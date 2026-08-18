"""Wrap external/SUICA_pro GAE embedder + INR train/predict."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
from anndata import AnnData
from omegaconf import OmegaConf


@contextmanager
def _suica_import_context(suica_root: Path):
    """Put SUICA_pro on sys.path and cwd so its absolute imports resolve."""
    root = str(suica_root.resolve())
    old_cwd = os.getcwd()
    inserted = False
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted and root in sys.path:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _latest_version_dir(logs_root: Path) -> Path:
    ll = logs_root / "lightning_logs"
    if not ll.exists():
        raise FileNotFoundError(f"No lightning_logs under {logs_root}")
    versions = sorted(
        ll.glob("version_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
    )
    if not versions:
        raise FileNotFoundError(f"No version_* under {ll}")
    return versions[-1]


def stack_flanking_3d(
    slice_a: AnnData,
    slice_b: AnnData,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
) -> AnnData:
    a = slice_a.copy()
    b = slice_b.copy()
    a.obsm["spatial"] = xyz_a
    b.obsm["spatial"] = xyz_b
    return ad.concat([a, b], join="outer", index_unique="-")


def normalize_query_coords_3d(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    *,
    keep_ratio: bool = True,
    preserve_z_scale: bool = True,
    z_scale_factor: float = 1.5,
) -> np.ndarray:
    """Same normalization as SUICA prepare_custom_coords.normalize_coords_3d."""
    from .paths import require_suica

    suica = require_suica()
    with _suica_import_context(suica):
        from prepare_custom_coords import normalize_coords_3d

        return normalize_coords_3d(
            query_coords.copy(),
            ref_coords,
            keep_ratio=keep_ratio,
            preserve_z_scale=preserve_z_scale,
            z_scale_factor=z_scale_factor,
        )


def run_suica_inr_middle(
    stacked: AnnData,
    query_coords: np.ndarray,
    work_dir: Path,
    *,
    embedder_epochs: int = 200,
    inr_epochs: int = 300,
    batch_size_embedder: int = 512,
    batch_size_inr: int = 2048,
    n_neighbors: int = 6,
    z_scale_factor: float = 1.5,
) -> AnnData:
    """
    Train GAE embedder + FFN-INR (SUICA_pro) on stacked flanking slices,
    then predict gene expression at ``query_coords`` (raw XYZ).
    """
    from .paths import require_suica

    suica = require_suica()
    # Absolute paths: SUICA training chdirs into external/SUICA_pro
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Persist InterpolAI world coords immediately (source of truth for plotting + hybrid spatial)
    query_coords = np.asarray(query_coords, dtype=np.float64)
    np.save(work_dir / "middle_spatial_raw.npy", query_coords)

    data_file = (work_dir / "flanking_3d.h5ad").resolve()
    stacked.write_h5ad(data_file)

    emb_logs = (work_dir / "GAE-3D-sparse" / "one_line").resolve()
    inr_logs = (work_dir / "GAE+FFN-3D-sparse" / "one_line").resolve()

    embedder_cfg = OmegaConf.create(
        {
            "case": "one_line",
            "dataset": {
                "type": "GraphST3D",
                "data_file": str(data_file),
                "val_proportion": 0.2,
                "keep_ratio": True,
                "n_neighbors": n_neighbors,
                "require_coordnorm": True,
                "use_anisotropic_knn": True,
                "z_weight": 2.0,
                "z_threshold": None,
                "preserve_z_scale": True,
                "z_scale_factor": z_scale_factor,
            },
            "pipeline": {
                "embedder": {
                    "model": "GAE",
                    "dim_hidden": [2048, 512, 128],
                    "dim_latent": 64,
                },
                "optimization": {
                    "seed": 8848,
                    "epochs": embedder_epochs,
                    "lr": 1e-5,
                    "val_freq": max(1, embedder_epochs // 5),
                    "logs": str(emb_logs),
                    "batch_size": batch_size_embedder,
                },
                "predict_mode": "all",
                "embedded_data": "embedded-all.h5ad",
            },
        }
    )

    with _suica_import_context(suica):
        from systems import train_embedder, train_inr, predict_inr

        train_embedder(embedder_cfg)

        emb_ver = _latest_version_dir(emb_logs)
        embedded_h5ad = emb_ver / "embedded-all.h5ad"
        emb_ckpt = emb_ver / "checkpoints" / "last.ckpt"
        if not embedded_h5ad.exists():
            # lightning may write under log_dir differently
            candidates = list(emb_ver.rglob("embedded-all.h5ad"))
            if not candidates:
                raise FileNotFoundError(f"embedded-all.h5ad not found under {emb_ver}")
            embedded_h5ad = candidates[0]
        if not emb_ckpt.exists():
            ckpts = list(emb_ver.rglob("last.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"last.ckpt not found under {emb_ver}")
            emb_ckpt = ckpts[0]

        embedded_h5ad = Path(embedded_h5ad).resolve()
        emb_ckpt = Path(emb_ckpt).resolve()

        inr_cfg = OmegaConf.create(
            {
                "case": "one_line",
                "dataset": {
                    "type": "ST3D",
                    "data_file": str(embedded_h5ad),
                    "val_proportion": 0.2,
                    "keep_ratio": True,
                    "require_coordnorm": False,
                    "preserve_z_scale": True,
                    "z_scale_factor": z_scale_factor,
                },
                "pipeline": {
                    "target": "embeddings",
                    "inr": {
                        "model": "FFN",
                        "num_hidden_layers": 3,
                        "num_hidden_features": 2048,
                        "phase": max(1, inr_epochs // 2),
                        "encoding_type": "enhanced",
                        "mapping_size": 128,
                        "encoding_scales": [1, 10, 100],
                        "anisotropic_3d": True,
                        "z_scales": [0.1, 1.0, 10.0],
                        "decoder": {
                            "ckpt": str(emb_ckpt),
                            "recon_loss": True,
                            "finetune": True,
                        },
                    },
                    "optimization": {
                        "seed": 8848,
                        "epochs": inr_epochs,
                        "lr": 1e-4,
                        "val_freq": max(1, inr_epochs // 5),
                        "logs": str(inr_logs),
                        "batch_size": batch_size_inr,
                    },
                    "predict_mode": "val",
                    "reconstructed_data": "reconstructed-val.h5ad",
                },
            }
        )
        train_inr(inr_cfg)

        inr_ver = _latest_version_dir(inr_logs)
        inr_ckpt = inr_ver / "checkpoints" / "last.ckpt"
        if not inr_ckpt.exists():
            ckpts = list(inr_ver.rglob("last.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"INR last.ckpt not found under {inr_ver}")
            inr_ckpt = ckpts[0]

        # custom coords in the same normalized space as training (ST3D with require_coordnorm False
        # still may expect coords already normalized in embedded file — use prepare_custom_coords
        # against the stacked raw spatial as reference, matching tutorial 3d_sparse settings.
        ref_spatial = np.asarray(stacked.obsm["spatial"], dtype=np.float64)
        query_norm = normalize_query_coords_3d(
            query_coords,
            ref_spatial,
            keep_ratio=True,
            preserve_z_scale=True,
            z_scale_factor=z_scale_factor,
        )
        coords_file = (work_dir / "custom_coords_3d_norm.npy").resolve()
        np.save(coords_file, query_norm.astype(np.float32))

        pred_cfg = OmegaConf.create(
            {
                "case": "one_line",
                "dataset": {
                    "type": "ST3D",
                    "data_file": str(embedded_h5ad),
                    "val_proportion": 0.2,
                    "keep_ratio": True,
                    "require_coordnorm": False,
                    "preserve_z_scale": True,
                    "z_scale_factor": z_scale_factor,
                },
                "pipeline": {
                    "target": "embeddings",
                    "prediction": {"ckpt": str(inr_ckpt)},
                    "inr": {
                        "model": "FFN",
                        "num_hidden_layers": 3,
                        "num_hidden_features": 2048,
                        "phase": 1000,
                        "encoding_type": "enhanced",
                        "mapping_size": 128,
                        "encoding_scales": [1, 10, 100],
                        "anisotropic_3d": True,
                        "z_scales": [0.1, 1.0, 10.0],
                        "decoder": {
                            "ckpt": str(emb_ckpt),
                            "recon_loss": True,
                            "finetune": True,
                        },
                    },
                    "optimization": {
                        "seed": 8848,
                        "epochs": inr_epochs,
                        "lr": 1e-4,
                        "val_freq": 200,
                        "logs": str(inr_logs),
                        "batch_size": batch_size_inr,
                    },
                    "predict_mode": "custom",
                    "custom_coords_file": str(coords_file),
                    "reconstructed_data": "reconstructed-custom-middle.h5ad",
                },
            }
        )
        predict_inr(pred_cfg)

    # Find output
    out_candidates = list(inr_logs.rglob("reconstructed-custom-middle.h5ad"))
    if not out_candidates:
        raise FileNotFoundError(f"INR prediction output not found under {inr_logs}")
    raw_out = ad.read_h5ad(out_candidates[-1])

    if "reconstructed_raw" in raw_out.obsm:
        from scipy import sparse as sp

        X = raw_out.obsm["reconstructed_raw"]
        if sp.issparse(X):
            X = X.toarray()
        result = AnnData(
            X=np.asarray(X, dtype=np.float32),
            obs=raw_out.obs.copy(),
            var=stacked.var.copy()
            if np.asarray(X).shape[1] == stacked.n_vars
            else raw_out.var.copy(),
        )
        if "fitted_embd" in raw_out.obsm:
            emb = raw_out.obsm["fitted_embd"]
            if sp.issparse(emb):
                emb = emb.toarray()
            result.obsm["fitted_embd"] = np.asarray(emb, dtype=np.float32)
        result.obsm["reconstructed_raw"] = np.asarray(X, dtype=np.float32)
    else:
        result = raw_out.copy()

    result.obsm["spatial"] = query_coords.copy()  # InterpolAI / world — plot + hybrid spatial
    result.obsm["spatial_normalized"] = query_norm  # GAE/INR input only
    np.save(work_dir / "middle_spatial_normalized.npy", query_norm.astype(np.float64))
    result.uns["unist"] = {
        "mode": "inr",
        "backend": "SUICA_pro",
        "work_dir": str(work_dir),
        "embedder_epochs": embedder_epochs,
        "inr_epochs": inr_epochs,
        "embedded_h5ad": str(embedded_h5ad),
        "coord_spaces": {
            "obsm.spatial": "InterpolAI world xy + z_mid (plotting + hybrid spatial term)",
            "obsm.spatial_normalized": "GAE/INR normalized coords (model input only)",
            "hybrid_spatial": "world obsm['spatial'] vs flanking obsm['spatial']",
            "hybrid_embedding": "fitted_embd vs GAE embeddings",
        },
    }
    return result
