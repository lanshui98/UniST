"""Cell-type transfer: spatial 1-NN (fast) or hybrid world-spatial + INR embedding (INR mode)."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _as_dense(X) -> np.ndarray:
    """Convert AnnData X / obsm arrays (dense, sparse, or object-wrapped sparse) to float64."""
    if X is None:
        raise ValueError("Cannot densify None")
    if sparse.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    if hasattr(X, "toarray") and callable(X.toarray):
        try:
            return np.asarray(X.toarray(), dtype=np.float64)
        except Exception:
            pass
    arr = np.asarray(X)
    if arr.dtype == object:
        if arr.shape == () or arr.size == 1:
            item = arr.item() if arr.shape == () else arr.ravel()[0]
            if sparse.issparse(item) or hasattr(item, "toarray"):
                return _as_dense(item)
        raise TypeError(
            f"Cannot densify object array of shape {arr.shape}; "
            f"element type={type(arr.ravel()[0]) if arr.size else None}"
        )
    return np.asarray(arr, dtype=np.float64)


def transfer_1nn(
    ref_coords: np.ndarray,
    query_coords: np.ndarray,
    ref_adata: AnnData,
    label_key: Optional[str] = None,
    transfer_expression: bool = False,
) -> AnnData:
    """Pure spatial 1-NN in **original / InterpolAI world** coordinates."""
    if ref_coords.shape[0] != ref_adata.n_obs:
        raise ValueError("ref_coords length must match ref_adata.n_obs")

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(ref_coords)
    distances, indices = nn.kneighbors(query_coords)
    idx = indices.ravel()

    if transfer_expression:
        X = _as_dense(ref_adata.X)[idx]
    else:
        X = np.zeros((query_coords.shape[0], ref_adata.n_vars), dtype=np.float32)

    out = AnnData(
        X=X,
        var=ref_adata.var.copy(),
        obs=pd.DataFrame(index=[f"middle_{i}" for i in range(query_coords.shape[0])]),
    )
    out.obsm["spatial"] = np.asarray(query_coords, dtype=np.float64).copy()
    out.obsm["nn_distance"] = distances.ravel()
    out.obs["nn_index"] = idx

    if label_key is not None:
        if label_key not in ref_adata.obs:
            raise KeyError(
                f"label_key '{label_key}' not in obs. Available: {list(ref_adata.obs.columns)}"
            )
        out.obs[label_key] = np.asarray(ref_adata.obs[label_key].iloc[idx])
        out.obs["predicted_celltype"] = out.obs[label_key].values

    return out


def _balanced_alpha(d_expr: np.ndarray, d_spatial: np.ndarray) -> float:
    """Median-based weight so expression and spatial terms have similar scale."""
    m_e = float(np.median(d_expr))
    m_s = float(np.median(d_spatial))
    return m_s / (m_e + m_s + 1e-12)


def transfer_hybrid_nn(
    ref_coords: np.ndarray,
    query_coords: np.ndarray,
    ref_embd: np.ndarray,
    query_embd: np.ndarray,
    ref_labels: np.ndarray,
    *,
    alpha: Union[float, str] = 0.05,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Hybrid 1-NN cell-type transfer.

    For each query point::

        d = alpha * d_emb + (1 - alpha) * d_spatial
        label = ref_labels[argmin(d)]

    ``ref_coords`` / ``query_coords`` must be in the **same** coordinate frame.
    For one_line_code INR mode this is **original world / InterpolAI** space
    (same as ``obsm['spatial']`` on flanking + middle), NOT GAE-normalized space.

    Embeddings are GAE ``embeddings`` vs INR ``fitted_embd``.
    """
    ref_coords = np.asarray(ref_coords, dtype=np.float64)
    query_coords = np.asarray(query_coords, dtype=np.float64)
    ref_embd = _as_dense(ref_embd)
    query_embd = _as_dense(query_embd)
    ref_labels = np.asarray(ref_labels)

    if ref_embd.shape[0] != ref_coords.shape[0]:
        raise ValueError("ref_embd and ref_coords length mismatch")
    if query_embd.shape[0] != query_coords.shape[0]:
        raise ValueError("query_embd and query_coords length mismatch")
    if ref_embd.shape[1] != query_embd.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: ref {ref_embd.shape[1]} vs query {query_embd.shape[1]}"
        )

    n_q = query_embd.shape[0]
    best_idx = np.empty(n_q, dtype=np.int64)
    alpha_used: Optional[float] = None if alpha == "auto" else float(alpha)

    if alpha == "auto":
        q0 = query_embd[: min(256, n_q)]
        c0 = query_coords[: min(256, n_q)]
        d_e = np.linalg.norm(q0[:, None, :] - ref_embd[None, :, :], axis=-1)
        d_s = np.linalg.norm(c0[:, None, :] - ref_coords[None, :, :], axis=-1)
        alpha_used = _balanced_alpha(d_e, d_s)

    assert alpha_used is not None

    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        qe = query_embd[start:end]
        qc = query_coords[start:end]
        d_expr = np.linalg.norm(qe[:, None, :] - ref_embd[None, :, :], axis=-1)
        d_spatial = np.linalg.norm(qc[:, None, :] - ref_coords[None, :, :], axis=-1)
        d = alpha_used * d_expr + (1.0 - alpha_used) * d_spatial
        best_idx[start:end] = d.argmin(axis=1)

    labels = ref_labels[best_idx]
    return labels, best_idx, float(alpha_used)


def annotate_with_hybrid(
    middle: AnnData,
    ref_adata: AnnData,
    ref_emb_adata: AnnData,
    label_key: str,
    *,
    alpha: Union[float, str] = 0.05,
) -> AnnData:
    """
    Hybrid cell-type annotation for INR middle slice.

    Spaces (do not mix):
    - **Spatial term (world)**: ``middle.obsm["spatial"]`` vs ``ref_adata.obsm["spatial"]``
      — InterpolAI / original ST coordinates (same frame as fast mode).
    - **Embedding term**: ``middle.obsm["fitted_embd"]`` vs
      ``ref_emb_adata.obsm["embeddings"]``.
    - ``spatial_normalized`` is only for INR training/prediction, **not** used here.
    """
    if "fitted_embd" not in middle.obsm:
        raise KeyError("middle is missing obsm['fitted_embd'] (INR embedding)")
    if "spatial" not in middle.obsm:
        raise KeyError("middle is missing obsm['spatial'] (world / InterpolAI coords)")
    if "spatial" not in ref_adata.obsm:
        raise KeyError("ref_adata is missing obsm['spatial'] (flanking world coords)")
    if "embeddings" not in ref_emb_adata.obsm:
        raise KeyError("ref embedder output missing obsm['embeddings']")
    if label_key not in ref_adata.obs:
        raise KeyError(f"label_key '{label_key}' not in flanking obs")
    if ref_emb_adata.n_obs != ref_adata.n_obs:
        raise ValueError(
            f"Embedded ref n_obs={ref_emb_adata.n_obs} != flanking n_obs={ref_adata.n_obs}"
        )

    ref_xyz = _as_dense(ref_adata.obsm["spatial"])
    query_xyz = _as_dense(middle.obsm["spatial"])
    if ref_xyz.shape[1] != query_xyz.shape[1]:
        # allow 2D vs 3D by padding/truncating to min dim for distance
        d = min(ref_xyz.shape[1], query_xyz.shape[1])
        ref_xyz = ref_xyz[:, :d]
        query_xyz = query_xyz[:, :d]

    labels, nn_idx, alpha_used = transfer_hybrid_nn(
        ref_coords=ref_xyz,
        query_coords=query_xyz,
        ref_embd=_as_dense(ref_emb_adata.obsm["embeddings"]),
        query_embd=_as_dense(middle.obsm["fitted_embd"]),
        ref_labels=np.asarray(ref_adata.obs[label_key]),
        alpha=alpha,
    )
    middle.obs[label_key] = labels
    middle.obs["predicted_celltype"] = labels
    middle.obs["nn_index"] = nn_idx
    middle.uns.setdefault("unist", {})
    middle.uns["unist"]["annotation"] = "hybrid_nn"
    middle.uns["unist"]["annotation_alpha"] = alpha_used
    middle.uns["unist"]["annotation_spatial_space"] = "world (obsm['spatial'])"
    middle.uns["unist"]["annotation_formula"] = (
        "d = alpha * ||fitted_embd - gae_emb|| + (1-alpha) * ||xyz_world_q - xyz_world_ref||"
    )
    return middle
