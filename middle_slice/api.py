"""One-line API: two ST slices → middle slice."""

from __future__ import annotations

from typing import Literal, Optional

import anndata as ad
import numpy as np
from anndata import AnnData

from .coords import (
    build_middle_query,
    get_spatial,
    infer_slice_z,
    normalize_coords,
    to_xyz,
)
from .fast_nn import transfer_1nn
from .inr_model import fit_predict_inr


Mode = Literal["fast", "inr"]


def predict_middle_slice(
    slice_a: AnnData,
    slice_b: AnnData,
    mode: Mode = "fast",
    *,
    spatial_key: str = "spatial",
    label_key: Optional[str] = None,
    z_a: Optional[float] = None,
    z_b: Optional[float] = None,
    query: Literal["union", "a", "b"] = "union",
    transfer_expression: bool = False,
    # INR options
    n_pcs: Optional[int] = 50,
    epochs: int = 300,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    annotate: bool = True,
) -> AnnData:
    """
    Predict the middle spatial transcriptomics slice between two flanking slices.

    Parameters
    ----------
    slice_a, slice_b : AnnData
        Flanking ST slices. Must share the same genes (``var_names``).
        Spatial coordinates in ``obsm[spatial_key]`` as (N, 2) or (N, 3).
    mode : {"fast", "inr"}
        - ``fast``: 1-nearest-neighbor transfer of cell-type labels (and optionally
          expression) from the flanking slices onto a middle-plane query.
        - ``inr``: train a lightweight Fourier-feature INR on both slices and
          predict gene expression at the middle plane.
    spatial_key : str
        Key in ``obsm`` for spatial coordinates.
    label_key : str, optional
        ``obs`` column for cell-type labels. Required for ``mode="fast"``
        annotation; optional for ``mode="inr"`` (spatial 1-NN annotation).
    z_a, z_b : float, optional
        Slice depths. If spatial is 2D, defaults to 0 and 1. If 3D, defaults
        to the median z of each slice.
    query : {"union", "a", "b"}
        How to build middle-plane XY positions.
    transfer_expression : bool
        In fast mode, also copy expression from the nearest neighbor.
    n_pcs, epochs, batch_size, lr, device :
        INR training options. Set ``n_pcs=None`` to predict all genes directly.
    annotate : bool
        In INR mode, also attach cell-type labels via spatial 1-NN when
        ``label_key`` is provided.

    Returns
    -------
    AnnData
        Middle slice with ``obsm["spatial"]``. Fast mode writes labels to
        ``obs[label_key]`` / ``obs["predicted_celltype"]``. INR mode stores
        predicted expression in ``X``.

    Examples
    --------
    Fast cell-type annotation::

        middle = unist.predict_middle_slice(a, b, mode="fast", label_key="cell_type")

    INR gene expression::

        middle = unist.predict_middle_slice(a, b, mode="inr")
    """
    if not slice_a.var_names.equals(slice_b.var_names):
        raise ValueError("slice_a and slice_b must have identical var_names (genes).")

    coords_a = get_spatial(slice_a, spatial_key)
    coords_b = get_spatial(slice_b, spatial_key)

    za = infer_slice_z(coords_a, z_a, fallback=0.0)
    zb = infer_slice_z(coords_b, z_b, fallback=1.0)
    if abs(za - zb) < 1e-12:
        zb = za + 1.0
    z_mid = 0.5 * (za + zb)

    xyz_a = to_xyz(coords_a, z=za)
    xyz_b = to_xyz(coords_b, z=zb)
    ref_coords = np.vstack([xyz_a, xyz_b])
    query_coords = build_middle_query(xyz_a, xyz_b, z_mid, source=query)

    ref = ad.concat([slice_a, slice_b], join="outer", index_unique="-")
    # Ensure spatial alignment after concat
    ref.obsm[spatial_key] = ref_coords

    if mode == "fast":
        if label_key is None and not transfer_expression:
            raise ValueError(
                "fast mode needs label_key (cell type) and/or transfer_expression=True."
            )
        return transfer_1nn(
            ref_coords,
            query_coords,
            ref,
            label_key=label_key,
            transfer_expression=transfer_expression,
        )

    if mode == "inr":
        ref_norm, _ = normalize_coords(ref_coords, ref=ref_coords, keep_ratio=True)
        query_norm, _ = normalize_coords(query_coords, ref=ref_coords, keep_ratio=True)

        from scipy import sparse

        X = ref.X
        if sparse.issparse(X):
            train_X = X.toarray().astype(np.float32)
        else:
            train_X = np.asarray(X, dtype=np.float32)

        pred_X, info = fit_predict_inr(
            ref_norm,
            train_X,
            query_norm,
            n_pcs=n_pcs,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

        import pandas as pd

        out = AnnData(
            X=pred_X,
            var=ref.var.copy(),
            obs=pd.DataFrame(index=[f"middle_{i}" for i in range(query_coords.shape[0])]),
        )
        out.obsm["spatial"] = query_coords.copy()
        out.obsm["spatial_normalized"] = query_norm.copy()
        out.uns["unist"] = {"mode": "inr", "method": "ffn-inr", **info, "z_mid": z_mid}

        if annotate and label_key is not None:
            labeled = transfer_1nn(
                ref_coords,
                query_coords,
                ref,
                label_key=label_key,
                transfer_expression=False,
            )
            out.obs[label_key] = labeled.obs[label_key].values
            out.obs["predicted_celltype"] = labeled.obs["predicted_celltype"].values

        return out

    raise ValueError(f"Unknown mode: {mode!r}. Use 'fast' or 'inr'.")
