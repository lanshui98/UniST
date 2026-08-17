"""Fast 1-nearest-neighbor cell-type (and optional expression) transfer."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _as_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def transfer_1nn(
    ref_coords: np.ndarray,
    query_coords: np.ndarray,
    ref_adata: AnnData,
    label_key: Optional[str] = None,
    transfer_expression: bool = False,
) -> AnnData:
    """
    For each query point, copy annotation (and optionally expression) from 1-NN.

    Parameters
    ----------
    ref_coords : (N, 3)
        Reference spatial coordinates (flanking slices stacked).
    query_coords : (M, 3)
        Middle-slice query coordinates.
    ref_adata : AnnData
        Concatenated flanking AnnData aligned with ref_coords.
    label_key : str, optional
        Column in ``ref_adata.obs`` to transfer (e.g. cell type).
    transfer_expression : bool
        If True, also copy gene expression from the nearest neighbor.
    """
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
    out.obsm["spatial"] = query_coords.copy()
    out.obsm["nn_distance"] = distances.ravel()
    out.obs["nn_index"] = idx

    if label_key is not None:
        if label_key not in ref_adata.obs:
            raise KeyError(
                f"label_key '{label_key}' not in ref_adata.obs. "
                f"Available: {list(ref_adata.obs.columns)}"
            )
        out.obs[label_key] = np.asarray(ref_adata.obs[label_key].iloc[idx])
        out.obs["predicted_celltype"] = out.obs[label_key].values

    out.uns["unist"] = {
        "mode": "fast",
        "method": "1-nearest-neighbor",
        "label_key": label_key,
        "transfer_expression": transfer_expression,
    }
    return out
