"""Spatial coordinate helpers for middle-slice prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from anndata import AnnData


def get_spatial(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(f"Expected spatial shape (N, 2) or (N, 3), got {coords.shape}")
    return coords


def to_xyz(
    coords: np.ndarray,
    z: Optional[float] = None,
    default_z: float = 0.0,
) -> np.ndarray:
    """Promote 2D coords to 3D; keep 3D as-is (optionally override z)."""
    if coords.shape[1] == 3:
        out = coords.copy()
        if z is not None:
            out[:, 2] = z
        return out
    z_val = default_z if z is None else z
    return np.column_stack([coords, np.full(coords.shape[0], z_val, dtype=np.float64)])


def infer_slice_z(coords: np.ndarray, z: Optional[float], fallback: float) -> float:
    if z is not None:
        return float(z)
    if coords.shape[1] == 3:
        return float(np.median(coords[:, 2]))
    return float(fallback)


def build_middle_query(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    z_mid: float,
    source: str = "union",
) -> np.ndarray:
    """
    Build query XY at the middle plane.

    Parameters
    ----------
    source : {"union", "a", "b"}
        Which slice XY positions to place at z_mid.
    """
    xy_a = coords_a[:, :2]
    xy_b = coords_b[:, :2]
    if source == "a":
        xy = xy_a
    elif source == "b":
        xy = xy_b
    elif source == "union":
        xy = np.vstack([xy_a, xy_b])
    else:
        raise ValueError(f"Unknown query source: {source!r}")
    return np.column_stack([xy, np.full(xy.shape[0], z_mid, dtype=np.float64)])


def normalize_coords(
    coords: np.ndarray,
    ref: Optional[np.ndarray] = None,
    keep_ratio: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Normalize coords to roughly [-1, 1] using ref (or coords) range."""
    ref = coords if ref is None else ref
    ref_min = ref.min(axis=0)
    ref_max = ref.max(axis=0)
    ref_range = np.maximum(ref_max - ref_min, 1e-8)
    normed = (coords - ref_min) / ref_range
    normed = (normed - 0.5) * 2.0
    if keep_ratio and coords.shape[1] >= 2:
        xy_max = max(ref_range[0], ref_range[1])
        normed[:, 0] *= ref_range[0] / xy_max
        normed[:, 1] *= ref_range[1] / xy_max
    stats = {"min": ref_min, "range": ref_range, "keep_ratio": keep_ratio}
    return normed.astype(np.float32), stats
