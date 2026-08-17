"""Rasterize ST spots to occupancy images and map InterpolAI pixels back to world coords.

Pipeline used by ``one_line_code``
----------------------------------
1. **ST → image**: project flanking slice xy into one shared bbox; paint binary
   occupancy (spot → pixel = 255). Grid size is adaptive from median nearest-
   neighbor spacing, clipped to ``[64, max_side]``.
2. **InterpolAI**: 3-channel occupancy pair → middle frame at ``t=0.5``.
3. **Back to original coord**: pixels above threshold → xy via the same bbox
   affine map; z set to ``z_mid = (z_a + z_b) / 2``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from anndata import AnnData


def get_xy(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not in adata.obsm")
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(f"Expected spatial (N,2) or (N,3), got {coords.shape}")
    return coords[:, :2]


def infer_z(adata: AnnData, spatial_key: str, z: Optional[float], fallback: float) -> float:
    if z is not None:
        return float(z)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.shape[1] == 3:
        return float(np.median(coords[:, 2]))
    return float(fallback)


def shared_bbox(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    pad_frac: float = 0.02,
) -> Tuple[float, float, float, float]:
    xy = np.vstack([xy_a, xy_b])
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    pad_x = dx * pad_frac
    pad_y = dy * pad_frac
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def auto_grid_size(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    bbox: Tuple[float, float, float, float],
    max_side: int = 1024,
) -> Tuple[int, int]:
    """Choose H,W so typical nearest-neighbor spacing is ~1 pixel, capped at max_side."""
    xy = np.vstack([xy_a, xy_b])
    xmin, xmax, ymin, ymax = bbox
    # rough spacing from median nearest distance on a sample
    n = min(len(xy), 2000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xy), size=n, replace=False) if len(xy) > n else np.arange(len(xy))
    sample = xy[idx]
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2).fit(sample)
    dists = nn.kneighbors(sample)[0][:, 1]
    spacing = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    spacing = max(spacing, 1e-6)
    width = xmax - xmin
    height = ymax - ymin
    w = int(np.clip(np.ceil(width / spacing), 64, max_side))
    h = int(np.clip(np.ceil(height / spacing), 64, max_side))
    return h, w


def rasterize_occupancy(
    xy: np.ndarray,
    bbox: Tuple[float, float, float, float],
    grid_hw: Tuple[int, int],
) -> np.ndarray:
    """Binary occupancy (H, W) uint8 in {0, 255}."""
    h, w = grid_hw
    xmin, xmax, ymin, ymax = bbox
    img = np.zeros((h, w), dtype=np.uint8)
    if len(xy) == 0:
        return img
    xs = (xy[:, 0] - xmin) / max(xmax - xmin, 1e-8) * (w - 1)
    ys = (xy[:, 1] - ymin) / max(ymax - ymin, 1e-8) * (h - 1)
    xs = np.clip(xs.round().astype(int), 0, w - 1)
    ys = np.clip(ys.round().astype(int), 0, h - 1)
    img[ys, xs] = 255
    return img


def to_3channel(img2d: np.ndarray) -> np.ndarray:
    if img2d.ndim == 2:
        return np.stack([img2d, img2d, img2d], axis=-1)
    if img2d.ndim == 3 and img2d.shape[-1] == 3:
        return img2d
    raise ValueError(f"Unsupported image shape {img2d.shape}")


def sample_coords_from_occupancy(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    z: float,
    threshold: float = 127.0,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Convert occupancy image → (N, 3) world coordinates at depth z.

    ``img`` may be (H,W) or (H,W,3); uses mean over channels.
    """
    if img.ndim == 3:
        gray = img.astype(np.float32).mean(axis=-1)
    else:
        gray = img.astype(np.float32)
    ys, xs = np.where(gray >= threshold)
    if len(xs) == 0:
        # fallback: any nonzero
        ys, xs = np.where(gray > 0)
    if len(xs) == 0:
        raise RuntimeError("InterpolAI middle occupancy is empty; cannot sample spots.")

    if max_points is not None and len(xs) > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys = xs[pick], ys[pick]

    h, w = gray.shape
    xmin, xmax, ymin, ymax = bbox
    x = xmin + xs.astype(np.float64) / max(w - 1, 1) * (xmax - xmin)
    y = ymin + ys.astype(np.float64) / max(h - 1, 1) * (ymax - ymin)
    return np.column_stack([x, y, np.full(len(x), z, dtype=np.float64)])
