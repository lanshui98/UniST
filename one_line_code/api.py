"""One-line API: two ST slices → middle slice via InterpolAI (+ 1-NN or SUICA INR).

Coordinate pipeline
-------------------
1. **ST → image**: rasterize flanking ``obsm['spatial']`` xy into a shared occupancy
   grid (adaptive resolution from spot spacing, capped by ``max_side``).
2. **InterpolAI**: synthesize the middle occupancy frame at ``time=0.5``.
3. **Image → original coords**: threshold the middle frame and map pixel indices
   back to the original world xy (same bbox), with ``z = (z_a + z_b) / 2``.

Downstream
----------
- ``mode="fast"``: spatial 1-NN cell-type transfer on those coords.
- ``mode="inr"``: SUICA_pro GAE+INR gene expression at those coords, then hybrid
  cell-type prediction: embedding distance + **world-space** spatial distance
  (same ``obsm['spatial']`` frame as InterpolAI / flanking slices).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import anndata as ad
import numpy as np
from anndata import AnnData

from .interpolai_backend import interpolate_middle_frame
from .nn_annotate import annotate_with_hybrid, transfer_1nn
from .occupancy import (
    auto_grid_size,
    get_xy,
    infer_z,
    rasterize_occupancy,
    sample_coords_from_occupancy,
    shared_bbox,
    to_3channel,
)
from .suica_backend import run_suica_inr_middle, stack_flanking_3d

Mode = Literal["fast", "inr"]


def _build_middle_coords_with_interpolai(
    slice_a: AnnData,
    slice_b: AnnData,
    *,
    spatial_key: str,
    z_a: Optional[float],
    z_b: Optional[float],
    tile_size: Tuple[int, int],
    max_side: int,
    occupancy_threshold: float,
    max_points: Optional[int],
    interpolai_model=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    ST spots → occupancy images → InterpolAI → spots in original coordinates.

    Returns query_xyz, xyz_a, xyz_b, z_mid, meta (bbox / grid info).
    """
    xy_a = get_xy(slice_a, spatial_key)
    xy_b = get_xy(slice_b, spatial_key)
    za = infer_z(slice_a, spatial_key, z_a, 0.0)
    zb = infer_z(slice_b, spatial_key, z_b, 1.0)
    if abs(za - zb) < 1e-12:
        zb = za + 1.0
    z_mid = 0.5 * (za + zb)

    # --- ST → shared occupancy grid ---
    bbox = shared_bbox(xy_a, xy_b)
    grid_hw = auto_grid_size(xy_a, xy_b, bbox, max_side=max_side)
    occ_a = to_3channel(rasterize_occupancy(xy_a, bbox, grid_hw))
    occ_b = to_3channel(rasterize_occupancy(xy_b, bbox, grid_hw))

    # --- InterpolAI middle frame ---
    mid_img = interpolate_middle_frame(
        occ_a, occ_b, model=interpolai_model, time=0.5, tile_size=tile_size
    )

    # --- image pixels → original world coordinates ---
    query = sample_coords_from_occupancy(
        mid_img,
        bbox,
        z_mid,
        threshold=occupancy_threshold,
        max_points=max_points,
    )

    xyz_a = np.column_stack([xy_a, np.full(len(xy_a), za)])
    xyz_b = np.column_stack([xy_b, np.full(len(xy_b), zb)])
    meta = {
        "bbox_xy": bbox,  # (xmin, xmax, ymin, ymax)
        "grid_hw": grid_hw,  # (H, W)
        "occupancy_threshold": occupancy_threshold,
        "pipeline": "ST→occupancy→InterpolAI→original_xy + z_mid",
    }
    return query, xyz_a, xyz_b, z_mid, meta


def predict_middle_slice(
    slice_a: AnnData,
    slice_b: AnnData,
    mode: Mode = "fast",
    *,
    spatial_key: str = "spatial",
    label_key: Optional[str] = None,
    z_a: Optional[float] = None,
    z_b: Optional[float] = None,
    transfer_expression: bool = False,
    # InterpolAI / occupancy
    tile_size: Tuple[int, int] = (1024, 1024),
    max_side: int = 1024,
    occupancy_threshold: float = 127.0,
    max_points: Optional[int] = None,
    interpolai_model=None,
    # SUICA INR
    work_dir: Optional[Union[str, Path]] = None,
    embedder_epochs: int = 200,
    inr_epochs: int = 300,
    annotate: bool = True,
    annotation_alpha: Union[float, str] = 0.05,
) -> AnnData:
    """
    Predict the middle ST slice between two flanking slices.

    **Geometry (always)**::

        ST spots  →  occupancy image (shared bbox / adaptive grid)
                  →  InterpolAI (t=0.5)
                  →  back to original xy + z_mid

    **Labels / expression**

    - ``mode="fast"``: spatial 1-NN cell-type (optional expression copy).
    - ``mode="inr"``: SUICA_pro GAE + FFN-INR gene expression, then **hybrid**
      cell-type prediction::

          d = alpha * d_embedding + (1 - alpha) * d_spatial

      where embeddings are GAE (flanking) vs INR ``fitted_embd`` (middle).
      ``annotation_alpha`` default ``0.05`` (tutorial); use ``"auto"`` for
      median-balanced weight.

    Parameters
    ----------
    slice_a, slice_b : AnnData
        Flanking slices with shared genes and ``obsm[spatial_key]``.
    mode : {"fast", "inr"}
    label_key : str, optional
        Cell-type column in ``.obs``.
    annotation_alpha : float or "auto"
        Hybrid annotation weight on embedding distance (INR mode only).
    work_dir : path, optional
        Scratch directory for SUICA logs/checkpoints (INR mode).

    Examples
    --------
    ::

        from unist import predict_middle_slice
        middle = predict_middle_slice(a, b, mode="fast", label_key="cell_type")
        middle = predict_middle_slice(
            a, b, mode="inr", label_key="cell_type", work_dir="./unist_run"
        )
    """
    if not slice_a.var_names.equals(slice_b.var_names):
        raise ValueError("slice_a and slice_b must have identical var_names.")

    query, xyz_a, xyz_b, z_mid, geom_meta = _build_middle_coords_with_interpolai(
        slice_a,
        slice_b,
        spatial_key=spatial_key,
        z_a=z_a,
        z_b=z_b,
        tile_size=tile_size,
        max_side=max_side,
        occupancy_threshold=occupancy_threshold,
        max_points=max_points,
        interpolai_model=interpolai_model,
    )

    ref = stack_flanking_3d(slice_a, slice_b, xyz_a, xyz_b)
    ref_coords = np.asarray(ref.obsm["spatial"], dtype=np.float64)

    if mode == "fast":
        if label_key is None and not transfer_expression:
            raise ValueError(
                "fast mode needs label_key and/or transfer_expression=True."
            )
        out = transfer_1nn(
            ref_coords,
            query,
            ref,
            label_key=label_key,
            transfer_expression=transfer_expression,
        )
        out.uns["unist"] = {
            "mode": "fast",
            "structure": "InterpolAI",
            "annotation": "1-NN",
            "z_mid": z_mid,
            **geom_meta,
        }
        return out

    if mode == "inr":
        if work_dir is None:
            tmp = tempfile.mkdtemp(prefix="unist_one_line_")
            work_path = Path(tmp).resolve()
        else:
            work_path = Path(work_dir).expanduser().resolve()

        out = run_suica_inr_middle(
            ref,
            query,
            work_path,
            embedder_epochs=embedder_epochs,
            inr_epochs=inr_epochs,
        )
        out.uns["unist"]["structure"] = "InterpolAI"
        out.uns["unist"]["z_mid"] = z_mid
        out.uns["unist"].update(geom_meta)

        if annotate and label_key is not None:
            emb_path = out.uns["unist"].get("embedded_h5ad")
            if not emb_path or not Path(emb_path).exists():
                raise FileNotFoundError(
                    "GAE embedded-all.h5ad missing; cannot run hybrid cell-type prediction."
                )
            ref_emb = ad.read_h5ad(emb_path)
            annotate_with_hybrid(
                out,
                ref_adata=ref,
                ref_emb_adata=ref_emb,
                label_key=label_key,
                alpha=annotation_alpha,
            )
        return out

    raise ValueError(f"Unknown mode: {mode!r}. Use 'fast' or 'inr'.")
