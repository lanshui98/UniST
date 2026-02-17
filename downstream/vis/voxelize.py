"""
Voxelize point cloud (spateo-style).

Supports voxelizing cell/region labels and gene expression: any point_data
on the input point cloud is carried over to the output voxel model's cell_data.
"""

from typing import Optional

import numpy as np
from pyvista import PolyData, UnstructuredGrid


def voxelize_pc(
    pc: PolyData,
    voxel_size: Optional[np.ndarray] = None,
) -> UnstructuredGrid:
    """
    Voxelize the point cloud (spateo-style).

    Use for both cell/region labels and gene expression: all keys in
    ``pc.point_data`` (e.g. region labels, gene expression values) are
    copied to the output voxel model's ``cell_data``.

    Parameters
    ----------
    pc : pyvista.PolyData
        Point cloud (e.g. from :func:`construct_pc`).
    voxel_size : np.ndarray, optional
        Per-point voxel size, shape ``(pc.n_points, 3)``. If None, PVGeo
        uses a single global voxel size (estimated from point spacing).

    Returns
    -------
    pyvista.UnstructuredGrid
        Voxel model with ``cell_data`` populated from ``pc.point_data``.

    Examples
    --------
    Voxelize cell/region labels:
        pc, _ = construct_pc(adata, groupby="region")
        vox = voxelize_pc(pc)

    Voxelize gene expression:
        pc, _ = construct_pc(adata, layer="X", spatial_key="spatial")
        adata.obs["expr"] = adata.X[:, gene_idx]
        pc.point_data["gene1"] = adata.obs["expr"].values
        vox = voxelize_pc(pc)
    """
    try:
        import PVGeo
    except ImportError:
        raise ImportError(
            "Voxelization requires PVGeo. Install with: pip install PVGeo"
        ) from None

    voxelizer = PVGeo.filters.VoxelizePoints()

    if voxel_size is not None:
        voxel_size = np.asarray(voxel_size)
        if voxel_size.shape != (pc.n_points, 3):
            raise ValueError(
                "voxel_size must have shape (n_points, 3), got %s"
                % (voxel_size.shape,)
            )
        voxelizer.set_deltas(
            voxel_size[:, 0],
            voxel_size[:, 1],
            voxel_size[:, 2],
        )
        voxelizer.set_estimate_grid(False)

    voxel_pc = voxelizer.apply(pc)

    pc_keys = list(pc.point_data.keys())
    for key in pc_keys:
        voxel_pc.cell_data[key] = pc.point_data[key]

    return voxel_pc
