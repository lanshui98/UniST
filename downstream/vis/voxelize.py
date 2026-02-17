"""
Grid-based voxelization to PyVista ImageData (VTI).

- points_to_imagedata: rasterize points + labels (or any point_data) into a 3D grid.
- expression_to_imagedata: rasterize gene (or any continuous) expression; each voxel
  gets the mean expression of points inside it.
Voxel size is set by grid_shape and physical ranges: dx = (x_max - x_min) / nx, etc.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import ImageData, PolyData


def points_to_imagedata(
    points: Union[PolyData, np.ndarray],
    grid_shape: Tuple[int, int, int],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    point_data: Optional[dict] = None,
    fill_mode: str = "last",
    label_to_value: Optional[dict] = None,
    empty_voxel_value: Optional[Union[int, float]] = None,
) -> ImageData:
    """
    Rasterize points into a 3D grid (ImageData) with explicit voxel size.

    Voxel size is determined by grid_shape and physical ranges:
    ``dx = (x_max - x_min) / nx``, similarly for dy, dz. Points are mapped to
    voxel indices and each voxel gets a value (last point wins, or mode/mean
    depending on fill_mode). No PVGeo dependency.

    Parameters
    ----------
    points : pyvista.PolyData or np.ndarray
        If PolyData: uses .points and (if point_data is None) .point_data.
        If array: shape (N, 3) in physical coordinates.
    grid_shape : tuple (nx, ny, nz)
        Number of voxels along each axis. Voxel size is (range/n) per axis.
    x_range, y_range, z_range : tuple (min, max), optional
        Physical extent. If None and points is PolyData/array, inferred from
        point bounds (with a small padding).
    point_data : dict, optional
        Name -> array (length N). If None and points is PolyData, uses
        points.point_data. Keys ending with "_rgba" are skipped (scalar only).
    fill_mode : str, default "last"
        When several points fall in the same voxel: "last" (overwrite),
        "mode" (categorical: most frequent), "mean" (numeric mean).
    label_to_value : dict, optional
        Map factor labels to numbers per key. E.g. ``{"pred_label": {"Tumor": 1, "Stroma": 0}}``.
        Labels not in the dict get 0. So you get 1=Tumor, 0=background/other in the grid.
    empty_voxel_value : int or float, optional
        Value for voxels with no points. If None, use 0 (int) or NaN (float).
        E.g. 0 so that 0=empty, 1=Tumor.

    Returns
    -------
    pyvista.ImageData
        Grid with .origin, .spacing, and cell_data for each key (one value per voxel).
        Can be saved as .vti or plotted with three_d_plot(..., key="...").

    Examples
    --------
    From point cloud (e.g. construct_pc output):
        pc, _ = construct_pc(adata, groupby="pred_label")
        grid = points_to_imagedata(
            pc,
            grid_shape=(532, 400, 34),
            x_range=(6000, 14000),
            y_range=(-1000, 5000),
            z_range=(57.97, 1043.48),
        )
        grid.save("mask.vti")

    From CSV-style coordinates and labels:
        pts = df[["x", "y", "z"]].to_numpy()
        point_data = {"pred_label": label_array}  # or use a PolyData
        grid = points_to_imagedata(pts, grid_shape=(100, 100, 50), point_data=point_data)

    Tumor = 1, others = 0:
        grid = points_to_imagedata(
            pc_model, grid_shape=(532, 400, 34),
            x_range=(6000, 14000), y_range=(-1000, 5000), z_range=(57.97, 1043.48),
            label_to_value={"pred_label": {"Tumor": 1}}, empty_voxel_value=0,
        )
    """
    if isinstance(points, PolyData):
        pts = np.asarray(points.points, dtype=float)
        if point_data is None:
            point_data = {k: np.asarray(points.point_data[k]) for k in points.point_data.keys()}
    else:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be (N, 3)")
        if point_data is None:
            point_data = {}

    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    n_pts = pts.shape[0]

    if x_range is None:
        x_range = (float(pts[:, 0].min()), float(pts[:, 0].max()))
    if y_range is None:
        y_range = (float(pts[:, 1].min()), float(pts[:, 1].max()))
    if z_range is None:
        z_range = (float(pts[:, 2].min()), float(pts[:, 2].max()))

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    # Map coordinates to voxel indices (0 .. n-1)
    xi = np.clip(((pts[:, 0] - x_min) / dx).astype(np.int64), 0, nx - 1)
    yi = np.clip(((pts[:, 1] - y_min) / dy).astype(np.int64), 0, ny - 1)
    zi = np.clip(((pts[:, 2] - z_min) / dz).astype(np.int64), 0, nz - 1)

    # ImageData: dimensions = (nx+1, ny+1, nz+1) -> nx*ny*nz cells
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.origin = (x_min, y_min, z_min)
    grid.spacing = (dx, dy, dz)

    for key, arr in point_data.items():
        if key.endswith("_rgba"):
            continue
        arr = np.asarray(arr)
        if arr.shape[0] != n_pts:
            continue
        if arr.ndim > 1:
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                continue

        # Map factor labels to numbers if requested (e.g. {"Tumor": 1})
        if label_to_value is not None and key in label_to_value:
            mapping = label_to_value[key]
            arr = np.array([mapping.get(x, 0) for x in arr], dtype=np.float64)

        if empty_voxel_value is not None:
            volume = np.full((nx, ny, nz), empty_voxel_value, dtype=np.float64)
        elif np.issubdtype(arr.dtype, np.floating):
            volume = np.full((nx, ny, nz), np.nan, dtype=np.float64)
        else:
            volume = np.zeros((nx, ny, nz), dtype=arr.dtype)

        if fill_mode == "last":
            volume[xi, yi, zi] = arr
        elif fill_mode == "mean" and np.issubdtype(arr.dtype, np.number):
            count = np.zeros((nx, ny, nz), dtype=np.float64)
            sum_ = np.zeros((nx, ny, nz), dtype=np.float64)
            np.add.at(sum_, (xi, yi, zi), arr.astype(np.float64))
            np.add.at(count, (xi, yi, zi), 1.0)
            np.divide(sum_, count, out=volume, where=count > 0)
            volume = np.where(count > 0, volume, np.nan)
        elif fill_mode == "mode":
            flat_idx = xi + nx * (yi + ny * zi)
            for u in np.unique(flat_idx):
                ix, iy, iz = u % nx, (u // nx) % ny, u // (nx * ny)
                sel = flat_idx == u
                vals, cnts = np.unique(arr[sel], return_counts=True)
                volume[ix, iy, iz] = vals[cnts.argmax()]
        else:
            volume[xi, yi, zi] = arr

        grid.cell_data[key] = volume.flatten(order="F")

    return grid


def expression_to_imagedata(
    points: Union[PolyData, np.ndarray],
    grid_shape: Tuple[int, int, int],
    expression: Union[dict, np.ndarray],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    gene_names: Optional[list] = None,
) -> ImageData:
    """
    Rasterize gene (or any continuous) expression into a 3D ImageData.

    Each voxel gets the **mean** expression of all points inside it. Use this
    for gene expression, intensity, or any numeric per-point quantity.

    Parameters
    ----------
    points : pyvista.PolyData or np.ndarray
        Coordinates, shape (N, 3). If PolyData, uses .points.
    grid_shape : tuple (nx, ny, nz)
        Number of voxels per axis. Voxel size = (range/n) per axis.
    expression : dict or np.ndarray
        If dict: name -> 1D array (length N) for each gene/feature.
        If array: shape (N,) for one gene, or (N, G) for G genes; then
        pass gene_names of length G (or default "gene_0", "gene_1", ...).
    x_range, y_range, z_range : tuple (min, max), optional
        Physical extent. If None, inferred from points.
    gene_names : list of str, optional
        Used when expression is (N, G); names for cell_data keys.
        If None, "gene_0", "gene_1", ... are used.

    Returns
    -------
    pyvista.ImageData
        Grid with cell_data[key] = mean expression per voxel for each key.
        Save as .vti or plot with three_d_plot(grid, key="GeneName", colormap="hot_r").

    Examples
    --------
    Single gene from adata:
        pts = adata.obsm["spatial"]
        expr = adata.X[:, gene_ix].toarray().ravel()
        grid = expression_to_imagedata(pts, (100, 100, 50), {"Myc": expr})

    Multiple genes (matrix):
        expr = adata.X[:, [i1, i2]].toarray()
        grid = expression_to_imagedata(
            adata.obsm["spatial"],
            (100, 100, 50),
            expr,
            gene_names=["GeneA", "GeneB"],
        )
    """
    if isinstance(points, PolyData):
        pts = np.asarray(points.points, dtype=float)
    else:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be (N, 3)")
    n_pts = pts.shape[0]

    if isinstance(expression, dict):
        expr_dict = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in expression.items()}
        for k, v in expr_dict.items():
            if v.shape[0] != n_pts:
                raise ValueError(f"expression['{k}'] length {v.shape[0]} != n_points {n_pts}")
    else:
        expr = np.asarray(expression, dtype=np.float64)
        if expr.ndim == 1:
            expr = expr.reshape(-1, 1)
        if expr.shape[0] != n_pts:
            raise ValueError(f"expression length {expr.shape[0]} != n_points {n_pts}")
        n_genes = expr.shape[1]
        names = gene_names if gene_names is not None else [f"gene_{j}" for j in range(n_genes)]
        if len(names) != n_genes:
            raise ValueError(f"gene_names length {len(names)} != expression columns {n_genes}")
        expr_dict = {names[j]: expr[:, j] for j in range(n_genes)}

    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    if x_range is None:
        x_range = (float(pts[:, 0].min()), float(pts[:, 0].max()))
    if y_range is None:
        y_range = (float(pts[:, 1].min()), float(pts[:, 1].max()))
    if z_range is None:
        z_range = (float(pts[:, 2].min()), float(pts[:, 2].max()))
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    xi = np.clip(((pts[:, 0] - x_min) / dx).astype(np.int64), 0, nx - 1)
    yi = np.clip(((pts[:, 1] - y_min) / dy).astype(np.int64), 0, ny - 1)
    zi = np.clip(((pts[:, 2] - z_min) / dz).astype(np.int64), 0, nz - 1)

    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.origin = (x_min, y_min, z_min)
    grid.spacing = (dx, dy, dz)

    for key, arr in expr_dict.items():
        sum_ = np.zeros((nx, ny, nz), dtype=np.float64)
        count = np.zeros((nx, ny, nz), dtype=np.float64)
        np.add.at(sum_, (xi, yi, zi), arr)
        np.add.at(count, (xi, yi, zi), 1.0)
        volume = np.full((nx, ny, nz), np.nan, dtype=np.float64)
        np.divide(sum_, count, out=volume, where=count > 0)
        grid.cell_data[key] = volume.flatten(order="F")

    return grid
