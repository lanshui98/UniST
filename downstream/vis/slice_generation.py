"""
Slice generation utilities for 3D point cloud visualization.

This module provides functions to generate and visualize slices from 3D point clouds
using plane-based slicing with PyVista.
"""

import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import List, Sequence, Optional, Tuple, Union, Dict, Any

ArrayLike = Union[np.ndarray, pv.PolyData]


def _as_polydata(pc: ArrayLike) -> pv.PolyData:
    """
    Convert input to PyVista PolyData.
    
    Parameters
    ----------
    pc : np.ndarray or pv.PolyData
        Point cloud data
    
    Returns
    -------
    pv.PolyData
        PyVista PolyData object
    """
    if isinstance(pc, pv.PolyData):
        return pc
    arr = np.asarray(pc, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    return pv.PolyData(arr)


def slice_points_plane(
    pc: ArrayLike,
    origin: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    width: float = 0.0,
    eps: float = 1e-6,
) -> pv.PolyData:
    """
    Extract points within a plane (or thin slab) from a point cloud.
    
    Parameters
    ----------
    pc : np.ndarray or pv.PolyData
        Point cloud data, shape (N, 3)
    origin : tuple of float
        Origin point of the plane (x, y, z)
    normal : tuple of float
        Normal vector of the plane (nx, ny, nz)
    width : float, default=0.0
        Thickness of the slice. If 0, extracts points exactly on the plane.
        If > 0, extracts points within width/2 distance from the plane.
    eps : float, default=1e-6
        Small epsilon for numerical stability
    
    Returns
    -------
    pv.PolyData
        PolyData containing points within the slice
    """
    P = _as_polydata(pc)
    pts = P.points
    n = np.asarray(normal, dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)
    
    # Calculate signed distance from each point to the plane
    d = (pts - np.asarray(origin)) @ n
    
    # Extract points within tolerance
    tol = max(width / 2.0, eps)
    mask = np.abs(d) <= tol
    
    return P.extract_points(
        np.where(mask)[0],
        adjacent_cells=False,
        include_cells=False
    )


def _range_along_normal(P: pv.PolyData, normal: Sequence[float]) -> Tuple[float, float]:
    """
    Calculate the range of point cloud along a normal direction.
    
    Parameters
    ----------
    P : pv.PolyData
        Point cloud
    normal : sequence of float
        Normal vector (nx, ny, nz)
    
    Returns
    -------
    tuple
        (min_projection, max_projection) along the normal direction
    """
    n = np.asarray(normal, dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)
    t = P.points @ n  # Projection of each point onto the normal
    return float(t.min()), float(t.max())


def generate_slice_origins(
    pc: ArrayLike,
    normal: Sequence[float] = (0, 0, 1),
    *,
    n_slices: Optional[int] = None,
    step: Optional[float] = None,
    margin: float = 0.0,
) -> List[np.ndarray]:
    """
    Generate a series of slice plane origins along a normal direction.
    
    Parameters
    ----------
    pc : np.ndarray or pv.PolyData
        Point cloud data
    normal : sequence of float, default=(0, 0, 1)
        Normal vector direction for slicing (nx, ny, nz)
    n_slices : int, optional
        Number of slices to generate (evenly spaced)
    step : float, optional
        Step size between slices
    margin : float, default=0.0
        Margin to shrink from both ends to avoid boundary noise
    
    Returns
    -------
    list of np.ndarray
        List of origin points for each slice plane
    
    Raises
    ------
    ValueError
        If both or neither n_slices and step are specified,
        or if range is too small after applying margin
    """
    if (n_slices is None) == (step is None):
        raise ValueError("Must specify exactly one of: n_slices or step")
    
    P = _as_polydata(pc)
    n = np.asarray(normal, dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)
    
    # Calculate range along normal
    tmin, tmax = _range_along_normal(P, n)
    tmin += margin
    tmax -= margin
    
    if tmax <= tmin:
        raise ValueError(
            f"Range too small or margin too large: "
            f"tmin={tmin}, tmax={tmax}, margin={margin}"
        )
    
    origins = []
    if n_slices is not None:
        # Generate evenly spaced slices
        for alpha in np.linspace(0.0, 1.0, n_slices):
            t = tmin * (1 - alpha) + tmax * alpha
            origins.append(n * t)
    else:
        # Generate slices with fixed step size
        tt = np.arange(tmin, tmax + 1e-12, step)
        origins = [n * t for t in tt]
    
    return origins


def slice_stack(
    pc: ArrayLike,
    normal: Sequence[float] = (0, 0, 1),
    *,
    n_slices: Optional[int] = None,
    step: Optional[float] = None,
    width: float = 0.0,
    margin: float = 0.0,
    eps: float = 1e-6,
) -> List[pv.PolyData]:
    """
    Generate a stack of slices from a point cloud.
    
    Parameters
    ----------
    pc : np.ndarray or pv.PolyData
        Point cloud data
    normal : sequence of float, default=(0, 0, 1)
        Normal vector direction for slicing (nx, ny, nz)
    n_slices : int, optional
        Number of slices to generate (evenly spaced)
    step : float, optional
        Step size between slices
    width : float, default=0.0
        Thickness of each slice
    margin : float, default=0.0
        Margin to shrink from both ends
    eps : float, default=1e-6
        Small epsilon for numerical stability
    
    Returns
    -------
    list of pv.PolyData
        List of PolyData objects, one for each slice
    """
    origins = generate_slice_origins(
        pc, normal,
        n_slices=n_slices,
        step=step,
        margin=margin
    )
    
    return [
        slice_points_plane(
            pc,
            origin=o,
            normal=normal,
            width=width,
            eps=eps
        )
        for o in origins
    ]


def _to_str_array(arr: np.ndarray) -> np.ndarray:
    """Convert object/bytes/None to strings uniformly (None/NaN -> 'Unknown')."""
    out = []
    for v in arr:
        if v is None:
            out.append("Unknown")
        elif isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            s = str(v)
            if s.lower() in ("nan", "none"):
                s = "Unknown"
            out.append(s)
    return np.asarray(out, dtype=object)


def make_region_cmap(region_colors: dict, data: np.ndarray):
    """
    Convert {region_name: hex} to (cmap, region_to_int, int_array)
    for direct use in PyVista add_mesh.
    
    Parameters
    ----------
    region_colors : dict
        Dictionary mapping region names to hex color strings (e.g., {'region1': '#FF0000'})
    data : np.ndarray
        Array of region names or labels (can be string or numeric)
    
    Returns
    -------
    tuple
        (cmap, region_to_int, int_array)
        - cmap: matplotlib.colors.ListedColormap object
        - region_to_int: Dictionary mapping region names to integer indices
        - int_array: Integer array for visualization in PyVista
    """
    region_names = list(region_colors.keys())
    region_hex = [region_colors[n] for n in region_names]

    # Build cmap and mapping table
    cmap = mcolors.ListedColormap(region_hex, name="region_cmap")
    region_to_int = {name: i for i, name in enumerate(region_names)}

    # Convert string labels to int
    if data.dtype.type is np.str_ or data.dtype == object:
        int_array = np.array([region_to_int[val] for val in data])
    else:
        int_array = data

    return cmap, region_to_int, int_array


def build_region_mapping_for_slices(
    slices: Sequence[pv.PolyData],
    key: str,
    regions_colors: Dict[str, str],
    unknown_color: str = "#808080"
) -> Tuple[mcolors.ListedColormap, Dict[str, int], Sequence[str]]:
    """
    Build a unified label->int mapping and colormap for all slices using the same regions_colors.
    Any labels not in regions_colors are marked as 'Unknown' and given unknown_color.
    Returns: (cmap, label2id, id2label)
    """
    # Fixed order = the order of the dict you provide
    id2label = list(regions_colors.keys())
    label2id = {name: i for i, name in enumerate(id2label)}
    palette = [regions_colors[name] for name in id2label]

    # Check all slices and add unknown categories
    unknown_seen = False
    for s in slices:
        if s.n_points == 0 or key not in s.point_data:
            continue
        lbls = s.point_data[key]
        if lbls.dtype.kind not in ("U", "S", "O"):  # Convert numeric to string for uniformity
            lbls = lbls.astype(object)
        lbls = _to_str_array(lbls)
        uniq = np.unique(lbls)
        for u in uniq:
            if u not in label2id:
                unknown_seen = True
    if unknown_seen and "Unknown" not in label2id:
        label2id["Unknown"] = len(id2label)
        id2label.append("Unknown")
        palette.append(unknown_color)

    cmap = mcolors.ListedColormap(palette, name="heart_regions_cmap")
    return cmap, label2id, id2label


def plot_slice_stack(
    slices: Sequence[pv.PolyData],
    *,
    key: str = "heart_regions",
    regions_colors: Dict[str, str],
    point_size: float = 2.5,
    opacity: float = 1.0,
    background: str = "white",
    show_axes: bool = True,
    spacing_axis: int = 1,      # 0=x, 1=y, 2=z
    spacing: float = 0.0,       # If >0, separate adjacent slices
    use_parallel_projection: bool = False,
    show_colorbar: bool = True
) -> pv.Plotter:
    """
    Overlay all slices at once, color them using the same regions_colors; optionally separate slices along an axis.
    
    Parameters
    ----------
    slices : sequence of pv.PolyData
        List of slice PolyData objects
    key : str, default="heart_regions"
        Key in point_data to use for coloring
    regions_colors : dict
        Dictionary mapping region names to hex color strings (e.g., {'region1': '#FF0000'})
        Required parameter for consistent coloring across all slices
    point_size : float, default=2.5
        Size of points in the visualization
    opacity : float, default=1.0
        Opacity of the points (0-1)
    background : str, default="white"
        Background color
    show_axes : bool, default=True
        Whether to show coordinate axes
    spacing_axis : int, default=1
        Axis along which to space slices (0=x, 1=y, 2=z)
    spacing : float, default=0.0
        Spacing between adjacent slices. If > 0, slices are separated along spacing_axis
    use_parallel_projection : bool, default=False
        Enable parallel projection for cleaner cross-sections
    show_colorbar : bool, default=True
        Whether to show colorbar with region labels
    
    Returns
    -------
    pv.Plotter
        PyVista Plotter object
    
    Raises
    ------
    ValueError
        If no valid slices are found or spacing_axis is invalid
    """
    assert spacing_axis in (0, 1, 2), "spacing_axis must be 0(x)/1(y)/2(z)"
    pl = pv.Plotter()
    pl.set_background(background)
    if use_parallel_projection:
        pl.enable_parallel_projection()

    # Build unified cmap + mapping
    cmap, label2id, id2label = build_region_mapping_for_slices(slices, key, regions_colors)

    any_added = False
    for i, s in enumerate(slices):
        if s.n_points == 0:
            continue
        if key not in s.point_data:
            continue

        s2 = s.copy()
        # Separate slices with spacing
        if spacing > 0:
            pts = s2.points.copy()
            pts[:, spacing_axis] += i * spacing
            s2.points = pts

        # Convert labels to int uniformly
        arr = s2.point_data[key]
        arr = _to_str_array(arr)
        arr_int = np.array([label2id.get(v, label2id.get("Unknown")) for v in arr], dtype=int)
        s2.point_data[key] = arr_int

        pl.add_mesh(
            s2,
            scalars=key,
            cmap=cmap,
            render_points_as_spheres=False,
            style="points",
            point_size=point_size,
            opacity=opacity,
            clim=(0, len(id2label)-1)  # Fixed range to avoid color misalignment from auto-scaling
        )
        any_added = True

    if not any_added:
        raise ValueError(f"No valid slices found or missing point_data['{key}'].")

    if show_axes:
        pl.add_axes()

    # Customize colorbar ticks and labels for categorical data
    if show_colorbar:
        # Create a scalar_bar with custom ticks to display category names
        sb = pl.add_scalar_bar(title=key, n_labels=len(id2label))
        # Set ticks at integer positions
        ticks = list(range(len(id2label)))
        # matplotlib-style tick/label settings
        # Note: Sometimes the backend cannot directly modify scalar_bar ticks,
        # but fixed clim + ListedColormap + n_labels usually gives evenly spaced ticks.
        # For finer control, you can operate on pl.scalar_bars[key] after rendering (varies by version).
        try:
            # Some versions support:
            sb.SetNumberOfLabels(len(id2label))
        except Exception:
            pass
        # Note: If your environment's scalar bar cannot directly display category names,
        # you can disable colorbar and use on-screen text legend (draw boxes + text yourself).
        # Here we provide a simple solution:
        pl.add_text("  |  ".join(id2label), position="lower_edge", font_size=10, color="black")

    pl.show()
    return pl
