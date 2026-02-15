"""
3D plot for PyVista models (point cloud, mesh).

API compatible with spateo's st.pl.three_d_plot for use with construct_pc.
"""

from typing import Optional, Union, Dict, Any

import numpy as np
import pyvista as pv
import matplotlib as mpl
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _legend_loc_to_xy(loc, size, border=0.05):
    """Convert legend_loc (str or tuple) to (x, y) for SetPosition. Mirrors PyVista map_loc_to_pos."""
    if isinstance(loc, (tuple, list)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    loc = str(loc).lower()
    w, h = size[0], size[1]
    if "right" in loc:
        x = 1 - w - border
    elif "left" in loc:
        x = border
    else:
        x = 0.5 - w / 2
    if "upper" in loc:
        y = 1 - h - border
    elif "lower" in loc:
        y = border
    else:
        y = 0.5 - h / 2
    return x, y


def three_d_plot(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[Union[str, list]] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany", "html", "trame"]] = False,
    off_screen: bool = False,
    window_size: tuple = (512, 512),
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    ambient: float = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Literal["points", "surface", "wireframe"] = "surface",
    model_size: float = 3.0,
    show_legend: bool = True,
    legend_loc: Union[str, tuple] = "lower right",
    legend_size: tuple = (0.1, 0.4),
    legend_kwargs: Optional[Dict[str, Any]] = None,
    show_axes: bool = False,
    text: Optional[str] = None,
    text_kwargs: Optional[Dict[str, Any]] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 24,
):
    """
    Visualize 3D model (point cloud or mesh) with PyVista (spateo-style API).

    Use with construct_pc:
        pc, cmap = construct_pc(adata, spatial_key="spatial", groupby="Myl2", colormap="hot_r")
        three_d_plot(pc, key="Myl2", colormap="hot_r", model_style="points", model_size=4.0,
                     show_legend=True, jupyter="trame", opacity=0.5)

    Parameters
    ----------
    model : pv.PolyData, UnstructuredGrid, or MultiBlock
        Model to plot (e.g. from construct_pc).
    key : str or list, optional
        Key in point_data for coloring. If None and model has one scalar array, it is used.
    filename : str, optional
        Save to image (e.g. .png) or animation (.gif, .mp4).
    jupyter : bool or str
        False: desktop window. True or "static": static image in notebook.
        "trame": trame widget. "none", "pythreejs", "ipygany", "html" also supported.
    off_screen : bool
        Render off-screen (e.g. for saving).
    window_size : tuple
        (width, height) in pixels.
    background : str
        Background color.
    cpo : str or list
        Camera position: "iso", "xy", "xz", "yz", or [position, focal_point, view_up].
    colormap : str or list, optional
        Matplotlib colormap name (e.g. "hot_r"). If None, use point_data[key_rgba] for coloring.
    ambient : float
        Ambient light (0–1).
    opacity : float
        Opacity of the model (0–1).
    model_style : "points" | "surface" | "wireframe"
        Rendering style.
    model_size : float
        Point size when model_style="points"; line width for wireframe.
    show_legend : bool
        Show scalar bar (continuous) or legend (categorical).
    legend_loc : str or tuple, default="lower right"
        Legend position. String: "lower right", "upper left", "center right", etc.
        Tuple (x, y): custom normalized position 0-1, e.g. (0.82, 0.35) for right-side vertical center.
    legend_size : tuple, default=(0.1, 0.4)
        Legend box size (width, height) in normalized 0-1; narrow bar like spateo.
    legend_kwargs : dict, optional
        Passed to legend logic; overrides legend_loc/legend_size when provided.
    show_axes : bool
        Show axes widget.
    text : str, optional
        Text to overlay on the plot.
    text_kwargs : dict, optional
        Passed to plotter.add_text: font_size, font_family, color (or font_color), position (text_loc).
    view_up : tuple, default=(0.5, 0.5, 1)
        Normal to orbital plane for .mp4/.gif generation.
    framerate : int, default=24
        Frames per second for .mp4/.gif.
    """
    _legend_loc = legend_loc
    _legend_size = legend_size
    if legend_kwargs:
        _legend_loc = legend_kwargs.get("legend_loc", _legend_loc)
        _legend_size = legend_kwargs.get("legend_size", _legend_size)

    if key is None:
        if hasattr(model, "array_names") and model.array_names:
            key = model.array_names[0]
            if key == "obs_index":
                key = model.array_names[1] if len(model.array_names) > 1 else None
        if key is None:
            raise ValueError("key must be provided or model must have point_data arrays.")

    # Resolve jupyter backend
    if jupyter is False:
        jupyter_backend = "none"
        off_screen_plot = off_screen
    elif jupyter is True:
        jupyter_backend = "static"
        off_screen_plot = True
    elif isinstance(jupyter, str):
        jupyter_backend = jupyter
        off_screen_plot = True
    else:
        jupyter_backend = "static"
        off_screen_plot = True

    plotter = pv.Plotter(
        off_screen=off_screen_plot,
        window_size=window_size,
        notebook=(jupyter_backend != "none"),
        lighting="light_kit",
    )
    plotter.background_color = background

    def _add_one(_model, _key, _cmap, _style, _size, _ambient, _opacity):
        render_spheres = _style == "points"
        render_tubes = _style == "wireframe"
        smooth = _style == "surface" or _style == "points"
        kwargs = dict(
            style=_style,
            render_points_as_spheres=render_spheres,
            render_lines_as_tubes=render_tubes,
            point_size=_size,
            line_width=_size,
            ambient=_ambient,
            opacity=_opacity,
            smooth_shading=smooth,
            show_scalar_bar=False,
        )
        rgba_key = f"{_key}_rgba"
        if _cmap is None and hasattr(_model, "point_data") and rgba_key in _model.point_data:
            kwargs["scalars"] = rgba_key
            kwargs["rgba"] = True
        else:
            arr_name = _key if (hasattr(_model, "array_names") and _key in _model.array_names) else None
            if arr_name is None and hasattr(_model, "point_data"):
                arr_name = _key if _key in _model.point_data else list(_model.point_data.keys())[0]
            kwargs["scalars"] = arr_name
            if _cmap and _cmap in list(mpl.colormaps()):
                kwargs["cmap"] = _cmap
            elif _cmap and isinstance(_cmap, str):
                kwargs["color"] = _cmap
        plotter.add_mesh(_model, **kwargs)

    if isinstance(model, (MultiBlock, list)):
        n_models = len(model)
        keys = key if isinstance(key, list) else [key] * n_models
        cmaps = colormap if isinstance(colormap, list) else [colormap] * n_models
        for m, k, cm in zip(model, keys, cmaps):
            _add_one(m, k, cm, model_style, model_size, ambient, opacity)
    else:
        _add_one(model, key, colormap, model_style, model_size, ambient, opacity)

    plotter.camera_position = cpo

    if show_legend:
        if colormap is None and hasattr(model, "point_data"):
            # Categorical legend from key_rgba (spateo add_str_legend style)
            m = model[0] if isinstance(model, (MultiBlock, list)) else model
            rk = f"{key}_rgba"
            if rk in m.point_data and key in m.point_data:
                lbls = np.asarray(m.point_data[key]).flatten()
                rgba = np.asarray(m.point_data[rk])
                if len(rgba.shape) == 2 and rgba.shape[0] == len(lbls):
                    uniq = np.unique(lbls)
                    uniq = uniq[uniq != "mask"]
                    if len(uniq) <= 20:
                        hex_colors = [mpl.colors.to_hex(rgba[np.where(lbls == u)[0][0]]) for u in uniq]
                        legend_entries = list(zip(uniq.astype(str).tolist(), hex_colors))
                        legend_num = len(legend_entries)
                        sz = _legend_size
                        if sz is None:
                            sz = (0.1 + 0.01 * legend_num, 0.1 + 0.012 * legend_num)
                        elif isinstance(sz, (int, float)):
                            sz = (sz, sz)
                        leg = plotter.add_legend(
                            legend_entries, face="circle", bcolor=None, loc=None, size=sz
                        )
                        x, y = _legend_loc_to_xy(_legend_loc, sz)
                        leg.SetPosition(x, y)
                        leg.SetPosition2(sz[0], sz[1])
        else:
            sb_kw = dict(title="", vertical=True)
            sz_sb = _legend_size
            if sz_sb is None:
                sz_sb = (0.1, 0.4)
            elif isinstance(sz_sb, (int, float)):
                sz_sb = (sz_sb, sz_sb)
            sb_kw["width"] = sz_sb[0]
            sb_kw["height"] = sz_sb[1]
            if isinstance(_legend_loc, (tuple, list)) and len(_legend_loc) >= 2:
                sb_kw["position_x"] = _legend_loc[0]
                sb_kw["position_y"] = _legend_loc[1]
            else:
                x, y = _legend_loc_to_xy(_legend_loc, sz_sb)
                sb_kw["position_x"] = x
                sb_kw["position_y"] = y
            plotter.add_scalar_bar(**sb_kw)

    if show_axes:
        plotter.add_axes()

    if text is not None:
        tk = text_kwargs or {}
        plotter.add_text(
            text=text,
            font=tk.get("font_family", tk.get("font", "arial")),
            font_size=tk.get("font_size", 12),
            color=tk.get("color", tk.get("font_color", "black")),
            position=tk.get("position", tk.get("text_loc", "upper_left")),
        )

    if filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("png", "tif", "tiff", "bmp", "jpeg", "jpg"):
            plotter.screenshot(filename)
        elif ext in ("svg", "eps", "ps", "pdf"):
            plotter.save_graphic(filename, raster=True, painter=True)
        elif ext == "gif":
            path = plotter.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
            plotter.open_gif(filename)
            plotter.orbit_on_path(path, write_frames=True, viewup=view_up, step=0.1)
        elif ext == "mp4":
            path = plotter.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
            plotter.open_movie(filename, framerate=framerate, quality=5)
            plotter.orbit_on_path(path, write_frames=True, viewup=view_up, step=0.1)
        plotter.close()
        return

    return plotter.show(
        return_cpos=True,
        jupyter_backend=jupyter_backend,
        cpos=cpo,
    )
