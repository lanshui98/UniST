"""
3D plotting utilities adapted from Spateo.

This module provides functions for visualizing 3D models using PyVista.
"""

from typing import Optional, Tuple, Union, Dict, List, Literal
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import pyvista as pv
from pyvista import PolyData, UnstructuredGrid, MultiBlock, Plotter

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _get_default_cmap():
    """Get or create default colormap."""
    if "default_cmap" not in mpl.colormaps():
        colors = ["#4B0082", "#800080", "#F97306", "#FFA500", "#FFD700", "#FFFFCB"]
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        from matplotlib.colors import LinearSegmentedColormap
        mpl.colormaps.register(LinearSegmentedColormap.from_list("default_cmap", list(zip(nodes, colors))))
    return "default_cmap"


def _set_jupyter(
    jupyter: Union[bool, Literal["trame", "panel", "none", "static", "html"]] = False,
    off_screen: bool = False,
):
    """Set jupyter backend and off_screen flags."""
    if jupyter is False:
        off_screen1, off_screen2 = off_screen, True
        jupyter_backend = "none"
    elif jupyter is True:
        off_screen1, off_screen2 = True, off_screen
        jupyter_backend = "static"
    elif jupyter in ["trame", "panel", "none", "static", "html"]:
        off_screen1, off_screen2 = True, off_screen
        jupyter_backend = jupyter
    else:
        raise ValueError(
            "`jupyter` value is wrong."
            "\nAvailable `jupyter` value are: `True`, `False`, `'trame'`, `'none'`, `'static'`."
        )
    return off_screen1, off_screen2, jupyter_backend


def create_plotter(
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (512, 512),
    background: str = "white",
    shape: Union[str, list, tuple] = (1, 1),
    show_axes: bool = True,
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk model.

    Args:
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        shape: Number of sub-render windows inside the main window. By default, there is only one render window.
        show_axes: Whether to add a camera orientation widget to the active renderer.
    
    Returns:
        plotter: The plotting object to display pyvista/vtk model.
    """
    _get_default_cmap()
    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=window_size,
        notebook=False if jupyter is False else True,
        lighting="light_kit",
        shape=shape,
    )
    plotter.background_color = background
    
    if jupyter != "trame":
        if show_axes:
            if shape == (1, 1):
                plotter.add_camera_orientation_widget()
            else:
                plotter.add_axes()
    return plotter


def add_model(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    key: Union[str, list] = None,
    colormap: Optional[Union[str, list]] = None,
    clim: Optional[Union[tuple, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, list, np.ndarray] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
):
    """
    Add model(s) to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.
                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        clim: Color limits for the colormap. A tuple or list of (min, max) values to set the color range.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0).
        opacity: Opacity of the model.
        model_style: Visualization style of the model. One of: 'surface', 'wireframe', 'points'.
        model_size: If model_style='points', point size. If model_style='wireframe', thickness of lines.
    """
    def _add_model(_p, _model, _key, _colormap, _clim, _style, _ambient, _opacity, _model_size):
        """Add any PyVista/VTK model to the scene."""
        if _style == "points":
            render_spheres, render_tubes, smooth_shading = True, False, True
        elif _style == "wireframe":
            render_spheres, render_tubes, smooth_shading = False, True, False
        else:
            render_spheres, render_tubes, smooth_shading = False, False, True
        
        mesh_kwargs = dict(
            style=_style,
            render_points_as_spheres=render_spheres,
            render_lines_as_tubes=render_tubes,
            point_size=_model_size,
            line_width=_model_size,
            ambient=_ambient,
            opacity=_opacity,
            smooth_shading=smooth_shading,
            show_scalar_bar=False,
        )

        if _colormap is None:
            added_kwargs = dict(
                scalars=f"{_key}_rgba" if _key in _model.array_names else _model.active_scalars_name,
                rgba=True,
            )
        else:
            added_kwargs = (
                dict(
                    scalars=_key if _key in _model.array_names else _model.active_scalars_name,
                    cmap=_colormap,
                    clim=_clim,
                )
                if _colormap in list(mpl.colormaps())
                else dict(color=_colormap)
            )

        mesh_kwargs.update(added_kwargs)
        _p.add_mesh(_model, **mesh_kwargs)

    # Add model(s) to the plotter.
    if isinstance(model, (MultiBlock, list)):
        n_model = len(model)
        keys = key if isinstance(key, list) else [key]
        keys = keys * n_model if len(keys) == 1 else keys
        cmaps = colormap if isinstance(colormap, list) else [colormap]
        cmaps = cmaps * n_model if len(cmaps) == 1 else cmaps
        clims = clim if isinstance(clim, list) else [clim]
        clims = clims * n_model if len(clims) == 1 else clims
        mts = model_style if isinstance(model_style, list) else [model_style]
        mts = mts * n_model if len(mts) == 1 else mts
        mss = model_size if isinstance(model_size, list) else [model_size]
        mss = mss * n_model if len(mss) == 1 else mss
        ams = ambient if isinstance(ambient, list) else [ambient]
        ams = ams * n_model if len(ams) == 1 else ams
        ops = opacity if isinstance(opacity, list) else [opacity]
        ops = ops * n_model if len(ops) == 1 else ops

        for sub_model, sub_key, sub_cmap, sub_clim, sub_mt, sub_ms, sub_am, sub_op in zip(
            model, keys, cmaps, clims, mts, mss, ams, ops
        ):
            _add_model(_p=plotter, _model=sub_model, _key=sub_key, _colormap=sub_cmap,
                      _clim=sub_clim, _style=sub_mt, _model_size=sub_ms, _ambient=sub_am, _opacity=sub_op)
    else:
        _add_model(_p=plotter, _model=model, _key=key, _colormap=colormap, _clim=clim,
                  _style=model_style, _model_size=model_size, _ambient=ambient, _opacity=opacity)


def add_str_legend(
    plotter: Plotter,
    labels: np.ndarray,
    colors: np.ndarray,
    legend_size: Optional[Tuple] = None,
    legend_loc: Literal[
        "upper right", "upper left", "lower left", "lower right",
        "center left", "center right", "lower center", "upper center", "center"
    ] = "lower right",
):
    """
    Add a string legend to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        labels: Legend labels.
        colors: The colors corresponding to the legend labels.
        legend_size: The percentage (0 to 1) width and height of the window for the legend.
        legend_loc: The location of the legend in the window.
    """
    legend_data = np.concatenate(
        [labels.reshape(-1, 1).astype(object), colors.reshape(-1, 1).astype(object)],
        axis=1,
    )
    legend_data = legend_data[legend_data[:, 0] != "mask", :]
    assert len(legend_data) != 0, "No legend can be added, please set `show_legend=False`."

    legend_entries = legend_data[np.lexsort(legend_data[:, ::-1].T)]
    if legend_size is None:
        legend_num = 10 if len(legend_entries) >= 10 else len(legend_entries)
        legend_size = (0.1 + 0.01 * legend_num, 0.1 + 0.012 * legend_num)

    plotter.add_legend(
        legend_entries.tolist(),
        face="circle",
        bcolor=None,
        loc=legend_loc,
        size=legend_size,
    )


def add_legend(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[Union[str, list]] = None,
    colormap: Optional[Union[str, list]] = None,
    legend_size: Optional[Tuple] = None,
    legend_loc: Union[str, tuple, list] = None,
    title: Optional[str] = "",
    title_font_size: Optional[Union[int, float]] = None,
    label_font_size: Optional[Union[int, float]] = None,
    font_color: str = "black",
    font_family: Literal["times", "courier", "arial"] = "arial",
    fmt="%.2e",
    n_labels: int = 5,
    vertical: bool = True,
):
    """
    Add a legend to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the legend labels.
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.
        legend_size: The percentage (0 to 1) width and height of the window for the legend.
        legend_loc: The location of the legend in the window.
        title: Title of the legend.
        title_font_size: Sets the size of the title font.
        label_font_size: Sets the size of the label font.
        font_color: The color of the font.
        font_family: The font of the text.
        fmt: printf format for labels.
        n_labels: Number of labels to use for the legend.
        vertical: Use vertical or horizontal legend.
    """
    if colormap is None:
        assert key is not None, "When colormap is None, key cannot be None at the same time."

        if isinstance(model, MultiBlock):
            keys = key if isinstance(key, list) else [key] * len(model)
            legend_label_data, legend_color_data = [], []
            for m, k in zip(model, keys):
                legend_label_data.append(np.asarray(m[k]).flatten())
                legend_color_data.append(np.asarray([mpl.colors.to_hex(i) for i in m[f"{k}_rgba"]]).flatten())
            legend_label_data = np.concatenate(legend_label_data, axis=0)
            legend_color_data = np.concatenate(legend_color_data, axis=0)
        else:
            legend_label_data = np.asarray(model[key]).flatten()
            legend_color_data = np.asarray([mpl.colors.to_hex(i) for i in model[f"{key}_rgba"]]).flatten()
        legend_data = np.concatenate([legend_label_data.reshape(-1, 1), legend_color_data.reshape(-1, 1)], axis=1)
        unique_legend_data = np.unique(legend_data, axis=0)

        add_str_legend(
            plotter=plotter,
            labels=unique_legend_data[:, 0],
            colors=unique_legend_data[:, 1],
            legend_size=legend_size,
            legend_loc="lower right" if legend_loc is None else legend_loc,
        )
    else:
        plotter.add_scalar_bar(
            title=title,
            n_labels=n_labels,
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            color=font_color,
            font_family=font_family,
            use_opacity=True,
            width=(0.1, 0.4) if legend_size is None else legend_size[0],
            height=(0.1, 0.4) if legend_size is None else legend_size[1],
            position_x=(0.85, 0.3) if legend_loc is None else legend_loc[0] if isinstance(legend_loc, (tuple, list)) else 0.85,
            position_y=(0.85, 0.3) if legend_loc is None else legend_loc[1] if isinstance(legend_loc, (tuple, list)) else 0.3,
            vertical=vertical,
            fmt=fmt,
        )


def add_outline(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    outline_width: float = 5.0,
    outline_color: Union[str, tuple] = "black",
    show_labels: bool = True,
    font_size: int = 16,
    font_color: Union[str, tuple] = "white",
    font_family: Literal["times", "courier", "arial"] = "arial",
):
    """
    Produce an outline of the full extent for the model.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        outline_width: The width of the outline.
        outline_color: The color of the outline.
        show_labels: Whether to add the length, width and height information of the model to the outline.
        font_family: The font family of the text.
        font_size: The size of the font.
        font_color: The color of the font.
    """
    model_outline = model.outline()
    plotter.add_mesh(model_outline, color=outline_color, line_width=outline_width)

    if show_labels is True:
        mo_points = np.asarray(model_outline.points)
        model_x = mo_points[:, 0].max() - mo_points[:, 0].min()
        model_y = mo_points[:, 1].max() - mo_points[:, 1].min()
        model_z = mo_points[:, 2].max() - mo_points[:, 2].min()
        model_x, model_y, model_z = (
            round(model_x.astype(float), 5),
            round(model_y.astype(float), 5),
            round(model_z.astype(float), 5),
        )

        momid_points = [
            mo_points[1, :] - [model_x / 2, 0, 0],
            mo_points[1, :] + [0, model_y / 2, 0],
            mo_points[1, :] + [0, 0, model_z / 2],
        ]
        momid_labels = [model_x, model_y, model_z]
        plotter.add_point_labels(
            points=momid_points,
            labels=momid_labels,
            bold=True,
            font_size=font_size,
            font_family=font_family,
            shape="rounded_rect",
            shape_color=outline_color,
            show_points=False,
            text_color=font_color,
        )


def add_text(
    plotter: Plotter,
    text: str,
    font_family: Literal["times", "courier", "arial"] = "arial",
    font_size: Union[int, float] = 15,
    font_color: Union[str, tuple, list] = "black",
    text_loc: Literal[
        "lower_left", "lower_right", "upper_left", "upper_right",
        "lower_edge", "upper_edge", "right_edge", "left_edge"
    ] = "upper_left",
):
    """
    Add text to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        text: The text to add the rendering.
        font_family: The font family of the text.
        font_size: The size of the font.
        font_color: The color of the font.
        text_loc: The location of the text in the window.
    """
    plotter.add_text(
        text=text,
        font=font_family,
        color=font_color,
        font_size=font_size,
        position=text_loc,
    )


def wrap_to_plotter(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Union[str, list] = None,
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    clim: Optional[Union[tuple, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
):
    """
    What needs to be added to the visualization window.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
        background: The background color of the window.
        cpo: Camera position of the active render window.
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.
        clim: Color limits for the colormap.
        ambient: When lighting is enabled, this is the amount of light.
        opacity: Opacity of the model.
        model_style: Visualization style of the model.
        model_size: Point size or line width.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the add_legend function.
        show_outline: whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the add_outline function.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the add_text function.
    """
    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    # Add model(s) to the plotter.
    add_model(
        plotter=plotter,
        model=model,
        key=key,
        colormap=colormap,
        clim=clim,
        ambient=ambient,
        opacity=opacity,
        model_size=model_size,
        model_style=model_style,
    )

    # Set the camera position of plotter.
    plotter.camera_position = cpo

    # Add a legend to the plotter.
    if show_legend:
        lg_kwargs = dict(
            title=key if isinstance(key, str) else key[-1],
            legend_size=None,
            legend_loc=None,
            font_color=cbg_rgb,
            title_font_size=-1,
            label_font_size=12,
            font_family="arial",
            fmt="%.2e",
            n_labels=5,
            vertical=True,
        )
        if not (legend_kwargs is None):
            lg_kwargs.update((k, legend_kwargs[k]) for k in lg_kwargs.keys() & legend_kwargs.keys())

        add_legend(
            plotter=plotter,
            model=model,
            key=key,
            colormap=colormap,
            **lg_kwargs,
        )

    # Add an outline to the plotter.
    if show_outline:
        ol_kwargs = dict(
            outline_width=5.0,
            outline_color=cbg_rgb,
            show_labels=True,
            font_size=16,
            font_color=bg_rgb,
            font_family="arial",
        )
        if not (outline_kwargs is None):
            ol_kwargs.update((k, outline_kwargs[k]) for k in ol_kwargs.keys() & outline_kwargs.keys())
        add_outline(plotter=plotter, model=model, **ol_kwargs)

    # Add text to the plotter.
    if not (text is None):
        t_kwargs = dict(
            font_family="arial",
            font_size=12,
            font_color=cbg_rgb,
            text_loc="upper_left",
        )
        if not (text_kwargs is None):
            t_kwargs.update((k, text_kwargs[k]) for k in t_kwargs.keys() & text_kwargs.keys())
        add_text(plotter=plotter, text=text, **t_kwargs)


def output_plotter(
    plotter: Plotter,
    filename: Optional[str] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    jupyter: Union[bool, Literal["trame", "panel", "none", "static", "html"]] = False,
):
    """
    Output plotter as image, gif file or mp4 file.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
        view_up: The normal to the orbital plane. Only available when filename ending with .mp4 or .gif.
        framerate: Frames per second. Only available when filename ending with .mp4 or .gif.
        jupyter: Whether to plot in jupyter notebook.

    Returns:
        cpo: List of camera position, focal point, and view up (if applicable).
        img: Numpy array of the last image (if applicable).
    """
    _, _, jupyter_backend = _set_jupyter(jupyter=jupyter)

    if filename is None:
        if jupyter is False or jupyter == "none":
            if plotter.shape == (1, 1):
                cpo, img = plotter.show(return_img=True, return_cpos=True, jupyter_backend=jupyter_backend)
                return cpo, img
            else:
                plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show(jupyter_backend=jupyter_backend)
    else:
        filename_format = filename.split(".")[-1]

        if filename_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
            if plotter.shape == (1, 1):
                cpo, img = plotter.show(
                    screenshot=filename,
                    return_img=True,
                    return_cpos=True,
                    jupyter_backend=jupyter_backend
                )
                return cpo, img
            else:
                plotter.show(screenshot=filename, jupyter_backend=jupyter_backend)
        elif filename_format in ["svg", "eps", "ps", "pdf", "tex"]:
            plotter.save_graphic(filename, title="PyVista Export", raster=True, painter=True)
            if plotter.shape == (1, 1):
                cpo, img = plotter.show(return_img=True, return_cpos=True, jupyter_backend=jupyter_backend)
                return cpo, img
            else:
                plotter.show(jupyter_backend=jupyter_backend)
        elif filename_format == "gif":
            path = plotter.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
            plotter.open_gif(filename)
            plotter.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
            plotter.close()
            return None
        elif filename_format == "mp4":
            path = plotter.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
            plotter.open_movie(filename, framerate=framerate, quality=5)
            plotter.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
            plotter.close()
            return None
        else:
            raise ValueError(
                "\nFilename is wrong."
                "\nIf outputting an image file, please enter a filename ending with "
                "`.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`, `.svg`, `.eps`, `.ps`, `.pdf`, `.tex`."
                "\nIf outputting a gif file, please enter a filename ending with `.gif`."
                "\nIf outputting a mp4 file, please enter a filename ending with `.mp4`."
            )


def save_plotter(
    plotter: Plotter,
    filename: str,
):
    """
    Save plotter as gltf file, html file, obj file or vtkjs file.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        filename: The filename of the file where the plotter is saved.
                  Writer type is inferred from the extension of the filename.
    """
    filename_format = filename.split(".")[-1]

    if filename_format == "gltf":
        plotter.export_gltf(filename)
    elif filename_format == "html":
        plotter.export_html(filename)
    elif filename_format == "obj":
        plotter.export_obj(filename)
    else:
        plotter.export_vtkjs(filename)


def three_d_plot(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Union[str, list] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany", "html", "trame"]] = False,
    off_screen: bool = False,
    window_size: tuple = (512, 512),
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    clim: Optional[Union[tuple, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 24,
    plotter_filename: Optional[str] = None,
    show_axes: bool = False,
):
    """
    Visualize reconstructed 3D model.

    This function is adapted from Spateo's st.pl.three_d_plot.

    Args:
        model: A reconstructed model.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
        jupyter: Whether to plot in jupyter notebook. Available options:
                'none', 'pythreejs', 'static', 'ipygany', 'panel', 'html', 'trame'
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is [512, 512].
        background: The background color of the window.
        cpo: Camera position of the active render window. Available options:
             - Iterable containing position, focal_point, and view up.
             - Iterable containing a view vector.
             - A string: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.
                  When the colormap is None, use {key}_rgba to map the scalars.
        clim: Color limits for the colormap. A tuple or list of (min, max) values.
        ambient: When lighting is enabled, this is the amount of light (0 to 1).
        opacity: Opacity of the model (0 to 1).
        model_style: Visualization style of the model: 'surface', 'wireframe', or 'points'.
        model_size: If model_style='points', point size. If model_style='wireframe', thickness of lines.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the add_legend function.
        show_outline: whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the add_outline function.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the add_text function.
        view_up: The normal to the orbital plane. Only available when filename ending with .mp4 or .gif.
        framerate: Frames per second. Only available when filename ending with .mp4 or .gif.
        plotter_filename: The filename of the file where the plotter is saved.
        show_axes: Whether to show coordinate axes.

    Returns:
        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with image extensions.
        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with image extensions.
    """
    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
        show_axes=show_axes,
    )
    model_kwargs = dict(
        background=background,
        colormap=colormap,
        clim=clim,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        show_outline=show_outline,
        outline_kwargs=outline_kwargs,
        text=text,
        text_kwargs=text_kwargs,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    wrap_to_plotter(plotter=p, model=model, key=key, cpo=cpo, **model_kwargs)
    cpo = p.show(return_cpos=True, jupyter_backend="none", cpos=cpo)

    # Create another plotting object to save pyvista/vtk model.
    p = create_plotter(off_screen=off_screen2, **plotter_kws)
    wrap_to_plotter(plotter=p, model=model, key=key, cpo=cpo, **model_kwargs)

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)

    # Output the plotting object.
    return output_plotter(
        plotter=p,
        filename=filename,
        view_up=view_up,
        framerate=framerate,
        jupyter=jupyter_backend,
    )
