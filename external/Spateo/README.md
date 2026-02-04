# Spateo Point Cloud Utilities

This module provides point cloud construction utilities adapted from Spateo.

## Functions

### `construct_pc`

Construct a point cloud model based on 3D coordinate information from an AnnData object.

**Usage:**

```python
import sys
import os
import anndata as ad

# Add external/Spateo to Python path (adjust path as needed)
_spateo_path = os.path.join(os.path.dirname(__file__), 'external', 'Spateo')
if os.path.exists(_spateo_path) and _spateo_path not in sys.path:
    sys.path.insert(0, _spateo_path)

from Spateo import construct_pc

# Example: Create point cloud from AnnData with 3D spatial coordinates
pc, plot_cmap = construct_pc(
    adata=adata.copy(),
    spatial_key="aligned_spatial_3D",
    groupby="heart_regions",
    key_added="heart_regions",
    colormap=regions_colors
)

# The returned pc is a PyVista PolyData object with:
# - pc.point_data["heart_regions"]: the groupby information
# - pc.point_data["heart_regions_rgba"]: the rgba colors
# - pc.point_data["obs_index"]: the obs_index of each coordinate
```

**Parameters:**

- `adata`: AnnData object
- `layer`: If `'X'`, uses `.X`, otherwise uses the representation given by `.layers[layer]`
- `spatial_key`: The key in `.obsm` that corresponds to the spatial coordinate of each bucket
- `groupby`: The key that stores clustering or annotation information in `.obs`, a gene name or a list of gene names in `.var`
- `key_added`: The key under which to add the labels
- `mask`: The part that you don't want to be displayed
- `colormap`: Colors to use for plotting pc. The default colormap is `'rainbow'`
- `alphamap`: The opacity of the colors to use for plotting pc. The default alphamap is `1.0`

**Returns:**

- `pc`: A PyVista PolyData point cloud with groupby information and colors
- `plot_cmap`: Recommended colormap parameter values for plotting

### `add_model_labels`

Add rgba color to each point of a PyVista model based on labels.

**Usage:**

```python
from Spateo import add_model_labels

model, plot_cmap = add_model_labels(
    model=pc,
    labels=groups,
    key_added="heart_regions",
    where="point_data",
    colormap=regions_colors,
    alphamap=1.0,
    inplace=False
)
```

## Dependencies

- `numpy`
- `pyvista`
- `anndata`
- `pandas`
- `matplotlib`

### `three_d_plot`

Visualize reconstructed 3D model with comprehensive customization options.

**Usage:**

```python
import sys
import os

# Add external/Spateo to Python path
_spateo_path = os.path.join(os.path.dirname(__file__), '..', 'external', 'Spateo')
if _spateo_path not in sys.path:
    sys.path.insert(0, _spateo_path)

from Spateo import three_d_plot

# Example: Visualize point cloud model
three_d_plot(
    model=pc_model,
    key="annotation",
    model_size=2,
    model_style="points",
    opacity=1,
    background="white",
    cpo=saved_oblique_view,
    jupyter="trame",
)
```

**Parameters:**

- `model`: A reconstructed model (PolyData, UnstructuredGrid, or MultiBlock)
- `key`: The key under which are the labels
- `model_size`: Point size (if model_style='points') or line width (if model_style='wireframe')
- `model_style`: Visualization style: 'points', 'surface', or 'wireframe'
- `opacity`: Opacity of the model (0-1)
- `background`: Background color
- `cpo`: Camera position (can be 'iso', 'xy', 'xz', 'yz', or a list of [position, focal_point, view_up])
- `jupyter`: Jupyter backend ('trame', 'panel', 'static', 'none', etc.)
- `window_size`: Window size in pixels (default: (512, 512))
- `colormap`: Matplotlib colormap name (if None, uses {key}_rgba)
- `show_legend`: Whether to show legend
- `show_outline`: Whether to show outline
- `show_axes`: Whether to show coordinate axes
- `filename`: Output filename (supports .png, .gif, .mp4, etc.)
- `plotter_filename`: Filename to save plotter (supports .gltf, .html, .obj, .vtkjs)

**Returns:**

- `cpo`: Camera position (if applicable)
- `img`: Numpy array of the last image (if applicable)

## Notes

This module is adapted from Spateo's:
- `tdr.models.models_individual.point_clouds`
- `tdr.models.utilities.label_utils`
- `plotting.static.three_d_plot.three_dims_plots`
