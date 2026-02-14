"""
Point cloud construction from AnnData for 3D visualization.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
import matplotlib as mpl
from anndata import AnnData
from pandas import DataFrame
from pyvista import PolyData


def _add_model_labels(
    model: PolyData,
    labels: np.ndarray,
    key_added: str,
    where: str = "point_data",
    colormap: Union[str, list, dict] = "rainbow",
    alphamap: Union[float, list, dict] = 1.0,
    mask_color: Optional[str] = "gainsboro",
    mask_alpha: Optional[float] = 0.0,
    inplace: bool = True,
) -> Tuple[Optional[PolyData], Optional[str]]:
    """
    Add labels and optionally rgba to model point_data.
    For numeric labels (e.g. gene expression): only set point_data[key_added], return plot_cmap.
    For categorical: set point_data[key_added] and point_data[key_added_rgba], return None.
    """
    if not inplace:
        model = model.copy()
    labels = np.asarray(labels).flatten()

    if not np.issubdtype(labels.dtype, np.number):
        # Categorical: build _rgba
        cu_arr = np.sort(np.unique(labels), axis=0).astype(object)
        raw_hex = labels.copy().astype(object)
        raw_alpha = labels.copy().astype(object)
        raw_hex[raw_hex == "mask"] = mpl.colors.to_hex(mask_color)
        raw_alpha[raw_alpha == "mask"] = mask_alpha

        if isinstance(colormap, str):
            if colormap in list(mpl.colormaps()):
                lscmap = mpl.colormaps[colormap]
                hex_list = [mpl.colors.to_hex(lscmap(i)) for i in np.linspace(0, 1, len(cu_arr))]
                for lb, c in zip(cu_arr, hex_list):
                    raw_hex[raw_hex == lb] = c
            else:
                raw_hex[raw_hex != "mask"] = mpl.colors.to_hex(colormap)
        elif isinstance(colormap, dict):
            for lb, c in colormap.items():
                raw_hex[raw_hex == lb] = mpl.colors.to_hex(c)
        elif isinstance(colormap, (list, np.ndarray)):
            hex_list = [mpl.colors.to_hex(c) for c in colormap]
            for lb, c in zip(cu_arr, hex_list):
                raw_hex[raw_hex == lb] = c
        else:
            raise ValueError("colormap must be str, list, or dict.")

        if isinstance(alphamap, (float, int)):
            raw_alpha[raw_alpha != "mask"] = float(alphamap)
        elif isinstance(alphamap, dict):
            for lb, a in alphamap.items():
                raw_alpha[raw_alpha == lb] = a
        else:
            raw_alpha = np.asarray(alphamap).astype(object)

        rgba = np.array([mpl.colors.to_rgba(c, alpha=float(a)) for c, a in zip(raw_hex, raw_alpha)], dtype=np.float32)
        model.point_data[f"{key_added}_rgba"] = rgba
        plot_cmap = None
    else:
        plot_cmap = colormap if isinstance(colormap, str) else None

    model.point_data[key_added] = labels
    return (model if not inplace else None), plot_cmap


def construct_pc(
    adata: AnnData,
    layer: str = "X",
    spatial_key: str = "spatial",
    groupby: Union[str, tuple, None] = None,
    key_added: str = "groups",
    mask: Union[str, int, float, list, None] = None,
    colormap: Union[str, list, dict] = "rainbow",
    alphamap: Union[float, list, dict] = 1.0,
) -> Tuple[PolyData, Optional[str]]:
    """
    Construct a point cloud model from AnnData 3D coordinates (spateo-style API).

    Parameters
    ----------
    adata : AnnData
        AnnData with 3D coordinates in obsm[spatial_key].
    layer : str
        Use .X or .layers[layer] for gene expression.
    spatial_key : str
        Key in .obsm for spatial coordinates (x, y, z).
    groupby : str, tuple, or None
        Key in .obs (e.g. cell type), or a gene name / list of gene names in .var.
        If None, all points get the same label.
    key_added : str
        Key under which to add labels (and key_added_rgba for categorical).
        Use key_added="Myl2" when groupby="Myl2" so three_d_plot(..., key="Myl2") works.
    mask : str, int, float, list, or None
        Values to replace with "mask" in displayed groups.
    colormap : str, list, or dict
        Colormap name (e.g. "hot_r", "rainbow"), list of colors, or dict label -> color.
    alphamap : float, list, or dict
        Opacity.

    Returns
    -------
    pc : pv.PolyData
        Point cloud with point_data[key_added] and optionally point_data[key_added_rgba], point_data["obs_index"].
    plot_cmap : str or None
        Recommended colormap for three_d_plot (for continuous/gene data); None when using _rgba.
    """
    adata = adata.copy()
    bucket_xyz = adata.obsm[spatial_key].astype(np.float64)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pc = pv.PolyData(bucket_xyz)

    mask_list = list(mask) if isinstance(mask, list) else ([mask] if mask is not None else [])
    obs_names = set(adata.obs_keys())
    gene_names = set(adata.var_names.tolist())

    if groupby is None:
        groups = np.asarray(["same"] * adata.n_obs, dtype=object)
    elif groupby in obs_names:
        groups = np.asarray(
            adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values,
            dtype=object,
        )
    elif groupby in gene_names or (isinstance(groupby, (list, tuple)) and set(groupby) <= gene_names):
        X = adata.X if layer == "X" else adata.layers[layer]
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(groupby, str):
            gidx = adata.var_names.get_loc(groupby)
            groups = np.asarray(X[:, gidx].flatten(), dtype=float)
        else:
            cols = [adata.var_names.get_loc(g) for g in groupby]
            groups = np.asarray(X[:, cols].sum(axis=1).flatten(), dtype=float)
    else:
        raise ValueError(
            "groupby must be a key in adata.obs_names or adata.var_names, "
            "or a list of adata.var_names."
        )

    _, plot_cmap = _add_model_labels(
        model=pc,
        labels=groups,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
        inplace=True,
    )
    pc.point_data["obs_index"] = np.array(adata.obs_names.tolist(), dtype=object)
    return pc, plot_cmap
