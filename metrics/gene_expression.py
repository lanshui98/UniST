"""
Gene expression evaluation metrics.

This module provides metrics for evaluating gene expression predictions,
including rasterization to uniform grids and various similarity metrics
for both 2D and 3D spatial transcriptomics data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from scipy.sparse import csr_matrix, vstack
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import anndata as ad


def get_slice_union_bounds_and_bins(
    adata_true: ad.AnnData,
    recon_list: List[ad.AnnData],
    bin_size: float,
    true_coords_key: str = "spatial",
    recon_coords_key: str = "spatial",
) -> Tuple[float, float, int, int]:
    """
    Get unified bounds and grid bins for a slice using union of true and reconstructed coordinates.
    
    For 2D: uses x, y coordinates
    For 3D: uses x, y coordinates (z is handled separately)
    
    Parameters
    ----------
    adata_true : AnnData
        Ground truth AnnData object
    recon_list : list of AnnData
        List of reconstructed AnnData objects
    bin_size : float
        Size of each bin for rasterization
    true_coords_key : str, default="spatial"
        Key in obsm for true coordinates
    recon_coords_key : str, default="spatial"
        Key in obsm for reconstructed coordinates
    
    Returns
    -------
    tuple
        (x_min, y_min, n_x_bins, n_y_bins)
    """
    # Get true coordinates (first 2 dimensions: x, y)
    coords_true = adata_true.obsm[true_coords_key][:, :2]
    
    # Get reconstructed coordinates
    coords_recon = []
    for res in recon_list:
        coords_recon.append(res.obsm[recon_coords_key][:, :2])
    
    # Union of all coordinates
    all_coords = np.vstack([coords_true] + coords_recon)
    
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    
    n_x_bins = int(np.ceil((x_max - x_min) / bin_size))
    n_y_bins = int(np.ceil((y_max - y_min) / bin_size))
    
    return float(x_min), float(y_min), int(n_x_bins), int(n_y_bins)


def get_volume_union_bounds_and_bins(
    adata_true: ad.AnnData,
    recon_list: List[ad.AnnData],
    bin_size: float,
    true_coords_key: str = "spatial",
    recon_coords_key: str = "spatial",
) -> Tuple[float, float, float, int, int, int]:
    """
    Get unified bounds and grid bins for a 3D volume using union of coordinates.
    
    Parameters
    ----------
    adata_true : AnnData
        Ground truth AnnData object
    recon_list : list of AnnData
        List of reconstructed AnnData objects
    bin_size : float
        Size of each bin for rasterization
    true_coords_key : str, default="spatial"
        Key in obsm for true coordinates
    recon_coords_key : str, default="spatial"
        Key in obsm for reconstructed coordinates
    
    Returns
    -------
    tuple
        (x_min, y_min, z_min, n_x_bins, n_y_bins, n_z_bins)
    """
    # Get true coordinates (x, y, z)
    coords_true = adata_true.obsm[true_coords_key][:, :3]
    
    # Get reconstructed coordinates
    coords_recon = []
    for res in recon_list:
        coords_recon.append(res.obsm[recon_coords_key][:, :3])
    
    # Union of all coordinates
    all_coords = np.vstack([coords_true] + coords_recon)
    
    x_min, y_min, z_min = all_coords.min(axis=0)
    x_max, y_max, z_max = all_coords.max(axis=0)
    
    n_x_bins = int(np.ceil((x_max - x_min) / bin_size))
    n_y_bins = int(np.ceil((y_max - y_min) / bin_size))
    n_z_bins = int(np.ceil((z_max - z_min) / bin_size))
    
    return float(x_min), float(y_min), float(z_min), int(n_x_bins), int(n_y_bins), int(n_z_bins)


def bin_and_pad_adata_2d(
    adata: ad.AnnData,
    bin_size: float,
    x_min: float,
    y_min: float,
    n_x_bins: int,
    n_y_bins: int,
    coords_key: str = "spatial"
) -> ad.AnnData:
    """
    Rasterize 2D AnnData to uniform grid bins.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    bin_size : float
        Size of each bin
    x_min : float
        Minimum x coordinate
    y_min : float
        Minimum y coordinate
    n_x_bins : int
        Number of bins in x direction
    n_y_bins : int
        Number of bins in y direction
    coords_key : str, default="spatial"
        Key in obsm for coordinates
    
    Returns
    -------
    AnnData
        Binned AnnData with mean expression per bin
    """
    coords = ((adata.obsm[coords_key][:, 0:2] - [x_min, y_min]) / bin_size).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, n_x_bins - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, n_y_bins - 1)
    bin_ids = coords[:, 0] * n_y_bins + coords[:, 1]
    
    X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
    bins = defaultdict(list)
    for i, b in enumerate(bin_ids):
        bins[b].append(i)
    
    total_bins = n_x_bins * n_y_bins
    n_genes = adata.shape[1]
    rows = []
    coords_binned = []
    
    for b in range(total_bins):
        if b in bins:
            mean_vec = X[bins[b]].mean(axis=0)
        else:
            mean_vec = csr_matrix((1, n_genes))
        rows.append(mean_vec)
        coords_binned.append((b // n_y_bins, b % n_y_bins))
    
    X_binned = vstack(rows).tocsr()
    coords_binned = np.array(coords_binned, dtype=int)
    
    out = ad.AnnData(X_binned)
    out.var_names = adata.var_names.copy()
    out.obsm[coords_key] = coords_binned
    out.obs_names = [f"{x}_{y}" for x, y in coords_binned]
    out.uns["__type"] = "UMI"
    return out


def bin_and_pad_adata_3d(
    adata: ad.AnnData,
    bin_size: float,
    x_min: float,
    y_min: float,
    z_min: float,
    n_x_bins: int,
    n_y_bins: int,
    n_z_bins: int,
    coords_key: str = "spatial"
) -> ad.AnnData:
    """
    Rasterize 3D AnnData to uniform grid bins.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    bin_size : float
        Size of each bin
    x_min : float
        Minimum x coordinate
    y_min : float
        Minimum y coordinate
    z_min : float
        Minimum z coordinate
    n_x_bins : int
        Number of bins in x direction
    n_y_bins : int
        Number of bins in y direction
    n_z_bins : int
        Number of bins in z direction
    coords_key : str, default="spatial"
        Key in obsm for coordinates
    
    Returns
    -------
    AnnData
        Binned AnnData with mean expression per bin
    """
    coords = ((adata.obsm[coords_key][:, 0:3] - [x_min, y_min, z_min]) / bin_size).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, n_x_bins - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, n_y_bins - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, n_z_bins - 1)
    bin_ids = coords[:, 0] * (n_y_bins * n_z_bins) + coords[:, 1] * n_z_bins + coords[:, 2]
    
    X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
    bins = defaultdict(list)
    for i, b in enumerate(bin_ids):
        bins[b].append(i)
    
    total_bins = n_x_bins * n_y_bins * n_z_bins
    n_genes = adata.shape[1]
    rows = []
    coords_binned = []
    
    for b in range(total_bins):
        if b in bins:
            mean_vec = X[bins[b]].mean(axis=0)
        else:
            mean_vec = csr_matrix((1, n_genes))
        rows.append(mean_vec)
        # Convert linear bin index back to 3D coordinates
        z = b % n_z_bins
        y = (b // n_z_bins) % n_y_bins
        x = b // (n_y_bins * n_z_bins)
        coords_binned.append((x, y, z))
    
    X_binned = vstack(rows).tocsr()
    coords_binned = np.array(coords_binned, dtype=int)
    
    out = ad.AnnData(X_binned)
    out.var_names = adata.var_names.copy()
    out.obsm[coords_key] = coords_binned
    out.obs_names = [f"{x}_{y}_{z}" for x, y, z in coords_binned]
    out.uns["__type"] = "UMI"
    return out


def mask_from_binned(adata_binned: ad.AnnData) -> np.ndarray:
    """
    Create binary mask from binned AnnData (bins with non-zero expression).
    
    Parameters
    ----------
    adata_binned : AnnData
        Binned AnnData object
    
    Returns
    -------
    np.ndarray
        Boolean mask (True for bins with expression > 0)
    """
    X = adata_binned.X
    if isinstance(X, csr_matrix):
        s = np.array(X.sum(axis=1)).ravel()
    else:
        s = np.sum(np.asarray(X), axis=1)
    return s > 0


def iou_from_binned_masks(true_binned: ad.AnnData, pred_binned: ad.AnnData) -> float:
    """
    Compute IoU from binned masks.
    
    Parameters
    ----------
    true_binned : AnnData
        Binned ground truth AnnData
    pred_binned : AnnData
        Binned predicted AnnData
    
    Returns
    -------
    float
        IoU score (0-1), or np.nan if union is empty
    """
    mt = mask_from_binned(true_binned)
    mp = mask_from_binned(pred_binned)
    inter = np.logical_and(mt, mp).sum()
    union = np.logical_or(mt, mp).sum()
    return np.nan if union == 0 else float(inter / union)


def masked_MSE(Yt: np.ndarray, Yp: np.ndarray) -> float:
    """
    Compute Mean Squared Error on masked regions (where Yt > 0).
    
    Parameters
    ----------
    Yt : np.ndarray
        Ground truth expression matrix
    Yp : np.ndarray
        Predicted expression matrix
    
    Returns
    -------
    float
        Masked MSE, or np.nan if no valid mask
    """
    m = (Yt > 0)
    return np.nan if not np.any(m) else float(((Yt[m] - Yp[m]) ** 2).mean())


def masked_MAE(Yt: np.ndarray, Yp: np.ndarray) -> float:
    """
    Compute Mean Absolute Error on masked regions (where Yt > 0).
    
    Parameters
    ----------
    Yt : np.ndarray
        Ground truth expression matrix
    Yp : np.ndarray
        Predicted expression matrix
    
    Returns
    -------
    float
        Masked MAE, or np.nan if no valid mask
    """
    m = (Yt > 0)
    return np.nan if not np.any(m) else float(np.abs(Yt[m] - Yp[m]).mean())


def cosine_similarity_mean(
    Yt: np.ndarray,
    Yp: np.ndarray,
    mask: bool = False,
    eps: float = 1e-8
) -> float:
    """
    Compute mean cosine similarity across bins/genes.
    
    Parameters
    ----------
    Yt : np.ndarray
        Ground truth expression matrix (bins x genes)
    Yp : np.ndarray
        Predicted expression matrix (bins x genes)
    mask : bool, default=False
        If True, only compute on bins where Yt > 0
    eps : float, default=1e-8
        Small epsilon to avoid division by zero
    
    Returns
    -------
    float
        Mean cosine similarity, or np.nan if no valid comparisons
    """
    Yt, Yp = np.asarray(Yt), np.asarray(Yp)
    cs = []
    
    for i in range(Yt.shape[0]):
        if mask:
            m = Yt[i] > 0
            if m.sum() == 0:
                continue
            xt, yp = Yt[i, m], Yp[i, m]
        else:
            xt, yp = Yt[i], Yp[i]
        
        den = float(np.linalg.norm(xt) * np.linalg.norm(yp))
        if den < eps:
            continue
        
        cs.append(float((xt * yp).sum()) / den)
    
    return float(np.mean(cs)) if cs else np.nan


def spearman_r_mean(Yt: np.ndarray, Yp: np.ndarray, mask: bool = False) -> float:
    """
    Compute mean Spearman correlation across bins/genes.
    
    Parameters
    ----------
    Yt : np.ndarray
        Ground truth expression matrix (bins x genes)
    Yp : np.ndarray
        Predicted expression matrix (bins x genes)
    mask : bool, default=False
        If True, only compute on bins where Yt > 0
    
    Returns
    -------
    float
        Mean Spearman correlation, or np.nan if no valid comparisons
    """
    Yt, Yp = np.asarray(Yt), np.asarray(Yp)
    vals = []
    
    for i in range(Yt.shape[0]):
        if mask:
            m = Yt[i] > 0
            if m.sum() < 2:
                continue
            x, y = Yt[i, m], Yp[i, m]
        else:
            x, y = Yt[i], Yp[i]
        
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue
        
        r = spearmanr(x, y).statistic
        if not np.isnan(r):
            vals.append(r)
    
    return float(np.mean(vals)) if vals else np.nan


def pearson_r_mean(Yt: np.ndarray, Yp: np.ndarray, mask: bool = False) -> float:
    """
    Compute mean Pearson correlation across bins/genes.
    
    Parameters
    ----------
    Yt : np.ndarray
        Ground truth expression matrix (bins x genes)
    Yp : np.ndarray
        Predicted expression matrix (bins x genes)
    mask : bool, default=False
        If True, only compute on bins where Yt > 0
    
    Returns
    -------
    float
        Mean Pearson correlation, or np.nan if no valid comparisons
    """
    Yt, Yp = np.asarray(Yt), np.asarray(Yp)
    vals = []
    
    for i in range(Yt.shape[0]):
        if mask:
            m = Yt[i] > 0
            if m.sum() < 2:
                continue
            x, y = Yt[i, m], Yp[i, m]
        else:
            x, y = Yt[i], Yp[i]
        
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue
        
        r = pearsonr(x, y).statistic
        if not np.isnan(r):
            vals.append(r)
    
    return float(np.mean(vals)) if vals else np.nan


def evaluate_gene_expression_2d(
    adata_true: ad.AnnData,
    adata_pred: ad.AnnData,
    bin_size: float,
    true_coords_key: str = "spatial",
    pred_coords_key: str = "spatial",
) -> Dict[str, float]:
    """
    Evaluate gene expression for 2D data with rasterization.
    
    Parameters
    ----------
    adata_true : AnnData
        Ground truth AnnData object
    adata_pred : AnnData
        Predicted AnnData object
    bin_size : float
        Size of each bin for rasterization
    true_coords_key : str, default="spatial"
        Key in obsm for true coordinates
    pred_coords_key : str, default="spatial"
        Key in obsm for predicted coordinates
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Get unified grid
    x_min, y_min, nx, ny = get_slice_union_bounds_and_bins(
        adata_true, [adata_pred], bin_size,
        true_coords_key=true_coords_key,
        recon_coords_key=pred_coords_key
    )
    
    # Bin both
    T_bin = bin_and_pad_adata_2d(
        adata_true, bin_size, x_min, y_min, nx, ny,
        coords_key=true_coords_key
    )
    P_bin = bin_and_pad_adata_2d(
        adata_pred, bin_size, x_min, y_min, nx, ny,
        coords_key=pred_coords_key
    )
    
    # Convert to dense arrays
    Xt = T_bin.X.toarray() if hasattr(T_bin.X, "toarray") else np.asarray(T_bin.X)
    Xp = P_bin.X.toarray() if hasattr(P_bin.X, "toarray") else np.asarray(P_bin.X)
    
    # Compute metrics
    iou = iou_from_binned_masks(T_bin, P_bin)
    
    metrics = {
        "bins": int(Xt.shape[0]),
        "genes": int(Xt.shape[1]),
        "bin_size": float(bin_size),
        "IoU_mask": float(iou) if iou == iou else np.nan,
        "MSE_masked": masked_MSE(Xt, Xp),
        "MAE_masked": masked_MAE(Xt, Xp),
        "Cosine_unmasked": cosine_similarity_mean(Xt, Xp, mask=False),
        "Cosine_masked": cosine_similarity_mean(Xt, Xp, mask=True),
        "Spearman_unmasked": spearman_r_mean(Xt, Xp, mask=False),
        "Spearman_masked": spearman_r_mean(Xt, Xp, mask=True),
        "Pearson_unmasked": pearson_r_mean(Xt, Xp, mask=False),
        "Pearson_masked": pearson_r_mean(Xt, Xp, mask=True),
    }
    
    return metrics


def evaluate_gene_expression_3d(
    adata_true: ad.AnnData,
    adata_pred: ad.AnnData,
    bin_size: float,
    true_coords_key: str = "spatial",
    pred_coords_key: str = "spatial",
) -> Dict[str, float]:
    """
    Evaluate gene expression for 3D data with rasterization.
    
    Parameters
    ----------
    adata_true : AnnData
        Ground truth AnnData object
    adata_pred : AnnData
        Predicted AnnData object
    bin_size : float
        Size of each bin for rasterization
    true_coords_key : str, default="spatial"
        Key in obsm for true coordinates
    pred_coords_key : str, default="spatial"
        Key in obsm for predicted coordinates
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Get unified grid
    x_min, y_min, z_min, nx, ny, nz = get_volume_union_bounds_and_bins(
        adata_true, [adata_pred], bin_size,
        true_coords_key=true_coords_key,
        recon_coords_key=pred_coords_key
    )
    
    # Bin both
    T_bin = bin_and_pad_adata_3d(
        adata_true, bin_size, x_min, y_min, z_min, nx, ny, nz,
        coords_key=true_coords_key
    )
    P_bin = bin_and_pad_adata_3d(
        adata_pred, bin_size, x_min, y_min, z_min, nx, ny, nz,
        coords_key=pred_coords_key
    )
    
    # Convert to dense arrays
    Xt = T_bin.X.toarray() if hasattr(T_bin.X, "toarray") else np.asarray(T_bin.X)
    Xp = P_bin.X.toarray() if hasattr(P_bin.X, "toarray") else np.asarray(P_bin.X)
    
    # Compute metrics
    iou = iou_from_binned_masks(T_bin, P_bin)
    
    metrics = {
        "bins": int(Xt.shape[0]),
        "genes": int(Xt.shape[1]),
        "bin_size": float(bin_size),
        "IoU_mask": float(iou) if iou == iou else np.nan,
        "MSE_masked": masked_MSE(Xt, Xp),
        "MAE_masked": masked_MAE(Xt, Xp),
        "Cosine_unmasked": cosine_similarity_mean(Xt, Xp, mask=False),
        "Cosine_masked": cosine_similarity_mean(Xt, Xp, mask=True),
        "Spearman_unmasked": spearman_r_mean(Xt, Xp, mask=False),
        "Spearman_masked": spearman_r_mean(Xt, Xp, mask=True),
        "Pearson_unmasked": pearson_r_mean(Xt, Xp, mask=False),
        "Pearson_masked": pearson_r_mean(Xt, Xp, mask=True),
    }
    
    return metrics


def evaluate_specific_genes_2d(
    adata_true: ad.AnnData,
    adata_pred: ad.AnnData,
    gene_names: Union[str, List[str]],
    bin_size: float,
    true_coords_key: str = "spatial",
    pred_coords_key: str = "spatial",
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate gene expression for specific genes in 2D data with rasterization.
    
    Parameters
    ----------
    adata_true : AnnData
        Ground truth AnnData object
    adata_pred : AnnData
        Predicted AnnData object
    gene_names : str or list of str
        Gene name(s) to evaluate. Can be a single gene name or a list of gene names.
    bin_size : float
        Size of each bin for rasterization
    true_coords_key : str, default="spatial"
        Key in obsm for true coordinates
    pred_coords_key : str, default="spatial"
        Key in obsm for predicted coordinates
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'overall': Overall metrics across all specified genes
        - 'per_gene': Dictionary with per-gene metrics (if multiple genes)
    """
    # Convert single gene to list
    if isinstance(gene_names, str):
        gene_names = [gene_names]
        return_per_gene = False
    else:
        return_per_gene = True
    
    # Get unified grid
    x_min, y_min, nx, ny = get_slice_union_bounds_and_bins(
        adata_true, [adata_pred], bin_size,
        true_coords_key=true_coords_key,
        recon_coords_key=pred_coords_key
    )
    
    # Bin both
    T_bin = bin_and_pad_adata_2d(
        adata_true, bin_size, x_min, y_min, nx, ny,
        coords_key=true_coords_key
    )
    P_bin = bin_and_pad_adata_2d(
        adata_pred, bin_size, x_min, y_min, nx, ny,
        coords_key=pred_coords_key
    )
    
    # Find gene indices
    gene_indices = []
    valid_genes = []
    for gene in gene_names:
        if gene in T_bin.var_names and gene in P_bin.var_names:
            idx = list(T_bin.var_names).index(gene)
            gene_indices.append(idx)
            valid_genes.append(gene)
        else:
            print(f"Warning: Gene '{gene}' not found in both datasets, skipping.")
    
    if not valid_genes:
        raise ValueError("No valid genes found in both datasets")
    
    # Convert to dense arrays and extract specific genes
    Xt_full = T_bin.X.toarray() if hasattr(T_bin.X, "toarray") else np.asarray(T_bin.X)
    Xp_full = P_bin.X.toarray() if hasattr(P_bin.X, "toarray") else np.asarray(P_bin.X)
    
    Xt = Xt_full[:, gene_indices]
    Xp = Xp_full[:, gene_indices]
    
    # Compute overall metrics (averaged across genes)
    overall_metrics = {
        "genes_evaluated": len(valid_genes),
        "gene_names": valid_genes,
        "bins": int(Xt.shape[0]),
        "bin_size": float(bin_size),
        "MSE_masked": masked_MSE(Xt, Xp),
        "MAE_masked": masked_MAE(Xt, Xp),
        "Cosine_unmasked": cosine_similarity_mean(Xt, Xp, mask=False),
        "Cosine_masked": cosine_similarity_mean(Xt, Xp, mask=True),
        "Spearman_unmasked": spearman_r_mean(Xt, Xp, mask=False),
        "Spearman_masked": spearman_r_mean(Xt, Xp, mask=True),
        "Pearson_unmasked": pearson_r_mean(Xt, Xp, mask=False),
        "Pearson_masked": pearson_r_mean(Xt, Xp, mask=True),
    }
    
    result = {"overall": overall_metrics}
    
    # Compute per-gene metrics if multiple genes
    if return_per_gene and len(valid_genes) > 1:
        per_gene_metrics = {}
        for i, gene in enumerate(valid_genes):
            Xt_gene = Xt[:, i:i+1]
            Xp_gene = Xp[:, i:i+1]
            
            per_gene_metrics[gene] = {
                "MSE_masked": masked_MSE(Xt_gene, Xp_gene),
                "MAE_masked": masked_MAE(Xt_gene, Xp_gene),
                "Cosine_unmasked": cosine_similarity_mean(Xt_gene, Xp_gene, mask=False),
                "Cosine_masked": cosine_similarity_mean(Xt_gene, Xp_gene, mask=True),
                "Spearman_unmasked": spearman_r_mean(Xt_gene, Xp_gene, mask=False),
                "Spearman_masked": spearman_r_mean(Xt_gene, Xp_gene, mask=True),
                "Pearson_unmasked": pearson_r_mean(Xt_gene, Xp_gene, mask=False),
                "Pearson_masked": pearson_r_mean(Xt_gene, Xp_gene, mask=True),
            }
        result["per_gene"] = per_gene_metrics
    
    return result


def evaluate_specific_genes_3d(
    adata_true: ad.AnnData,
    adata_pred: ad.AnnData,
    gene_names: Union[str, List[str]],
    bin_size: float,
    true_coords_key: str = "spatial",
    pred_coords_key: str = "spatial",
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate gene expression for specific genes in 3D data with rasterization.
    
    Parameters
    ----------
    adata_true : AnnData
        Ground truth AnnData object
    adata_pred : AnnData
        Predicted AnnData object
    gene_names : str or list of str
        Gene name(s) to evaluate. Can be a single gene name or a list of gene names.
    bin_size : float
        Size of each bin for rasterization
    true_coords_key : str, default="spatial"
        Key in obsm for true coordinates
    pred_coords_key : str, default="spatial"
        Key in obsm for predicted coordinates
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'overall': Overall metrics across all specified genes
        - 'per_gene': Dictionary with per-gene metrics (if multiple genes)
    """
    # Convert single gene to list
    if isinstance(gene_names, str):
        gene_names = [gene_names]
        return_per_gene = False
    else:
        return_per_gene = True
    
    # Get unified grid
    x_min, y_min, z_min, nx, ny, nz = get_volume_union_bounds_and_bins(
        adata_true, [adata_pred], bin_size,
        true_coords_key=true_coords_key,
        recon_coords_key=pred_coords_key
    )
    
    # Bin both
    T_bin = bin_and_pad_adata_3d(
        adata_true, bin_size, x_min, y_min, z_min, nx, ny, nz,
        coords_key=true_coords_key
    )
    P_bin = bin_and_pad_adata_3d(
        adata_pred, bin_size, x_min, y_min, z_min, nx, ny, nz,
        coords_key=pred_coords_key
    )
    
    # Find gene indices
    gene_indices = []
    valid_genes = []
    for gene in gene_names:
        if gene in T_bin.var_names and gene in P_bin.var_names:
            idx = list(T_bin.var_names).index(gene)
            gene_indices.append(idx)
            valid_genes.append(gene)
        else:
            print(f"Warning: Gene '{gene}' not found in both datasets, skipping.")
    
    if not valid_genes:
        raise ValueError("No valid genes found in both datasets")
    
    # Convert to dense arrays and extract specific genes
    Xt_full = T_bin.X.toarray() if hasattr(T_bin.X, "toarray") else np.asarray(T_bin.X)
    Xp_full = P_bin.X.toarray() if hasattr(P_bin.X, "toarray") else np.asarray(P_bin.X)
    
    Xt = Xt_full[:, gene_indices]
    Xp = Xp_full[:, gene_indices]
    
    # Compute overall metrics (averaged across genes)
    overall_metrics = {
        "genes_evaluated": len(valid_genes),
        "gene_names": valid_genes,
        "bins": int(Xt.shape[0]),
        "bin_size": float(bin_size),
        "MSE_masked": masked_MSE(Xt, Xp),
        "MAE_masked": masked_MAE(Xt, Xp),
        "Cosine_unmasked": cosine_similarity_mean(Xt, Xp, mask=False),
        "Cosine_masked": cosine_similarity_mean(Xt, Xp, mask=True),
        "Spearman_unmasked": spearman_r_mean(Xt, Xp, mask=False),
        "Spearman_masked": spearman_r_mean(Xt, Xp, mask=True),
        "Pearson_unmasked": pearson_r_mean(Xt, Xp, mask=False),
        "Pearson_masked": pearson_r_mean(Xt, Xp, mask=True),
    }
    
    result = {"overall": overall_metrics}
    
    # Compute per-gene metrics if multiple genes
    if return_per_gene and len(valid_genes) > 1:
        per_gene_metrics = {}
        for i, gene in enumerate(valid_genes):
            Xt_gene = Xt[:, i:i+1]
            Xp_gene = Xp[:, i:i+1]
            
            per_gene_metrics[gene] = {
                "MSE_masked": masked_MSE(Xt_gene, Xp_gene),
                "MAE_masked": masked_MAE(Xt_gene, Xp_gene),
                "Cosine_unmasked": cosine_similarity_mean(Xt_gene, Xp_gene, mask=False),
                "Cosine_masked": cosine_similarity_mean(Xt_gene, Xp_gene, mask=True),
                "Spearman_unmasked": spearman_r_mean(Xt_gene, Xp_gene, mask=False),
                "Spearman_masked": spearman_r_mean(Xt_gene, Xp_gene, mask=True),
                "Pearson_unmasked": pearson_r_mean(Xt_gene, Xp_gene, mask=False),
                "Pearson_masked": pearson_r_mean(Xt_gene, Xp_gene, mask=True),
            }
        result["per_gene"] = per_gene_metrics
    
    return result
