"""
Voxel-based metrics for structural accuracy evaluation.

This module provides voxel-based metrics for both 2D and 3D evaluation,
including Dice Coefficient, Jaccard Index (IoU), and distance-based metrics
computed on all foreground voxels (not just boundaries).
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional


def dice_coefficient(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice Coefficient (F1 Score) between two binary masks.
    
    Supports both 2D and 3D masks.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask
    mask_pred : np.ndarray
        Predicted binary mask
    smooth : float, default=1e-6
        Smoothing factor to avoid division by zero
    
    Returns
    -------
    float
        Dice Coefficient (0-1), where 1 is perfect overlap
    """
    mask_true = (mask_true > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    
    intersection = (mask_true * mask_pred).sum()
    union = mask_true.sum() + mask_pred.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float((2.0 * intersection + smooth) / (union + smooth))


def jaccard_index(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Compute Jaccard Index (Intersection over Union, IoU) between two binary masks.
    
    Supports both 2D and 3D masks.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask
    mask_pred : np.ndarray
        Predicted binary mask
    smooth : float, default=1e-6
        Smoothing factor to avoid division by zero
    
    Returns
    -------
    float
        Jaccard Index / IoU (0-1), where 1 is perfect overlap
    """
    mask_true = (mask_true > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    
    intersection = (mask_true * mask_pred).sum()
    union = (mask_true + mask_pred - mask_true * mask_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float((intersection + smooth) / (union + smooth))


def _points_from_foreground(mask: np.ndarray) -> np.ndarray:
    """
    Extract all foreground voxel coordinates from a mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (2D or 3D)
    
    Returns
    -------
    np.ndarray
        NxD array of foreground voxel coordinates (D=2 for 2D, D=3 for 3D)
    """
    mask = (mask > 0).astype(bool)
    
    if mask.ndim == 2:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([ys, xs], axis=1).astype(np.float32)
    elif mask.ndim == 3:
        zs, ys, xs = np.nonzero(mask)
        if zs.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack([zs, ys, xs], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.ndim}")


def hausdorff_distance_voxel(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    percentile: Optional[float] = None
) -> float:
    """
    Compute Hausdorff Distance on all foreground voxels (not just boundaries).
    
    This is a voxel-based version that considers all foreground voxels,
    not just the boundary/surface voxels.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask (2D or 3D)
    mask_pred : np.ndarray
        Predicted binary mask (2D or 3D)
    percentile : float, optional
        If provided, compute HD at this percentile (e.g., 95 for HD95).
        If None, compute classical HD.
    
    Returns
    -------
    float
        Hausdorff Distance in pixels/voxels, or np.nan if no foreground voxels
    """
    A = _points_from_foreground(mask_true)
    B = _points_from_foreground(mask_pred)
    
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    
    D = cdist(A, B)
    dA = D.min(axis=1)
    dB = D.min(axis=0)
    
    if percentile is None:
        return float(max(dA.max(), dB.max()))
    else:
        return float(max(
            np.percentile(dA, percentile),
            np.percentile(dB, percentile)
        ))


def average_surface_distance_voxel(
    mask_true: np.ndarray,
    mask_pred: np.ndarray
) -> float:
    """
    Compute Average Surface Distance on all foreground voxels.
    
    This is a voxel-based version that considers all foreground voxels,
    not just the boundary/surface voxels.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask (2D or 3D)
    mask_pred : np.ndarray
        Predicted binary mask (2D or 3D)
    
    Returns
    -------
    float
        Average Surface Distance in pixels/voxels, or np.nan if no foreground voxels
    """
    A = _points_from_foreground(mask_true)
    B = _points_from_foreground(mask_pred)
    
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    
    D = cdist(A, B)
    dA = D.min(axis=1)
    dB = D.min(axis=0)
    
    return float(0.5 * (dA.mean() + dB.mean()))


def chamfer_distance_voxel(
    mask_true: np.ndarray,
    mask_pred: np.ndarray
) -> float:
    """
    Compute Chamfer Distance on all foreground voxels.
    
    This is a voxel-based version that considers all foreground voxels,
    not just the boundary/surface voxels.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask (2D or 3D)
    mask_pred : np.ndarray
        Predicted binary mask (2D or 3D)
    
    Returns
    -------
    float
        Chamfer Distance in pixels/voxels, or np.nan if no foreground voxels
    """
    A = _points_from_foreground(mask_true)
    B = _points_from_foreground(mask_pred)
    
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    
    D = cdist(A, B)
    dA = D.min(axis=1).mean()
    dB = D.min(axis=0).mean()
    
    return float(dA + dB)


def false_positive_rate(
    mask_true: np.ndarray,
    mask_pred: np.ndarray
) -> float:
    """
    Compute False Positive Rate (FPR).
    
    FPR = FP / (FP + TN) = FP / (Total background in ground truth)
    
    Supports both 2D and 3D masks.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask
    mask_pred : np.ndarray
        Predicted binary mask
    
    Returns
    -------
    float
        False Positive Rate (0-1)
    """
    mask_true = (mask_true > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    
    # False positives: predicted positive but actually negative
    fp = ((mask_pred > 0) & (mask_true == 0)).sum()
    # True negatives: predicted negative and actually negative
    tn = ((mask_pred == 0) & (mask_true == 0)).sum()
    
    if fp + tn == 0:
        return 0.0
    
    return float(fp / (fp + tn))


def false_negative_rate(
    mask_true: np.ndarray,
    mask_pred: np.ndarray
) -> float:
    """
    Compute False Negative Rate (FNR).
    
    FNR = FN / (FN + TP) = FN / (Total foreground in ground truth)
    
    Supports both 2D and 3D masks.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask
    mask_pred : np.ndarray
        Predicted binary mask
    
    Returns
    -------
    float
        False Negative Rate (0-1)
    """
    mask_true = (mask_true > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    
    # False negatives: predicted negative but actually positive
    fn = ((mask_pred == 0) & (mask_true > 0)).sum()
    # True positives: predicted positive and actually positive
    tp = ((mask_pred > 0) & (mask_true > 0)).sum()
    
    if fn + tp == 0:
        return 0.0
    
    return float(fn / (fn + tp))
