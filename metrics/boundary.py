"""
Boundary-based metrics for structural accuracy evaluation.

This module provides boundary-based metrics for both 2D and 3D evaluation,
including Hausdorff Distance, Average Surface Distance, Chamfer Distance,
and Boundary IoU.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, binary_dilation
from typing import Optional, Union


def boundary_map(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Extract binary boundary map from a mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary or grayscale mask (2D or 3D)
    thickness : int, default=1
        Thickness of the boundary band in pixels/voxels
    
    Returns
    -------
    np.ndarray
        Binary boundary map (1 on boundary, 0 elsewhere)
    """
    m = (mask > 0).astype(np.uint8)
    if thickness <= 0:
        thickness = 1
    
    core = binary_erosion(m, iterations=1).astype(np.uint8)
    bnd = (m ^ core).astype(np.uint8)
    
    if thickness > 1:
        bnd = binary_dilation(bnd, iterations=thickness - 1).astype(np.uint8)
    
    return bnd


def boundary_iou(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    thickness: int = 1,
    tolerance: int = 0
) -> float:
    """
    Compute IoU on boundary bands.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask
    mask_pred : np.ndarray
        Predicted binary mask
    thickness : int, default=1
        Thickness of the boundary band
    tolerance : int, default=0
        Tolerance for boundary matching (dilation iterations)
    
    Returns
    -------
    float
        Boundary IoU score (0-1)
    """
    bt = boundary_map(mask_true, thickness=thickness)
    bp = boundary_map(mask_pred, thickness=thickness)
    
    if tolerance > 0:
        bt = binary_dilation(bt, iterations=tolerance).astype(np.uint8)
        bp = binary_dilation(bp, iterations=tolerance).astype(np.uint8)
    
    inter = np.logical_and(bt, bp).sum()
    union = np.logical_or(bt, bp).sum()
    
    return 1.0 if union == 0 else float(inter / union)


def _points_from_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary point coordinates from a 2D mask.
    
    Parameters
    ----------
    mask : np.ndarray
        2D binary mask
    
    Returns
    -------
    np.ndarray
        Nx2 array of (row, col) boundary coordinates
    """
    b = boundary_map(mask, thickness=1)
    ys, xs = np.nonzero(b)
    
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    # Stack [row, col] which corresponds to [y, x] in image coordinates
    return np.stack([ys, xs], axis=1).astype(np.float32)


def _points_from_surface(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface point coordinates from a 3D mask.
    
    Parameters
    ----------
    mask : np.ndarray
        3D binary mask
    
    Returns
    -------
    np.ndarray
        Nx3 array of (z, row, col) surface coordinates
    """
    b = boundary_map(mask, thickness=1)
    zs, ys, xs = np.nonzero(b)
    
    if zs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    return np.stack([zs, ys, xs], axis=1).astype(np.float32)


def hausdorff_distance(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    percentile: Optional[float] = None
) -> float:
    """
    Compute Hausdorff Distance (HD) or HD at given percentile (e.g., HD95).
    
    Supports both 2D and 3D masks.
    
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
        Hausdorff Distance in pixels/voxels, or np.nan if no boundary points
    """
    if mask_true.ndim == 2:
        A = _points_from_boundary(mask_true)
        B = _points_from_boundary(mask_pred)
    elif mask_true.ndim == 3:
        A = _points_from_surface(mask_true)
        B = _points_from_surface(mask_pred)
    else:
        raise ValueError(f"Unsupported mask dimension: {mask_true.ndim}")
    
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


def average_surface_distance(
    mask_true: np.ndarray,
    mask_pred: np.ndarray
) -> float:
    """
    Compute Average Surface Distance (ASD): symmetric mean surface distance.
    
    Supports both 2D and 3D masks.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask (2D or 3D)
    mask_pred : np.ndarray
        Predicted binary mask (2D or 3D)
    
    Returns
    -------
    float
        Average Surface Distance in pixels/voxels, or np.nan if no boundary points
    """
    if mask_true.ndim == 2:
        A = _points_from_boundary(mask_true)
        B = _points_from_boundary(mask_pred)
    elif mask_true.ndim == 3:
        A = _points_from_surface(mask_true)
        B = _points_from_surface(mask_pred)
    else:
        raise ValueError(f"Unsupported mask dimension: {mask_true.ndim}")
    
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    
    D = cdist(A, B)
    dA = D.min(axis=1)
    dB = D.min(axis=0)
    
    return float(0.5 * (dA.mean() + dB.mean()))


def chamfer_distance(
    mask_true: np.ndarray,
    mask_pred: np.ndarray
) -> float:
    """
    Compute Chamfer Distance between two binary masks.
    
    Supports both 2D and 3D masks.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth binary mask (2D or 3D)
    mask_pred : np.ndarray
        Predicted binary mask (2D or 3D)
    
    Returns
    -------
    float
        Chamfer Distance in pixels/voxels, or np.nan if no boundary points
    """
    if mask_true.ndim == 2:
        A = _points_from_boundary(mask_true)
        B = _points_from_boundary(mask_pred)
    elif mask_true.ndim == 3:
        A = _points_from_surface(mask_true)
        B = _points_from_surface(mask_pred)
    else:
        raise ValueError(f"Unsupported mask dimension: {mask_true.ndim}")
    
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    
    D = cdist(A, B)
    dA = D.min(axis=1).mean()
    dB = D.min(axis=0).mean()
    
    return float(dA + dB)
