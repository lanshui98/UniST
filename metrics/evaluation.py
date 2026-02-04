"""
Main evaluation functions for structural accuracy assessment.

This module provides high-level functions for evaluating reconstrcution results
on 2D slices or 3D volumes, with support for both boundary-based and
voxel-based metrics.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tifffile

from .boundary import (
    boundary_iou,
    hausdorff_distance,
    average_surface_distance,
    chamfer_distance,
)
from .voxel import (
    dice_coefficient,
    jaccard_index,
    hausdorff_distance_voxel,
    average_surface_distance_voxel,
    chamfer_distance_voxel,
    false_positive_rate,
    false_negative_rate,
)


def get_file_map(directory: Union[str, Path], pattern: str) -> Optional[Dict[int, str]]:
    """
    Parse filenames using a regex pattern and return a {index: filename} map.
    
    Parameters
    ----------
    directory : str or Path
        Directory to scan for files
    pattern : str
        Regex pattern with a capture group for the index
    
    Returns
    -------
    dict or None
        Mapping from index to filename, or None if directory not found
    """
    file_map = {}
    directory = Path(directory)
    
    try:
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return None
        
        for f in directory.iterdir():
            if f.suffix.lower() in ('.tif', '.tiff'):
                match = re.search(pattern, f.name)
                if match:
                    # Index is captured by the first group in the regex
                    index = int(match.group(1))
                    file_map[index] = f.name
    except Exception as e:
        print(f"An unexpected error occurred while scanning {directory}: {e}")
        return None
    
    return file_map


def evaluate_slice_pair(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    compute_boundary: bool = True,
    compute_voxel: bool = True,
    boundary_thickness: int = 1,
    boundary_tolerance: int = 0
) -> Dict[str, float]:
    """
    Evaluate a single 2D slice pair with multiple metrics.
    
    Parameters
    ----------
    mask_true : np.ndarray
        Ground truth 2D binary mask
    mask_pred : np.ndarray
        Predicted 2D binary mask
    compute_boundary : bool, default=True
        Whether to compute boundary-based metrics
    compute_voxel : bool, default=True
        Whether to compute voxel-based metrics
    boundary_thickness : int, default=1
        Thickness of boundary band for boundary metrics
    boundary_tolerance : int, default=0
        Tolerance for boundary matching
    
    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    # Ensure binary and same shape
    mask_true = (mask_true > 0).astype(np.uint8)
    mask_pred = (mask_pred > 0).astype(np.uint8)
    
    if mask_true.shape != mask_pred.shape:
        raise ValueError(
            f"Masks have different shapes: {mask_true.shape} vs {mask_pred.shape}"
        )
    
    results = {}
    
    # Voxel-based metrics
    if compute_voxel:
        results['Dice'] = dice_coefficient(mask_true, mask_pred)
        results['IoU'] = jaccard_index(mask_true, mask_pred)
        results['HD_voxel'] = hausdorff_distance_voxel(mask_true, mask_pred)
        results['HD95_voxel'] = hausdorff_distance_voxel(mask_true, mask_pred, percentile=95)
        results['ASD_voxel'] = average_surface_distance_voxel(mask_true, mask_pred)
        results['ChamferDistance_voxel'] = chamfer_distance_voxel(mask_true, mask_pred)
        results['FPR'] = false_positive_rate(mask_true, mask_pred)
        results['FNR'] = false_negative_rate(mask_true, mask_pred)
    
    # Boundary-based metrics
    if compute_boundary:
        results['BoundaryIoU'] = boundary_iou(
            mask_true, mask_pred,
            thickness=boundary_thickness,
            tolerance=boundary_tolerance
        )
        results['HD'] = hausdorff_distance(mask_true, mask_pred)
        results['HD95'] = hausdorff_distance(mask_true, mask_pred, percentile=95)
        results['ASD'] = average_surface_distance(mask_true, mask_pred)
        results['ChamferDistance'] = chamfer_distance(mask_true, mask_pred)
    
    return results


def evaluate_volume_pair(
    volume_true: np.ndarray,
    volume_pred: np.ndarray,
    compute_boundary: bool = True,
    compute_voxel: bool = True,
    boundary_thickness: int = 1,
    boundary_tolerance: int = 0
) -> Dict[str, float]:
    """
    Evaluate a 3D volume pair with multiple metrics.
    
    Parameters
    ----------
    volume_true : np.ndarray
        Ground truth 3D binary volume
    volume_pred : np.ndarray
        Predicted 3D binary volume
    compute_boundary : bool, default=True
        Whether to compute boundary-based metrics
    compute_voxel : bool, default=True
        Whether to compute voxel-based metrics
    boundary_thickness : int, default=1
        Thickness of boundary band for boundary metrics
    boundary_tolerance : int, default=0
        Tolerance for boundary matching
    
    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    # Ensure binary and same shape
    volume_true = (volume_true > 0).astype(np.uint8)
    volume_pred = (volume_pred > 0).astype(np.uint8)
    
    if volume_true.shape != volume_pred.shape:
        raise ValueError(
            f"Volumes have different shapes: {volume_true.shape} vs {volume_pred.shape}"
        )
    
    results = {}
    
    # Voxel-based metrics
    if compute_voxel:
        results['Dice'] = dice_coefficient(volume_true, volume_pred)
        results['IoU'] = jaccard_index(volume_true, volume_pred)
        results['HD_voxel'] = hausdorff_distance_voxel(volume_true, volume_pred)
        results['HD95_voxel'] = hausdorff_distance_voxel(volume_true, volume_pred, percentile=95)
        results['ASD_voxel'] = average_surface_distance_voxel(volume_true, volume_pred)
        results['ChamferDistance_voxel'] = chamfer_distance_voxel(volume_true, volume_pred)
        results['FPR'] = false_positive_rate(volume_true, volume_pred)
        results['FNR'] = false_negative_rate(volume_true, volume_pred)
    
    # Boundary-based metrics (3D surface)
    if compute_boundary:
        results['BoundaryIoU'] = boundary_iou(
            volume_true, volume_pred,
            thickness=boundary_thickness,
            tolerance=boundary_tolerance
        )
        results['HD'] = hausdorff_distance(volume_true, volume_pred)
        results['HD95'] = hausdorff_distance(volume_true, volume_pred, percentile=95)
        results['ASD'] = average_surface_distance(volume_true, volume_pred)
        results['ChamferDistance'] = chamfer_distance(volume_true, volume_pred)
    
    return results


def evaluate_slices(
    true_dir: Union[str, Path],
    pred_dir: Union[str, Path],
    true_pattern: str,
    pred_pattern: str,
    output_csv: Optional[Union[str, Path]] = None,
    compute_boundary: bool = True,
    compute_voxel: bool = True,
    boundary_thickness: int = 1,
    boundary_tolerance: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate 2D slices from two directories with matching file indices.
    
    Parameters
    ----------
    true_dir : str or Path
        Directory containing ground truth slices
    pred_dir : str or Path
        Directory containing predicted slices
    true_pattern : str
        Regex pattern for ground truth filenames (with index capture group)
    pred_pattern : str
        Regex pattern for predicted filenames (with index capture group)
    output_csv : str or Path, optional
        Path to save results CSV file
    compute_boundary : bool, default=True
        Whether to compute boundary-based metrics
    compute_voxel : bool, default=True
        Whether to compute voxel-based metrics
    boundary_thickness : int, default=1
        Thickness of boundary band
    boundary_tolerance : int, default=0
        Tolerance for boundary matching
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    pd.DataFrame
        DataFrame with per-slice metrics
    """
    true_dir = Path(true_dir)
    pred_dir = Path(pred_dir)
    
    # Get file maps
    true_map = get_file_map(true_dir, true_pattern)
    pred_map = get_file_map(pred_dir, pred_pattern)
    
    if true_map is None or pred_map is None:
        raise FileNotFoundError("Could not read one or both directories")
    
    # Find common indices
    common_indices = sorted(list(set(true_map.keys()) & set(pred_map.keys())))
    
    if not common_indices:
        raise ValueError(
            f"No matching file indices found between directories.\n"
            f"True pattern: '{true_pattern}' | Pred pattern: '{pred_pattern}'"
        )
    
    if verbose:
        print(f"Found {len(common_indices)} matched pairs for evaluation.")
        print("-" * 50)
    
    all_results = []
    
    for file_num in common_indices:
        true_filename = true_map[file_num]
        pred_filename = pred_map[file_num]
        
        true_path = true_dir / true_filename
        pred_path = pred_dir / pred_filename
        
        if verbose:
            print(f"Processing pair {file_num}: {true_filename} vs {pred_filename}")
        
        try:
            # Load images
            mask_true = tifffile.imread(true_path)
            mask_pred = tifffile.imread(pred_path)
            
            # Evaluate
            metrics = evaluate_slice_pair(
                mask_true, mask_pred,
                compute_boundary=compute_boundary,
                compute_voxel=compute_voxel,
                boundary_thickness=boundary_thickness,
                boundary_tolerance=boundary_tolerance
            )
            
            # Add metadata
            row = {
                'ID': file_num,
                'True_File': true_filename,
                'Pred_File': pred_filename,
                **metrics
            }
            
            all_results.append(row)
            
            if verbose:
                print(f"  ✓ Computed {len(metrics)} metrics")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    if results_df.empty:
        if verbose:
            print("No successful results were generated.")
        return results_df
    
    # Save results
    if output_csv is not None:
        output_csv = Path(output_csv)
        results_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n✓ Results saved to: {output_csv}")
    
    # Display summary
    if verbose and not results_df.empty:
        metric_cols = [c for c in results_df.columns if c not in ['ID', 'True_File', 'Pred_File']]
        if metric_cols:
            summary = results_df[metric_cols].agg(['mean', 'std'])
            print("\nSummary Statistics:")
            print(summary.to_string(float_format='%.4f'))
    
    return results_df


def evaluate_volume(
    volume_true: np.ndarray,
    volume_pred: np.ndarray,
    compute_boundary: bool = True,
    compute_voxel: bool = True,
    boundary_thickness: int = 1,
    boundary_tolerance: int = 0
) -> Dict[str, float]:
    """
    Evaluate a 3D volume pair.
    
    This is a convenience wrapper around evaluate_volume_pair.
    
    Parameters
    ----------
    volume_true : np.ndarray
        Ground truth 3D binary volume
    volume_pred : np.ndarray
        Predicted 3D binary volume
    compute_boundary : bool, default=True
        Whether to compute boundary-based metrics
    compute_voxel : bool, default=True
        Whether to compute voxel-based metrics
    boundary_thickness : int, default=1
        Thickness of boundary band
    boundary_tolerance : int, default=0
        Tolerance for boundary matching
    
    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    return evaluate_volume_pair(
        volume_true, volume_pred,
        compute_boundary=compute_boundary,
        compute_voxel=compute_voxel,
        boundary_thickness=boundary_thickness,
        boundary_tolerance=boundary_tolerance
    )
