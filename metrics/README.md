# Metrics for Structural Accuracy Evaluation

This module provides comprehensive metrics for evaluating structural accuracy of segmentation results, supporting both 2D slice-level and 3D volume-level evaluation with boundary-based and voxel-based metrics.

## Features

- **2D and 3D Support**: Evaluate both individual slices and entire volumes
- **Boundary-based Metrics**: Hausdorff Distance, Average Surface Distance, Chamfer Distance, Boundary IoU
- **Voxel-based Metrics**: Dice Coefficient, Jaccard Index (IoU), Volume Overlap Error, Volume Similarity, FPR, FNR
- **Batch Evaluation**: Process multiple slice pairs with automatic file matching

## Metrics Overview

### Boundary-based Metrics

- **Boundary IoU**: Intersection over Union on boundary bands
- **Hausdorff Distance (HD)**: Maximum distance between boundaries
- **HD95**: 95th percentile Hausdorff Distance (more robust to outliers)
- **Average Surface Distance (ASD)**: Symmetric mean surface distance
- **Chamfer Distance**: Average distance from each boundary point to the nearest point on the other boundary

### Voxel-based Metrics

- **Dice Coefficient**: F1 score for overlap (2|A∩B| / (|A| + |B|))
- **Jaccard Index (IoU)**: Intersection over Union (|A∩B| / |A∪B|)
- **Hausdorff Distance (Voxel)**: HD computed on all foreground voxels (not just boundaries)
- **HD95 (Voxel)**: 95th percentile HD on all foreground voxels
- **Average Surface Distance (Voxel)**: ASD computed on all foreground voxels
- **Chamfer Distance (Voxel)**: Chamfer distance computed on all foreground voxels
- **False Positive Rate (FPR)**: Rate of false positives
- **False Negative Rate (FNR)**: Rate of false negatives

**Note**: The voxel-based distance metrics consider all foreground voxels, while boundary-based metrics only consider boundary/surface voxels. Voxel-based metrics are more computationally expensive but provide a different perspective on shape similarity.

## Usage

### Basic Example: Evaluate Single Slice Pair

```python
import numpy as np
from metrics import evaluate_slice_pair

# Load your masks
mask_true = np.load('ground_truth.npy')  # 2D binary mask
mask_pred = np.load('prediction.npy')   # 2D binary mask

# Evaluate with all metrics
results = evaluate_slice_pair(
    mask_true, mask_pred,
    compute_boundary=True,
    compute_voxel=True
)

print(f"Dice: {results['Dice']:.4f}")
print(f"IoU: {results['IoU']:.4f}")
print(f"HD95: {results['HD95']:.4f}")
```

### Batch Evaluation: Evaluate Multiple Slices

```python
from metrics import evaluate_slices
import pandas as pd

# Evaluate slices from two directories
results_df = evaluate_slices(
    true_dir="/path/to/ground_truth",
    pred_dir="/path/to/predictions",
    true_pattern=r'Epithelial_(\d{3})\.tif',
    pred_pattern=r'slice_(\d{3})\.tif$',
    output_csv="evaluation_results.csv",
    compute_boundary=True,
    compute_voxel=True,
    boundary_thickness=1,
    boundary_tolerance=1,
    verbose=True
)

# View results
print(results_df)
```

### 3D Volume Evaluation

```python
from metrics import evaluate_volume
import tifffile

# Load 3D volumes
volume_true = tifffile.imread('ground_truth_volume.tif')
volume_pred = tifffile.imread('prediction_volume.tif')

# Evaluate
results = evaluate_volume(
    volume_true, volume_pred,
    compute_boundary=True,
    compute_voxel=True
)

print(f"3D Dice: {results['Dice']:.4f}")
print(f"3D HD95: {results['HD95']:.4f}")
```

### Using Individual Metrics

```python
from metrics import (
    dice_coefficient,
    hausdorff_distance,
    boundary_iou,
    average_surface_distance
)

# Voxel-based
dice = dice_coefficient(mask_true, mask_pred)
iou = jaccard_index(mask_true, mask_pred)

# Boundary-based
hd = hausdorff_distance(mask_true, mask_pred)
hd95 = hausdorff_distance(mask_true, mask_pred, percentile=95)
asd = average_surface_distance(mask_true, mask_pred)
boundary_iou_score = boundary_iou(mask_true, mask_pred, thickness=1, tolerance=1)
```

## File Matching

The `evaluate_slices` function uses regex patterns to match files by index:

- **Pattern Format**: Must include a capture group `(\d+)` for the index
- **Example Patterns**:
  - `r'Epithelial_(\d{3})\.tif'` - Matches `Epithelial_001.tif`, `Epithelial_002.tif`, etc.
  - `r'slice_(\d{3})\.tif$'` - Matches `slice_001.tif`, `slice_002.tif`, etc.
  - `r'image_(\d+)\.tif'` - Matches `image_1.tif`, `image_10.tif`, etc.

Files are matched by their extracted index values.

## Output Format

The evaluation functions return pandas DataFrames with the following columns:

- `ID`: File index
- `True_File`: Ground truth filename
- `Pred_File`: Predicted filename
- Metric columns (Dice, IoU, HD, HD95, ASD, etc.)

## Gene Expression Evaluation

The module also provides gene expression evaluation metrics with rasterization support for both 2D and 3D data.

### Gene Expression Metrics

- **Masked MSE/MAE**: Mean Squared/Absolute Error on regions with expression
- **Cosine Similarity**: Mean cosine similarity (masked and unmasked)
- **Spearman Correlation**: Mean Spearman correlation (masked and unmasked)
- **Pearson Correlation**: Mean Pearson correlation (masked and unmasked)
- **IoU Mask**: Intersection over Union of expression masks

### Usage Example

```python
import anndata as ad
from metrics import evaluate_gene_expression_2d, evaluate_gene_expression_3d

# Load AnnData objects
adata_true = ad.read_h5ad("ground_truth.h5ad")
adata_pred = ad.read_h5ad("prediction.h5ad")

# Evaluate 2D gene expression
metrics_2d = evaluate_gene_expression_2d(
    adata_true, adata_pred,
    bin_size=20.0,
    true_coords_key="spatial",
    pred_coords_key="spatial"
)

print(f"MSE (masked): {metrics_2d['MSE_masked']:.4f}")
print(f"Pearson (masked): {metrics_2d['Pearson_masked']:.4f}")

# Evaluate 3D gene expression
metrics_3d = evaluate_gene_expression_3d(
    adata_true, adata_pred,
    bin_size=20.0,
    true_coords_key="spatial",
    pred_coords_key="spatial"
)
```

### Rasterization

The gene expression evaluation automatically:
1. Creates a unified grid from true and predicted coordinates
2. Bins spots into grid cells (computing mean expression per bin)
3. Evaluates metrics on the binned data

This ensures fair comparison even when spot locations differ slightly.

## Notes

- All masks are automatically converted to binary (values > 0 become 1)
- Boundary metrics require non-empty boundaries (returns NaN if empty)
- Distance metrics are in pixels (2D) or voxels (3D)
- The module handles shape mismatches gracefully with informative error messages
- Gene expression evaluation requires AnnData objects with coordinates in `obsm`