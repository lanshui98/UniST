"""
Metrics for structural accuracy evaluation.

This module provides comprehensive metrics for evaluating structural accuracy
of segmentation results, supporting both 2D and 3D evaluation with
boundary-based and voxel-based metrics, as well as gene expression evaluation.
"""

from .boundary import (
    boundary_map,
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
from .evaluation import evaluate_slices, evaluate_volume
from .gene_expression import (
    evaluate_gene_expression_2d,
    evaluate_gene_expression_3d,
    evaluate_specific_genes_2d,
    evaluate_specific_genes_3d,
    masked_MSE,
    masked_MAE,
    cosine_similarity_mean,
    spearman_r_mean,
    pearson_r_mean,
)

__all__ = [
    # Boundary-based metrics
    'boundary_map',
    'boundary_iou',
    'hausdorff_distance',
    'average_surface_distance',
    'chamfer_distance',
    # Voxel-based metrics
    'dice_coefficient',
    'jaccard_index',
    'hausdorff_distance_voxel',
    'average_surface_distance_voxel',
    'chamfer_distance_voxel',
    'false_positive_rate',
    'false_negative_rate',
    # Evaluation functions
    'evaluate_slices',
    'evaluate_volume',
    # Gene expression evaluation
    'evaluate_gene_expression_2d',
    'evaluate_gene_expression_3d',
    'evaluate_specific_genes_2d',
    'evaluate_specific_genes_3d',
    'masked_MSE',
    'masked_MAE',
    'cosine_similarity_mean',
    'spearman_r_mean',
    'pearson_r_mean',
]
