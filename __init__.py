"""
UniST - A Unified Computational Framework for 3D Spatial Transcriptomics Reconstruction.

This package provides three main functionalities:
- Upsampling: Point cloud upsampling using RepKPU
- Interpolation: Slice interpolation using InterpolAI
- Imputation: Gene expression imputation using SUICA_pro
"""

__version__ = "0.1.0"
__author__ = "Lan Shui"
__email__ = "lshui@mdanderson.org"

# Users should import directly from submodules:
#   from metrics import dice_coefficient, boundary_iou
#   from downstream.morph import dilation, erosion
#   from downstream.vis import slice_points_plane, slice_stack
#   from interpolation.ReHo import interpolate_missing_slices_spline

__all__ = ["__version__", "__author__", "__email__"]
