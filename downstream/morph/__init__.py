"""
Morphological operations for 3D image data using VTK.

This module provides morphological operations (dilation, erosion, closing, opening)
for vtkImageData objects, particularly useful for connecting structures
along the Z-axis in 3D spatial transcriptomics data.

It also provides geometric measurements (volume and surface area) for voxel data.
"""

from .operations import dilation, erosion, closing, opening, periphery_mask, boundary_mask
from .geometry import (
    calculate_volume,
    calculate_surface_area,
    calculate_volume_and_surface_area,
)

__all__ = [
    'dilation',
    'erosion',
    'closing',
    'opening',
    'periphery_mask',
    'boundary_mask',
    'calculate_volume',
    'calculate_surface_area',
    'calculate_volume_and_surface_area',
]
