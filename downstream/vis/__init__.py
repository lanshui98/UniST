"""
Visualization utilities for 3D spatial transcriptomics data.

This module provides visualization functions for point clouds, slices, and
3D reconstructions using PyVista.
"""

from .slice_generation import (
    slice_points_plane,
    generate_slice_origins,
    slice_stack,
    plot_slice_stack,
)

__all__ = [
    'slice_points_plane',
    'generate_slice_origins',
    'slice_stack',
    'plot_slice_stack',
]
