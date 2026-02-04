"""
Interpolation module for UniST.

This module provides slice interpolation functionality using InterpolAI,
including utilities for image preprocessing and post-processing.
"""

from . import utils
from .utils import (
    convert_to_3channel,
    convert_directory_to_3channel,
    merge_to_volume,
    collect_interpolated_maps,
)

__all__ = [
    'utils',
    'convert_to_3channel',
    'convert_directory_to_3channel',
    'merge_to_volume',
    'collect_interpolated_maps',
]
