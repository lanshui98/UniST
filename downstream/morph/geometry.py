"""
Geometric measurements for voxel data.

This module provides functions to calculate volume and surface area
for binary voxel data (3D arrays or vtkImageData).
"""

import numpy as np
import vtk
from typing import Union, Tuple, Optional


def _as_numpy_array(data: Union[np.ndarray, vtk.vtkImageData], 
                    foreground_value: int = 1) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Convert input to numpy array and get spacing.
    
    Parameters
    ----------
    data : np.ndarray or vtk.vtkImageData
        Input voxel data
    foreground_value : int, default=1
        Value representing foreground voxels
    
    Returns
    -------
    tuple
        (binary_array, spacing)
        - binary_array: Binary numpy array (True for foreground, False for background)
          Array shape is (z, y, x) to match VTK convention
        - spacing: Tuple of (spacing_x, spacing_y, spacing_z) in physical units
    """
    if isinstance(data, vtk.vtkImageData):
        # Extract numpy array from VTK
        dims = data.GetDimensions()  # Returns (x, y, z)
        spacing = data.GetSpacing()  # Returns (spacing_x, spacing_y, spacing_z)
        
        # Get scalar data
        scalars = data.GetPointData().GetScalars()
        if scalars is None:
            raise ValueError("vtkImageData has no scalar data")
        
        arr = vtk.util.numpy_support.vtk_to_numpy(scalars)
        # VTK stores data in (x, y, z) order, reshape accordingly
        arr = arr.reshape(dims[::-1])  # Reshape to (z, y, x) for numpy convention
        
        binary = (arr == foreground_value)
        return binary, spacing
    else:
        # Assume numpy array - keep original shape
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got {arr.ndim}D")
        
        binary = (arr == foreground_value)
        # Default spacing is 1.0 for each dimension
        spacing = (1.0, 1.0, 1.0)
        return binary, spacing


def calculate_volume(
    data: Union[np.ndarray, vtk.vtkImageData],
    foreground_value: int = 1,
    spacing: Optional[Tuple[float, float, float]] = None
) -> float:
    """
    Calculate the volume of foreground voxels in voxel data.
    
    Parameters
    ----------
    data : np.ndarray or vtk.vtkImageData
        Input voxel data (3D binary array or vtkImageData)
    foreground_value : int, default=1
        Value representing foreground voxels
    spacing : tuple of float, optional
        Physical spacing of voxels (spacing_x, spacing_y, spacing_z).
        If None and data is numpy array, assumes spacing of 1.0 for all dimensions.
        If data is vtkImageData, uses spacing from the image data.
    
    Returns
    -------
    float
        Volume in physical units (spacing_x * spacing_y * spacing_z * voxel_count)
    
    Examples
    --------
    >>> import numpy as np
    >>> from downstream.morph.geometry import calculate_volume
    >>> 
    >>> # Create a 3D binary array
    >>> data = np.zeros((100, 100, 50), dtype=np.uint8)
    >>> data[40:60, 40:60, 20:30] = 1  # A rectangular region
    >>> 
    >>> # Calculate volume with default spacing (1.0)
    >>> volume = calculate_volume(data)
    >>> print(f"Volume: {volume} cubic units")
    >>> 
    >>> # Calculate volume with custom spacing (e.g., microns)
    >>> volume = calculate_volume(data, spacing=(0.5, 0.5, 2.0))
    >>> print(f"Volume: {volume} cubic microns")
    """
    binary, data_spacing = _as_numpy_array(data, foreground_value)
    
    # Use provided spacing or spacing from data
    if spacing is None:
        spacing = data_spacing
    
    # Count foreground voxels
    voxel_count = np.sum(binary)
    
    # Calculate volume
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    volume = voxel_count * voxel_volume
    
    return float(volume)


def calculate_surface_area(
    data: Union[np.ndarray, vtk.vtkImageData],
    foreground_value: int = 1,
    spacing: Optional[Tuple[float, float, float]] = None
) -> float:
    """
    Calculate the surface area of foreground voxels in voxel data.
    
    The surface area is computed by counting exposed faces of foreground voxels.
    Each exposed face contributes to the total surface area.
    
    Parameters
    ----------
    data : np.ndarray or vtk.vtkImageData
        Input voxel data (3D binary array or vtkImageData)
    foreground_value : int, default=1
        Value representing foreground voxels
    spacing : tuple of float, optional
        Physical spacing of voxels (spacing_x, spacing_y, spacing_z).
        If None and data is numpy array, assumes spacing of 1.0 for all dimensions.
        If data is vtkImageData, uses spacing from the image data.
    
    Returns
    -------
    float
        Surface area in physical units (sum of exposed face areas)
    
    Examples
    --------
    >>> import numpy as np
    >>> from downstream.morph.geometry import calculate_surface_area
    >>> 
    >>> # Create a 3D binary array
    >>> data = np.zeros((100, 100, 50), dtype=np.uint8)
    >>> data[40:60, 40:60, 20:30] = 1  # A rectangular region
    >>> 
    >>> # Calculate surface area with default spacing
    >>> surface_area = calculate_surface_area(data)
    >>> print(f"Surface area: {surface_area} square units")
    >>> 
    >>> # Calculate surface area with custom spacing
    >>> surface_area = calculate_surface_area(data, spacing=(0.5, 0.5, 2.0))
    >>> print(f"Surface area: {surface_area} square microns")
    """
    binary, data_spacing = _as_numpy_array(data, foreground_value)
    
    # Use provided spacing or spacing from data
    if spacing is None:
        spacing = data_spacing
    
    # Pad binary array to handle boundaries (pad with False/background)
    # binary shape is (z, y, x)
    padded = np.pad(binary, pad_width=1, mode='constant', constant_values=False)
    
    # Count exposed faces
    # A face is exposed if the current voxel is foreground and the neighbor is background
    # binary shape: (z, y, x)
    # padded shape: (z+2, y+2, x+2)
    # padded[1:-1, 1:-1, 1:-1] corresponds to binary
    
    surface_area = 0.0
    
    # Face in +x direction (right): check neighbor at (z, y, x+1)
    # Area of x-face = spacing_y * spacing_z
    neighbor_x_plus = padded[1:-1, 1:-1, 2:]  # Shape: (z, y, x)
    exposed = binary & ~neighbor_x_plus
    surface_area += np.sum(exposed) * spacing[1] * spacing[2]
    
    # Face in -x direction (left): check neighbor at (z, y, x-1)
    neighbor_x_minus = padded[1:-1, 1:-1, :-2]  # Shape: (z, y, x)
    exposed = binary & ~neighbor_x_minus
    surface_area += np.sum(exposed) * spacing[1] * spacing[2]
    
    # Face in +y direction (front): check neighbor at (z, y+1, x)
    # Area of y-face = spacing_x * spacing_z
    neighbor_y_plus = padded[1:-1, 2:, 1:-1]  # Shape: (z, y, x)
    exposed = binary & ~neighbor_y_plus
    surface_area += np.sum(exposed) * spacing[0] * spacing[2]
    
    # Face in -y direction (back): check neighbor at (z, y-1, x)
    neighbor_y_minus = padded[1:-1, :-2, 1:-1]  # Shape: (z, y, x)
    exposed = binary & ~neighbor_y_minus
    surface_area += np.sum(exposed) * spacing[0] * spacing[2]
    
    # Face in +z direction (top): check neighbor at (z+1, y, x)
    # Area of z-face = spacing_x * spacing_y
    neighbor_z_plus = padded[2:, 1:-1, 1:-1]  # Shape: (z, y, x)
    exposed = binary & ~neighbor_z_plus
    surface_area += np.sum(exposed) * spacing[0] * spacing[1]
    
    # Face in -z direction (bottom): check neighbor at (z-1, y, x)
    neighbor_z_minus = padded[:-2, 1:-1, 1:-1]  # Shape: (z, y, x)
    exposed = binary & ~neighbor_z_minus
    surface_area += np.sum(exposed) * spacing[0] * spacing[1]
    
    return float(surface_area)


def calculate_volume_and_surface_area(
    data: Union[np.ndarray, vtk.vtkImageData],
    foreground_value: int = 1,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, float]:
    """
    Calculate both volume and surface area of foreground voxels.
    
    This is a convenience function that computes both metrics in a single pass.
    
    Parameters
    ----------
    data : np.ndarray or vtk.vtkImageData
        Input voxel data (3D binary array or vtkImageData)
    foreground_value : int, default=1
        Value representing foreground voxels
    spacing : tuple of float, optional
        Physical spacing of voxels (spacing_x, spacing_y, spacing_z).
        If None and data is numpy array, assumes spacing of 1.0 for all dimensions.
        If data is vtkImageData, uses spacing from the image data.
    
    Returns
    -------
    tuple
        (volume, surface_area)
        - volume: Volume in physical units
        - surface_area: Surface area in physical units
    
    Examples
    --------
    >>> import numpy as np
    >>> from downstream.morph.geometry import calculate_volume_and_surface_area
    >>> 
    >>> data = np.zeros((100, 100, 50), dtype=np.uint8)
    >>> data[40:60, 40:60, 20:30] = 1
    >>> 
    >>> volume, surface_area = calculate_volume_and_surface_area(data)
    >>> print(f"Volume: {volume}, Surface area: {surface_area}")
    """
    volume = calculate_volume(data, foreground_value, spacing)
    surface_area = calculate_surface_area(data, foreground_value, spacing)
    return volume, surface_area
