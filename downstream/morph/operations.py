"""
Morphological operations implementation using VTK.

This module provides dilation, erosion, closing, and opening operations for vtkImageData.
These operations are particularly useful for bridging gaps along the Z-axis
in 3D spatial transcriptomics data.
"""

import vtk

try:
    import pyvista as pv
except ImportError:
    pv = None


def _wrap_output(vtk_output):
    """Return PyVista wrapper when available so result has .save() etc."""
    if pv is not None:
        return pv.wrap(vtk_output)
    return vtk_output


def dilation(
    image_data,
    dilate_value=1,
    erode_value=0,
    kernel_size=(1, 1, 2)
):
    """
    Perform morphological dilation on a vtkImageData object.
    
    Dilation expands the foreground regions (specified by dilate_value)
    by the kernel size, which is useful for bridging gaps.
    
    Parameters
    ----------
    image_data : vtk.vtkImageData
        Input image data to dilate
    dilate_value : int, default=1
        The pixel/voxel value that represents the foreground (e.g., tumor regions).
        All voxels with this value will be expanded (dilated) by the kernel size.
        The default value of 1 is specific to certain data formats where
        foreground structures are labeled as 1.
    erode_value : int, default=0
        The pixel/voxel value that represents the background.
        This value is used by VTK to distinguish foreground from background,
        but background voxels are not eroded during dilation operation.
        The default value of 0 represents empty/background regions.
    kernel_size : tuple of int, default=(1, 1, 2)
        Kernel size as (kernel_x, kernel_y, kernel_z).
        Larger Z value helps bridge gaps along the Z-axis.
        For example, (1, 1, 2) expands 1 voxel in XY and 2 voxels in Z.
    
    Returns
    -------
    vtk.vtkImageData
        Dilated image data
    
    Notes
    -----
    In VTK's vtkImageDilateErode3D:
    - SetDilateValue(dilate_value): Voxels with this value will be expanded
    - SetErodeValue(erode_value): Used to distinguish background (not eroded during dilation)
    
    The values 1 and 0 are data-specific labels. Adjust these based on your
    actual data format where different values may represent foreground/background.
    """
    dilate = vtk.vtkImageDilateErode3D()
    dilate.SetInputData(image_data)
    dilate.SetDilateValue(dilate_value)
    dilate.SetErodeValue(erode_value)
    dilate.SetKernelSize(kernel_size[0], kernel_size[1], kernel_size[2])
    dilate.Update()
    return _wrap_output(dilate.GetOutput())


def erosion(
    image_data,
    dilate_value=0,
    erode_value=1,
    kernel_size=(1, 1, 2)
):
    """
    Perform morphological erosion on a vtkImageData object.
    
    Erosion shrinks the foreground regions (specified by erode_value)
    by the kernel size, which is useful for removing small protrusions.
    
    Parameters
    ----------
    image_data : vtk.vtkImageData
        Input image data to erode
    dilate_value : int, default=0
        The pixel/voxel value that represents the background.
        Used to distinguish background from foreground during erosion.
    erode_value : int, default=1
        The pixel/voxel value that represents the foreground (e.g., tumor regions).
        All voxels with this value will be shrunk (eroded) by the kernel size.
        The default value of 1 matches the foreground label used in dilation.
    kernel_size : tuple of int, default=(1, 1, 2)
        Kernel size as (kernel_x, kernel_y, kernel_z).
        Should match the kernel size used in dilation for closing operation.
    
    Returns
    -------
    vtk.vtkImageData
        Eroded image data
    
    Notes
    -----
    In VTK's vtkImageDilateErode3D:
    - SetErodeValue(erode_value): Voxels with this value will be shrunk
    - SetDilateValue(dilate_value): Used to distinguish background
    
    Note that the dilate_value and erode_value are swapped compared to dilation()
    because we're performing the inverse operation.
    """
    erode = vtk.vtkImageDilateErode3D()
    erode.SetInputData(image_data)
    erode.SetDilateValue(dilate_value)
    erode.SetErodeValue(erode_value)
    erode.SetKernelSize(kernel_size[0], kernel_size[1], kernel_size[2])
    erode.Update()
    return _wrap_output(erode.GetOutput())


def closing(
    image_data,
    foreground_value=1,
    background_value=0,
    kernel_size=(1, 1, 2)
):
    """
    Perform morphological closing on a vtkImageData object.
    
    Closing is a combination of dilation followed by erosion.
    It is useful for filling small holes and connecting nearby structures,
    particularly along the Z-axis in 3D data.
    
    Parameters
    ----------
    image_data : vtk.vtkImageData
        Input image data to close
    foreground_value : int, default=1
        The value that represents the foreground (e.g., tumor regions)
    background_value : int, default=0
        The value that represents the background
    kernel_size : tuple of int, default=(1, 1, 2)
        Kernel size as (kernel_x, kernel_y, kernel_z).
        Larger Z value helps bridge gaps along the Z-axis.
        For example, (1, 1, 2) expands 1 voxel in XY and 2 voxels in Z.
    
    Returns
    -------
    vtk.vtkImageData
        Closed image data
    
    Notes
    -----
    Closing operation:
    1. First applies dilation to expand foreground regions and bridge gaps
    2. Then applies erosion to restore the original size while keeping connections
    
    This is particularly useful for 3D spatial transcriptomics data where
    slices may have gaps that need to be connected along the Z-axis.
    """
    # Step 1: Dilation - bridge gaps
    dilated = dilation(
        image_data,
        dilate_value=foreground_value,
        erode_value=background_value,
        kernel_size=kernel_size
    )
    
    # Step 2: Erosion - restore size while keeping connections
    closed = erosion(
        dilated,
        dilate_value=background_value,
        erode_value=foreground_value,
        kernel_size=kernel_size
    )
    
    return _wrap_output(closed)


def opening(
    image_data,
    foreground_value=1,
    background_value=0,
    kernel_size=(1, 1, 2)
):
    """
    Perform morphological opening on a vtkImageData object.
    
    Opening is a combination of erosion followed by dilation.
    It is useful for removing small foreground regions, noise, and smoothing boundaries,
    while preserving the overall shape and size of larger structures.
    
    Parameters
    ----------
    image_data : vtk.vtkImageData
        Input image data to open
    foreground_value : int, default=1
        The value that represents the foreground (e.g., tumor regions)
    background_value : int, default=0
        The value that represents the background
    kernel_size : tuple of int, default=(1, 1, 2)
        Kernel size as (kernel_x, kernel_y, kernel_z).
        Larger Z value affects more voxels along the Z-axis.
        For example, (1, 1, 2) affects 1 voxel in XY and 2 voxels in Z.
    
    Returns
    -------
    vtk.vtkImageData
        Opened image data
    
    Notes
    -----
    Opening operation:
    1. First applies erosion to shrink foreground regions and remove small protrusions
    2. Then applies dilation to restore the original size while keeping small regions removed
    
    This is the inverse of closing:
    - Closing: dilation then erosion - fills holes, connects structures
    - Opening: erosion then dilation - removes small regions, smooths boundaries
    
    Opening is particularly useful for:
    - Removing noise and small artifacts
    - Smoothing object boundaries
    - Separating connected objects
    """
    # Step 1: Erosion - remove small regions and protrusions
    eroded = erosion(
        image_data,
        dilate_value=background_value,
        erode_value=foreground_value,
        kernel_size=kernel_size
    )
    
    # Step 2: Dilation - restore size while keeping small regions removed
    opened = dilation(
        eroded,
        dilate_value=foreground_value,
        erode_value=background_value,
        kernel_size=kernel_size
    )
    
    return _wrap_output(opened)
