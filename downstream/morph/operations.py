"""
Morphological operations implementation using VTK.

This module provides dilation, erosion, closing, and opening operations for vtkImageData.
These operations are particularly useful for bridging gaps along the Z-axis
in 3D spatial transcriptomics data.
"""

import numpy as np
import vtk
from vtk.util import numpy_support

try:
    import pyvista as pv
except ImportError:
    pv = None


def _wrap_output(vtk_output):
    """Return PyVista wrapper when available so result has .save() etc."""
    if pv is not None:
        return pv.wrap(vtk_output)
    return vtk_output


def _remap_nonzero_to(vtk_image, value):
    """In-place: set all non-zero scalars to value (for point and cell data)."""
    if value == 1:
        return
    for data in (vtk_image.GetPointData(), vtk_image.GetCellData()):
        s = data.GetScalars()
        if s is None:
            continue
        arr = numpy_support.vtk_to_numpy(s)
        out = np.where(arr != 0, value, 0)
        out = out.astype(np.float64 if isinstance(value, float) else np.int32)
        new_s = numpy_support.numpy_to_vtk(out.ravel(), deep=True)
        new_s.SetName(s.GetName() if s.GetName() else "")
        data.SetScalars(new_s)


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


def periphery_mask(
    image_data,
    foreground_value=1,
    background_value=0,
    kernel_size=(5, 5, 3),
    output_value=1,
):
    """
    Compute a mask of the periphery (outer ring) around the foreground region.

    Performs dilation then subtracts the original volume, so the result is
    output_value where the dilated region extends beyond the original foreground (0 elsewhere).
    Useful for defining a "shell" or invasive margin around a tumor mask.

    Parameters
    ----------
    image_data : vtk.vtkImageData or pyvista.ImageData
        Input volume with foreground (e.g. tumor) and background.
    foreground_value : int, default=1
        Voxel value that represents the foreground to dilate (e.g. tumor = 1 or 4).
    background_value : int, default=0
        Background value for the dilation step.
    kernel_size : tuple of int, default=(5, 5, 3)
        Dilation kernel (kernel_x, kernel_y, kernel_z). Larger values give a thicker ring.
    output_value : int or float, default=1
        Value assigned to the periphery voxels (non-zero in the ring). Use e.g. 2 or 255 for a distinct label.

    Returns
    -------
    pyvista.ImageData or vtk.vtkImageData
        Volume the same size as input; output_value in the periphery ring, 0 elsewhere.
        Has .save() when PyVista is available.

    Example
    -------
    periphery = periphery_mask(closed_volume_data, foreground_value=4, background_value=0, kernel_size=(5, 5, 3), output_value=2)
    periphery.save("periphery_mask.vti")
    """
    dilate = vtk.vtkImageDilateErode3D()
    dilate.SetInputData(image_data)
    dilate.SetDilateValue(foreground_value)
    dilate.SetErodeValue(background_value)
    dilate.SetKernelSize(kernel_size[0], kernel_size[1], kernel_size[2])
    dilate.Update()
    dilated_vtk = dilate.GetOutput()

    subtract = vtk.vtkImageMathematics()
    subtract.SetOperationToSubtract()
    subtract.SetInput1Data(dilated_vtk)
    subtract.SetInput2Data(image_data)
    subtract.Update()
    result = subtract.GetOutput()
    _remap_nonzero_to(result, output_value)
    return _wrap_output(result)


def boundary_mask(
    image_data,
    foreground_value=1,
    background_value=0,
    kernel_size=(3, 3, 3),
    output_value=1,
):
    """
    Compute a mask of the boundary (surface shell) of the foreground region.

    Performs a small erosion then subtracts the eroded volume from the original,
    so the result is output_value on the boundary voxels (thickness controlled by kernel).
    Useful for defining the tumor boundary or invasive front.

    Parameters
    ----------
    image_data : vtk.vtkImageData or pyvista.ImageData
        Input volume (e.g. closed tumor mask) with foreground and background.
    foreground_value : int, default=1
        Voxel value that represents the foreground to erode (e.g. tumor = 1 or 4).
    background_value : int, default=0
        Background value for the erosion step.
    kernel_size : tuple of int, default=(3, 3, 3)
        Erosion kernel (kernel_x, kernel_y, kernel_z). Size controls boundary thickness.
    output_value : int or float, default=1
        Value assigned to the boundary voxels (non-zero on the shell). Use e.g. 2 or 255 for a distinct label.

    Returns
    -------
    pyvista.ImageData or vtk.vtkImageData
        Volume the same size as input; output_value on the boundary shell, 0 elsewhere.
        Has .save() when PyVista is available.

    Example
    -------
    boundary = boundary_mask(closed_volume_data, foreground_value=4, background_value=0, kernel_size=(3, 3, 3), output_value=2)
    boundary.save("boundary_mask.vti")
    """
    erode = vtk.vtkImageDilateErode3D()
    erode.SetInputData(image_data)
    erode.SetDilateValue(background_value)
    erode.SetErodeValue(foreground_value)
    erode.SetKernelSize(kernel_size[0], kernel_size[1], kernel_size[2])
    erode.Update()
    eroded_vtk = erode.GetOutput()

    subtract = vtk.vtkImageMathematics()
    subtract.SetOperationToSubtract()
    subtract.SetInput1Data(image_data)
    subtract.SetInput2Data(eroded_vtk)
    subtract.Update()
    result = subtract.GetOutput()
    _remap_nonzero_to(result, output_value)
    return _wrap_output(result)
