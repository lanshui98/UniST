"""
ReHo: Registration-based Hermite interpolation for slice interpolation.

This module provides Python wrappers for MATLAB-based slice interpolation
algorithms using curvature-regularized, registration-based spline interpolation.

Requires MATLAB Engine API for Python to be installed.
"""

import os
import numpy as np
from pathlib import Path

# Get the directory containing this __init__.py
_REHO_DIR = Path(__file__).parent

# Try to import MATLAB engine
try:
    import matlab.engine
    _MATLAB_AVAILABLE = True
except ImportError:
    _MATLAB_AVAILABLE = False
    matlab = None

__all__ = [
    'slice_interp_spline_intensity_spline',
    'interpolate_missing_slices_spline',
    'is_matlab_available',
]


def is_matlab_available():
    """Check if MATLAB Engine API is available."""
    return _MATLAB_AVAILABLE


def _get_matlab_engine():
    """Get or start MATLAB engine."""
    if not _MATLAB_AVAILABLE:
        raise RuntimeError(
            "MATLAB Engine API for Python is not installed. "
            "Please install it using: pip install matlabengine"
        )
    
    try:
        # Try to get existing engine
        eng = matlab.engine.find_matlab()
        if eng:
            return matlab.engine.connect_matlab(eng[0])
    except:
        pass
    
    # Start new engine
    return matlab.engine.start_matlab()


def _add_reho_path(eng):
    """Add ReHo directory to MATLAB path."""
    eng.addpath(str(_REHO_DIR), nargout=0)


def slice_interp_spline_intensity_spline(
    I,
    z,
    R,
    lambda_val=10,
    tau=10,
    TOL=0.001,
    maxIter=100,
    borderSize=0.1,
    vx=None,
    vy=None
):
    """
    Curvature regularized, registration-based spline-spline slice interpolation.
    
    Python wrapper for sliceInterp_spline_intensitySpline.m
    
    Parameters
    ----------
    I : numpy.ndarray
        3D array of size M x N x P; list of matrices of same dimensions
    z : numpy.ndarray or list
        1D array of size P; z positions of the parallel slices
    R : float
        Refinement factor; ratio of new dz over old dz
    lambda_val : float, default=10
        Curvature regularization factor
    tau : float, default=10
        Implicit gradient descent step size
    TOL : float, default=0.001
        Stopping criteria; tolerance of the max(L1(L2)) norm
    maxIter : int, default=100
        Maximal amount of iterations
    borderSize : float, default=0.1
        Relative border size w.r.t. M and N
    vx : numpy.ndarray, optional
        3D array of size MxNx(P-1); initial displacement field in x direction
    vy : numpy.ndarray, optional
        3D array of size MxNx(P-1); initial displacement field in y direction
    
    Returns
    -------
    Iinterp : numpy.ndarray
        3D array of size MxNx(R(P-1)+1); interpolated slices + original ones
    zinterp : numpy.ndarray
        1D array of size (R(P-1)+1); z positions of interpolated slices
    vx : numpy.ndarray
        3D array of size MxNx(P-1); registered displacement field in x
    vy : numpy.ndarray
        3D array of size MxNx(P-1); registered displacement field in y
    
    Raises
    ------
    RuntimeError
        If MATLAB Engine API is not available
    """
    eng = _get_matlab_engine()
    _add_reho_path(eng)
    
    # Convert numpy arrays to MATLAB arrays
    I_matlab = matlab.double(I.tolist())
    z_matlab = matlab.double(z if isinstance(z, list) else z.tolist())
    
    # Prepare optional arguments
    args = [I_matlab, z_matlab, float(R), float(lambda_val), float(tau), 
            float(TOL), int(maxIter), float(borderSize)]
    
    if vx is not None:
        args.append(matlab.double(vx.tolist()))
    if vy is not None:
        args.append(matlab.double(vy.tolist()))
    
    # Call MATLAB function
    result = eng.sliceInterp_spline_intensitySpline(*args, nargout=4)
    
    # Convert results back to numpy arrays
    Iinterp = np.array(result[0])
    zinterp = np.array(result[1]).flatten()
    vx_out = np.array(result[2])
    vy_out = np.array(result[3])
    
    return Iinterp, zinterp, vx_out, vy_out


def interpolate_missing_slices_spline(
    in_dir,
    out_dir,
    copy_original=True,
    binarize=False,
    thresh=127,
    lambda_val=10,
    tau=100,
    TOL=0.01,
    maxIter=1000,
    borderSize=0.1
):
    """
    Interpolate missing slices between existing slices using spline interpolation.
    
    Python wrapper for interpolate_missing_slices_spline.m
    
    Parameters
    ----------
    in_dir : str or Path
        Input directory containing slice_XXX.tif files
    out_dir : str or Path
        Output directory for interpolated slices
    copy_original : bool, default=True
        Whether to copy original slices to output directory
    binarize : bool, default=False
        Whether to binarize the output
    thresh : int, default=127
        Threshold for binarization (0-255)
    lambda_val : float, default=10
        Curvature regularization factor
    tau : float, default=100
        Implicit gradient descent step size
    TOL : float, default=0.01
        Stopping criteria tolerance
    maxIter : int, default=1000
        Maximum iterations
    borderSize : float, default=0.1
        Relative border size
    
    Raises
    ------
    RuntimeError
        If MATLAB Engine API is not available
    """
    eng = _get_matlab_engine()
    _add_reho_path(eng)
    
    # Convert paths to strings
    in_dir = str(Path(in_dir).absolute())
    out_dir = str(Path(out_dir).absolute())
    
    # Call MATLAB function
    eng.interpolate_missing_slices_spline(
        in_dir,
        out_dir,
        'CopyOriginal', copy_original,
        'Binarize', binarize,
        'Thresh', float(thresh),
        'Lambda', float(lambda_val),
        'Tau', float(tau),
        'TOL', float(TOL),
        'MaxIter', int(maxIter),
        'BorderSize', float(borderSize),
        nargout=0
    )
