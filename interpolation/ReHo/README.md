# ReHo: Registration-based Hermite Interpolation

This module provides Python wrappers for MATLAB-based slice interpolation algorithms using curvature-regularized, registration-based spline interpolation.

## Citation

If you use this algorithm, please cite:

```
Antal Horvath, Simon Pezold, Matthias Weigel, Katrin Parmar, and Philippe Cattin
"High Order Slice Interpolation for Medical Images."
Simulation and Synthesis in Medical Imaging (MICCAI 2017 Workshop SASHIMI). 2017
```

## Requirements

1. **MATLAB** (R2014b or later)
2. **MATLAB Engine API for Python**

### Installing MATLAB Engine API for Python

1. Navigate to your MATLAB installation directory:
   ```bash
   cd /path/to/matlab/extern/engines/python
   ```

2. Install the engine:
   ```bash
   python setup.py install
   ```
   
   Or using pip (if available):
   ```bash
   pip install matlabengine
   ```

## Usage

### Basic Example

```python
from interpolation.ReHo import (
    slice_interp_spline_intensity_spline,
    interpolate_missing_slices_spline
)
import numpy as np

# Check if MATLAB is available
from interpolation.ReHo import is_matlab_available
if not is_matlab_available():
    print("MATLAB Engine API is not installed")
    exit(1)

# Example 1: Direct interpolation
I = np.random.rand(100, 100, 2)  # 2 slices
z = [0, 1]  # z positions
R = 5  # refinement factor

Iinterp, zinterp, vx, vy = slice_interp_spline_intensity_spline(
    I, z, R,
    lambda_val=10,
    tau=10,
    TOL=0.001,
    maxIter=100
)

# Example 2: Interpolate missing slices from directory
interpolate_missing_slices_spline(
    in_dir="/path/to/input/slices",
    out_dir="/path/to/output/slices",
    copy_original=True,
    binarize=False,
    lambda_val=10,
    tau=100,
    TOL=0.01,
    maxIter=1000
)
```

### Function Parameters

#### `slice_interp_spline_intensity_spline`

- `I`: 3D numpy array (M x N x P) - input slices
- `z`: 1D array of z positions
- `R`: Refinement factor
- `lambda_val`: Curvature regularization factor (default: 10)
- `tau`: Implicit gradient descent step size (default: 10)
- `TOL`: Stopping criteria tolerance (default: 0.001)
- `maxIter`: Maximum iterations (default: 100)
- `borderSize`: Relative border size (default: 0.1)
- `vx`, `vy`: Optional initial displacement fields

#### `interpolate_missing_slices_spline`

- `in_dir`: Input directory with `slice_XXX.tif` files
- `out_dir`: Output directory for interpolated slices
- `copy_original`: Copy original slices to output (default: True)
- `binarize`: Binarize output (default: False)
- `thresh`: Binarization threshold 0-255 (default: 127)
- `lambda_val`: Curvature regularization (default: 10)
- `tau`: Step size (default: 100)
- `TOL`: Tolerance (default: 0.01)
- `maxIter`: Max iterations (default: 1000)
- `borderSize`: Border size (default: 0.1)

## MATLAB Functions

The module contains two MATLAB functions:

1. **`sliceInterp_spline_intensitySpline.m`**: Core interpolation algorithm
2. **`interpolate_missing_slices_spline.m`**: Wrapper for batch processing

These functions are automatically added to MATLAB path when called from Python.

## Notes

- The MATLAB engine will be started automatically on first use
- Make sure MATLAB is properly installed and accessible from command line
- Large images may take significant time to process
- The algorithm uses iterative registration, so convergence depends on image quality and parameters
