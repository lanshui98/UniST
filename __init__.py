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

import sys
import os

# Get package root directory (now at root level)
_package_root = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Upsampling module (RepKPU_ops)
# ============================================================================
_external_repkpu_path = os.path.join(_package_root, 'external', 'RepKPU_ops')
if os.path.exists(_external_repkpu_path) and _external_repkpu_path not in sys.path:
    sys.path.insert(0, _external_repkpu_path)

try:
    from models.repkpu import RepKPU, RepKPU_o
except ImportError as e:
    raise ImportError(
        f"Could not import RepKPU from external/RepKPU_ops/: {e}\n"
        f"Expected path: {_external_repkpu_path}\n"
        "Please ensure RepKPU_ops code is in external/RepKPU_ops/ directory."
    )

# ============================================================================
# Interpolation module (InterpolAI)
# ============================================================================
_external_interpolai_path = os.path.join(_package_root, 'external', 'InterpolAI')
if os.path.exists(_external_interpolai_path) and _external_interpolai_path not in sys.path:
    sys.path.insert(0, _external_interpolai_path)

try:
    from interpolation.interpolation_function_auto import (
        interpolate_from_image_list,
        list_skip_images
    )
    from interpolation.interpolation_function_skip import (
        interpolate_from_image_stack_skip
    )
    from interpolation.interpolation_functions_no_skip import (
        interpolate_from_image_stack_no_skip
    )
    _INTERPOLATION_AVAILABLE = True
except ImportError as e:
    _INTERPOLATION_AVAILABLE = False
    # Define placeholders if interpolation is not available
    interpolate_from_image_list = None
    list_skip_images = None
    interpolate_from_image_stack_skip = None
    interpolate_from_image_stack_no_skip = None

# ============================================================================
# Imputation module (SUICA_pro)
# ============================================================================
_external_suica_path = os.path.join(_package_root, 'external', 'SUICA_pro')
if os.path.exists(_external_suica_path) and _external_suica_path not in sys.path:
    sys.path.insert(0, _external_suica_path)

try:
    # Import main functions from systems module
    from systems import (
        train_inr,
        train_embedder,
        fit_griddata,
        predict_inr
    )
    # Import network classes
    from networks import (
        NGP,
        SirenNet as SIREN,
        FourierFeatureNet as FFN,
        GAE,
        AE
    )
    _IMPUTATION_AVAILABLE = True
except ImportError as e:
    _IMPUTATION_AVAILABLE = False
    # Define placeholders if imputation is not available
    train_inr = None
    train_embedder = None
    fit_griddata = None
    predict_inr = None
    NGP = None
    SIREN = None
    FFN = None
    GAE = None
    AE = None

__all__ = [
    # Upsampling
    "RepKPU",
    "RepKPU_o",
    # Interpolation
    "interpolate_from_image_list",
    "list_skip_images",
    "interpolate_from_image_stack_skip",
    "interpolate_from_image_stack_no_skip",
    # Imputation
    "train_inr",
    "train_embedder",
    "fit_griddata",
    "predict_inr",
    "NGP",
    "SIREN",
    "FFN",
    "GAE",
    "AE",
]
