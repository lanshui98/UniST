"""
UniST - A Unified Computational Framework for 3D Spatial Transcriptomics Reconstruction.

This package provides three main functionalities:
- Upsampling: Point cloud upsampling using RepKPU
- Interpolation: Slice interpolation using InterpolAI
- Imputation: Gene expression imputation using SUICA_pro

Quick middle-slice API::

    from unist import predict_middle_slice
    middle = predict_middle_slice(slice_a, slice_b, mode="fast", label_key="cell_type")
    middle = predict_middle_slice(slice_a, slice_b, mode="inr", work_dir="./unist_run")
"""

from typing import Any

__version__ = "0.1.1"
__author__ = "Lan Shui"
__email__ = "lshui@mdanderson.org"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "predict_middle_slice",
]


def __getattr__(name: str) -> Any:
    if name == "predict_middle_slice":
        from unist.one_line_code import predict_middle_slice

        return predict_middle_slice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
