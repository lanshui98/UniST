"""
Command-line interface for InterpolAI interpolation.

This module provides a CLI wrapper for the external InterpolAI code in external/InterpolAI/.
"""

import argparse
import os
import sys

# Add external/InterpolAI to Python path
# Try multiple possible paths to handle different environments (local, Colab, etc.)
from pathlib import Path

_possible_paths = [
    Path(__file__).parent.parent.parent / 'external' / 'InterpolAI',
    Path.cwd() / 'external' / 'InterpolAI',
    Path.cwd() / 'UniST' / 'external' / 'InterpolAI',
]

_external_interpolai_path = None
for path in _possible_paths:
    path = path.resolve()
    if path.exists():
        _external_interpolai_path = str(path)
        break

# Fallback to original method
if _external_interpolai_path is None:
    _package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _external_interpolai_path = os.path.join(_package_root, 'external', 'InterpolAI')

# Add to sys.path if it exists
if not os.path.exists(_external_interpolai_path):
    print(f"Error: InterpolAI path not found: {_external_interpolai_path}")
    sys.exit(1)

# Ensure __init__.py files exist (for Colab compatibility)
interpolation_dir = os.path.join(_external_interpolai_path, "interpolation")
interpolation_init = os.path.join(interpolation_dir, "__init__.py")
if os.path.exists(interpolation_dir) and not os.path.exists(interpolation_init):
    with open(interpolation_init, 'w') as f:
        f.write('"""InterpolAI interpolation functions."""\n')

interpolai_init = os.path.join(_external_interpolai_path, "__init__.py")
if not os.path.exists(interpolai_init):
    with open(interpolai_init, 'w') as f:
        f.write('"""InterpolAI: Slice interpolation for 3D spatial transcriptomics."""\n')

if _external_interpolai_path not in sys.path:
    sys.path.insert(0, _external_interpolai_path)

# Import dependencies
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Error: Could not import TensorFlow: {e}")
    sys.exit(1)

# Import interpolation functions
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
except ImportError as e:
    print(f"Error: Could not import InterpolAI functions: {e}")
    print(f"Please ensure InterpolAI is properly set up in {_external_interpolai_path}")
    sys.exit(1)


def load_model():
    """Load the TensorFlow SavedModel from InterpolAI package."""
    model_path = os.path.join(_external_interpolai_path, "interpolation", "model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please ensure the model directory exists in external/InterpolAI/interpolation/model/"
        )
    
    return tf.saved_model.load(model_path)


def run_auto(tile_size, pth, model):
    """Run interpolation in auto mode."""
    image_files = [f for f in os.listdir(pth) if f.endswith(('tif', 'png', 'jpg'))]
    skip_images = list_skip_images(pth)
    interpolate_from_image_list(pth, skip_images, tile_size, model, image_files)


def run_no_skip(tile_size, pth, skip, model):
    """Run interpolation in no_skip mode."""
    image_files = [f for f in os.listdir(pth) if f.endswith(('tif', 'png', 'jpg'))]
    interpolate_from_image_stack_no_skip(pth, skip, tile_size, model, image_files)


def run_skip(tile_size, pth, skip, model):
    """Run interpolation in skip mode."""
    image_files = [f for f in os.listdir(pth) if f.endswith(('tif', 'png', 'jpg'))]
    interpolate_from_image_stack_skip(pth, skip, tile_size, model, image_files)


def main():
    """Main entry point for the interpolation CLI."""
    parser = argparse.ArgumentParser(
        description="InterpolAI: Slice interpolation for 3D spatial transcriptomics"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "no_skip", "skip"],
        required=True,
        help="Interpolation mode"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        required=True,
        help="Tile size for processing (e.g., --tile_size 1024 1024)"
    )
    parser.add_argument(
        "--pth",
        type=str,
        required=True,
        help="Path to folder containing input images"
    )
    parser.add_argument(
        "--skip",
        nargs='+',
        type=int,
        default=[1],
        help="Skip values (required for no_skip and skip modes)"
    )

    args = parser.parse_args()
    
    # Convert relative path to absolute path
    if not os.path.isabs(args.pth):
        args.pth = os.path.abspath(args.pth)
    
    if not os.path.exists(args.pth):
        print(f"Error: Path does not exist: {args.pth}")
        sys.exit(1)
    
    try:
        model = load_model()
        if args.mode == "auto":
            run_auto(tuple(args.tile_size), args.pth, model)
        elif args.mode == "no_skip":
            run_no_skip(tuple(args.tile_size), args.pth, args.skip, model)
        elif args.mode == "skip":
            run_skip(tuple(args.tile_size), args.pth, args.skip, model)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
