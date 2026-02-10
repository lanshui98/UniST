"""
Command-line interface for InterpolAI interpolation.

This module provides a CLI wrapper for the external InterpolAI code in external/InterpolAI/.
"""

import argparse
import os
import sys

# Ensure project root is in Python path for external imports
from pathlib import Path

# Get project root: go up from interpolation/main.py to project root
# __file__ is at interpolation/main.py, so parent.parent is project root
_project_root = Path(__file__).parent.parent.resolve()

# Verify by checking if external directory exists
if not (_project_root / 'external').exists():
    # Fallback: try to find project root by looking for external directory
    current = Path(__file__).parent
    while current != current.parent:
        if (current / 'external').exists():
            _project_root = current.resolve()
            break
        current = current.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Ensure __init__.py files exist for package structure
_external_dir = _project_root / 'external'
_interpolai_dir = _external_dir / 'InterpolAI'
_interpolation_dir = _interpolai_dir / 'interpolation'

# Create __init__.py files if they don't exist
if _external_dir.exists() and not (_external_dir / '__init__.py').exists():
    (_external_dir / '__init__.py').write_text('"""External dependencies."""\n')

if _interpolai_dir.exists() and not (_interpolai_dir / '__init__.py').exists():
    (_interpolai_dir / '__init__.py').write_text('"""InterpolAI: Slice interpolation for 3D spatial transcriptomics."""\n')

if _interpolation_dir.exists() and not (_interpolation_dir / '__init__.py').exists():
    (_interpolation_dir / '__init__.py').write_text('"""InterpolAI interpolation functions."""\n')

# Import dependencies
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Error: Could not import TensorFlow: {e}")
    sys.exit(1)

# Import interpolation functions
try:
    from external.InterpolAI.interpolation.interpolation_function_auto import (
        interpolate_from_image_list,
        list_skip_images
    )
    from external.InterpolAI.interpolation.interpolation_function_skip import (
        interpolate_from_image_stack_skip
    )
    from external.InterpolAI.interpolation.interpolation_functions_no_skip import (
        interpolate_from_image_stack_no_skip
    )
except ImportError as e:
    print(f"Error: Could not import InterpolAI functions: {e}")
    print(f"Please ensure InterpolAI is properly set up in {_interpolai_dir}")
    sys.exit(1)


def load_model():
    """Load the TensorFlow SavedModel from InterpolAI package."""
    model_path = _interpolation_dir / "model"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please ensure the model directory exists in external/InterpolAI/interpolation/model/"
        )
    
    return tf.saved_model.load(str(model_path))


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
