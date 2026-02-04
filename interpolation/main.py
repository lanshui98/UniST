"""
Command-line interface for InterpolAI interpolation.

This module provides a CLI wrapper for the external InterpolAI code in external/InterpolAI/.
"""

import argparse
import os
import sys

# Add external/InterpolAI to Python path
_package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_external_interpolai_path = os.path.join(_package_root, 'external', 'InterpolAI')
if os.path.exists(_external_interpolai_path) and _external_interpolai_path not in sys.path:
    sys.path.insert(0, _external_interpolai_path)

try:
    import tensorflow as tf
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
    print(f"Error: Could not import InterpolAI: {e}")
    print(f"\nExpected path: {_external_interpolai_path}")
    print("Please ensure InterpolAI code is in external/InterpolAI/ directory.")
    sys.exit(1)


def load_model():
    """Load the TensorFlow SavedModel from InterpolAI package."""
    import interpolation
    interpolation_path = os.path.dirname(interpolation.__file__)
    model_path = os.path.join(interpolation_path, "model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please ensure InterpolAI is properly set up and the model directory exists."
        )
    model = tf.saved_model.load(model_path)
    return model


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
    
    if not os.path.exists(args.pth):
        print(f"Error: Path does not exist: {args.pth}")
        sys.exit(1)
    
    print("Loading model...")
    try:
        model = load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"Running interpolation in '{args.mode}' mode...")
    try:
        if args.mode == "auto":
            run_auto(tuple(args.tile_size), args.pth, model)
        elif args.mode == "no_skip":
            run_no_skip(tuple(args.tile_size), args.pth, args.skip, model)
        elif args.mode == "skip":
            run_skip(tuple(args.tile_size), args.pth, args.skip, model)
        print("Interpolation completed successfully!")
    except Exception as e:
        print(f"Error during interpolation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
