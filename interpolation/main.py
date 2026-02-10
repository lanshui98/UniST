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
if os.path.exists(_external_interpolai_path):
    if _external_interpolai_path not in sys.path:
        sys.path.insert(0, _external_interpolai_path)
        print(f"✅ Added InterpolAI to path: {_external_interpolai_path}")
else:
    print(f"⚠️  Warning: InterpolAI path not found: {_external_interpolai_path}")
    print(f"Tried paths: {_possible_paths}")
    print(f"Current working directory: {os.getcwd()}")

# Try to import interpolation functions
try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported (version: {tf.__version__})")
except ImportError as e:
    print(f"❌ Error: Could not import TensorFlow: {e}")
    sys.exit(1)

# Import interpolation functions
try:
    # Try importing from the external InterpolAI directory
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
    print(f"✅ InterpolAI functions imported successfully")
except ImportError as e:
    print(f"❌ Error: Could not import InterpolAI functions: {e}")
    print(f"\nExpected path: {_external_interpolai_path}")
    print(f"Path exists: {os.path.exists(_external_interpolai_path) if _external_interpolai_path else False}")
    print(f"Current sys.path entries (first 5):")
    for i, p in enumerate(sys.path[:5]):
        print(f"  {i}: {p}")
    print("\nPlease ensure:")
    print("  1. InterpolAI code is in external/InterpolAI/ directory")
    print("  2. You are running from the UniST root directory")
    print("  3. In Colab, ensure you have run: %cd UniST")
    sys.exit(1)


def load_model():
    """Load the TensorFlow SavedModel from InterpolAI package."""
    # Try to get model path from the external InterpolAI directory
    if _external_interpolai_path and os.path.exists(_external_interpolai_path):
        # Model should be at external/InterpolAI/interpolation/model
        model_path = os.path.join(_external_interpolai_path, "interpolation", "model")
    else:
        # Fallback: try to get from imported interpolation module
        try:
            import interpolation
            interpolation_path = os.path.dirname(interpolation.__file__)
            model_path = os.path.join(interpolation_path, "model")
        except Exception:
            # Last resort: relative path from current working directory
            model_path = os.path.join("external", "InterpolAI", "interpolation", "model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please ensure InterpolAI is properly set up and the model directory exists. "
            f"Expected location: {os.path.join(_external_interpolai_path or 'external/InterpolAI', 'interpolation', 'model')}"
        )
    print(f"Loading model from: {model_path}")
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
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(args.pth):
        # Try to resolve relative to current working directory
        args.pth = os.path.abspath(args.pth)
        # If still doesn't exist, try relative to project root
        if not os.path.exists(args.pth):
            project_root = Path(__file__).parent.parent.parent
            potential_path = project_root / args.pth
            if potential_path.exists():
                args.pth = str(potential_path.resolve())
    
    if not os.path.exists(args.pth):
        print(f"Error: Path does not exist: {args.pth}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    print(f"Using input directory: {args.pth}")
    
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
