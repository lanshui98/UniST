import argparse
import os
import tensorflow as tf

# Import your functions
from interpolation.interpolation_function_auto import *
from interpolation.interpolation_function_skip import *
from interpolation.interpolation_functions_no_skip import *

def load_model():
    model_path = os.path.join("interpolation", "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.saved_model.load(model_path)
    return model

def run_auto(tile_size, pth, model):
    image_files = [f for f in os.listdir(pth) if f.endswith(('tif', 'png', 'jpg'))]
    skip_images = list_skip_images(pth)
    interpolate_from_image_list(pth, skip_images, tile_size, model, image_files)

def run_no_skip(tile_size, pth, skip, model):
    image_files = [f for f in os.listdir(pth) if f.endswith(('tif', 'png', 'jpg'))]
    interpolate_from_image_stack_no_skip(pth, skip, tile_size, model, image_files)

def run_skip(tile_size, pth, skip, model):
    image_files = [f for f in os.listdir(pth) if f.endswith(('tif', 'png', 'jpg'))]
    interpolate_from_image_stack_skip(pth, skip, tile_size, model, image_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interpolation modes from CLI")
    parser.add_argument("--mode", choices=["auto", "no_skip", "skip"], required=True, help="Choose a mode")
    parser.add_argument("--tile_size", type=int, nargs=2, required=True, help="Tile size (e.g., --tile_size 1024 1024)")
    parser.add_argument("--pth", type=str, required=True, help="Path to image folder")
    parser.add_argument('--skip', nargs='+', type=int, default=[1], help='Skip values')

    args = parser.parse_args()
    model = load_model()

    if args.mode == "auto":
        run_auto(tuple(args.tile_size), args.pth, model)
    elif args.mode == "no_skip":
        if args.skip is None:
            raise ValueError("Skip value is required for no_skip mode")
        run_no_skip(tuple(args.tile_size), args.pth, args.skip, model)
    elif args.mode == "skip":
        if args.skip is None:
            raise ValueError("Skip value is required for skip mode")
        run_skip(tuple(args.tile_size), args.pth, args.skip, model)
