import numpy as np
import os
import tensorflow as tf
import imageio
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import time
import re

Image.MAX_IMAGE_PIXELS = None

def load_image(img_path: str):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    image = imageio.imread(img_path)
    image = image.astype(np.float32) / _UINT8_MAX_F
    return image

def get_file_extension(filename):
    return os.path.splitext(filename)[1]

def pad_and_tile_image(image: np.ndarray, tile_size: tuple):
    h, w, _ = image.shape
    tile_height, tile_width = tile_size
    pad_h = (tile_height - h % tile_height) % tile_height
    pad_w = (tile_width - w % tile_width) % tile_width
    padded_image = np.pad(image, pad_width=[(0, pad_h), (0, pad_w), (0, 0)], mode='constant', constant_values=0)
    padded_height, padded_width, _ = padded_image.shape
    tiled_array = padded_image.reshape(padded_height // tile_height,
                                       tile_height,
                                       padded_width // tile_width,
                                       tile_width,
                                       -1)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array, (pad_h, pad_w)

def stitch_tiles(tiles: np.ndarray, pad_h: int, pad_w: int, tile_size: tuple):
    tile_height, tile_width = tile_size
    tile_rows, tile_cols, _, _, _ = tiles.shape
    stitched_image = tiles.swapaxes(1, 2).reshape(tile_rows * tile_height, tile_cols * tile_width, -1)
    if pad_h > 0 or pad_w > 0:
        stitched_image = stitched_image[:-pad_h or None, :-pad_w or None]

    return stitched_image
def extract_number_from_filename(filename):
    match = re.search(r'(\d+)(?=\D*$)', filename)  # Extract last number in filename
    return int(match.group(1)) if match else None

def list_skip_images(pthims):
    image_files = [f for f in os.listdir(pthims) if f.endswith(('tif', 'png', 'jpg'))]

    # Extract numeric values from filenames
    list_z_nums = sorted(filter(None, [extract_number_from_filename(f) for f in image_files]))

    # Identify missing numbers in the sequence
    full_range = set(range(min(list_z_nums), max(list_z_nums) + 1))
    missing_numbers = sorted(full_range - set(list_z_nums))

    # Create input image pairs for each missing number
    skip_inputs = {}
    added_pairs = set()  # To track unique pairs

    for missing in missing_numbers:
        prev_image = max([num for num in list_z_nums if num < missing], default=None)
        next_image = min([num for num in list_z_nums if num > missing], default=None)

        if prev_image is not None and next_image is not None:
            skip_index = next_image - prev_image - 1  # Defines the "skip" level
            pair = (prev_image, next_image)

            if skip_index not in skip_inputs:
                skip_inputs[skip_index] = []

            if pair not in added_pairs:  # Ensure uniqueness
                skip_inputs[skip_index].append([prev_image, next_image])
                added_pairs.add(pair)

    return skip_inputs

def interpolate_from_image_list(pthims, skip_images, TILE_SIZE, model, image_files):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    image_files = natsorted(image_files)  # Sort image files naturally
    file_extension = get_file_extension(image_files[0])  # Get file extension
    image_dict = {extract_number_from_filename(f): f for f in image_files}  # Map numbers to filenames
    image2 = None  # Initialize image2 to None

    for skip_num, image_pairs in skip_images.items():
        print(f'Interpolating for skip {skip_num}:')
        output_folder = os.path.join(pthims, f'int_{skip_num}')
        os.makedirs(output_folder, exist_ok=True)
        times = np.linspace(0, 1, skip_num + 2)[1:-1]

        for img1_num, img2_num in image_pairs:
            image1_path = os.path.join(pthims, image_dict[img1_num])
            image2_path = os.path.join(pthims, image_dict[img2_num])

            # Check if all output images already exist
            all_exist = True
            for idx, time_value in enumerate(times):
                base_name, ext = os.path.splitext(image_dict[img1_num])
                match = re.search(r'(\d+)$', base_name)
                if match:
                    number_part = match.group(1)
                    new_number = str(int(number_part) + (idx + 1)).zfill(len(number_part))
                    filename = f"{base_name[:-len(number_part)]}{new_number}{file_extension}"
                else:
                    filename = f"{base_name}_int{idx + 1}{file_extension}"
                output_path = os.path.join(output_folder, filename)
                if not os.path.exists(output_path):
                    all_exist = False
                    break

            if all_exist:
                print(f'All interpolated images from {image_dict[img1_num]} and {image_dict[img2_num]} already exist. Skipping...')
                continue

            image1 = load_image(image1_path)
            image2 = load_image(image2_path)

            if max(image1.shape[:2]) > TILE_SIZE[0] or max(image2.shape[:2]) > TILE_SIZE[1]:
                print('Stitching needed')
                tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
                tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
                tile_height, tile_width = TILE_SIZE
                num_channels = image1.shape[-1]
                tile_rows, tile_cols, _, _, _ = tiles1.shape
            else:
                print('No stitching needed')
                with tqdm(total=len(times), desc=f"Interpolating {image_dict[img1_num]}", unit="frame") as pbar:
                    for idx, time_value in enumerate(times):
                        base_name, ext = os.path.splitext(image_dict[img1_num])
                        match = re.search(r'(\d+)$', base_name)
                        if match:
                            number_part = match.group(1)
                            new_number = str(int(number_part) + (idx + 1)).zfill(len(number_part))
                            filename = f"{base_name[:-len(number_part)]}{new_number}{file_extension}"
                        else:
                            filename = f"{base_name}_int{idx + 1}{file_extension}"
                        output_path = os.path.join(output_folder, filename)
                        if os.path.exists(output_path):
                            print(f'Already interpolated from {image_dict[img1_num]} and {image_dict[img2_num]}')
                            pbar.update(1)
                            continue
                        time1 = np.array([time_value], dtype=np.float32)
                        input_data = {
                            'time': np.array([time1], dtype=np.float32),
                            'x0': np.expand_dims(image1, axis=0),
                            'x1': np.expand_dims(image2, axis=0)
                        }
                        mid_frame = model(input_data)
                        generated_image = mid_frame['image'][0].numpy()
                        image_in_uint8_range = np.clip(generated_image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
                        image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
                        imageio.imwrite(output_path, image_in_uint8, format=file_extension.lstrip('.'))
                        pbar.update(1)
                continue

            with tqdm(total=len(times), desc=f"Interpolating {image_dict[img1_num]}", unit="frame") as pbar:
                for idx, time_value in enumerate(times):
                    tiles_dict = []
                    for tile_row in range(tile_rows):
                        for tile_col in range(tile_cols):
                            tile1 = tiles1[tile_row, tile_col]
                            tile2 = tiles2[tile_row, tile_col]
                            time1 = np.array([time_value], dtype=np.float32)
                            input_data = {
                                'time': np.array([time1], dtype=np.float32),
                                'x0': np.expand_dims(tile1, axis=0),
                                'x1': np.expand_dims(tile2, axis=0)
                            }
                            mid_frame = model(input_data)
                            generated_tile = mid_frame['image'][0].numpy()
                            image_in_uint8_range = np.clip(generated_tile * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
                            image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
                            tiles_dict.append(image_in_uint8)
                    base_name, ext = os.path.splitext(image_dict[img1_num])
                    match = re.search(r'(\d+)$', base_name)
                    if match:
                        number_part = match.group(1)
                        new_number = str(int(number_part) + (idx + 1)).zfill(len(number_part))
                        stitched_filename = f"{base_name[:-len(number_part)]}{new_number}{file_extension}"
                    else:
                        stitched_filename = f"{base_name}_int{idx + 1}{file_extension}"
                    stitched_output_path = os.path.join(output_folder, stitched_filename)
                    interpolated_tiles_np = np.array(tiles_dict).reshape(tile_rows, tile_cols, tile_height, tile_width, num_channels)
                    stitched_image = stitch_tiles(interpolated_tiles_np, pad_h, pad_w, TILE_SIZE)
                    imageio.imwrite(stitched_output_path, stitched_image, format=file_extension.lstrip('.'))
                    pbar.update(1)

