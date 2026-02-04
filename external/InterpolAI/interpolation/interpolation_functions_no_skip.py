import numpy as np
import os
import tensorflow as tf
import imageio
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import time

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

def interpolate_from_image_stack_no_skip(pthims, skips, TILE_SIZE, model, image_files=None):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    image_files = [f for f in os.listdir(pthims) if f.endswith(('tif', 'png', 'jpg'))]
    image_files = natsorted(image_files)
    file_extension = get_file_extension(image_files[0])
    image2 = None

    for skip_num in skips:
        print(f'Interpolating int {skip_num}:')
        output_folder = os.path.join(pthims, f'int_{skip_num}_no_skip')
        os.makedirs(output_folder, exist_ok=True)
        times = np.linspace(0, 1, skip_num + 2)[1:-1]

        for i in range(len(image_files) - 1):
            image1_path = os.path.join(pthims, image_files[i])
            image2_path = os.path.join(pthims, image_files[i + 1])

            output_checkpath = os.path.join(output_folder, f"{os.path.splitext(image_files[i])[0]}_int{skip_num}{file_extension}")
            if os.path.exists(output_checkpath):
                print(f'already interpolated from input images {image_files[i]} and {image_files[i + 1]}')
                continue

            if image2 is not None:
                print('loading image 1 from image 2')
                image1 = image2
            else:
                image1 = load_image(image1_path)

            image2 = load_image(image2_path)

            if max(image1.shape[:2]) > TILE_SIZE[0] or max(image2.shape[:2]) > TILE_SIZE[1]:
                print('stitching')
                tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
                tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
                tile_height, tile_width = TILE_SIZE
                num_channels = image1.shape[-1]
                tile_rows, tile_cols, _, _, _ = tiles1.shape
            else:
                print('not stitching')
                with tqdm(total=len(times), desc=f"Interpolating {image_files[i]}", unit="frame") as pbar:
                    for idx, time_value in enumerate(times):
                        start_time = time.time()
                        filename = f"{os.path.splitext(image_files[i])[0]}_int{idx + 1}{file_extension}"
                        output_path = os.path.join(output_folder, filename)

                        if os.path.exists(output_path):
                            print(f'already interpolated from input images {image_files[i]} and {image_files[i + 1]}')
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

                        elapsed_time = time.time() - start_time
                        print(f"Time to generate {filename}: {elapsed_time:.2f} seconds")
                        pbar.update(1)
                continue

            with tqdm(total=len(times), desc=f"Interpolating {image_files[i]}", unit="frame") as pbar:
                for idx, time_value in enumerate(times):
                    start_time = time.time()
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

                    stitched_filename = f"{os.path.splitext(image_files[i])[0]}_int{idx + 1}{file_extension}"
                    stitched_output_path = os.path.join(output_folder, stitched_filename)

                    interpolated_tiles_np = np.array(tiles_dict).reshape(tile_rows, tile_cols, tile_height, tile_width, num_channels)
                    stitched_image = stitch_tiles(interpolated_tiles_np, pad_h, pad_w, TILE_SIZE)

                    # imageio.imwrite(stitched_output_path, stitched_image, format=file_extension.lstrip('.'),
                    #                 jpg=True)

                    imageio.imwrite(stitched_output_path, stitched_image, format=file_extension.lstrip('.'))

                    elapsed_time = time.time() - start_time
                    print(f"Time to generate {stitched_filename}: {elapsed_time:.2f} seconds")
                    pbar.update(1)