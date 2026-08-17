"""Call package InterpolAI to synthesize a middle occupancy frame."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .occupancy import to_3channel
from .paths import interpolai_dir, require_interpolai_model


def _ensure_interpolai_on_path():
    import sys

    root = interpolai_dir().parent  # external/
    # Prefer importing via external.InterpolAI...
    proj = interpolai_dir().parents[1]
    for p in (str(proj), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_interpolai_model():
    require_interpolai_model()
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "InterpolAI requires TensorFlow. Install with: pip install 'unist[full]' "
            "or pip install tensorflow>=2.16"
        ) from e
    model_path = require_interpolai_model()
    return tf.saved_model.load(str(model_path))


def interpolate_middle_frame(
    img0: np.ndarray,
    img1: np.ndarray,
    *,
    model=None,
    time: float = 0.5,
    tile_size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """
    Run InterpolAI between two 3-channel images; return uint8 (H,W,3) at ``time``.

    Uses the same tiling logic as ``external/InterpolAI``.
    """
    _ensure_interpolai_on_path()
    from external.InterpolAI.interpolation.interpolation_function_auto import (
        pad_and_tile_image,
        stitch_tiles,
    )

    if model is None:
        model = load_interpolai_model()

    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    x0 = to_3channel(img0).astype(np.float32) / _UINT8_MAX_F
    x1 = to_3channel(img1).astype(np.float32) / _UINT8_MAX_F
    if x0.shape != x1.shape:
        raise ValueError(f"Image shapes must match, got {x0.shape} vs {x1.shape}")

    time_arr = np.array([[time]], dtype=np.float32)

    if max(x0.shape[:2]) > tile_size[0] or max(x0.shape[:2]) > tile_size[1]:
        tiles0, (pad_h, pad_w) = pad_and_tile_image(x0, tile_size)
        tiles1, _ = pad_and_tile_image(x1, tile_size)
        tile_h, tile_w = tile_size
        n_ch = x0.shape[-1]
        rows, cols = tiles0.shape[:2]
        out_tiles = []
        for r in range(rows):
            for c in range(cols):
                inp = {
                    "time": time_arr,
                    "x0": np.expand_dims(tiles0[r, c], 0),
                    "x1": np.expand_dims(tiles1[r, c], 0),
                }
                gen = model(inp)["image"][0].numpy()
                u8 = (np.clip(gen * _UINT8_MAX_F, 0, _UINT8_MAX_F) + 0.5).astype(np.uint8)
                out_tiles.append(u8)
        arr = np.array(out_tiles).reshape(rows, cols, tile_h, tile_w, n_ch)
        return stitch_tiles(arr, pad_h, pad_w, tile_size)

    inp = {
        "time": time_arr,
        "x0": np.expand_dims(x0, 0),
        "x1": np.expand_dims(x1, 0),
    }
    gen = model(inp)["image"][0].numpy()
    return (np.clip(gen * _UINT8_MAX_F, 0, _UINT8_MAX_F) + 0.5).astype(np.uint8)
