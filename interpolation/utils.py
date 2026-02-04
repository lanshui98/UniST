"""
Utility functions for image preprocessing before interpolation.

This module provides functions to prepare images for InterpolAI,
including converting single-channel images to multi-channel format
and converting them back to binary volumes.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import tifffile as tiff
import imageio
from natsort import natsorted


def convert_to_3channel(
    image: np.ndarray,
    inplace: bool = False
) -> np.ndarray:
    """
    Convert a single-channel 2D image to 3-channel by duplicating channels.
    
    Parameters
    ----------
    image : np.ndarray
        Input image. Must be:
        - 2D: (H, W) → converted to (H, W, 3)
        - Already 3-channel: (H, W, 3) → returned as-is
    inplace : bool, default=False
        If True, modify the input array in-place (only if possible).
        If False, create a new array.
        Note: In-place conversion is not possible for 2D to 3D conversion.
    
    Returns
    -------
    np.ndarray
        Image with 3 channels. Shape:
        - (H, W, 3) for 2D input
        - Original shape if already 3-channel
    
    Raises
    ------
    ValueError
        If image is not 2D or already 3-channel
    
    Examples
    --------
    >>> img_2d = np.random.rand(100, 100)  # Single channel
    >>> img_3ch = convert_to_3channel(img_2d)  # Shape: (100, 100, 3)
    """
    # Check if already 3-channel
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    
    # 2D image: (H, W) → (H, W, 3)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    
    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}. "
            "Expected 2D image (H, W) or 3-channel image (H, W, 3)."
        )


def convert_directory_to_3channel(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.{tif,tiff}",
    verbose: bool = True
) -> int:
    """
    Convert all single-channel images in a directory to 3-channel format.
    
    This function processes all TIFF images in the input directory,
    converting single-channel images to 3-channel by duplicating channels.
    Images that are already 3-channel are skipped.
    
    Parameters
    ----------
    input_dir : str or Path
        Input directory containing TIFF images
    output_dir : str or Path, optional
        Output directory for converted images. If None, images are saved
        in-place (overwriting originals). Default is None.
    pattern : str, default="*.{tif,tiff}"
        File pattern to match. Supports glob patterns.
    verbose : bool, default=True
        If True, print progress messages.
    
    Returns
    -------
    int
        Number of images successfully converted
    
    Examples
    --------
    >>> # Convert images to a new directory
    >>> convert_directory_to_3channel(
    ...     input_dir="/path/to/input",
    ...     output_dir="/path/to/output"
    ... )
    
    >>> # Convert images in-place (overwrite originals)
    >>> convert_directory_to_3channel(
    ...     input_dir="/path/to/images",
    ...     output_dir=None
    ... )
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    patterns = pattern.split(",") if "," in pattern else [pattern]
    tif_files = []
    for p in patterns:
        tif_files.extend(sorted(input_dir.glob(p.strip())))
    
    if not tif_files:
        if verbose:
            print(f"⚠️  No files found matching pattern '{pattern}' in {input_dir}")
        return 0
    
    converted_count = 0
    skipped_count = 0
    
    for tif_path in tif_files:
        try:
            # Read image
            img = tiff.imread(tif_path)
            
            # Check if already 3-channel
            if img.ndim == 3 and img.shape[-1] == 3:
                if verbose:
                    print(f"⏭️  Already 3-channel, skip: {tif_path.name}")
                skipped_count += 1
                # If output_dir is specified, copy the file
                if output_dir is not None:
                    out_path = output_dir / tif_path.name
                    tiff.imwrite(out_path, img)
                continue
            
            # Skip 3D volumes (not supported)
            if img.ndim == 3 and img.shape[-1] != 3:
                if verbose:
                    print(f"⚠️  Unsupported 3D volume shape {img.shape}, skip: {tif_path.name}")
                skipped_count += 1
                continue
            
            # Convert to 3-channel
            img_3ch = convert_to_3channel(img)
            
            # Determine output path
            if output_dir is not None:
                out_path = output_dir / tif_path.name
            else:
                out_path = tif_path  # Overwrite original
            
            # Save converted image
            tiff.imwrite(out_path, img_3ch)
            
            if verbose:
                print(f"✅ Converted: {tif_path.name} → {img_3ch.shape}")
            converted_count += 1
            
        except Exception as e:
            if verbose:
                print(f"❌ Error processing {tif_path.name}: {e}")
            continue
    
    if verbose:
        print(f"\n📊 Summary: {converted_count} converted, {skipped_count} skipped")
    
    return converted_count


def _extract_num(fname: str) -> Optional[int]:
    """Extract numeric index from filename."""
    m = re.search(r'(\d+)(?=\D*$)', fname)
    return int(m.group(1)) if m else None


def _to_binary(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Convert image to binary (0 or 255).
    
    Parameters
    ----------
    img : np.ndarray
        Input image, shape: (H, W) or (H, W, 3) or (H, W, 4)
    threshold : int, default=127
        Threshold for binarization
    
    Returns
    -------
    np.ndarray
        Binary image as uint8 with values {0, 255}
    """
    if img.ndim == 3:
        # Ignore alpha channel, take mean of first 3 channels
        img = img[..., :3].mean(axis=2)
    
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    binmap = (img > threshold).astype(np.uint8) * 255
    return binmap


def _read_img(path: Union[str, Path]) -> np.ndarray:
    """
    Read image and normalize to uint8.
    
    Parameters
    ----------
    path : str or Path
        Path to image file
    
    Returns
    -------
    np.ndarray
        Image as uint8 array
    """
    arr = imageio.imread(path)
    # If float (0-1), map to 0-255
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def collect_interpolated_maps(
    interp_root: Union[str, Path],
    exts: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
) -> Dict[int, Path]:
    """
    Collect all interpolated images from int_* subdirectories.
    
    Parameters
    ----------
    interp_root : str or Path
        Root directory containing interpolation subdirectories (int_1, int_2, etc.)
    exts : tuple of str, default=('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        File extensions to match
    
    Returns
    -------
    dict
        Mapping from slice index to file path
    """
    idx2path = {}
    interp_root = Path(interp_root)
    
    if not interp_root.is_dir():
        return idx2path
    
    # Find all int_* subdirectories
    subdirs = [
        d for d in interp_root.iterdir()
        if d.is_dir() and d.name.startswith('int_')
    ]
    subdirs = natsorted(subdirs, key=lambda x: x.name)
    
    for sd in subdirs:
        for f in natsorted(sd.iterdir()):
            if not f.suffix.lower().lstrip('.') in [ext.lstrip('.') for ext in exts]:
                continue
            idx = _extract_num(f.name)
            if idx is not None:
                # Use setdefault to keep first occurrence
                idx2path.setdefault(idx, f)
    
    return idx2path


def merge_to_volume(
    base_dir: Union[str, Path],
    interp_dir: Union[str, Path],
    out_binary_dir: Union[str, Path] = "binary_slices",
    out_stack_path: Union[str, Path] = "volume.tif",
    threshold: int = 127,
    exts: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
    bigtiff: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Merge base slices and interpolated slices into a binary volume.
    
    This function:
    1. Collects base slices from base_dir
    2. Collects interpolated slices from int_* subdirectories in interp_dir
    3. Converts all images to binary (0/255)
    4. Stacks them into a 3D volume (Z, H, W)
    5. Saves individual binary slices and the volume stack
    
    Parameters
    ----------
    base_dir : str or Path
        Directory containing original base slices
    interp_dir : str or Path
        Directory containing interpolation subdirectories (int_1, int_2, etc.)
    out_binary_dir : str or Path, default="binary_slices"
        Output directory for individual binary slices
    out_stack_path : str or Path, default="volume.tif"
        Output path for the volume stack
    threshold : int, default=127
        Threshold for binarization (0-255)
    exts : tuple of str, default=('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        File extensions to match
    bigtiff : bool, default=True
        Use BigTIFF format for large volumes
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    np.ndarray
        Volume stack as uint8 array, shape (Z, H, W) with values {0, 255}
    
    Raises
    ------
    RuntimeError
        If no slices are found
    ValueError
        If slice dimensions don't match
    """
    base_dir = Path(base_dir)
    interp_dir = Path(interp_dir)
    out_binary_dir = Path(out_binary_dir)
    out_stack_path = Path(out_stack_path)
    
    # Create output directory
    out_binary_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Collect base slices
    base_files = [
        f for f in base_dir.iterdir()
        if f.suffix.lower().lstrip('.') in [ext.lstrip('.') for ext in exts]
    ]
    base_files = natsorted(base_files, key=lambda x: _extract_num(x.name))
    base_idx2path = {}
    for f in base_files:
        idx = _extract_num(f.name)
        if idx is not None:
            base_idx2path[idx] = f
    
    if not base_idx2path:
        raise RuntimeError(f"No base slices found in {base_dir}")
    
    # 2) Collect interpolated slices
    interp_idx2path = collect_interpolated_maps(interp_dir, exts=exts)
    
    # 3) Build complete index range
    all_indices = sorted(set(base_idx2path.keys()) | set(interp_idx2path.keys()))
    if not all_indices:
        raise RuntimeError("No slices (base or interpolated) found.")
    
    if verbose:
        print(f"Found {len(base_idx2path)} base slices and {len(interp_idx2path)} interpolated slices")
        print(f"Total unique indices: {len(all_indices)} (min: {min(all_indices)}, max: {max(all_indices)})")
    
    # 4) Process each slice
    volume_slices = []
    H_ref, W_ref = None, None
    
    for idx in all_indices:
        if idx in base_idx2path:
            src = base_idx2path[idx]
            src_type = "base"
        elif idx in interp_idx2path:
            src = interp_idx2path[idx]
            src_type = "interp"
        else:
            continue
        
        # Read and convert to binary
        img = _read_img(src)
        binmap = _to_binary(img, threshold=threshold)
        
        # Check dimensions
        if H_ref is None:
            H_ref, W_ref = binmap.shape
        else:
            if binmap.shape != (H_ref, W_ref):
                raise ValueError(
                    f"Slice size mismatch at index {idx}: "
                    f"got {binmap.shape}, expected {(H_ref, W_ref)}"
                )
        
        # Save individual binary slice
        out_name = f"slice_{idx:03d}.tif"
        out_path = out_binary_dir / out_name
        imageio.imwrite(out_path, binmap.astype(np.uint8))
        
        volume_slices.append(binmap)
        
        if verbose and len(volume_slices) % 10 == 0:
            print(f"Processed {len(volume_slices)}/{len(all_indices)} slices...")
    
    # 5) Stack into volume
    volume = np.stack(volume_slices, axis=0).astype(np.uint8)  # (Z, H, W)
    
    # Save volume stack
    tiff.imwrite(
        out_stack_path,
        volume,
        bigtiff=bigtiff,
        photometric='minisblack'
    )
    
    if verbose:
        print(f"\n Saved binary slices to: {out_binary_dir.absolute()}")
        print(f" Saved volume to: {out_stack_path.absolute()}")
        print(f"   Volume shape: {volume.shape}, dtype: {volume.dtype}")
        print(f"   Base indices (sample): {sorted(list(base_idx2path.keys()))[:10]}")
        print(f"   Interp indices (sample): {sorted(list(interp_idx2path.keys()))[:10]}")
        print(f"   All indices range: {min(all_indices)} - {max(all_indices)}")
    
    return volume
