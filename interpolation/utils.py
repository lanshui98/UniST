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
    # Handle patterns like "*.{tif,tiff}" by expanding to multiple patterns
    if "{tif,tiff}" in pattern or "{tif,tiff}" in pattern:
        patterns = [
            pattern.replace("{tif,tiff}", "tif"),
            pattern.replace("{tif,tiff}", "tiff")
        ]
    elif "," in pattern:
        patterns = [p.strip() for p in pattern.split(",")]
    else:
        patterns = [pattern]
    
    tif_files = []
    for p in patterns:
        tif_files.extend(sorted(input_dir.glob(p.strip())))
    
    # Remove duplicates while preserving order
    seen = set()
    tif_files = [f for f in tif_files if not (f in seen or seen.add(f))]
    
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


def _extract_interp_slice_num(fname: str) -> Optional[int]:
    """
    Extract slice index from interpolated filename.

    For names like `xxxx_int1.tif`, the trailing `int1` describes interpolation
    step, not slice index. This function first removes trailing `_int<k>` and then
    extracts the numeric slice index from the remaining stem.
    """
    stem = Path(fname).stem
    # Remove trailing interpolation suffix such as "_int1" or "_int_1"
    stem_wo_int = re.sub(r'[_-]?int[_-]?\d+$', '', stem, flags=re.IGNORECASE)
    idx = _extract_num(stem_wo_int)
    if idx is not None:
        return idx
    # Fallback for uncommon names that only contain one numeric token.
    return _extract_num(stem)


def _parse_z_bin_int(filename: str) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Parse z value, bin index, and interpolation index from filename.

    Expected style:
      occupancy_z1043.48_bin34_gray_int1.tif
      occupancy_z1043.48_bin34_gray.tif  (int defaults to 0)
    """
    match = re.search(r"z([\d.]+)_bin(\d+).*?(?:_int(\d+))?", filename)
    if match:
        z_val = float(match.group(1))
        bin_val = int(match.group(2))
        int_val = int(match.group(3)) if match.group(3) else 0
        return z_val, bin_val, int_val
    return None, None, None


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
    exts: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
    verbose: bool = False,
) -> Dict[int, Path]:
    """
    Collect all interpolated images from int_* subdirectories.
    
    Parameters
    ----------
    interp_root : str or Path
        Root directory containing interpolation subdirectories (int_1, int_2, etc.)
    exts : tuple of str, default=('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        File extensions to match
    verbose : bool, default=False
        If True, print per-directory collection statistics
    
    Returns
    -------
    dict
        Mapping from slice index to file path
    """
    idx2path = {}
    interp_root = Path(interp_root)
    
    if not interp_root.is_dir():
        return idx2path
    
    # Find interpolation subdirectories named int_*
    subdirs = [
        d for d in interp_root.iterdir()
        if d.is_dir() and d.name.startswith("int_")
    ]
    subdirs = natsorted(subdirs, key=lambda x: x.name)

    if verbose:
        print(f"Found {len(subdirs)} interpolation subdirectories under: {interp_root}")
    
    for sd in subdirs:
        total_files = 0
        matched_ext = 0
        parsed_with_bin = 0
        parsed_with_fallback = 0
        kept = 0

        # Recursively find all image files inside each int_* directory
        files = natsorted(
            [f for f in sd.rglob("*") if f.is_file()],
            key=lambda p: p.name
        )
        for f in files:
            total_files += 1
            if f.suffix.lower().lstrip('.') not in [ext.lstrip('.') for ext in exts]:
                continue
            matched_ext += 1
            # Prefer bin index for occupancy_z..._bin... naming
            _, bin_idx, _ = _parse_z_bin_int(f.name)
            if bin_idx is not None:
                idx = bin_idx
                parsed_with_bin += 1
            else:
                idx = _extract_interp_slice_num(f.name)
                if idx is not None:
                    parsed_with_fallback += 1
            if idx is not None:
                # Use setdefault to keep first occurrence
                prev = idx2path.setdefault(idx, f)
                if prev == f:
                    kept += 1

        if verbose:
            print(
                f"  {sd.name}: files={total_files}, ext_matched={matched_ext}, "
                f"parsed(bin/fallback)={parsed_with_bin}/{parsed_with_fallback}, kept={kept}"
            )

    if verbose:
        print(f"Collected {len(idx2path)} unique interpolated slice indices")
    
    return idx2path


def collect_interpolated_paths(
    interp_root: Union[str, Path],
    exts: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
    verbose: bool = False,
) -> list:
    """
    Collect all interpolated image paths from int_* subdirectories (no dedup).

    Paths are sorted by (z, bin, int) when the filename matches z/bin format;
    otherwise they are sorted by fallback numeric index.
    """
    interp_root = Path(interp_root)
    if not interp_root.is_dir():
        return []

    subdirs = [d for d in interp_root.iterdir() if d.is_dir() and d.name.startswith("int_")]
    subdirs = natsorted(subdirs, key=lambda x: x.name)
    ext_set = {ext.lower().lstrip('.') for ext in exts}

    if verbose:
        print(f"Found {len(subdirs)} interpolation subdirectories under: {interp_root}")

    collected = []
    for sd in subdirs:
        total_files = 0
        matched_ext = 0
        parsed_with_zbin = 0
        parsed_with_fallback = 0
        kept = 0

        files = natsorted([f for f in sd.rglob("*") if f.is_file()], key=lambda p: p.name)
        for f in files:
            total_files += 1
            if f.suffix.lower().lstrip('.') not in ext_set:
                continue
            matched_ext += 1

            z, bin_idx, int_idx = _parse_z_bin_int(f.name)
            if z is not None and bin_idx is not None:
                sort_key = (0, z, bin_idx, int_idx, f.name)
                parsed_with_zbin += 1
                kept += 1
                collected.append((sort_key, f))
                continue

            idx = _extract_interp_slice_num(f.name)
            if idx is not None:
                sort_key = (1, idx, 0, 0, f.name)
                parsed_with_fallback += 1
                kept += 1
                collected.append((sort_key, f))

        if verbose:
            print(
                f"  {sd.name}: files={total_files}, ext_matched={matched_ext}, "
                f"parsed(zbin/fallback)={parsed_with_zbin}/{parsed_with_fallback}, kept={kept}"
            )

    collected = sorted(collected, key=lambda x: x[0])
    ordered_paths = [p for _, p in collected]

    if verbose:
        print(f"Collected {len(ordered_paths)} interpolated slices (no dedup)")

    return ordered_paths


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
    ext_set = {ext.lower().lstrip('.') for ext in exts}
    base_records = []
    for f in base_dir.iterdir():
        if not f.is_file() or f.suffix.lower().lstrip('.') not in ext_set:
            continue

        z, bin_idx, _ = _parse_z_bin_int(f.name)
        if z is not None and bin_idx is not None:
            sort_key = (0, z, bin_idx, 0, f.name)
        else:
            idx = _extract_num(f.name)
            if idx is None:
                continue
            sort_key = (1, idx, 0, 0, f.name)
        base_records.append((sort_key, "base", f))

    base_records = sorted(base_records, key=lambda x: x[0])
    if not base_records:
        raise RuntimeError(f"No base slices found in {base_dir}")

    # 2) Collect interpolated slices (keep all)
    interp_paths = collect_interpolated_paths(interp_dir, exts=exts, verbose=verbose)
    interp_records = []
    for f in interp_paths:
        z, bin_idx, int_idx = _parse_z_bin_int(f.name)
        if z is not None and bin_idx is not None:
            sort_key = (0, z, bin_idx, int_idx, f.name)
        else:
            idx = _extract_interp_slice_num(f.name)
            if idx is None:
                continue
            sort_key = (1, idx, 0, 0, f.name)
        interp_records.append((sort_key, "interp", f))

    all_records = sorted(base_records + interp_records, key=lambda x: x[0])
    if not all_records:
        raise RuntimeError("No slices (base or interpolated) found.")

    if verbose:
        print(
            f"Found {len(base_records)} base slices and "
            f"{len(interp_records)} interpolated slices"
        )
        print(f"Total merged slices: {len(all_records)}")

    # 3) Process each slice in sorted order
    volume_slices = []
    H_ref, W_ref = None, None

    for merged_idx, (_, src_type, src) in enumerate(all_records, start=1):
        # Read and convert to binary
        img = _read_img(src)
        binmap = _to_binary(img, threshold=threshold)

        # Check dimensions
        if H_ref is None:
            H_ref, W_ref = binmap.shape
        else:
            if binmap.shape != (H_ref, W_ref):
                raise ValueError(
                    f"Slice size mismatch at merged index {merged_idx}: "
                    f"got {binmap.shape}, expected {(H_ref, W_ref)}"
                )

        # Save individual binary slice by merged order (keeps all int slices)
        out_name = f"slice_{merged_idx:03d}.tif"
        out_path = out_binary_dir / out_name
        imageio.imwrite(out_path, binmap.astype(np.uint8))

        volume_slices.append(binmap)

        if verbose and len(volume_slices) % 10 == 0:
            print(f"Processed {len(volume_slices)}/{len(all_records)} slices...")

    # 4) Stack into volume
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
        print(f"   Base slices: {len(base_records)}")
        print(f"   Interpolated slices: {len(interp_records)}")
        print(f"   Merged slices: {len(all_records)}")
    
    return volume
