"""
This script converts 3-channel interpolated images back to single-channel
binary format and merges them with base slices into a 3D volume.
"""

import argparse
import sys
from pathlib import Path

from .utils import merge_to_volume


def main():
    """Main entry point for the volume merging CLI."""
    parser = argparse.ArgumentParser(
        description="Merge base and interpolated slices into a binary volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m interpolation.merge_volume \\
      --base_dir /path/to/base_slices \\
      --interp_dir /path/to/interpolation_output \\
      --out_stack_path volume.tif

  # With custom output directories
  python -m interpolation.merge_volume \\
      --base_dir /path/to/base_slices \\
      --interp_dir /path/to/interpolation_output \\
      --out_binary_dir binary_slices \\
      --out_stack_path volume.tif \\
      --threshold 127
        """
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing original base slices"
    )
    parser.add_argument(
        "--interp_dir",
        type=str,
        required=True,
        help="Directory containing interpolation subdirectories (int_1, int_2, etc.)"
    )
    parser.add_argument(
        "--out_binary_dir",
        type=str,
        default="binary_slices",
        help="Output directory for individual binary slices (default: 'binary_slices')"
    )
    parser.add_argument(
        "--out_stack_path",
        type=str,
        default="volume.tif",
        help="Output path for the volume stack (default: 'volume.tif')"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Threshold for binarization (0-255, default: 127)"
    )
    parser.add_argument(
        "--exts",
        type=str,
        nargs='+',
        default=['.tif', '.tiff', '.png', '.jpg', '.jpeg'],
        help="File extensions to match (default: .tif .tiff .png .jpg .jpeg)"
    )
    parser.add_argument(
        "--no_bigtiff",
        action="store_true",
        help="Disable BigTIFF format (use for smaller volumes)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    interp_dir = Path(args.interp_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not interp_dir.exists():
        print(f"Error: Interpolation directory does not exist: {interp_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        volume = merge_to_volume(
            base_dir=base_dir,
            interp_dir=interp_dir,
            out_binary_dir=args.out_binary_dir,
            out_stack_path=args.out_stack_path,
            threshold=args.threshold,
            exts=tuple(args.exts),
            bigtiff=not args.no_bigtiff,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n✅ Successfully created volume with shape {volume.shape}")
        sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
