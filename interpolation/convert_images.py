"""
This script prepares images for InterpolAI by converting single-channel
images to 3-channel format through channel duplication.
"""

import argparse
import sys
from pathlib import Path

from .utils import convert_directory_to_3channel


def main():
    """Main entry point for the image conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert single-channel images to 3-channel format for InterpolAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert images to a new directory
  python -m interpolation.convert_images \\
      --input_dir /path/to/input \\
      --output_dir /path/to/output

  # Convert images in-place (overwrite originals)
  python -m interpolation.convert_images \\
      --input_dir /path/to/images \\
      --inplace

  # Convert with custom file pattern
  python -m interpolation.convert_images \\
      --input_dir /path/to/input \\
      --output_dir /path/to/output \\
      --pattern "*.tif"
        """
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing TIFF images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for converted images. If not specified and --inplace is not set, "
             "images will be saved to input_dir with '_3ch' suffix."
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original images in-place (use with caution!)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="File pattern to match (default: '*.tif'). Use comma-separated for multiple: '*.tif,*.tiff'"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if args.inplace:
        output_dir = None
        if not args.quiet:
            print("⚠️  Warning: Converting images in-place (original files will be overwritten)")
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: create output directory with '_3ch' suffix
        output_dir = input_dir.parent / f"{input_dir.name}_3ch"
        if not args.quiet:
            print(f"Output directory not specified. Using: {output_dir}")
    
    try:
        count = convert_directory_to_3channel(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=args.pattern,
            verbose=not args.quiet
        )
        
        if count == 0:
            if not args.quiet:
                print("No images were converted.")
            sys.exit(1)
        else:
            if not args.quiet:
                print(f"\n✅ Successfully converted {count} image(s)")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
