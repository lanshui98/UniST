#!/usr/bin/env python
"""
Setup script for building CUDA extensions required by RepKPU_ops.

This script builds the Chamfer3D and pointops CUDA extensions needed for
point cloud upsampling functionality.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.absolute()


def build_chamfer3d(project_root):
    """Build Chamfer3D CUDA extension."""
    chamfer3d_dir = project_root / "external" / "RepKPU_ops" / "models" / "Chamfer3D"
    
    if not chamfer3d_dir.exists():
        raise FileNotFoundError(
            f"Chamfer3D directory not found at {chamfer3d_dir}\n"
            "Please ensure RepKPU_ops code is in external/RepKPU_ops/ directory."
        )
    
    print("=" * 60)
    print("Building Chamfer3D CUDA extension...")
    print("=" * 60)
    
    os.chdir(chamfer3d_dir)
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "install"],
            check=True,
            capture_output=False
        )
        print("✓ Chamfer3D extension built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build Chamfer3D extension: {e}")
        return False
    finally:
        os.chdir(project_root)


def build_pointops(project_root):
    """Build pointops CUDA extension."""
    pointops_dir = project_root / "external" / "RepKPU_ops" / "models" / "pointops"
    
    if not pointops_dir.exists():
        raise FileNotFoundError(
            f"pointops directory not found at {pointops_dir}\n"
            "Please ensure RepKPU_ops code is in external/RepKPU_ops/ directory."
        )
    
    print("=" * 60)
    print("Building pointops CUDA extension...")
    print("=" * 60)
    
    os.chdir(pointops_dir)
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "install"],
            check=True,
            capture_output=False
        )
        print("✓ pointops extension built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build pointops extension: {e}")
        return False
    finally:
        os.chdir(project_root)


def main():
    """Main entry point."""
    project_root = get_project_root()
    
    print("=" * 60)
    print("UniST CUDA Extensions Setup")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print()
    
    # Check CUDA architecture
    cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_arch:
        print(f"Using CUDA architecture: {cuda_arch}")
    else:
        print("Warning: TORCH_CUDA_ARCH_LIST not set.")
        print("Defaulting to architecture 8.0 (A100, RTX 3090, etc.)")
        print("For other GPUs, set TORCH_CUDA_ARCH_LIST before running:")
        print("  export TORCH_CUDA_ARCH_LIST=\"7.0\"  # V100")
        print("  export TORCH_CUDA_ARCH_LIST=\"7.5\"  # RTX 2080, Titan RTX")
        print("  export TORCH_CUDA_ARCH_LIST=\"8.0\"  # A100, RTX 3090")
        print("  export TORCH_CUDA_ARCH_LIST=\"8.6\"  # RTX 3090, A6000")
        print()
    
    # Build extensions
    success = True
    
    if not build_chamfer3d(project_root):
        success = False
    
    print()
    
    if not build_pointops(project_root):
        success = False
    
    print()
    print("=" * 60)
    if success:
        print("✓ All CUDA extensions built successfully!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some CUDA extensions failed to build.")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Ensure CUDA toolkit is installed")
        print("2. Ensure PyTorch with CUDA support is installed")
        print("3. Ensure ninja is installed: pip install ninja")
        print("4. Check that TORCH_CUDA_ARCH_LIST matches your GPU")
        return 1


if __name__ == "__main__":
    sys.exit(main())
