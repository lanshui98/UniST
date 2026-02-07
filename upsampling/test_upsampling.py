import os
import sys
from pathlib import Path
from glob import glob

# Add RepKPU_ops to path
_possible_paths = [
    Path(__file__).parent.parent / 'external' / 'RepKPU_ops',
    Path.cwd() / 'external' / 'RepKPU_ops',
    Path.cwd() / 'UniST' / 'external' / 'RepKPU_ops',
]

for path in _possible_paths:
    if path.exists():
        sys.path.insert(0, str(path.resolve()))
        break

import torch
import argparse
from models.repkpu import RepKPU, RepKPU_o
from cfgs.upsampling import parse_pu1k_args, parse_pugan_o_args, parse_pugan_args
from cfgs.utils import reset_model_args

# Import and patch test_flexible to support recursive search
sys.path.insert(0, str(Path(__file__).parent.parent / 'external' / 'RepKPU_ops'))
from test import test_flexible as _original_test_flexible

def test_flexible(model, args):
    """Wrapper that adds recursive file search and debug info."""
    # Debug: print input directory info
    print(f"\n{'='*60}")
    print(f"Input Directory: {args.input_dir}")
    print(f"Directory exists: {os.path.exists(args.input_dir)}")
    print(f"Directory is absolute: {os.path.isabs(args.input_dir)}")
    
    if os.path.exists(args.input_dir):
        # List directory contents
        try:
            dir_contents = os.listdir(args.input_dir)
            print(f"Directory contents ({len(dir_contents)} items):")
            for item in dir_contents[:10]:  # Show first 10 items
                item_path = os.path.join(args.input_dir, item)
                item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                print(f"  [{item_type}] {item}")
            if len(dir_contents) > 10:
                print(f"  ... and {len(dir_contents) - 10} more items")
        except Exception as e:
            print(f"Error listing directory: {e}")
        
        # Try to find .xyz files
        print(f"\nSearching for .xyz files...")
        # Direct search
        xyz_files = glob(os.path.join(args.input_dir, '*.xyz'))
        print(f"  Direct search: found {len(xyz_files)} files")
        
        # Recursive search
        if len(xyz_files) == 0:
            xyz_files = glob(os.path.join(args.input_dir, '**', '*.xyz'), recursive=True)
            print(f"  Recursive search: found {len(xyz_files)} files")
            if len(xyz_files) > 0:
                print(f"  Sample files found:")
                for f in xyz_files[:5]:
                    print(f"    {f}")
        
        if len(xyz_files) == 0:
            print(f"\n⚠️  WARNING: No .xyz files found in {args.input_dir}")
            print(f"   Please check:")
            print(f"   1. The directory path is correct")
            print(f"   2. Files have .xyz extension")
            print(f"   3. Files are readable")
    else:
        print(f"❌ ERROR: Directory does not exist: {args.input_dir}")
    
    print(f"{'='*60}\n")
    
    # Monkey patch glob in test module to support recursive search
    import test as test_module
    original_glob = glob
    
    def patched_glob(pattern, recursive=False):
        result = original_glob(pattern, recursive=recursive)
        if len(result) == 0 and '*.xyz' in pattern and not recursive:
            # Try recursive search
            recursive_pattern = pattern.replace('*.xyz', '**/*.xyz')
            result = original_glob(recursive_pattern, recursive=True)
        return result
    
    # Temporarily replace glob
    test_module.glob = patched_glob
    
    try:
        # Call original function
        return _original_test_flexible(model, args)
    finally:
        # Restore original glob
        test_module.glob = original_glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pu1k', type=str)
    parser.add_argument('--r', default=4, type=float)
    parser.add_argument('--o', action='store_true')
    parser.add_argument('--flexible', action='store_true')
    parser.add_argument('--input_dir', default='./output', type=str)
    parser.add_argument('--gt_dir', default='./output', type=str)
    parser.add_argument('--save_dir', default='pcd', type=str)
    parser.add_argument('--ckpt', default='external/RepKPU_ops/pretrain/ckpt-best.pth', type=str)
    parser.add_argument('--no_gt', action='store_true')
    args = parser.parse_args()
    
    # Normalize paths (handle both relative and absolute)
    if not os.path.isabs(args.input_dir):
        args.input_dir = str(Path(__file__).parent.parent / args.input_dir)
    if not os.path.isabs(args.gt_dir):
        args.gt_dir = str(Path(__file__).parent.parent / args.gt_dir)
    if not os.path.isabs(args.save_dir):
        args.save_dir = str(Path(__file__).parent.parent / args.save_dir)
    if not os.path.isabs(args.ckpt):
        args.ckpt = str(Path(__file__).parent.parent / args.ckpt)
    
    if args.dataset == 'pugan':
        if args.o:
            reset_model_args(parse_pugan_o_args(), args)
            model = RepKPU_o(args)
        else:
            reset_model_args(parse_pugan_args(), args)
            model = RepKPU(args)
    else:
        reset_model_args(parse_pu1k_args(), args)
        model = RepKPU(args)
    
    model = model.cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))
    
    test_flexible(model, args)
