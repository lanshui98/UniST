"""
Test script for RepKPU upsampling.

This script provides a command-line interface that uses the external RepKPU_ops code.
"""

import os
import sys
import argparse
import torch
from glob import glob
import open3d as o3d
import numpy as np
from einops import rearrange

# Add external/RepKPU_ops to Python path
_package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_external_repkpu_path = os.path.join(_package_root, 'external', 'RepKPU_ops')
if os.path.exists(_external_repkpu_path) and _external_repkpu_path not in sys.path:
    sys.path.insert(0, _external_repkpu_path)

# Import from external RepKPU_ops package
try:
    from models.repkpu import RepKPU, RepKPU_o
    from cfgs.upsampling import parse_pu1k_args, parse_pugan_o_args, parse_pugan_args
    from cfgs.utils import reset_model_args
    from models.utils import (
        normalize_point_cloud, FPS, extract_knn_patch, 
        chamfer_sqrt
    )
    from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
    chamfer_dist = chamfer_3DDist()
except ImportError as e:
    print(f"Error: Could not import from RepKPU_ops: {e}")
    print(f"\nExpected path: {_external_repkpu_path}")
    print("Please ensure RepKPU_ops code is in external/RepKPU_ops/ directory.")
    sys.exit(1)


def upsampling(args, model, input_pcd):
    """Upsample point cloud using the model."""
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = args.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    coarse_pts, _ = model.forward(patches)
    coarse_pts = coarse_pts
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1] * args.up_rate)
    return coarse_pts


def test_flexible(model, args):
    """Test with flexible upsampling rate."""
    print("=== Starting Flexible Scale Upsampling Test ===")
    with torch.no_grad():
        model.eval()
        test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
        
        print(f"Found {len(test_input_path)} files")
        
        total_cd = 0
        counter = 0
        txt_result = []

        for i, path in enumerate(test_input_path):
            pcd_name = os.path.basename(path)
            print(f"\n{'='*50}")
            print(f"Processing file {i+1}/{len(test_input_path)}: {pcd_name}")
            print(f"{'='*50}")

            try:
                # Read point cloud
                pcd = o3d.io.read_point_cloud(path)
                input_points = np.asarray(pcd.points)
                print(f"  ✓ Successfully read {input_points.shape[0]} points")

                if input_points.shape[0] == 0:
                    print("  ❌ Error: Point cloud is empty")
                    txt_result.append(f'{pcd_name}: Point cloud is empty')
                    continue

                # Calculate target point count
                input_num = input_points.shape[0]
                target_num = int(input_num * args.r)
                print(f"  📊 Input points: {input_num}")
                print(f"  📈 Upsampling rate: {args.r}")
                print(f"  🎯 Target points: {target_num}")

                # Load GT if available
                gt = None
                if not args.no_gt:
                    try:
                        gt_path = os.path.join(args.gt_dir, pcd_name)
                        if os.path.exists(gt_path):
                            gt_pcd = o3d.io.read_point_cloud(gt_path)
                            gt_points = np.asarray(gt_pcd.points)
                            gt = torch.from_numpy(gt_points).float().cuda().unsqueeze(0)
                            print(f"  ✅ GT loaded: {gt.shape}")
                    except Exception as gt_error:
                        print(f"  ⚠️ Failed to load GT: {gt_error}")

                # Convert to tensor
                input_pcd = torch.from_numpy(input_points).float().cuda()
                input_pcd = rearrange(input_pcd, 'n c -> c n').unsqueeze(0)

                # Normalize
                input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)

                # Model upsampling
                pcd_upsampled = upsampling(args, model, input_pcd)

                # Denormalize
                pcd_upsampled = centroid + pcd_upsampled * furthest_distance

                # Point count control
                if pcd_upsampled.shape[-1] > target_num:
                    pcd_upsampled = pcd_upsampled[:, :, (pcd_upsampled.shape[-1]-target_num):]

                # Prepare for saving
                saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').contiguous()
                saved_pcd = saved_pcd.detach().cpu().numpy()

                # Data validity check
                nan_count = np.sum(np.isnan(saved_pcd))
                inf_count = np.sum(np.isinf(saved_pcd))
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ❌ Error: Data contains invalid values")
                    txt_result.append(f'{pcd_name}: Data contains NaN/Inf')
                    continue

                # Save file
                save_folder = os.path.join(args.save_dir, 'xyz')
                os.makedirs(save_folder, exist_ok=True)
                output_path = os.path.join(save_folder, pcd_name)
                
                try:
                    np.savetxt(output_path, saved_pcd, fmt='%.6f')
                    print(f"  ✅ File saved: {output_path}")
                except Exception as save_error:
                    print(f"  ❌ File save exception: {save_error}")
                    txt_result.append(f'{pcd_name}: File save exception')
                    continue

                # Calculate Chamfer distance
                if gt is not None:
                    try:
                        cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()
                        cd_scaled = cd * 1e3
                        print(f"  ✅ Chamfer distance: {cd_scaled:.6f}")
                        txt_result.append(f'{pcd_name}: {cd_scaled:.6f}')
                        total_cd += cd
                        counter += 1.0
                    except Exception as cd_error:
                        print(f"  ❌ Chamfer distance calculation failed: {cd_error}")
                        txt_result.append(f'{pcd_name}: Chamfer distance calculation failed')
                        continue
                else:
                    txt_result.append(f'{pcd_name}: Successfully processed (evaluation skipped)')
                    counter += 1.0

                print(f"\n✅ File {pcd_name} processing completed!")

            except Exception as e:
                print(f"\n❌ Exception occurred while processing file {pcd_name}:")
                print(f"   Error: {str(e)}")
                import traceback
                traceback.print_exc()
                txt_result.append(f'{pcd_name}: Processing exception - {str(e)}')
                continue

        # Final results
        print(f"\n{'='*60}")
        print(f"Processing completed! Successfully processed {counter} files")
        print(f"{'='*60}")
        
        if counter > 0:
            avg_cd = total_cd / counter * 1e3
            print(f"Average Chamfer distance: {avg_cd:.6f}")
            txt_result.append(f'overall: {avg_cd:.6f}')
        else:
            txt_result.append('overall: No files processed successfully')

        # Save results
        result_file = os.path.join(args.save_dir, 'cd.txt')
        with open(result_file, "w") as f:
            for ll in txt_result:
                f.write(ll + '\n')
        
        print(f"📄 Results saved to: {result_file}")

    return total_cd / counter * 1e3 if counter > 0 else float('nan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniST RepKPU Testing Arguments')
    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--r', default=4, type=float, help='upsampling rate')
    parser.add_argument('--o', action='store_true', help='using original model')
    parser.add_argument('--flexible', action='store_true', help='arbitrary scale?')
    parser.add_argument('--input_dir', default='./output', type=str, help='path to folder of input point clouds')
    parser.add_argument('--gt_dir', default='./output', type=str, help='path to folder of gt point clouds')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud and results')
    parser.add_argument('--ckpt', default='external/RepKPU_ops/pretrain/ckpt-best.pth', type=str, help='checkpoints')
    parser.add_argument('--no_gt', action='store_true', help='skip evaluation (no ground truth)')
    args = parser.parse_args()
    
    # Parse dataset-specific arguments
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
    
    # Load model
    model = model.cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))
    
    # Run test
    if args.flexible:
        test_flexible(model, args)
    else:
        print("Note: Non-flexible mode not yet implemented. Use --flexible flag.")
        test_flexible(model, args)
