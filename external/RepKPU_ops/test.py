import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
import argparse
from models.repkpu import RepKPU, RepKPU_o
from cfgs.upsampling import parse_pu1k_args, parse_pugan_o_args, parse_pugan_args
from cfgs.utils import *
from dataset.dataset import PUDataset
import torch.optim as optim
from glob import glob
import open3d as o3d
from einops import repeat
from models.utils import *
import time
from datetime import datetime

def _normalize_point_cloud(pc):
    # b, n, 3
    centroid = torch.mean(pc, dim=1, keepdim = True) # b, 1, 3
    pc = pc - centroid # b, n, 3
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] # b, 1, 1
    pc = pc / furthest_distance
    return pc

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

def upsampling(args, model, input_pcd):
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = args.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    coarse_pts, _= model.forward(patches)
    coarse_pts = coarse_pts
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1]* args.up_rate)
    return coarse_pts

def _midpoint_interpolate(up_rate, sparse_pts):
    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate) + 1
    k = int(2 * up_rate)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    interpolated_pts = mid_pts
    interpolated_pts = FPS(interpolated_pts, up_pts_num)
    return interpolated_pts

# supprt 4x and 16x
def test_debug(model, args):
    print("=== Starting Step-by-Step Debugging ===")
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
                # Step 1: File reading
                print("🔍 Step 1: Reading point cloud file...")
                pcd = o3d.io.read_point_cloud(path)
                input_points = np.asarray(pcd.points)
                print(f"  ✓ Successfully read {input_points.shape[0]} points")
                print(f"  📊 Point cloud shape: {input_points.shape}")
                print(f"  📋 Data range: X[{input_points[:,0].min():.3f}, {input_points[:,0].max():.3f}]")
                print(f"              Y[{input_points[:,1].min():.3f}, {input_points[:,1].max():.3f}]") 
                print(f"              Z[{input_points[:,2].min():.3f}, {input_points[:,2].max():.3f}]")

                if input_points.shape[0] == 0:
                    print("  ❌ Error: Point cloud is empty")
                    txt_result.append(f'{pcd_name}: Point cloud is empty')
                    continue

                # Step 2: Convert to tensor
                print("\n🔄 Step 2: Converting to PyTorch tensor...")
                input_pcd = torch.from_numpy(input_points).float().cuda()
                print(f"  ✓ Converted to tensor: {input_pcd.shape}")
                print(f"  📱 Device: {input_pcd.device}")
                print(f"  🔢 Data type: {input_pcd.dtype}")
                
                input_pcd = rearrange(input_pcd, 'n c -> c n').unsqueeze(0)
                print(f"  ✓ After rearrangement: {input_pcd.shape}")

                # Step 3: Normalization
                print("\n📏 Step 3: Normalizing point cloud...")
                input_pcd_norm, centroid, furthest_distance = normalize_point_cloud(input_pcd)
                print(f"  ✓ Normalization completed")
                print(f"  📊 Normalized shape: {input_pcd_norm.shape}")
                print(f"  📍 Centroid: {centroid.shape}")
                print(f"  📐 Furthest distance: {furthest_distance.shape}")
                print(f"  🔍 Centroid value: {centroid.squeeze().cpu().numpy()}")
                print(f"  🔍 Furthest distance value: {furthest_distance.squeeze().cpu().item():.6f}")

                # Step 4: Upsampling parameter check
                print(f"\n⚙️ Step 4: Checking upsampling parameters...")
                print(f"  📈 Upsampling rate (r): {args.r}")
                print(f"  🎯 Points per patch (num_points): {args.num_points}")
                print(f"  📦 Patch rate (patch_rate): {args.patch_rate}")
                
                pcd_pts_num = input_pcd_norm.shape[-1]
                patch_pts_num = args.num_points
                sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
                print(f"  📊 Input point count: {pcd_pts_num}")
                print(f"  📊 Calculated sample count: {sample_num}")
                
                if sample_num <= 0:
                    print(f"  ❌ Error: Invalid sample count ({sample_num})")
                    txt_result.append(f'{pcd_name}: Invalid sample count')
                    continue

                # Step 5: Upsampling processing
                print(f"\n🚀 Step 5: Executing upsampling...")
                try:
                    pcd_upsampled = upsampling(args, model, input_pcd_norm)
                    print(f"  ✓ Upsampling completed: {pcd_upsampled.shape}")
                except Exception as upsample_error:
                    print(f"  ❌ Upsampling failed: {upsample_error}")
                    import traceback
                    traceback.print_exc()
                    txt_result.append(f'{pcd_name}: Upsampling failed')
                    continue

                # Step 6: Denormalization
                print(f"\n🔄 Step 6: Denormalizing...")
                pcd_upsampled = centroid + pcd_upsampled * furthest_distance
                print(f"  ✓ Denormalization completed: {pcd_upsampled.shape}")

                # Step 7: 16x upsampling processing
                if args.r == 16:
                    print(f"\n🔄 Step 7: 16x upsampling second stage...")
                    pcd_upsampled, centroid, furthest_distance = normalize_point_cloud(pcd_upsampled)
                    pcd_upsampled = upsampling(args, model, pcd_upsampled)
                    pcd_upsampled = centroid + pcd_upsampled * furthest_distance
                    print(f"  ✓ 16x upsampling completed: {pcd_upsampled.shape}")

                # Step 8: Prepare data for saving
                print(f"\n💾 Step 8: Preparing data for saving...")
                saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').cpu().numpy()
                print(f"  ✓ Converted to numpy: {saved_pcd.shape}")
                print(f"  🔢 Data type: {saved_pcd.dtype}")
                
                # Data validity check
                nan_count = np.sum(np.isnan(saved_pcd))
                inf_count = np.sum(np.isinf(saved_pcd))
                print(f"  🔍 NaN count: {nan_count}")
                print(f"  🔍 Inf count: {inf_count}")
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ❌ Error: Data contains invalid values")
                    txt_result.append(f'{pcd_name}: Data contains NaN/Inf')
                    continue

                # Step 9: Create save directory
                print(f"\n📁 Step 9: Creating save directory...")
                save_folder = os.path.join(args.save_dir, 'xyz')
                print(f"  📂 Target directory: {save_folder}")
                
                try:
                    os.makedirs(save_folder, exist_ok=True)
                    print(f"  ✓ Directory created successfully")
                except Exception as dir_error:
                    print(f"  ❌ Directory creation failed: {dir_error}")
                    txt_result.append(f'{pcd_name}: Directory creation failed')
                    continue

                # Step 10: Save file
                print(f"\n💾 Step 10: Saving file...")
                output_path = os.path.join(save_folder, pcd_name)
                print(f"  📄 Output path: {output_path}")
                
                try:
                    print(f"  🔄 Starting file write...")
                    np.savetxt(output_path, saved_pcd, fmt='%.6f')
                    print(f"  ✓ File write completed")
                    
                    # Immediate verification
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"  ✅ File verification successful! Size: {file_size} bytes")
                        
                        # Try reading for verification
                        try:
                            verify_data = np.loadtxt(output_path)
                            print(f"  ✅ File content verification: {verify_data.shape}")
                        except Exception as verify_error:
                            print(f"  ⚠️ File content verification failed: {verify_error}")
                    else:
                        print(f"  ❌ File save failed: File does not exist")
                        txt_result.append(f'{pcd_name}: File save failed')
                        continue
                        
                except Exception as save_error:
                    print(f"  ❌ File save exception: {save_error}")
                    import traceback
                    traceback.print_exc()
                    txt_result.append(f'{pcd_name}: File save exception')
                    continue

                # Step 11: Chamfer distance calculation
                print(f"\n📊 Step 11: Calculating evaluation metrics...")
                
                # Load GT if available and no_gt flag is not set
                gt = None
                if not args.no_gt:
                    try:
                        gt_path = os.path.join(args.gt_dir, pcd_name)
                        print(f"  📂 Loading GT from: {gt_path}")
                        
                        if os.path.exists(gt_path):
                            gt_pcd = o3d.io.read_point_cloud(gt_path)
                            gt_points = np.asarray(gt_pcd.points)
                            gt = torch.from_numpy(gt_points).float().cuda().unsqueeze(0)
                            print(f"  ✅ GT loaded: {gt.shape}")
                        else:
                            print(f"  ⚠️ GT file not found, skipping evaluation")
                    except Exception as gt_error:
                        print(f"  ⚠️ Failed to load GT: {gt_error}")
                
                if gt is not None:
                    try:
                        print(f"  🎯 Preparing to calculate Chamfer distance...")
                        print(f"  📊 Predicted point cloud: {pcd_upsampled.shape}")
                        print(f"  📊 GT point cloud: {gt.shape}")
                        
                        # Dimension check
                        if pcd_upsampled.shape[-1] == 0 or gt.shape[1] == 0:
                            print(f"  ❌ Dimension error: Cannot calculate Chamfer distance")
                            txt_result.append(f'{pcd_name}: Dimension error')
                            continue
                        
                        cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()
                        cd_scaled = cd * 1e3
                        print(f"  ✅ Chamfer distance: {cd_scaled:.6f}")
                        
                        txt_result.append(f'{pcd_name}: {cd_scaled:.6f}')
                        total_cd += cd
                        counter += 1.0
                        
                    except Exception as cd_error:
                        print(f"  ❌ Chamfer distance calculation failed: {cd_error}")
                        import traceback
                        traceback.print_exc()
                        txt_result.append(f'{pcd_name}: Chamfer distance calculation failed')
                        continue
                else:
                    print(f"  ⏭️ Skipping evaluation (no GT)")
                    txt_result.append(f'{pcd_name}: Successfully processed (evaluation skipped)')
                    counter += 1.0

                print(f"\n✅ File {pcd_name} processing completed!")

            except Exception as e:
                print(f"\n❌ Exception occurred while processing file {pcd_name}:")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {str(e)}")
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


def test_flexible(model, args):
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
                # Step 1: File reading
                print("🔍 Step 1: Reading point cloud file...")
                pcd = o3d.io.read_point_cloud(path)
                input_points = np.asarray(pcd.points)
                print(f"  ✓ Successfully read {input_points.shape[0]} points")
                print(f"  📊 Point cloud shape: {input_points.shape}")

                if input_points.shape[0] == 0:
                    print("  ❌ Error: Point cloud is empty")
                    txt_result.append(f'{pcd_name}: Point cloud is empty')
                    continue

                # Step 2: Calculate target point count
                print("\n🎯 Step 2: Calculating target point count...")
                input_num = input_points.shape[0]
                target_num = int(input_num * args.r)
                print(f"  📊 Input points: {input_num}")
                print(f"  📈 Upsampling rate: {args.r}")
                print(f"  🎯 Target points: {target_num}")

                # Step 3: Load GT if available
                gt = None
                if not args.no_gt:
                    try:
                        gt_path = os.path.join(args.gt_dir, pcd_name)
                        print(f"\n📂 Step 3: Loading ground truth...")
                        print(f"  📁 GT path: {gt_path}")
                        
                        if os.path.exists(gt_path):
                            gt_pcd = o3d.io.read_point_cloud(gt_path)
                            gt_points = np.asarray(gt_pcd.points)
                            gt = torch.from_numpy(gt_points).float().cuda().unsqueeze(0)
                            print(f"  ✅ GT loaded: {gt.shape}")
                        else:
                            print(f"  ⚠️ GT file not found, will skip evaluation")
                    except Exception as gt_error:
                        print(f"  ⚠️ Failed to load GT: {gt_error}")
                else:
                    print(f"\n⏭️ Step 3: Skipping GT loading (no_gt flag set)")

                # Step 4: Convert to tensor
                print("\n🔄 Step 4: Converting to PyTorch tensor...")
                input_pcd = torch.from_numpy(input_points).float().cuda()
                print(f"  ✓ Converted to tensor: {input_pcd.shape}")
                print(f"  📱 Device: {input_pcd.device}")
                
                input_pcd = rearrange(input_pcd, 'n c -> c n').unsqueeze(0)
                print(f"  ✓ After rearrangement: {input_pcd.shape}")

                # Step 5: Normalization
                print("\n📏 Step 5: Normalizing point cloud...")
                input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
                print(f"  ✓ Normalization completed")
                print(f"  📊 Normalized shape: {input_pcd.shape}")
                print(f"  📍 Centroid: {centroid.squeeze().cpu().numpy()}")
                print(f"  📐 Furthest distance: {furthest_distance.squeeze().cpu().item():.6f}")

                # Step 6: Model upsampling
                print(f"\n🚀 Step 6: Executing model upsampling...")
                try:
                    pcd_upsampled = upsampling(args, model, input_pcd)
                    print(f"  ✓ Model upsampling completed: {pcd_upsampled.shape}")
                except Exception as upsample_error:
                    print(f"  ❌ Model upsampling failed: {upsample_error}")
                    import traceback
                    traceback.print_exc()
                    txt_result.append(f'{pcd_name}: Model upsampling failed')
                    continue

                # Step 7: Denormalization
                print(f"\n🔄 Step 7: Denormalizing...")
                pcd_upsampled = centroid + pcd_upsampled * furthest_distance
                print(f"  ✓ Denormalization completed: {pcd_upsampled.shape}")

                # Step 8: Point count control
                print(f"\n✂️ Step 8: Controlling output point count...")
                print(f"  📊 Current point count: {pcd_upsampled.shape[-1]}")
                print(f"  🎯 Target point count: {target_num}")
                
                if pcd_upsampled.shape[-1] > target_num:
                    pcd_upsampled = pcd_upsampled[:, :, (pcd_upsampled.shape[-1]-target_num):]
                    print(f"  ✂️ Point count after trimming: {pcd_upsampled.shape[-1]}")
                else:
                    print(f"  ✓ Point count meets requirement")

                # Step 9: Prepare data for saving
                print(f"\n💾 Step 9: Preparing data for saving...")
                saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').contiguous()
                saved_pcd = saved_pcd.detach().cpu().numpy()
                print(f"  ✓ Converted to numpy: {saved_pcd.shape}")
                print(f"  🔢 Data type: {saved_pcd.dtype}")
                
                # Data validity check
                nan_count = np.sum(np.isnan(saved_pcd))
                inf_count = np.sum(np.isinf(saved_pcd))
                print(f"  🔍 NaN count: {nan_count}")
                print(f"  🔍 Inf count: {inf_count}")
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ❌ Error: Data contains invalid values")
                    txt_result.append(f'{pcd_name}: Data contains NaN/Inf')
                    continue

                # Step 10: Create save directory
                print(f"\n📁 Step 10: Creating save directory...")
                save_folder = os.path.join(args.save_dir, 'xyz')
                print(f"  📂 Target directory: {save_folder}")
                
                try:
                    os.makedirs(save_folder, exist_ok=True)
                    print(f"  ✓ Directory created successfully")
                except Exception as dir_error:
                    print(f"  ❌ Directory creation failed: {dir_error}")
                    txt_result.append(f'{pcd_name}: Directory creation failed')
                    continue

                # Step 11: Save file
                print(f"\n💾 Step 11: Saving file...")
                output_path = os.path.join(save_folder, pcd_name)
                print(f"  📄 Output path: {output_path}")
                
                try:
                    print(f"  🔄 Starting file write...")
                    np.savetxt(output_path, saved_pcd, fmt='%.6f')
                    print(f"  ✓ File write completed")
                    
                    # Immediate verification
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"  ✅ File verification successful! Size: {file_size} bytes")
                        
                        # Try reading for verification
                        try:
                            verify_data = np.loadtxt(output_path)
                            print(f"  ✅ File content verification: {verify_data.shape}")
                        except Exception as verify_error:
                            print(f"  ⚠️ File content verification failed: {verify_error}")
                    else:
                        print(f"  ❌ File save failed: File does not exist")
                        txt_result.append(f'{pcd_name}: File save failed')
                        continue
                        
                except Exception as save_error:
                    print(f"  ❌ File save exception: {save_error}")
                    import traceback
                    traceback.print_exc()
                    txt_result.append(f'{pcd_name}: File save exception')
                    continue

                # Step 12: Chamfer distance calculation
                print(f"\n📊 Step 12: Calculating evaluation metrics...")
                if gt is not None:
                    try:
                        print(f"  🎯 Preparing to calculate Chamfer distance...")
                        print(f"  📊 Predicted point cloud: {pcd_upsampled.shape}")
                        print(f"  📊 GT point cloud: {gt.shape}")
                        
                        # Dimension check
                        if pcd_upsampled.shape[-1] == 0 or gt.shape[1] == 0:
                            print(f"  ❌ Dimension error: Cannot calculate Chamfer distance")
                            txt_result.append(f'{pcd_name}: Dimension error')
                            continue
                        
                        cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()
                        cd_scaled = cd * 1e3
                        print(f"  ✅ Chamfer distance: {cd_scaled:.6f}")
                        
                        txt_result.append(f'{pcd_name}: {cd_scaled:.6f}')
                        total_cd += cd
                        counter += 1.0
                        
                    except Exception as cd_error:
                        print(f"  ❌ Chamfer distance calculation failed: {cd_error}")
                        import traceback
                        traceback.print_exc()
                        txt_result.append(f'{pcd_name}: Chamfer distance calculation failed')
                        continue
                else:
                    print(f"  ⏭️ Skipping evaluation (no GT)")
                    txt_result.append(f'{pcd_name}: Successfully processed (evaluation skipped)')
                    counter += 1.0

                print(f"\n✅ File {pcd_name} processing completed!")

            except Exception as e:
                print(f"\n❌ Exception occurred while processing file {pcd_name}:")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {str(e)}")
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
    parser = argparse.ArgumentParser(description='Testing Arguments')
    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--r', default=4, type=float, help='upsampling rate')
    parser.add_argument('--o', action='store_true', help='using original model')
    parser.add_argument('--flexible', action='store_true', help='aribitrary scale?')
    parser.add_argument('--input_dir', default='./output', type=str, help='path to folder of input point clouds')
    parser.add_argument('--gt_dir', default='./output', type=str, help='path to folder of gt point clouds')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud and results')
    parser.add_argument('--ckpt', default='./output', type=str, help='checkpoints')
    parser.add_argument('--no_gt', action='store_true', help='skip evaluation (no ground truth)')
    args = parser.parse_args()
    
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
    model.load_state_dict(torch.load(args.ckpt))
    if not args.flexible:
        test_debug(model, args)
    else:
        test_flexible(model, args)