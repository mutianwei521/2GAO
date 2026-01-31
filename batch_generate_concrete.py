"""
Batch Defect Generation Script - ConcreteImages Dataset Version
Calls main_contrastive.py (2-GAO model) for image defect generation
Adapted for ConcreteImages dataset directory structure

Logic: 50 good images × each defect type × defect_count(1-4) 
Output: outputs_concrete/defect_N/defect_type/
"""

import os
import random
import shutil
import subprocess
import sys
import datetime
import warnings
import cv2
import numpy as np
warnings.filterwarnings('ignore')

# ==================== Configuration Options ====================

# ConcreteImages dataset path
CONCRETE_ROOT = "concreteImages/concrete"

# Image saving configuration
SAVE_CONFIG = {
    'save_feathered_blend': True,
    'save_non_feathered_blend': True,
    'save_comparison_grid': True,
    'save_contrastive_defect': True,
    'save_original_good': True,
    'save_reference_bad': True,
    'save_good_object_masks': True,
    'save_combined_defect_masks': True,
    'save_bad_defect_masks': True,
    'save_defect_heatmaps': True,
    'save_attention_heatmaps': False,
    'save_other_files': True
}

# Feature toggle configuration
FEATURE_CONFIG = {
    'enable_feature_alignment': True,
    'ioa_threshold': 0.5,
    'save_attention_heatmaps': False,
    'measure_inference_time': True
}

# ==================== End of Configuration Options ====================

# Import all helper functions from previous file (for brevity, only function names listed)
def get_image_size(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return (img.shape[1], img.shape[0])
    return None

def resize_image_to_original(image_path, target_size, interpolation=cv2.INTER_LANCZOS4):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        current_size = (img.shape[1], img.shape[0])
        if current_size != target_size:
            resized = cv2.resize(img, target_size, interpolation=interpolation)
            cv2.imwrite(image_path, resized)
            return True
        return True
    except Exception as e:
        print(f"   Warning: Failed to resize {image_path}: {e}")
        return False

def generate_defect_heatmap(image_path, mask_path, output_path):
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        image = cv2.imread(image_path)
        if image is not None and mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        if dist_transform.max() > 0:
            dist_normalized = dist_transform / dist_transform.max()
        else:
            dist_normalized = dist_transform
        probability_map = np.exp(-3 * (1 - dist_normalized)) * (binary_mask > 0).astype(np.float32)
        kernel_size = max(mask.shape) // 10
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 31)
        probability_map_smooth = cv2.GaussianBlur(probability_map, (kernel_size, kernel_size), 0)
        probability_map_normalized = cv2.normalize(probability_map_smooth, None, 0, 255, cv2.NORM_MINMAX)
        probability_map_uint8 = probability_map_normalized.astype(np.uint8)
        heatmap = cv2.applyColorMap(probability_map_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, heatmap)
        return True
    except Exception as e:
        print(f"   Error generating defect heatmap: {e}")
        return False

def find_good_pairs_in_subfolder(subfolder_path):
    pairs = []
    if not os.path.exists(subfolder_path):
        return pairs
    files = os.listdir(subfolder_path)
    files_set = set(files)
    image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
    for file in files:
        if file.endswith(image_extensions) and '_mask' not in file:
            base_name = os.path.splitext(file)[0]
            mask_png = f"{base_name}_mask.png"
            mask_jpg = f"{base_name}_mask.jpg"
            if mask_png in files_set:
                pairs.append((os.path.join(subfolder_path, file), os.path.join(subfolder_path, mask_png)))
            elif mask_jpg in files_set:
                pairs.append((os.path.join(subfolder_path, file), os.path.join(subfolder_path, mask_jpg)))
    return pairs

def find_all_good_pairs(good_directory):
    all_pairs = []
    if not os.path.exists(good_directory):
        return all_pairs
    for subfolder in os.listdir(good_directory):
        subfolder_path = os.path.join(good_directory, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"   Scanning good subfolder: {subfolder}")
            pairs = find_good_pairs_in_subfolder(subfolder_path)
            all_pairs.extend(pairs)
            print(f"   Found {len(pairs)} good pairs in {subfolder}")
    return all_pairs

def find_bad_pairs_in_subfolder(subfolder_path, defect_type):
    pairs = []
    files = os.listdir(subfolder_path)
    files_set = set(files)
    image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
    for file in files:
        if file.endswith(image_extensions) and '_mask' not in file:
            base_name = os.path.splitext(file)[0]
            mask_png = f"{base_name}_mask.png"
            mask_jpg = f"{base_name}_mask.jpg"
            if mask_png in files_set:
                pairs.append({
                    'img_path': os.path.join(subfolder_path, file),
                    'mask_path': os.path.join(subfolder_path, mask_png),
                    'subfolder': defect_type,
                    'filename': base_name,
                    'full_id': f"{defect_type}_{base_name}"
                })
            elif mask_jpg in files_set:
                pairs.append({
                    'img_path': os.path.join(subfolder_path, file),
                    'mask_path': os.path.join(subfolder_path, mask_jpg),
                    'subfolder': defect_type,
                    'filename': base_name,
                    'full_id': f"{defect_type}_{base_name}"
                })
    return pairs

def find_all_bad_pairs(bad_directory):
    all_pairs = []
    if not os.path.exists(bad_directory):
        return all_pairs
    image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
    
    # Process root directory files
    root_files = [f for f in os.listdir(bad_directory) if os.path.isfile(os.path.join(bad_directory, f))]
    if root_files:
        print(f"   Scanning bad root directory")
        files_set = set(root_files)
        pairs_count = 0
        for file in root_files:
            if file.endswith(image_extensions) and '_mask' not in file:
                base_name = os.path.splitext(file)[0]
                mask_png = f"{base_name}_mask.png"
                mask_jpg = f"{base_name}_mask.jpg"
                if mask_png in files_set or mask_jpg in files_set:
                    mask_file = mask_png if mask_png in files_set else mask_jpg
                    all_pairs.append({
                        'img_path': os.path.join(bad_directory, file),
                        'mask_path': os.path.join(bad_directory, mask_file),
                        'subfolder': 'root',
                        'filename': base_name,
                        'full_id': f"root_{base_name}"
                    })
                    pairs_count += 1
        print(f"   Found {pairs_count} pairs in root directory")
    
    # Traverse subfolders
    for defect_type in os.listdir(bad_directory):
        defect_folder = os.path.join(bad_directory, defect_type)
        if not os.path.isdir(defect_folder):
            continue
        print(f"   Scanning defect type: {defect_type}")
        try:
            pairs = find_bad_pairs_in_subfolder(defect_folder, defect_type)
            all_pairs.extend(pairs)
            print(f"   Found {len(pairs)} pairs in {defect_type}")
        except Exception as e:
            print(f"   Warning: Error scanning {defect_type}: {e}")
    return all_pairs

def setup_temp_experiment_concrete(good_pair, bad_pairs, num_bad, exp_id):
    selected_bad = random.sample(bad_pairs, min(num_bad, len(bad_pairs)))
    temp_dir = f"temp_concrete_{exp_id}"
    good_dir = os.path.join(temp_dir, "good")
    bad_dir = os.path.join(temp_dir, "bad")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    
    original_size = get_image_size(good_pair[0])
    shutil.copy2(good_pair[0], os.path.join(good_dir, "good.png"))
    shutil.copy2(good_pair[1], os.path.join(good_dir, "good_mask.png"))
    
    bad_names = []
    for i, bad_info in enumerate(selected_bad):
        unique_name = f"{bad_info['subfolder']}_{bad_info['filename']}"
        bad_names.append(unique_name)
        shutil.copy2(bad_info['img_path'], os.path.join(bad_dir, f"bad_{i+1:02d}.png"))
        shutil.copy2(bad_info['mask_path'], os.path.join(bad_dir, f"bad_{i+1:02d}_mask.png"))
    
    good_name = os.path.splitext(os.path.basename(good_pair[0]))[0]
    return good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad, original_size

def run_experiment_concrete(good_dir, bad_dir, output_dir):
    prompt = "concrete crack"
    cmd = [
        sys.executable, "main_contrastive.py",
        "--good-dir", good_dir,
        "--bad-dir", bad_dir,
        "--output-dir", output_dir,
        "--prompt", prompt,
        "--num-inference-steps", "100",
        "--r", "0.25",
        "--num-optimization-steps", "25",
        "--optimization-interval", "5",
        "--feather-radius", "10"
    ]
    
    if FEATURE_CONFIG['enable_feature_alignment']:
        cmd.extend(["--enable-feature-alignment", "--ioa-threshold", str(FEATURE_CONFIG['ioa_threshold'])])
    if FEATURE_CONFIG['save_attention_heatmaps']:
        cmd.extend(["--save-attention-heatmaps"])
    if FEATURE_CONFIG['measure_inference_time']:
        cmd.extend(["--measure-inference-time"])
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               encoding='utf-8', errors='ignore', timeout=3600)
        if result.returncode == 0:
            return True, "Success"
        else:
            error_msg = ""
            if result.stderr:
                error_msg += f"STDERR: {result.stderr[:500]}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout[-500:]}"
            return False, error_msg if error_msg else "Unknown error"
    except subprocess.TimeoutExpired:
        return False, "Process timed out after 1 hour"
    except Exception as e:
        return False, str(e)

def read_inference_time(output_dir):
    try:
        time_file_path = os.path.join(output_dir, "inference_times.txt")
        if os.path.exists(time_file_path):
            with open(time_file_path, 'r', encoding='utf-8') as f:
                for line in f.read().split('\n'):
                    if 'Total inference time:' in line:
                        return float(line.split(':')[1].strip().split()[0])
        return None
    except:
        return None

def reorganize_output_files(original_output_dir, good_name, bad_names, exp_id, main_output_dir=None, original_size=None):
    if not os.path.exists(original_output_dir):
        return {}
    if main_output_dir is None:
        main_output_dir = "outputs_concrete"
    
    # Use short filename to avoid Windows path length limit
    # Format: good_image_name(truncated)_experiment_id
    short_good_name = good_name[:30] if len(good_name) > 30 else good_name
    exp_name = f"{short_good_name}_{exp_id}"
    file_categories = {
        "feathered_blend_image.png": "feathered_blend",
        "non_feathered_blend_image.png": "non_feathered_blend",
        "comparison_grid.png": "comparison_grid",
        "contrastive_defect_image.png": "contrastive_defect",
        "original_good_image.png": "original_good",
        "reference_bad_image.png": "reference_bad"
    }
    mask_categories = {
        "good_object_mask.png": "good_object_masks",
        "combined_defect_mask.png": "combined_defect_masks",
        "bad_defect_mask_1.png": "bad_defect_masks",
        "bad_defect_mask_2.png": "bad_defect_masks"
    }
    
    reorganized_files = {}
    timestamp = None
    all_categories = set(file_categories.values()) | set(mask_categories.values()) | {"defect_heatmaps"}
    for cat in all_categories:
        os.makedirs(os.path.join(main_output_dir, cat), exist_ok=True)
    
    # Process main files
    for filename, category in file_categories.items():
        if not SAVE_CONFIG.get(f"save_{category}", True):
            continue
        source_file = os.path.join(original_output_dir, filename)
        if os.path.exists(source_file):
            category_dir = os.path.join(main_output_dir, category)
            file_ext = os.path.splitext(filename)[1]
            new_filename = f"{exp_name}{file_ext}"
            target_file = os.path.join(category_dir, new_filename)
            if os.path.exists(target_file):
                if timestamp is None:
                    timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                new_filename = f"{exp_name}_{timestamp}{file_ext}"
                target_file = os.path.join(category_dir, new_filename)
            shutil.move(source_file, target_file)
            if original_size is not None and category != 'comparison_grid':
                resize_image_to_original(target_file, original_size)
            reorganized_files[category] = target_file
    
    # Process mask files
    for filename, category in mask_categories.items():
        if not SAVE_CONFIG.get(f"save_{category}", True):
            continue
        source_file = os.path.join(original_output_dir, filename)
        if os.path.exists(source_file):
            mask_category_dir = os.path.join(main_output_dir, category)
            file_ext = os.path.splitext(filename)[1]
            new_filename = f"{exp_name}{file_ext}"
            target_file = os.path.join(mask_category_dir, new_filename)
            if os.path.exists(target_file):
                if timestamp is None:
                    timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                new_filename = f"{exp_name}_{timestamp}{file_ext}"
                target_file = os.path.join(mask_category_dir, new_filename)
            shutil.move(source_file, target_file)
            if original_size is not None:
                resize_image_to_original(target_file, original_size, interpolation=cv2.INTER_NEAREST)
            if category not in reorganized_files:
                reorganized_files[category] = []
            if isinstance(reorganized_files[category], list):
                reorganized_files[category].append(target_file)
            else:
                reorganized_files[category] = [reorganized_files[category], target_file]
    
    # Generate defect heatmap
    if SAVE_CONFIG.get('save_defect_heatmaps', True):
        feathered_blend_path = reorganized_files.get('feathered_blend')
        combined_mask_path = reorganized_files.get('combined_defect_masks')
        if isinstance(combined_mask_path, list) and len(combined_mask_path) > 0:
            combined_mask_path = combined_mask_path[0]
        if feathered_blend_path and combined_mask_path:
            heatmap_dir = os.path.join(main_output_dir, "defect_heatmaps")
            heatmap_filename = f"{exp_name}_defect_heatmap.png"
            heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
            if os.path.exists(heatmap_path):
                if timestamp is None:
                    timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                heatmap_filename = f"{exp_name}_defect_heatmap_{timestamp}.png"
                heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
            if generate_defect_heatmap(feathered_blend_path, combined_mask_path, heatmap_path):
                reorganized_files["defect_heatmaps"] = heatmap_path
    
    return reorganized_files


def run_full_concrete_dataset(base_output_dir="outputs_concrete", num_good_samples=50, defect_counts=None):
    """
    Run ConcreteImages full batch generation
    
    Logic (same as VisA):
    - Randomly sample num_good_samples good images (default 50)
    - For each defect_count (1, 2, 3, 4):
        - For each good image:
            - For each defect_type:
                - Perform 1 generation
    - Output directory: base_output_dir/defect_N/defect_type/
    
    Total experiments = num_good_samples × num_defect_types × len(defect_counts)
    Example: 50 × 8 × 4 = 1600
    """
    if defect_counts is None:
        defect_counts = [1, 2, 3, 4]
    
    print("=" * 70)
    print("ConcreteImages Full Dataset Batch Generation")
    print("=" * 70)
    
    # Scan images
    good_dir = os.path.join(CONCRETE_ROOT, "good")
    bad_dir = os.path.join(CONCRETE_ROOT, "bad")
    
    print(f"\n[1] Scanning images...")
    good_pairs = find_all_good_pairs(good_dir)
    bad_pairs = find_all_bad_pairs(bad_dir)
    
    print(f"   Total good pairs: {len(good_pairs)}")
    print(f"   Total bad pairs: {len(bad_pairs)}")
    
    if len(good_pairs) == 0 or len(bad_pairs) == 0:
        print("[ERROR] No valid image pairs found!")
        return
    
    # Group by defect type
    print(f"\n[2] Grouping bad images by defect type...")
    bad_by_type = {}
    for bad_pair in bad_pairs:
        defect_type = bad_pair['subfolder']
        if defect_type not in bad_by_type:
            bad_by_type[defect_type] = []
        bad_by_type[defect_type].append(bad_pair)
    
    print(f"   Defect types: {len(bad_by_type)}")
    for dtype, pairs in sorted(bad_by_type.items()):
        print(f"      - {dtype}: {len(pairs)} images")
    
    # Randomly sample good images
    print(f"\n[3] Sampling {num_good_samples} good images...")
    sampled_good = random.sample(good_pairs, min(num_good_samples, len(good_pairs)))
    print(f"   Sampled: {len(sampled_good)} good images")
    
    # Calculate total experiments
    total_experiments = len(sampled_good) * len(bad_by_type) * len(defect_counts)
    
    print(f"\n[4] Experiment Configuration")
    print(f"   Good samples: {len(sampled_good)}")
    print(f"   Defect types: {len(bad_by_type)}")
    print(f"   Defect counts: {defect_counts}")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Output: {base_output_dir}/defect_N/defect_type/")
    print("=" * 70)
    
    # Run all experiments
    all_results = {}
    global_exp_count = 0
    
    for defect_count in defect_counts:
        print(f"\n{'#'*70}")
        print(f"# Defect Count: {defect_count}")
        print(f"{'#'*70}")
        
        experiment_log = []
        
        for good_idx, good_pair in enumerate(sampled_good):
            print(f"\n[Good {good_idx + 1}/{len(sampled_good)}] {os.path.basename(good_pair[0])}")
            
            for defect_type, available_bad in sorted(bad_by_type.items()):
                global_exp_count += 1
                exp_id = f"d{defect_count}_{global_exp_count:04d}"
                
                print(f"  [{global_exp_count}/{total_experiments}] {defect_type} ", end="")
                
                # Set up temporary directory
                good_exp_dir, bad_exp_dir, temp_dir, good_name, bad_names, selected_bad, original_size = \
                    setup_temp_experiment_concrete(good_pair, available_bad, defect_count, exp_id)
                
                # Build output directory: base_output_dir/defect_N/defect_type/
                output_dir = os.path.join(base_output_dir, f"defect_{defect_count}", defect_type)
                
                # Run experiment
                output_exp_dir = os.path.join(temp_dir, "output")
                success, message = run_experiment_concrete(good_exp_dir, bad_exp_dir, output_exp_dir)
                
                if success:
                    print("[OK]")
                    inference_time = read_inference_time(output_exp_dir)
                    reorganized = reorganize_output_files(output_exp_dir, good_name, bad_names, 
                                                          exp_id, output_dir, original_size)
                    exp_record = {
                        'experiment_id': exp_id,
                        'status': 'SUCCESS',
                        'good_image': good_name,
                        'bad_images': bad_names,
                        'defect_type': defect_type,
                        'defect_count': defect_count,
                        'inference_time': inference_time
                    }
                else:
                    print(f"[FAIL]")
                    exp_record = {
                        'experiment_id': exp_id,
                        'status': 'FAILED',
                        'good_image': good_name,
                        'defect_type': defect_type,
                        'defect_count': defect_count,
                        'error_message': message[:100]
                    }
                
                experiment_log.append(exp_record)
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        # Statistics for current defect_count
        successful = len([e for e in experiment_log if e['status'] == 'SUCCESS'])
        all_results[defect_count] = {
            'total': len(experiment_log),
            'success': successful,
            'experiments': experiment_log
        }
        print(f"\n[Defect {defect_count} Complete] {successful}/{len(experiment_log)} successful")
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("BATCH GENERATION COMPLETED")
    print("=" * 70)
    
    total_exp = 0
    total_success = 0
    for defect_count, result in sorted(all_results.items()):
        total_exp += result['total']
        total_success += result['success']
        print(f"Defect {defect_count}: {result['success']}/{result['total']}")
    
    print(f"\nTOTAL: {total_success}/{total_exp} experiments successful")
    print(f"Output: {os.path.abspath(base_output_dir)}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ConcreteImages Dataset Batch Defect Generation")
    parser.add_argument("--num-good", "-g", type=int, default=50,
                       help="Number of good images to sample (default: 50)")
    parser.add_argument("--defect-counts", "-d", type=str, default="1,2,3,4",
                       help="Comma-separated defect counts (default: 1,2,3,4)")
    parser.add_argument("--output", "-o", type=str, default="outputs_concrete",
                       help="Base output directory (default: outputs_concrete)")
    
    args = parser.parse_args()
    
    # Parse defect counts
    defect_counts = [int(x.strip()) for x in args.defect_counts.split(",")]
    
    print("=" * 70)
    print("ConcreteImages Dataset Batch Defect Generation")
    print("=" * 70)
    print(f"Good samples: {args.num_good}")
    print(f"Defect counts: {defect_counts}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
    
    run_full_concrete_dataset(
        base_output_dir=args.output,
        num_good_samples=args.num_good,
        defect_counts=defect_counts
    )
