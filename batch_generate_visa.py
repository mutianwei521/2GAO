"""
Batch Defect Generation Script - VisA Dataset Version
Calls main_contrastive.py (2-GAO model) for image defect generation
Adapted for VisA dataset directory structure
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

# VisA dataset path
VISA_ROOT = "visaImages"

# Image saving configuration
SAVE_CONFIG = {
    'save_feathered_blend': True,        # Save feathered blend images
    'save_non_feathered_blend': True,    # Save non-feathered blend images
    'save_comparison_grid': True,        # Save comparison grid images
    'save_contrastive_defect': True,     # Save contrastive defect images
    'save_original_good': True,          # Save original good images
    'save_reference_bad': True,          # Save reference bad images
    'save_good_object_masks': True,      # Save good image object masks
    'save_combined_defect_masks': True,  # Save combined defect masks
    'save_bad_defect_masks': True,       # Save bad image defect masks
    'save_defect_heatmaps': True,        # Save defect heatmaps
    'save_attention_heatmaps': False,    # Save attention heatmaps
    'save_other_files': True             # Save other auxiliary files
}

# Feature toggle configuration
FEATURE_CONFIG = {
    'enable_feature_alignment': True,    # Enable feature alignment
    'ioa_threshold': 0.5,                # IoA threshold
    'save_attention_heatmaps': False,    # Save attention heatmaps
    'measure_inference_time': True       # Measure inference time
}

# ==================== End of Configuration Options ====================


def get_image_size(image_path):
    """Get image dimensions"""
    img = cv2.imread(image_path)
    if img is not None:
        return (img.shape[1], img.shape[0])  # (width, height)
    return None


def resize_image_to_original(image_path, target_size, interpolation=cv2.INTER_LANCZOS4):
    """
    Resize image to target dimensions
    
    Args:
        image_path: Image path
        target_size: Target dimensions (width, height)
        interpolation: Interpolation method
    
    Returns:
        bool: Whether successful
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        current_size = (img.shape[1], img.shape[0])
        if current_size != target_size:
            resized = cv2.resize(img, target_size, interpolation=interpolation)
            cv2.imwrite(image_path, resized)
            return True
        return True  # Same size, no resize needed
    except Exception as e:
        print(f"   Warning: Failed to resize {image_path}: {e}")
        return False


def generate_defect_heatmap(image_path, mask_path, output_path):
    """Generate defect probability heatmap"""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   Warning: Cannot read mask {mask_path}")
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


def find_good_pairs_visa(good_directory):
    """
    Find image-mask pairs in VisA dataset good directory
    VisA format: XXX.JPG + XXX_mask.png
    """
    pairs = []
    if not os.path.exists(good_directory):
        return pairs

    files = os.listdir(good_directory)
    files_set = set(files)
    image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
    
    for file in files:
        if file.endswith(image_extensions) and '_mask' not in file:
            base_name = os.path.splitext(file)[0]
            mask_file = f"{base_name}_mask.png"
            
            if mask_file in files_set:
                pairs.append((
                    os.path.join(good_directory, file),
                    os.path.join(good_directory, mask_file)
                ))

    return pairs


def find_bad_pairs_visa(bad_directory):
    """
    Find image-mask pairs in all subfolders (defect types) of VisA dataset bad directory
    VisA structure: bad/defect_type/XXX.JPG + XXX_mask.png
    """
    all_pairs = []
    if not os.path.exists(bad_directory):
        return all_pairs

    image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
    
    # Traverse all defect type subfolders
    for defect_type in os.listdir(bad_directory):
        defect_folder = os.path.join(bad_directory, defect_type)
        
        if not os.path.isdir(defect_folder):
            continue
        
        print(f"   Scanning defect type: {defect_type}")
        
        try:
            files = os.listdir(defect_folder)
        except PermissionError:
            print(f"   Warning: Cannot access {defect_folder}")
            continue
        
        files_set = set(files)
        pairs_count = 0
        
        for file in files:
            if file.endswith(image_extensions) and '_mask' not in file:
                base_name = os.path.splitext(file)[0]
                mask_file = f"{base_name}_mask.png"
                
                if mask_file in files_set:
                    all_pairs.append({
                        'img_path': os.path.join(defect_folder, file),
                        'mask_path': os.path.join(defect_folder, mask_file),
                        'subfolder': defect_type,  # Defect type
                        'filename': base_name,
                        'full_id': f"{defect_type}_{base_name}"
                    })
                    pairs_count += 1
        
        print(f"   Found {pairs_count} pairs in {defect_type}")
    
    return all_pairs


def setup_temp_experiment_visa(good_pairs, bad_pairs, num_bad, exp_id):
    """Set up temporary experiment directory (VisA version)"""
    
    selected_good = random.choice(good_pairs)
    selected_bad = random.sample(bad_pairs, min(num_bad, len(bad_pairs)))
    
    temp_dir = f"temp_visa_{exp_id}"
    good_dir = os.path.join(temp_dir, "good")
    bad_dir = os.path.join(temp_dir, "bad")
    
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    
    # Get original image size
    original_size = get_image_size(selected_good[0])
    
    # Copy good files
    shutil.copy2(selected_good[0], os.path.join(good_dir, "good.png"))
    shutil.copy2(selected_good[1], os.path.join(good_dir, "good_mask.png"))
    
    # Copy bad files
    bad_names = []
    for i, bad_info in enumerate(selected_bad):
        img_path = bad_info['img_path']
        mask_path = bad_info['mask_path']
        defect_type = bad_info['subfolder']
        filename = bad_info['filename']
        
        unique_name = f"{defect_type}_{filename}"
        bad_names.append(unique_name)
        
        shutil.copy2(img_path, os.path.join(bad_dir, f"bad_{i+1:02d}.png"))
        shutil.copy2(mask_path, os.path.join(bad_dir, f"bad_{i+1:02d}_mask.png"))
        
        print(f"   Copied: {defect_type}/{filename} -> bad_{i+1:02d}")
    
    good_name = os.path.splitext(os.path.basename(selected_good[0]))[0]
    
    # Return original size information
    return good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad, original_size


def generate_smart_prompt_visa(category, selected_bad_info):
    """
    Generate intelligent prompt based on VisA object category and defect types
    Defect types are extracted from subfolder names
    """
    # VisA product category mapping
    product_tokens = {
        "candle": "candle",
        "capsules": "capsule",
        "cashew": "cashew",
        "chewinggum": "chewing gum",
        "fryum": "fryum",
        "macaroni1": "macaroni",
        "macaroni2": "macaroni",
        "pcb1": "pcb circuit board",
        "pcb2": "pcb circuit board",
        "pcb3": "pcb circuit board",
        "pcb4": "pcb circuit board",
        "pipe_fryum": "pipe fryum"
    }
    
    # VisA defect type mapping (from subfolder name to more general description)
    defect_type_mapping = {
        # Common defects
        "bubble": "bubble",
        "discolor": "discoloration",
        "scratch": "scratch",
        "leak": "leak",
        "misshape": "deformed",
        "bent": "bent",
        "melt": "melted",
        "missing": "missing",
        "crack": "crack",
        "hole": "hole",
        "stain": "stain",
        "contamination": "contamination",
        "burn": "burn",
        "damage": "damage",
        "defect": "defect"
    }
    
    product_token = product_tokens.get(category, category)
    
    # Extract defect types from selected bad image info
    defect_types = []
    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']  # e.g.: "bubble_discolor_scratch"
        
        # Parse combined defect types
        defect_parts = subfolder.split('_')
        for part in defect_parts:
            mapped = defect_type_mapping.get(part, part)
            if mapped not in defect_types:
                defect_types.append(mapped)
    
    # Generate prompt
    if len(defect_types) == 0:
        anomaly_token = "defect"
    elif len(defect_types) == 1:
        anomaly_token = defect_types[0]
    elif len(defect_types) <= 3:
        anomaly_token = " ".join(defect_types)
    else:
        anomaly_token = "damaged"
    
    prompt = f"{product_token} {anomaly_token}"
    return prompt, defect_types


def run_experiment_visa(good_dir, bad_dir, output_dir, category="capsules", selected_bad_info=None):
    """Run VisA experiment"""
    
    if selected_bad_info:
        prompt, defect_types = generate_smart_prompt_visa(category, selected_bad_info)
        print(f"   Generated prompt: '{prompt}' (defect types: {defect_types})")
    else:
        prompt = f"{category} defect"
        print(f"   Using default prompt: '{prompt}'")
    
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
        cmd.extend(["--enable-feature-alignment"])
        cmd.extend(["--ioa-threshold", str(FEATURE_CONFIG['ioa_threshold'])])
    
    if FEATURE_CONFIG['save_attention_heatmaps']:
        cmd.extend(["--save-attention-heatmaps"])
    
    if FEATURE_CONFIG['measure_inference_time']:
        cmd.extend(["--measure-inference-time"])
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        
        print(f"   Running command: {' '.join(cmd[:5])}...")
        
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
    """Read inference time file"""
    try:
        time_file_path = os.path.join(output_dir, "inference_times.txt")
        if os.path.exists(time_file_path):
            with open(time_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'Total inference time:' in line:
                        time_str = line.split(':')[1].strip().split()[0]
                        return float(time_str)
        return None
    except Exception as e:
        print(f"   Warning: Could not read inference time: {e}")
        return None


def reorganize_output_files(original_output_dir, good_name, bad_names, exp_id, main_output_dir=None, original_size=None):
    """
    Reorganize output files into categorized directories
    
    Args:
        original_output_dir: Original output directory
        good_name: Good image name
        bad_names: Bad image name list
        exp_id: Experiment ID
        main_output_dir: Main output directory
        original_size: Original image dimensions (width, height), used for resize
    """
    
    if not os.path.exists(original_output_dir):
        return {}
    
    if main_output_dir is None:
        main_output_dir = "outputs_visa"
    
    exp_name = f"{good_name}_{'_'.join(bad_names)}_{exp_id}"
    
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
    
    # Pre-create all directories
    all_categories = set(file_categories.values()) | set(mask_categories.values()) | {"defect_heatmaps"}
    for cat in all_categories:
        os.makedirs(os.path.join(main_output_dir, cat), exist_ok=True)
    
    # Process main file categories
    for filename, category in file_categories.items():
        save_key = f"save_{category}"
        if not SAVE_CONFIG.get(save_key, True):
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
            
            # Resize to original dimensions
            if original_size is not None and category != 'comparison_grid':
                if resize_image_to_original(target_file, original_size):
                    print(f"   Moved & resized {filename} -> {category}/{new_filename}")
                else:
                    print(f"   Moved {filename} -> {category}/{new_filename} (resize failed)")
            else:
                print(f"   Moved {filename} -> {category}/{new_filename}")
            
            reorganized_files[category] = target_file
    
    # Process mask files
    for filename, category in mask_categories.items():
        save_key = f"save_{category}"
        if not SAVE_CONFIG.get(save_key, True):
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
            
            # Resize mask to original dimensions
            if original_size is not None:
                resize_image_to_original(target_file, original_size, interpolation=cv2.INTER_NEAREST)
            
            if category not in reorganized_files:
                reorganized_files[category] = []
            if isinstance(reorganized_files[category], list):
                reorganized_files[category].append(target_file)
            else:
                reorganized_files[category] = [reorganized_files[category], target_file]
            
            print(f"   Moved {filename} -> {category}/{new_filename}")
    
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
                print(f"   Generated defect heatmap -> defect_heatmaps/{heatmap_filename}")
    
    return reorganized_files


def get_visa_categories():
    """Get all available categories in VisA dataset"""
    categories = []
    if os.path.exists(VISA_ROOT):
        for item in os.listdir(VISA_ROOT):
            item_path = os.path.join(VISA_ROOT, item)
            if os.path.isdir(item_path):
                good_path = os.path.join(item_path, "good")
                bad_path = os.path.join(item_path, "bad")
                if os.path.exists(good_path) and os.path.exists(bad_path):
                    categories.append(item)
    return sorted(categories)


def run_batch_visa(category, num_experiments=10, num_bad_per_exp=1, output_dir="outputs_visa"):
    """
    Run batch generation on VisA dataset
    
    Args:
        category: VisA category (e.g., 'capsules', 'pcb1', 'candle')
        num_experiments: Number of experiments
        num_bad_per_exp: Number of bad images per experiment
        output_dir: Output directory
    """
    print("=" * 60)
    print("VisA Dataset Batch Defect Generation")
    print("=" * 60)
    print(f"Category: {category}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Build paths
    category_path = os.path.join(VISA_ROOT, category)
    good_dir = os.path.join(category_path, "good")
    bad_dir = os.path.join(category_path, "bad")
    
    # Scan images
    print(f"\n[1] Scanning images...")
    good_pairs = find_good_pairs_visa(good_dir)
    bad_pairs = find_bad_pairs_visa(bad_dir)
    
    print(f"   Good pairs: {len(good_pairs)}")
    print(f"   Bad pairs: {len(bad_pairs)}")
    
    if len(good_pairs) == 0 or len(bad_pairs) == 0:
        print("[ERROR] No valid image pairs found!")
        return
    
    # Run experiments
    experiment_log = []
    
    for exp_num in range(1, num_experiments + 1):
        exp_id = f"visa{exp_num:03d}"
        print(f"\n[Experiment {exp_num}/{num_experiments}]")
        
        # Set up temporary directory
        good_exp_dir, bad_exp_dir, temp_dir, good_name, bad_names, selected_bad, original_size = \
            setup_temp_experiment_visa(good_pairs, bad_pairs, num_bad_per_exp, exp_id)
        
        print(f"   Good: {good_name}")
        print(f"   Bad: {bad_names}")
        if original_size:
            print(f"   Original size: {original_size[0]}x{original_size[1]}")
        
        # Run experiment
        output_exp_dir = os.path.join(temp_dir, "output")
        success, message = run_experiment_visa(good_exp_dir, bad_exp_dir, output_exp_dir, 
                                               category, selected_bad)
        
        if success:
            print(f"[SUCCESS] Experiment completed!")
            
            # Read inference time
            inference_time = read_inference_time(output_exp_dir)
            if inference_time:
                print(f"   Inference time: {inference_time:.2f}s")
            
            # Reorganize files (including resize to original dimensions)
            print(f"\n[Reorganizing output files...]")
            reorganized = reorganize_output_files(output_exp_dir, good_name, bad_names, 
                                                  exp_id, output_dir, original_size)
            
            exp_record = {
                'experiment_id': exp_id,
                'status': 'SUCCESS',
                'good_image': good_name,
                'bad_images': bad_names,
                'inference_time': inference_time,
                'generated_files': list(reorganized.values()),
                'error_message': None
            }
        else:
            print(f"[FAILED] {message[:100]}")
            exp_record = {
                'experiment_id': exp_id,
                'status': 'FAILED',
                'good_image': good_name,
                'bad_images': bad_names,
                'inference_time': None,
                'generated_files': [],
                'error_message': message
            }
        
        experiment_log.append(exp_record)
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    # Generate report
    print("\n" + "=" * 60)
    print("Batch Generation Completed!")
    print("=" * 60)
    
    successful = len([e for e in experiment_log if e['status'] == 'SUCCESS'])
    print(f"Success: {successful}/{num_experiments}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    return experiment_log


def run_full_visa_dataset(base_output_dir="outputs_visa", num_experiments=50, defect_counts=None):
    """
    Traverse all VisA categories, trying different defect counts for each
    
    Args:
        base_output_dir: Base output directory
        num_experiments: Number of experiments per configuration
        defect_counts: Defect count list, default [1, 2, 3, 4]
    """
    if defect_counts is None:
        defect_counts = [1, 2, 3, 4]
    
    categories = get_visa_categories()
    
    print("=" * 70)
    print("VisA Full Dataset Batch Generation")
    print("=" * 70)
    print(f"Categories: {len(categories)}")
    print(f"Defect counts: {defect_counts}")
    print(f"Experiments per config: {num_experiments}")
    print(f"Total experiments: {len(categories) * len(defect_counts) * num_experiments}")
    print(f"Output directory: {base_output_dir}/[category]/[defect_count]/")
    print("=" * 70)
    
    all_results = {}
    
    for cat_idx, category in enumerate(categories):
        print(f"\n{'#' * 70}")
        print(f"# Category {cat_idx + 1}/{len(categories)}: {category}")
        print(f"{'#' * 70}")
        
        all_results[category] = {}
        
        for defect_count in defect_counts:
            print(f"\n--- Defect count: {defect_count} ---")
            
            # Build output directory: outputs_visa/category/defect_count/
            output_dir = os.path.join(base_output_dir, category, f"defect_{defect_count}")
            
            try:
                experiment_log = run_batch_visa(
                    category=category,
                    num_experiments=num_experiments,
                    num_bad_per_exp=defect_count,
                    output_dir=output_dir
                )
                
                if experiment_log:
                    successful = len([e for e in experiment_log if e['status'] == 'SUCCESS'])
                    all_results[category][defect_count] = {
                        'total': len(experiment_log),
                        'success': successful,
                        'output_dir': output_dir
                    }
                else:
                    all_results[category][defect_count] = {
                        'total': 0,
                        'success': 0,
                        'output_dir': output_dir
                    }
            except Exception as e:
                print(f"[ERROR] Failed for {category} with {defect_count} defects: {e}")
                all_results[category][defect_count] = {
                    'total': 0,
                    'success': 0,
                    'error': str(e)
                }
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("FULL DATASET GENERATION SUMMARY")
    print("=" * 70)
    
    total_experiments = 0
    total_success = 0
    
    for category, defect_results in all_results.items():
        print(f"\n{category}:")
        for defect_count, result in defect_results.items():
            status = f"{result['success']}/{result['total']}"
            total_experiments += result['total']
            total_success += result['success']
            print(f"  Defect {defect_count}: {status}")
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_success}/{total_experiments} experiments successful")
    print(f"Output base directory: {os.path.abspath(base_output_dir)}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VisA Dataset Batch Defect Generation")
    parser.add_argument("--category", "-c", type=str, default=None,
                       help="Single category to process (default: all categories)")
    parser.add_argument("--num-experiments", "-n", type=int, default=50,
                       help="Number of experiments per configuration (default: 50)")
    parser.add_argument("--defect-counts", "-d", type=str, default="1,2,3,4",
                       help="Comma-separated defect counts (default: 1,2,3,4)")
    parser.add_argument("--output", "-o", type=str, default="outputs_visa",
                       help="Base output directory (default: outputs_visa)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available categories and exit")
    
    args = parser.parse_args()
    
    # Display available categories
    categories = get_visa_categories()
    
    if args.list:
        print("Available VisA categories:")
        for cat in categories:
            print(f"  - {cat}")
        sys.exit(0)
    
    # Parse defect counts
    defect_counts = [int(x.strip()) for x in args.defect_counts.split(",")]
    
    print("=" * 70)
    print("VisA Dataset Batch Defect Generation")
    print("=" * 70)
    print(f"Available categories: {categories}")
    print(f"Defect counts: {defect_counts}")
    print(f"Experiments per config: {args.num_experiments}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
    
    if args.category:
        # Single category mode
        if args.category not in categories:
            print(f"[ERROR] Category '{args.category}' not found!")
            print(f"Available: {categories}")
            sys.exit(1)
        
        for defect_count in defect_counts:
            output_dir = os.path.join(args.output, args.category, f"defect_{defect_count}")
            print(f"\n--- Running {args.category} with {defect_count} defect(s) ---")
            run_batch_visa(
                category=args.category,
                num_experiments=args.num_experiments,
                num_bad_per_exp=defect_count,
                output_dir=output_dir
            )
    else:
        # Full dataset mode
        run_full_visa_dataset(
            base_output_dir=args.output,
            num_experiments=args.num_experiments,
            defect_counts=defect_counts
        )
