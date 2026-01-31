"""
Batch Defect Generation Script (Simplified Version)
Calls main_contrastive.py (2-GAO model) for image defect generation
Removed all evaluation metric calculation code
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
    'save_defect_heatmaps': True,        # Save defect heatmaps (defect areas overlaid on generated image)
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


def generate_defect_heatmap(image_path, mask_path, output_path):
    """
    Generate defect probability heatmap: higher probability appears redder, lower appears bluer
    Uses distance transform and Gaussian blur to create natural gradient effects
    
    Args:
        image_path: Path to generated defect image (used for size reference)
        mask_path: Path to defect mask
        output_path: Output heatmap path
    
    Returns:
        bool: Whether generation was successful
    """
    try:
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   Warning: Cannot read mask {mask_path}")
            return False
        
        # Read original image for size reference
        image = cv2.imread(image_path)
        if image is not None and mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Binarize mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Use distance transform to create gradient effect
        # Calculate distance from defect region outward
        dist_transform = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalize distance transform result
        if dist_transform.max() > 0:
            dist_normalized = dist_transform / dist_transform.max()
        else:
            dist_normalized = dist_transform
        
        # Invert distance (so defect center has higher probability)
        # and use exponential function for more natural decay
        probability_map = np.exp(-3 * (1 - dist_normalized)) * (binary_mask > 0).astype(np.float32)
        
        # Apply Gaussian blur to entire image, creating gradient from defect region outward
        # First extend the influence range of defect region
        kernel_size = max(mask.shape) // 10
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 31)  # At least 31 pixels
        
        # Apply Gaussian blur to probability_map to smooth edges
        probability_map_smooth = cv2.GaussianBlur(probability_map, (kernel_size, kernel_size), 0)
        
        # Normalize to 0-255 range
        probability_map_normalized = cv2.normalize(probability_map_smooth, None, 0, 255, cv2.NORM_MINMAX)
        probability_map_uint8 = probability_map_normalized.astype(np.uint8)
        
        # Apply JET colormap: low values=blue, high values=red
        heatmap = cv2.applyColorMap(probability_map_uint8, cv2.COLORMAP_JET)
        
        # Save heatmap
        cv2.imwrite(output_path, heatmap)
        return True
        
    except Exception as e:
        print(f"   Error generating defect heatmap: {e}")
        return False


def find_good_pairs(good_directory):
    """Find image-mask pairs in good directory (optimized version)"""
    pairs = []
    if not os.path.exists(good_directory):
        return pairs

    files = os.listdir(good_directory)
    files_set = set(files)  # Use set for better lookup efficiency
    image_extensions = ('.png', '.jpg', '.jpeg')
    
    for file in files:
        file_lower = file.lower()
        if file_lower.endswith(image_extensions) and 'mask' not in file_lower:
            base_name = os.path.splitext(file)[0]
            # Check if mask file exists
            for mask_suffix in ('_mask.png', 'mask.png'):
                mask_file = base_name + mask_suffix
                if mask_file in files_set:
                    pairs.append((
                        os.path.join(good_directory, file),
                        os.path.join(good_directory, mask_file)
                    ))
                    break

    return pairs


def find_bad_pairs_from_subfolders(bad_directory):
    """Find image-mask pairs in all subfolders of bad directory (optimized version)"""
    all_pairs = []
    if not os.path.exists(bad_directory):
        return all_pairs

    image_extensions = ('.png', '.jpg', '.jpeg')
    
    # Traverse all subfolders
    for subfolder in os.listdir(bad_directory):
        subfolder_path = os.path.join(bad_directory, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"   Scanning subfolder: {subfolder}")

        try:
            files = os.listdir(subfolder_path)
        except PermissionError:
            print(f"   Warning: Cannot access {subfolder_path}")
            continue

        files_set = set(files)  # Use set for better lookup efficiency
        pairs_count = 0

        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(image_extensions) and 'mask' not in file_lower:
                base_name = os.path.splitext(file)[0]
                
                for mask_suffix in ('_mask.png', 'mask.png'):
                    mask_file = base_name + mask_suffix
                    if mask_file in files_set:
                        all_pairs.append({
                            'img_path': os.path.join(subfolder_path, file),
                            'mask_path': os.path.join(subfolder_path, mask_file),
                            'subfolder': subfolder,
                            'filename': base_name,
                            'full_id': f"{subfolder}_{base_name}"
                        })
                        pairs_count += 1
                        break

        print(f"   Found {pairs_count} pairs in {subfolder}")

    return all_pairs


def setup_temp_experiment(good_pairs, bad_pairs, num_bad, exp_id):
    """Set up temporary experiment directory"""

    # Randomly select good image
    selected_good = random.choice(good_pairs)

    # Randomly select bad images
    selected_bad = random.sample(bad_pairs, min(num_bad, len(bad_pairs)))

    # Create temporary directory
    temp_dir = f"temp_simple_{exp_id}"
    good_dir = os.path.join(temp_dir, "good")
    bad_dir = os.path.join(temp_dir, "bad")

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    # Copy good files
    shutil.copy2(selected_good[0], os.path.join(good_dir, "good.png"))
    shutil.copy2(selected_good[1], os.path.join(good_dir, "good_mask.png"))

    # Copy bad files
    bad_names = []
    for i, bad_info in enumerate(selected_bad):
        img_path = bad_info['img_path']
        mask_path = bad_info['mask_path']
        subfolder = bad_info['subfolder']
        filename = bad_info['filename']

        img_ext = os.path.splitext(img_path)[1]
        mask_ext = os.path.splitext(mask_path)[1]

        unique_name = f"{subfolder}_{filename}"
        bad_names.append(unique_name)

        shutil.copy2(img_path, os.path.join(bad_dir, f"bad_{i+1:02d}{img_ext}"))
        shutil.copy2(mask_path, os.path.join(bad_dir, f"bad_{i+1:02d}_mask{mask_ext}"))

        print(f"   Copied: {subfolder}/{filename} -> bad_{i+1:02d}")

    good_name = os.path.splitext(os.path.basename(selected_good[0]))[0]

    return good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad


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


def reorganize_output_files(original_output_dir, good_name, bad_names, exp_id, main_output_dir=None):
    """Reorganize output files into categorized directories (optimized version)"""

    if not os.path.exists(original_output_dir):
        return {}

    if main_output_dir is None:
        main_output_dir = "outputs_contrastive2"
    
    # Generate experiment name (calculate only once)
    exp_name = f"{good_name}_{'_'.join(bad_names)}_{exp_id}"

    # Define file categories
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

    # Pre-create all potentially needed directories (create once, avoid repeated checks)
    all_categories = set(file_categories.values()) | set(mask_categories.values()) | {'defect_heatmaps'}
    for category in all_categories:
        os.makedirs(os.path.join(main_output_dir, category), exist_ok=True)

    reorganized_files = {}
    timestamp = None  # Lazy timestamp generation

    # Process main file categories
    for filename, category in file_categories.items():
        if not SAVE_CONFIG.get(f"save_{category}", True):
            continue

        source_file = os.path.join(original_output_dir, filename)

        if os.path.exists(source_file):
            category_dir = os.path.join(main_output_dir, category)
            new_filename = f"{exp_name}.png"
            target_file = os.path.join(category_dir, new_filename)

            if os.path.exists(target_file):
                if timestamp is None:
                    timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                new_filename = f"{exp_name}_{timestamp}.png"
                target_file = os.path.join(category_dir, new_filename)

            shutil.move(source_file, target_file)  # Use move instead of copy
            reorganized_files[category] = target_file
            print(f"   Moved {filename} -> {category}/{new_filename}")

    # Process mask files
    for filename, category in mask_categories.items():
        if not SAVE_CONFIG.get(f"save_{category}", True):
            continue

        source_file = os.path.join(original_output_dir, filename)

        if os.path.exists(source_file):
            mask_category_dir = os.path.join(main_output_dir, category)
            new_filename = f"{exp_name}.png"
            target_file = os.path.join(mask_category_dir, new_filename)

            if os.path.exists(target_file):
                if timestamp is None:
                    timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                new_filename = f"{exp_name}_{timestamp}.png"
                target_file = os.path.join(mask_category_dir, new_filename)

            shutil.move(source_file, target_file)  # Use move instead of copy

            if category not in reorganized_files:
                reorganized_files[category] = []
            reorganized_files[category].append(target_file)

            print(f"   Moved {filename} -> {category}/{new_filename}")

    # Process heatmap files
    if SAVE_CONFIG.get('save_attention_heatmaps', True):
        # Check attention_heatmaps subfolder
        attention_source_dir = os.path.join(original_output_dir, "attention_heatmaps")
        if os.path.exists(attention_source_dir) and os.path.isdir(attention_source_dir):
            attention_target_dir = os.path.join(main_output_dir, "attention_heatmaps")
            os.makedirs(attention_target_dir, exist_ok=True)
            
            attention_files = []
            for heatmap_file in os.listdir(attention_source_dir):
                source_heatmap = os.path.join(attention_source_dir, heatmap_file)
                if os.path.isfile(source_heatmap):
                    name, ext = os.path.splitext(heatmap_file)
                    new_heatmap_filename = f"{name}_{exp_name}{ext}"
                    target_heatmap = os.path.join(attention_target_dir, new_heatmap_filename)
                    
                    if os.path.exists(target_heatmap):
                        timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                        new_heatmap_filename = f"{name}_{exp_name}_{timestamp}{ext}"
                        target_heatmap = os.path.join(attention_target_dir, new_heatmap_filename)
                    
                    shutil.copy2(source_heatmap, target_heatmap)
                    attention_files.append(target_heatmap)
                    print(f"   Moved {heatmap_file} -> attention_heatmaps/{new_heatmap_filename}")
            
            if attention_files:
                reorganized_files["attention_heatmaps"] = attention_files
        
        # Check heatmap files in main directory
        for file in os.listdir(original_output_dir):
            if 'attention' in file.lower() or 'heatmap' in file.lower():
                source_file = os.path.join(original_output_dir, file)
                if os.path.isfile(source_file):
                    attention_target_dir = os.path.join(main_output_dir, "attention_heatmaps")
                    os.makedirs(attention_target_dir, exist_ok=True)
                    
                    name, ext = os.path.splitext(file)
                    new_filename = f"{name}_{exp_name}{ext}"
                    target_file = os.path.join(attention_target_dir, new_filename)
                    
                    if os.path.exists(target_file):
                        timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
                        new_filename = f"{name}_{exp_name}_{timestamp}{ext}"
                        target_file = os.path.join(attention_target_dir, new_filename)
                    
                    shutil.copy2(source_file, target_file)
                    print(f"   Moved {file} -> attention_heatmaps/{new_filename}")
                    
                    if "attention_heatmaps" not in reorganized_files:
                        reorganized_files["attention_heatmaps"] = []
                    reorganized_files["attention_heatmaps"].append(target_file)

    # Generate defect heatmap (using already moved files)
    if SAVE_CONFIG.get('save_defect_heatmaps', True):
        # Use already moved file paths
        feathered_blend_path = reorganized_files.get('feathered_blend')
        combined_mask_path = reorganized_files.get('combined_defect_masks')
        
        # combined_defect_masks is a list, get first one
        if isinstance(combined_mask_path, list) and len(combined_mask_path) > 0:
            combined_mask_path = combined_mask_path[0]
        
        if feathered_blend_path and combined_mask_path and os.path.exists(feathered_blend_path) and os.path.exists(combined_mask_path):
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
            else:
                print(f"   Warning: Failed to generate defect heatmap")
        else:
            print(f"   Warning: Missing files for defect heatmap generation")

    return reorganized_files


def generate_smart_prompt(category, selected_bad_info):
    """Generate intelligent prompt based on object category and selected defect types"""

    product_tokens = {
        "bottle": "bottle", "cable": "cable", "capsule": "capsule",
        "carpet": "carpet", "grid": "grid", "hazelnut": "hazelnut",
        "leather": "leather", "metal_nut": "metal nut", "pill": "pill",
        "screw": "screw", "tile": "tile", "toothbrush": "toothbrush",
        "transistor": "transistor", "wood": "wood", "zipper": "zipper"
    }

    defect_type_mapping = {
        "broken_large": "crack", "broken_small": "crack", "contamination": "stain",
        "bent_wire": "bent", "cable_swap": "swap", "combined": "combined",
        "cut_inner_insulation": "cut", "cut_outer_insulation": "cut",
        "missing_cable": "missing", "missing_wire": "missing", "poke_insulation": "poke",
        "crack": "crack", "faulty_imprint": "imprint", "poke": "poke",
        "scratch": "scratch", "squeeze": "squeeze", "color": "color",
        "cut": "cut", "hole": "hole", "metal_contamination": "contamination",
        "thread": "thread", "bent": "bent", "broken": "broken", "glue": "glue",
        "print": "print", "fold": "fold", "flip": "flip",
        "pill_type": "type", "manipulated_front": "manipulated",
        "scratch_head": "scratch", "scratch_neck": "scratch",
        "thread_side": "thread", "thread_top": "thread",
        "glue_strip": "glue", "gray_stroke": "stroke", "oil": "oil", "rough": "rough",
        "defective": "defective", "bent_lead": "bent", "cut_lead": "cut",
        "damaged_case": "damaged", "misplaced": "misplaced", "liquid": "liquid",
        "broken_teeth": "broken", "fabric_border": "fabric", "fabric_interior": "fabric",
        "split_teeth": "split", "squeezed_teeth": "squeezed"
    }

    product_token = product_tokens.get(category, category)

    defect_types = []
    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']
        mapped_defect = defect_type_mapping.get(subfolder, subfolder)
        if mapped_defect not in defect_types:
            defect_types.append(mapped_defect)

    if len(defect_types) == 1:
        anomaly_token = defect_types[0]
    else:
        anomaly_token = "damaged"

    prompt = f"{product_token} {anomaly_token}"
    return prompt, defect_types


def run_simple_experiment(good_dir, bad_dir, output_dir, category="bottle", selected_bad_info=None):
    """Run simple experiment"""

    if selected_bad_info:
        prompt, defect_types = generate_smart_prompt(category, selected_bad_info)
        print(f"   Generated prompt: '{prompt}' (defect types: {defect_types})")
    else:
        prompt = f"{category} damaged"
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


def generate_experiment_report(experiment_log, output_dir="outputs_contrastive2"):
    """Generate experiment report txt file (simplified version, no evaluation metrics)"""

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"experiment_report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)

    total_experiments = len(experiment_log)
    successful_experiments = len([exp for exp in experiment_log if exp['status'] == 'SUCCESS'])
    failed_experiments = len([exp for exp in experiment_log if exp['status'] == 'FAILED'])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Batch Defect Generation Experiment Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generation time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Successful experiments: {successful_experiments}\n")
        f.write(f"Failed experiments: {failed_experiments}\n")
        f.write(f"Success rate: {successful_experiments/total_experiments*100:.1f}%\n")
        f.write("=" * 80 + "\n\n")

        f.write("Detailed Experiment Log:\n")
        f.write("-" * 80 + "\n")

        for exp in experiment_log:
            f.write(f"Experiment {exp['experiment_id']}:\n")
            f.write(f"  Status: {exp['status']}\n")
            f.write(f"  Good image: {exp['good_image']}\n")
            f.write(f"  Bad images: {', '.join(exp['bad_images'])}\n")

            if exp.get('inference_time') is not None:
                f.write(f"  Inference time: {exp['inference_time']:.2f} seconds\n")

            if exp['status'] == 'SUCCESS':
                f.write(f"  Generated files:\n")
                for file_info in exp['generated_files']:
                    f.write(f"    - {file_info}\n")

            if exp['error_message']:
                f.write(f"  Error message: {exp['error_message']}\n")

            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("Output directory structure:\n")
        full_output_path = os.path.abspath(output_dir)
        f.write(f"Full path: {full_output_path}\n\n")

        f.write(f"{os.path.basename(output_dir)}/\n")
        f.write("├── feathered_blend/         # Feathered blend images (main results)\n")
        f.write("├── non_feathered_blend/     # Non-feathered blend images\n")
        f.write("├── comparison_grid/         # Comparison grid images\n")
        f.write("├── contrastive_defect/      # Contrastive defect images\n")
        f.write("├── original_good/           # Original good images\n")
        f.write("├── reference_bad/           # Reference bad images\n")
        f.write("├── good_object_masks/       # Good image object mask files\n")
        f.write("├── combined_defect_masks/   # Combined defect mask files\n")
        f.write("├── bad_defect_masks/        # Bad image defect mask files\n")
        f.write("└── experiment_report_*.txt  # Experiment report files\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    return report_path


def main():
    print("Batch Defect Generation - Simplified Version")
    print("=" * 50)

    # Configuration - All MVTEC object categories
    MVTEC_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    
    NUM_EXPERIMENTS = 20  # Number of experiments per configuration

    for category in MVTEC_CATEGORIES:
        print("\n" + "=" * 80)
        print(f"[TARGET] PROCESSING CATEGORY: {category.upper()}")
        print("=" * 80)

        BASE_DIR = f"images/{category}"
        GOOD_DIR = os.path.join(BASE_DIR, "good")
        BAD_DIR = os.path.join(BASE_DIR, "bad")

        if not os.path.exists(BASE_DIR):
            print(f"[WARNING] Category directory {BASE_DIR} not found! Skipping...")
            continue

        if not os.path.exists(GOOD_DIR) or not os.path.exists(BAD_DIR):
            print(f"[WARNING] Good or Bad directory not found! Skipping...")
            continue

        # Loop through experiments with 1 to 4 defects
        for NUM_BAD in range(1, 5):
            stroutput = f"outputs_contrastive3/{category}/{NUM_BAD}"
            experiment_log = []

            print(f"\n[CONFIG] {category.upper()} with {NUM_BAD} defects, Output: {stroutput}")

            # Find images
            print("[SCAN] Scanning images...")
            good_pairs = find_good_pairs(GOOD_DIR)
            bad_pairs = find_bad_pairs_from_subfolders(BAD_DIR)

            print(f"   Found {len(good_pairs)} good pairs, {len(bad_pairs)} bad pairs")

            if len(good_pairs) == 0 or len(bad_pairs) < NUM_BAD:
                print(f"[WARNING] Not enough images! Skipping...")
                continue

            # Run experiments
            successful = 0

            for i in range(NUM_EXPERIMENTS):
                exp_id = f"{i+1:02d}"
                print(f"\n[TEST] Experiment {i+1}/{NUM_EXPERIMENTS}")

                try:
                    # Set up experiment
                    good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad_info = setup_temp_experiment(
                        good_pairs, bad_pairs, NUM_BAD, exp_id
                    )

                    exp_record = {
                        'experiment_id': exp_id,
                        'good_image': good_name,
                        'bad_images': bad_names,
                        'status': 'FAILED',
                        'generated_files': [],
                        'error_message': None,
                        'inference_time': None
                    }

                    # Run experiment
                    temp_output_dir = f"temp_output_{exp_id}"
                    success, message = run_simple_experiment(good_dir, bad_dir, temp_output_dir, 
                                                            category, selected_bad_info)

                    if success:
                        inference_time = read_inference_time(temp_output_dir)
                        if inference_time:
                            print(f"   Inference time: {inference_time:.2f}s")
                            exp_record['inference_time'] = inference_time

                        # Reorganize output files
                        reorganized_files = reorganize_output_files(temp_output_dir, good_name, 
                                                                    bad_names, exp_id, stroutput)

                        if reorganized_files:
                            successful += 1
                            exp_record['status'] = 'SUCCESS'
                            for file_category, files in reorganized_files.items():
                                if isinstance(files, list):
                                    exp_record['generated_files'].append(f"{file_category}: {len(files)} files")
                                else:
                                    exp_record['generated_files'].append(f"{file_category}: {os.path.basename(files)}")

                        # Clean up temporary output directory
                        if os.path.exists(temp_output_dir):
                            shutil.rmtree(temp_output_dir)
                    else:
                        print(f"   FAILED: {message[:100]}")
                        exp_record['error_message'] = message
                        if os.path.exists(temp_output_dir):
                            shutil.rmtree(temp_output_dir)

                    experiment_log.append(exp_record)

                    # Clean up temporary experiment directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

                except Exception as e:
                    print(f"[ERROR] {e}")
                    exp_record = {
                        'experiment_id': exp_id,
                        'good_image': 'UNKNOWN',
                        'bad_images': ['UNKNOWN'],
                        'status': 'ERROR',
                        'generated_files': [],
                        'error_message': str(e),
                        'inference_time': None
                    }
                    experiment_log.append(exp_record)

            print(f"\n[RESULT] {category} with {NUM_BAD} defects: {successful}/{NUM_EXPERIMENTS} successful")

            # Generate report
            if experiment_log:
                report_path = generate_experiment_report(experiment_log, stroutput)
                print(f"   Report saved: {report_path}")

    print("\n" + "=" * 80)
    print("[COMPLETE] ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
