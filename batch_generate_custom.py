"""
Batch Defect Generation Script - Custom Dataset Version
========================================================

This script generates synthetic defect images using the 2-GAO framework.
It is designed to work with custom datasets following a specific directory structure.

Usage:
    python batch_generate_custom.py --dataset-root your_dataset --object-name "product"
    python batch_generate_custom.py --dataset-root your_dataset --object-name "product" --defect-counts 1,2,3,4

See CUSTOM_DATASET.md for detailed documentation.
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
import argparse
warnings.filterwarnings('ignore')


# ==============================================================================
# ========================== USER CONFIGURATION ================================
# ==============================================================================
# 
# Modify the settings below to customize the generation process for your dataset.
# All configurable options are grouped here for easy access.
#
# ==============================================================================

# ------------------------------ Dataset Paths ---------------------------------

# Root directory of your dataset (contains 'good' and 'bad' folders)
DATASET_ROOT = "your_dataset"

# Object name used in prompt generation (e.g., "capsule", "screw", "product")
OBJECT_NAME = "product"

# Output directory for generated images
OUTPUT_DIR = "outputs_custom"


# ------------------------------ Defect Type Mapping ---------------------------
# Map folder names to prompt tokens for better generation quality.
# Keys: folder names in bad/ directory
# Values: descriptive words for the prompt

DEFECT_TYPE_MAPPING = {
    # Common defect types - add your own defect types here
    "scratch": "scratch",
    "crack": "crack",
    "stain": "stain",
    "hole": "hole",
    "dent": "dent",
    "contamination": "contamination",
    "discoloration": "discoloration",
    "missing": "missing",
    "bent": "bent",
    "broken": "broken",
    "bubble": "bubble",
    "burn": "burn",
    "corrosion": "corrosion",
    "defect": "defect",
    # Add more mappings as needed:
    # "your_folder_name": "prompt_word",
}


# ------------------------------ Generation Parameters -------------------------

GENERATION_CONFIG = {
    'num_inference_steps': 100,      # Number of diffusion steps (50-100 recommended)
    'r': 0.25,                       # Retention coefficient (0.15-0.35 recommended)
    'num_optimization_steps': 25,    # Attention optimization iterations
    'optimization_interval': 5,      # Apply optimization every N steps
    'feather_radius': 10,            # Blending feather radius in pixels
}


# ------------------------------ Feature Alignment (IoA) -----------------------

FEATURE_CONFIG = {
    'enable_feature_alignment': True,    # Enable IoA-based spatial alignment
    'ioa_threshold': 0.5,                # IoA threshold (0.3-0.9, higher = stricter)
    'save_attention_heatmaps': False,    # Save attention visualization
    'measure_inference_time': True,      # Record inference time
}


# ------------------------------ Output Saving Options -------------------------

SAVE_CONFIG = {
    'save_feathered_blend': True,        # Final blended images (recommended)
    'save_non_feathered_blend': True,    # Non-feathered blend for comparison
    'save_comparison_grid': True,        # Before/after comparison grid
    'save_contrastive_defect': True,     # Intermediate contrastive output
    'save_original_good': True,          # Copy of source good image
    'save_reference_bad': True,          # Copy of source defect image
    'save_good_object_masks': True,      # Object masks used
    'save_combined_defect_masks': True,  # Generated defect masks
    'save_bad_defect_masks': True,       # Source defect masks
    'save_defect_heatmaps': True,        # Defect probability heatmaps
    'save_attention_heatmaps': False,    # Attention visualizations
    'save_other_files': True,            # Other auxiliary files
}


# ==============================================================================
# ======================== END OF USER CONFIGURATION ===========================
# ==============================================================================


# ------------------------------ Utility Functions -----------------------------

def get_image_size(image_path):
    """Get image dimensions (width, height)."""
    img = cv2.imread(image_path)
    if img is not None:
        return (img.shape[1], img.shape[0])
    return None


def resize_image_to_original(image_path, target_size, interpolation=cv2.INTER_LANCZOS4):
    """
    Resize image to target dimensions.
    
    Args:
        image_path: Path to the image file
        target_size: Target size as (width, height)
        interpolation: OpenCV interpolation method
    
    Returns:
        bool: True if successful
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
    except Exception as e:
        print(f"   Warning: Failed to resize {image_path}: {e}")
        return False


def generate_defect_heatmap(image_path, mask_path, output_path):
    """
    Generate a defect probability heatmap from a binary mask.
    
    The heatmap uses distance transform to create smooth probability
    gradients from the defect center to its boundaries.
    """
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


# ------------------------------ Data Loading Functions ------------------------

def find_good_pairs(good_directory):
    """
    Find image-mask pairs in the good directory.
    
    Expected naming convention: image.png + image_mask.png
    
    Args:
        good_directory: Path to the 'good' folder
    
    Returns:
        List of tuples: [(image_path, mask_path), ...]
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


def find_bad_pairs(bad_directory):
    """
    Find image-mask pairs in the bad directory (organized by defect type).
    
    Expected structure: bad/defect_type/image.png + image_mask.png
    
    Args:
        bad_directory: Path to the 'bad' folder
    
    Returns:
        List of dicts with keys: img_path, mask_path, subfolder, filename, full_id
    """
    all_pairs = []
    if not os.path.exists(bad_directory):
        return all_pairs

    image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
    
    # Iterate through defect type subfolders
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
                        'subfolder': defect_type,
                        'filename': base_name,
                        'full_id': f"{defect_type}_{base_name}"
                    })
                    pairs_count += 1
        
        print(f"   Found {pairs_count} pairs in {defect_type}")
    
    return all_pairs


# ------------------------------ Experiment Setup ------------------------------

def setup_temp_experiment(good_pairs, bad_pairs, num_bad, exp_id):
    """
    Set up temporary experiment directory with selected images.
    
    Args:
        good_pairs: List of (image, mask) tuples for good images
        bad_pairs: List of dicts for bad images
        num_bad: Number of defect images to use per experiment
        exp_id: Experiment identifier string
    
    Returns:
        Tuple: (good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad, original_size)
    """
    selected_good = random.choice(good_pairs)
    selected_bad = random.sample(bad_pairs, min(num_bad, len(bad_pairs)))
    
    temp_dir = f"temp_custom_{exp_id}"
    good_dir = os.path.join(temp_dir, "good")
    bad_dir = os.path.join(temp_dir, "bad")
    
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    
    # Get original image size for later resizing
    original_size = get_image_size(selected_good[0])
    
    # Copy good image files
    shutil.copy2(selected_good[0], os.path.join(good_dir, "good.png"))
    shutil.copy2(selected_good[1], os.path.join(good_dir, "good_mask.png"))
    
    # Copy bad image files
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
    
    return good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad, original_size


def generate_smart_prompt(object_name, selected_bad_info, defect_mapping):
    """
    Generate an optimized prompt based on object name and defect types.
    
    Args:
        object_name: Name of the object being synthesized
        selected_bad_info: List of dicts with 'subfolder' keys
        defect_mapping: Dict mapping folder names to prompt tokens
    
    Returns:
        Tuple: (prompt_string, defect_types_list)
    """
    defect_types = []
    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']
        
        # Parse compound defect types (e.g., "crack_scratch")
        defect_parts = subfolder.split('_')
        for part in defect_parts:
            mapped = defect_mapping.get(part, part)
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
    
    prompt = f"{object_name} {anomaly_token}"
    return prompt, defect_types


# ------------------------------ Core Generation -------------------------------

def run_experiment(good_dir, bad_dir, output_dir, object_name, selected_bad_info):
    """
    Run the 2-GAO generation process.
    
    Args:
        good_dir: Path to temporary good directory
        bad_dir: Path to temporary bad directory
        output_dir: Path for output files
        object_name: Object name for prompt
        selected_bad_info: List of bad image info dicts
    
    Returns:
        Tuple: (success: bool, message: str)
    """
    # Generate optimized prompt
    prompt, defect_types = generate_smart_prompt(object_name, selected_bad_info, DEFECT_TYPE_MAPPING)
    print(f"   Generated prompt: '{prompt}' (defect types: {defect_types})")
    
    # Build command
    cmd = [
        sys.executable, "main_contrastive.py",
        "--good-dir", good_dir,
        "--bad-dir", bad_dir,
        "--output-dir", output_dir,
        "--prompt", prompt,
        "--num-inference-steps", str(GENERATION_CONFIG['num_inference_steps']),
        "--r", str(GENERATION_CONFIG['r']),
        "--num-optimization-steps", str(GENERATION_CONFIG['num_optimization_steps']),
        "--optimization-interval", str(GENERATION_CONFIG['optimization_interval']),
        "--feather-radius", str(GENERATION_CONFIG['feather_radius']),
    ]
    
    # Add optional flags
    if FEATURE_CONFIG['enable_feature_alignment']:
        cmd.extend(["--enable-feature-alignment"])
        cmd.extend(["--ioa-threshold", str(FEATURE_CONFIG['ioa_threshold'])])
    
    if FEATURE_CONFIG['save_attention_heatmaps']:
        cmd.extend(["--save-attention-heatmaps"])
    
    if FEATURE_CONFIG['measure_inference_time']:
        cmd.extend(["--measure-inference-time"])
    
    # Execute
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
    """Read inference time from output directory."""
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


# ------------------------------ Output Organization ---------------------------

def reorganize_output_files(original_output_dir, good_name, bad_names, exp_id, 
                           main_output_dir=None, original_size=None):
    """
    Reorganize generated files into categorized directories.
    
    Args:
        original_output_dir: Raw output directory from generation
        good_name: Name of the good image used
        bad_names: List of bad image names used
        exp_id: Experiment identifier
        main_output_dir: Target output directory
        original_size: Original image size for resizing
    
    Returns:
        Dict mapping category names to output file paths
    """
    if not os.path.exists(original_output_dir):
        return {}
    
    if main_output_dir is None:
        main_output_dir = OUTPUT_DIR
    
    exp_name = f"{good_name}_{'_'.join(bad_names)}_{exp_id}"
    
    # File category mappings
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
    
    # Pre-create directories
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
            
            # Resize to original size
            if original_size is not None and category != 'comparison_grid':
                resize_image_to_original(target_file, original_size)
            
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
            
            # Resize mask with nearest neighbor interpolation
            if original_size is not None:
                resize_image_to_original(target_file, original_size, interpolation=cv2.INTER_NEAREST)
            
            if category not in reorganized_files:
                reorganized_files[category] = []
            if isinstance(reorganized_files[category], list):
                reorganized_files[category].append(target_file)
            else:
                reorganized_files[category] = [reorganized_files[category], target_file]
            
            print(f"   Moved {filename} -> {category}/{new_filename}")
    
    # Generate defect heatmaps
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


# ------------------------------ Batch Execution -------------------------------

def run_batch(dataset_root, object_name, num_experiments=10, num_bad_per_exp=1, output_dir=None):
    """
    Run batch generation on a custom dataset.
    
    Args:
        dataset_root: Path to dataset root directory
        object_name: Object name for prompts
        num_experiments: Number of experiments to run
        num_bad_per_exp: Number of defect images per experiment
        output_dir: Output directory (defaults to OUTPUT_DIR)
    
    Returns:
        List of experiment result dicts
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print("=" * 60)
    print("Custom Dataset Batch Defect Generation")
    print("=" * 60)
    print(f"Dataset: {dataset_root}")
    print(f"Object: {object_name}")
    print(f"Experiments: {num_experiments}")
    print(f"Defects per experiment: {num_bad_per_exp}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Build paths
    good_dir = os.path.join(dataset_root, "good")
    bad_dir = os.path.join(dataset_root, "bad")
    
    # Scan for images
    print(f"\n[1] Scanning images...")
    good_pairs = find_good_pairs(good_dir)
    bad_pairs = find_bad_pairs(bad_dir)
    
    print(f"   Good pairs: {len(good_pairs)}")
    print(f"   Bad pairs: {len(bad_pairs)}")
    
    if len(good_pairs) == 0 or len(bad_pairs) == 0:
        print("[ERROR] No valid image pairs found!")
        print("Please check:")
        print("  - Good images are in: {}/good/".format(dataset_root))
        print("  - Bad images are in: {}/bad/<defect_type>/".format(dataset_root))
        print("  - Masks are named: <image_name>_mask.png")
        return []
    
    # Run experiments
    experiment_log = []
    
    for exp_num in range(1, num_experiments + 1):
        exp_id = f"custom{exp_num:03d}"
        print(f"\n[Experiment {exp_num}/{num_experiments}]")
        
        # Setup temporary directory
        good_exp_dir, bad_exp_dir, temp_dir, good_name, bad_names, selected_bad, original_size = \
            setup_temp_experiment(good_pairs, bad_pairs, num_bad_per_exp, exp_id)
        
        print(f"   Good: {good_name}")
        print(f"   Bad: {bad_names}")
        if original_size:
            print(f"   Original size: {original_size[0]}x{original_size[1]}")
        
        # Run generation
        output_exp_dir = os.path.join(temp_dir, "output")
        success, message = run_experiment(good_exp_dir, bad_exp_dir, output_exp_dir, 
                                         object_name, selected_bad)
        
        if success:
            print(f"[SUCCESS] Experiment completed!")
            
            # Read inference time
            inference_time = read_inference_time(output_exp_dir)
            if inference_time:
                print(f"   Inference time: {inference_time:.2f}s")
            
            # Reorganize output files
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
        
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    # Print summary
    print("\n" + "=" * 60)
    print("Batch Generation Completed!")
    print("=" * 60)
    
    successful = len([e for e in experiment_log if e['status'] == 'SUCCESS'])
    print(f"Success: {successful}/{num_experiments}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    return experiment_log


# ------------------------------ Main Entry Point ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Custom Dataset Batch Defect Generation using 2-GAO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_generate_custom.py --dataset-root my_dataset --object-name "screw"
  python batch_generate_custom.py -d my_dataset -o "product" -n 100 --defect-counts 1,2,3,4
  
See CUSTOM_DATASET.md for detailed documentation.
        """
    )
    
    parser.add_argument("--dataset-root", "-d", type=str, default=DATASET_ROOT,
                       help=f"Path to dataset root directory (default: {DATASET_ROOT})")
    parser.add_argument("--object-name", "-o", type=str, default=OBJECT_NAME,
                       help=f"Object name for prompt generation (default: {OBJECT_NAME})")
    parser.add_argument("--num-experiments", "-n", type=int, default=50,
                       help="Number of experiments per defect count (default: 50)")
    parser.add_argument("--defect-counts", "-c", type=str, default="1",
                       help="Comma-separated defect counts (default: 1)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                       help=f"Output directory (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    # Parse defect counts
    defect_counts = [int(x.strip()) for x in args.defect_counts.split(",")]
    
    print("=" * 70)
    print("2-GAO Custom Dataset Batch Defect Generation")
    print("=" * 70)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Object name: {args.object_name}")
    print(f"Defect counts: {defect_counts}")
    print(f"Experiments per config: {args.num_experiments}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
    
    # Run for each defect count
    for defect_count in defect_counts:
        print(f"\n{'#' * 70}")
        print(f"# Running with {defect_count} defect(s) per image")
        print(f"{'#' * 70}")
        
        output_dir = os.path.join(args.output, f"defect_{defect_count}")
        
        run_batch(
            dataset_root=args.dataset_root,
            object_name=args.object_name,
            num_experiments=args.num_experiments,
            num_bad_per_exp=defect_count,
            output_dir=output_dir
        )
    
    print("\n" + "=" * 70)
    print("All batch generations completed!")
    print("=" * 70)
