"""
Quick Test Script - Concrete Dataset Version
Tests defect generation functionality and records inference time
"""

import os
import random
import shutil
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from batch_generate_concrete
from batch_generate_concrete import (
    find_all_good_pairs,
    find_all_bad_pairs,
    setup_temp_experiment_concrete,
    run_experiment_concrete,
    reorganize_output_files,
    read_inference_time
)


def quick_test_concrete(num_defects=1):
    """
    Quick test for Concrete dataset defect generation
    
    Args:
        num_defects: Number of defects to use per generation (1-4)
    """
    print("=" * 60)
    print("Quick Test - Concrete Dataset")
    print("=" * 60)
    
    # Configuration
    CONCRETE_ROOT = "concreteImages"
    category = "concrete"
    exp_id = f"concrete_test_{num_defects}defects"
    output_dir = "outputs_concrete_test"
    
    BASE_DIR = os.path.join(CONCRETE_ROOT, category)
    GOOD_DIR = os.path.join(BASE_DIR, "good")
    BAD_DIR = os.path.join(BASE_DIR, "bad")
    
    print(f"Category: {category}")
    print(f"Defects per image: {num_defects}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Find images
    print("\n[1] Scanning images...")
    good_pairs = find_all_good_pairs(GOOD_DIR)
    bad_pairs = find_all_bad_pairs(BAD_DIR)
    
    print(f"   Good pairs: {len(good_pairs)}")
    print(f"   Bad pairs: {len(bad_pairs)}")
    
    if len(good_pairs) == 0:
        print("[ERROR] No good images found!")
        return False
    
    if len(bad_pairs) < num_defects:
        print(f"[ERROR] Not enough bad images! Need {num_defects}, found {len(bad_pairs)}")
        return False
    
    # Randomly select a good image pair
    selected_good = random.choice(good_pairs)
    
    # Randomly select bad images
    selected_bad = random.sample(bad_pairs, min(num_defects, len(bad_pairs)))
    
    # Set up experiment
    print("\n[2] Setting up experiment...")
    result = setup_temp_experiment_concrete(selected_good, selected_bad, num_defects, exp_id)
    good_dir, bad_dir, temp_dir, good_name, bad_names, _, original_size = result
    
    print(f"   Good: {good_name}")
    print(f"   Bad: {bad_names}")
    if original_size:
        print(f"   Original size: {original_size[0]}x{original_size[1]}")
    
    # Run experiment
    print("\n[3] Running experiment...")
    temp_output_dir = os.path.join(temp_dir, "output")
    
    start_time = time.time()
    success, message = run_experiment_concrete(good_dir, bad_dir, temp_output_dir)
    total_time = time.time() - start_time
    
    if success:
        print("[SUCCESS] Experiment completed!")
        
        # Read inference time
        inference_time = read_inference_time(temp_output_dir)
        if inference_time:
            print(f"\n" + "=" * 40)
            print(f"  INFERENCE TIME: {inference_time:.2f} seconds")
            print(f"  (Total elapsed: {total_time:.2f} seconds)")
            print("=" * 40)
        else:
            print(f"\n   Total elapsed time: {total_time:.2f} seconds")
        
        # Reorganize output files
        print("\n[4] Reorganizing output files...")
        reorganized_files = reorganize_output_files(
            temp_output_dir, good_name, bad_names, exp_id, output_dir, original_size
        )
        
        if reorganized_files:
            print(f"\n[RESULT] Files saved to {output_dir}/")
            for cat, files in reorganized_files.items():
                if isinstance(files, list):
                    print(f"   {cat}: {len(files)} files")
                else:
                    print(f"   {cat}: {os.path.basename(files)}")
            
            # Check if heatmaps exist
            if "defect_heatmaps" in reorganized_files:
                print("\n✅ Defect heatmap successfully generated!")
    else:
        print(f"[FAILED] {message}")
    
    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("Quick Test Completed!")
    print("=" * 60)
    
    return success


def run_inference_time_test():
    """
    Test inference time with different numbers of defects
    """
    print("\n" + "#" * 70)
    print("# INFERENCE TIME TEST - Concrete Dataset")
    print("#" * 70)
    
    # Configuration
    CONCRETE_ROOT = "concreteImages"
    category = "concrete"
    
    BASE_DIR = os.path.join(CONCRETE_ROOT, category)
    GOOD_DIR = os.path.join(BASE_DIR, "good")
    BAD_DIR = os.path.join(BASE_DIR, "bad")
    
    good_pairs = find_all_good_pairs(GOOD_DIR)
    bad_pairs = find_all_bad_pairs(BAD_DIR)
    
    print(f"Good pairs: {len(good_pairs)}")
    print(f"Bad pairs: {len(bad_pairs)}")
    
    results = {}
    
    for num_defects in [1, 2, 3, 4]:
        print(f"\n\n{'='*60}")
        print(f"Testing with {num_defects} defect(s)...")
        print("=" * 60)
        
        if len(good_pairs) == 0 or len(bad_pairs) < num_defects:
            print(f"[SKIP] Not enough images for {num_defects} defects")
            continue
        
        exp_id = f"time_test_{num_defects}"
        
        # Random selection
        selected_good = random.choice(good_pairs)
        selected_bad = random.sample(bad_pairs, num_defects)
        
        result = setup_temp_experiment_concrete(selected_good, selected_bad, num_defects, exp_id)
        good_dir, bad_dir, temp_dir, good_name, bad_names, _, original_size = result
        
        temp_output_dir = os.path.join(temp_dir, "output")
        
        start_time = time.time()
        success, message = run_experiment_concrete(good_dir, bad_dir, temp_output_dir)
        elapsed_time = time.time() - start_time
        
        if success:
            inference_time = read_inference_time(temp_output_dir)
            results[num_defects] = {
                'inference_time': inference_time,
                'elapsed_time': elapsed_time,
                'status': 'SUCCESS'
            }
            print(f"   Inference time: {inference_time:.2f}s" if inference_time else "")
            print(f"   Elapsed time: {elapsed_time:.2f}s")
        else:
            results[num_defects] = {
                'status': 'FAILED',
                'error': message
            }
            print(f"   FAILED: {message[:100]}")
        
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("INFERENCE TIME SUMMARY - Concrete Dataset")
    print("=" * 70)
    print(f"{'Defects':<10} {'Inference Time':<20} {'Status':<10}")
    print("-" * 70)
    
    for num_defects, data in sorted(results.items()):
        if data['status'] == 'SUCCESS':
            inf_time = data.get('inference_time')
            time_str = f"{inf_time:.2f}s" if inf_time else "N/A"
            print(f"{num_defects:<10} {time_str:<20} {data['status']:<10}")
        else:
            print(f"{num_defects:<10} {'N/A':<20} {data['status']:<10}")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Test for Concrete Dataset")
    parser.add_argument("--defects", "-d", type=int, default=1,
                       help="Number of defects per image (default: 1)")
    parser.add_argument("--time-test", "-t", action="store_true",
                       help="Run inference time test for 1-4 defects")
    
    args = parser.parse_args()
    
    if args.time_test:
        run_inference_time_test()
    else:
        quick_test_concrete(num_defects=args.defects)
