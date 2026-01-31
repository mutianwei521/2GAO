"""
Quick Test Script - Verify Defect Heatmap Output Functionality
Defect heatmap: Overlays defect mask as heatmap on generated image
"""

import os
import random
import shutil
import sys

# Add parent directory to path to import batch_generate
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from batch_generate
from batch_generate import (
    find_good_pairs, 
    find_bad_pairs_from_subfolders,
    setup_temp_experiment,
    run_simple_experiment,
    reorganize_output_files,
    read_inference_time
)

def quick_test():
    print("=" * 60)
    print("Quick Test - Attention Heatmap Verification")
    print("=" * 60)
    
    category = "bottle"
    num_bad = 1
    exp_id = "test01"
    output_dir = "outputs_test"
    
    BASE_DIR = f"images/{category}"
    GOOD_DIR = os.path.join(BASE_DIR, "good")
    BAD_DIR = os.path.join(BASE_DIR, "bad")
    
    print(f"Category: {category}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Find images
    print("\n[1] Scanning images...")
    good_pairs = find_good_pairs(GOOD_DIR)
    bad_pairs = find_bad_pairs_from_subfolders(BAD_DIR)
    
    print(f"   Good pairs: {len(good_pairs)}")
    print(f"   Bad pairs: {len(bad_pairs)}")
    
    if len(good_pairs) == 0 or len(bad_pairs) < num_bad:
        print("[ERROR] Not enough images!")
        return False
    
    # Set up experiment
    print("\n[2] Setting up experiment...")
    good_dir, bad_dir, temp_dir, good_name, bad_names, selected_bad_info = setup_temp_experiment(
        good_pairs, bad_pairs, num_bad, exp_id
    )
    print(f"   Good: {good_name}")
    print(f"   Bad: {bad_names}")
    
    # Run experiment
    print("\n[3] Running experiment (with attention heatmaps enabled)...")
    temp_output_dir = f"temp_output_{exp_id}"
    
    success, message = run_simple_experiment(
        good_dir, bad_dir, temp_output_dir, category, selected_bad_info
    )
    
    if success:
        print("[SUCCESS] Experiment completed!")
        
        inference_time = read_inference_time(temp_output_dir)
        if inference_time:
            print(f"   Inference time: {inference_time:.2f}s")
        
        # Reorganize output files
        print("\n[4] Reorganizing output files...")
        reorganized_files = reorganize_output_files(
            temp_output_dir, good_name, bad_names, exp_id, output_dir
        )
        
        if reorganized_files:
            print(f"\n[RESULT] Files saved to {output_dir}/")
            for category, files in reorganized_files.items():
                if isinstance(files, list):
                    print(f"   {category}: {len(files)} files")
                else:
                    print(f"   {category}: {os.path.basename(files)}")
            
            # Check if heatmaps exist
            if "attention_heatmaps" in reorganized_files:
                print("\n✅ Attention heatmaps successfully generated!")
            else:
                print("\n⚠️ No attention heatmaps found in output")
        
        # Clean up temporary directory
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
    else:
        print(f"[FAILED] {message}")
    
    # Clean up temporary experiment directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("Quick Test Completed!")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    quick_test()
