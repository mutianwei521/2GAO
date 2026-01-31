"""
VisA Dataset Quick Test Script
Only runs 1 experiment to verify functionality
"""

import os
import sys
import random
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_generate_visa import (
    find_good_pairs_visa, 
    find_bad_pairs_visa,
    setup_temp_experiment_visa,
    run_experiment_visa,
    reorganize_output_files,
    read_inference_time,
    get_visa_categories,
    VISA_ROOT
)


def quick_test_visa(category=None, output_dir="outputs_visa_test"):
    """
    Quick test for VisA dataset generation functionality
    Only runs 1 experiment for verification
    """
    print("=" * 60)
    print("Quick Test - VisA Dataset Defect Generation")
    print("=" * 60)
    
    # Auto-select category
    if category is None:
        available = get_visa_categories()
        if not available:
            print("[ERROR] No VisA categories found!")
            print(f"Please ensure {VISA_ROOT} directory exists with proper structure.")
            return False
        category = random.choice(available)
    
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
    
    if len(good_pairs) == 0:
        print("[ERROR] No good image pairs found!")
        return False
    if len(bad_pairs) == 0:
        print("[ERROR] No bad image pairs found!")
        return False
    
    # Set up experiment
    print(f"\n[2] Setting up experiment...")
    exp_id = "test01"
    
    good_exp_dir, bad_exp_dir, temp_dir, good_name, bad_names, selected_bad, original_size = \
        setup_temp_experiment_visa(good_pairs, bad_pairs, num_bad=1, exp_id=exp_id)
    
    print(f"   Good: {good_name}")
    print(f"   Bad: {bad_names}")
    if original_size:
        print(f"   Original size: {original_size[0]}x{original_size[1]}")
    
    # Run experiment
    print(f"\n[3] Running experiment...")
    output_exp_dir = os.path.join(temp_dir, "output")
    
    success, message = run_experiment_visa(
        good_exp_dir, bad_exp_dir, output_exp_dir,
        category=category,
        selected_bad_info=selected_bad
    )
    
    if success:
        print(f"[SUCCESS] Experiment completed!")
        
        # Read inference time
        inference_time = read_inference_time(output_exp_dir)
        if inference_time:
            print(f"   Inference time: {inference_time:.2f}s")
        
        # Reorganize files (including resize to original dimensions)
        print(f"\n[4] Reorganizing output files (resizing to {original_size[0]}x{original_size[1]})...")
        reorganized = reorganize_output_files(
            output_exp_dir, good_name, bad_names, exp_id, output_dir, original_size
        )
        
        # Display results
        print(f"\n[RESULT] Files saved to {output_dir}/")
        for category_name, file_path in reorganized.items():
            if isinstance(file_path, list):
                print(f"   {category_name}: {len(file_path)} files")
            else:
                print(f"   {category_name}: {os.path.basename(file_path)}")
        
        # Check for attention heatmaps
        heatmap_dir = os.path.join(output_exp_dir, "attention_heatmaps")
        if os.path.exists(heatmap_dir) and os.listdir(heatmap_dir):
            print(f"\n✓ Attention heatmaps found")
        else:
            print(f"\n⚠️ No attention heatmaps found in output")
        
        result = True
    else:
        print(f"[FAILED] {message}")
        result = False
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
        print(f"\n[Cleanup] Removed temp directory")
    except Exception as e:
        print(f"\n[Warning] Could not remove temp directory: {e}")
    
    print("\n" + "=" * 60)
    print("Quick Test Completed!")
    print("=" * 60)
    
    return result


def test_all_categories(output_dir="outputs_visa_test_all"):
    """
    Test all VisA categories (only test 1 image per category)
    """
    print("=" * 60)
    print("Testing All VisA Categories")
    print("=" * 60)
    
    categories = get_visa_categories()
    print(f"Found {len(categories)} categories: {categories}")
    
    results = {}
    for cat in categories:
        print(f"\n--- Testing {cat} ---")
        cat_output = os.path.join(output_dir, cat)
        success = quick_test_visa(category=cat, output_dir=cat_output)
        results[cat] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for cat, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {cat}: {status}")
    
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{len(categories)} passed")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test for VisA dataset generation")
    parser.add_argument("--category", "-c", type=str, default=None,
                       help="VisA category to test (default: random)")
    parser.add_argument("--output", "-o", type=str, default="outputs_visa_test",
                       help="Output directory")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Test all categories")
    
    args = parser.parse_args()
    
    if args.all:
        test_all_categories()
    else:
        quick_test_visa(category=args.category, output_dir=args.output)
