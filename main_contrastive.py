"""
对比学习缺陷生成主程序
使用有缺陷图引导无缺陷图生成缺陷
"""

import argparse
import os
import sys
from contrastive_defect_generator import ContrastiveDefectGenerator

def find_image_mask_pairs(directory):
    """
    在指定目录中查找图像和对应的mask文件
    
    Args:
        directory: 目录路径
        
    Returns:
        pairs: [(image_path, mask_path), ...] 图像和mask路径对列表
    """
    if not os.path.exists(directory):
        return []
    
    pairs = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            if not file.lower().endswith('_mask.png'):  # 排除mask文件
                image_files.append(file)
    
    # 为每个图像文件查找对应的mask
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        
        # 尝试不同的mask命名模式
        base_name = os.path.splitext(image_file)[0]
        possible_mask_names = [
            f"{base_name}_mask.png",
            f"{base_name}_mask.jpg",
            f"{base_name}.png" if not image_file.endswith('.png') else f"{base_name}_gt.png",
            f"mask_{base_name}.png"
        ]
        
        mask_path = None
        for mask_name in possible_mask_names:
            potential_mask_path = os.path.join(directory, mask_name)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        
        if mask_path:
            pairs.append((image_path, mask_path))
        else:
            print(f"[WARNING] No mask found for {image_file}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Contrastive Defect Generation")
    
    # 基本参数
    parser.add_argument("--prompt", type=str, default="bottle damaged",
                       help="Text prompt containing product and anomaly keywords")
    parser.add_argument("--good-dir", type=str, default="images/good",
                       help="Directory containing good images and their object masks")
    parser.add_argument("--bad-dir", type=str, default="images/bad", 
                       help="Directory containing bad images and their defect masks")
    parser.add_argument("--output-dir", type=str, default="outputs_contrastive2/2",
                       help="Output directory for generated images")
    
    # 生成参数
    parser.add_argument("--num-inference-steps", type=int, default=100,
                       help="Number of denoising steps")
    parser.add_argument("--r", type=float, default=0.25,
                       help="Preservation ratio (0-1): controls partial forward diffusion")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate for attention optimization")
    parser.add_argument("--num-optimization-steps", type=int, default=25,
                       help="Number of optimization steps per interval")
    parser.add_argument("--optimization-interval", type=int, default=5,
                       help="Interval between optimization steps")
    parser.add_argument("--feather-radius", type=int, default=15,
                       help="Feathering radius for edge smoothing (0 = no feathering)")
    # 特征对齐参数
    parser.add_argument("--enable-feature-alignment", action="store_true", default=True,
                       help="Enable feature alignment to move defects within object mask")
    parser.add_argument("--ioa-threshold", type=float, default=0.5,
                       help="IoA threshold for feature alignment (0.0-1.0)")

    # 可选功能开关
    parser.add_argument("--save-attention-heatmaps", action="store_true", default=False,
                       help="Save attention heatmaps")
    parser.add_argument("--measure-inference-time", action="store_true", default=False,
                       help="Measure and save inference time")
    parser.add_argument("--defect-variation", type=float, default=0.0,
                       help="Add randomness to defect generation (0.0=identical, 0.5=moderate, 1.0=high variation)")
    parser.add_argument("--variation-seed", type=int, default=None,
                       help="Random seed for defect variation (for reproducibility)")
    
    # 注意力热力图参数
    parser.add_argument("--extract-attention", action="store_true", default=False,
                       help="Extract and save attention heatmaps")
    parser.add_argument("--no-extract-attention", dest="extract_attention", action="store_false",
                       help="Disable attention heatmap extraction")

    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-inpainting",
                       help="Stable Diffusion model ID")
    parser.add_argument("--cache-dir", type=str, default="models",
                       help="Model cache directory")
    
    args = parser.parse_args()
    
    print("Starting Contrastive Defect Generation")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Good images directory: {args.good_dir}")
    print(f"Bad images directory: {args.bad_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(args.good_dir):
        print(f"[ERROR] Good images directory '{args.good_dir}' does not exist!")
        print("Please create the directory and add good images with their object masks.")
        sys.exit(1)
    
    if not os.path.exists(args.bad_dir):
        print(f"[ERROR] Bad images directory '{args.bad_dir}' does not exist!")
        print("Please create the directory and add bad images with their defect masks.")
        sys.exit(1)
    
    # 查找图像和mask对
    print("Searching for image-mask pairs...")
    good_pairs = find_image_mask_pairs(args.good_dir)
    bad_pairs = find_image_mask_pairs(args.bad_dir)

    print(f"   Found {len(good_pairs)} good image-mask pairs")
    print(f"   Found {len(bad_pairs)} bad image-mask pairs")
    
    if len(good_pairs) == 0:
        print("[ERROR] No good image-mask pairs found!")
        print("Please ensure your good images directory contains:")
        print("  - Image files (e.g., image1.png)")
        print("  - Corresponding mask files (e.g., image1_mask.png)")
        sys.exit(1)
    
    if len(bad_pairs) == 0:
        print("[ERROR] No bad image-mask pairs found!")
        print("Please ensure your bad images directory contains:")
        print("  - Image files (e.g., defect1.png)")
        print("  - Corresponding mask files (e.g., defect1_mask.png)")
        sys.exit(1)
    
    # 选择第一对无缺陷图像
    good_image_path, good_mask_path = good_pairs[0]

    # 使用所有缺陷图像对
    bad_image_paths = [pair[0] for pair in bad_pairs]
    bad_mask_paths = [pair[1] for pair in bad_pairs]

    print(f"[INFO] Using good image: {os.path.basename(good_image_path)}")
    print(f"[INFO] Using good mask: {os.path.basename(good_mask_path)}")
    print(f"[INFO] Using {len(bad_image_paths)} bad images:")
    for i, (img_path, mask_path) in enumerate(zip(bad_image_paths, bad_mask_paths)):
        print(f"   {i+1}. Image: {os.path.basename(img_path)}, Mask: {os.path.basename(mask_path)}")
    
    try:
        # 初始化生成器
        print("\n[INIT] Initializing Contrastive Defect Generator...")
        generator = ContrastiveDefectGenerator(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.cache_dir
        )
        
        # 提取缺陷类型和生成智能prompt
        defect_types = []
        selected_bad_info = []

        for bad_path in bad_image_paths:
            # 获取父目录名作为缺陷类型
            parent_dir = os.path.basename(os.path.dirname(bad_path))
            filename = os.path.splitext(os.path.basename(bad_path))[0]

            if parent_dir not in defect_types:
                defect_types.append(parent_dir)

            selected_bad_info.append({
                'subfolder': parent_dir,
                'filename': filename
            })

        print(f"[DETECT] Detected defect types: {defect_types}")

        # 生成智能prompt
        try:
            from smart_prompt_generator import generate_smart_prompt, generate_individual_prompts

            # 从args.prompt中提取category（假设格式为"category ..."）
            category = args.prompt.split()[0] if args.prompt else "unknown"

            # 生成组合prompt和单独prompts
            combined_prompt, _, _ = generate_smart_prompt(category, selected_bad_info)
            individual_prompt_info = generate_individual_prompts(category, selected_bad_info)
            individual_prompts = [info['prompt'] for info in individual_prompt_info]

            print(f"[PROMPT] Generated smart prompts:")
            print(f"   Combined: {combined_prompt}")
            print(f"   Individual: {individual_prompts}")

            # 使用生成的prompt
            final_prompt = combined_prompt

        except Exception as e:
            print(f"[WARNING] Could not generate smart prompt: {e}")
            print(f"   Using original prompt: {args.prompt}")
            final_prompt = args.prompt
            individual_prompts = None

        # 生成缺陷图像
        print("\n[GENERATE] Generating contrastive defect image...")
        file_paths = generator.generate_contrastive_defect(
            good_image_path=good_image_path,
            good_mask_path=good_mask_path,
            bad_image_paths=bad_image_paths,
            bad_mask_paths=bad_mask_paths,
            prompt=final_prompt,
            individual_prompts=individual_prompts,
            num_inference_steps=args.num_inference_steps,
            r=args.r,
            learning_rate=args.learning_rate,
            num_optimization_steps=args.num_optimization_steps,
            optimization_interval=args.optimization_interval,
            feather_radius=args.feather_radius,
            defect_variation=args.defect_variation,
            variation_seed=args.variation_seed,
            output_dir=args.output_dir,
            extract_attention=args.save_attention_heatmaps,
            defect_types=defect_types,
            enable_feature_alignment=args.enable_feature_alignment,
            ioa_threshold=args.ioa_threshold,
            measure_inference_time=args.measure_inference_time
        )
        
        # 显示结果
        print("\n[SUCCESS] Generation completed successfully!")
        print("[FILES] Generated files:")
        for key, path in file_paths.items():
            if isinstance(path, str):
                # 显示完整路径，如果太长则换行显示
                if len(path) > 80:
                    print(f"   {key}:")
                    print(f"      {path}")
                else:
                    print(f"   {key}: {path}")
            elif isinstance(path, list):
                print(f"   {key}: {len(path)} files")
                for i, p in enumerate(path[:3]):  # 只显示前3个
                    if len(p) > 80:
                        print(f"      {i+1}. {os.path.basename(p)}")
                        print(f"         Full path: {p}")
                    else:
                        print(f"      {i+1}. {p}")
                if len(path) > 3:
                    print(f"      ... and {len(path) - 3} more files")

        # 显示输出目录的完整路径
        full_output_path = os.path.abspath(args.output_dir)
        print(f"\n[COMPLETE] Check the results in:")
        print(f"   {full_output_path}")

        # 如果路径很长，也显示相对路径
        if len(full_output_path) > 60:
            print(f"   (Relative: {args.output_dir})")
        
    except Exception as e:
        print(f"\n[ERROR] Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
