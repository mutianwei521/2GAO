"""
对比学习缺陷生成器
使用有缺陷图引导无缺陷图生成缺陷的对比学习方法
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import os
import random
from typing import List, Dict, Tuple
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
import cv2
from attention_heatmap_extractor import extract_attention_heatmaps

class ContrastiveDefectGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", 
                 device: str = "cuda", cache_dir: str = "./models"):
        """
        初始化对比学习缺陷生成器
        
        Args:
            model_id: Stable Diffusion模型ID
            device: 计算设备
            cache_dir: 模型缓存目录
        """
        self.device = device
        self.model_id = model_id
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        # 初始化Stable Diffusion管道
        print("[INIT] Loading Stable Diffusion model...")

        # 添加参数来处理模型文件格式问题
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*safetensors.*")

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=cache_dir,
                local_files_only=False,
                use_safetensors=False  # 明确指定使用 .bin 文件
            ).to(device)
        
        # 设置调度器
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # 获取组件
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        
        # 冻结所有模型参数
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        print("[SUCCESS] Model loaded successfully!")
    
    def load_image_and_mask(self, image_path: str, mask_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载图像和mask

        Args:
            image_path: 图像路径
            mask_path: mask路径

        Returns:
            image_tensor: 图像张量 [1, 3, H, W]
            mask_tensor: mask张量 [1, 1, H, W]
        """
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.dtype).to(self.device)

        # 加载mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512))
        mask_array = np.array(mask) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(self.dtype).to(self.device)

        return image_tensor, mask_tensor
    
    def parse_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """
        解析prompt，提取product和anomaly tokens
        
        Args:
            prompt: 输入prompt，如 "nutshell crack scratches"
            
        Returns:
            product_token: 产品token
            anomaly_tokens: 异常token列表
        """
        # 预定义的产品词汇
        product_keywords = [
            "nutshell", "nut", "bottle", "cable", "capsule", "hazelnut", 
            "metal", "pill", "screw", "toothbrush", "transistor", "zipper",
            "carpet", "grid", "leather", "tile", "wood"
        ]
        
        # 预定义的异常词汇及其变体
        # 预定义的异常词汇及其变体，包括MVTEC的所有缺陷类型
        anomaly_keywords = {
            # bottle
            "broken_large": ["broken_large", "broken", "large_break", "big_break", "major_break"],
            "broken_small": ["broken_small", "small_break", "minor_break", "tiny_break"],
            "contamination": ["contamination", "contaminated", "contaminant", "dirty", "impurity"],

            # cable
            "bent_wire": ["bent_wire", "bent", "wire_bent", "curved_wire", "twisted_wire"],
            "cable_swap": ["cable_swap", "swapped", "wrong_cable", "misplaced_cable"],
            "cut_inner_insulation": ["cut_inner_insulation", "inner_cut", "insulation_cut", "inner_damage"],
            "cut_outer_insulation": ["cut_outer_insulation", "outer_cut", "outer_damage", "external_cut"],
            "missing_cable": ["missing_cable", "missing", "absent_cable", "no_cable"],
            "missing_wire": ["missing_wire", "wire_missing", "absent_wire", "no_wire"],
            "poke_insulation": ["poke_insulation", "poked", "punctured", "pierced"],

            # capsule
            "crack": ["crack", "cracks", "cracked", "cracking", "fracture"],
            "faulty_imprint": ["faulty_imprint", "bad_imprint", "wrong_imprint", "defective_print"],
            "poke": ["poke", "poked", "puncture", "pierced", "hole"],
            "scratch": ["scratch", "scratches", "scratched", "scratching", "scrape"],
            "squeeze": ["squeeze", "squeezed", "compressed", "deformed", "crushed"],

            # carpet
            "color": ["color", "discolored", "color_change", "faded", "stained"],
            "cut": ["cut", "cuts", "cutting", "slice", "torn"],
            "hole": ["hole", "holes", "holed", "puncture", "opening"],
            "metal_contamination": ["metal_contamination", "metal", "metallic", "foreign_metal"],
            "thread": ["thread", "threads", "loose_thread", "thread_pull", "fiber"],

            # grid
            "bent": ["bent", "bending", "curved", "warped", "twisted"],
            "broken": ["broken", "break", "breaks", "fractured", "damaged"],
            "glue": ["glue", "adhesive", "glued", "sticky", "residue"],

            # hazelnut
            "print": ["print", "printed", "marking", "imprint", "stamp"],

            # leather
            "fold": ["fold", "folded", "crease", "wrinkle", "bend"],

            # metal_nut
            "flip": ["flip", "flipped", "inverted", "upside_down", "reversed"],

            # pill
            "pill_type": ["pill_type", "wrong_type", "different_pill", "incorrect_pill"],

            # screw
            "manipulated_front": ["manipulated_front", "front_damage", "front_altered", "modified_front"],
            "scratch_head": ["scratch_head", "head_scratch", "head_damage", "scratched_head"],
            "scratch_neck": ["scratch_neck", "neck_scratch", "neck_damage", "scratched_neck"],
            "thread_side": ["thread_side", "side_thread", "lateral_thread", "side_damage"],
            "thread_top": ["thread_top", "top_thread", "upper_thread", "top_damage"],

            # tile
            "glue_strip": ["glue_strip", "adhesive_strip", "glue_line", "sticky_strip"],
            "gray_stroke": ["gray_stroke", "grey_stroke", "gray_line", "stroke"],
            "oil": ["oil", "oily", "grease", "lubricant", "stain"],
            "rough": ["rough", "roughness", "coarse", "uneven", "textured"],

            # toothbrush
            "defective": ["defective", "defect", "faulty", "broken", "damaged"],

            # transistor
            "bent_lead": ["bent_lead", "lead_bent", "curved_lead", "twisted_lead"],
            "cut_lead": ["cut_lead", "lead_cut", "severed_lead", "broken_lead"],
            "damaged_case": ["damaged_case", "case_damage", "broken_case", "cracked_case"],
            "misplaced": ["misplaced", "displaced", "wrong_position", "shifted"],

            # wood
            "liquid": ["liquid", "wet", "moisture", "water", "fluid"],

            # zipper
            "broken_teeth": ["broken_teeth", "teeth_broken", "damaged_teeth", "missing_teeth"],
            "fabric_border": ["fabric_border", "border_fabric", "edge_fabric", "fabric_edge"],
            "fabric_interior": ["fabric_interior", "interior_fabric", "inner_fabric", "internal_fabric"],
            "split_teeth": ["split_teeth", "teeth_split", "separated_teeth", "divided_teeth"],
            "squeezed_teeth": ["squeezed_teeth", "teeth_squeezed", "compressed_teeth", "crushed_teeth"],

            # 通用缺陷词汇（combined -> damage）
            "damage": ["damage", "damaged", "defect", "defective", "fault", "faulty"],
            "combined": ["damage", "damaged", "defect", "defective", "fault", "faulty"]  # combined映射到damage
        }
        
        words = prompt.lower().split()
        
        # 找到产品token
        product_token = None
        for word in words:
            if word in product_keywords:
                product_token = word
                break
        
        if product_token is None:
            product_token = words[0] if words else "object"
        
        # 找到异常tokens
        anomaly_tokens = []
        for word in words:
            for base_anomaly, variants in anomaly_keywords.items():
                if word in variants:
                    if base_anomaly not in anomaly_tokens:
                        anomaly_tokens.append(base_anomaly)
                    break
        
        if not anomaly_tokens:
            # 如果没有找到预定义的异常词汇，使用除产品词汇外的其他词汇
            anomaly_tokens = [word for word in words if word != product_token]
        
        return product_token, anomaly_tokens
    
    def encode_text(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """
        编码文本并返回token indices
        
        Args:
            text: 输入文本
            
        Returns:
            text_embeddings: 文本嵌入
            token_indices: token索引列表
        """
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 获取token indices
        token_ids = text_inputs.input_ids[0].tolist()
        
        # 找到实际文本token的索引（排除特殊token）
        token_indices = []
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        for i, token in enumerate(tokens):
            if token not in ['<|startoftext|>', '<|endoftext|>', '<|padding|>'] and not token.startswith('<'):
                token_indices.append(i)
        
        # 编码
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
            # 确保文本嵌入的数据类型与模型一致
            text_embeddings = text_embeddings.to(self.dtype)

        return text_embeddings, token_indices
    
    def encode_images(self, good_image: torch.Tensor, bad_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像到潜在空间
        
        Args:
            good_image: 无缺陷图像
            bad_image: 有缺陷图像
            
        Returns:
            good_latents: 无缺陷图像潜在表示
            bad_latents: 有缺陷图像潜在表示
        """
        with torch.no_grad():
            # 编码无缺陷图像
            good_latents = self.vae.encode(good_image * 2 - 1).latent_dist.sample()
            good_latents = good_latents * self.vae.config.scaling_factor
            
            # 编码有缺陷图像
            bad_latents = self.vae.encode(bad_image * 2 - 1).latent_dist.sample()
            bad_latents = bad_latents * self.vae.config.scaling_factor
        
        return good_latents, bad_latents
    
    def add_noise(self, latents: torch.Tensor, timestep: int, noise: torch.Tensor) -> torch.Tensor:
        """
        添加噪声到潜在表示

        Args:
            latents: 潜在表示
            timestep: 时间步
            noise: 噪声

        Returns:
            noisy_latents: 加噪后的潜在表示
        """
        # 获取噪声调度参数
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep].to(latents.device, latents.dtype)
        beta_prod_t = (1 - alpha_prod_t).to(latents.device, latents.dtype)

        # 确保噪声与latents类型匹配
        noise = noise.to(latents.dtype)

        # 添加噪声
        noisy_latents = (alpha_prod_t ** 0.5) * latents + (beta_prod_t ** 0.5) * noise

        return noisy_latents
    
    def extract_attention_maps_from_unet(self, latents: torch.Tensor, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        通过UNet前向传播提取注意力图

        Args:
            latents: 潜在表示
            text_embeddings: 文本嵌入

        Returns:
            attention_maps: 模拟的注意力图字典
        """
        # 这里我们创建一个简化的注意力图提取
        # 在实际应用中，你可能需要修改UNet来直接返回注意力权重

        attention_maps = {}

        # 获取latents的空间维度
        _, _, h, w = latents.shape
        num_tokens = text_embeddings.shape[1]

        # 创建模拟的注意力图
        # 在实际实现中，这应该从UNet的交叉注意力层获取
        for layer_idx in range(4):  # 假设有4个注意力层
            # 创建随机注意力图作为占位符
            # 实际应用中应该从UNet内部获取真实的注意力权重
            attention_map = torch.randn(2, h, w, num_tokens, device=latents.device)
            attention_map = F.softmax(attention_map.view(2, h*w, num_tokens), dim=1).view(2, h, w, num_tokens)
            attention_maps[f"layer_{layer_idx}"] = attention_map

        return attention_maps
    
    def compute_contrastive_loss(self,
                                attention_maps: Dict[str, torch.Tensor],
                                product_token_indices: List[int],
                                anomaly_token_indices: List[int],
                                good_mask: torch.Tensor,
                                bad_masks_list: List[torch.Tensor],
                                current_defect_idx: int = 0) -> torch.Tensor:
        """
        计算对比损失

        Args:
            attention_maps: 注意力图
            product_token_indices: 产品token索引
            anomaly_token_indices: 异常token索引
            good_mask: 无缺陷图像的物体mask
            bad_masks_list: 有缺陷图像的缺陷mask列表
            current_defect_idx: 当前优化的缺陷图像索引

        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0
        num_maps = 0
        
        for _, attention_map in attention_maps.items():
            if attention_map.dim() != 4:
                continue
                
            _, h, w, num_tokens = attention_map.shape
            
            # 调整mask尺寸到注意力图尺寸
            good_mask_resized = F.interpolate(good_mask.float(), size=(h, w), mode='nearest').squeeze(1)

            # 使用当前优化的缺陷mask
            current_bad_mask = bad_masks_list[current_defect_idx]
            bad_mask_resized = F.interpolate(current_bad_mask.float(), size=(h, w), mode='nearest').squeeze(1)
            
            # 第一部分损失：产品token在无缺陷图像物体位置的注意力
            loss1 = 0.0
            for token_idx in product_token_indices:
                if token_idx < num_tokens:
                    # 获取产品token的注意力
                    product_attention = attention_map[0, :, :, token_idx]  # 无缺陷图像（batch中第一个）
                    
                    # Softmax归一化
                    product_attention_flat = product_attention.view(-1)
                    product_attention_norm = F.softmax(product_attention_flat, dim=0).view(h, w)
                    
                    # 与物体mask做点积并求和
                    attention_score = torch.sum(product_attention_norm * good_mask_resized[0])
                    loss1 += 1 - attention_score
            
            # 第二部分损失：异常token在有缺陷图像缺陷位置的注意力
            loss2 = 0.0
            for token_idx in anomaly_token_indices:
                if token_idx < num_tokens:
                    # 获取异常token的注意力
                    anomaly_attention = attention_map[1, :, :, token_idx]  # 有缺陷图像（batch中第二个）
                    
                    # Softmax归一化
                    anomaly_attention_flat = anomaly_attention.view(-1)
                    anomaly_attention_norm = F.softmax(anomaly_attention_flat, dim=0).view(h, w)
                    
                    # 与缺陷mask做点积并求和
                    attention_score = torch.sum(anomaly_attention_norm * bad_mask_resized[0])
                    loss2 += 1 - attention_score
            
            # 组合两部分损失
            combined_loss = loss1 + loss2
            total_loss += combined_loss
            num_maps += 1
        
        if num_maps > 0:
            total_loss = total_loss / num_maps
        
        return total_loss

    def transfer_defect_features(self,
                                good_latents: torch.Tensor,
                                bad_latents: torch.Tensor,
                                bad_mask: torch.Tensor) -> torch.Tensor:
        """
        将缺陷特征从有缺陷图像转移到无缺陷图像

        Args:
            good_latents: 无缺陷图像潜在表示
            bad_latents: 有缺陷图像潜在表示
            bad_mask: 缺陷位置mask

        Returns:
            updated_latents: 更新后的潜在表示
        """
        # 调整mask尺寸到潜在空间
        _, _, h, w = good_latents.shape
        bad_mask_latent = F.interpolate(bad_mask.float(), size=(h, w), mode='nearest').to(good_latents.dtype)

        # 将缺陷区域的特征从bad_latents转移到good_latents
        updated_latents = good_latents * (1 - bad_mask_latent) + bad_latents * bad_mask_latent

        return updated_latents

    def find_random_placement(self,
                            defect_mask: torch.Tensor,
                            object_mask: torch.Tensor,
                            placement_range: float = 1.0) -> Tuple[torch.Tensor, int, int]:
        """
        在物体mask内找到一个随机位置放置缺陷

        Args:
            defect_mask: 缺陷mask [1, 1, H, W]
            object_mask: 物体mask [1, 1, H, W]
            placement_range: 放置范围倍数 (0.5=小范围, 1.0=全范围, 2.0=大范围)

        Returns:
            new_mask: 新位置的缺陷mask
            offset_y, offset_x: 偏移量（用于显示）
        """
        # 转换为numpy进行处理
        defect_np = defect_mask.cpu().squeeze().numpy()
        object_np = object_mask.cpu().squeeze().numpy()

        h, w = defect_np.shape

        # 获取缺陷和物体的坐标
        defect_coords = np.where(defect_np > 0.5)
        object_coords = np.where(object_np > 0.5)

        if len(defect_coords[0]) == 0 or len(object_coords[0]) == 0:
            return defect_mask.clone(), 0, 0

        # 获取缺陷的边界框
        defect_min_y, defect_max_y = defect_coords[0].min(), defect_coords[0].max()
        defect_min_x, defect_max_x = defect_coords[1].min(), defect_coords[1].max()
        defect_h = defect_max_y - defect_min_y + 1
        defect_w = defect_max_x - defect_min_x + 1

        # 获取物体的边界框
        obj_min_y, obj_max_y = object_coords[0].min(), object_coords[0].max()
        obj_min_x, obj_max_x = object_coords[1].min(), object_coords[1].max()

        # 计算可放置的范围（确保缺陷完全在物体内）
        available_h = obj_max_y - obj_min_y + 1 - defect_h
        available_w = obj_max_x - obj_min_x + 1 - defect_w

        if available_h <= 0 or available_w <= 0:
            # 缺陷太大，无法完全放入物体内，返回原位置
            print(f"     Defect too large for object, keeping original position")
            return defect_mask.clone(), 0, 0

        # 根据placement_range调整可用范围
        if placement_range != 1.0:
            # 计算范围中心
            center_h = available_h // 2
            center_w = available_w // 2

            # 调整范围大小
            adjusted_h = max(1, int(available_h * placement_range))
            adjusted_w = max(1, int(available_w * placement_range))

            # 计算新的起始位置（保持居中）
            start_h = max(0, center_h - adjusted_h // 2)
            start_w = max(0, center_w - adjusted_w // 2)

            # 确保不超出边界
            end_h = min(available_h, start_h + adjusted_h)
            end_w = min(available_w, start_w + adjusted_w)

            available_h = end_h - start_h
            available_w = end_w - start_w

            # 随机选择新的左上角位置
            new_top_y = obj_min_y + start_h + random.randint(0, max(0, available_h - 1))
            new_top_x = obj_min_x + start_w + random.randint(0, max(0, available_w - 1))
        else:
            # 使用全范围
            new_top_y = obj_min_y + random.randint(0, available_h)
            new_top_x = obj_min_x + random.randint(0, available_w)

        # 计算偏移量
        offset_y = new_top_y - defect_min_y
        offset_x = new_top_x - defect_min_x

        # 创建新的mask
        new_mask_np = np.zeros_like(defect_np)

        # 将缺陷复制到新位置
        for i, (y, x) in enumerate(zip(defect_coords[0], defect_coords[1])):
            new_y = y + offset_y
            new_x = x + offset_x

            # 确保新坐标在范围内
            if 0 <= new_y < h and 0 <= new_x < w:
                new_mask_np[new_y, new_x] = defect_np[y, x]

        # 转换回tensor
        new_mask = torch.from_numpy(new_mask_np).unsqueeze(0).unsqueeze(0).to(defect_mask.device)

        return new_mask, offset_y, offset_x

    def apply_defect_with_placement(self,
                                  good_latents: torch.Tensor,
                                  bad_latents_list: List[torch.Tensor],
                                  bad_masks_list: List[torch.Tensor],
                                  good_mask: torch.Tensor,
                                  alignment_info: List[Dict] = None,
                                  current_step: int = 0,
                                  total_steps: int = 1) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        将多个缺陷应用到无缺陷图像上，支持特征对齐

        Args:
            good_latents: 无缺陷图像潜在表示
            bad_latents_list: 缺陷图像潜在表示列表
            bad_masks_list: 缺陷mask列表
            good_mask: 物体mask
            alignment_info: 特征对齐信息
            current_step: 当前优化步骤
            total_steps: 总优化步骤

        Returns:
            updated_latents: 更新后的潜在表示
            actual_masks: 实际使用的mask列表
        """
        updated_latents = good_latents.clone()
        _, _, h, w = good_latents.shape

        # 调整物体mask到潜在空间
        good_mask_latent = F.interpolate(good_mask.float(), size=(h, w), mode='nearest')

        # 记录实际使用的mask
        actual_masks = []

        for i, (bad_latents, bad_mask) in enumerate(zip(bad_latents_list, bad_masks_list)):
            print(f"   Applying defect {i+1}/{len(bad_latents_list)}...")

            # 获取当前缺陷的对齐信息
            if alignment_info and i < len(alignment_info):
                align_info = alignment_info[i]
                if align_info['needs_alignment']:
                    # 计算当前步骤的插值位置
                    progress = current_step / max(total_steps - 1, 1)  # 0到1的进度
                    current_mask = self.interpolate_mask_position(
                        align_info['original_mask'],
                        align_info['target_position'],
                        progress
                    )
                    print(f"     Feature alignment progress: {progress:.2f}")
                else:
                    current_mask = bad_mask
                    print(f"     Using original position (IoA: {align_info['ioa']:.3f})")
            else:
                current_mask = bad_mask
                print(f"     Using original position (no alignment)")

            # 调整mask到潜在空间尺寸
            current_mask_latent = F.interpolate(current_mask.float(), size=(h, w), mode='nearest')

            # 确保缺陷只在物体区域内
            effective_mask = (current_mask_latent * good_mask_latent).to(good_latents.dtype)

            # 应用缺陷到好图像
            updated_latents = updated_latents * (1 - effective_mask) + bad_latents * effective_mask
            actual_masks.append(current_mask)

        return updated_latents, actual_masks

    def add_defect_variation(self,
                           latents: torch.Tensor,
                           variation_strength: float = 0.0) -> torch.Tensor:
        """
        为潜在表示添加随机变化

        Args:
            latents: 输入的潜在表示
            variation_strength: 变化强度 (0.0-1.0)

        Returns:
            varied_latents: 添加变化后的潜在表示
        """
        if variation_strength <= 0.0:
            return latents

        # 生成随机噪声
        noise = torch.randn_like(latents) * variation_strength * 0.1

        # 添加随机变化
        varied_latents = latents + noise

        return varied_latents

    def apply_defect_variation_to_masks(self,
                                      masks: List[torch.Tensor],
                                      variation_strength: float = 0.0) -> List[torch.Tensor]:
        """
        为缺陷mask添加随机变化

        Args:
            masks: 缺陷mask列表
            variation_strength: 变化强度 (0.0-1.0)

        Returns:
            varied_masks: 变化后的mask列表
        """
        if variation_strength <= 0.0:
            return masks

        varied_masks = []
        for mask in masks:
            if variation_strength > 0.0:
                # 对mask进行轻微的形态学变化
                mask_np = mask.cpu().squeeze().numpy()

                # 确保数据类型为uint8
                mask_np = (mask_np * 255).astype(np.uint8)

                # 随机膨胀或腐蚀
                if random.random() < 0.5:
                    # 膨胀（扩大缺陷）
                    kernel_size = int(3 + variation_strength * 5)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    # 腐蚀（缩小缺陷）
                    kernel_size = int(2 + variation_strength * 3)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)

                # 转换回float并归一化
                mask_np = mask_np.astype(np.float32) / 255.0

                # 添加随机噪声到mask边缘
                if variation_strength > 0.3:
                    noise = np.random.random(mask_np.shape) * variation_strength * 0.3
                    mask_np = np.clip(mask_np + noise, 0, 1)

                # 转换回tensor
                varied_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(mask.device)
                varied_masks.append(varied_mask)
            else:
                varied_masks.append(mask)

        return varied_masks

    def create_feathered_mask(self, mask: np.ndarray, feather_radius: int = 10) -> np.ndarray:
        """
        创建羽化边缘的mask

        Args:
            mask: 原始mask (0-255)
            feather_radius: 羽化半径

        Returns:
            feathered_mask: 羽化后的mask (0-1)
        """
        # 确保mask是0-255范围
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # 使用高斯模糊进行羽化
        feathered = cv2.GaussianBlur(mask, (feather_radius*2+1, feather_radius*2+1), feather_radius/3)

        # 归一化到0-1范围
        feathered = feathered.astype(np.float32) / 255.0

        return feathered

    def blend_images_with_feathered_mask(self,
                                       original_image: np.ndarray,
                                       generated_image: np.ndarray,
                                       mask: np.ndarray,
                                       feather_radius: int = 10) -> np.ndarray:
        """
        使用羽化mask将生成图像的缺陷部分与原始图像合并

        Args:
            original_image: 原始图像 (H, W, 3) 范围0-255
            generated_image: 生成图像 (H, W, 3) 范围0-255
            mask: 缺陷mask (H, W) 范围0-255，白色为缺陷区域
            feather_radius: 羽化半径

        Returns:
            blended_image: 合并后的图像 (H, W, 3) 范围0-255
        """
        # 创建羽化mask
        feathered_mask = self.create_feathered_mask(mask, feather_radius)

        # 确保图像数据类型正确
        original_image = original_image.astype(np.float32)
        generated_image = generated_image.astype(np.float32)

        # 确保mask形状正确
        if len(feathered_mask.shape) == 4:  # (1, 1, H, W)
            feathered_mask = feathered_mask.squeeze()  # (H, W)
        elif len(feathered_mask.shape) == 3:  # (1, H, W) or (H, W, 1)
            feathered_mask = feathered_mask.squeeze()  # (H, W)

        # 调整mask尺寸以匹配图像
        if feathered_mask.shape != original_image.shape[:2]:
            import cv2
            feathered_mask = cv2.resize(feathered_mask, (original_image.shape[1], original_image.shape[0]))

        # 扩展mask维度以匹配图像通道
        if len(feathered_mask.shape) == 2:
            feathered_mask = np.expand_dims(feathered_mask, axis=2)
        feathered_mask = np.repeat(feathered_mask, 3, axis=2)

        # 使用羽化mask进行混合
        # mask=1的地方使用生成图像，mask=0的地方使用原始图像
        blended_image = original_image * (1 - feathered_mask) + generated_image * feathered_mask

        # 转换回uint8
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

        return blended_image

    def create_comparison_grid(self,
                             original_image: np.ndarray,
                             generated_image: np.ndarray,
                             blended_image: np.ndarray,
                             reference_bad_image: np.ndarray,
                             defect_mask: np.ndarray,
                             non_feathered_image: np.ndarray = None) -> np.ndarray:
        """
        创建对比网格图像

        Args:
            original_image: 原始无缺陷图像
            generated_image: 生成的缺陷图像
            blended_image: 羽化合并图像
            reference_bad_image: 参考缺陷图像
            defect_mask: 缺陷mask
            non_feathered_image: 无羽化合并图像

        Returns:
            comparison_grid: 对比网格图像
        """
        # 确保所有图像尺寸一致
        h, w = original_image.shape[:2]

        # 检查图像尺寸是否有效
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: h={h}, w={w}")

        # 调整所有图像到相同尺寸
        images = [original_image, generated_image, blended_image, reference_bad_image]
        resized_images = []
        for i, img in enumerate(images):
            if img is None:
                raise ValueError(f"Image {i} is None")
            if len(img.shape) < 2:
                raise ValueError(f"Image {i} has invalid shape: {img.shape}")
            if img.shape[:2] != (h, w):
                if img.shape[0] <= 0 or img.shape[1] <= 0:
                    raise ValueError(f"Image {i} has invalid dimensions: {img.shape}")
                img = cv2.resize(img, (w, h))
            resized_images.append(img)

        # 创建mask可视化（转换为3通道）
        if defect_mask is None:
            raise ValueError("defect_mask is None")
        if len(defect_mask.shape) < 2:
            raise ValueError(f"defect_mask has invalid shape: {defect_mask.shape}")

        # 确保defect_mask是2D数组
        if len(defect_mask.shape) == 4:  # (1, 1, H, W)
            defect_mask_2d = defect_mask.squeeze()  # (H, W)
        elif len(defect_mask.shape) == 3:  # (1, H, W) or (H, W, 1)
            defect_mask_2d = defect_mask.squeeze()  # (H, W)
        else:
            defect_mask_2d = defect_mask  # Already 2D

        mask_vis = np.stack([defect_mask_2d, defect_mask_2d, defect_mask_2d], axis=2)
        if mask_vis.max() <= 1.0:
            mask_vis = (mask_vis * 255).astype(np.uint8)

        # 检查mask尺寸并调整
        if mask_vis.shape[:2] != (h, w):
            mask_vis = cv2.resize(mask_vis, (w, h))

        # 创建网格布局
        if non_feathered_image is not None:
            # 如果有无羽化图像，创建2x4网格
            # 第一行：原始图像、生成图像、无羽化合并、羽化合并
            # 第二行：参考缺陷图像、缺陷mask、空白、空白
            non_feathered_resized = cv2.resize(non_feathered_image, (w, h)) if non_feathered_image.shape[:2] != (h, w) else non_feathered_image

            top_row = np.hstack([resized_images[0], resized_images[1], non_feathered_resized, resized_images[2]])

            # 创建空白图像
            blank = np.ones_like(resized_images[0]) * 255
            bottom_row = np.hstack([resized_images[3], mask_vis, blank, blank])
        else:
            # 原始2x3网格
            # 第一行：原始图像、生成图像、羽化合并图像
            # 第二行：参考缺陷图像、缺陷mask、空白
            top_row = np.hstack([resized_images[0], resized_images[1], resized_images[2]])

            # 创建空白图像
            blank = np.ones_like(resized_images[0]) * 255
            bottom_row = np.hstack([resized_images[3], mask_vis, blank])

        # 组合成完整网格
        comparison_grid = np.vstack([top_row, bottom_row])

        # 添加标签区域（简单的白色条带）
        label_height = 30
        grid_h, grid_w = comparison_grid.shape[:2]

        # 为每个子图添加标签背景
        labeled_grid = np.ones((grid_h + label_height * 2, grid_w, 3), dtype=np.uint8) * 255
        labeled_grid[label_height:label_height + grid_h] = comparison_grid

        return labeled_grid

    def calculate_feature_alignment(self, bad_masks: List[torch.Tensor], good_mask: torch.Tensor, ioa_threshold: float):
        """
        计算特征对齐信息

        Args:
            bad_masks: 缺陷mask列表
            good_mask: 正常图物体mask
            ioa_threshold: IoA阈值

        Returns:
            alignment_info: 对齐信息列表
        """
        alignment_info = []

        for i, bad_mask in enumerate(bad_masks):
            # 计算交集
            intersection = torch.logical_and(bad_mask > 0.5, good_mask > 0.5)
            intersection_pixels = torch.sum(intersection).item()

            # 计算缺陷mask的总像素数
            bad_mask_pixels = torch.sum(bad_mask > 0.5).item()

            # 计算IoA (Intersection over Area)
            ioa = intersection_pixels / max(bad_mask_pixels, 1)  # 避免除零

            # 判断是否需要对齐
            needs_alignment = ioa < ioa_threshold

            target_position = None
            if needs_alignment:
                # 在正常图mask区域内随机选择目标位置
                target_position = self.find_target_position(bad_mask, good_mask, ioa_threshold)

            alignment_info.append({
                'ioa': ioa,
                'needs_alignment': needs_alignment,
                'target_position': target_position,
                'original_mask': bad_mask.clone()
            })

        return alignment_info

    def find_target_position(self, bad_mask: torch.Tensor, good_mask: torch.Tensor, target_ioa: float):
        """
        在正常图mask区域内找到目标位置

        Args:
            bad_mask: 缺陷mask
            good_mask: 正常图物体mask
            target_ioa: 目标IoA值

        Returns:
            target_position: 目标位置 (y, x)
        """
        import random

        # 获取缺陷mask的尺寸
        bad_mask_binary = bad_mask > 0.5

        # 处理4D tensor: [1, 1, H, W] -> 只取最后两个维度
        if bad_mask_binary.dim() == 4:
            bad_mask_binary = bad_mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif bad_mask_binary.dim() == 3:
            bad_mask_binary = bad_mask_binary.squeeze(0)  # [H, W]

        bad_h, bad_w = torch.where(bad_mask_binary)

        if len(bad_h) == 0:
            return (0, 0)

        bad_min_h, bad_max_h = bad_h.min().item(), bad_h.max().item()
        bad_min_w, bad_max_w = bad_w.min().item(), bad_w.max().item()
        bad_height = bad_max_h - bad_min_h + 1
        bad_width = bad_max_w - bad_min_w + 1

        # 获取正常图mask的有效区域
        good_mask_binary = good_mask > 0.5

        # 处理4D tensor: [1, 1, H, W] -> 只取最后两个维度
        if good_mask_binary.dim() == 4:
            good_mask_binary = good_mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif good_mask_binary.dim() == 3:
            good_mask_binary = good_mask_binary.squeeze(0)  # [H, W]

        good_h, good_w = torch.where(good_mask_binary)

        if len(good_h) == 0:
            return (0, 0)

        good_min_h, good_max_h = good_h.min().item(), good_h.max().item()
        good_min_w, good_max_w = good_w.min().item(), good_w.max().item()

        # 尝试多次找到合适的位置
        max_attempts = 100
        best_position = (good_min_h, good_min_w)
        best_ioa = 0

        for _ in range(max_attempts):
            # 随机选择位置，确保缺陷完全在正常图mask内
            max_start_h = max(good_min_h, good_max_h - bad_height + 1)
            max_start_w = max(good_min_w, good_max_w - bad_width + 1)

            if max_start_h <= good_min_h or max_start_w <= good_min_w:
                continue

            start_h = random.randint(good_min_h, max_start_h)
            start_w = random.randint(good_min_w, max_start_w)

            # 计算在这个位置的IoA
            test_mask = torch.zeros_like(good_mask)
            end_h = min(start_h + bad_height, good_mask.shape[0])
            end_w = min(start_w + bad_width, good_mask.shape[1])

            # 将缺陷mask放置到测试位置
            mask_h_offset = bad_min_h
            mask_w_offset = bad_min_w

            for h in range(start_h, end_h):
                for w in range(start_w, end_w):
                    mask_h = h - start_h + mask_h_offset
                    mask_w = w - start_w + mask_w_offset
                    if (mask_h < bad_mask.shape[0] and mask_w < bad_mask.shape[1] and
                        bad_mask[mask_h, mask_w] > 0.5):
                        test_mask[h, w] = 1.0

            # 计算IoA
            intersection = torch.logical_and(test_mask > 0.5, good_mask > 0.5)
            intersection_pixels = torch.sum(intersection).item()
            test_mask_pixels = torch.sum(test_mask > 0.5).item()

            if test_mask_pixels > 0:
                ioa = intersection_pixels / test_mask_pixels
                if ioa > best_ioa:
                    best_ioa = ioa
                    best_position = (start_h, start_w)

                    # 如果达到目标IoA，提前退出
                    if ioa >= target_ioa:
                        break

        return best_position

    def interpolate_mask_position(self, original_mask: torch.Tensor, target_position: Tuple[int, int], progress: float):
        """
        在原始位置和目标位置之间插值mask位置

        Args:
            original_mask: 原始mask
            target_position: 目标位置 (y, x)
            progress: 插值进度 (0.0 到 1.0)

        Returns:
            interpolated_mask: 插值后的mask
        """
        if progress <= 0.0:
            return original_mask.clone()
        elif progress >= 1.0:
            # 完全移动到目标位置
            return self.move_mask_to_position(original_mask, target_position)
        else:
            # 插值移动
            return self.move_mask_to_position_with_interpolation(original_mask, target_position, progress)

    def move_mask_to_position(self, mask: torch.Tensor, target_position: Tuple[int, int]):
        """
        将mask移动到目标位置

        Args:
            mask: 原始mask
            target_position: 目标位置 (y, x)

        Returns:
            moved_mask: 移动后的mask
        """
        # 创建新的mask
        new_mask = torch.zeros_like(mask)

        # 获取原始mask的边界
        mask_binary = mask > 0.5
        if not torch.any(mask_binary):
            return new_mask

        # 处理4D tensor: [1, 1, H, W] -> 只取最后两个维度
        if mask_binary.dim() == 4:
            mask_binary = mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif mask_binary.dim() == 3:
            mask_binary = mask_binary.squeeze(0)  # [H, W]

        mask_h, mask_w = torch.where(mask_binary)
        min_h, max_h = mask_h.min().item(), mask_h.max().item()
        min_w, max_w = mask_w.min().item(), mask_w.max().item()

        # 计算偏移
        target_y, target_x = target_position
        offset_y = target_y - min_h
        offset_x = target_x - min_w

        # 处理4D tensor: 获取2D mask进行操作
        if mask.dim() == 4:
            mask_2d = mask.squeeze(0).squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0).squeeze(0)  # [H, W]
        elif mask.dim() == 3:
            mask_2d = mask.squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0)  # [H, W]
        else:
            mask_2d = mask
            new_mask_2d = new_mask

        # 移动mask
        for h in range(min_h, max_h + 1):
            for w in range(min_w, max_w + 1):
                if mask_2d[h, w].item() > 0.5:
                    new_h = h + offset_y
                    new_w = w + offset_x
                    if (0 <= new_h < mask_2d.shape[0] and 0 <= new_w < mask_2d.shape[1]):
                        new_mask_2d[new_h, new_w] = mask_2d[h, w]

        # 如果原始mask是4D，需要将结果重新reshape
        if mask.dim() == 4:
            new_mask[0, 0] = new_mask_2d
        elif mask.dim() == 3:
            new_mask[0] = new_mask_2d

        return new_mask

    def move_mask_to_position_with_interpolation(self, mask: torch.Tensor, target_position: Tuple[int, int], progress: float):
        """
        使用插值将mask移动到目标位置

        Args:
            mask: 原始mask
            target_position: 目标位置 (y, x)
            progress: 插值进度 (0.0 到 1.0)

        Returns:
            interpolated_mask: 插值后的mask
        """
        # 获取原始位置
        mask_binary = mask > 0.5
        if not torch.any(mask_binary):
            return mask.clone()

        # 处理4D tensor: [1, 1, H, W] -> 只取最后两个维度
        if mask_binary.dim() == 4:
            mask_binary = mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif mask_binary.dim() == 3:
            mask_binary = mask_binary.squeeze(0)  # [H, W]

        mask_h, mask_w = torch.where(mask_binary)
        original_center_h = mask_h.float().mean().item()
        original_center_w = mask_w.float().mean().item()

        # 计算目标中心位置
        target_y, target_x = target_position

        # 插值计算当前中心位置
        current_center_h = original_center_h + (target_y - original_center_h) * progress
        current_center_w = original_center_w + (target_x - original_center_w) * progress

        # 计算偏移
        offset_y = int(current_center_h - original_center_h)
        offset_x = int(current_center_w - original_center_w)

        # 创建新mask
        new_mask = torch.zeros_like(mask)

        # 处理4D tensor: 获取2D mask进行操作
        if mask.dim() == 4:
            mask_2d = mask.squeeze(0).squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0).squeeze(0)  # [H, W]
        elif mask.dim() == 3:
            mask_2d = mask.squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0)  # [H, W]
        else:
            mask_2d = mask
            new_mask_2d = new_mask

        # 应用偏移
        for h in range(mask_2d.shape[0]):
            for w in range(mask_2d.shape[1]):
                if mask_2d[h, w].item() > 0.5:
                    new_h = h + offset_y
                    new_w = w + offset_x
                    if (0 <= new_h < mask_2d.shape[0] and 0 <= new_w < mask_2d.shape[1]):
                        new_mask_2d[new_h, new_w] = mask_2d[h, w]

        # 如果原始mask是4D，需要将结果重新reshape
        if mask.dim() == 4:
            new_mask[0, 0] = new_mask_2d
        elif mask.dim() == 3:
            new_mask[0] = new_mask_2d

        return new_mask

    def generate_contrastive_defect(self,
                                  good_image_path: str,
                                  good_mask_path: str,
                                  bad_image_paths: List[str],
                                  bad_mask_paths: List[str],
                                  prompt: str = None,
                                  individual_prompts: List[str] = None,
                                  num_inference_steps: int = 50,
                                  r: float = 0.75,
                                  learning_rate: float = 0.01,
                                  num_optimization_steps: int = 5,
                                  optimization_interval: int = 5,
                                  feather_radius: int = 15,
                                  defect_variation: float = 0.0,
                                  variation_seed: int = None,
                                  output_dir: str = "outputs_contrastive",
                                  extract_attention: bool = True,
                                  defect_types: List[str] = None,
                                  enable_feature_alignment: bool = False,
                                  ioa_threshold: float = 0.5,
                                  measure_inference_time: bool = False) -> Dict[str, str]:
        """
        生成对比学习缺陷图像

        Args:
            good_image_path: 无缺陷图像路径
            good_mask_path: 无缺陷图像物体mask路径
            bad_image_paths: 有缺陷图像路径列表
            bad_mask_paths: 有缺陷图像缺陷mask路径列表
            prompt: 文本提示
            num_inference_steps: 推理步数
            r: 保留系数，控制前向扩散程度 (0-1)
            learning_rate: 学习率
            num_optimization_steps: 优化步数
            optimization_interval: 优化间隔
            feather_radius: 羽化半径，0表示无羽化
            random_placement: 是否随机放置缺陷
            placement_seed: 随机放置的种子
            placement_range: 随机放置范围倍数 (0.5=小范围, 1.0=全范围, 2.0=大范围)
            defect_variation: 缺陷变化程度 (0.0=完全相同, 1.0=高度变化)
            variation_seed: 变化随机种子
            output_dir: 输出目录

        Returns:
            file_paths: 生成文件路径字典
        """
        print("[START] Starting contrastive defect generation...")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置变化随机种子
        if variation_seed is not None:
            torch.manual_seed(variation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(variation_seed)

        # 初始化推理时间测量
        inference_times = []
        if measure_inference_time:
            import time

        # 1. 加载图像和mask
        print("[LOAD] Loading images and masks...")
        good_image, good_mask = self.load_image_and_mask(good_image_path, good_mask_path)

        # 加载多张缺陷图像和mask
        bad_images = []
        bad_masks = []
        print(f"   Loading {len(bad_image_paths)} defect images...")
        for i, (bad_img_path, bad_mask_path) in enumerate(zip(bad_image_paths, bad_mask_paths)):
            bad_img, bad_mask = self.load_image_and_mask(bad_img_path, bad_mask_path)
            bad_images.append(bad_img)
            bad_masks.append(bad_mask)
            print(f"   [SUCCESS] Loaded defect image {i+1}: {os.path.basename(bad_img_path)}")

        # 应用缺陷变化到mask
        if defect_variation > 0.0:
            print(f"   Applying defect variation (strength: {defect_variation:.2f})...")
            bad_masks = self.apply_defect_variation_to_masks(bad_masks, defect_variation)

        # 特征对齐处理
        alignment_info = []
        if enable_feature_alignment:
            print(f"[ALIGN] Performing feature alignment (IoA threshold: {ioa_threshold:.2f})...")
            alignment_info = self.calculate_feature_alignment(bad_masks, good_mask, ioa_threshold)
            for i, info in enumerate(alignment_info):
                print(f"   Defect {i+1}: IoA={info['ioa']:.3f}, needs_alignment={info['needs_alignment']}")
                if info['needs_alignment']:
                    print(f"      Target position: {info['target_position']}")

        # 合并所有缺陷mask（用于后续处理）
        combined_bad_mask = bad_masks[0].clone()
        for mask in bad_masks[1:]:
            combined_bad_mask = torch.maximum(combined_bad_mask, mask)

        # 2. 解析prompt
        print("[PARSE] Parsing prompts...")

        # 处理prompt输入：支持单个prompt或每个缺陷图对应一个prompt
        if individual_prompts is not None and len(individual_prompts) == len(bad_image_paths):
            print(f"   Using individual prompts for each defect image:")
            for i, ind_prompt in enumerate(individual_prompts):
                print(f"   {i+1}. {ind_prompt}")

            # 使用第一个prompt作为主prompt进行解析
            main_prompt = individual_prompts[0]
            product_token, anomaly_tokens = self.parse_prompt(main_prompt)

            # 解析所有individual prompts
            all_anomaly_tokens = []
            for ind_prompt in individual_prompts:
                _, tokens = self.parse_prompt(ind_prompt)
                all_anomaly_tokens.extend(tokens)

            # 去重并保持顺序
            unique_anomaly_tokens = []
            for token in all_anomaly_tokens:
                if token not in unique_anomaly_tokens:
                    unique_anomaly_tokens.append(token)

            anomaly_tokens = unique_anomaly_tokens

        elif prompt is not None:
            print(f"   Using combined prompt: {prompt}")
            product_token, anomaly_tokens = self.parse_prompt(prompt)
        else:
            raise ValueError("Either 'prompt' or 'individual_prompts' must be provided")
        print(f"   Product token: {product_token}")
        print(f"   Anomaly tokens: {anomaly_tokens}")

        # 3. 编码文本
        print("[ENCODE] Encoding text...")
        product_embeddings, product_token_indices = self.encode_text(product_token)
        anomaly_text = " ".join(anomaly_tokens)
        anomaly_embeddings, anomaly_token_indices = self.encode_text(anomaly_text)

        # 4. 编码图像
        print("[ENCODE] Encoding images...")

        # 开始推理时间测量
        if measure_inference_time:
            inference_start_time = time.time()
            print("[TIME] Starting inference time measurement...")

        # 编码无缺陷图像
        good_latents_orig, _ = self.encode_images(good_image, bad_images[0])  # 只需要good_latents

        # 编码所有缺陷图像
        bad_latents_list = []
        print(f"   Encoding {len(bad_images)} defect images...")
        for i, bad_img in enumerate(bad_images):
            _, bad_latents = self.encode_images(good_image, bad_img)  # 只需要bad_latents

            # 应用缺陷变化到潜在表示
            if defect_variation > 0.0:
                bad_latents = self.add_defect_variation(bad_latents, defect_variation)
                print(f"   [SUCCESS] Encoded defect image {i+1} (with variation)")
            else:
                print(f"   [SUCCESS] Encoded defect image {i+1}")

            bad_latents_list.append(bad_latents)

        # 选择第一张作为主要参考（用于初始化）
        primary_bad_latents = bad_latents_list[0]

        # 5. 设置时间步
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps

        # 6. 计算部分前向扩散的停止点
        t_stop = int(num_inference_steps * (1 - r))
        print(f"[FORWARD] Partial forward diffusion: stopping at step {t_stop} (r={r})")

        # 7. 生成相同的噪声
        noise_shape = good_latents_orig.shape
        noise = torch.randn(noise_shape, device=self.device, dtype=good_latents_orig.dtype)

        # 8. 部分前向扩散：只添加部分噪声
        if t_stop > 0:
            timestep_start = timesteps[t_stop]

            # 为每个缺陷图像添加不同的噪声（增加变化性）
            if defect_variation > 0.0:
                # 为good_latents添加基础噪声
                good_latents = self.add_noise(good_latents_orig, timestep_start, noise)

                # 为每个bad_latents添加略有不同的噪声
                varied_bad_latents_list = []
                for i, bad_lat in enumerate(bad_latents_list):
                    # 生成略有不同的噪声
                    variation_noise = noise + torch.randn_like(noise) * defect_variation * 0.1
                    varied_bad_lat = self.add_noise(bad_lat, timestep_start, variation_noise)
                    varied_bad_latents_list.append(varied_bad_lat)

                bad_latents_list = varied_bad_latents_list
                bad_latents = bad_latents_list[0]  # 更新主要参考
                print(f"   Added varied noise up to timestep {timestep_start}")
            else:
                good_latents = self.add_noise(good_latents_orig, timestep_start, noise)
                bad_latents = self.add_noise(primary_bad_latents, timestep_start, noise)
                print(f"   Added noise up to timestep {timestep_start}")
        else:
            # 如果r=1，则从原始图像开始（无噪声）
            good_latents = good_latents_orig.clone()
            bad_latents = primary_bad_latents.clone()
            print("   Starting from original images (no noise)")

        # 9. 反向扩散过程（从t_stop开始）
        print("[DIFFUSION] Starting reverse diffusion with contrastive optimization...")

        # 只处理从t_stop到0的时间步
        active_timesteps = timesteps[t_stop:]

        for i, timestep in enumerate(active_timesteps):
            step_num = i + 1
            total_steps = len(active_timesteps)
            print(f"  Step {step_num}/{total_steps}: timestep {timestep} (from t_stop={t_stop})")

            # 准备输入
            good_latents = good_latents.detach().requires_grad_(True)

            # 组合latents用于批处理
            latent_model_input = torch.cat([good_latents, bad_latents], dim=0)

            # 组合文本嵌入
            text_embeddings = torch.cat([product_embeddings, anomaly_embeddings], dim=0)

            # 为inpainting模型准备输入
            # 需要将mask调整到latent空间并添加到输入中
            _, _, h, w = latent_model_input.shape

            # 创建组合mask（good_mask用于good_latents，combined_bad_mask用于bad_latents）
            good_mask_latent = F.interpolate(good_mask.float(), size=(h, w), mode='nearest')
            bad_mask_latent = F.interpolate(combined_bad_mask.float(), size=(h, w), mode='nearest')
            combined_mask = torch.cat([good_mask_latent, bad_mask_latent], dim=0).to(latent_model_input.dtype)

            # 创建masked latents（用于inpainting）
            masked_latents = latent_model_input * (1 - combined_mask)

            # 组合输入：latents + mask + masked_latents
            inpaint_input = torch.cat([latent_model_input, combined_mask, masked_latents], dim=1)

            # UNet预测
            with torch.no_grad():
                noise_pred = self.unet(
                    inpaint_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False
                )[0]

            # 分离预测结果
            noise_pred_good, noise_pred_bad = noise_pred.chunk(2)

            # 优化步骤
            if step_num % optimization_interval == 0:
                print(f"    [OPTIMIZE] Optimizing attention at step {step_num}...")

                # 获取当前步骤的实际mask位置
                _, current_actual_masks = self.apply_defect_with_placement(
                    good_latents, bad_latents_list, bad_masks, good_mask, alignment_info, step_num, num_optimization_steps
                )

                # 对每张缺陷图片单独进行优化
                for defect_idx in range(len(bad_masks)):
                    print(f"      Optimizing defect {defect_idx+1}/{len(bad_masks)}...")

                    # 选择当前缺陷图像和实际使用的mask
                    current_bad_latents = bad_latents_list[defect_idx]
                    current_bad_mask = current_actual_masks[defect_idx]  # 使用实际位置的mask

                    for opt_step in range(num_optimization_steps):
                        # 重新计算UNet输出以获取注意力图
                        latent_model_input = torch.cat([good_latents, current_bad_latents], dim=0)

                        # 为inpainting模型准备输入
                        _, _, h, w = latent_model_input.shape
                        good_mask_latent = F.interpolate(good_mask.float(), size=(h, w), mode='nearest')
                        bad_mask_latent = F.interpolate(current_bad_mask.float(), size=(h, w), mode='nearest')
                        combined_mask = torch.cat([good_mask_latent, bad_mask_latent], dim=0).to(latent_model_input.dtype)
                        masked_latents = latent_model_input * (1 - combined_mask)
                        inpaint_input = torch.cat([latent_model_input, combined_mask, masked_latents], dim=1)

                        noise_pred = self.unet(
                            inpaint_input,
                            timestep,
                            encoder_hidden_states=text_embeddings,
                            return_dict=False
                        )[0]

                        # 提取注意力图
                        attention_maps = self.extract_attention_maps_from_unet(latent_model_input, text_embeddings)

                        # 计算对比损失（使用实际位置的mask）
                        contrastive_loss = self.compute_contrastive_loss(
                            attention_maps,
                            product_token_indices,
                            anomaly_token_indices,
                            good_mask,
                            current_actual_masks,
                            current_defect_idx=defect_idx
                        )

                        print(f"        Defect {defect_idx+1} step {opt_step+1}: Loss = {contrastive_loss.item():.6f}")

                        # 反向传播
                        if contrastive_loss.requires_grad:
                            contrastive_loss.backward(retain_graph=True)

                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(good_latents, max_norm=1.0)

                            # 更新good_latents
                            with torch.no_grad():
                                good_latents -= learning_rate * good_latents.grad
                                good_latents.grad.zero_()

            # 调度器步骤
            with torch.no_grad():
                good_latents = self.pipe.scheduler.step(noise_pred_good, timestep, good_latents).prev_sample
                bad_latents = self.pipe.scheduler.step(noise_pred_bad, timestep, bad_latents).prev_sample

                # 转移缺陷特征（支持特征对齐）
                good_latents, actual_bad_masks = self.apply_defect_with_placement(
                    good_latents, bad_latents_list, bad_masks, good_mask, alignment_info, i, num_inference_steps
                )

        # 8. 解码最终结果
        print("[DECODE] Decoding final result...")
        with torch.no_grad():
            final_image = self.vae.decode(good_latents / self.vae.config.scaling_factor).sample
            final_image = (final_image / 2 + 0.5).clamp(0, 1)
            final_image = final_image.cpu().permute(0, 2, 3, 1).numpy()[0]
            final_image = (final_image * 255).astype(np.uint8)

        # 结束推理时间测量
        if measure_inference_time:
            inference_end_time = time.time()
            total_inference_time = inference_end_time - inference_start_time
            inference_times.append(total_inference_time)
            print(f"[TIME] Inference completed in {total_inference_time:.2f} seconds")

        # 准备原始图像用于合并
        original_good_image = good_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        original_good_image = (original_good_image * 255).astype(np.uint8)

        # 准备缺陷mask用于合并（与正常图mask取交集）
        print("[OPTIMIZE] Optimizing final mask with object mask intersection...")

        # 将good_mask调整到与combined_bad_mask相同的尺寸
        good_mask_resized = F.interpolate(good_mask.float(), size=combined_bad_mask.shape[-2:], mode='nearest')

        # 计算交集：只保留在正常图物体区域内的缺陷
        optimized_mask = combined_bad_mask * good_mask_resized.squeeze(0).squeeze(0)

        # 转换为numpy
        defect_mask_np = optimized_mask.cpu().numpy()
        defect_mask_np = (defect_mask_np * 255).astype(np.uint8)

        print(f"   Original defect pixels: {torch.sum(combined_bad_mask > 0.5).item()}")
        print(f"   Optimized defect pixels: {torch.sum(optimized_mask > 0.5).item()}")
        print(f"   Removed pixels outside object: {torch.sum(combined_bad_mask > 0.5).item() - torch.sum(optimized_mask > 0.5).item()}")

        # 准备参考缺陷图像（使用第一张缺陷图像）
        reference_bad_image = bad_images[0].cpu().permute(0, 2, 3, 1).numpy()[0]
        reference_bad_image = (reference_bad_image * 255).astype(np.uint8)

        # 9. 创建合并图像（有羽化和无羽化版本）
        print("[BLEND] Creating blend images...")

        # 无羽化合并图像
        print(f"   Creating non-feathered blend image...")
        non_feathered_image = self.blend_images_with_feathered_mask(
            original_image=original_good_image,
            generated_image=final_image,
            mask=defect_mask_np,
            feather_radius=0  # 无羽化
        )

        # 羽化合并图像（如果羽化半径大于0）
        if feather_radius > 0:
            print(f"   Creating feathered blend image (radius={feather_radius})...")
            feathered_image = self.blend_images_with_feathered_mask(
                original_image=original_good_image,
                generated_image=final_image,
                mask=defect_mask_np,
                feather_radius=feather_radius
            )
        else:
            print("   Feather radius is 0, using non-feathered image as feathered version")
            feathered_image = non_feathered_image.copy()

        # 10. 保存结果
        print("[SAVE] Saving results...")
        file_paths = {}

        # 保存最终生成的缺陷图像
        final_image_pil = Image.fromarray(final_image)
        final_path = os.path.join(output_dir, "contrastive_defect_image.png")
        final_image_pil.save(final_path)
        file_paths["final_defect_image"] = final_path

        # 保存无羽化合并图像
        non_feathered_pil = Image.fromarray(non_feathered_image)
        non_feathered_path = os.path.join(output_dir, "non_feathered_blend_image.png")
        non_feathered_pil.save(non_feathered_path)
        file_paths["non_feathered_blend_image"] = non_feathered_path

        # 保存羽化合并图像
        feathered_pil = Image.fromarray(feathered_image)
        feathered_path = os.path.join(output_dir, "feathered_blend_image.png")
        feathered_pil.save(feathered_path)
        file_paths["feathered_blend_image"] = feathered_path

        # 创建并保存对比网格图像
        print("[GRID] Creating comparison grid...")
        comparison_grid = self.create_comparison_grid(
            original_image=original_good_image,
            generated_image=final_image,
            blended_image=feathered_image,  # 使用羽化版本作为主要展示
            reference_bad_image=reference_bad_image,
            defect_mask=defect_mask_np,
            non_feathered_image=non_feathered_image  # 添加无羽化版本
        )
        comparison_grid_pil = Image.fromarray(comparison_grid)
        comparison_path = os.path.join(output_dir, "comparison_grid.png")
        comparison_grid_pil.save(comparison_path)
        file_paths["comparison_grid"] = comparison_path

        # 保存输入图像副本用于对比
        good_image_pil = Image.open(good_image_path)
        good_copy_path = os.path.join(output_dir, "original_good_image.png")
        good_image_pil.save(good_copy_path)
        file_paths["original_good"] = good_copy_path

        # 保存参考缺陷图像（使用第一张）
        bad_image_pil = Image.open(bad_image_paths[0])
        bad_copy_path = os.path.join(output_dir, "reference_bad_image.png")
        bad_image_pil.save(bad_copy_path)
        file_paths["reference_bad"] = bad_copy_path

        # 保存所有缺陷图像副本
        for i, bad_img_path in enumerate(bad_image_paths):
            bad_img_pil = Image.open(bad_img_path)
            bad_img_copy_path = os.path.join(output_dir, f"bad_image_{i+1}.png")
            bad_img_pil.save(bad_img_copy_path)
            file_paths[f"bad_image_{i+1}"] = bad_img_copy_path

        # 保存mask副本
        good_mask_pil = Image.open(good_mask_path)
        good_mask_copy_path = os.path.join(output_dir, "good_object_mask.png")
        good_mask_pil.save(good_mask_copy_path)
        file_paths["good_mask"] = good_mask_copy_path

        # 保存所有缺陷mask副本
        for i, bad_mask_path in enumerate(bad_mask_paths):
            bad_mask_pil = Image.open(bad_mask_path)
            bad_mask_copy_path = os.path.join(output_dir, f"bad_defect_mask_{i+1}.png")
            bad_mask_pil.save(bad_mask_copy_path)
            file_paths[f"bad_mask_{i+1}"] = bad_mask_copy_path

        # 保存合并的缺陷mask
        combined_mask_np = (combined_bad_mask.cpu().squeeze().numpy() * 255).astype(np.uint8)
        combined_mask_pil = Image.fromarray(combined_mask_np)
        combined_mask_path = os.path.join(output_dir, "combined_defect_mask.png")
        combined_mask_pil.save(combined_mask_path)
        file_paths["combined_defect_mask"] = combined_mask_path

        # 提取注意力热力图
        if extract_attention and defect_types:
            print("[ATTENTION] Extracting attention heatmaps...")
            try:
                # 解析prompt中的anomaly tokens
                prompt_parts = prompt.split()
                anomaly_tokens = []

                # 从prompt中提取可能的anomaly tokens
                product_tokens = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                                "leather", "metal", "nut", "pill", "screw", "tile",
                                "toothbrush", "transistor", "wood", "zipper"]

                for token in prompt_parts:
                    if token.lower() not in product_tokens and len(token) > 2:
                        anomaly_tokens.append(token)

                if not anomaly_tokens:
                    anomaly_tokens = defect_types

                # 生成实验名称
                experiment_name = f"exp_{len(bad_image_paths)}defects"

                # 提取注意力热力图
                heatmap_path = extract_attention_heatmaps(
                    self, prompt, anomaly_tokens, defect_types,
                    experiment_name, output_dir
                )

                if heatmap_path:
                    file_paths["attention_heatmap"] = heatmap_path

            except Exception as e:
                print(f"[WARNING] Could not extract attention heatmaps: {e}")

        # 保存推理时间
        if measure_inference_time and inference_times:
            time_file_path = os.path.join(output_dir, "inference_times.txt")
            with open(time_file_path, 'w') as f:
                f.write("Inference Time Measurement\n")
                f.write("=" * 30 + "\n")
                f.write(f"Total inference time: {inference_times[0]:.4f} seconds\n")
                f.write(f"Measurement includes: image encoding, forward diffusion, optimization, and decoding\n")
                f.write(f"Number of inference steps: {num_inference_steps}\n")
                f.write(f"Number of optimization steps: {num_optimization_steps}\n")
                f.write(f"Optimization interval: {optimization_interval}\n")
                if enable_feature_alignment:
                    f.write(f"Feature alignment enabled (IoA threshold: {ioa_threshold})\n")
                else:
                    f.write(f"Feature alignment disabled\n")

            file_paths["inference_times"] = time_file_path
            print(f"[TIME] Inference time saved to: {time_file_path}")

        print("[SUCCESS] Contrastive defect generation completed!")
        print(f"[RESULTS] Results saved to: {output_dir}")

        return file_paths
