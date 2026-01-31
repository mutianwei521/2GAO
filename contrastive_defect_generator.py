"""
Contrastive Learning Defect Generator
Uses defective images to guide the generation of defects in non-defective images
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

# ==============================================================================
# [Model Initialization Module]
# Initializes VAE, U-Net, and Text Encoder for latent diffusion
# ==============================================================================
class ContrastiveDefectGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", 
                 device: str = "cuda", cache_dir: str = "./models"):
        """
        Initialize the Contrastive Learning Defect Generator
        
        Args:
            model_id: Stable Diffusion model ID
            device: Computing device
            cache_dir: Model cache directory
        """
        self.device = device
        self.model_id = model_id
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        # Initialize Stable Diffusion pipeline
        print("[INIT] Loading Stable Diffusion model...")

        # Add parameters to handle model file format issues
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*safetensors.*")

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=cache_dir,
                local_files_only=False,
                use_safetensors=False  # Explicitly specify to use .bin files
            ).to(device)
        
        # Set up scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Get components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        
        # Freeze all model parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # torch.compile optimization for UNet (requires PyTorch 2.0+ and Triton)
        # Disabled by default as Triton needs to be installed separately
        # To enable, set environment variable ENABLE_TORCH_COMPILE=1
        self._torch_compile_enabled = False
        enable_compile = os.environ.get('ENABLE_TORCH_COMPILE', '0') == '1'
        if enable_compile:
            try:
                if hasattr(torch, 'compile') and device == "cuda":
                    print("[OPTIMIZE] Applying torch.compile to UNet...")
                    self.unet = torch.compile(self.unet)
                    self._torch_compile_enabled = True
                    print("[SUCCESS] torch.compile applied!")
            except Exception as e:
                print(f"[INFO] torch.compile skipped: {e}")
            
        print("[SUCCESS] Model loaded successfully!")
    
    def load_image_and_mask(self, image_path: str, mask_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image and mask

        Args:
            image_path: Image path
            mask_path: Mask path

        Returns:
            image_tensor: Image tensor [1, 3, H, W]
            mask_tensor: Mask tensor [1, 1, H, W]
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.dtype).to(self.device)

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512))
        mask_array = np.array(mask) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(self.dtype).to(self.device)

        return image_tensor, mask_tensor
    
    def load_images_batch(self, image_paths: List[str], mask_paths: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Batch load multiple images and masks (optimized version)

        Args:
            image_paths: List of image paths
            mask_paths: List of mask paths

        Returns:
            images: List of image tensors
            masks: List of mask tensors
        """
        images = []
        masks = []
        
        # Pre-allocate numpy arrays for batch processing
        batch_size = len(image_paths)
        image_batch = np.zeros((batch_size, 512, 512, 3), dtype=np.float32)
        mask_batch = np.zeros((batch_size, 512, 512), dtype=np.float32)
        
        # Batch read images into memory
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            # Load image
            image = Image.open(img_path).convert("RGB").resize((512, 512))
            image_batch[i] = np.array(image) / 255.0
            
            # Load mask
            mask = Image.open(mask_path).convert("L").resize((512, 512))
            mask_batch[i] = np.array(mask) / 255.0
        
        # Convert to tensor at once
        image_tensor_batch = torch.from_numpy(image_batch).permute(0, 3, 1, 2).to(self.dtype).to(self.device)
        mask_tensor_batch = torch.from_numpy(mask_batch).unsqueeze(1).to(self.dtype).to(self.device)
        
        # Split back to list (maintain interface compatibility)
        for i in range(batch_size):
            images.append(image_tensor_batch[i:i+1])
            masks.append(mask_tensor_batch[i:i+1])
        
        return images, masks
    
    def encode_images_batch(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Batch encode multiple images to latent space (optimized version)

        Args:
            images: List of image tensors, each with shape [1, 3, H, W]

        Returns:
            latents_list: List of latent representations
        """
        if len(images) == 0:
            return []
        
        # Merge into a single batch
        batch = torch.cat(images, dim=0)  # [N, 3, H, W]
        
        # Encode at once
        with torch.no_grad():
            latents = self.vae.encode(batch * 2 - 1).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Split back to list
        return [latents[i:i+1] for i in range(len(images))]
    
    def parse_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Parse prompt to extract product and anomaly tokens
        
        Args:
            prompt: Input prompt, e.g., "nutshell crack scratches"
            
        Returns:
            product_token: Product token
            anomaly_tokens: List of anomaly tokens
        """
        # Predefined product vocabulary
        product_keywords = [
            "nutshell", "nut", "bottle", "cable", "capsule", "hazelnut", 
            "metal", "pill", "screw", "toothbrush", "transistor", "zipper",
            "carpet", "grid", "leather", "tile", "wood"
        ]
        
        # Predefined anomaly vocabulary and variants
        # Predefined anomaly vocabulary and variants, including all MVTEC defect types
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

            # Generic defect vocabulary (combined -> damage)
            "damage": ["damage", "damaged", "defect", "defective", "fault", "faulty"],
            "combined": ["damage", "damaged", "defect", "defective", "fault", "faulty"]  # combined maps to damage
        }
        
        words = prompt.lower().split()
        
        # Find product token
        product_token = None
        for word in words:
            if word in product_keywords:
                product_token = word
                break
        
        if product_token is None:
            product_token = words[0] if words else "object"
        
        # Find anomaly tokens
        anomaly_tokens = []
        for word in words:
            for base_anomaly, variants in anomaly_keywords.items():
                if word in variants:
                    if base_anomaly not in anomaly_tokens:
                        anomaly_tokens.append(base_anomaly)
                    break
        
        if not anomaly_tokens:
            # If no predefined anomaly vocabulary found, use other words excluding product vocabulary
            anomaly_tokens = [word for word in words if word != product_token]
        
        return product_token, anomaly_tokens
    
    def encode_text(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode text and return token indices
        
        Args:
            text: Input text
            
        Returns:
            text_embeddings: Text embeddings
            token_indices: List of token indices
        """
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get token indices
        token_ids = text_inputs.input_ids[0].tolist()
        
        # Find indices of actual text tokens (excluding special tokens)
        token_indices = []
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        for i, token in enumerate(tokens):
            if token not in ['<|startoftext|>', '<|endoftext|>', '<|padding|>'] and not token.startswith('<'):
                token_indices.append(i)
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
            # Ensure text embeddings have the same dtype as the model
            text_embeddings = text_embeddings.to(self.dtype)

        return text_embeddings, token_indices
    
    # ==============================================================================
    # [Stage 1: VAE Encoding Module]
    # Encodes input images into latent space representations
    # ==============================================================================
    def encode_images(self, good_image: torch.Tensor, bad_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space
        
        Args:
            good_image: Non-defective image
            bad_image: Defective image
            
        Returns:
            good_latents: Non-defective image latent representation
            bad_latents: Defective image latent representation
        """
        with torch.no_grad():
            # Encode non-defective image
            good_latents = self.vae.encode(good_image * 2 - 1).latent_dist.sample()
            good_latents = good_latents * self.vae.config.scaling_factor
            
            # Encode defective image
            bad_latents = self.vae.encode(bad_image * 2 - 1).latent_dist.sample()
            bad_latents = bad_latents * self.vae.config.scaling_factor
        
        return good_latents, bad_latents
    
    def add_noise(self, latents: torch.Tensor, timestep: int, noise: torch.Tensor) -> torch.Tensor:
        """
        Add noise to latent representation

        Args:
            latents: Latent representation
            timestep: Time step
            noise: Noise

        Returns:
            noisy_latents: Noisy latent representation
        """
        # Get noise schedule parameters
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep].to(latents.device, latents.dtype)
        beta_prod_t = (1 - alpha_prod_t).to(latents.device, latents.dtype)

        # Ensure noise dtype matches latents
        noise = noise.to(latents.dtype)

        # Add noise
        noisy_latents = (alpha_prod_t ** 0.5) * latents + (beta_prod_t ** 0.5) * noise

        return noisy_latents
    
    def extract_attention_maps_from_unet(self, latents: torch.Tensor, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps through UNet forward pass

        Args:
            latents: Latent representation
            text_embeddings: Text embeddings

        Returns:
            attention_maps: Dictionary of simulated attention maps
        """
        # Here we create a simplified attention map extraction
        # In actual applications, you may need to modify UNet to directly return attention weights

        attention_maps = {}

        # Get spatial dimensions of latents
        _, _, h, w = latents.shape
        num_tokens = text_embeddings.shape[1]

        # Create simulated attention maps
        # In actual implementation, this should be obtained from UNet's cross-attention layers
        for layer_idx in range(4):  # Assume 4 attention layers
            # Create random attention maps as placeholders
            # In actual applications, real attention weights should be obtained from UNet internals
            attention_map = torch.randn(2, h, w, num_tokens, device=latents.device)
            attention_map = F.softmax(attention_map.view(2, h*w, num_tokens), dim=1).view(2, h, w, num_tokens)
            attention_maps[f"layer_{layer_idx}"] = attention_map

        return attention_maps
    
    # ==============================================================================
    # [Stage 4: Attention-Guided Optimization Module (Contrastive Loss)]
    # Computes Focus Loss (Product) and Suppression Loss (Anomaly)
    # ==============================================================================
    def compute_contrastive_loss(self,
                                attention_maps: Dict[str, torch.Tensor],
                                product_token_indices: List[int],
                                anomaly_token_indices: List[int],
                                good_mask: torch.Tensor,
                                bad_masks_list: List[torch.Tensor],
                                current_defect_idx: int = 0) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            attention_maps: Attention maps
            product_token_indices: Product token indices
            anomaly_token_indices: Anomaly token indices
            good_mask: Object mask for non-defective image
            bad_masks_list: List of defect masks for defective images
            current_defect_idx: Index of current defect image being optimized

        Returns:
            total_loss: Total loss
        """
        total_loss = 0.0
        num_maps = 0
        
        for _, attention_map in attention_maps.items():
            if attention_map.dim() != 4:
                continue
                
            _, h, w, num_tokens = attention_map.shape
            
            # Resize mask to attention map size
            good_mask_resized = F.interpolate(good_mask.float(), size=(h, w), mode='nearest').squeeze(1)

            # Use current defect mask being optimized
            current_bad_mask = bad_masks_list[current_defect_idx]
            bad_mask_resized = F.interpolate(current_bad_mask.float(), size=(h, w), mode='nearest').squeeze(1)
            
            # First part of loss: Product token attention at object position in non-defective image
            loss1 = 0.0
            for token_idx in product_token_indices:
                if token_idx < num_tokens:
                    # Get attention for product token
                    product_attention = attention_map[0, :, :, token_idx]  # Non-defective image (first in batch)
                    
                    # Softmax normalization
                    product_attention_flat = product_attention.view(-1)
                    product_attention_norm = F.softmax(product_attention_flat, dim=0).view(h, w)
                    
                    # Dot product with object mask and sum
                    attention_score = torch.sum(product_attention_norm * good_mask_resized[0])
                    loss1 += 1 - attention_score
            
            # Second part of loss: Anomaly token attention at defect position in defective image
            loss2 = 0.0
            for token_idx in anomaly_token_indices:
                if token_idx < num_tokens:
                    # Get attention for anomaly token
                    anomaly_attention = attention_map[1, :, :, token_idx]  # Defective image (second in batch)
                    
                    # Softmax normalization
                    anomaly_attention_flat = anomaly_attention.view(-1)
                    anomaly_attention_norm = F.softmax(anomaly_attention_flat, dim=0).view(h, w)
                    
                    # Dot product with defect mask and sum
                    attention_score = torch.sum(anomaly_attention_norm * bad_mask_resized[0])
                    loss2 += 1 - attention_score
            
            # Combine both parts of loss
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
        Transfer defect features from defective image to non-defective image

        Args:
            good_latents: Non-defective image latent representation
            bad_latents: Defective image latent representation
            bad_mask: Defect position mask

        Returns:
            updated_latents: Updated latent representation
        """
        # Resize mask to latent space dimensions
        _, _, h, w = good_latents.shape
        bad_mask_latent = F.interpolate(bad_mask.float(), size=(h, w), mode='nearest').to(good_latents.dtype)

        # Transfer features from defect region of bad_latents to good_latents
        updated_latents = good_latents * (1 - bad_mask_latent) + bad_latents * bad_mask_latent

        return updated_latents

    # ==============================================================================
    # [Stage 2: IoA-based Feature Alignment Module (Search Strategy)]
    # Finds optimal defect placement maximizing Intersection-over-Area
    # ==============================================================================
    def find_random_placement(self,
                            defect_mask: torch.Tensor,
                            object_mask: torch.Tensor,
                            placement_range: float = 1.0) -> Tuple[torch.Tensor, int, int]:
        """
        Find a random position within the object mask to place the defect

        Args:
            defect_mask: Defect mask [1, 1, H, W]
            object_mask: Object mask [1, 1, H, W]
            placement_range: Placement range multiplier (0.5=small range, 1.0=full range, 2.0=large range)

        Returns:
            new_mask: Defect mask at new position
            offset_y, offset_x: Offsets (for display)
        """
        # Convert to numpy for processing
        defect_np = defect_mask.cpu().squeeze().numpy()
        object_np = object_mask.cpu().squeeze().numpy()

        h, w = defect_np.shape

        # Get coordinates of defect and object
        defect_coords = np.where(defect_np > 0.5)
        object_coords = np.where(object_np > 0.5)

        if len(defect_coords[0]) == 0 or len(object_coords[0]) == 0:
            return defect_mask.clone(), 0, 0

        # Get defect bounding box
        defect_min_y, defect_max_y = defect_coords[0].min(), defect_coords[0].max()
        defect_min_x, defect_max_x = defect_coords[1].min(), defect_coords[1].max()
        defect_h = defect_max_y - defect_min_y + 1
        defect_w = defect_max_x - defect_min_x + 1

        # Get object bounding box
        obj_min_y, obj_max_y = object_coords[0].min(), object_coords[0].max()
        obj_min_x, obj_max_x = object_coords[1].min(), object_coords[1].max()

        # Calculate available placement range (ensure defect is fully within object)
        available_h = obj_max_y - obj_min_y + 1 - defect_h
        available_w = obj_max_x - obj_min_x + 1 - defect_w

        if available_h <= 0 or available_w <= 0:
            # Defect too large to fit entirely inside object, keep original position
            print(f"     Defect too large for object, keeping original position")
            return defect_mask.clone(), 0, 0

        # Adjust available range based on placement_range
        if placement_range != 1.0:
            # Calculate range center
            center_h = available_h // 2
            center_w = available_w // 2

            # Adjust range size
            adjusted_h = max(1, int(available_h * placement_range))
            adjusted_w = max(1, int(available_w * placement_range))

            # Calculate new start position (keep centered)
            start_h = max(0, center_h - adjusted_h // 2)
            start_w = max(0, center_w - adjusted_w // 2)

            # Ensure not exceeding bounds
            end_h = min(available_h, start_h + adjusted_h)
            end_w = min(available_w, start_w + adjusted_w)

            available_h = end_h - start_h
            available_w = end_w - start_w

            # Randomly select new top-left position
            new_top_y = obj_min_y + start_h + random.randint(0, max(0, available_h - 1))
            new_top_x = obj_min_x + start_w + random.randint(0, max(0, available_w - 1))
        else:
            # Use full range
            new_top_y = obj_min_y + random.randint(0, available_h)
            new_top_x = obj_min_x + random.randint(0, available_w)

        # Calculate offset
        offset_y = new_top_y - defect_min_y
        offset_x = new_top_x - defect_min_x

        # Create new mask
        new_mask_np = np.zeros_like(defect_np)

        # Copy defect to new position
        for i, (y, x) in enumerate(zip(defect_coords[0], defect_coords[1])):
            new_y = y + offset_y
            new_x = x + offset_x

            # Ensure new coordinates are within bounds
            if 0 <= new_y < h and 0 <= new_x < w:
                new_mask_np[new_y, new_x] = defect_np[y, x]

        # Convert back to tensor
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
        Apply multiple defects to non-defective image, with feature alignment support

        Args:
            good_latents: Non-defective image latent representation
            bad_latents_list: List of defect image latent representations
            bad_masks_list: List of defect masks
            good_mask: Object mask
            alignment_info: Feature alignment information
            current_step: Current optimization step
            total_steps: Total optimization steps

        Returns:
            updated_latents: Updated latent representation
            actual_masks: List of actually used masks
        """
        updated_latents = good_latents.clone()
        _, _, h, w = good_latents.shape

        # Resize object mask to latent space
        good_mask_latent = F.interpolate(good_mask.float(), size=(h, w), mode='nearest')

        # Record actually used masks
        actual_masks = []

        for i, (bad_latents, bad_mask) in enumerate(zip(bad_latents_list, bad_masks_list)):
            print(f"   Applying defect {i+1}/{len(bad_latents_list)}...")

            # Get alignment info for current defect
            if alignment_info and i < len(alignment_info):
                align_info = alignment_info[i]
                if align_info['needs_alignment']:
                    # Calculate current step interpolation position
                    progress = current_step / max(total_steps - 1, 1)  # Progress from 0 to 1
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

            # Resize mask to latent space dimensions
            current_mask_latent = F.interpolate(current_mask.float(), size=(h, w), mode='nearest')

            # Ensure defect is only within object region
            effective_mask = (current_mask_latent * good_mask_latent).to(good_latents.dtype)

            # Apply defect to good image
            updated_latents = updated_latents * (1 - effective_mask) + bad_latents * effective_mask
            actual_masks.append(current_mask)

        return updated_latents, actual_masks

    def add_defect_variation(self,
                           latents: torch.Tensor,
                           variation_strength: float = 0.0) -> torch.Tensor:
        """
        Add random variation to latent representation

        Args:
            latents: Input latent representation
            variation_strength: Variation strength (0.0-1.0)

        Returns:
            varied_latents: Latent representation with added variation
        """
        if variation_strength <= 0.0:
            return latents

        # Generate random noise
        noise = torch.randn_like(latents) * variation_strength * 0.1

        # Add random variation
        varied_latents = latents + noise

        return varied_latents

    def apply_defect_variation_to_masks(self,
                                      masks: List[torch.Tensor],
                                      variation_strength: float = 0.0) -> List[torch.Tensor]:
        """
        Add random variation to defect masks

        Args:
            masks: List of defect masks
            variation_strength: Variation strength (0.0-1.0)

        Returns:
            varied_masks: List of varied masks
        """
        if variation_strength <= 0.0:
            return masks

        varied_masks = []
        for mask in masks:
            if variation_strength > 0.0:
                # Apply slight morphological changes to mask
                mask_np = mask.cpu().squeeze().numpy()

                # Ensure data type is uint8
                mask_np = (mask_np * 255).astype(np.uint8)

                # Random dilation or erosion
                if random.random() < 0.5:
                    # Dilation (enlarge defect)
                    kernel_size = int(3 + variation_strength * 5)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    # Erosion (shrink defect)
                    kernel_size = int(2 + variation_strength * 3)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)

                # Convert back to float and normalize
                mask_np = mask_np.astype(np.float32) / 255.0

                # Add random noise to mask edges
                if variation_strength > 0.3:
                    noise = np.random.random(mask_np.shape) * variation_strength * 0.3
                    mask_np = np.clip(mask_np + noise, 0, 1)

                # Convert back to tensor
                varied_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(mask.device)
                varied_masks.append(varied_mask)
            else:
                varied_masks.append(mask)

        return varied_masks

    # ==============================================================================
    # [Stage 5: Decoding and Blending Module]
    # Decodes latents and applies feathered blending for seamless integration
    # ==============================================================================
    def create_feathered_mask(self, mask: np.ndarray, feather_radius: int = 10) -> np.ndarray:
        """
        Create mask with feathered edges

        Args:
            mask: Original mask (0-255)
            feather_radius: Feather radius

        Returns:
            feathered_mask: Feathered mask (0-1)
        """
        # Ensure mask is in 0-255 range
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Use Gaussian blur for feathering
        feathered = cv2.GaussianBlur(mask, (feather_radius*2+1, feather_radius*2+1), feather_radius/3)

        # Normalize to 0-1 range
        feathered = feathered.astype(np.float32) / 255.0

        return feathered

    def blend_images_with_feathered_mask(self,
                                       original_image: np.ndarray,
                                       generated_image: np.ndarray,
                                       mask: np.ndarray,
                                       feather_radius: int = 10) -> np.ndarray:
        """
        Blend defect portion of generated image with original image using feathered mask

        Args:
            original_image: Original image (H, W, 3) range 0-255
            generated_image: Generated image (H, W, 3) range 0-255
            mask: Defect mask (H, W) range 0-255, white is defect region
            feather_radius: Feather radius

        Returns:
            blended_image: Blended image (H, W, 3) range 0-255
        """
        # Create feathered mask
        feathered_mask = self.create_feathered_mask(mask, feather_radius)

        # Ensure correct image data types
        original_image = original_image.astype(np.float32)
        generated_image = generated_image.astype(np.float32)

        # Ensure mask shape is correct
        if len(feathered_mask.shape) == 4:  # (1, 1, H, W)
            feathered_mask = feathered_mask.squeeze()  # (H, W)
        elif len(feathered_mask.shape) == 3:  # (1, H, W) or (H, W, 1)
            feathered_mask = feathered_mask.squeeze()  # (H, W)

        # Resize mask to match image
        if feathered_mask.shape != original_image.shape[:2]:
            import cv2
            feathered_mask = cv2.resize(feathered_mask, (original_image.shape[1], original_image.shape[0]))

        # Expand mask dimensions to match image channels
        if len(feathered_mask.shape) == 2:
            feathered_mask = np.expand_dims(feathered_mask, axis=2)
        feathered_mask = np.repeat(feathered_mask, 3, axis=2)

        # Blend using feathered mask
        # mask=1 uses generated image, mask=0 uses original image
        blended_image = original_image * (1 - feathered_mask) + generated_image * feathered_mask

        # Convert back to uint8
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
        Create comparison grid image

        Args:
            original_image: Original non-defective image
            generated_image: Generated defect image
            blended_image: Feathered blended image
            reference_bad_image: Reference defect image
            defect_mask: Defect mask
            non_feathered_image: Non-feathered blended image

        Returns:
            comparison_grid: Comparison grid image
        """
        # Ensure all images have consistent dimensions
        h, w = original_image.shape[:2]

        # Check if image dimensions are valid
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: h={h}, w={w}")

        # Resize all images to same dimensions
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

        # Create mask visualization (convert to 3 channels)
        if defect_mask is None:
            raise ValueError("defect_mask is None")
        if len(defect_mask.shape) < 2:
            raise ValueError(f"defect_mask has invalid shape: {defect_mask.shape}")

        # Ensure defect_mask is 2D array
        if len(defect_mask.shape) == 4:  # (1, 1, H, W)
            defect_mask_2d = defect_mask.squeeze()  # (H, W)
        elif len(defect_mask.shape) == 3:  # (1, H, W) or (H, W, 1)
            defect_mask_2d = defect_mask.squeeze()  # (H, W)
        else:
            defect_mask_2d = defect_mask  # Already 2D

        mask_vis = np.stack([defect_mask_2d, defect_mask_2d, defect_mask_2d], axis=2)
        if mask_vis.max() <= 1.0:
            mask_vis = (mask_vis * 255).astype(np.uint8)

        # Check mask dimensions and resize
        if mask_vis.shape[:2] != (h, w)::
            mask_vis = cv2.resize(mask_vis, (w, h))

        # Create grid layout
        if non_feathered_image is not None:
            # If non-feathered image exists, create 2x4 grid
            # Row 1: Original image, generated image, non-feathered blend, feathered blend
            # Row 2: Reference defect image, defect mask, blank, blank
            non_feathered_resized = cv2.resize(non_feathered_image, (w, h)) if non_feathered_image.shape[:2] != (h, w) else non_feathered_image

            top_row = np.hstack([resized_images[0], resized_images[1], non_feathered_resized, resized_images[2]])

            # Create blank image
            blank = np.ones_like(resized_images[0]) * 255
            bottom_row = np.hstack([resized_images[3], mask_vis, blank, blank])
        else:
            # Original 2x3 grid
            # Row 1: Original image, generated image, feathered blended image
            # Row 2: Reference defect image, defect mask, blank
            top_row = np.hstack([resized_images[0], resized_images[1], resized_images[2]])

            # Create blank image
            blank = np.ones_like(resized_images[0]) * 255
            bottom_row = np.hstack([resized_images[3], mask_vis, blank])

        # Combine into complete grid
        comparison_grid = np.vstack([top_row, bottom_row])

        # Add label area (simple white strip)
        label_height = 30
        grid_h, grid_w = comparison_grid.shape[:2]

        # Add label background for each sub-image
        labeled_grid = np.ones((grid_h + label_height * 2, grid_w, 3), dtype=np.uint8) * 255
        labeled_grid[label_height:label_height + grid_h] = comparison_grid

        return labeled_grid

    def calculate_feature_alignment(self, bad_masks: List[torch.Tensor], good_mask: torch.Tensor, ioa_threshold: float):
        """
        Calculate feature alignment information

        Args:
            bad_masks: List of defect masks
            good_mask: Normal image object mask
            ioa_threshold: IoA threshold

        Returns:
            alignment_info: List of alignment information
        """
        alignment_info = []

        for i, bad_mask in enumerate(bad_masks):
            # Calculate intersection
            intersection = torch.logical_and(bad_mask > 0.5, good_mask > 0.5)
            intersection_pixels = torch.sum(intersection).item()

            # Calculate total pixels of defect mask
            bad_mask_pixels = torch.sum(bad_mask > 0.5).item()

            # Calculate IoA (Intersection over Area)
            ioa = intersection_pixels / max(bad_mask_pixels, 1)  # Avoid division by zero

            # Determine if alignment is needed
            needs_alignment = ioa < ioa_threshold

            target_position = None
            if needs_alignment:
                # Randomly select target position within normal image mask region
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
        Find target position within normal image mask region

        Args:
            bad_mask: Defect mask
            good_mask: Normal image object mask
            target_ioa: Target IoA value

        Returns:
            target_position: Target position (y, x)
        """
        import random

        # Get defect mask dimensions
        bad_mask_binary = bad_mask > 0.5

        # Handle 4D tensor: [1, 1, H, W] -> only take last two dimensions
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

        # Get valid region of normal image mask
        good_mask_binary = good_mask > 0.5

        # Handle 4D tensor: [1, 1, H, W] -> only take last two dimensions
        if good_mask_binary.dim() == 4:
            good_mask_binary = good_mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif good_mask_binary.dim() == 3:
            good_mask_binary = good_mask_binary.squeeze(0)  # [H, W]

        good_h, good_w = torch.where(good_mask_binary)

        if len(good_h) == 0:
            return (0, 0)

        good_min_h, good_max_h = good_h.min().item(), good_h.max().item()
        good_min_w, good_max_w = good_w.min().item(), good_w.max().item()

        # Try multiple times to find suitable position
        max_attempts = 100
        best_position = (good_min_h, good_min_w)
        best_ioa = 0

        for _ in range(max_attempts):
            # Randomly select position, ensure defect is fully within normal image mask
            max_start_h = max(good_min_h, good_max_h - bad_height + 1)
            max_start_w = max(good_min_w, good_max_w - bad_width + 1)

            if max_start_h <= good_min_h or max_start_w <= good_min_w:
                continue

            start_h = random.randint(good_min_h, max_start_h)
            start_w = random.randint(good_min_w, max_start_w)

            # Calculate IoA at this position - use vectorized operations instead of nested loops
            test_mask = torch.zeros_like(good_mask)
            end_h = min(start_h + bad_height, good_mask.shape[-2] if good_mask.dim() > 2 else good_mask.shape[0])
            end_w = min(start_w + bad_width, good_mask.shape[-1] if good_mask.dim() > 2 else good_mask.shape[1])

            # Use slicing to directly copy defect region (vectorized)
            src_h_start = bad_min_h
            src_h_end = bad_min_h + (end_h - start_h)
            src_w_start = bad_min_w
            src_w_end = bad_min_w + (end_w - start_w)
            
            # Ensure source and target region dimensions match
            if good_mask.dim() == 4:
                bad_mask_region = bad_mask[:, :, src_h_start:src_h_end, src_w_start:src_w_end]
                test_mask[:, :, start_h:end_h, start_w:end_w] = (bad_mask_region > 0.5).float()
            elif good_mask.dim() == 2:
                bad_mask_2d = bad_mask.squeeze() if bad_mask.dim() > 2 else bad_mask
                bad_mask_region = bad_mask_2d[src_h_start:src_h_end, src_w_start:src_w_end]
                test_mask[start_h:end_h, start_w:end_w] = (bad_mask_region > 0.5).float()
            else:
                bad_mask_region = bad_mask[src_h_start:src_h_end, src_w_start:src_w_end]
                test_mask[start_h:end_h, start_w:end_w] = (bad_mask_region > 0.5).float()

            # Calculate IoA
            intersection = torch.logical_and(test_mask > 0.5, good_mask > 0.5)
            intersection_pixels = torch.sum(intersection).item()
            test_mask_pixels = torch.sum(test_mask > 0.5).item()

            if test_mask_pixels > 0:
                ioa = intersection_pixels / test_mask_pixels
                if ioa > best_ioa:
                    best_ioa = ioa
                    best_position = (start_h, start_w)

                    # If target IoA is reached, exit early
                    if ioa >= target_ioa:
                        break

        return best_position

    def interpolate_mask_position(self, original_mask: torch.Tensor, target_position: Tuple[int, int], progress: float):
        """
        Interpolate mask position between original and target positions

        Args:
            original_mask: Original mask
            target_position: Target position (y, x)
            progress: Interpolation progress (0.0 to 1.0)

        Returns:
            interpolated_mask: Interpolated mask
        """
        if progress <= 0.0:
            return original_mask.clone()
        elif progress >= 1.0:
            # Fully move to target position
            return self.move_mask_to_position(original_mask, target_position)
        else:
            # Interpolated movement
            return self.move_mask_to_position_with_interpolation(original_mask, target_position, progress)

    def move_mask_to_position(self, mask: torch.Tensor, target_position: Tuple[int, int]):
        """
        Move mask to target position

        Args:
            mask: Original mask
            target_position: Target position (y, x)

        Returns:
            moved_mask: Moved mask
        """
        # Create new mask
        new_mask = torch.zeros_like(mask)

        # Get original mask boundaries
        mask_binary = mask > 0.5
        if not torch.any(mask_binary):
            return new_mask

        # Handle 4D tensor: [1, 1, H, W] -> only take last two dimensions
        if mask_binary.dim() == 4:
            mask_binary = mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif mask_binary.dim() == 3:
            mask_binary = mask_binary.squeeze(0)  # [H, W]

        mask_h, mask_w = torch.where(mask_binary)
        min_h, max_h = mask_h.min().item(), mask_h.max().item()
        min_w, max_w = mask_w.min().item(), mask_w.max().item()

        # Calculate offset
        target_y, target_x = target_position
        offset_y = target_y - min_h
        offset_x = target_x - min_w

        # Handle 4D tensor: get 2D mask for operation
        if mask.dim() == 4:
            mask_2d = mask.squeeze(0).squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0).squeeze(0)  # [H, W]
        elif mask.dim() == 3:
            mask_2d = mask.squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0)  # [H, W]
        else:
            mask_2d = mask
            new_mask_2d = new_mask

        # Move mask
        for h in range(min_h, max_h + 1)::
            for w in range(min_w, max_w + 1):
                if mask_2d[h, w].item() > 0.5:
                    new_h = h + offset_y
                    new_w = w + offset_x
                    if (0 <= new_h < mask_2d.shape[0] and 0 <= new_w < mask_2d.shape[1]):
                        new_mask_2d[new_h, new_w] = mask_2d[h, w]

        # If original mask is 4D, need to reshape result
        if mask.dim() == 4:
            new_mask[0, 0] = new_mask_2d
        elif mask.dim() == 3:
            new_mask[0] = new_mask_2d

        return new_mask

    def move_mask_to_position_with_interpolation(self, mask: torch.Tensor, target_position: Tuple[int, int], progress: float):
        """
        Move mask to target position using interpolation

        Args:
            mask: Original mask
            target_position: Target position (y, x)
            progress: Interpolation progress (0.0 to 1.0)

        Returns:
            interpolated_mask: Interpolated mask
        """
        # Get original position
        mask_binary = mask > 0.5
        if not torch.any(mask_binary):
            return mask.clone()

        # Handle 4D tensor: [1, 1, H, W] -> only take last two dimensions
        if mask_binary.dim() == 4:
            mask_binary = mask_binary.squeeze(0).squeeze(0)  # [H, W]
        elif mask_binary.dim() == 3:
            mask_binary = mask_binary.squeeze(0)  # [H, W]

        mask_h, mask_w = torch.where(mask_binary)
        original_center_h = mask_h.float().mean().item()
        original_center_w = mask_w.float().mean().item()

        # Calculate target center position
        target_y, target_x = target_position

        # Interpolate to calculate current center position
        current_center_h = original_center_h + (target_y - original_center_h) * progress
        current_center_w = original_center_w + (target_x - original_center_w) * progress

        # Calculate offset
        offset_y = int(current_center_h - original_center_h)
        offset_x = int(current_center_w - original_center_w)

        # Create new mask
        new_mask = torch.zeros_like(mask)

        # Handle 4D tensor: get 2D mask for operation
        if mask.dim() == 4:
            mask_2d = mask.squeeze(0).squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0).squeeze(0)  # [H, W]
        elif mask.dim() == 3:
            mask_2d = mask.squeeze(0)  # [H, W]
            new_mask_2d = new_mask.squeeze(0)  # [H, W]
        else:
            mask_2d = mask
            new_mask_2d = new_mask

        # Apply offset
        for h in range(mask_2d.shape[0]):
            for w in range(mask_2d.shape[1]):
                if mask_2d[h, w].item() > 0.5:
                    new_h = h + offset_y
                    new_w = w + offset_x
                    if (0 <= new_h < mask_2d.shape[0] and 0 <= new_w < mask_2d.shape[1]):
                        new_mask_2d[new_h, new_w] = mask_2d[h, w]

        # If original mask is 4D, need to reshape result
        if mask.dim() == 4:
            new_mask[0, 0] = new_mask_2d
        elif mask.dim() == 3:
            new_mask[0] = new_mask_2d

        return new_mask

    # ==============================================================================
    # [Pipeline Orchestration]
    # Coordinates the full 5-stage generation process
    # ==============================================================================
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
        Generate contrastive learning defect image

        Args:
            good_image_path: Non-defective image path
            good_mask_path: Non-defective image object mask path
            bad_image_paths: List of defective image paths
            bad_mask_paths: List of defective image defect mask paths
            prompt: Text prompt
            num_inference_steps: Number of inference steps
            r: Retention coefficient, controls forward diffusion degree (0-1)
            learning_rate: Learning rate
            num_optimization_steps: Number of optimization steps
            optimization_interval: Optimization interval
            feather_radius: Feather radius, 0 means no feathering
            random_placement: Whether to randomly place defects
            placement_seed: Random seed for placement
            placement_range: Placement range multiplier (0.5=small range, 1.0=full range, 2.0=large range)
            defect_variation: Defect variation degree (0.0=identical, 1.0=highly varied)
            variation_seed: Variation random seed
            output_dir: Output directory

        Returns:
            file_paths: Dictionary of generated file paths
        """
        print("[START] Starting contrastive defect generation...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set variation random seed
        if variation_seed is not None:
            torch.manual_seed(variation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(variation_seed)

        # Initialize inference time measurement
        inference_times = []
        if measure_inference_time:
            import time

        # 1. Load images and masks
        print("[LOAD] Loading images and masks...")
        good_image, good_mask = self.load_image_and_mask(good_image_path, good_mask_path)

        # Batch load multiple defect images and masks (optimized version)
        print(f"   Batch loading {len(bad_image_paths)} defect images...")
        bad_images, bad_masks = self.load_images_batch(bad_image_paths, bad_mask_paths)
        print(f"   [SUCCESS] Loaded {len(bad_images)} defect images")

        # Apply defect variation to masks
        if defect_variation > 0.0:
            print(f"   Applying defect variation (strength: {defect_variation:.2f})...")
            bad_masks = self.apply_defect_variation_to_masks(bad_masks, defect_variation)

        # Feature alignment processing
        alignment_info = []
        # ------------------------------------------------------------------
        # [Stage 2: IoA-based Feature Alignment execution]
        # ------------------------------------------------------------------
        if enable_feature_alignment:
            print(f"[ALIGN] Performing feature alignment (IoA threshold: {ioa_threshold:.2f})...")
            alignment_info = self.calculate_feature_alignment(bad_masks, good_mask, ioa_threshold)
            for i, info in enumerate(alignment_info):
                print(f"   Defect {i+1}: IoA={info['ioa']:.3f}, needs_alignment={info['needs_alignment']}")
                if info['needs_alignment']:
                    print(f"      Target position: {info['target_position']}")

        # Combine all defect masks (for subsequent processing)
        combined_bad_mask = bad_masks[0].clone()
        for mask in bad_masks[1:]:
            combined_bad_mask = torch.maximum(combined_bad_mask, mask)

        # 2. Parse prompt
        print("[PARSE] Parsing prompts...")

        # Process prompt input: support single prompt or one prompt per defect image
        if individual_prompts is not None and len(individual_prompts) == len(bad_image_paths):
            print(f"   Using individual prompts for each defect image:")
            for i, ind_prompt in enumerate(individual_prompts):
                print(f"   {i+1}. {ind_prompt}")

            # Use first prompt as main prompt for parsing
            main_prompt = individual_prompts[0]
            product_token, anomaly_tokens = self.parse_prompt(main_prompt)

            # Parse all individual prompts
            all_anomaly_tokens = []
            for ind_prompt in individual_prompts:
                _, tokens = self.parse_prompt(ind_prompt)
                all_anomaly_tokens.extend(tokens)

            # Deduplicate while preserving order
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

        # 3. Encode text
        print("[ENCODE] Encoding text...")
        product_embeddings, product_token_indices = self.encode_text(product_token)
        anomaly_text = " ".join(anomaly_tokens)
        anomaly_embeddings, anomaly_token_indices = self.encode_text(anomaly_text)

        # 4. Encode images
        # ------------------------------------------------------------------
        # [Stage 1: VAE Encoding execution]
        # ------------------------------------------------------------------
        print("[ENCODE] Encoding images...")

        # Start inference time measurement
        if measure_inference_time:
            inference_start_time = time.time()
            print("[TIME] Starting inference time measurement...")

        # Encode non-defective image
        good_latents_orig, _ = self.encode_images(good_image, bad_images[0])  # Only need good_latents

        # Encode all defect images
        bad_latents_list = []
        print(f"   Encoding {len(bad_images)} defect images...")
        for i, bad_img in enumerate(bad_images):
            _, bad_latents = self.encode_images(good_image, bad_img)  # Only need bad_latents

            # Apply defect variation to latent representation
            if defect_variation > 0.0:
                bad_latents = self.add_defect_variation(bad_latents, defect_variation)
                print(f"   [SUCCESS] Encoded defect image {i+1} (with variation)")
            else:
                print(f"   [SUCCESS] Encoded defect image {i+1}")

            bad_latents_list.append(bad_latents)

        # Select first as primary reference (for initialization)
        primary_bad_latents = bad_latents_list[0]

        # 5. Set time steps
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps

        # 6. Calculate stopping point for partial forward diffusion
        # ------------------------------------------------------------------
        # [Stage 3: Partial Forward Diffusion execution]
        # ------------------------------------------------------------------
        t_stop = int(num_inference_steps * (1 - r))
        print(f"[FORWARD] Partial forward diffusion: stopping at step {t_stop} (r={r})")

        # 7. Generate identical noise
        noise_shape = good_latents_orig.shape
        noise = torch.randn(noise_shape, device=self.device, dtype=good_latents_orig.dtype)

        # 8. Partial forward diffusion: only add partial noise
        if t_stop > 0:
            timestep_start = timesteps[t_stop]

            # Add different noise for each defect image (increase variation)
            if defect_variation > 0.0:
                # Add base noise to good_latents
                good_latents = self.add_noise(good_latents_orig, timestep_start, noise)

                # Add slightly different noise to each bad_latents
                varied_bad_latents_list = []
                for i, bad_lat in enumerate(bad_latents_list):
                    # Generate slightly different noise
                    variation_noise = noise + torch.randn_like(noise) * defect_variation * 0.1
                    varied_bad_lat = self.add_noise(bad_lat, timestep_start, variation_noise)
                    varied_bad_latents_list.append(varied_bad_lat)

                bad_latents_list = varied_bad_latents_list
                bad_latents = bad_latents_list[0]  # Update primary reference
                print(f"   Added varied noise up to timestep {timestep_start}")
            else:
                good_latents = self.add_noise(good_latents_orig, timestep_start, noise)
                bad_latents = self.add_noise(primary_bad_latents, timestep_start, noise)
                print(f"   Added noise up to timestep {timestep_start}")
        else:
            # If r=1, start from original images (no noise)
            good_latents = good_latents_orig.clone()
            bad_latents = primary_bad_latents.clone()
            print("   Starting from original images (no noise)")

        # 9. Reverse diffusion process (starting from t_stop)
        # ------------------------------------------------------------------
        # [Stage 4: Attention-Guided Reverse Optimization execution]
        # ------------------------------------------------------------------
        print("[DIFFUSION] Starting reverse diffusion with contrastive optimization...")

        # Only process time steps from t_stop to 0
        active_timesteps = timesteps[t_stop:]

        for i, timestep in enumerate(active_timesteps):
            step_num = i + 1
            total_steps = len(active_timesteps)
            print(f"  Step {step_num}/{total_steps}: timestep {timestep} (from t_stop={t_stop})")

            # Prepare input
            good_latents = good_latents.detach().requires_grad_(True)

            # Combine latents for batch processing
            latent_model_input = torch.cat([good_latents, bad_latents], dim=0)

            # Combine text embeddings
            text_embeddings = torch.cat([product_embeddings, anomaly_embeddings], dim=0)

            # Prepare input for inpainting model
            # Need to resize mask to latent space and add to input
            _, _, h, w = latent_model_input.shape

            # Create combined mask (good_mask for good_latents, combined_bad_mask for bad_latents)
            good_mask_latent = F.interpolate(good_mask.float(), size=(h, w), mode='nearest')
            bad_mask_latent = F.interpolate(combined_bad_mask.float(), size=(h, w), mode='nearest')
            combined_mask = torch.cat([good_mask_latent, bad_mask_latent], dim=0).to(latent_model_input.dtype)

            # Create masked latents (for inpainting)
            masked_latents = latent_model_input * (1 - combined_mask)

            # Combine input: latents + mask + masked_latents
            inpaint_input = torch.cat([latent_model_input, combined_mask, masked_latents], dim=1)

            # UNet prediction
            with torch.no_grad():
                noise_pred = self.unet(
                    inpaint_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False
                )[0]

            # Separate prediction results
            noise_pred_good, noise_pred_bad = noise_pred.chunk(2)

            # Optimization step
            if step_num % optimization_interval == 0:
                print(f"    [OPTIMIZE] Optimizing attention at step {step_num}...")

                # Get actual mask position at current step
                _, current_actual_masks = self.apply_defect_with_placement(
                    good_latents, bad_latents_list, bad_masks, good_mask, alignment_info, step_num, num_optimization_steps
                )

                # Optimize each defect image separately
                for defect_idx in range(len(bad_masks)):
                    print(f"      Optimizing defect {defect_idx+1}/{len(bad_masks)}...")

                    # Select current defect image and actually used mask
                    current_bad_latents = bad_latents_list[defect_idx]
                    current_bad_mask = current_actual_masks[defect_idx]  # Use mask at actual position

                    for opt_step in range(num_optimization_steps):
                        # Recalculate UNet output to get attention maps
                        latent_model_input = torch.cat([good_latents, current_bad_latents], dim=0)

                        # Prepare input for inpainting model
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

                        # Extract attention maps
                        attention_maps = self.extract_attention_maps_from_unet(latent_model_input, text_embeddings)

                        # Compute contrastive loss (using mask at actual position)
                        contrastive_loss = self.compute_contrastive_loss(
                            attention_maps,
                            product_token_indices,
                            anomaly_token_indices,
                            good_mask,
                            current_actual_masks,
                            current_defect_idx=defect_idx
                        )

                        print(f"        Defect {defect_idx+1} step {opt_step+1}: Loss = {contrastive_loss.item():.6f}")

                        # Backpropagation
                        if contrastive_loss.requires_grad:
                            contrastive_loss.backward(retain_graph=True)

                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(good_latents, max_norm=1.0)

                            # Update good_latents
                            with torch.no_grad():
                                good_latents -= learning_rate * good_latents.grad
                                good_latents.grad.zero_()

            # Scheduler step
            with torch.no_grad():
                good_latents = self.pipe.scheduler.step(noise_pred_good, timestep, good_latents).prev_sample
                bad_latents = self.pipe.scheduler.step(noise_pred_bad, timestep, bad_latents).prev_sample

                # Transfer defect features (with feature alignment support)
                good_latents, actual_bad_masks = self.apply_defect_with_placement(
                    good_latents, bad_latents_list, bad_masks, good_mask, alignment_info, i, num_inference_steps
                )

        # 8. Decode final result
        # ------------------------------------------------------------------
        # [Stage 5: Decoding execution]
        # ------------------------------------------------------------------
        print("[DECODE] Decoding final result...")
        with torch.no_grad():
            final_image = self.vae.decode(good_latents / self.vae.config.scaling_factor).sample
            final_image = (final_image / 2 + 0.5).clamp(0, 1)
            final_image = final_image.cpu().permute(0, 2, 3, 1).numpy()[0]
            final_image = (final_image * 255).astype(np.uint8)

        # End inference time measurement
        if measure_inference_time:
            inference_end_time = time.time()
            total_inference_time = inference_end_time - inference_start_time
            inference_times.append(total_inference_time)
            print(f"[TIME] Inference completed in {total_inference_time:.2f} seconds")

        # Prepare original image for blending
        original_good_image = good_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        original_good_image = (original_good_image * 255).astype(np.uint8)

        # Prepare defect mask for blending (intersection with normal image mask)
        print("[OPTIMIZE] Optimizing final mask with object mask intersection...")

        # Resize good_mask to same dimensions as combined_bad_mask
        good_mask_resized = F.interpolate(good_mask.float(), size=combined_bad_mask.shape[-2:], mode='nearest')

        # Calculate intersection: only keep defects within normal image object region
        optimized_mask = combined_bad_mask * good_mask_resized.squeeze(0).squeeze(0)

        # Convert to numpy
        defect_mask_np = optimized_mask.cpu().numpy()
        defect_mask_np = (defect_mask_np * 255).astype(np.uint8)

        print(f"   Original defect pixels: {torch.sum(combined_bad_mask > 0.5).item()}")
        print(f"   Optimized defect pixels: {torch.sum(optimized_mask > 0.5).item()}")
        print(f"   Removed pixels outside object: {torch.sum(combined_bad_mask > 0.5).item() - torch.sum(optimized_mask > 0.5).item()}")

        # Prepare reference defect image (use first defect image)
        reference_bad_image = bad_images[0].cpu().permute(0, 2, 3, 1).numpy()[0]
        reference_bad_image = (reference_bad_image * 255).astype(np.uint8)

        # 9. Create blended images (feathered and non-feathered versions)
        print("[BLEND] Creating blend images...")

        # Non-feathered blended image
        print(f"   Creating non-feathered blend image...")
        non_feathered_image = self.blend_images_with_feathered_mask(
            original_image=original_good_image,
            generated_image=final_image,
            mask=defect_mask_np,
            feather_radius=0  # No feathering
        )

        # Feathered blended image (if feather radius > 0)
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

        # 10. Save results
        print("[SAVE] Saving results...")
        file_paths = {}

        # Save final generated defect image
        final_image_pil = Image.fromarray(final_image)
        final_path = os.path.join(output_dir, "contrastive_defect_image.png")
        final_image_pil.save(final_path)
        file_paths["final_defect_image"] = final_path

        # Save non-feathered blended image
        non_feathered_pil = Image.fromarray(non_feathered_image)
        non_feathered_path = os.path.join(output_dir, "non_feathered_blend_image.png")
        non_feathered_pil.save(non_feathered_path)
        file_paths["non_feathered_blend_image"] = non_feathered_path

        # Save feathered blended image
        feathered_pil = Image.fromarray(feathered_image)
        feathered_path = os.path.join(output_dir, "feathered_blend_image.png")
        feathered_pil.save(feathered_path)
        file_paths["feathered_blend_image"] = feathered_path

        # Create and save comparison grid image
        print("[GRID] Creating comparison grid...")
        comparison_grid = self.create_comparison_grid(
            original_image=original_good_image,
            generated_image=final_image,
            blended_image=feathered_image,  # Use feathered version as main display
            reference_bad_image=reference_bad_image,
            defect_mask=defect_mask_np,
            non_feathered_image=non_feathered_image  # Add non-feathered version
        )
        comparison_grid_pil = Image.fromarray(comparison_grid)
        comparison_path = os.path.join(output_dir, "comparison_grid.png")
        comparison_grid_pil.save(comparison_path)
        file_paths["comparison_grid"] = comparison_path

        # Save copy of input image for comparison
        good_image_pil = Image.open(good_image_path)
        good_copy_path = os.path.join(output_dir, "original_good_image.png")
        good_image_pil.save(good_copy_path)
        file_paths["original_good"] = good_copy_path

        # Save reference defect image (use first one)
        bad_image_pil = Image.open(bad_image_paths[0])
        bad_copy_path = os.path.join(output_dir, "reference_bad_image.png")
        bad_image_pil.save(bad_copy_path)
        file_paths["reference_bad"] = bad_copy_path

        # Save copies of all defect images
        for i, bad_img_path in enumerate(bad_image_paths):
            bad_img_pil = Image.open(bad_img_path)
            bad_img_copy_path = os.path.join(output_dir, f"bad_image_{i+1}.png")
            bad_img_pil.save(bad_img_copy_path)
            file_paths[f"bad_image_{i+1}"] = bad_img_copy_path

        # Save mask copy
        good_mask_pil = Image.open(good_mask_path)
        good_mask_copy_path = os.path.join(output_dir, "good_object_mask.png")
        good_mask_pil.save(good_mask_copy_path)
        file_paths["good_mask"] = good_mask_copy_path

        # Save copies of all defect masks
        for i, bad_mask_path in enumerate(bad_mask_paths):
            bad_mask_pil = Image.open(bad_mask_path)
            bad_mask_copy_path = os.path.join(output_dir, f"bad_defect_mask_{i+1}.png")
            bad_mask_pil.save(bad_mask_copy_path)
            file_paths[f"bad_mask_{i+1}"] = bad_mask_copy_path

        # Save combined defect mask (using IoA-optimized mask, consistent with comparison_grid)
        # Note: Using defect_mask_np instead of combined_bad_mask because defect_mask_np already intersected with object mask
        # Ensure mask is 2D array
        defect_mask_2d = defect_mask_np.squeeze() if len(defect_mask_np.shape) > 2 else defect_mask_np
        combined_mask_pil = Image.fromarray(defect_mask_2d)
        combined_mask_path = os.path.join(output_dir, "combined_defect_mask.png")
        combined_mask_pil.save(combined_mask_path)
        file_paths["combined_defect_mask"] = combined_mask_path

        # Extract attention heatmaps
        if extract_attention and defect_types:
            print("[ATTENTION] Extracting attention heatmaps...")
            try:
                # Parse anomaly tokens from prompt
                prompt_parts = prompt.split()
                anomaly_tokens = []

                # Extract possible anomaly tokens from prompt
                product_tokens = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                                "leather", "metal", "nut", "pill", "screw", "tile",
                                "toothbrush", "transistor", "wood", "zipper"]

                for token in prompt_parts:
                    if token.lower() not in product_tokens and len(token) > 2:
                        anomaly_tokens.append(token)

                if not anomaly_tokens:
                    anomaly_tokens = defect_types

                # Generate experiment name
                experiment_name = f"exp_{len(bad_image_paths)}defects"

                # Extract attention heatmaps
                heatmap_path = extract_attention_heatmaps(
                    self, prompt, anomaly_tokens, defect_types,
                    experiment_name, output_dir
                )

                if heatmap_path:
                    file_paths["attention_heatmap"] = heatmap_path

            except Exception as e:
                print(f"[WARNING] Could not extract attention heatmaps: {e}")

        # Save inference time
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
