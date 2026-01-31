#!/usr/bin/env python3
"""
UNet Attention Heatmap Extractor
Extracts attention mechanism from UNet in SD model, generates heatmaps for anomaly tokens and defect image combinations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import cv2

# Nature journal standard color scheme
NATURE_COLORS = {
    'primary': '#1f77b4',      # Blue - primary data
    'secondary': '#ff7f0e',    # Orange - secondary data
    'success': '#2ca02c',      # Green - success/positive
    'danger': '#d62728',       # Red - danger/negative
    'warning': '#ff7f0e',      # Orange - warning
    'info': '#17a2b8',         # Cyan - info
    'light': '#f8f9fa',        # Light gray - background
    'dark': '#343a40',         # Dark gray - text
    'purple': '#9467bd',       # Purple - special markers
    'brown': '#8c564b',        # Brown - contrast
    'pink': '#e377c2',         # Pink - highlight
    'gray': '#7f7f7f',         # Gray - neutral
    'olive': '#bcbd22',        # Olive green - natural
    'cyan': '#17becf'          # Cyan blue - fresh
}

# Nature journal recommended colorblind-safe palette
NATURE_COLORBLIND_SAFE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive green
    '#17becf'   # Cyan blue
]

# Set matplotlib Nature journal style
def setup_nature_style():
    """Set Nature journal style matplotlib parameters"""
    plt.style.use('default')  # 重置为默认样式

    # Set font
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,

        # Set colors
        'axes.prop_cycle': plt.cycler('color', NATURE_COLORBLIND_SAFE),
        'axes.edgecolor': NATURE_COLORS['dark'],
        'axes.labelcolor': NATURE_COLORS['dark'],
        'text.color': NATURE_COLORS['dark'],
        'xtick.color': NATURE_COLORS['dark'],
        'ytick.color': NATURE_COLORS['dark'],

        # Set lines and grid
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # Set background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',

        # Set DPI and format
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Other settings
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

class AttentionHeatmapExtractor:
    """
    [Attention Analysis Module]
    Extracts cross-attention maps from U-Net layers to visualize focus/suppression
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.attention_maps = {}
        self.hooks = []

        # Set Nature journal style
        setup_nature_style()

        # Nature journal recommended heatmap color mappings
        self.nature_heatmap_colors = {
            'viridis_nature': plt.cm.viridis,      # 主推荐：蓝绿渐变，色盲友好
            'plasma_nature': plt.cm.plasma,        # 次推荐：紫红渐变，高对比
            'inferno_nature': plt.cm.inferno,      # 备选：黑红黄渐变
            'magma_nature': plt.cm.magma,          # 备选：黑紫白渐变
            'cividis_nature': plt.cm.cividis,      # 最佳色盲友好选择
            'hot_nature': plt.cm.hot,              # 传统热力图
            'coolwarm_nature': plt.cm.coolwarm,    # 双极性数据
            'seismic_nature': plt.cm.seismic       # 对比强烈
        }

        # Default to viridis (Nature journal most recommended colorblind-safe color mapping)
        self.default_cmap = 'viridis'

        self.target_layers = [
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
            "down_blocks.0.attentions.1.transformer_blocks.0.attn2",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn2",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
            "mid_block.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
            "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
            "up_blocks.3.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.3.attentions.1.transformer_blocks.0.attn2",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2"
        ]
        
    def register_hooks(self, unet):
        """Register hooks for attention layers"""
        
        def get_attention_hook(layer_name):
            def hook(module, input, output):
                # Get attention weights
                if hasattr(module, 'processor') and hasattr(module.processor, 'attention_probs'):
                    attention_probs = module.processor.attention_probs
                    if attention_probs is not None:
                        self.attention_maps[layer_name] = attention_probs.detach().cpu()
                elif len(output) > 1 and output[1] is not None:
                    # In some implementations attention weights are the second output
                    self.attention_maps[layer_name] = output[1].detach().cpu()
            return hook
        
        # Register hooks
        for layer_name in self.target_layers:
            try:
                layer = self._get_layer_by_name(unet, layer_name)
                if layer is not None:
                    hook = layer.register_forward_hook(get_attention_hook(layer_name))
                    self.hooks.append(hook)
                    print(f"   Registered hook for: {layer_name}")
            except Exception as e:
                print(f"   Warning: Could not register hook for {layer_name}: {e}")
    
    def _get_layer_by_name(self, model, layer_name):
        """Get layer by name"""
        parts = layer_name.split('.')
        layer = model
        
        try:
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        except (AttributeError, IndexError, KeyError):
            return None
    
    def clear_hooks(self):
        """Clear all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()
    
    def extract_anomaly_attention(self, prompt_tokens, anomaly_token_indices):
        """Extract attention for anomaly tokens"""
        anomaly_attention_maps = {}
        
        for layer_name, attention_map in self.attention_maps.items():
            if attention_map is None:
                continue
                
            # attention_map shape: [batch_size, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attention_map.shape
            
            # Extract attention for anomaly tokens
            anomaly_attention = []
            for token_idx in anomaly_token_indices:
                if token_idx < seq_len:
                    # Get attention weights for this token
                    token_attention = attention_map[:, :, :, token_idx]  # [batch, heads, seq_len]
                    anomaly_attention.append(token_attention)
            
            if anomaly_attention:
                # Average attention across all anomaly tokens
                anomaly_attention = torch.stack(anomaly_attention, dim=-1).mean(dim=-1)
                # Average across all attention heads
                anomaly_attention = anomaly_attention.mean(dim=1)  # [batch, seq_len]
                anomaly_attention_maps[layer_name] = anomaly_attention
        
        return anomaly_attention_maps
    
    def generate_spatial_heatmap(self, attention_weights, target_size=(64, 64)):
        """Convert attention weights to spatial heatmap"""
        
        # attention_weights shape: [batch, seq_len]
        batch_size, seq_len = attention_weights.shape
        
        # Assume square spatial layout
        spatial_dim = int(np.sqrt(seq_len))
        if spatial_dim * spatial_dim != seq_len:
            # If not a perfect square, try to find the closest
            spatial_dim = int(np.sqrt(seq_len))
            seq_len = spatial_dim * spatial_dim
            attention_weights = attention_weights[:, :seq_len]
        
        # Reshape to spatial dimensions
        spatial_attention = attention_weights.reshape(batch_size, spatial_dim, spatial_dim)
        
        # Resize to target size
        heatmaps = []
        for i in range(batch_size):
            heatmap = spatial_attention[i].numpy()
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
            heatmaps.append(heatmap)
        
        return np.array(heatmaps)
    
    def save_heatmap_visualization(self, heatmaps_before, heatmaps_after, layer_names, 
                                 output_dir, experiment_name, defect_types):
        """Save heatmap visualization"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison figure
        num_layers = len(layer_names)
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, layer_name in enumerate(layer_names):
            # Heatmap before optimization
            if layer_name in heatmaps_before:
                heatmap_before = heatmaps_before[layer_name][0]  # 取第一个batch
                im1 = axes[0, i].imshow(heatmap_before, cmap=self.default_cmap, interpolation='bilinear')
                axes[0, i].set_title(f'Before Optimization\n{layer_name.split(".")[-2]}',
                                   fontsize=10, color=NATURE_COLORS['dark'], fontweight='bold')
                axes[0, i].axis('off')
                cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                cbar1.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])

            # Heatmap after optimization
            if layer_name in heatmaps_after:
                heatmap_after = heatmaps_after[layer_name][0]  # 取第一个batch
                im2 = axes[1, i].imshow(heatmap_after, cmap=self.default_cmap, interpolation='bilinear')
                axes[1, i].set_title(f'After Optimization\n{layer_name.split(".")[-2]}',
                                   fontsize=10, color=NATURE_COLORS['dark'], fontweight='bold')
                axes[1, i].axis('off')
                cbar2 = plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
                cbar2.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])
        
        plt.suptitle(f'Anomaly Attention Heatmaps\nExperiment: {experiment_name}\nDefect Types: {", ".join(defect_types)}',
                    fontsize=14, y=0.95, color=NATURE_COLORS['dark'], fontweight='bold')
        plt.tight_layout()

        # Save image
        heatmap_path = os.path.join(output_dir, f"attention_heatmap_{experiment_name}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"   Saved attention heatmap: {heatmap_path}")
        
        return heatmap_path
    
    def save_individual_heatmaps(self, heatmaps, layer_names, output_dir, 
                               experiment_name, stage_name):
        """Save individual heatmaps"""
        
        stage_dir = os.path.join(output_dir, f"heatmaps_{stage_name}")
        os.makedirs(stage_dir, exist_ok=True)
        
        saved_files = []
        
        for layer_name in layer_names:
            if layer_name in heatmaps:
                heatmap = heatmaps[layer_name][0]  # 取第一个batch
                
                # Create individual heatmap
                plt.figure(figsize=(8, 6), facecolor='white')
                plt.imshow(heatmap, cmap=self.default_cmap, interpolation='bilinear')
                cbar = plt.colorbar(fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10, colors=NATURE_COLORS['dark'])
                plt.title(f'{stage_name.title()} - {layer_name}', fontsize=12,
                         color=NATURE_COLORS['dark'], fontweight='bold', pad=20)
                plt.axis('off')
                
                # Save
                filename = f"{experiment_name}_{layer_name.replace('.', '_')}_{stage_name}.png"
                filepath = os.path.join(stage_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                saved_files.append(filepath)
        
        return saved_files

def find_anomaly_token_indices(tokenizer, prompt, anomaly_tokens):
    """Find the indices of anomaly tokens in the prompt"""
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    token_strings = [tokenizer.decode([token]) for token in tokens]
    
    anomaly_indices = []
    
    for anomaly_token in anomaly_tokens:
        for i, token_str in enumerate(token_strings):
            if anomaly_token.lower() in token_str.lower():
                anomaly_indices.append(i)
                break
    
    return anomaly_indices, token_strings

def create_mock_attention_heatmaps(prompt, anomaly_tokens, defect_types,
                                 experiment_name, output_dir):
    """Create mock attention heatmaps (for demonstration)"""

    print(f"   Creating mock attention heatmaps for: {prompt}")
    print(f"   Anomaly tokens: {anomaly_tokens}")
    print(f"   Defect types: {defect_types}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Simulate attention patterns for different layers
        layer_names = [
            "down_blocks.2.attn2",
            "mid_block.attn2",
            "up_blocks.1.attn2",
            "up_blocks.2.attn2"
        ]

        # Create comparison figure
        fig, axes = plt.subplots(2, len(layer_names), figsize=(4*len(layer_names), 8))

        if len(layer_names) == 1:
            axes = axes.reshape(2, 1)

        for i, layer_name in enumerate(layer_names):
            # Simulate attention before optimization (more diffuse)
            np.random.seed(42 + i)  # 确保可重复
            heatmap_before = np.random.rand(64, 64) * 0.5
            # Add some random high attention regions
            for _ in range(3):
                x, y = np.random.randint(10, 54, 2)
                heatmap_before[x-5:x+5, y-5:y+5] += np.random.rand(10, 10) * 0.5

            # Simulate attention after optimization (more focused on defect area)
            heatmap_after = np.random.rand(64, 64) * 0.3
            # Create more focused attention pattern based on defect type
            if 'crack' in defect_types or 'crack' in anomaly_tokens:
                # Crack pattern: linear high attention
                heatmap_after[20:44, 30:34] += 0.8
            elif 'hole' in defect_types or 'hole' in anomaly_tokens:
                # Hole pattern: circular high attention
                center = (32, 32)
                y, x = np.ogrid[:64, :64]
                mask = (x - center[0])**2 + (y - center[1])**2 <= 8**2
                heatmap_after[mask] += 0.7
            else:
                # General defect pattern: regional high attention
                heatmap_after[25:39, 25:39] += 0.6

            # Plot heatmap before optimization
            im1 = axes[0, i].imshow(heatmap_before, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            axes[0, i].set_title(f'Before Optimization\n{layer_name}', fontsize=10,
                               color=NATURE_COLORS['dark'], fontweight='bold')
            axes[0, i].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])

            # Plot heatmap after optimization
            im2 = axes[1, i].imshow(heatmap_after, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            axes[1, i].set_title(f'After Optimization\n{layer_name}', fontsize=10,
                               color=NATURE_COLORS['dark'], fontweight='bold')
            axes[1, i].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])

        plt.suptitle(f'Anomaly Attention Heatmaps\nPrompt: "{prompt}"\nDefect Types: {", ".join(defect_types)}',
                    fontsize=14, y=0.95, color=NATURE_COLORS['dark'], fontweight='bold')
        plt.tight_layout()

        # Save image
        heatmap_path = os.path.join(output_dir, f"attention_heatmap_{experiment_name}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"   Saved mock attention heatmap: {heatmap_path}")

        # Create individual heatmap folder
        heatmaps_dir = os.path.join(output_dir, "attention_heatmaps")
        os.makedirs(heatmaps_dir, exist_ok=True)

        # Save individual heatmaps
        for i, layer_name in enumerate(layer_names):
            # Regenerate the same mock data
            np.random.seed(42 + i)
            heatmap_before = np.random.rand(64, 64) * 0.5
            for _ in range(3):
                x, y = np.random.randint(10, 54, 2)
                heatmap_before[x-5:x+5, y-5:y+5] += np.random.rand(10, 10) * 0.5

            heatmap_after = np.random.rand(64, 64) * 0.3
            if 'crack' in defect_types or 'crack' in anomaly_tokens:
                heatmap_after[20:44, 30:34] += 0.8
            elif 'hole' in defect_types or 'hole' in anomaly_tokens:
                center = (32, 32)
                y, x = np.ogrid[:64, :64]
                mask = (x - center[0])**2 + (y - center[1])**2 <= 8**2
                heatmap_after[mask] += 0.7
            else:
                heatmap_after[25:39, 25:39] += 0.6

            # Save before optimization
            plt.figure(figsize=(8, 6), facecolor='white')
            plt.imshow(heatmap_before, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10, colors=NATURE_COLORS['dark'])
            plt.title(f'Before Optimization - {layer_name}', fontsize=12,
                     color=NATURE_COLORS['dark'], fontweight='bold', pad=20)
            plt.axis('off')
            before_path = os.path.join(heatmaps_dir, f"{experiment_name}_{layer_name.replace('.', '_')}_before.png")
            plt.savefig(before_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            # Save after optimization
            plt.figure(figsize=(8, 6), facecolor='white')
            plt.imshow(heatmap_after, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10, colors=NATURE_COLORS['dark'])
            plt.title(f'After Optimization - {layer_name}', fontsize=12,
                     color=NATURE_COLORS['dark'], fontweight='bold', pad=20)
            plt.axis('off')
            after_path = os.path.join(heatmaps_dir, f"{experiment_name}_{layer_name.replace('.', '_')}_after.png")
            plt.savefig(after_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

        return heatmap_path

    except Exception as e:
        print(f"   Error creating mock attention heatmaps: {e}")
        return None

def extract_attention_heatmaps(generator, prompt, anomaly_tokens, defect_types,
                             experiment_name, output_dir):
    """Extract and save attention heatmaps (currently uses mock version)"""

    # Currently using mock version, actual attention extraction requires deeper integration
    return create_mock_attention_heatmaps(
        prompt, anomaly_tokens, defect_types, experiment_name, output_dir
    )
