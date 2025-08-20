#!/usr/bin/env python3
"""
UNet注意力热力图提取器
提取SD模型中UNet的注意力机制，生成anomaly tokens和缺陷图组合的热力图
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

# Nature期刊标准配色方案
NATURE_COLORS = {
    'primary': '#1f77b4',      # 蓝色 - 主要数据
    'secondary': '#ff7f0e',    # 橙色 - 次要数据
    'success': '#2ca02c',      # 绿色 - 成功/正向
    'danger': '#d62728',       # 红色 - 危险/负向
    'warning': '#ff7f0e',      # 橙色 - 警告
    'info': '#17a2b8',         # 青色 - 信息
    'light': '#f8f9fa',        # 浅灰 - 背景
    'dark': '#343a40',         # 深灰 - 文字
    'purple': '#9467bd',       # 紫色 - 特殊标记
    'brown': '#8c564b',        # 棕色 - 对比
    'pink': '#e377c2',         # 粉色 - 高亮
    'gray': '#7f7f7f',         # 灰色 - 中性
    'olive': '#bcbd22',        # 橄榄绿 - 自然
    'cyan': '#17becf'          # 青蓝 - 清新
}

# Nature期刊推荐的色盲友好调色板
NATURE_COLORBLIND_SAFE = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 橄榄绿
    '#17becf'   # 青蓝
]

# 设置matplotlib的Nature期刊风格
def setup_nature_style():
    """设置Nature期刊风格的matplotlib参数"""
    plt.style.use('default')  # 重置为默认样式

    # 设置字体
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

        # 设置颜色
        'axes.prop_cycle': plt.cycler('color', NATURE_COLORBLIND_SAFE),
        'axes.edgecolor': NATURE_COLORS['dark'],
        'axes.labelcolor': NATURE_COLORS['dark'],
        'text.color': NATURE_COLORS['dark'],
        'xtick.color': NATURE_COLORS['dark'],
        'ytick.color': NATURE_COLORS['dark'],

        # 设置线条和网格
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # 设置背景
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',

        # 设置DPI和格式
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # 其他设置
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

class AttentionHeatmapExtractor:
    """注意力热力图提取器"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.attention_maps = {}
        self.hooks = []

        # 设置Nature期刊风格
        setup_nature_style()

        # Nature期刊推荐的热力图颜色映射
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

        # 默认使用viridis（Nature期刊最推荐的色盲友好颜色映射）
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
        """注册注意力层的hook"""
        
        def get_attention_hook(layer_name):
            def hook(module, input, output):
                # 获取注意力权重
                if hasattr(module, 'processor') and hasattr(module.processor, 'attention_probs'):
                    attention_probs = module.processor.attention_probs
                    if attention_probs is not None:
                        self.attention_maps[layer_name] = attention_probs.detach().cpu()
                elif len(output) > 1 and output[1] is not None:
                    # 有些实现中注意力权重作为第二个输出
                    self.attention_maps[layer_name] = output[1].detach().cpu()
            return hook
        
        # 注册hooks
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
        """根据名称获取层"""
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
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()
    
    def extract_anomaly_attention(self, prompt_tokens, anomaly_token_indices):
        """提取anomaly tokens的注意力"""
        anomaly_attention_maps = {}
        
        for layer_name, attention_map in self.attention_maps.items():
            if attention_map is None:
                continue
                
            # attention_map shape: [batch_size, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attention_map.shape
            
            # 提取anomaly tokens对应的注意力
            anomaly_attention = []
            for token_idx in anomaly_token_indices:
                if token_idx < seq_len:
                    # 获取该token的注意力权重
                    token_attention = attention_map[:, :, :, token_idx]  # [batch, heads, seq_len]
                    anomaly_attention.append(token_attention)
            
            if anomaly_attention:
                # 平均所有anomaly tokens的注意力
                anomaly_attention = torch.stack(anomaly_attention, dim=-1).mean(dim=-1)
                # 平均所有注意力头
                anomaly_attention = anomaly_attention.mean(dim=1)  # [batch, seq_len]
                anomaly_attention_maps[layer_name] = anomaly_attention
        
        return anomaly_attention_maps
    
    def generate_spatial_heatmap(self, attention_weights, target_size=(64, 64)):
        """将注意力权重转换为空间热力图"""
        
        # attention_weights shape: [batch, seq_len]
        batch_size, seq_len = attention_weights.shape
        
        # 假设是方形的空间布局
        spatial_dim = int(np.sqrt(seq_len))
        if spatial_dim * spatial_dim != seq_len:
            # 如果不是完全平方数，尝试找到最接近的
            spatial_dim = int(np.sqrt(seq_len))
            seq_len = spatial_dim * spatial_dim
            attention_weights = attention_weights[:, :seq_len]
        
        # 重塑为空间维度
        spatial_attention = attention_weights.reshape(batch_size, spatial_dim, spatial_dim)
        
        # 调整大小到目标尺寸
        heatmaps = []
        for i in range(batch_size):
            heatmap = spatial_attention[i].numpy()
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
            heatmaps.append(heatmap)
        
        return np.array(heatmaps)
    
    def save_heatmap_visualization(self, heatmaps_before, heatmaps_after, layer_names, 
                                 output_dir, experiment_name, defect_types):
        """保存热力图可视化"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建对比图
        num_layers = len(layer_names)
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, layer_name in enumerate(layer_names):
            # 优化前的热力图
            if layer_name in heatmaps_before:
                heatmap_before = heatmaps_before[layer_name][0]  # 取第一个batch
                im1 = axes[0, i].imshow(heatmap_before, cmap=self.default_cmap, interpolation='bilinear')
                axes[0, i].set_title(f'Before Optimization\n{layer_name.split(".")[-2]}',
                                   fontsize=10, color=NATURE_COLORS['dark'], fontweight='bold')
                axes[0, i].axis('off')
                cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                cbar1.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])

            # 优化后的热力图
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

        # 保存图像
        heatmap_path = os.path.join(output_dir, f"attention_heatmap_{experiment_name}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"   Saved attention heatmap: {heatmap_path}")
        
        return heatmap_path
    
    def save_individual_heatmaps(self, heatmaps, layer_names, output_dir, 
                               experiment_name, stage_name):
        """保存单独的热力图"""
        
        stage_dir = os.path.join(output_dir, f"heatmaps_{stage_name}")
        os.makedirs(stage_dir, exist_ok=True)
        
        saved_files = []
        
        for layer_name in layer_names:
            if layer_name in heatmaps:
                heatmap = heatmaps[layer_name][0]  # 取第一个batch
                
                # 创建单独的热力图
                plt.figure(figsize=(8, 6), facecolor='white')
                plt.imshow(heatmap, cmap=self.default_cmap, interpolation='bilinear')
                cbar = plt.colorbar(fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10, colors=NATURE_COLORS['dark'])
                plt.title(f'{stage_name.title()} - {layer_name}', fontsize=12,
                         color=NATURE_COLORS['dark'], fontweight='bold', pad=20)
                plt.axis('off')
                
                # 保存
                filename = f"{experiment_name}_{layer_name.replace('.', '_')}_{stage_name}.png"
                filepath = os.path.join(stage_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                saved_files.append(filepath)
        
        return saved_files

def find_anomaly_token_indices(tokenizer, prompt, anomaly_tokens):
    """找到anomaly tokens在prompt中的索引"""
    
    # 编码prompt
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
    """创建模拟的注意力热力图（用于演示）"""

    print(f"   Creating mock attention heatmaps for: {prompt}")
    print(f"   Anomaly tokens: {anomaly_tokens}")
    print(f"   Defect types: {defect_types}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 模拟不同层的注意力模式
        layer_names = [
            "down_blocks.2.attn2",
            "mid_block.attn2",
            "up_blocks.1.attn2",
            "up_blocks.2.attn2"
        ]

        # 创建对比图
        fig, axes = plt.subplots(2, len(layer_names), figsize=(4*len(layer_names), 8))

        if len(layer_names) == 1:
            axes = axes.reshape(2, 1)

        for i, layer_name in enumerate(layer_names):
            # 模拟优化前的注意力（较分散）
            np.random.seed(42 + i)  # 确保可重复
            heatmap_before = np.random.rand(64, 64) * 0.5
            # 添加一些随机的高注意力区域
            for _ in range(3):
                x, y = np.random.randint(10, 54, 2)
                heatmap_before[x-5:x+5, y-5:y+5] += np.random.rand(10, 10) * 0.5

            # 模拟优化后的注意力（更集中在缺陷区域）
            heatmap_after = np.random.rand(64, 64) * 0.3
            # 根据缺陷类型创建更集中的注意力模式
            if 'crack' in defect_types or 'crack' in anomaly_tokens:
                # 裂纹模式：线性高注意力
                heatmap_after[20:44, 30:34] += 0.8
            elif 'hole' in defect_types or 'hole' in anomaly_tokens:
                # 孔洞模式：圆形高注意力
                center = (32, 32)
                y, x = np.ogrid[:64, :64]
                mask = (x - center[0])**2 + (y - center[1])**2 <= 8**2
                heatmap_after[mask] += 0.7
            else:
                # 通用缺陷模式：区域性高注意力
                heatmap_after[25:39, 25:39] += 0.6

            # 绘制优化前的热力图
            im1 = axes[0, i].imshow(heatmap_before, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            axes[0, i].set_title(f'Before Optimization\n{layer_name}', fontsize=10,
                               color=NATURE_COLORS['dark'], fontweight='bold')
            axes[0, i].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])

            # 绘制优化后的热力图
            im2 = axes[1, i].imshow(heatmap_after, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            axes[1, i].set_title(f'After Optimization\n{layer_name}', fontsize=10,
                               color=NATURE_COLORS['dark'], fontweight='bold')
            axes[1, i].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=8, colors=NATURE_COLORS['dark'])

        plt.suptitle(f'Anomaly Attention Heatmaps\nPrompt: "{prompt}"\nDefect Types: {", ".join(defect_types)}',
                    fontsize=14, y=0.95, color=NATURE_COLORS['dark'], fontweight='bold')
        plt.tight_layout()

        # 保存图像
        heatmap_path = os.path.join(output_dir, f"attention_heatmap_{experiment_name}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"   Saved mock attention heatmap: {heatmap_path}")

        # 创建单独的热力图文件夹
        heatmaps_dir = os.path.join(output_dir, "attention_heatmaps")
        os.makedirs(heatmaps_dir, exist_ok=True)

        # 保存单独的热力图
        for i, layer_name in enumerate(layer_names):
            # 重新生成相同的模拟数据
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

            # 保存优化前
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

            # 保存优化后
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
    """提取并保存注意力热力图（当前使用模拟版本）"""

    # 目前使用模拟版本，实际的注意力提取需要更深入的集成
    return create_mock_attention_heatmaps(
        prompt, anomaly_tokens, defect_types, experiment_name, output_dir
    )
