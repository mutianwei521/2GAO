[English](CUSTOM_DATASET.md) | **中文**

# 自定义数据集使用教程

## 1. 概述

2-GAO 是一个**无需训练**的缺陷生成框架，只需要最少量的数据：
- **1张好图**（无缺陷）及其物体mask
- **1张或多张缺陷图**及其缺陷mask

本教程将介绍如何准备自定义数据集并运行批量生成。

## 2. 数据目录结构

您的数据集应遵循以下结构：

```
your_dataset/
├── good/
│   ├── image1.png           # 好图（无缺陷）
│   ├── image1_mask.png      # 物体mask（白色=物体区域）
│   ├── image2.png
│   ├── image2_mask.png
│   └── ...
└── bad/
    ├── defect_type_1/       # 文件夹名称 = 缺陷类型
    │   ├── img001.png       # 缺陷图
    │   ├── img001_mask.png  # 缺陷mask（白色=缺陷区域）
    │   ├── img002.png
    │   ├── img002_mask.png
    │   └── ...
    ├── defect_type_2/
    │   ├── img001.png
    │   ├── img001_mask.png
    │   └── ...
    └── ...
```

## 3. 图像和Mask要求

| 要求 | 规格 |
|------|------|
| **图像格式** | PNG 或 JPG |
| **Mask格式** | PNG（灰度图） |
| **Mask命名** | `{图像名}_mask.png` |
| **Mask内容** | 二值：0（黑色）= 背景，255（白色）= 感兴趣区域 |
| **好图Mask** | 应覆盖整个物体区域 |
| **缺陷Mask** | 应仅覆盖缺陷区域 |

## 4. Mask准备建议

1. **物体Mask（好图）**：使用 [Segment Anything Model (SAM)](https://segment-anything.com/) 进行自动分割，或在图像编辑软件中手动绘制。

2. **缺陷Mask**：
   - 精确绘制缺陷边界
   - 仅包含缺陷区域，不包含整个物体
   - 可选择使用羽化边缘以获得更好的融合效果

3. **质量检查**：确保mask与对应图像完美对齐。

## 5. 运行批量生成

```bash
# 单类别，单缺陷数量
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "您的产品名" \
    --num-experiments 50 \
    --defect-counts 1

# 多缺陷数量
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "您的产品名" \
    --num-experiments 50 \
    --defect-counts 1,2,3,4

# 自定义输出目录
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "您的产品名" \
    --output outputs_custom \
    --num-experiments 100
```

## 6. 配置选项

编辑 `batch_generate_custom.py` 顶部的配置部分：

```python
# ==================== 用户配置 ====================

# 数据集路径
DATASET_ROOT = "your_dataset"

# 物体名称（用于生成prompt）
OBJECT_NAME = "product"

# 缺陷类型映射（文件夹名 -> prompt词）
DEFECT_TYPE_MAPPING = {
    "scratch": "scratch",      # 划痕
    "crack": "crack",          # 裂纹
    "stain": "stain",          # 污渍
    # 在此添加您的缺陷类型
}

# 生成参数
GENERATION_CONFIG = {
    'num_inference_steps': 100,      # 扩散步数
    'r': 0.25,                       # 保留系数
    'num_optimization_steps': 25,    # 注意力优化步数
    'optimization_interval': 5,      # 优化间隔
    'feather_radius': 10,            # 融合羽化半径
}

# 特征对齐 (IoA)
FEATURE_CONFIG = {
    'enable_feature_alignment': True,
    'ioa_threshold': 0.5,
}

# 输出保存选项
SAVE_CONFIG = {
    'save_feathered_blend': True,
    'save_comparison_grid': True,
    'save_defect_heatmaps': True,
    # ... 更多选项
}

# ==================== 配置结束 ====================
```

## 7. 输出结构

生成的文件按以下结构组织：

```
outputs_custom/
├── feathered_blend/          # 最终融合图像
├── combined_defect_masks/    # 生成的缺陷mask
├── defect_heatmaps/          # 缺陷概率热力图
├── comparison_grid/          # 前后对比图
├── original_good/            # 源好图
├── reference_bad/            # 源缺陷图
└── ...
```

## 8. 常见问题

| 问题 | 解决方案 |
|------|----------|
| "No valid image pairs found" | 检查mask命名规范（`{name}_mask.png`） |
| 缺陷出现在物体外部 | 增加IoA阈值或检查物体mask |
| 缺陷质量差 | 在DEFECT_TYPE_MAPPING中使用更具描述性的prompt |
| 内存不足 | 减少批量大小或使用更小的图像 |

---

## 许可证

本项目使用 MIT 许可证发布。
