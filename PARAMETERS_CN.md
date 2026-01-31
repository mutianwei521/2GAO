[English](PARAMETERS.md) | **中文**

# 2-GAO 参数参考手册

## 快速参考表

| 组件 | 参数 | 默认值 | 范围 | 描述 |
|------|------|--------|------|------|
| **扩散** | `num_inference_steps` | 100 | 50-150 | 去噪步数 |
| **扩散** | `r` | 0.25 | 0.15-0.35 | 保留系数 |
| **优化** | `num_optimization_steps` | 25 | 10-50 | 注意力优化迭代次数 |
| **优化** | `optimization_interval` | 5 | 1-10 | 每N步进行一次优化 |
| **优化** | `learning_rate` | 0.01 | 0.001-0.1 | 优化学习率 |
| **IoA** | `enable_feature_alignment` | True | True/False | 启用空间对齐 |
| **IoA** | `ioa_threshold` | 0.5 | 0.3-0.9 | IoA阈值 |
| **融合** | `feather_radius` | 10-15 | 0-30 | 羽化半径（像素） |
| **模型** | `model_id` | SD-Inpainting | - | Stable Diffusion模型 |
| **设备** | `device` | cuda | cuda/cpu | 计算设备 |

---

## 1. batch_generate.py 参数

### 1.1 保存配置 (SAVE_CONFIG)

控制输出文件的保存选项。

```python
SAVE_CONFIG = {
    'save_feathered_blend': True,        # 保存羽化融合图像
    'save_non_feathered_blend': True,    # 保存非羽化融合图像
    'save_comparison_grid': True,        # 保存对比网格图像
    'save_contrastive_defect': True,     # 保存对比缺陷图像
    'save_original_good': True,          # 保存原始好图
    'save_reference_bad': True,          # 保存参考坏图
    'save_good_object_masks': True,      # 保存好图物体mask
    'save_combined_defect_masks': True,  # 保存组合缺陷mask
    'save_bad_defect_masks': True,       # 保存坏图缺陷mask
    'save_defect_heatmaps': True,        # 保存缺陷热力图
    'save_attention_heatmaps': False,    # 保存注意力热力图（较慢）
    'save_other_files': True             # 保存其他辅助文件
}
```

### 1.2 功能配置 (FEATURE_CONFIG)

```python
FEATURE_CONFIG = {
    'enable_feature_alignment': True,    # 启用IoA特征对齐
    'ioa_threshold': 0.5,                # IoA对齐阈值
    'save_attention_heatmaps': False,    # 保存注意力热力图
    'measure_inference_time': True       # 测量推理时间
}
```

---

## 2. main_contrastive.py 参数

### 2.1 输入输出路径

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--good-dir` | str | `images/good` | 好图及物体mask目录 |
| `--bad-dir` | str | `images/bad` | 缺陷图及缺陷mask目录 |
| `--output-dir` | str | `outputs_contrastive2` | 输出目录 |
| `--prompt` | str | `bottle damaged` | 生成用的文本提示 |

### 2.2 扩散参数

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|------|------|--------|------|------|
| `--num-inference-steps` | int | **100** | 50-150 | 去噪步数。越高质量越好，速度越慢 |
| `--r` | float | **0.25** | 0.15-0.35 | 保留系数。越低变化越大 |

### 2.3 优化参数

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|------|------|--------|------|------|
| `--learning-rate` | float | **0.01** | 0.001-0.1 | 注意力优化学习率 |
| `--num-optimization-steps` | int | **25** | 10-50 | 每个间隔的优化迭代次数 |
| `--optimization-interval` | int | **5** | 1-10 | 每N个扩散步骤进行一次优化 |

### 2.4 特征对齐 (IoA)

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|------|------|--------|------|------|
| `--enable-feature-alignment` | flag | **True** | - | 启用IoA空间对齐 |
| `--ioa-threshold` | float | **0.5** | 0.3-0.9 | 有效缺陷放置的最小IoA |

**IoA阈值指南：**
- 0.3：宽松约束，允许部分重叠
- 0.5：平衡（推荐）
- 0.7：严格，要求大部分缺陷在物体内
- 0.9：非常严格，要求几乎完全包含

### 2.5 融合参数

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|------|------|--------|------|------|
| `--feather-radius` | int | **10-15** | 0-30 | 边缘平滑的羽化半径 |

### 2.6 模型与设备

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model-id` | str | `runwayml/stable-diffusion-inpainting` | SD模型ID |
| `--cache-dir` | str | `models` | 模型缓存目录 |
| `--device` | str | `cuda` | 计算设备 (cuda/cpu) |

---

## 3. 论文实验设置

论文中使用的默认配置：

```bash
python main_contrastive.py \
    --num-inference-steps 100 \
    --r 0.25 \
    --num-optimization-steps 25 \
    --optimization-interval 5 \
    --feather-radius 10 \
    --enable-feature-alignment \
    --ioa-threshold 0.5
```

---

## 4. 命令示例

```bash
# 基本生成
python main_contrastive.py \
    --good-dir images/bottle/good \
    --bad-dir images/bottle/bad \
    --prompt "bottle crack"

# 批量生成 (MVTec-AD)
python batch_generate.py

# 自定义数据集
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "您的产品"

# 高质量模式
python main_contrastive.py \
    --num-inference-steps 150 \
    --r 0.20 \
    --num-optimization-steps 50

# 快速预览模式
python main_contrastive.py \
    --num-inference-steps 50 \
    --r 0.30 \
    --num-optimization-steps 10
```

---

## 5. 故障排除

| 问题 | 调整参数 | 建议 |
|------|----------|------|
| 缺陷出现在物体外部 | `--ioa-threshold` | 增加到 0.7-0.9 |
| 缺陷模糊 | `--num-inference-steps` | 增加到 100-150 |
| 边缘不自然 | `--feather-radius` | 增加到 15-20 |
| 生成速度慢 | `--num-inference-steps` | 减少到 50 |
| 内存不足 | `--device` | 使用 `cpu` |

---

## 6. 输出文件结构

| 目录 | 内容 | 描述 |
|------|------|------|
| `feathered_blend/` | `*.png` | 最终融合图像（主要输出） |
| `non_feathered_blend/` | `*.png` | 非羽化融合图像 |
| `comparison_grid/` | `*.png` | 前后对比图 |
| `combined_defect_masks/` | `*.png` | 生成的缺陷mask |
| `defect_heatmaps/` | `*.png` | 缺陷概率热力图 |
| `original_good/` | `*.png` | 源好图 |
| `reference_bad/` | `*.png` | 源缺陷图 |
| `inference_times.txt` | text | 计时信息 |
