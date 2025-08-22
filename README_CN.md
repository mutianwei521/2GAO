# 2-GAO：对比学习缺陷生成系统 - 安装和使用指南

## 📋 系统概述

本系统是一个基于Stable Diffusion的对比学习缺陷生成工具，能够使用有缺陷的图像引导无缺陷图像生成相应的缺陷。系统采用注意力机制优化和特征对齐技术，实现高质量的缺陷生成。

## 🔧 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐8GB+ VRAM)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8-3.11 (推荐3.10)
- **CUDA**: 11.8+ (用于GPU加速)

## 📦 安装步骤

### 1. 克隆或下载源代码

确保您有以下核心文件：
```
main_contrastive.py                 # 主程序入口
contrastive_defect_generator.py     # 核心生成器
smart_prompt_generator.py           # 智能提示生成器
attention_heatmap_extractor.py      # 注意力热力图提取器
requirements.txt                    # 依赖包列表
```

### 2. 创建Python虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. 安装依赖包

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果在Windows上遇到xformers安装问题，可以跳过：
pip install torch torchvision diffusers transformers accelerate
pip install opencv-python Pillow numpy scikit-image matplotlib tqdm safetensors scipy
```

### 4. 验证安装

```bash
# 检查Python语法
python -m py_compile main_contrastive.py
python -m py_compile contrastive_defect_generator.py
python -m py_compile smart_prompt_generator.py
python -m py_compile attention_heatmap_extractor.py

# 测试导入
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
```

## 📁 数据准备

### 目录结构
```
your_project/
├── main_contrastive.py
├── contrastive_defect_generator.py
├── smart_prompt_generator.py
├── attention_heatmap_extractor.py
├── images/
│   ├── good/                    # 无缺陷图像目录
│   │   ├── good_image.png       # 无缺陷图像
│   │   └── good_image_mask.png  # 物体区域mask
│   └── bad/                     # 有缺陷图像目录
│       ├── defect1.png          # 缺陷图像1
│       ├── defect1_mask.png     # 缺陷区域mask1
│       ├── defect2.png          # 缺陷图像2
│       └── defect2_mask.png     # 缺陷区域mask2
└── outputs/                     # 输出目录（自动创建）
```

### Mask文件说明
- **物体mask (good_image_mask.png)**: 白色区域表示物体位置，黑色区域表示背景
- **缺陷mask (defect_mask.png)**: 白色区域表示缺陷位置，黑色区域表示正常区域

## 🚀 使用方法

### 基本使用

```bash
# 最简单的使用方式
python main_contrastive.py --prompt "bottle crack"

# 指定输入和输出目录
python main_contrastive.py \
    --prompt "bottle crack" \
    --good-dir "images/good" \
    --bad-dir "images/bad" \
    --output-dir "outputs"
```

### 高级参数配置

```bash
python main_contrastive.py \
    --prompt "bottle crack hole" \
    --good-dir "images/good" \
    --bad-dir "images/bad" \
    --output-dir "outputs_contrastive" \
    --num-inference-steps 100 \
    --r 0.25 \
    --learning-rate 0.01 \
    --num-optimization-steps 25 \
    --optimization-interval 5 \
    --feather-radius 15 \
    --enable-feature-alignment \
    --ioa-threshold 0.5 \
    --save-attention-heatmaps \
    --measure-inference-time
```

## 📊 参数说明

### 核心参数
- `--prompt`: 文本提示词，包含产品和缺陷类型
- `--good-dir`: 无缺陷图像目录
- `--bad-dir`: 有缺陷图像目录  
- `--output-dir`: 输出目录

### 生成参数
- `--num-inference-steps`: 去噪步数 (默认: 100)
- `--r`: 保留比例，控制部分前向扩散 (默认: 0.25)
- `--learning-rate`: 注意力优化学习率 (默认: 0.01)
- `--num-optimization-steps`: 每次优化的步数 (默认: 25)
- `--optimization-interval`: 优化间隔 (默认: 5)

### 后处理参数
- `--feather-radius`: 边缘羽化半径 (默认: 15)
- `--defect-variation`: 缺陷变化程度 (0.0-1.0, 默认: 0.0)

### 功能开关
- `--enable-feature-alignment`: 启用特征对齐
- `--ioa-threshold`: IoA阈值 (默认: 0.5)
- `--save-attention-heatmaps`: 保存注意力热力图
- `--measure-inference-time`: 测量推理时间

### 设备参数
- `--device`: 计算设备 (cuda/cpu, 默认: cuda)
- `--model-id`: Stable Diffusion模型ID
- `--cache-dir`: 模型缓存目录 (默认: models)

## 📁 输出文件说明

运行完成后，输出目录将包含以下文件：

### 主要输出
- `contrastive_defect_image.png`: 最终生成的缺陷图像
- `feathered_blend_image.png`: 羽化混合图像
- `non_feathered_blend_image.png`: 非羽化混合图像
- `comparison_grid.png`: 对比网格图

### 中间文件
- `original_good_image.png`: 原始无缺陷图像
- `good_object_mask.png`: 物体区域mask
- `combined_defect_mask.png`: 组合缺陷mask
- `reference_bad_image.png`: 参考缺陷图像

### 可选输出
- `attention_heatmaps/`: 注意力热力图文件夹 (如果启用)
- `inference_times.txt`: 推理时间记录 (如果启用)

## 💡 实际使用示例

### 示例1：瓶子裂纹生成
```bash
# 准备数据(MVTEC AD dataset)
mkdir -p images/good images/bad
# 将bottle_good.png和bottle_good_mask.png放入images/good/
# 将bottle_crack.png和bottle_crack_mask.png放入images/bad/

# 运行生成
python main_contrastive.py \
    --prompt "bottle crack" \
    --good-dir "images/good" \
    --bad-dir "images/bad" \
    --output-dir "outputs_bottle_crack" \
    --num-inference-steps 100 \
    --enable-feature-alignment \
    --save-attention-heatmaps
```

### 示例2：多缺陷生成
```bash
# 准备多个缺陷图像(MVTEC AD dataset)
# images/bad/crack.png, crack_mask.png
# images/bad/hole.png, hole_mask.png
# images/bad/scratch.png, scratch_mask.png

python main_contrastive.py \
    --prompt "nutshell damage" \
    --good-dir "images/good" \
    --bad-dir "images/bad" \
    --output-dir "outputs_multi_defects" \
    --r 0.25 \
    --feather-radius 20 \
    --enable-feature-alignment \
    --ioa-threshold 0.7
```

### 示例3：快速原型测试
```bash
# 快速测试配置
python main_contrastive.py \
    --prompt "cable bent" \
    --num-inference-steps 25 \
    --r 0.5 \
    --num-optimization-steps 10 \
    --optimization-interval 3
```

## 🔧 高级配置

### 模型选择
系统默认使用 `runwayml/stable-diffusion-inpainting`，您也可以尝试其他模型：
```bash
# 使用不同的Stable Diffusion模型
python main_contrastive.py \
    --model-id "stabilityai/stable-diffusion-2-inpainting" \
    --cache-dir "./models"
```

### 性能调优
```bash
# 最高质量配置（需要强大GPU）
python main_contrastive.py \
    --num-inference-steps 150 \
    --r 0.2 \
    --learning-rate 0.005 \
    --num-optimization-steps 50 \
    --optimization-interval 3

# 平衡配置（推荐）
python main_contrastive.py \
    --num-inference-steps 100 \
    --r 0.25 \
    --learning-rate 0.01 \
    --num-optimization-steps 25 \
    --optimization-interval 5

# 快速配置（用于测试）
python main_contrastive.py \
    --num-inference-steps 50 \
    --r 0.5 \
    --learning-rate 0.02 \
    --num-optimization-steps 15 \
    --optimization-interval 8
```

## 📊 输出文件详细说明

运行完成后，您将在输出目录中找到以下文件：

### 主要结果文件
1. **contrastive_defect_image.png** - 最终生成的缺陷图像（主要结果）
2. **comparison_grid.png** - 包含原图、生成图、参考图的对比网格
3. **feathered_blend_image.png** - 边缘羽化处理后的混合图像
4. **non_feathered_blend_image.png** - 未羽化的硬边缘混合图像

### 中间过程文件
5. **original_good_image.png** - 输入的无缺陷图像
6. **good_object_mask.png** - 物体区域mask
7. **combined_defect_mask.png** - 组合后的缺陷mask
8. **reference_bad_image.png** - 参考的缺陷图像

### 可选分析文件
9. **attention_heatmaps/** - 注意力热力图文件夹（如果启用）
10. **inference_times.txt** - 推理时间记录（如果启用）

## 🎯 最佳实践总结

1. **数据准备**：确保mask文件准确标注目标区域
2. **参数选择**：从默认参数开始，根据结果逐步调优
3. **质量控制**：使用`--save-attention-heatmaps`查看注意力分布
4. **性能平衡**：根据硬件能力选择合适的推理步数
5. **结果评估**：查看comparison_grid.png进行视觉评估

