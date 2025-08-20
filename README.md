# 对比学习缺陷生成系统 - 安装和使用指南

## 📋 系统概述

本系统是一个基于Stable Diffusion的对比学习缺陷生成工具，能够使用有缺陷的图像引导无缺陷图像生成相应的缺陷。系统采用注意力机制优化和特征对齐技术，实现高质量的缺陷生成。

## 🔧 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐8GB+ VRAM)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8-3.11
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

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少推理步数
   --num-inference-steps 50
   
   # 使用CPU
   --device cpu
   ```

2. **模型下载失败**
   ```bash
   # 指定本地模型缓存目录
   --cache-dir "./models"
   ```

3. **Unicode编码错误**
   - 确保所有Python文件使用UTF-8编码
   - 在Windows上可能需要设置环境变量：`set PYTHONIOENCODING=utf-8`

4. **依赖包冲突**
   ```bash
   # 重新创建虚拟环境
   rm -rf venv
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或 venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

### 性能优化

1. **启用xformers** (可选，可能在Windows上有问题)
   ```bash
   pip install xformers
   ```

2. **使用半精度浮点**
   - 系统自动在CUDA设备上使用float16

3. **调整批处理大小**
   - 系统自动根据可用内存调整

## 📞 技术支持

如果遇到问题，请检查：
1. Python版本是否在3.8-3.11范围内
2. CUDA版本是否与PyTorch兼容
3. 所有依赖包是否正确安装
4. 输入图像和mask文件是否正确配对

## 📋 必需的子程序文件列表

### 核心程序文件 (必需)

1. **main_contrastive.py** - 主程序入口
   - 功能：命令行参数解析、图像-mask对查找、程序流程控制
   - 依赖：contrastive_defect_generator.py, smart_prompt_generator.py

2. **contrastive_defect_generator.py** - 核心生成器类
   - 功能：对比学习缺陷生成、注意力优化、特征对齐
   - 依赖：attention_heatmap_extractor.py
   - 大小：约1200行代码

3. **smart_prompt_generator.py** - 智能提示生成器
   - 功能：根据产品类别和缺陷类型生成智能prompt
   - 包含：MVTEC数据集的完整词汇表
   - 大小：约400行代码

4. **attention_heatmap_extractor.py** - 注意力热力图提取器
   - 功能：提取UNet注意力机制，生成热力图可视化
   - 包含：Nature期刊标准配色方案
   - 大小：约500行代码

### 配置文件 (必需)

5. **requirements.txt** - Python依赖包列表
   ```
   torch>=2.0.0
   torchvision>=0.15.0
   diffusers>=0.21.0
   transformers>=4.25.0
   accelerate>=0.20.0
   opencv-python>=4.8.0
   Pillow>=9.5.0
   numpy>=1.24.0
   scikit-image>=0.20.0
   matplotlib>=3.7.0
   tqdm>=4.65.0
   safetensors>=0.3.0
   scipy>=1.10.0
   ```

### 可选文件 (推荐)

6. **test_contrastive.py** - 测试程序
   - 功能：验证系统安装和基本功能
   - 用途：快速测试和问题诊断

7. **README_contrastive.md** - 详细使用说明
   - 包含：完整的使用示例和参数说明

### 重要说明

**系统完全自包含**: 上述4个核心Python文件包含了所有必需的功能，无需额外的自定义模块。所有依赖都是标准的Python包，可通过pip安装。

**无需额外配置文件**: 系统使用内置的配置和词汇表，包括：
- MVTEC数据集的完整产品词汇表
- 缺陷类型映射表
- Nature期刊标准配色方案
- 默认的模型参数配置

## 🔄 完整的安装和运行流程

### 步骤1：准备文件
```bash
# 确保您有以下4个核心文件：
main_contrastive.py
contrastive_defect_generator.py
smart_prompt_generator.py
attention_heatmap_extractor.py
requirements.txt
```

### 步骤2：环境设置
```bash
# 创建项目目录
mkdir defect_generation_system
cd defect_generation_system

# 复制所有必需文件到此目录

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 步骤3：准备数据
```bash
# 创建数据目录
mkdir -p images/good images/bad

# 将您的图像文件放入相应目录：
# images/good/: 无缺陷图像和对应的物体mask
# images/bad/: 有缺陷图像和对应的缺陷mask
```

### 步骤4：运行测试
```bash
# 验证安装
python -c "import torch, diffusers, transformers; print('Installation OK')"

# 验证核心模块导入
python -c "from contrastive_defect_generator import ContrastiveDefectGenerator; print('Core modules OK')"

# 运行基本测试（如果有test_contrastive.py）
python test_contrastive.py
```

### 步骤4.1：创建简单验证脚本（可选）
如果您想创建一个简单的验证脚本，可以保存以下内容为 `verify_installation.py`：

```python
#!/usr/bin/env python3
"""
安装验证脚本
"""

def verify_installation():
    print("=== 对比学习缺陷生成系统 - 安装验证 ===")

    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")

    # 检查核心依赖
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch未安装")
        return False

    try:
        import diffusers
        print(f"✓ Diffusers: {diffusers.__version__}")
    except ImportError:
        print("✗ Diffusers未安装")
        return False

    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers未安装")
        return False

    # 检查核心模块
    try:
        from contrastive_defect_generator import ContrastiveDefectGenerator
        print("✓ 核心生成器模块")
    except ImportError as e:
        print(f"✗ 核心模块导入失败: {e}")
        return False

    try:
        from smart_prompt_generator import generate_smart_prompt
        print("✓ 智能提示生成器模块")
    except ImportError as e:
        print(f"✗ 提示生成器模块导入失败: {e}")
        return False

    try:
        from attention_heatmap_extractor import extract_attention_heatmaps
        print("✓ 注意力热力图提取器模块")
    except ImportError as e:
        print(f"✗ 热力图提取器模块导入失败: {e}")
        return False

    print("\n🎉 所有模块验证通过！系统已准备就绪。")
    return True

if __name__ == "__main__":
    verify_installation()
```

然后运行：
```bash
python verify_installation.py
```

### 步骤5：开始生成
```bash
# 基本使用
python main_contrastive.py --prompt "your_product defect_type"

# 示例
python main_contrastive.py --prompt "bottle crack"
```

## 📊 系统架构说明

### 数据流程
```
输入图像 → 图像编码 → 部分前向扩散 → 注意力优化 → 去噪生成 → 后处理 → 输出图像
    ↓           ↓            ↓            ↓         ↓        ↓
  Mask处理 → 特征对齐 → 对比学习损失 → 梯度更新 → 图像解码 → 边缘羽化
```

### 核心技术
1. **对比学习**: 使用有缺陷图像引导无缺陷图像生成
2. **注意力优化**: 通过梯度下降优化注意力权重
3. **特征对齐**: IoA-based特征对齐确保缺陷在物体区域内
4. **部分扩散**: 使用参数r控制前向扩散程度
5. **边缘羽化**: 平滑缺陷边缘，提高真实感

## 🎯 使用建议

### 最佳实践
1. **图像质量**: 使用高质量、清晰的输入图像
2. **Mask精度**: 确保mask准确标注目标区域
3. **参数调优**: 根据具体任务调整关键参数
4. **批量处理**: 对于大量数据，考虑使用批处理脚本

### 参数调优指南
- **高质量生成**: `--num-inference-steps 100 --r 0.25`
- **快速测试**: `--num-inference-steps 50 --r 0.5`
- **精细控制**: `--learning-rate 0.01 --num-optimization-steps 25`
- **边缘平滑**: `--feather-radius 15`

## 💡 实际使用示例

### 示例1：瓶子裂纹生成
```bash
# 准备数据
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
# 准备多个缺陷图像
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

## ❓ 常见问题解答

### Q1: 出现"CUDA out of memory"错误怎么办？
**A1**:
```bash
# 方法1：减少推理步数
python main_contrastive.py --num-inference-steps 50

# 方法2：使用CPU
python main_contrastive.py --device cpu

# 方法3：减少优化步数
python main_contrastive.py --num-optimization-steps 10
```

### Q2: 生成的缺陷质量不好怎么办？
**A2**:
```bash
# 增加推理步数和优化强度
python main_contrastive.py \
    --num-inference-steps 100 \
    --r 0.25 \
    --learning-rate 0.02 \
    --num-optimization-steps 30 \
    --enable-feature-alignment
```

### Q3: 如何确保缺陷在物体区域内？
**A3**:
```bash
# 启用特征对齐功能
python main_contrastive.py \
    --enable-feature-alignment \
    --ioa-threshold 0.5  # 调整IoA阈值
```

### Q4: 生成的图像边缘不自然怎么办？
**A4**:
```bash
# 增加羽化半径
python main_contrastive.py --feather-radius 25

# 或者调整优化参数
python main_contrastive.py \
    --optimization-interval 3 \
    --num-optimization-steps 20
```

### Q5: 如何批量处理多个图像？
**A5**: 目前系统处理单个good图像和多个bad图像。对于批量处理，建议编写简单的脚本：
```python
import os
import subprocess

good_images = ["image1.png", "image2.png", "image3.png"]
for i, good_img in enumerate(good_images):
    cmd = f"""python main_contrastive.py \
        --prompt "bottle crack" \
        --good-dir "batch_good_{i}" \
        --bad-dir "batch_bad" \
        --output-dir "batch_outputs_{i}" """
    subprocess.run(cmd, shell=True)
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

## 📞 技术支持和故障排除

### 系统要求检查
```bash
# 检查Python版本（需要3.8-3.11）
python --version

# 检查CUDA版本
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 常见错误解决
1. **模块导入错误**：确保所有4个核心文件在同一目录
2. **内存不足**：减少推理步数或使用CPU
3. **模型下载失败**：检查网络连接或使用本地模型
4. **Unicode错误**：确保文件编码为UTF-8

### 获取帮助
```bash
# 查看完整参数列表
python main_contrastive.py --help

# 运行验证脚本
python verify_installation.py
```

## 📄 许可证和引用

本项目仅供学习和研究使用。使用时请确保遵守相关的开源协议和学术规范。

如果您在研究中使用了本系统，请考虑引用相关的学术论文。

---

**最后更新**: 2024年
**版本**: 1.0
**兼容性**: Python 3.8-3.11, PyTorch 2.0+, CUDA 11.8+
