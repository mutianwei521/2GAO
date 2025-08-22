# 2-GAO： A Contrastive Learning Defect Generation System - Installation and User Guide

## 📋 System Overview

This system is a contrastive learning defect generation tool based on Stable Diffusion. It can generate defects on defect-free images guided by defective samples. The system leverages attention mechanism optimization and feature alignment techniques to achieve high-quality defect generation.

**Some results of this research could be found in:
https://drive.google.com/file/d/1FEvOEMTT9A-Ykt7jTK17nSAblMLfGHZa/view** **or the folder:** [/outputsResults](https://github.com/mutianwei521/2GAO/tree/main/outputsResults)


## 🔧 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU (recommended 8GB+ VRAM)
- **RAM**: 16GB+
- **Storage**: 10GB+ free space

### Software Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8–3.11 (recommended 3.10)
- **CUDA**: 11.8+ (for GPU acceleration)

## 📦 Installation Steps

### 1. Clone or Download the Source Code

Ensure you have the following core files:
```
main_contrastive.py                 # Main entry script
contrastive_defect_generator.py     # Core generator
smart_prompt_generator.py           # Smart prompt generator
attention_heatmap_extractor.py      # Attention heatmap extractor
requirements.txt                    # Dependency list
```

### 2. Create a Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# If you encounter issues installing xformers on Windows, skip it:
pip install torch torchvision diffusers transformers accelerate
pip install opencv-python Pillow numpy scikit-image matplotlib tqdm safetensors scipy
```

### 4. Verify Installation

```bash
# Check Python syntax
python -m py_compile main_contrastive.py
python -m py_compile contrastive_defect_generator.py
python -m py_compile smart_prompt_generator.py
python -m py_compile attention_heatmap_extractor.py

# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
```

## 📁 Data Preparation

### Directory Structure
```
your_project/
├── main_contrastive.py
├── contrastive_defect_generator.py
├── smart_prompt_generator.py
├── attention_heatmap_extractor.py
├── images/
│   ├── good/                    # Defect-free images
│   │   ├── good_image.png
│   │   └── good_image_mask.png  # Object mask
│   └── bad/                     # Defective images
│       ├── defect1.png
│       ├── defect1_mask.png
│       ├── defect2.png
│       └── defect2_mask.png
└── outputs/                     # Output directory (auto-created)
```

### Mask Files
- **Object mask (good_image_mask.png)**: White = object, black = background
- **Defect mask (defect_mask.png)**: White = defect, black = normal area

## 🚀 Usage

### Basic Usage
```bash
# Simplest usage
python main_contrastive.py --prompt "bottle crack"

# Specify input and output directories
python main_contrastive.py     --prompt "bottle crack"     --good-dir "images/good"     --bad-dir "images/bad"     --output-dir "outputs"
```

### Advanced Parameters
```bash
python main_contrastive.py     --prompt "bottle crack hole"     --good-dir "images/good"     --bad-dir "images/bad"     --output-dir "outputs_contrastive"     --num-inference-steps 100     --r 0.25     --learning-rate 0.01     --num-optimization-steps 25     --optimization-interval 5     --feather-radius 15     --enable-feature-alignment     --ioa-threshold 0.5     --save-attention-heatmaps     --measure-inference-time
```

## 📊 Parameter Explanation

### Core Parameters
- `--prompt`: Text prompt (product + defect type)
- `--good-dir`: Directory of defect-free images
- `--bad-dir`: Directory of defective images
- `--output-dir`: Output directory

### Generation Parameters
- `--num-inference-steps`: Denoising steps (default: 100)
- `--r`: Retention ratio for partial diffusion (default: 0.25)
- `--learning-rate`: Learning rate for attention optimization (default: 0.01)
- `--num-optimization-steps`: Steps per optimization (default: 25)
- `--optimization-interval`: Interval between optimizations (default: 5)

### Post-processing Parameters
- `--feather-radius`: Feathering radius (default: 15)
- `--defect-variation`: Defect variation intensity (0.0–1.0, default: 0.0)

### Feature Toggles
- `--enable-feature-alignment`: Enable feature alignment
- `--ioa-threshold`: IoA threshold (default: 0.5)
- `--save-attention-heatmaps`: Save attention heatmaps
- `--measure-inference-time`: Measure inference time

### Device Parameters
- `--device`: Device (cuda/cpu, default: cuda)
- `--model-id`: Stable Diffusion model ID
- `--cache-dir`: Model cache directory (default: models)

## 📁 Output Files

### Main Outputs
- `contrastive_defect_image.png`: Final generated defect image
- `feathered_blend_image.png`: Feathered blend image
- `non_feathered_blend_image.png`: Hard-edge blend image
- `comparison_grid.png`: Comparison grid

### Intermediate Files
- `original_good_image.png`: Input defect-free image
- `good_object_mask.png`: Object mask
- `combined_defect_mask.png`: Combined defect mask
- `reference_bad_image.png`: Reference defective image

### Optional Outputs
- `attention_heatmaps/`: Attention heatmaps (if enabled)
- `inference_times.txt`: Inference time logs (if enabled)

## 💡 Practical Examples

### Example 1: Bottle Crack
```bash
mkdir -p images/good images/bad
# Put bottle_good.png and bottle_good_mask.png in images/good/
# Put bottle_crack.png and bottle_crack_mask.png in images/bad/

python main_contrastive.py     --prompt "bottle crack"     --good-dir "images/good"     --bad-dir "images/bad"     --output-dir "outputs_bottle_crack"     --num-inference-steps 100     --enable-feature-alignment     --save-attention-heatmaps
```

### Example 2: Multiple Defects
```bash
# Prepare multiple defect images
# images/bad/crack.png, crack_mask.png
# images/bad/hole.png, hole_mask.png
# images/bad/scratch.png, scratch_mask.png

python main_contrastive.py     --prompt "nutshell damage"     --good-dir "images/good"     --bad-dir "images/bad"     --output-dir "outputs_multi_defects"     --r 0.25     --feather-radius 20     --enable-feature-alignment     --ioa-threshold 0.7
```

### Example 3: Quick Prototype
```bash
python main_contrastive.py     --prompt "cable bent"     --num-inference-steps 25     --r 0.5     --num-optimization-steps 10     --optimization-interval 3
```

## 🔧 Advanced Settings

### Model Selection
The system defaults to `runwayml/stable-diffusion-inpainting`, but you can try other models:
```bash
python main_contrastive.py     --model-id "stabilityai/stable-diffusion-2-inpainting"     --cache-dir "./models"
```

### Performance Tuning
```bash
# High quality (requires strong GPU)
python main_contrastive.py     --num-inference-steps 150     --r 0.2     --learning-rate 0.005     --num-optimization-steps 50     --optimization-interval 3

# Balanced (recommended)
python main_contrastive.py     --num-inference-steps 100     --r 0.25     --learning-rate 0.01     --num-optimization-steps 25     --optimization-interval 5

# Fast (testing)
python main_contrastive.py     --num-inference-steps 50     --r 0.5     --learning-rate 0.02     --num-optimization-steps 15     --optimization-interval 8
```

## 🎯 Best Practices

1. **Data Preparation**: Ensure accurate mask labeling.
2. **Parameter Selection**: Start with defaults and fine-tune gradually.
3. **Quality Control**: Use `--save-attention-heatmaps` to inspect attention.
4. **Performance Balance**: Adjust inference steps based on hardware.
5. **Result Evaluation**: Use `comparison_grid.png` for visual comparison.
