**English** | [中文](PARAMETERS_CN.md)

# 2-GAO Parameter Reference

## Quick Reference Table

| Component | Parameter | Default | Range | Description |
|-----------|-----------|---------|-------|-------------|
| **Diffusion** | `num_inference_steps` | 100 | 50-150 | Number of denoising steps |
| **Diffusion** | `r` | 0.25 | 0.15-0.35 | Retention coefficient |
| **Optimization** | `num_optimization_steps` | 25 | 10-50 | Attention optimization iterations |
| **Optimization** | `optimization_interval` | 5 | 1-10 | Apply optimization every N steps |
| **Optimization** | `learning_rate` | 0.01 | 0.001-0.1 | Optimization learning rate |
| **IoA** | `enable_feature_alignment` | True | True/False | Enable spatial alignment |
| **IoA** | `ioa_threshold` | 0.5 | 0.3-0.9 | IoA threshold |
| **Blending** | `feather_radius` | 10-15 | 0-30 | Feathering radius (pixels) |
| **Model** | `model_id` | SD-Inpainting | - | Stable Diffusion model |
| **Device** | `device` | cuda | cuda/cpu | Compute device |

---

## 1. batch_generate.py Parameters

### 1.1 Save Configuration (SAVE_CONFIG)

Controls which output files are saved.

```python
SAVE_CONFIG = {
    'save_feathered_blend': True,        # Final blended images with feathering
    'save_non_feathered_blend': True,    # Blended images without feathering
    'save_comparison_grid': True,        # Before/after comparison grid
    'save_contrastive_defect': True,     # Raw contrastive output
    'save_original_good': True,          # Copy of source good image
    'save_reference_bad': True,          # Copy of source defect image
    'save_good_object_masks': True,      # Object masks used
    'save_combined_defect_masks': True,  # Generated defect masks
    'save_bad_defect_masks': True,       # Source defect masks
    'save_defect_heatmaps': True,        # Defect probability heatmaps
    'save_attention_heatmaps': False,    # Attention visualizations (slow)
    'save_other_files': True             # Other auxiliary files
}
```

### 1.2 Feature Configuration (FEATURE_CONFIG)

```python
FEATURE_CONFIG = {
    'enable_feature_alignment': True,    # Enable IoA-based spatial alignment
    'ioa_threshold': 0.5,                # IoA threshold for alignment
    'save_attention_heatmaps': False,    # Save attention heatmaps
    'measure_inference_time': True       # Record inference time
}
```

---

## 2. main_contrastive.py Parameters

### 2.1 Input/Output Paths

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--good-dir` | str | `images/good` | Directory with good images and object masks |
| `--bad-dir` | str | `images/bad` | Directory with defect images and defect masks |
| `--output-dir` | str | `outputs_contrastive2` | Output directory |
| `--prompt` | str | `bottle damaged` | Text prompt for generation |

### 2.2 Diffusion Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--num-inference-steps` | int | **100** | 50-150 | Denoising steps. Higher = better quality, slower |
| `--r` | float | **0.25** | 0.15-0.35 | Retention coefficient. Lower = more variation |

### 2.3 Optimization Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--learning-rate` | float | **0.01** | 0.001-0.1 | Learning rate for attention optimization |
| `--num-optimization-steps` | int | **25** | 10-50 | Optimization iterations per interval |
| `--optimization-interval` | int | **5** | 1-10 | Apply optimization every N diffusion steps |

### 2.4 Feature Alignment (IoA)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--enable-feature-alignment` | flag | **True** | - | Enable IoA-based spatial alignment |
| `--ioa-threshold` | float | **0.5** | 0.3-0.9 | Minimum IoA for valid defect placement |

**IoA Threshold Guidelines:**
- 0.3: Loose constraint, allows partial overlap
- 0.5: Balanced (recommended)
- 0.7: Strict, requires most of defect within object
- 0.9: Very strict, requires almost complete containment

### 2.5 Blending Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--feather-radius` | int | **10-15** | 0-30 | Feathering radius for edge smoothing |

### 2.6 Model & Device

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-id` | str | `runwayml/stable-diffusion-inpainting` | SD model ID |
| `--cache-dir` | str | `models` | Model cache directory |
| `--device` | str | `cuda` | Compute device (cuda/cpu) |

---

## 3. Paper Experiment Settings

Default configuration used in the paper:

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

## 4. Command Examples

```bash
# Basic generation
python main_contrastive.py \
    --good-dir images/bottle/good \
    --bad-dir images/bottle/bad \
    --prompt "bottle crack"

# Batch generation (MVTec-AD)
python batch_generate.py

# Custom dataset
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "your_product"

# High-quality mode
python main_contrastive.py \
    --num-inference-steps 150 \
    --r 0.20 \
    --num-optimization-steps 50

# Fast preview mode
python main_contrastive.py \
    --num-inference-steps 50 \
    --r 0.30 \
    --num-optimization-steps 10
```

---

## 5. Troubleshooting

| Issue | Parameter to Adjust | Recommendation |
|-------|---------------------|----------------|
| Defects outside object | `--ioa-threshold` | Increase to 0.7-0.9 |
| Blurry defects | `--num-inference-steps` | Increase to 100-150 |
| Unnatural edges | `--feather-radius` | Increase to 15-20 |
| Slow generation | `--num-inference-steps` | Decrease to 50 |
| Memory error | `--device` | Use `cpu` |

---

## 6. Output File Structure

| Directory | Content | Description |
|-----------|---------|-------------|
| `feathered_blend/` | `*.png` | Final blended images (main output) |
| `non_feathered_blend/` | `*.png` | Non-feathered blended images |
| `comparison_grid/` | `*.png` | Before/after comparison |
| `combined_defect_masks/` | `*.png` | Generated defect masks |
| `defect_heatmaps/` | `*.png` | Defect probability heatmaps |
| `original_good/` | `*.png` | Source good images |
| `reference_bad/` | `*.png` | Source defect images |
| `inference_times.txt` | text | Timing information |
