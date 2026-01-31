**English** | [中文](CUSTOM_DATASET_CN.md)

# Custom Dataset Tutorial

## 1. Overview

2-GAO is a **training-free** defect generation framework that requires minimal data:
- **1 good image** (defect-free) with its object mask
- **1+ defect images** with their defect masks

This guide explains how to prepare your custom dataset and run batch generation.

## 2. Data Directory Structure

Your dataset should follow this structure:

```
your_dataset/
├── good/
│   ├── image1.png           # Good image (defect-free)
│   ├── image1_mask.png      # Object mask (white = object region)
│   ├── image2.png
│   ├── image2_mask.png
│   └── ...
└── bad/
    ├── defect_type_1/       # Folder name = defect type
    │   ├── img001.png       # Defect image
    │   ├── img001_mask.png  # Defect mask (white = defect region)
    │   ├── img002.png
    │   ├── img002_mask.png
    │   └── ...
    ├── defect_type_2/
    │   ├── img001.png
    │   ├── img001_mask.png
    │   └── ...
    └── ...
```

## 3. Image and Mask Requirements

| Requirement | Specification |
|-------------|---------------|
| **Image format** | PNG or JPG |
| **Mask format** | PNG (grayscale) |
| **Mask naming** | `{image_name}_mask.png` |
| **Mask content** | Binary: 0 (black) = background, 255 (white) = region of interest |
| **Good mask** | Should cover the entire object region |
| **Defect mask** | Should cover only the defect region |

## 4. Mask Preparation Tips

1. **Object masks (good images)**: Use [Segment Anything Model (SAM)](https://segment-anything.com/) for automatic segmentation, or manually draw in image editing software.

2. **Defect masks**: 
   - Draw precisely around the defect boundary
   - Include only the defect area, not the entire object
   - Use feathered edges for better blending (optional)

3. **Quality check**: Ensure masks align perfectly with the corresponding images.

## 5. Running Batch Generation

```bash
# Single category, single defect count
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "your_product" \
    --num-experiments 50 \
    --defect-counts 1

# Multiple defect counts
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "your_product" \
    --num-experiments 50 \
    --defect-counts 1,2,3,4

# Custom output directory
python batch_generate_custom.py \
    --dataset-root your_dataset \
    --object-name "your_product" \
    --output outputs_custom \
    --num-experiments 100
```

## 6. Configuration Options

Edit the configuration section at the top of `batch_generate_custom.py`:

```python
# ==================== USER CONFIGURATION ====================

# Dataset path
DATASET_ROOT = "your_dataset"

# Object name (used in prompt generation)
OBJECT_NAME = "product"

# Defect type mapping (folder name -> prompt token)
DEFECT_TYPE_MAPPING = {
    "scratch": "scratch",
    "crack": "crack",
    "stain": "stain",
    # Add your defect types here
}

# Generation parameters
GENERATION_CONFIG = {
    'num_inference_steps': 100,      # Diffusion steps
    'r': 0.25,                       # Retention coefficient
    'num_optimization_steps': 25,    # Attention optimization steps
    'optimization_interval': 5,      # Optimization interval
    'feather_radius': 10,            # Blending feather radius
}

# Feature alignment (IoA)
FEATURE_CONFIG = {
    'enable_feature_alignment': True,
    'ioa_threshold': 0.5,
}

# Output saving options
SAVE_CONFIG = {
    'save_feathered_blend': True,
    'save_comparison_grid': True,
    'save_defect_heatmaps': True,
    # ... more options
}

# ==================== END CONFIGURATION ====================
```

## 7. Output Structure

Generated files are organized as follows:

```
outputs_custom/
├── feathered_blend/          # Final blended images
├── combined_defect_masks/    # Generated defect masks
├── defect_heatmaps/          # Defect probability heatmaps
├── comparison_grid/          # Before/after comparisons
├── original_good/            # Source good images
├── reference_bad/            # Source defect images
└── ...
```

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| "No valid image pairs found" | Check mask naming convention (`{name}_mask.png`) |
| Defects appear outside object | Increase IoA threshold or check object mask |
| Poor defect quality | Use more descriptive prompts in DEFECT_TYPE_MAPPING |
| Out of memory | Reduce batch size or use smaller images |

---

## License

This project is released under the MIT License.
