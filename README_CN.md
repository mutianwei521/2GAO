[English](README.md) | **中文**

# 2-GAO：基于对比优化的工业缺陷生成系统

## 📋 系统概述

本系统是基于Stable Diffusion的对比学习缺陷生成工具。通过缺陷样本引导，在无缺陷图像上生成缺陷，利用注意力优化和特征对齐技术。

**结果展示:**
- [Google Drive](https://drive.google.com/file/d/1FEvOEMTT9A-Ykt7jTK17nSAblMLfGHZa/view)
- [/outputsResults](https://github.com/mutianwei521/2GAO/tree/main/outputsResults)

**原始图像数据链接 (放入对应文件夹即可）**
- MVTec-AD dataset: [Google Drive](https://drive.google.com/file/d/1BZzy7apJ91cr33-_KG01_Qh-jAeN_zVg/view?usp=drive_link)
- VISA dataset: [Google Drive](https://drive.google.com/file/d/1jaKbzgfHThh7AGqfYQvdphyZfxERbHmq/view?usp=drive_link)
- Concrete crack dataset: [Google Drive](https://drive.google.com/file/d/1ysoPO7OU6GQm2tVTrXvRP0BKX236Aype/view?usp=drive_link)

### 🏗️ 网络架构
![Overall Network Architecture](paper/2gao_03.png)
*图：整体框架包含五个阶段：(1) VAE编码，(2) IoA对齐，(3) 前向扩散，(4) 注意力引导反向优化，以及 (5) 解码。*

---
### 🏗️ 结果展示
![Result Show](paper/mvtec_qual_group1_1defect.png)
*图：工业物体的原始图像、参考图像以及合成缺陷图像（2-GAO方法生成）综合对比：（a）瓶子，（b）电缆，（c）胶囊，（d）地毯。*

## 🔧 系统要求

### 硬件
- **GPU**: NVIDIA GPU（推荐8GB+显存）
- **RAM**: 16GB+
- **存储**: 10GB+可用空间

### 软件
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8–3.11（推荐3.10）
- **CUDA**: 11.8+（GPU加速）

---

## 📦 安装

### 1. 克隆仓库
```bash
git clone https://github.com/mutianwei521/2GAO.git
cd 2GAO
```

### 2. 创建虚拟环境
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. 安装依赖
```bash
# Windows（推荐）:
pip install -r requirements_windows.txt

# Linux/macOS:
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
```

---

## 📁 项目结构

```
2GAO/
├── main_contrastive.py              # 主入口
├── contrastive_defect_generator.py  # 核心生成器
├── smart_prompt_generator.py        # 智能提示生成器
├── attention_heatmap_extractor.py   # 注意力提取器
├── run_ablation.py                  # 消融实验运行器
├── requirements.txt                 # Linux依赖
├── requirements_windows.txt         # Windows依赖
├── batch_generate_mvtec.py          # MVTec数据集批量
├── batch_generate_visa.py           # VISA数据集批量
├── batch_generate_concrete.py       # Concrete数据集批量
├── batch_generate_custom.py         # 自定义数据集批量
├── test/
│   ├── quick_test_mvtec.py          # MVTec快速测试
│   ├── quick_test_visa.py           # VISA快速测试
│   ├── quick_test_concrete.py       # Concrete快速测试
│   ├── evaluate_visa_metrics.py     # VISA评估
│   ├── evaluate_concrete_metrics.py # Concrete评估
│   └── evaluate_mvtec_metrics.py    # MVTec评估
├── visaImages/                      # VISA数据集图像
├── concreteImages/                  # Concrete数据集图像
├── mvtecImages/                     # MVTec数据集图像
├── outputs_visa/                    # VISA输出目录
├── outputs_concrete/                # Concrete输出目录
└── outputs_mvtec/                   # MVTec输出目录
```

---

## 🚀 主程序

### main_contrastive.py
单图像缺陷生成的主入口。

```bash
python main_contrastive.py \
    --prompt "bottle crack" \
    --good-dir "images/good" \
    --bad-dir "images/bad" \
    --output-dir "outputs" \
    --num-inference-steps 100 \
    --r 0.25 \
    --enable-feature-alignment \
    --save-attention-heatmaps
```

### contrastive_defect_generator.py
核心生成器模块（由main_contrastive.py导入）。

### attention_heatmap_extractor.py
提取并可视化注意力热图。

### smart_prompt_generator.py
根据图像内容生成优化的提示词。

---

## 🔬 快速测试程序

### test/quick_test_mvtec.py
MVTec数据集快速测试。
```bash
python test/quick_test_mvtec.py --category bottle --num-defects 2
```
参数:
- `--category`: MVTec分类（bottle, cable, capsule等）
- `--num-defects`: 缺陷数量（1-4）

### test/quick_test_visa.py
VISA数据集快速测试。
```bash
python test/quick_test_visa.py --category candle --num-defects 2
```
参数:
- `--category`: VISA分类（candle, capsules, cashew等）
- `--num-defects`: 缺陷数量（1-4）

### test/quick_test_concrete.py
Concrete裂缝数据集快速测试。
```bash
python test/quick_test_concrete.py --category CFD --num-defects 2
```
参数:
- `--category`: Concrete分类（CFD, CRACK500, DeepCrack等）
- `--num-defects`: 缺陷数量（1-4）

---

## 📦 批量生成程序

### batch_generate_mvtec.py
MVTec数据集批量生成（15个分类）。
```bash
python batch_generate_mvtec.py \
    --mvtec-dir "mvtecImages" \
    --output-dir "outputs_mvtec" \
    --num-samples 50 \
    --num-defects 1 2 3 4
```
MVTec分类: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

### batch_generate_visa.py
VISA数据集批量生成（12个分类）。
```bash
python batch_generate_visa.py \
    --visa-dir "visaImages" \
    --output-dir "outputs_visa" \
    --num-samples 50 \
    --num-defects 1 2 3 4
```
VISA分类: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

### batch_generate_concrete.py
Concrete裂缝数据集批量生成（8个分类）。
```bash
python batch_generate_concrete.py \
    --concrete-dir "concreteImages" \
    --output-dir "outputs_concrete" \
    --num-samples 50 \
    --num-defects 1 2 3 4
```
Concrete分类: CFD, CRACK500, DeepCrack, Eugen, Rissbilder, Volker, crack, cracktree200

### batch_generate_custom.py
自定义数据集批量生成。详见 [CUSTOM_DATASET_CN.md](CUSTOM_DATASET_CN.md)。
```bash
python batch_generate_custom.py \
    --config "config/custom_dataset.yaml" \
    --output-dir "outputs_custom"
```

---

## 📊 评估程序

### test/evaluate_visa_metrics.py
评估VISA数据集生成结果。
```bash
python test/evaluate_visa_metrics.py \
    --output-dir "outputs_visa" \
    --save-csv
```
指标: I-AUC, I-F1, P-AUC, P-F1, PRO, IS, LPIPS  
输出结构: `outputs_visa/{category}/defect_{N}/`

### test/evaluate_concrete_metrics.py
评估Concrete数据集生成结果。
```bash
python test/evaluate_concrete_metrics.py \
    --output-dir "outputs_concrete" \
    --save-csv
```
输出结构: `outputs_concrete/defect_{N}/{category}/`

### test/evaluate_mvtec_metrics.py
评估MVTec数据集生成结果。
```bash
python test/evaluate_mvtec_metrics.py \
    --output-dir "outputs_mvtec" \
    --save-csv
```
输出表格:
- Table A1: 各分类IS/LPIPS
- Table A2: 各场景指标（I-AUC, I-F1, P-AUC, P-F1, PRO）
- Table A3: 详细的分类-缺陷指标

输出结构: `outputs_mvtec/{category}/{1,2,3,4}/`

---

## 🔬 消融实验（论文复现）

### run_ablation.py
复现论文第2.4节和附录中的消融实验。
使用DAAM注意力提取生成注意力图和论文图表。

```bash
# 运行所有消融实验（生成图6-12）
python run_ablation.py --mode all

# 语义模糊性验证（图7）
# 对比通用与具体提示词的注意力分布
python run_ablation.py --mode semantic

# 注意力引导验证（图8）
# 对比有/无Focus Loss和Suppression Loss
python run_ablation.py --mode attention

# 潜空间纠缠验证（图9-11）
# 多缺陷注意力解耦（2/3/4个缺陷）
python run_ablation.py --mode entanglement

# IoA对齐验证（图12）
# 几何有效性：部分/无重叠校正
python run_ablation.py --mode ioa

# 超参数敏感性（图6）
# IoA阈值、扩散步数、优化步数分析
python run_ablation.py --mode hyperparameter

# 仅打印表S2-S6
python run_ablation.py --mode tables
```

**生成的图表：**
- 图6：超参数敏感性分析（4个子图）
- 图7：语义模糊性 - 通用vs具体提示词
- 图8：注意力引导 - Focus/Suppression Loss效果
- 图9-11：潜空间纠缠 - 多缺陷解耦
- 图12：IoA对齐 - 几何有效性

**消融组件（表S2）：**
- **w/o Prompt Guidance**: 语义一致性（I-AUC: 75.63%）
- **w/o Attention Guidance**: 空间精度（PRO: 75.64%）
- **w/o Contrastive Loss**: 多缺陷解耦（PRO: 78.95%）
- **w/o IoA Alignment**: 几何有效性（PRO: 82.34%）
- **Full Model**: 全组件（I-AUC: 100%, PRO: 99.90%）

---

## 📁 输出文件

| 目录 | 内容 | 描述 |
|------|------|------|
| `feathered_blend/` | `*.png` | 最终融合图像（主要输出） |
| `non_feathered_blend/` | `*.png` | 非羽化融合图像 |
| `comparison_grid/` | `*.png` | 前后对比图 |
| `combined_defect_masks/` | `*.png` | 生成的缺陷mask |
| `defect_heatmaps/` | `*.png` | 缺陷概率热力图 |
| `original_good/` | `*.png` | 源好图 |
| `reference_bad/` | `*.png` | 源缺陷图 |

---

## 📚 文档

- [PARAMETERS_CN.md](PARAMETERS_CN.md) - 参数参考手册
- [CUSTOM_DATASET_CN.md](CUSTOM_DATASET_CN.md) - 自定义数据集教程

---

## 💡 示例

### 完整VISA工作流
```bash
# 生成
python batch_generate_visa.py \
    --visa-dir "visaImages" \
    --output-dir "outputs_visa" \
    --num-samples 50

# 评估
python test/evaluate_visa_metrics.py \
    --output-dir "outputs_visa" \
    --save-csv
```

### 完整MVTec工作流
```bash
# 生成
python batch_generate_mvtec.py \
    --mvtec-dir "mvtecImages" \
    --output-dir "outputs_mvtec" \
    --num-samples 50

# 评估
python test/evaluate_mvtec_metrics.py \
    --output-dir "outputs_mvtec" \
    --save-csv
```

---

## 🎯 最佳实践

1. **数据准备**: 确保掩码标注准确
2. **从简单开始**: 先用默认参数，再微调
3. **质量检查**: 使用 `--save-attention-heatmaps`
4. **硬件平衡**: 根据GPU调整步数
5. **评估结果**: 使用评估脚本获取指标

---

## 📜 许可证

MIT License




