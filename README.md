# Benchmarking 18 Deep Learning Segmentation Architectures for Sentinel-2 Snow Cover Mapping in the Hindu Kush

## Executive Summary
This repository contains the computational pipelines, deep learning architectures, and empirical benchmarking experiments for high-resolution cryospheric snow cover mapping across the Swat River Basin in the Hindu Kush range, Northern Pakistan.

We systematically benchmarked eighteen (18) semantic segmentation models grouped across four panels: CNN baselines, advanced/attention-augmented CNNs, pure Vision Transformers (ViTs), and hybrid CNN-Transformer architectures. The study evaluates model performance under complex alpine conditions: steep topographic shadowing, snow-cloud misclassification, and patchy seasonal snowpack boundaries.

---

## Study Basin & Dataset Details

| Parameter | Technical Specification |
| :--- | :--- |
| Geographic Region | Swat River Basin, Hindu Kush Mountains, Northern Pakistan |
| Satellite Platform | Sentinel-2 MultiSpectral Instrument (MSI) Level-2A Surface Reflectance |
| Spectral Bands Evaluated | 6 Bands: Blue (B2), Green (B3), Red (B4), NIR (B8), SWIR-1 (B11), SWIR-2 (B12) |
| Dataset Scale | 9,026 paired multispectral image-mask patches |
| Patch Resolution | 256 x 256 pixels |
| Terrain Conditions | High-relief alpine topography, steep ridge shading, mixed cryospheric surfaces |

*Note: Due to storage size constraints, the complete 9,026-patch multispectral dataset is archived on local storage and is accessible for academic evaluation upon reasonable request.*

---

## Architecture Taxonomy & Directory Organization

The eighteen benchmarked architectures are categorized into four dedicated subdirectories within `notebooks/`, aligning with Panels A, B, C, and D:

### 1. `notebooks/cnn_baseline_models/` (Panel A: Baseline CNN Architectures)
* **01. U-Net** (Backbone: ResNet-34)
* **02. U-Net** (Backbone: EfficientNet-B4)
* **03. SegNet** (Backbone: VGG-16)
* **04. LinkNet** (Backbone: ResNet-50)
* **05. Res-U-Net** (Backbone: ResNet-50) — *Top Overall Performance*
* **06. FPN** (Backbone: VGG-16)

### 2. `notebooks/advanced_cnn_models/` (Panel B: Advanced & Attention CNNs)
* **07. U-Net++** (Backbone: ResNet-34)
* **08. DeepLabV3+** (Backbone: ResNet-50)
* **09. MA-Net** (Backbone: ResNet-50)
* **10. HRNetV2** (Backbone: HRNet-W48)
* **11. PAN** (Backbone: ResNet-50)
* **12. Attention U-Net** (Backbone: VGG-16)
* **13. PSPNet** (Backbone: VGG-16)

### 3. `notebooks/vision_transformers/` (Panel C: Hierarchical Vision Transformers)
* **14. Swin-U-Net** (Backbone: Swin-T Small)
* **15. SegFormer** (Backbone: MiT-B5)
* **16. UNETR** (Backbone: ViT-B/16)

### 4. `notebooks/sota_hybrids/` (Panel D: Hybrid CNN-Transformer Models)
* **17. TransUNet** (Backbone: ViT-R50+)
* **18. CTFuse** (Backbone: CNN + Transformer)

---

## Quantitative Benchmarking Results

All 18 models were trained and evaluated on an independent test split from the 9,026 Sentinel-2 patch dataset under consistent loss functions and hyperparameter configurations:

| # | Model | Backbone | Panel | Architecture Group | DSC (Dice / F1) | mIoU | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | U-Net | ResNet-34 | A | Baseline CNN | 0.9507 | 0.9063 | 0.0717 |
| 02 | U-Net | EfficientNet-B4 | A | Baseline CNN | 0.9326 | 0.8763 | 0.0809 |
| 03 | SegNet | VGG-16 | A | Baseline CNN | 0.9466 | 0.8989 | 0.0783 |
| 04 | LinkNet | ResNet-50 | A | Baseline CNN | 0.9485 | 0.9025 | 0.0748 |
| 05* | Res-U-Net | ResNet-50 | A | Baseline CNN | 0.9536 | 0.9109 | 0.0686 |
| 06 | FPN | VGG-16 | A | Baseline CNN | 0.9288 | 0.8675 | 0.1011 |
| 07 | U-Net++ | ResNet-34 | B | Advanced CNN | 0.9527 | 0.9103 | 0.0686 |
| 08 | DeepLabV3+ | ResNet-50 | B | Advanced CNN | 0.9245 | 0.8602 | 0.1082 |
| 09 | MA-Net | ResNet-50 | B | Advanced CNN | 0.9505 | 0.9060 | 0.0727 |
| 10 | HRNetV2 | HRNet-W48 | B | Advanced CNN | 0.9055 | 0.8289 | 0.1855 |
| 11 | PAN | ResNet-50 | B | Advanced CNN | 0.9231 | 0.8577 | 0.1050 |
| 12 | Attention U-Net | VGG-16 | B | Advanced CNN | 0.9504 | 0.9059 | 0.0721 |
| 13 | PSPNet | VGG-16 | B | Advanced CNN | 0.9066 | 0.8290 | 0.1321 |
| 14 | Swin-U-Net | Swin-T Small | C | Vision Transformer | 0.9314 | 0.8725 | 0.0988 |
| 15 | SegFormer | MiT-B5 | C | Vision Transformer | 0.9432 | 0.8930 | 0.0815 |
| 16 | UNETR | ViT-B/16 | C | Vision Transformer | 0.9468 | 0.8995 | 0.0765 |
| 17 | TransUNet | ViT-R50+ | D | SOTA Hybrid | 0.8633 | 0.7613 | 0.1834 |
| 18 | CTFuse | CNN+Transformer | D | SOTA Hybrid | 0.8731 | 0.7734 | 0.1784 |

*\* Denotes the highest-performing architecture across the entire 18-model benchmark.*

### Key Empirical Findings
* **Top Overall Performers:** Res-U-Net (ResNet-50) achieved the highest overall score with a DSC of 0.9536, mIoU of 0.9109, and the lowest test loss of 0.0686, followed closely by U-Net++ (ResNet-34) (DSC: 0.9527, mIoU: 0.9103).
* **Vision Transformer Benchmarks:** Among pure attention architectures (Panel C), UNETR (ViT-B/16) led the category (DSC: 0.9468, mIoU: 0.8995, Loss: 0.0765), closely followed by SegFormer (MiT-B5) (DSC: 0.9432, mIoU: 0.8930).
* **Attention Mechanisms in Steep Topography:** Integrating channel and spatial attention into convolutional pipelines (such as Attention U-Net with DSC 0.9504 and MA-Net with DSC 0.9505) substantially mitigated false negatives in deep ridge shadows.

---

## Execution & Environment Setup

### 1. Clone the Repository


### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Inspecting the Experiments
Open any notebook inside `notebooks/cnn_baseline_models/`, `notebooks/advanced_cnn_models/`, `notebooks/vision_transformers/`, or `notebooks/sota_hybrids/` to review individual model architectures, loss formulations, and convergence curves.

---

## Author & Academic Affiliation
* **Shakir Ullah Khan**
* Bachelor of Science in Civil Engineering (Gold Medalist)
* University of Engineering and Technology (UET), Peshawar, Pakistan
* Email: 22bnciv1222@uetpeshawar.edu.pk
* LinkedIn: www.linkedin.com/in/shakir-ullah-khan-409811397
