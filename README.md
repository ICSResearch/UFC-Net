# 🚀 UFC‑Net: Unrolling Fixed-point Continuous Network for Deep Compressive Sensing

> Welcome aboard — this repo implements **UFC‑Net**, a novel deep compressive sensing framework.  
> Interpretability, high performance, and some engineering vibes included ☕  

---

## 📑 Table of Contents
1. [Poster & Paper](#poster--paper)
2. [Supplementary Materials](#supplementary-materials)
3. [Abstract](#abstract)
4. [Pretrained Models & Requirements](#pretrained-models--requirements)
5. [How to Run](#how-to-run)
6. [Notes & Tips](#notes--tips)

---

## 🖼️ Poster & Paper

**UFC‑Net Poster (CVPR 2024)**  
![UFC‑Net Poster](./posters/UFC-Net_poster.png) <!-- 下载后放在该路径 -->

**Official Paper:**  
- PDF: [CVPR 2024 Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_UFC-Net_Unrolling_Fixed-point_Continuous_Network_for_Deep_Compressive_Sensing_CVPR_2024_paper.pdf)  
- Poster Page: [CVPR Virtual Poster #91](https://cvpr.thecvf.com/virtual/2024/poster/30588)  

---

## 📄 Supplementary Materials
- PDF: [Supplementary File](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_UFC-Net_Unrolling_Fixed-point_CVPR_2024_supplemental.pdf)  
*(Contains additional ablation studies, implementation details, and more visualizations.)*

---

## 📝 Abstract
> Deep unfolding networks (DUNs), renowned for their interpretability and superior performance, have invigorated the realm of compressive sensing (CS). Nonetheless, existing DUNs frequently suffer from issues related to insufficient feature extraction and feature attrition during the iterative steps. In this paper, we propose Unrolling Fixed‑point Continuous Network (UFC‑Net), a novel deep CS framework motivated by the traditional fixed-point continuous optimization algorithm. Specifically, we introduce Convolution‑guided Attention Module (CAM) to serve as a critical constituent within the reconstruction phase, encompassing tailored components such as Multi-head Attention Residual Block (MARB), Auxiliary Iterative Reconstruction Block (AIRB), etc. MARB effectively integrates multi‑head attention mechanisms with convolution to reinforce feature extraction, transcending the confinement of localized attributes and facilitating the apprehension of long-range correlations. Meanwhile, AIRB introduces auxiliary variables, significantly bolstering the preservation of features within each iterative stage. Extensive experiments demonstrate that our proposed UFC‑Net achieves remarkable performance both on image CS and CS‑MRI in contrast to state-of-the-art methods.

*(TL;DR: better reconstruction + stable training + impressive feature preservation.)*

---

## 📦 Pretrained Models & Requirements

- **Pretrained Models:** [Google Drive Folder](https://drive.google.com/drive/folders/1vBgjCj9As_Uwe3I5rhclddRDQ-Xp05_h?usp=drive_link)  
- **Requirements:**  
  - Python == 3.11.5  
  - PyTorch == 1.12.0  

*(Make sure to use the correct PyTorch version to avoid compatibility issues.)*

---

## ⚙️ How to Run

```bash
# Clone repo
git clone https://github.com/ICSResearch/UFC-Net.git
cd UFC-Net

# Install dependencies
pip install -r requirements.txt

# Run training / testing
# Example: Image Compressive Sensing
python train_image_cs.py --config configs/your_config.yaml

# Example: CS-MRI
python train_mri_cs.py --config configs/mri_config.yaml
