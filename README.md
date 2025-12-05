# 🚀 UFC‑Net: Unrolling Fixed-point Continuous Network for Deep Compressive Sensing

> Welcome aboard! This repository implements **UFC‑Net**, a state-of-the-art deep compressive sensing framework.  
> High interpretability, superior performance, and clear engineering practices included ☕  

---

## 📑 Table of Contents
1. [Poster & Paper](#poster--paper)
2. [Supplementary Materials](#supplementary-materials)
3. [Video Demo / Presentation](#video-demo--presentation)
4. [Abstract](#abstract)
5. [Pretrained Models & Requirements](#pretrained-models--requirements)
6. [How to Run](#how-to-run)
7. [Notes & Tips](#notes--tips)
8. [If this code is helpful, please cite](#if-this-code-is-helpful-please-cite)

---

## 🖼️ Poster & Paper

**UFC‑Net Poster (CVPR 2024)**  
![UFC‑Net Poster](./posters/UFC-Net_poster.png)  <!-- 下载后放在该路径 -->

**Official Paper:**  
- [CVPR 2024 Paper PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_UFC-Net_Unrolling_Fixed-point_Continuous_Network_for_Deep_Compressive_Sensing_CVPR_2024_paper.pdf)  
- [CVPR Virtual Poster #91](https://cvpr.thecvf.com/virtual/2024/poster/30588)  

---

## 📄 Supplementary Materials

- [Supplementary PDF](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_UFC-Net_Unrolling_Fixed-point_CVPR_2024_supplemental.pdf)  
*Contains additional ablation studies, implementation details, and visualizations.*

---

## 🎥 Video Demo / Presentation

Check out the official YouTube video:  
[![UFC‑Net Demo](https://img.youtube.com/vi/vaZjIJOCE4g/0.jpg)](https://www.youtube.com/watch?time_continue=2&v=vaZjIJOCE4g&embeds_referring_euri=https%3A%2F%2Fcvpr.thecvf.com%2F)  

> Click the thumbnail to watch the demo — see UFC‑Net’s performance on image CS and CS-MRI tasks.

---

## 📝 Abstract

Deep unfolding networks (DUNs), renowned for their interpretability and superior performance, have invigorated the realm of compressive sensing (CS). Nonetheless, existing DUNs frequently suffer from insufficient feature extraction and feature attrition during iterative steps.  

We propose **Unrolling Fixed‑point Continuous Network (UFC‑Net)**, a novel deep CS framework inspired by fixed-point continuous optimization algorithms. Key components include:

- **Convolution-guided Attention Module (CAM):** Reinforces feature extraction.  
- **Multi-head Attention Residual Block (MARB):** Integrates multi-head attention with convolution to capture long-range correlations.  
- **Auxiliary Iterative Reconstruction Block (AIRB):** Introduces auxiliary variables to preserve features during iterations.  

Extensive experiments show **UFC‑Net** achieves superior performance on both image CS and CS-MRI compared to state-of-the-art methods.  

*TL;DR: Better reconstruction, stable training, and impressive feature preservation.*

---

## 📦 Pretrained Models & Requirements

- **Pretrained Models:** [Google Drive Folder](https://drive.google.com/drive/folders/1vBgjCj9As_Uwe3I5rhclddRDQ-Xp05_h?usp=drive_link)  
- **Requirements:**  
  - Python == 3.11.5  
  - PyTorch == 1.12.0  

*⚠️ Ensure correct PyTorch version to avoid compatibility issues.*

---

## ⚙️ How to Run

```bash
# Clone repo
git clone https://github.com/ICSResearch/UFC-Net.git
cd UFC-Net

# Install dependencies
pip install -r requirements.txt

# Run training / testing
# Image Compressive Sensing
python train_image_cs.py --config configs/your_config.yaml

# CS-MRI
python train_mri_cs.py --config configs/mri_config.yaml
```
## 📚 If this code is helpful, please cite
If you find UFC‑Net useful in your research, please cite the original paper:
```bash
@InProceedings{Wang_2024_CVPR,
  author    = {Xiaoyang Wang and Hongping Gan},
  title     = {UFC‑Net: Unrolling Fixed-point Continuous Network for Deep Compressive Sensing},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  pages     = {25149–25159}
}
```

