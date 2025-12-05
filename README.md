# 🚀 UFC‑Net: Unrolling Fixed-point Continuous Network for Deep Compressive Sensing

> Welcome aboard — this repo implements **UFC‑Net**, a fresh take on deep compressive sensing.  
> Want robustness, clarity, and some good engineering vibes? Keep reading ☕  

---

## 📑 Table of Contents  
1. [Official Paper & Resources](#official-paper--resources)  
2. [Abstract](#abstract)  
3. [Pretrained Models & Requirements](#pretrained-models--requirements)  
4. [How to Run](#how-to-run)  
5. [Notes & Tips / “Things we learned the hard way”](#notes--tips)  

---

## 📄 Official Paper & Resources  
- **Paper (CVPR 2024):** [UFC‑Net: Unrolling Fixed‑point Continuous Network for Deep Compressive Sensing](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_UFC-Net_Unrolling_Fixed-point_Continuous_Network_for_Deep_Compressive_Sensing_CVPR_2024_paper.pdf) :contentReference[oaicite:7]{index=7}  
- **GitHub Code:** https://github.com/ICSResearch/UFC-Net :contentReference[oaicite:8]{index=8}  
- **Official Poster (CVPR 2024):** Accessible via CVPR’s Virtual Poster Session (#91) :contentReference[oaicite:9]{index=9}  

---

## 📝 Abstract  
> Deep unfolding networks (DUNs), renowned for their interpretability and superior performance, have invigorated the realm of compressive sensing (CS). Nonetheless, existing DUNs frequently suffer from issues related to insufficient feature extraction and feature attrition during the iterative steps. In this paper, we propose Unrolling Fixed‑point Continuous Network (UFC‑Net), a novel deep CS framework motivated by the traditional fixed‑point continuous optimization algorithm. Specifically, we introduce Convolution‑guided Attention Module (CAM) to serve as a critical constituent within the reconstruction phase, encompassing tailored components such as Multi-head Attention Residual Block (MARB), Auxiliary Iterative Reconstruction Block (AIRB), etc. MARB effectively integrates multi‑head attention mechanisms with convolution to reinforce feature extraction, transcending the confinement of localized attributes and facilitating the apprehension of long-range correlations. Meanwhile, AIRB introduces auxiliary variables, significantly bolstering the preservation of features within each iterative stage. Extensive experiments demonstrate that our proposed UFC‑Net achieves remarkable performance both on image CS and CS‑MRI (magnetic resonance imaging) compared to state-of-the-art methods.  

*(If you don’t care about the math, the TL;DR is: better reconstruction + stable training + nice artifacts — just run the code.)*

---

## 📦 Pretrained Models & Requirements  

- **Pretrained models:** available at [this Google Drive folder](*put your link here*)  — ready to plug and play.  
- **Requirements:**  
  - Python == 3.11.5  
  - PyTorch == 1.12.0  

*(Yes, newer PyTorch — so maybe upgrade your environment before running.)*

---

## ⚙️ How to Run  

```bash
# clone repo  
git clone https://github.com/ICSResearch/UFC-Net.git  
cd UFC-Net

# install dependencies (you might want a venv or conda env)  
pip install -r requirements.txt

# run training / testing  
# e.g., for image CS  
python train_image_cs.py --config configs/your_config.yaml

# for CS‑MRI experiment  
python train_mri_cs.py --config configs/mri_config.yaml
