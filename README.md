# UFC-Net: Unrolling Fixed-point Continuous Network for Deep Compressive Sensing
## Abstract
Deep unfolding networks (DUNs), renowned for their interpretability and superior performance, have invigorated the realm of compressive sensing (CS). Nonetheless, existing DUNs frequently suffer from issues related to insufficient feature extraction and feature attrition during the iterative steps. In this paper, we propose Unrolling Fixed-point Continuous Network (**UFC-Net**), a novel deep CS framework motivated by the traditional fixed-point continuous optimization algorithm. Specifically, we introduce Convolution-guided Attention Module (CAM) to serve as a critical constituent within the reconstruction phase, encompassing tailored components such as Multi-head Attention Residual Block (MARB), Auxiliary Iterative Reconstruction Block (AIRB), etc. MARB effectively integrates multi-head attention mechanisms with convolution to reinforce feature extraction, transcending the confinement of localized attributes and facilitating the apprehension of long-range correlations. Meanwhile, AIRB introduces auxiliary variables, significantly bolstering the preservation of features within each iterative stage. Extensive experiments demonstrate that our proposed UFC-Net achieves remarkable  performance both on image CS and CS-magnetic resonance imaging (CS-MRI) in contrast to state-of-the-art methods.

## Pretrained Models
https://drive.google.com/drive/folders/1vBgjCj9As_Uwe3I5rhclddRDQ-Xp05_h?usp=drive_link

## Requirements
> Python == 3.11.5

> PyTorch == 1.12.0
