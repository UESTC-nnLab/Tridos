# Tridos
This is the respository of  Tridos.

Triple-domain Feature Learning with Frequency-aware Memory Enhancement for Moving Infrared Small Target Detection

We have submited our paper to "IEEE Trans. on Geoscience and Remote Sensing (TGRS)".

Currently, it is still in the process of refinement. After the formal publication of the paper, the code will be further improved.
![frame](frame.png)
## Abstract
Moving infrared small target detection presents significant challenges due to tiny target sizes and low contrast against backgrounds. Currently-existing methods primarily focus on extracting target features only from the spatial-temporal domain. For further enhancing feature representation, more information domains such as frequency are believed to be potentially valuable.  To extend target feature learning, we propose a new Triple-domain Strategy (Tridos) with the frequency-aware memory enhancement on the spatial-temporal domain. In our scheme, it effectively detaches and enhances frequency features by a local-global frequency-aware module with Fourier transform. Inspired by the human visual system, our memory enhancement aims to capture the target spatial relations between video frames. Furthermore, it encodes temporal dynamics motion features via differential learning and residual enhancing.
Additionally, we further design a residual compensation unit to reconcile possible cross-domain feature mismatches.
To our best knowledge, our Tridos is the first work to explore target feature learning comprehensively in spatial-temporal-frequency domains. The extensive experiments on three datasets (DAUB, ITSDT-15K, and IRDST) validate that our triple-domain learning scheme could be obviously superior to state-of-the-art ones.

## Prerequisite
- python == 3.8
- pytorch == 1.10.0
- einops == 0.7.0
- opencv-python == 4.7.0.72
- scikit-learn == 1.2.2
- scipy == 1.9.1
- Tested on Ubuntu 20.04.6, with CUDA 12.0, and 1x NVIDIA 3090(24 GB)

## PR Curve
![PR](PR.png)

## Contact
IF any questions, please contact with Weiwei Duan via email: [dwwuestc@163.com]().
