This code package implements the prototypical part network as applied to spatial proteomics (ProteinPNet)
from the paper "ProteinPNet" (to appear at NeurIPS 2025 workshop for Imageomics), by Louis McConnell, Jieran Sun, Theo Maffei, Raphael Gottardo, and Marianna Rapsomaniki.

This code package was licensed under MIT License (see LICENSE for more information regarding the use
and the distribution of this code package).

Prerequisites: PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)

Instructions for preparing the data:
1. Download the dataset from Cords et. al 2024 (https://github.com/AI4SCR/ai4bmr-datasets)
2. Apply PCA on each of the image samples. 
3. Run main.py.
