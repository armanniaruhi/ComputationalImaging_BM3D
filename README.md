# Computational Imaging

## Denoising Project

### Team Members:
- Arman Niaruhi
- Haitham El Euch

### Project Description:
This project explores computational image denoising using **BM3D (Block Matching and 3D Filtering)** alongside other denoising techniques. The main goal is to reduce noise while preserving fine details, particularly in microscopic images. The project evaluates different parameter configurations and quality metrics to assess denoising performance.

### Denoising Algorithms Implemented:
- **Gaussian Filter**: Smooths images but can blur fine details.
- **Bilateral Filter**: Preserves edges while reducing noise.
- **Non-Local Means Filter**: Averages similar image patches for noise removal.
- **BM3D**: A two-step denoising algorithm using collaborative 3D filtering.
- **Bregman TV & Chambolle TV**: Total Variation-based methods for edge-preserving smoothing.
- **Wavelet Transform**: Reduces noise in the frequency domain.

### Experimental Setup:
- **Datasets**: 
  - *Leech Dataset* 
  - *ResTarget Dataset*
- **Evaluation Metrics**:
  - **BRISQUE**: Perceptual quality assessment using statistical features.
  - **TV Loss**: Measures image smoothness by penalizing intensity variations.
  - **CLIP-IQA**: Uses CLIP embeddings to evaluate perceptual quality.
  - **SNR**: Signal-to-noise ratio to measure noise suppression effectiveness.

### Key Observations:
- **Higher Sigma Values**: Reduce noise but blur fine details.
- **BM3D Strength**: Preserves structures better than traditional methods.
- **Challenges with Microscopic Images**: Existing metrics may not fully align with human perception.
- **Parameter Optimization**: Patch size, block count, transformation type, and threshold values significantly affect results.

### Results & Conclusion:
- BM3D remains a strong traditional denoising method, especially for Gaussian noise.
- No single metric perfectly evaluates microscopic image denoising.
- Further work needed in developing a suitable metric and exploring deep learning-based denoising.

### Future Work:
- Identify or develop better evaluation metrics for microscopic images.
- Investigate deep learning models for handling complex noise patterns.

### References:
- [A Tour of Modern Image Filtering by Milanfar, P. IEEE Signal Processing Magazine, 2013](https://users.soe.ucsc.edu/~milanfar/publications/journal/ModernTour.pdf)
- [BM3D: Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering](https://webpages.tuni.fi/foi/GCF-BM3D/BM3D_TIP_2007.pdf)
- [Public Dataset](https://www.cellpose.org)
