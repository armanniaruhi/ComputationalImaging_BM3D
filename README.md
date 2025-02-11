# Computational Imaging

## Denoising Project

### Team Members:
- Arman Niaruhi
- Haitham El Euch

### Project Description:
In this project, we will implement several denoising algorithms and evaluate their performance on various types of images. The goal is to enhance image quality by removing noise while preserving important details. We have specifiecally used the BM3D method to denoise the images.

### Denoising Algorithms Implemented:
- **Gaussian Filter**: A linear filter used to remove Gaussian noise by averaging pixel values in a neighborhood.
- **Bilateral Filter**: A non-linear filter that smooths images while preserving edges by considering both spatial distance and intensity difference.
- **Non-Local Means Filter**: A denoising technique that averages similar patches in the image, even if they are far apart spatially.
- **BM3D Filter (Block-Matching and 3D Filtering)**: A state-of-the-art method that groups similar image patches into 3D blocks and applies collaborative filtering to reduce noise.
- **Bergman TV**: A method based on total variation (TV) minimization for denoising.
- **Chambolle TV**: Another method based on total variation minimization, focusing on edge-preserving denoising.
- **Wavelet Transform**: A denoising technique that decomposes the image into different frequency components, selectively reducing noise in the wavelet domain.

### Project Plan:
1. **Step 1**: Implement the denoising methods and test them on a public dataset.
2. **Step 2**: Use the BM3D algorithm in details and try different parameters from the algorithm
3. **Step 3**: Apply the methods to our custom datasets (Leech and ResTarget) and analyze the results.
4. **Step 4**: Compare the denoising results across methods and datasets, and prepare a presentation of findings.

### Denoising using BM3D:
BM3D (Block-Matching and 3D Filtering) is one of the most powerful denoising techniques. It operates in three main steps:

1. **Block Matching**: The image is divided into small patches (blocks). Each block is compared to other blocks in the image to find similar ones, regardless of their location.
2. **3D Collaborative Filtering**: After grouping similar blocks, a 3D filtering operation is applied to the group of blocks, effectively reducing noise while retaining key features and structures in the image.
3. **Aggregation**: The final denoised image is obtained by aggregating the filtered blocks and resolving overlaps.

BM3D works well in preserving fine details and edges while eliminating noise, making it ideal for complex images and datasets.

### Project References:
- [A Tour of Modern Image Filtering by Milanfar, P. IEEE Signal Processing Magazine, 2013](https://users.soe.ucsc.edu/~milanfar/publications/journal/ModernTour.pdf)
- [Public Dataset](https://www.cellpose.org)
