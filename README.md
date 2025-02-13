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

### Web App for Results Visualization:
To make the results more accessible, we developed a **web application** where users can:
- View denoised images for different configurations.
- Compare scores from **BRISQUE, TV Loss, CLIP-IQA, and SNR**.
- Analyze the impact of parameter tuning on denoising performance.

🔗 **Check out our results here**: Please run the plot_table.py to see the results in the app.

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


---
# Instruction to use the different application in the repository
---
## Create a Conda Environment from a YAML File

To create a new conda environment using a YAML file (`environment.yml`), follow these steps:

1. Ensure that you have the `environment.yml` file ready. This file should contain the necessary dependencies and configuration for your environment.

2. Open your terminal or command prompt.

3. Run the following command to create the environment:

    ```bash
    conda env create -n new -f environment.yml
    ```

   - `-n new`: This specifies the name of the new environment (`new` in this case). You can replace `new` with your desired environment name.
   - `-f environment.yml`: This points to the `environment.yml` file that contains the environment configuration.

4. Once the command finishes, you can activate the new environment with:

    ```bash
    conda activate new
    ```

    Replace `new` with the name of your environment if you chose a different name.

5. Your conda environment is now set up and ready to use.

---

# BM3D Denoising Experiment Guide

The module in this project is called **Denoising_bm3d.ipynb**.

## How to Use

1. **Select Dataset and Parameters:**
   - `dataset`: Choose between "leech" or "resTarget".
   - `y`: Set a value between 15 and 22.
   - `sigma`: Adjust noise level (integer between 1 and 50 recommended).
   - `stage`: Choose BM3D processing stage (`BM3DStages.ALL_STAGES` or `BM3DStages.HARD_THRESHOLDING`).

2. **Set Transform and Filtering Parameters:**
   - `transform_2d_ht`: Defaults to "bior1.5".
   - `transform_2d_wiener`: Defaults to "bior1.5".
   - `max_3d_size_ht` and `max_3d_size_wiener`: Recommended values are 16 or 32.
   - `bs_wiener` and `bs_ht`: Recommended values are 8 or 16.
   - `step_wiener` and `step_ht`: Choose from 1, 3, 5, or 7.
   - `search_window_ht` and `search_window_wiener`: Set to 39.
   - `tau_match`: Default 1500.
   - `tau_match_wiener`: Default 400.
   - `lambda_thr3d`: Default 2.7.
   - `gamma`: Default 2.0.
   - `beta_wiener`: Default 2.0.

3. **Run BM3D Denoising:**
   ```python
   results = bm3d.process_bm3d(
       dataset, y, sigma, stage,
       transform_2d_ht="bior1.5",
       transform_2d_wiener="bior1.5",
       max_3d_size_ht=32,
       max_3d_size_wiener=32,
       bs_wiener=8,
       bs_ht=8,
       step_wiener=3,
       step_ht=3,
       search_window_ht=39,
       search_window_wiener=39,
       tau_match=1500,
       tau_match_wiener=400,
       lambda_thr3d=2.7,
       gamma=2.0,
       beta_wiener=2.0)
### View and Save Results:
The script extracts the following metrics:
- computation_time
- lpips_score
- clipiqa_score
- tv
- brsique_score
- snr_mean_db
- noisy_img
- denoised_img

The denoised image is saved in `.exr` format.

### Output Metrics

- **Computation Time**: Total time taken for processing.
- **LPIPS Score**: Perceptual similarity score.
- **CLIP-IQA Score**: Measures image quality.
- **Total Variation (TV)**: Image smoothness measurement.
- **BRISQUE Score**: Blind image quality assessment.
- **SNR (Mean dB)**: Signal-to-noise ratio improvement.


---

# Denoising Techniques in `Denoising_synthetic.ipynb`

This notebook explores several denoising techniques applied to synthetic noisy images, with the goal of reducing various types of noise while preserving important details. The following denoising methods were evaluated and compared:

## 1. **Gaussian Noise Denoising**
   - **Noise Type:** Gaussian noise
   - **Method:** A Gaussian filter is used to smooth the image and reduce Gaussian noise added to the image.
   - **Evaluation:** The denoising performance was evaluated by tuning the kernel size (`d`) and the sigma value (`sigmaX`) using a grid search.
   - **Results:** Best parameters were found using PSNR (Peak Signal-to-Noise Ratio) evaluation.
   - **Best Parameters:** Based on the grid search, the best denoising parameters were selected, resulting in the highest PSNR.

## 2. **Salt & Pepper Noise Denoising**
   - **Noise Type:** Salt and pepper noise
   - **Method:** Median filter denoising technique was applied to remove the random black and white pixels resembling salt and pepper grains.
   - **Evaluation:** The performance of the median filter was evaluated by tuning the filter size (`size`) using grid search.
   - **Results:** The best filter size was selected to achieve optimal denoising performance based on PSNR.
   - **Best Parameters:** The best filter size was selected, and the denoised image showed the highest PSNR.

## 3. **Poisson Noise Denoising**
   - **Noise Type:** Poisson noise
   - **Method:** Poisson noise was added, and various denoising techniques were explored to reduce this type of noise.
   - **Evaluation:** A grid search was conducted over different Poisson noise-specific parameters to evaluate performance.

## 4. **Speckle Noise Denoising**
   - **Noise Type:** Speckle noise
   - **Method:** Speckle noise was removed using techniques like bilateral filtering and wavelet-based denoising.
   - **Evaluation:** Speckle noise was reduced by adjusting the filter parameters for the bilateral filter and wavelet methods.

## 5. **Bilateral Filter Denoising**
   - **Noise Type:** Gaussian, Salt & Pepper, Speckle Noise
   - **Method:** The bilateral filter was used for noise reduction while preserving edges. Parameters like the filter diameter (`d`), color standard deviation (`sigma_color`), and spatial standard deviation (`sigma_space`) were tuned via grid search.
   - **Evaluation:** The performance was evaluated based on the PSNR after adjusting the filter parameters.
   - **Best Parameters:** The best parameters were chosen to provide optimal denoising results.

## 6. **Non-Local Means (NLM) Denoising**
   - **Noise Type:** Gaussian, Salt & Pepper, Speckle Noise
   - **Method:** Non-local means denoising was applied to smooth out noise while preserving textures and edges. The key parameter (`h`) was tuned for optimal results.
   - **Evaluation:** The method was evaluated based on the PSNR, and the best parameter (`h`) was selected for each noise type.

## 7. **Total Variation (TV) Bregman Denoising**
   - **Noise Type:** Gaussian, Poisson, Speckle Noise
   - **Method:** TV Bregman denoising was applied, which works by minimizing the total variation of the image while reducing noise. Parameters like `weight` and `max_iter` were tuned to optimize denoising.
   - **Evaluation:** The best combination of `weight` and `max_iter` was selected based on PSNR for optimal denoising.

## 8. **Total Variation (TV) Chambolle Denoising**
   - **Noise Type:** Gaussian, Poisson, Speckle Noise
   - **Method:** TV Chambolle denoising was applied to the denoised image from the median filter step. This method minimizes total variation with additional parameters `weight`, `epsilon`, and `max_iter`.
   - **Evaluation:** The evaluation was based on the PSNR, with the best parameters chosen for denoising.

### Summary
Each denoising method was evaluated by adjusting various parameters using grid search to achieve the best PSNR values. The results demonstrate the effectiveness of each technique in reducing different types of noise while preserving key details in the images.

