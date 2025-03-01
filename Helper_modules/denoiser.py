import numpy as np
import cv2 as cv  # Assuming you are using OpenCV
from Helper_modules.helper_func import extract_patches, group_similar_patches, process_patch, aggregate_patches
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage import restoration
from scipy.ndimage import median_filter

class ImageDenoiser:
    def __init__(self, original_image, noisy_image):
        """
        Initialize the ImageDenoiser with the original and noisy images.

        Parameters:
            original_image (np.ndarray): The original image (float32).
            noisy_image (np.ndarray): The noisy image (float32).
        """
        self.original_image = original_image
        self.noisy_image = noisy_image

    def denoise_with_bilateral_filter(self, d=15, sigma_color=150, sigma_space=150):
        """
        Denoise the noisy image using a bilateral filter.

        Parameters:
            d (int): Diameter of the pixel neighborhood used during filtering. Default is 15.
            sigma_color (float): Filter sigma in color space. Default is 150.
            sigma_space (float): Filter sigma in coordinate space. Default is 150.

        Returns:
            np.ndarray: The denoised image (float32).
        """
        # Convert the noisy image to 8-bit format
        data_8bit = (self.noisy_image * 255).astype(np.uint8)

        # Apply the bilateral filter
        denoised_image_bilateral = cv.bilateralFilter(data_8bit, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # Convert the filtered data back to float32 for visualization
        denoised_image_bilateral = denoised_image_bilateral.astype(np.float32) / 255.0
        return denoised_image_bilateral

    def denoise_with_gaussian_blur(self, kernel_size=(5, 5), sigmaX=1.0):
        """
        Denoise the noisy image using Gaussian blur.

        Parameters:
            kernel_size (tuple): Size of the Gaussian kernel. Default is (5, 5).
            sigmaX (float): Standard deviation in the X direction. Default is 1.

        Returns:
            np.ndarray: The denoised image (float32).
        """
        # Convert the noisy image to 8-bit format
        data_8bit = (self.noisy_image * 255).astype(np.uint8)

        # Apply Gaussian blur
        denoised_image_gaussian = cv.GaussianBlur(data_8bit, ksize=kernel_size, sigmaX=sigmaX)

        # Convert the filtered data back to float32 for visualization
        denoised_image_gaussian = denoised_image_gaussian.astype(np.float32) / 255.0
        return denoised_image_gaussian

    def denoise_with_nonlocal_means(self, h=50, templateWindowSize=20, searchWindowSize=5):
        """
        Denoise the noisy image using Nonlocal Means Denoising.

        Parameters:
            h (float): Parameter regulating filter strength. Default is 50.
            templateWindowSize (int): Size of the window used to compute the weights. Default is 20.
            searchWindowSize (int): Size of the window to search for pixels. Default is 5.

        Returns:
            np.ndarray: The denoised image (float32).
        """
        # Convert the noisy image to 8-bit format
        data_8bit = (self.noisy_image * 255).astype(np.uint8)

        # Apply Nonlocal Means Denoising
        filtered_data = cv.fastNlMeansDenoising(data_8bit, None, h, templateWindowSize, searchWindowSize)

        # Convert the filtered data back to float32 for visualization
        filtered_data_float = filtered_data.astype(np.float32) / 255.0
        return filtered_data_float
    
    def denoise_with_tv_bregman(self, weight=0.1, max_iter=100):
        denoised_image = restoration.denoise_tv_bregman(self.noisy_image, weight=weight, max_num_iter=max_iter)
        return denoised_image.astype(np.float32)

    def denoise_with_median_filter(self, size=9):
        denoised_image = median_filter(self.noisy_image, size=size)
        return denoised_image.astype(np.float32)

    def denoise_with_tv_chambolle(self, img, weight=0.1, epsilon=0.001, max_num_iter=200):
        """
        Denoise the noisy image using Total Variation (TV) minimization with the Chambolle algorithm.

        Parameters:
            weight (float): Denoising weight. Larger values remove more noise. Default is 0.1.
            epsilon (float): Tolerance for the stopping criterion. Default is 0.001.
            max_num_iter (int): Maximum number of iterations. Default is 200.

        Returns:
            np.ndarray: The denoised image (float32).
        """
        denoised_image = restoration.denoise_tv_chambolle(image= img, weight=weight, 
                                                          eps=epsilon, max_num_iter=max_num_iter)
        return denoised_image

    def denoise_with_dictionary_learning(self, patch_size=8, n_components=256, alpha=1.0, max_patches=10000):
        """
        Denoise the noisy image using dictionary learning with optimizations.

        Parameters:
            patch_size (int): The size of each patch to be extracted. Default is 8.
            n_components (int): Number of dictionary components to learn. Default is 256.
            alpha (float): Regularization parameter controlling the sparsity of the representation. Default is 1.0.
            max_patches (int): Maximum number of patches to use for training the dictionary. Default is 10000.

        Returns:
            np.ndarray: The denoised image (float32).
        """
        # Extract patches from the noisy image
        noisy_patches, positions = extract_patches(self.noisy_image, patch_size, stride=patch_size)

        # Reshape patches for dictionary learning
        patches_reshaped = noisy_patches.reshape(noisy_patches.shape[0], -1)

        # Sample a subset of patches for learning (if more than max_patches available)
        num_patches = min(max_patches, patches_reshaped.shape[0])
        indices = np.random.choice(patches_reshaped.shape[0], size=num_patches, replace=False)
        patches_sampled = patches_reshaped[indices]

        # Perform dictionary learning using MiniBatch
        dictionary_learning = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, batch_size=100, random_state=42)
        X_transformed = dictionary_learning.fit_transform(patches_sampled)

        # Reconstruct the denoised patches
        denoised_patches_reshaped = np.dot(X_transformed, dictionary_learning.components_)

        # Reshape back to original patch shape
        denoised_patches = denoised_patches_reshaped.reshape(num_patches, patch_size, patch_size)

        # Aggregate the denoised patches back into the full image
        denoised_image = aggregate_patches(denoised_patches, positions[:num_patches], self.noisy_image.shape, patch_size)

        return denoised_image