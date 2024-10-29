import numpy as np
import cv2 as cv  # Assuming you are using OpenCV
from Helper_modules.helper_func import extract_patches, group_similar_patches, process_patch, aggregate_patches


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

    def denoise_with_gaussian_blur(self, kernel_size=(5, 5), sigmaX=1):
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

    def denoise_with_bm3d(self, patch_size=8, stride=4, threshold=25, similarity_threshold=30):
        """
        Denoise the noisy image using BM3D (Block Matching and 3D Filtering).

        Parameters:
            patch_size (int): The size of each patch to be extracted. Default is 8.
            stride (int): The step size for patch extraction (overlapping control). Default is 4.
            threshold (float): Threshold value for the frequency domain denoising. Default is 25.
            similarity_threshold (float): Similarity threshold for grouping patches.

        Returns:
            np.ndarray: The denoised image (float32).
        """
        # Extract patches from the noisy image
        noisy_patches, positions = extract_patches(self.noisy_image, patch_size, stride)

        # Group similar patches into 3D blocks
        similar_patch_groups = group_similar_patches(noisy_patches, positions, similarity_threshold)

        # Process each 3D block using DCT and thresholding
        denoised_patches = []
        for group in similar_patch_groups:
            denoised_group = np.array([process_patch(patch, threshold) for patch in group])
            denoised_patches.append(np.mean(denoised_group, axis=0))  # Mittelwert über den 3D-Block

        # Aggregate the patches back into the full image, averaging overlapping areas
        denoised_image = aggregate_patches(denoised_patches, positions, self.noisy_image.shape, patch_size)

        return denoised_image