import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
from scipy.spatial.distance import cdist


# Function to plot original and noisy images side by side and display PSNR
def plot_images(original, noisy, denoised_image, method_name, noise_type):
    psnr_value = calculate_psnr(denoised_image, original)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Noisy image with method name and noise type
    ax[1].imshow(noisy, cmap='gray')
    ax[1].set_title(f'Noisy Image\n(Method: {method_name}, Noise: {noise_type})')
    ax[1].axis('off')
    
    # Denoised image with PSNR
    ax[2].imshow(denoised_image, cmap='gray')
    ax[2].set_title(f'Denoised Image\n(PSNR: {psnr_value:.2f} dB)')
    ax[2].axis('off')
    plt.savefig(f"Results/plots/methode_{method_name},noise_{noise_type}.png")  # Save the figure if a path is provided
    plt.show()

def plot(img):
    plt.imshow(img)
    plt.show

def calculate_psnr(original_image, noisy_image): 
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_image - noisy_image) ** 2)
    
    if mse == 0:
        return float('inf')  # No noise, PSNR is infinite
    
    # Define MAX_I (maximum possible pixel value for the image)
    max_pixel_value = 1.0
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def plot_noisy_images(original_image, gaussian_noisy_image, salt_und_pepper_noisy_image, poisson_noisy_image, speckle_noisy_image):
    # Create a figure with 1 row and 5 columns for the images
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # List of images and titles
    images = [original_image, gaussian_noisy_image, salt_und_pepper_noisy_image, poisson_noisy_image, speckle_noisy_image]
    titles = ['Original', 'Gaussian Noise', 'Salt & Pepper Noise', 'Poisson Noise', 'Speckle Noise']

    # Loop through the images and plot each one
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')  # Assuming grayscale images
        ax.set_title(titles[i])
        ax.axis('off')  # Hide the axes for a cleaner look

    # Show the plot
    plt.tight_layout()
    plt.show()

def extract_patches(image, patch_size, stride):
    """
    Extract overlapping patches from an image.

    Parameters:
        image (np.ndarray): The input image from which patches are extracted.
        patch_size (int): The size of each patch (e.g., 8 for an 8x8 patch).
        stride (int): The step size to control the overlap of patches.

    Returns:
        np.ndarray: Array of extracted patches.
        list: List of (row, col) positions for each patch in the original image.
    """
    patches = []
    positions = []
    h, w = image.shape

    # Slide over the image with the given stride to extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))

    return np.array(patches), positions

def group_similar_patches(patches, positions, similarity_threshold=30):
    """
    Gruppiert ähnliche Patches basierend auf einem Ähnlichkeitsschwellenwert.

    Parameters:
        patches (np.ndarray): Array der extrahierten Patches.
        positions (list): Liste der Positionen jedes Patches im Bild.
        similarity_threshold (float): Schwellenwert für die Ähnlichkeit.

    Returns:
        list: Liste der 3D-Stacks ähnlicher Patches.
    """
    similar_groups = []
    for i, patch in enumerate(patches):
        # Berechne die euklidische Distanz zwischen dem aktuellen Patch und allen anderen
        distances = cdist([patch.flatten()], patches.reshape(patches.shape[0], -1), metric='euclidean')

        # Wähle ähnliche Patches aus basierend auf dem Schwellenwert
        similar_patches_indices = np.where(distances[0] < similarity_threshold)[0]
        similar_patches = patches[similar_patches_indices]
        similar_groups.append(similar_patches)

    return similar_groups

def process_patch(patch, threshold):
    """
    Apply DCT (Discrete Cosine Transform) and thresholding to a patch.

    Parameters:
        patch (np.ndarray): The input patch to be processed.
        threshold (float): Threshold value for frequency domain denoising.

    Returns:
        np.ndarray: The denoised patch after inverse DCT.
    """
    # Perform DCT to move the patch to the frequency domain
    dct_patch = dct(dct(patch.T, norm='ortho').T, norm='ortho')

    # Apply soft thresholding in the frequency domain
    thresholded_patch = np.sign(dct_patch) * np.maximum(np.abs(dct_patch) - threshold, 0)

    # Apply inverse DCT to bring the patch back to the spatial domain
    return idct(idct(thresholded_patch.T, norm='ortho').T, norm='ortho')

def aggregate_patches(patches, positions, image_shape, patch_size):
    """
    Aggregate denoised patches into the full image by averaging overlaps.

    Parameters:
        patches (np.ndarray): Array of denoised patches.
        positions (list): List of (row, col) positions for each patch in the original image.
        image_shape (tuple): The shape of the original image (height, width).
        patch_size (int): The size of each patch.

    Returns:
        np.ndarray: The fully denoised image with overlapping regions averaged.
    """
    aggregated_image = np.zeros(image_shape)
    weight_matrix = np.zeros(image_shape)

    for patch, (i, j) in zip(patches, positions):
        aggregated_image[i:i + patch_size, j:j + patch_size] += patch
        weight_matrix[i:i + patch_size, j:j + patch_size] += 1

    # Avoid division by zero by only dividing where weight_matrix is non-zero
    aggregated_image = np.divide(
        aggregated_image,
        np.where(weight_matrix != 0, weight_matrix, 1),  # Avoid zero division by using 1 where weight_matrix is zero
        out=np.zeros_like(aggregated_image),  # Fill zeros for locations with zero weight
        where=(weight_matrix != 0)  # Only divide where weight_matrix is non-zero
        )

    return aggregated_image
