import matplotlib.pyplot as plt
import numpy as np


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