import numpy as np
from skimage.util import random_noise


def add_gaussian_noise(image, mean=0, sigma=0.1):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 1)  # Keep values in [0, 1]
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)
    
    # Add salt (white)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 1

    # Add pepper (black)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def add_poisson_noise(image):
    noisy_image = random_noise(image, mode='poisson')
    return noisy_image

def add_speckle_noise(image):
    gauss = np.random.randn(*image.shape)
    noisy_image = image + image * gauss
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image





