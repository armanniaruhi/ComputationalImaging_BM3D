from PIL import Image
import numpy as np

# Dataset : http://www.cellpose.org/
def load_and_normalize_image(image_path):
    """
    Load an image from the specified path and normalize the pixel values to the range [0, 1].
    
    Parameters:
    - image_path: Path to the input image.

    Returns:
    - normalized_image: Image with pixel values normalized between 0 and 1.
    """
    # Load the image using Pillow
    original_image = Image.open(image_path)
    # Convert to grayscale if it's not already (optional)
    original_image = original_image.convert('L')  # 'L' mode stands for grayscale
    # Convert to a numpy array and normalize pixel values
    normalized_image = np.array(original_image) / 255.0
    return normalized_image