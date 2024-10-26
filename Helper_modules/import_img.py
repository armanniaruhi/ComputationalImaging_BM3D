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
    return np.array(Image.open(image_path).convert('L')) / 255.0