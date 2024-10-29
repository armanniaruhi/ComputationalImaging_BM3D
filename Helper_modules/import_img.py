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
    original_image = Image.open(image_path).convert('L')
    # Define the new size (width, height)
    
    # # Get the original dimensions
    # original_width, original_height = original_image.size

    # # Calculate the new height to maintain the aspect ratio
    # aspect_ratio = original_height / original_width
    # new_height = int(desired_width * aspect_ratio)

    # # Resize the image
    # new_size = (desired_width, new_height)
    # resized_image = original_image.resize(new_size, Image.LANCZOS)

    # Convert to a numpy array and normalize pixel values
    normalized_image = np.array(original_image) / 255.0
    return normalized_image