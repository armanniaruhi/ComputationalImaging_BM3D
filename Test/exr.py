import OpenEXR
import Imath
import numpy as np

# File path to the EXR file
file_path = r"/home/arman/Documents/arman/Uni/Master/Semester 3/ip_repository/ImageProcessing/Dataset/leech/2024_04_25_11_54_01_img_x_15_y_15_r_0_g_1_b_0_cropped.exr"

# Open the EXR file
exr_file = OpenEXR.InputFile(file_path)

# Get the header and dimensions
header = exr_file.header()
dw = header['dataWindow']
width = dw.max.x - dw.min.x + 1
height = dw.max.y - dw.min.y + 1
print(f"Dimensions: {width}x{height}")

# Channels to extract
channels = ['R', 'G', 'B']
use_float16 = False

# Determine pixel type
pixel_type = Imath.PixelType(Imath.PixelType.HALF)

# Allocate memory and load channels
img = np.empty((height, width, len(channels)), dtype=np.float16)

for i, channel in enumerate(channels):
    raw_data = exr_file.channel(channel, pixel_type)
    img[:, :, i] = np.frombuffer(raw_data, dtype=np.float16).reshape(height, width)

# Verify the loaded data
print(f"Image shape: {img.shape}")
