import os
import numpy as np
import torch
import bm3d
from bm3d import bm3d
import bm3d.profiles as profiles
import OpenEXR
import Imath
import math
import time
from piq import BRISQUELoss, TVLoss, CLIPIQA
import lpips
from contextlib import redirect_stdout


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose GPU or CPU
base_dir = os.getcwd()  # Current working directory

def process_exr(file_path):
    """
    Processes an EXR file by loading its data, adjusting exposure and gamma,
    and optionally displaying the original and adjusted images.

    Parameters:
        file_path (str): Path to the EXR file.

    Returns:
        tuple: (noisy_img, noisy_img_adjusted), where:
               - noisy_img is the original HDR image as a NumPy array.
               - noisy_img_adjusted is the adjusted image.
    """
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Get the header and dimensions
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Channels to extract
    channels = ['R', 'G', 'B']

    # Determine pixel type
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)

    # Allocate memory and load channels
    img = np.empty((height, width, len(channels)), dtype=np.float16)
    for i, channel in enumerate(channels):
        raw_data = exr_file.channel(channel, pixel_type)
        img[:, :, i] = np.frombuffer(raw_data, dtype=np.float16).reshape(height, width)

    # Convert to float32 for processing
    noisy_img = img.astype(np.float32)

    return noisy_img

def calculate_snr(image_3channel):
    """Calculate SNR (Signal-to-Noise Ratio) in decibels using mean and max-min methods."""
    # Initialize lists for SNR per channel
    snr_mean_per_channel = []
    snr_max_min_per_channel = []

    for channel in range(image_3channel.shape[2]):
        channel_data = image_3channel[:, :, channel]
        
        # Signal using max-min
        signal_max_min = channel_data.max() - channel_data.min()
        
        # Signal using mean
        signal_mean = np.mean(channel_data)
        
        # Noise (Standard deviation)
        noise = np.std(channel_data)
        
        # SNR using max-min and mean methods
        snr_max_min = signal_max_min / noise
        snr_mean = signal_mean / noise
        
        # Append to respective lists
        snr_max_min_per_channel.append(snr_max_min)
        snr_mean_per_channel.append(snr_mean)

    # Compute overall SNR for max-min and mean methods
    overall_snr_max_min = np.mean(snr_max_min_per_channel)
    overall_snr_mean = np.mean(snr_mean_per_channel)

    # Convert to decibels
    snr_max_min_db = 20 * math.log(overall_snr_max_min, 10)
    snr_mean_db = 20 * math.log(overall_snr_mean, 10)

    return snr_max_min_db, snr_mean_db

# Convert the NumPy array to a PyTorch tensor
def numpy_to_torch(img_np):
    # Ensure the dimensions are in the format [Channels, Height, Width]
    if len(img_np.shape) == 2:  # Grayscale image
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    else:  # Color image
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).to(device)  # Rearrange dimensions

    # Normalize the tensor values to [0, 1] (optional, depending on your use case)
    img_tensor = torch.clamp(img_tensor, min=0)
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    
    return img_tensor


def load_lpips_model():
    with open(os.devnull, "w") as f, redirect_stdout(f):
        model = lpips.LPIPS(net='vgg').to(device)
    return model



def process_bm3d(
    dataset,
    y,sigma,stage,
    transform_2d_ht="dct",
    transform_2d_wiener = "dct",
    max_3d_size_ht=32,
    max_3d_size_wiener=32,
    bs_wiener=8,
    step_wiener=3,
    bs_ht=8,
    search_window_ht=39,
    search_window_wiener=39,
    step_ht=3,
    tau_match=3000,
    tau_match_wiener=400,
    lambda_thr3d=2.7,
    gamma=2.0,
    beta_wiener=2.0,):

    # Configure BM3D profile
    profile = profiles.BM3DProfile()
    # Transforms used
    profile.transform_2d_ht_name = transform_2d_ht # 'bior1.5'
    profile.transform_2d_wiener_name = transform_2d_wiener

    # Block matching
    profile.gamma = gamma  # 3.0  # Block matching correction factor

    # Hard-thresholding (HT) parameters:
    profile.bs_ht = bs_ht  # 8  # N1 x N1 is the block size used for the hard-thresholding (HT) filtering
    profile.step_ht = (
        step_ht  # 3# sliding step to process every next reference block
    )
    profile.max_3d_size_ht = max_3d_size_ht  # 16  # maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
    profile.search_window_ht = search_window_ht  # 39  # side length of the search neighborhood for full-search block-matching (BM), must be odd
    profile.tau_match = (
        tau_match  # 3000  # threshold for the block-distance (d-distance)
    )

    # None in these parameters results in automatic parameter selection for them
    profile.lambda_thr3d = lambda_thr3d  # None  # 2.7  # threshold parameter for the hard-thresholding in 3D transform domain
    profile.mu2 = lambda_thr3d  # None  # 1.0

    # Wiener filtering parameters:
    profile.bs_wiener = bs_wiener  # 8
    profile.step_wiener = step_wiener  # 3
    profile.max_3d_size_wiener = max_3d_size_wiener  # 32
    profile.search_window_wiener = search_window_wiener  # 39
    profile.tau_match_wiener = tau_match_wiener  # 400
    profile.beta_wiener = beta_wiener  # 2.0
    profile.dec_level = (
        0  # dec. levels of the dyadic wavelet 2D transform for blocks
    )

    file_path = f"{base_dir}/Dataset/{dataset}/2024_04_25_11_54_01_img_x_15_y_{y}_r_0_g_1_b_0_cropped.exr"
    # Load and preprocess noisy image
    noisy_img = process_exr(file_path)  # 800x800
    noisy_img = (noisy_img[:-10, :-10] * 255).astype("uint8")

    # start timer
    start_time = time.time()

    denoised_img = bm3d(
        noisy_img, sigma_psd=sigma, stage_arg=stage, profile=profile
    )

    end_time = time.time()

    _, snr_mean_db = calculate_snr(denoised_img)
    img_tensor = numpy_to_torch(denoised_img)

    # BRISQUELoss
    try:
        loss = BRISQUELoss(reduction="sum").to(device)
        brsique_score = loss(img_tensor)
        brsique_score.backward()
        brsique_score = brsique_score.item()
    except:
        brsique_score = None

    # TV
    loss = TVLoss().to(device)
    tv = loss(img_tensor)
    tv.backward()
    tv = tv.item()

    # CLIP IQA
    try:
        clipiqa = CLIPIQA().to(device)
        clipiqa_score = clipiqa(img_tensor).item()#
    except:
        clipiqa_score = None


    # LPIPS
    try:
        noisy_img_torch = numpy_to_torch(noisy_img)
        loss_fn = load_lpips_model()
        lpips_score = loss_fn(noisy_img_torch, img_tensor).item()
    except:
        lpips_score = None

    denoised_img_float32 = (
        denoised_img.astype("float32") / 255.0
    )  # Normalize to [0, 1] range
    # OpenEXR requires the data to be split into channels (R, G, B)
    R = denoised_img_float32[:, :, 0].tobytes()  # Red channel
    G = denoised_img_float32[:, :, 1].tobytes()  # Green channel
    B = denoised_img_float32[:, :, 2].tobytes()  # Blue channel

    # Define EXR header
    header = OpenEXR.Header(
        denoised_img_float32.shape[1], denoised_img_float32.shape[0]
    )  # Width, Height

    # # Write the data to an EXR file
    # path_final = output_base_folder + f"{type}-{index}_image_res.exr"
    # exr_file = OpenEXR.OutputFile(path_final, header)
    # exr_file.writePixels({"R": R, "G": G, "B": B})
    # exr_file.close()
    computation_time = end_time- start_time
    return computation_time, lpips_score, clipiqa_score, tv, snr_mean_db, noisy_img,denoised_img