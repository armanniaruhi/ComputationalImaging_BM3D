import numpy as np
from itertools import product
import csv
from Helper_modules import helper_func
class DenoisingEvaluator:
    def __init__(self, image, noisy_image, result_dir='Results'):
        self.image = image
        self.noisy_image = noisy_image
        self.result_dir = result_dir
        self.all_psnrs, self.all_denoised, self.method_names = [], [], []

    def evaluate_denoising(self, denoise_func, param_grid, csv_filename, method_name):
        best_psnr = -np.inf
        best_params = None
        psnr_results = []

        # Open CSV file to save results
        with open(f'{self.result_dir}/{csv_filename}', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(list(param_grid.keys()) + ["PSNR"])  # CSV header with parameter names

            # Iterate over all parameter combinations
            for param_values in product(*param_grid.values()):
                params = dict(zip(param_grid.keys(), param_values))
                denoised_image = denoise_func(**params)
                
                # Calculate PSNR as the performance metric
                psnr_value = self.calculate_psnr(denoised_image, self.image)
                psnr_results.append((params, psnr_value))
                
                # Write parameters and PSNR to CSV
                csvwriter.writerow(list(params.values()) + [psnr_value])
                
                # Update the best PSNR and parameters
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    best_params = params
        self.all_denoised.append(denoise_func(**best_params))
        self.all_psnrs.append(int(best_psnr))
        self.method_names.append(method_name)
        return best_psnr, best_params

    def calculate_psnr(self, denoised_image, original_image):
        # Define your PSNR calculation function here
        return helper_func.calculate_psnr(denoised_image, original_image)