import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import torch
import numpy as np

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset
from utils.metrics import psnr, ssim_metric, mse


# ==========================================================
# Reproducibility
# ==========================================================

torch.manual_seed(42)
np.random.seed(42)


# ==========================================================
# Settings
# ==========================================================

NUM_IMAGES = 100
NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10, 0.15]


# ==========================================================
# Gaussian Noise
# ==========================================================

def add_gaussian_noise(image, mean=0.0, std=0.03):
    noise = torch.randn_like(image) * std + mean
    noisy = image + noise
    return torch.clamp(noisy, 0, 1)


# ==========================================================
# Evaluation Function
# ==========================================================

def evaluate_model(model_path, output_csv):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===================================")
    print(f"Loading model: {model_path}")
    print("===================================")

    model = StegoModel().to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    dataset = StegoDataset()

    total_results = {}

    for noise in NOISE_LEVELS:
        total_results[noise] = {
            "psnr": [],
            "ssim": [],
            "mse": []
        }

    with open(output_csv, mode="w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Image_Index",
            "Noise_STD",
            "PSNR",
            "SSIM",
            "Secret_MSE"
        ])

        # ======================================================
        # Evaluate Fixed Images
        # ======================================================

        for image_index in range(NUM_IMAGES):

            cover, secret = dataset[image_index]

            cover = cover.unsqueeze(0).to(device)
            secret = secret.unsqueeze(0).to(device)

            with torch.no_grad():
                stego, _ = model(cover, secret)

            # --------------------------------------------
            # Test every noise level
            # --------------------------------------------

            for noise_std in NOISE_LEVELS:

                noisy_stego = add_gaussian_noise(
                    stego,
                    std=noise_std
                )

                with torch.no_grad():
                    recovered = model.decoder(noisy_stego)

                cover_cpu = cover[0].cpu()
                noisy_cpu = noisy_stego[0].cpu()

                secret_cpu = secret[0].cpu()
                recovered_cpu = recovered[0].cpu()

                current_psnr = psnr(
                    cover_cpu,
                    noisy_cpu
                )

                current_ssim = ssim_metric(
                    cover_cpu,
                    noisy_cpu
                )

                current_mse = mse(
                    secret_cpu,
                    recovered_cpu
                )

                writer.writerow([
                    image_index,
                    noise_std,
                    current_psnr,
                    current_ssim,
                    current_mse
                ])

                total_results[noise_std]["psnr"].append(current_psnr)
                total_results[noise_std]["ssim"].append(current_ssim)
                total_results[noise_std]["mse"].append(current_mse)

            if (image_index + 1) % 10 == 0:
                print(f"Processed {image_index + 1}/{NUM_IMAGES} images...")

    # ======================================================
    # Print Summary
    # ======================================================

    print("\n===================================")
    print(f"AVERAGE RESULTS ({model_path})")
    print("===================================")

    for noise in NOISE_LEVELS:

        avg_psnr = np.mean(total_results[noise]["psnr"])
        avg_ssim = np.mean(total_results[noise]["ssim"])
        avg_mse = np.mean(total_results[noise]["mse"])

        print(f"\nNoise STD: {noise:.2f}")
        print(f"Average PSNR      : {avg_psnr:.2f}")
        print(f"Average SSIM      : {avg_ssim:.4f}")
        print(f"Average Secret MSE: {avg_mse:.6f}")

    print(f"\nCSV saved as: {output_csv}")


# ==========================================================
# Run Both Models
# ==========================================================

if __name__ == "__main__":

    evaluate_model(
        "stego_model.pth",
        "baseline_results.csv"
    )

    evaluate_model(
        "robust_stego_model.pth",
        "robust_results.csv"
    )