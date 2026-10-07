import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset
from utils.metrics import psnr, ssim_metric, mse

torch.manual_seed(42)
np.random.seed(42)

# =========================
# Noise Function
# =========================

def add_gaussian_noise(image, mean=0.0, std=0.05):

    noise = torch.randn_like(image) * std + mean

    noisy_image = image + noise

    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image

# =========================
# Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StegoModel().to(device)

model.load_state_dict(
    torch.load("stego_model.pth", map_location=device)
)

model.eval()

print("Using device:", device)

# =========================
# Dataset
# =========================

dataset = StegoDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# =========================
# Noise Levels
# =========================

noise_levels = [0.01, 0.03, 0.05, 0.10, 0.15]

# =========================
# Experiment
# =========================

# Fixed sample for reproducibility
cover, secret = dataset[0]

# Add batch dimension
cover = cover.unsqueeze(0).to(device)
secret = secret.unsqueeze(0).to(device)

with torch.no_grad():

    stego, recovered = model(cover, secret)

print("\n==============================")
print("MULTI-NOISE ROBUSTNESS TEST")
print("==============================")

for noise_std in noise_levels:

    # Add noise
    noisy_stego = add_gaussian_noise(
        stego,
        std=noise_std
    )

    # Recover secret
    with torch.no_grad():

        recovered_noisy = model.decoder(noisy_stego)

    # CPU copies
    cover_cpu = cover[0].cpu()
    noisy_stego_cpu = noisy_stego[0].cpu()

    secret_cpu = secret[0].cpu()
    recovered_noisy_cpu = recovered_noisy[0].cpu()

    # Metrics
    current_psnr = psnr(cover_cpu, noisy_stego_cpu)
    current_ssim = ssim_metric(cover_cpu, noisy_stego_cpu)
    current_mse = mse(secret_cpu, recovered_noisy_cpu)

    # Results
    print(f"\nNoise STD: {noise_std}")
    print(f"PSNR: {current_psnr:.2f}")
    print(f"SSIM: {current_ssim:.4f}")
    print(f"Secret MSE: {current_mse:.6f}")

print("\nExperiment Complete.")