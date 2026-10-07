import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset
from utils.metrics import psnr, ssim_metric, mse
from utils.visualization import show_images

torch.manual_seed(42)
np.random.seed(42)

# =========================
# Gaussian Noise Function
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
# Load Data
# =========================

dataset = StegoDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

cover, secret = dataset[0]

# Add batch dimension
cover = cover.unsqueeze(0).to(device)
secret = secret.unsqueeze(0).to(device)

# =========================
# Generate Stego
# =========================

with torch.no_grad():

    stego, recovered_normal = model(cover, secret)

# =========================
# Add Noise
# =========================

noise_std = 0.05

noisy_stego = add_gaussian_noise(
    stego,
    std=noise_std
)

# =========================
# Recover From Noisy Stego
# =========================

with torch.no_grad():

    recovered_noisy = model.decoder(noisy_stego)

# =========================
# Metrics
# =========================

cover_cpu = cover[0].cpu()
stego_cpu = stego[0].cpu()
noisy_stego_cpu = noisy_stego[0].cpu()

secret_cpu = secret[0].cpu()

recovered_normal_cpu = recovered_normal[0].cpu()
recovered_noisy_cpu = recovered_noisy[0].cpu()

# Stego quality
normal_psnr = psnr(cover_cpu, stego_cpu)
normal_ssim = ssim_metric(cover_cpu, stego_cpu)

noise_psnr = psnr(cover_cpu, noisy_stego_cpu)
noise_ssim = ssim_metric(cover_cpu, noisy_stego_cpu)

# Secret recovery quality
normal_mse = mse(secret_cpu, recovered_normal_cpu)
noisy_mse = mse(secret_cpu, recovered_noisy_cpu)

# =========================
# Results
# =========================

print("\n========== NORMAL ==========")
print(f"PSNR: {normal_psnr:.2f}")
print(f"SSIM: {normal_ssim:.4f}")
print(f"Secret MSE: {normal_mse:.6f}")

print("\n========== NOISY ==========")
print(f"PSNR: {noise_psnr:.2f}")
print(f"SSIM: {noise_ssim:.4f}")
print(f"Secret MSE: {noisy_mse:.6f}")

# =========================
# Visualizations
# =========================

print("\nShowing Normal Recovery...")
show_images(stego_cpu, recovered_normal_cpu)

print("Showing Noisy Recovery...")
show_images(noisy_stego_cpu, recovered_noisy_cpu)