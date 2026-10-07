import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset
from utils.metrics import psnr, ssim_metric, mse

# =========================
# Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StegoModel().to(device)
model.load_state_dict(torch.load("stego_model.pth", map_location=device))
model.eval()

print("Using device:", device)

# =========================
# Dataset
# =========================

dataset = StegoDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

# =========================
# Storage
# =========================

psnr_scores = []
ssim_scores = []
mse_scores = []

# =========================
# Evaluation Loop
# =========================

num_samples = 100

with torch.no_grad():

    for i, (cover, secret) in enumerate(loader):

        if i >= num_samples:
            break

        cover = cover.to(device)
        secret = secret.to(device)

        # Forward pass
        stego, recovered = model(cover, secret)

        # Move to CPU
        cover_cpu = cover[0].cpu()
        stego_cpu = stego[0].cpu()

        secret_cpu = secret[0].cpu()
        recovered_cpu = recovered[0].cpu()

        # Metrics
        current_psnr = psnr(cover_cpu, stego_cpu)
        current_ssim = ssim_metric(cover_cpu, stego_cpu)
        current_mse = mse(secret_cpu, recovered_cpu)

        # Store
        psnr_scores.append(current_psnr)
        ssim_scores.append(current_ssim)
        mse_scores.append(current_mse)

        print(
            f"Sample {i+1}: "
            f"PSNR={current_psnr:.2f}, "
            f"SSIM={current_ssim:.4f}, "
            f"MSE={current_mse:.6f}"
        )

# =========================
# Final Results
# =========================

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)
avg_mse = np.mean(mse_scores)

print("\n========================")
print("FINAL AVERAGE RESULTS")
print("========================")

print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average Secret MSE: {avg_mse:.6f}")

print("\nEvaluation Complete.")