import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset
from utils.metrics import psnr, ssim_metric, mse
from utils.visualization import show_images

# =========================
# Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StegoModel().to(device)
model.load_state_dict(torch.load("stego_model.pth", map_location=device))
model.eval()

print("Using device:", device)

# =========================
# Load Sample Data
# =========================

dataset = StegoDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=True)

cover, secret = next(iter(loader))

cover = cover.to(device)
secret = secret.to(device)

# =========================
# Inference
# =========================

with torch.no_grad():
    stego, recovered = model(cover, secret)

# =========================
# Metrics
# =========================

cover_cpu = cover[0].cpu()
stego_cpu = stego[0].cpu()

secret_cpu = secret[0].cpu()
recovered_cpu = recovered[0].cpu()

print("\n--- Evaluation Results ---")
print("PSNR (Cover vs Stego):", psnr(cover_cpu, stego_cpu))
print("SSIM (Cover vs Stego):", ssim_metric(cover_cpu, stego_cpu))
print("Secret MSE:", mse(secret_cpu, recovered_cpu))

# =========================
# Visualizations
# =========================

print("\nShowing Cover vs Secret...")
show_images(cover_cpu, secret_cpu)

print("Showing Stego vs Recovered...")
show_images(stego_cpu, recovered_cpu)