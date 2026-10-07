import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset

# =========================
# Gaussian Noise Function
# =========================

def add_gaussian_noise(image, mean=0.0, std=0.03):

    noise = torch.randn_like(image) * std + mean

    noisy_image = image + noise

    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image

# =========================
# Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = StegoModel().to(device)

dataset = StegoDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

beta = 1.0

epochs = 5

# =========================
# Training Loop
# =========================

for epoch in range(epochs):

    total_loss = 0

    for i, (cover, secret) in enumerate(loader):

        cover = cover.to(device)
        secret = secret.to(device)

        # =========================
        # Forward Pass
        # =========================

        stego, _ = model(cover, secret)

        # =========================
        # Add Noise To Stego
        # =========================

        noisy_stego = add_gaussian_noise(
            stego,
            std=0.03
        )

        # =========================
        # Decode From Noisy Stego
        # =========================

        recovered = model.decoder(noisy_stego)

        # =========================
        # Losses
        # =========================

        cover_loss = criterion(stego, cover)

        secret_loss = criterion(recovered, secret)

        loss = cover_loss + beta * secret_loss

        # =========================
        # Backpropagation
        # =========================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # =========================
        # Batch Logging
        # =========================

        if i % 50 == 0:

            print(
                f"Epoch {epoch+1}, "
                f"Batch {i}, "
                f"Loss: {loss.item():.4f}"
            )

    print(f"\nEpoch [{epoch+1}/{epochs}] "
          f"Total Loss: {total_loss:.4f}\n")

# =========================
# Save Robust Model
# =========================

torch.save(
    model.state_dict(),
    "robust_stego_model.pth"
)

print("Robust model training complete.")