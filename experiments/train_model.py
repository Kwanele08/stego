import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.stego_model import StegoModel
from utils.dataset_loader import StegoDataset

# =========================
# Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StegoModel().to(device)

dataset = StegoDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.MSELoss()

beta = 1.0  # weight for secret loss

# =========================
# Training Loop
# =========================

epochs = 5  # keep small for now

for epoch in range(epochs):

    total_loss = 0

    for i, (cover, secret) in enumerate(loader):

        cover = cover.to(device)
        secret = secret.to(device)

        stego, recovered = model(cover, secret)

        cover_loss = criterion(stego, cover)
        secret_loss = criterion(recovered, secret)

        loss = cover_loss + beta * secret_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 ADD THIS
        if i % 50 == 0:
            print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# =========================
# Save Model
# =========================

torch.save(model.state_dict(), "stego_model.pth")

print("Training complete and model saved.")