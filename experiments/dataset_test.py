import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from utils.dataset_loader import StegoDataset
from utils.visualization import show_images

dataset = StegoDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

cover, secret = next(iter(loader))

print("Cover shape:", cover.shape)
print("Secret shape:", secret.shape)

show_images(cover[0], secret[0])