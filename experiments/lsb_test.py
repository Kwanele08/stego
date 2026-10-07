import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from utils.dataset_loader import StegoDataset
from utils.visualization import show_images
from utils.metrics import psnr, ssim_metric

from baselines.lsb_steganography import embed_lsb, extract_lsb

dataset = StegoDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=True)

cover, secret = next(iter(loader))

# Embed
stego = embed_lsb(cover[0], secret[0])

# Extract
recovered = extract_lsb(stego)

# Metrics
print("PSNR:", psnr(cover[0], stego))
print("SSIM:", ssim_metric(cover[0], stego))

# Visualize
show_images(cover[0], secret[0])
show_images(stego, recovered)