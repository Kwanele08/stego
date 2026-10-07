import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

def mse(img1, img2):
    return torch.mean((img1 - img2) ** 2).item()

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))

def ssim_metric(img1, img2):

    img1 = img1.squeeze().cpu().numpy().transpose(1,2,0)
    img2 = img2.squeeze().cpu().numpy().transpose(1,2,0)

    score, _ = ssim(
        img1,
        img2,
        full=True,
        channel_axis=2,
        data_range=1.0   # 🔥 FIX HERE
    )

    return score