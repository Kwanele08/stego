import numpy as np
import torch
import cv2

# =========================
# Helper Functions
# =========================

def tensor_to_image(tensor):
    img = tensor.permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img

def image_to_tensor(img):
    img = img.astype(np.float32) / 255.0

    if len(img.shape) == 2:  # grayscale
        img = np.expand_dims(img, axis=-1)

    return torch.tensor(img).permute(2,0,1)


# =========================
# EMBEDDING FUNCTION (2-bit LSB)
# =========================

def embed_lsb(cover, secret):
    """
    cover: [3,128,128]
    secret: [1,32,32]
    """

    cover_img = tensor_to_image(cover)

    # Convert secret to 3 channels
    secret_img = tensor_to_image(secret.repeat(3,1,1))

    # Resize secret to match cover
    secret_img = cv2.resize(secret_img, (128,128))

    # Reduce to 2-bit values (0–3)
    secret_2bit = (secret_img // 64).astype(np.uint8)

    # Flatten
    cover_flat = cover_img.flatten()
    secret_flat = secret_2bit.flatten()

    # Embed 2 bits into each pixel
    for i in range(len(secret_flat)):
        cover_flat[i] = (cover_flat[i] & 252) | secret_flat[i]
        # 252 = 11111100 → clears last 2 bits

    # Reshape back
    stego_img = cover_flat.reshape(128,128,3)

    return image_to_tensor(stego_img)


# =========================
# EXTRACTION FUNCTION (2-bit LSB)
# =========================

def extract_lsb(stego):
    """
    stego: [3,128,128]
    """

    stego_img = tensor_to_image(stego)

    flat = stego_img.flatten()

    # Extract last 2 bits
    bits = np.array([pixel & 3 for pixel in flat], dtype=np.uint8)

    # Reshape
    secret_img = bits.reshape(128,128,3)

    # Scale back to 0–255 range
    secret_img = (secret_img * 85).astype(np.uint8)
    # 85 ≈ 255 / 3

    # Convert to grayscale
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2GRAY)

    # Resize back to original size
    secret_img = cv2.resize(secret_img, (32,32))

    return image_to_tensor(secret_img)