import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Basic Conv Block
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# =========================
# Multi-Scale Block
# =========================

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv3 = ConvBlock(in_channels, out_channels, 3)
        self.conv5 = ConvBlock(in_channels, out_channels, 5)
        self.conv7 = ConvBlock(in_channels, out_channels, 7)

    def forward(self, x):
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        return torch.cat([c3, c5, c7], dim=1)


# =========================
# Encoder Network
# =========================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 3 (cover) + 1 (secret) = 4 channels
        self.block1 = MultiScaleBlock(4, 32)   # → 96 channels
        self.block2 = MultiScaleBlock(96, 32)  # → 96 channels
        self.block3 = MultiScaleBlock(96, 32)  # → 96 channels

        self.final = nn.Conv2d(96, 3, kernel_size=1)

    def forward(self, cover, secret):

        # Upsample secret to match cover size
        secret = F.interpolate(secret, size=cover.shape[2:])

        x = torch.cat([cover, secret], dim=1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        stego = torch.sigmoid(self.final(x))

        return stego


# =========================
# Decoder Network
# =========================

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = MultiScaleBlock(3, 32)
        self.block2 = MultiScaleBlock(96, 32)
        self.block3 = MultiScaleBlock(96, 32)

        self.final = nn.Conv2d(96, 1, kernel_size=1)

    def forward(self, stego):

        x = self.block1(stego)
        x = self.block2(x)
        x = self.block3(x)

        secret = torch.sigmoid(self.final(x))

        # Downsample back to 32x32
        secret = F.interpolate(secret, size=(32,32))

        return secret


# =========================
# Full Model Wrapper
# =========================

class StegoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, cover, secret):

        stego = self.encoder(cover, secret)
        recovered = self.decoder(stego)

        return stego, recovered