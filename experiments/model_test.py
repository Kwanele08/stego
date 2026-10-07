import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.stego_model import StegoModel

model = StegoModel()

cover = torch.randn(1,3,128,128)
secret = torch.randn(1,1,32,32)

stego, recovered = model(cover, secret)

print("Stego shape:", stego.shape)
print("Recovered shape:", recovered.shape)