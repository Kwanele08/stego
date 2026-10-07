import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import random
import cv2
import numpy as np

class StegoDataset(Dataset):
    def __init__(self, root='./data', train=True):

        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )

        self.cover_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

        self.secret_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        cover_img,_ = self.dataset[idx]

        secret_idx = random.randint(0,len(self.dataset)-1)
        secret_img,_ = self.dataset[secret_idx]

        cover = self.cover_transform(cover_img)
        secret = self.secret_transform(secret_img)

        return cover, secret