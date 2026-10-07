import matplotlib.pyplot as plt
import torch

def show_images(cover, secret):

    cover = cover.permute(1,2,0).cpu().numpy()
    secret = secret.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1,2, figsize=(8,4))

    ax[0].imshow(cover)
    ax[0].set_title("Cover Image")

    ax[1].imshow(secret, cmap='gray')
    ax[1].set_title("Secret Image")

    for a in ax:
        a.axis("off")

    plt.show()