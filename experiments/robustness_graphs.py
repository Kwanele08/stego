import matplotlib.pyplot as plt

# =========================
# Experimental Data
# =========================

noise_levels = [0.01, 0.03, 0.05, 0.10, 0.15]

psnr_values = [34.62, 29.46, 25.70, 20.06, 16.68]

ssim_values = [0.9170, 0.6472, 0.4405, 0.2050, 0.1166]

mse_values = [0.000843, 0.005333, 0.012957, 0.040993, 0.074332]

# =========================
# PSNR Graph
# =========================

plt.figure(figsize=(8,5))

plt.plot(noise_levels, psnr_values, marker='o')

plt.title("PSNR vs Gaussian Noise Level")

plt.xlabel("Noise Standard Deviation")

plt.ylabel("PSNR (dB)")

plt.grid(True)

plt.savefig("psnr_vs_noise.png")

plt.show()

# =========================
# SSIM Graph
# =========================

plt.figure(figsize=(8,5))

plt.plot(noise_levels, ssim_values, marker='o')

plt.title("SSIM vs Gaussian Noise Level")

plt.xlabel("Noise Standard Deviation")

plt.ylabel("SSIM")

plt.grid(True)

plt.savefig("ssim_vs_noise.png")

plt.show()

# =========================
# Secret MSE Graph
# =========================

plt.figure(figsize=(8,5))

plt.plot(noise_levels, mse_values, marker='o')

plt.title("Secret MSE vs Gaussian Noise Level")

plt.xlabel("Noise Standard Deviation")

plt.ylabel("Secret MSE")

plt.grid(True)

plt.savefig("mse_vs_noise.png")

plt.show()

print("Graphs generated successfully.")