import matplotlib.pyplot as plt

# =========================
# Noise Levels
# =========================

noise_levels = [0.01, 0.03, 0.05, 0.10, 0.15]

# =========================
# ORIGINAL MODEL RESULTS
# =========================

original_psnr = [34.62, 29.46, 25.70, 20.06, 16.68]

original_ssim = [0.9170, 0.6472, 0.4405, 0.2050, 0.1166]

original_mse = [0.000843, 0.005333, 0.012957, 0.040993, 0.074332]

# =========================
# ROBUST MODEL RESULTS
# =========================

robust_psnr = [28.12, 26.31, 24.05, 19.40, 16.32]

robust_ssim = [0.8502, 0.6046, 0.4062, 0.1812, 0.1025]

robust_mse = [0.000371, 0.000676, 0.001099, 0.005138, 0.012537]

# =========================
# PSNR COMPARISON
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    noise_levels,
    original_psnr,
    marker='o',
    label='Original Model'
)

plt.plot(
    noise_levels,
    robust_psnr,
    marker='o',
    label='Robust Model'
)

plt.title("PSNR Comparison Under Gaussian Noise")

plt.xlabel("Noise Standard Deviation")

plt.ylabel("PSNR (dB)")

plt.legend()

plt.grid(True)

plt.savefig("comparison_psnr.png")

plt.show()

# =========================
# SSIM COMPARISON
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    noise_levels,
    original_ssim,
    marker='o',
    label='Original Model'
)

plt.plot(
    noise_levels,
    robust_ssim,
    marker='o',
    label='Robust Model'
)

plt.title("SSIM Comparison Under Gaussian Noise")

plt.xlabel("Noise Standard Deviation")

plt.ylabel("SSIM")

plt.legend()

plt.grid(True)

plt.savefig("comparison_ssim.png")

plt.show()

# =========================
# SECRET MSE COMPARISON
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    noise_levels,
    original_mse,
    marker='o',
    label='Original Model'
)

plt.plot(
    noise_levels,
    robust_mse,
    marker='o',
    label='Robust Model'
)

plt.title("Secret Recovery MSE Comparison")

plt.xlabel("Noise Standard Deviation")

plt.ylabel("Secret MSE")

plt.legend()

plt.grid(True)

plt.savefig("comparison_mse.png")

plt.show()

print("Comparative graphs generated successfully.")