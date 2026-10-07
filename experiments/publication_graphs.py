import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Load summary statistics
# -------------------------------------------------------

summary = pd.read_csv("summary_results.csv")

baseline = summary[summary["Model"] == "Baseline"]
robust = summary[summary["Model"] == "Robust"]

# -------------------------------------------------------
# Global Figure Settings
# -------------------------------------------------------

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["legend.fontsize"] = 11

# -------------------------------------------------------
# Helper Function
# -------------------------------------------------------

def save_error_graph(
    x,
    y1,
    err1,
    y2,
    err2,
    ylabel,
    title,
    filename,
    log_scale=False
):

    plt.figure()

    plt.errorbar(
        x,
        y1,
        yerr=err1,
        marker='o',
        linewidth=2,
        capsize=5,
        label="Baseline"
    )

    plt.errorbar(
        x,
        y2,
        yerr=err2,
        marker='s',
        linewidth=2,
        capsize=5,
        label="Robust"
    )

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Noise Standard Deviation")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename, dpi=600)

    plt.show()


# -------------------------------------------------------
# PSNR
# -------------------------------------------------------

save_error_graph(

    baseline["Noise_STD"],

    baseline["Avg_PSNR"],
    baseline["Std_PSNR"],

    robust["Avg_PSNR"],
    robust["Std_PSNR"],

    "Average PSNR (dB)",

    "PSNR Under Gaussian Noise",

    "publication_psnr.png"

)

# -------------------------------------------------------
# SSIM
# -------------------------------------------------------

save_error_graph(

    baseline["Noise_STD"],

    baseline["Avg_SSIM"],
    baseline["Std_SSIM"],

    robust["Avg_SSIM"],
    robust["Std_SSIM"],

    "Average SSIM",

    "SSIM Under Gaussian Noise",

    "publication_ssim.png"

)

# -------------------------------------------------------
# Secret MSE (Log Scale)
# -------------------------------------------------------

save_error_graph(

    baseline["Noise_STD"],

    baseline["Avg_MSE"],
    baseline["Std_MSE"],

    robust["Avg_MSE"],
    robust["Std_MSE"],

    "Secret Reconstruction MSE",

    "Secret Recovery Error",

    "publication_secret_mse.png",

    log_scale=True

)

print("\nPublication-quality figures generated successfully.")