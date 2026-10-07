import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Load Results
# ==========================================================

baseline = pd.read_csv("baseline_results.csv")
robust = pd.read_csv("robust_results.csv")

# ==========================================================
# Compute Summary Statistics
# ==========================================================

summary_rows = []

for model_name, data in [("Baseline", baseline), ("Robust", robust)]:

    for noise in sorted(data["Noise_STD"].unique()):

        subset = data[data["Noise_STD"] == noise]

        summary_rows.append({
            "Model": model_name,
            "Noise_STD": noise,

            "Avg_PSNR": subset["PSNR"].mean(),
            "Std_PSNR": subset["PSNR"].std(),

            "Avg_SSIM": subset["SSIM"].mean(),
            "Std_SSIM": subset["SSIM"].std(),

            "Avg_MSE": subset["Secret_MSE"].mean(),
            "Std_MSE": subset["Secret_MSE"].std()
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv("summary_results.csv", index=False)

print("\nSummary Statistics\n")
print(summary)

# ==========================================================
# Split Data
# ==========================================================

baseline_summary = summary[summary["Model"] == "Baseline"]
robust_summary = summary[summary["Model"] == "Robust"]

# ==========================================================
# Improvement Calculation
# ==========================================================

improvement = (
    (baseline_summary["Avg_MSE"].values -
     robust_summary["Avg_MSE"].values)
    /
    baseline_summary["Avg_MSE"].values
) * 100

improvement_df = pd.DataFrame({
    "Noise_STD": baseline_summary["Noise_STD"].values,
    "Improvement (%)": improvement
})

improvement_df.to_csv(
    "percentage_improvement.csv",
    index=False
)

print("\nPercentage Improvement\n")
print(improvement_df)

# ==========================================================
# Graph Function
# ==========================================================

def save_graph(x,
               y1,
               y2,
               ylabel,
               title,
               filename):

    plt.figure(figsize=(8,5))

    plt.plot(
        x,
        y1,
        marker='o',
        label="Baseline"
    )

    plt.plot(
        x,
        y2,
        marker='s',
        label="Robust"
    )

    plt.xlabel("Noise Standard Deviation")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename, dpi=300)

    plt.show()


# ==========================================================
# PSNR
# ==========================================================

save_graph(

    baseline_summary["Noise_STD"],

    baseline_summary["Avg_PSNR"],

    robust_summary["Avg_PSNR"],

    "Average PSNR",

    "Average PSNR vs Noise",

    "average_psnr.png"

)

# ==========================================================
# SSIM
# ==========================================================

save_graph(

    baseline_summary["Noise_STD"],

    baseline_summary["Avg_SSIM"],

    robust_summary["Avg_SSIM"],

    "Average SSIM",

    "Average SSIM vs Noise",

    "average_ssim.png"

)

# ==========================================================
# Secret MSE
# ==========================================================

save_graph(

    baseline_summary["Noise_STD"],

    baseline_summary["Avg_MSE"],

    robust_summary["Avg_MSE"],

    "Average Secret MSE",

    "Average Secret MSE vs Noise",

    "average_secret_mse.png"

)

# ==========================================================
# Percentage Improvement
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    improvement_df["Noise_STD"],

    improvement_df["Improvement (%)"],

    marker='o'

)

plt.xlabel("Noise Standard Deviation")

plt.ylabel("Improvement (%)")

plt.title("Secret Recovery Improvement")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "percentage_improvement.png",
    dpi=300
)

plt.show()

print("\nAnalysis complete.")