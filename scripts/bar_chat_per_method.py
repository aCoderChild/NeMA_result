import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the method you want to visualize
method = "sft"  # Change to "dpo", "w-reinforce", "ppo", "sft", "npo", etc.

# Input files
icr_csv = "icr_results.csv"
mapo_csv = "mapo_results.csv"
lacomsa_csv = "lacomsa_results.csv"
# baseline_csv = "baseline_results.csv"

# Read and clean CSVs
def load_and_filter(csv_file, model_prefix, method=None):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    if method is not None:
        # Filter for the specific method
        df = df[df["model"].str.contains(f"{model_prefix}_enesru_llama3_8b_{method}")]
    # else:
        # For baseline, match the exact model name in the baseline CSV
        # df = df[df["model"].str.strip() == "Llama-3-8B-SFT-DPO"]
    return df[["source", "length_controlled_winrate"]].set_index("source")

df_icr = load_and_filter(icr_csv, "icr", method)
df_mapo = load_and_filter(mapo_csv, "mapo", method)
df_lacomsa = load_and_filter(lacomsa_csv, "lacomsa", method)
# df_baseline = load_and_filter(baseline_csv, "baseline", method=None)

# Merge for plotting
df_plot = pd.concat([
    df_icr.rename(columns={"length_controlled_winrate": "icr"}),
    df_mapo.rename(columns={"length_controlled_winrate": "mapo"}),
    df_lacomsa.rename(columns={"length_controlled_winrate": "lacomsa"}),
    # df_baseline.rename(columns={"length_controlled_winrate": "baseline"})
], axis=1)

# Plot
ax = df_plot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Length Controlled Winrate")
plt.xlabel("Language")
plt.title(f"Length Controlled Winrate for '{method}'")
plt.legend(title="Model Type")
plt.tight_layout()

# Save
os.makedirs("visualisation", exist_ok=True)
plt.savefig(f"visualisation/lc_winrate_{method}_bar_chart.png")
plt.close()
print(f"Bar chart saved")