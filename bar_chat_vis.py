import pandas as pd
import matplotlib.pyplot as plt
import os

# Input files
result_csv = "lacomsa_results.csv"
baseline_csv = "baseline_results.csv"

# Read CSVs and strip column names
df_result = pd.read_csv(result_csv)
df_result.columns = df_result.columns.str.strip()

df_baseline = pd.read_csv(baseline_csv)
df_baseline.columns = df_baseline.columns.str.strip()

# Standardize columns for baseline
if "win_rate" not in df_baseline.columns:
    df_baseline["win_rate"] = None

# Combine and keep only relevant columns
df_result = df_result[["source", "model", "length_controlled_winrate"]]
df_baseline = df_baseline[["source", "model", "length_controlled_winrate"]]

df_all = pd.concat([df_result, df_baseline], ignore_index=True)

# Pivot for plotting
pivot = df_all.pivot(index="source", columns="model", values="length_controlled_winrate")

# Plot
pivot.plot(kind="bar", figsize=(10, 6))
plt.ylabel("Length Controlled Winrate")
plt.xlabel("Language")
plt.title("Length Controlled Winrate by Language and Model")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

os.makedirs("visualisation", exist_ok=True)
plt.savefig(f"visualisation/{result_csv}.png")
plt.close()
print("Bar chart saved")