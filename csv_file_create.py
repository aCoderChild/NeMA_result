import os
import csv

# Path to the mapo folder and output CSV
dir = "lacomsa"
output_csv = "lacomsa_results.csv"

# The columns to match runs/15k_train.csv
columns = [
    "source", "model", "win_rate", "standard_error", "mode", "avg_length",
    "n_wins", "n_wins_base", "n_draws", "n_total", "discrete_win_rate",
    "length_controlled_winrate", "lc_standard_error"
]

# Helper function to extract stats from a result folder
def extract_stats_from_folder(folder_path, expected_model=None):
    leaderboard_path = os.path.join(folder_path, "leaderboard.csv")
    if not os.path.exists(leaderboard_path):
        return None

    with open(leaderboard_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Strip whitespace from keys and values
            row = {k.strip(): v.strip() for k, v in row.items()}
            # If there's a model column, match it
            if expected_model:
                model_in_row = row.get("model", "") or row.get("", "")  # Sometimes the first column is unnamed
                if expected_model in model_in_row:
                    return {
                        "win_rate": row.get("win_rate", ""),
                        "standard_error": row.get("standard_error", ""),
                        "mode": row.get("mode", ""),
                        "avg_length": row.get("avg_length", ""),
                        "n_wins": row.get("n_wins", ""),
                        "n_wins_base": row.get("n_wins_base", ""),
                        "n_draws": row.get("n_draws", ""),
                        "n_total": row.get("n_total", ""),
                        "discrete_win_rate": row.get("discrete_win_rate", ""),
                        "length_controlled_winrate": row.get("length_controlled_winrate", ""),
                        "lc_standard_error": row.get("lc_standard_error", "")
                    }
    return None

import re

def extract_model_from_folder(folder_name):
    # This regex extracts {icr/mapo}_enesru_llama3_8b_{method}
    match = re.search(rf"{dir}_enesru_llama3_8b_[a-zA-Z\-]+", folder_name) # change between mapo, icr or lacomsa
    if match:
        return match.group(0)
    return folder_name

"""
def extract_model_from_folder(folder_name):
    # This regex extracts {icr/mapo}_enesru_llama3_8b_{method}(_\d+)? (with optional version)
    match = re.search(r"icr_enesru_llama3_8b_[a-zA-Z-]+(?:_\d+)?", folder_name)
    if match:
        return match.group(0)
    return folder_name
"""

rows = []

for entry in os.listdir(dir):
    # Only process folders that match the mapo_enesru_llama3 pattern
    if entry.startswith(f"de-results-{dir}_enesru_llama3_") or \
       entry.startswith(f"en-results-{dir}_enesru_llama3_") or \
       entry.startswith(f"es-results-{dir}_enesru_llama3_") or \
       entry.startswith(f"fr-results-{dir}_enesru_llama3_") or \
       entry.startswith(f"ru-results-{dir}_enesru_llama3_"):
        # Extract language and method
        parts = entry.split("_")
        lang = entry[:2]
        method = "_".join(parts[4:-1]) if "w-reinforce" in entry else parts[4]
        model = extract_model_from_folder(entry)

        stats = extract_stats_from_folder(os.path.join(dir, entry), model)
        row = {
            "source": lang,
            "model": model,
            "win_rate": stats["win_rate"],
            "standard_error": stats["standard_error"],
            "mode": stats["mode"],
            "avg_length": stats["avg_length"],
            "n_wins": stats["n_wins"],
            "n_wins_base": stats["n_wins_base"],
            "n_draws": stats["n_draws"],
            "n_total": stats["n_total"],
            "discrete_win_rate": stats["discrete_win_rate"],
            "length_controlled_winrate": stats["length_controlled_winrate"],
            "lc_standard_error": stats["lc_standard_error"]
        }
        rows.append(row)

lang_order = ['de', 'en', 'es', 'fr', 'ru']
model_order = ['w-reinforce', 'dpo', 'ppo', 'sft', 'npo']
# model_order = ['w-reinforce_150426', 'dpo_150426', 'ppo_150426', 'sft_150426', 'npo_150426']

def model_sort_key(model_name):
    for method in model_order:
        if f"_{method}" in model_name:
            return model_order.index(method)
    return len(model_order)

def model_sort_key(model_name):
    # Extract the method part from the model name
    for method in model_order:
        if model_name.endswith(method):
            return model_order.index(method)
    return len(model_order)  # If not found, put at the end

rows.sort(key=lambda x: (lang_order.index(x["source"]), model_sort_key(x["model"])))

# Write to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Results written to {output_csv}")