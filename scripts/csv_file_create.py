import os
import csv

# Path to the {method} folder and output CSV
dir = "responses/mapo"
method_name = "mapo"
output_csv = "results/full_npo/RAIL CrossLingual Transfer - MAPO_Results_Run_01.csv"

# The columns to match runs
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
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            # Strip whitespace from keys and values
            row = {k.strip(): v.strip() for k, v in row.items()}
            # If there's a model column, match it
            if expected_model:
                model_in_row = row.get("model", "") or row.get("", "")  # Sometimes the first column is unnamed
                # IMPORTANT: Require the method-specific prefix to avoid matching other methods.
                required_prefix = f"{method_name}_enesru_llama3_8b_"
                if model_in_row.startswith(required_prefix) and expected_model in model_in_row:
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
    # If we didn't find an exact match but an expected_model was provided,
    # try a looser match based on the method suffix (e.g. 'npo_checkpoint-1' -> 'npo_checkpoint-1'
    # or 'npo') to handle leaderboards that store shorter model names.
    # IMPORTANT: require the model row to start with "lacomsa_enesru_llama3_8b_" to avoid
    # picking up rows from other methods (e.g., mapo_enesru_llama3_8b_npo when looking for lacomsa npo).
    if expected_model:
        method_suffix = expected_model.split(f"{method_name}_enesru_llama3_8b_")[-1]
        required_prefix = f"{method_name}_enesru_llama3_8b_"
        with open(leaderboard_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                model_in_row = row.get("model", "") or row.get("", "")
                # Match only if it starts with the correct prefix AND ends with the suffix
                if model_in_row.startswith(required_prefix) and (method_suffix in model_in_row or model_in_row in method_suffix):
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

# Previous simpler regex caused two issues:
# 1) It didn't include underscores in the model part (so "npo_checkpoint-1" matched only "npo").
# 2) The commented triple-quoted snippet contained an invalid escape sequence and produced
#    a SyntaxWarning. The function below matches the full model token and strips any
#    trailing "_v<digits>" version suffix.
def extract_model_from_folder(folder_name):
    # Capture e.g. '{method}_enesru_llama3_8b_npo_checkpoint-1' (optionally followed by '_v0')
    match = re.search(rf"({method_name}_enesru_llama3_8b_[A-Za-z0-9_-]+)(?:_v\d+)?", folder_name)
    if match:
        model = match.group(1)
        # Ensure we remove any trailing _vN that may have been captured
        model = re.sub(r"_v\d+$", "", model)
        return model
    return folder_name

rows = []

for entry in os.listdir(dir):
    # Only process folders that match {method}_enesru_llama3 pattern
    if entry.startswith(f"de-results-{method_name}_enesru_llama3_") or \
       entry.startswith(f"en-results-{method_name}_enesru_llama3_") or \
       entry.startswith(f"es-results-{method_name}_enesru_llama3_") or \
       entry.startswith(f"fr-results-{method_name}_enesru_llama3_") or \
       entry.startswith(f"ru-results-{method_name}_enesru_llama3_"):
        # Extract language and method
        parts = entry.split("_")
        lang = entry[:2]
        # Some method names contain multiple parts (e.g. "npo_checkpoint-1").
        # Join all parts between the model prefix and the trailing version tag.
        method = "_".join(parts[4:-1]) if len(parts) > 5 else parts[4]
        model = extract_model_from_folder(entry)

        # Use the shorter `method` token when looking up leaderboard rows
        # because leaderboard 'model' values may omit the full prefix.
        stats = extract_stats_from_folder(os.path.join(dir, entry), method)
        if stats is None:
            print(f"Skipping {entry}: no leaderboard/stats found or no matching model row")
            continue

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
model_order = ['w-reinforce', 'dpo_250426', 'ppo', 'sft', 'npo_checkpoint-1', 'npo_checkpoint-2', 'npo_checkpoint-3', 'npo_checkpoint-4', 'npo_checkpoint-5', 'npo_checkpoint-10', 'npo_checkpoint-20', 'npo_checkpoint-30', 'npo']
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