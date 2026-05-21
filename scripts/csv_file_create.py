import argparse
import csv
import os
import re
from pathlib import Path


RESULTS_DIR = Path("results/full_npo")
LANG_ORDER = ["de", "en", "es", "fr", "ru"]
COLUMNS = [
    "source",
    "model",
    "win_rate",
    "standard_error",
    "mode",
    "avg_length",
    "n_wins",
    "n_wins_base",
    "n_draws",
    "n_total",
    "discrete_win_rate",
    "length_controlled_winrate",
    "lc_standard_error",
]

METHODS = {
    "icr": {
        "input_dirs": [Path("responses/icr"), Path("responses/random/icr")],
        "output_csv": RESULTS_DIR / "RAIL CrossLingual Transfer - ICR_Results_Run_01.csv",
    },
    "lacomsa": {
        "input_dirs": [Path("responses/lacomsa"), Path("responses/random/lacomsa")],
        "output_csv": RESULTS_DIR / "RAIL CrossLingual Transfer - LACOMSA_Results_Run_01.csv",
    },
    "mapo": {
        "input_dirs": [Path("responses/mapo"), Path("responses/random/mapo")],
        "output_csv": RESULTS_DIR / "RAIL CrossLingual Transfer - MAPO_Results_Run_01.csv",
    },
}

MODEL_ORDER = [
    "w-reinforce",
    "w-reinforce_random",
    "w-reinforce_negative",
    "dpo_250426",
    "dpo",
    "ppo_150426",
    "ppo",
    "sft_150426",
    "sft",
    "npo_checkpoint-1",
    "npo_checkpoint-2",
    "npo_checkpoint-3",
    "npo_checkpoint-4",
    "npo_checkpoint-5",
    "npo_checkpoint-10",
    "npo_checkpoint-20",
    "npo_checkpoint-30",
    "npo_150426",
    "npo",
]


def stats_from_row(row):
    return {column: row.get(column, "") for column in COLUMNS if column not in {"source", "model"}}


def normalized_row(row):
    return {str(k).strip(): str(v).strip() for k, v in row.items()}


def extract_stats_from_folder(folder_path, model_candidates):
    leaderboard_path = folder_path / "leaderboard.csv"
    if not leaderboard_path.exists():
        return None

    with leaderboard_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = [normalized_row(row) for row in reader]

    candidates = set(model_candidates)
    for row in rows:
        model_in_row = row.get("model", "") or row.get("", "")
        if model_in_row in candidates:
            return stats_from_row(row)

    for row in rows:
        model_in_row = row.get("model", "") or row.get("", "")
        if any(model_in_row.endswith(candidate) for candidate in candidates):
            return stats_from_row(row)

    return None


def strip_version(model_name):
    return re.sub(r"_v\d+$", "", model_name)


def parse_result_folder(entry, method_name):
    pattern = rf"^({'|'.join(LANG_ORDER)})-results-({re.escape(method_name)}(?:_enesru_llama3_8b)?_.+)$"
    match = re.match(pattern, entry)
    if not match:
        return None

    lang, raw_model = match.groups()
    raw_model = strip_version(raw_model)

    if raw_model.startswith(f"{method_name}_enesru_llama3_8b_"):
        model = raw_model
        suffix = raw_model.removeprefix(f"{method_name}_enesru_llama3_8b_")
        candidates = [
            raw_model,
            f"{method_name}_{suffix}",
            suffix,
        ]
    else:
        suffix = raw_model.removeprefix(f"{method_name}_")
        model = f"{method_name}_enesru_llama3_8b_{suffix}"
        candidates = [
            raw_model,
            model,
            suffix,
        ]

    return lang, model, candidates


def collect_rows(method_name, input_dirs):
    rows = []
    seen = set()

    for input_dir in input_dirs:
        if not input_dir.exists():
            continue

        for entry in sorted(os.listdir(input_dir)):
            parsed = parse_result_folder(entry, method_name)
            if parsed is None:
                continue

            lang, model, candidates = parsed
            key = (lang, model)
            if key in seen:
                continue

            stats = extract_stats_from_folder(input_dir / entry, candidates)
            if stats is None:
                print(f"Skipping {input_dir / entry}: no matching leaderboard row")
                continue

            rows.append({"source": lang, "model": model, **stats})
            seen.add(key)

    return rows


def model_suffix(model_name):
    return model_name.split("_8b_", 1)[-1]


def model_sort_key(model_name):
    suffix = model_suffix(model_name)
    for index, expected_suffix in enumerate(MODEL_ORDER):
        if suffix == expected_suffix:
            return index
    return len(MODEL_ORDER)


def write_method_csv(method_name):
    config = METHODS[method_name]
    rows = collect_rows(method_name, config["input_dirs"])
    rows.sort(key=lambda row: (LANG_ORDER.index(row["source"]), model_sort_key(row["model"]), row["model"]))

    output_csv = config["output_csv"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {output_csv} ({len(rows)} rows)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create cross-lingual result CSVs from response leaderboard folders."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=["icr", "lacomsa"],
        help="Methods to generate. Defaults to icr and lacomsa.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for method_name in args.methods:
        write_method_csv(method_name)


if __name__ == "__main__":
    main()
