#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LANGUAGES = ["es", "ru", "en", "fr", "de"]
MODEL_ORDER = ["baseline", "w-reinforce", "w-reinforce_0.1", "w-reinforce_10"]
MODEL_LABELS = {
    "baseline": "baseline",
    "w-reinforce": "w-reinforce",
    "w-reinforce_0.1": "w-reinforce-0.1",
    "w-reinforce_10": "w-reinforce-10",
}

BASELINE_LC = {
    "en": 23.80,
    "es": 15.33,
    "ru": 14.77,
    "de": 12.86,
    "fr": 16.39,
}

BASELINE = {
    "en": 22.61,
    "es": 16.21,
    "ru": 19.32,
    "de": 15.28,
    "fr": 18.88,
}

def parse_model_type(model_name: str) -> str | None:
    normalized = model_name.strip().lower()
    if "_8b_" not in normalized:
        return None

    tail = normalized.split("_8b_", 1)[1]
    if re.search(r"(?:^|_)w-reinforce_0\.1(?:_|$)", tail):
        return "w-reinforce_0.1"
    if re.search(r"(?:^|_)w-reinforce_10(?:_|$)", tail):
        return "w-reinforce_10"
    if re.search(r"(?:^|_)w-reinforce(?:_|$)", tail):
        return "w-reinforce"
    return None


def load_win_rates(csv_path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {model: {} for model in MODEL_ORDER}
    values["baseline"] = {lang: BASELINE_LC[lang] for lang in LANGUAGES if lang in BASELINE_LC}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for raw_row in reader:
            row = {
                (key.strip() if isinstance(key, str) else key): (
                    value.strip() if isinstance(value, str) else value
                )
                for key, value in raw_row.items()
            }

            source = (row.get("source", "") or "").strip().lower()
            if source not in LANGUAGES:
                continue

            model_type = parse_model_type((row.get("model", "") or "").strip())
            if model_type not in MODEL_ORDER:
                continue

            raw_win_rate = row.get("length_controlled_winrate", "")
            if raw_win_rate in ("", None):
                continue

            try:
                win_rate = float(raw_win_rate)
            except (TypeError, ValueError):
                continue

            prev = values[model_type].get(source)
            if prev is None or win_rate > prev:
                values[model_type][source] = win_rate

    return values


def plot_grouped_bars(values: dict[str, dict[str, float]], output_path: Path) -> None:
    group_gap = 1.25
    x = np.arange(len(LANGUAGES)) * group_gap
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for i, model in enumerate(MODEL_ORDER):
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        heights = [values.get(model, {}).get(lang, 0.0) for lang in LANGUAGES]
        ax.bar(x + offset, heights, width=width, label=MODEL_LABELS[model])

    y_max = max(
        values.get(model, {}).get(lang, 0.0)
        for model in MODEL_ORDER
        for lang in LANGUAGES
    )
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([lang.upper() for lang in LANGUAGES])
    ax.set_xlabel("language")
    ax.set_ylabel("length_controlled_winrate")
    ax.set_title("LACOMSA length controlled winrate by language and reinforce variant")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="model")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped bars of length_controlled_winrate for baseline, w-reinforce, "
            "w-reinforce-0.1, and w-reinforce-10 from the LACOMSA full_npo results CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/full_npo/RAIL CrossLingual Transfer - LACOMSA_Results_Run_01.csv"),
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualisation/length_controlled_winrate_reinforce_lacomsa_grouped.png"),
        help="Path to output PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_win_rates(args.input)
    plot_grouped_bars(values, args.output)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()