#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_WIN_RATE = {
    "en": 22.61,
    "es": 16.21,
    "ru": 19.32,
    "de": 15.28,
    "fr": 18.88,
}
LANGUAGES = ["en", "es", "ru", "de", "fr"]
LANG_COLORS = {
    "en": "tab:blue",
    "es": "tab:orange",
    "ru": "tab:green",
    "de": "tab:red",
    "fr": "tab:purple",
}
METHODS = {
    "ICR": Path("results/final/icr_w_reinforce_dpo_subset.csv"),
    "LACOMSA": Path("results/final/lacomsa_w_reinforce_dpo_subset.csv"),
}
MODEL_ORDER = [
    "w-reinforce",
    "w-reinforce_0.1_1.0",
    "w-reinforce_10.0_1.0",
    "dpo",
    "w-reinforce_random",
    "w-reinforce_positive",
    "w-reinforce_negative",
]


def load_values(csv_path: Path) -> dict[str, dict[str, float]]:
    values = {model: {} for model in MODEL_ORDER}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [field.strip() for field in reader.fieldnames]

        for raw_row in reader:
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            lang = row.get("source", "").lower()
            model = row.get("model", "")
            if lang not in LANGUAGES or model not in MODEL_ORDER:
                continue
            values[model][lang] = float(row["win_rate"])

    return values


def present_models(values: dict[str, dict[str, float]]) -> list[str]:
    return [model for model in MODEL_ORDER if values.get(model)]


def plot_chart(input_paths: dict[str, Path], output_path: Path) -> None:
    all_values = {method: load_values(path) for method, path in input_paths.items()}
    all_models = {
        method: present_models(values)
        for method, values in all_values.items()
    }

    fig, axes = plt.subplots(1, len(input_paths), figsize=(20, 7), sharey=True)
    if len(input_paths) == 1:
        axes = [axes]

    max_bar = max(
        value
        for values in all_values.values()
        for model_values in values.values()
        for value in model_values.values()
    )
    y_max = max(max_bar, max(BASELINE_WIN_RATE.values())) * 1.12

    for ax, (method, values) in zip(axes, all_values.items()):
        models = all_models[method]
        x = np.arange(len(models))
        width = min(0.75 / len(LANGUAGES), 0.16)

        for lang_index, lang in enumerate(LANGUAGES):
            offset = (lang_index - (len(LANGUAGES) - 1) / 2) * width
            heights = [values.get(model, {}).get(lang, 0.0) for model in models]
            ax.bar(
                x + offset,
                heights,
                width=width,
                color=LANG_COLORS[lang],
                alpha=0.82,
                label=f"{lang.upper()} win rate",
            )

        for lang in LANGUAGES:
            ax.axhline(
                BASELINE_WIN_RATE[lang],
                color=LANG_COLORS[lang],
                linestyle="--",
                linewidth=1.3,
                alpha=0.45,
                label=f"{lang.upper()} baseline win rate",
            )

        ax.set_title(method, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel("Rate")
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_ylim(0, y_max)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.suptitle(
        "W-Reinforce Variants & DPO: Win Rate",
        fontsize=15,
        y=0.97,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot win rate bars for W-Reinforce variants and DPO."
    )
    parser.add_argument(
        "--icr",
        type=Path,
        default=METHODS["ICR"],
        help="Input ICR subset CSV.",
    )
    parser.add_argument(
        "--lacomsa",
        type=Path,
        default=METHODS["LACOMSA"],
        help="Input LACOMSA subset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/figures/w_reinforce_winrate_barchart.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_chart({"ICR": args.icr, "LACOMSA": args.lacomsa}, args.output)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()
