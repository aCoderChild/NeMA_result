#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


CHECKPOINTS = [1, 2, 3, 4, 5, 10, 20, 30, 36]
LANGUAGE_ORDER = ["en", "es", "fr", "de", "ru"]
LANGUAGE_LABELS = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
}
LANGUAGE_COLORS = {
    "en": "#2563EB",
    "es": "#F97316",
    "fr": "#7C3AED",
    "de": "#DC2626",
    "ru": "#059669",
}
BASE_FONT_SIZE = 27
TITLE_FONT_SIZE = 32
ANNOTATION_FONT_SIZE = 23
AXIS_LABEL_FONT_SIZE = 27
FIGURE_SIZE = (17.5, 10.5)


def checkpoint_from_model(model_name: str) -> int | None:
    model_name = model_name.strip().lower()
    if model_name == "npo":
        return 36

    match = re.fullmatch(r"npo_checkpoint-(\d+)", model_name)
    if not match:
        return None

    return int(match.group(1))


def load_npo_winrates(csv_path: Path) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = defaultdict(dict)

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [field.strip() for field in reader.fieldnames]

        for raw_row in reader:
            row = {
                key.strip(): value.strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            language = row.get("source", "").lower()
            checkpoint = checkpoint_from_model(row.get("model", ""))

            if not language or checkpoint not in CHECKPOINTS:
                continue

            values[language][checkpoint] = float(row["win_rate"])

    return values


def ordered_languages(values: dict[str, dict[int, float]]) -> list[str]:
    ordered = [language for language in LANGUAGE_ORDER if language in values]
    return ordered + sorted(set(values) - set(ordered))


def plot_npo_winrates(
    values: dict[str, dict[int, float]],
    output_path: Path,
    method_name: str,
    axis_label_font_size: int,
) -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "font.size": BASE_FONT_SIZE,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    for language in ordered_languages(values):
        y_values = [values[language].get(checkpoint) for checkpoint in CHECKPOINTS]
        ax.plot(
            CHECKPOINTS,
            y_values,
            marker="o",
            linewidth=4.2,
            markersize=9.5,
            markeredgecolor="white",
            markeredgewidth=2.0,
            color=LANGUAGE_COLORS.get(language),
            label=LANGUAGE_LABELS.get(language, language.upper()),
        )

    ax.axvline(
        CHECKPOINTS[-1],
        color="#94A3B8",
        linestyle=(0, (4, 4)),
        linewidth=2.0,
        alpha=0.8,
    )
    ax.text(
        CHECKPOINTS[-1],
        ax.get_ylim()[1] * 0.96,
        "NPO",
        color="#64748B",
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight="bold",
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E2E8F0"},
    )

    ax.set_title(
        f"{method_name.upper()} NPO Win Rate Across Checkpoints",
        fontsize=TITLE_FONT_SIZE,
        fontweight="normal",
        pad=20,
    )
    ax.set_xlabel("Checkpoint", fontsize=axis_label_font_size)
    ax.set_ylabel("Win Rate", fontsize=axis_label_font_size)
    ax.set_xticks(CHECKPOINTS)
    ax.set_xticklabels(
        [str(x) if x != 36 else "36\nNPO" for x in CHECKPOINTS],
        rotation=35,
        ha="right",
    )
    ax.set_xlim(min(CHECKPOINTS) - 0.7, max(CHECKPOINTS) + 2.3)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.55)
    ax.grid(False, axis="x")
    ax.legend(
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.30)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NPO win rate scores by checkpoint, with one line per language."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/final/icr.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/figures/npo_winrate_linegraph_icr.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--method",
        default="ICR",
        help="Method name to show in the chart title.",
    )
    parser.add_argument(
        "--axis-label-font-size",
        type=int,
        default=AXIS_LABEL_FONT_SIZE,
        help="Font size for the x-axis and y-axis labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_npo_winrates(args.input)
    plot_npo_winrates(values, args.output, args.method, args.axis_label_font_size)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()
