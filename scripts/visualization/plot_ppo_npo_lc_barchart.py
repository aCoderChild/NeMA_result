#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


APPROACH_ORDER = ["PPO", "NPO (Best Ckpt)"]
APPROACH_LABELS = {
    "PPO": "PPO",
    "NPO (Best Ckpt)": "NPO",
}
INCLUDED_METHODS = {"ICR", "LaCoMSA"}
LANGUAGE_TABLE_ORDER = ["es", "ru", "en", "de", "fr"]
LANGUAGE_PLOT_ORDER = ["de", "en", "es", "fr", "ru"]
LANGUAGE_LABELS = {
    "de": "DE",
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "ru": "RU",
}
DEFAULT_INPUTS = [
    Path("table_methods_comparison.tex"),
]
FONT_SCALE = 4.0
FIGURE_SIZE = (23, 15)


def extract_lc_values(table_paths: list[Path]) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    row_pattern = re.compile(
        r"(?:\\multirow\{\d+\}\{\*\}\{\\textbf\{(?P<multirow_method>[^}]+)\}\}|\\textbf\{(?P<single_method>[^}]+)\}|)\s*&\s*"
        r"(?P<approach>[^&]+)\s*&(?P<rest>.*?)\\\\"
    )

    for table_path in table_paths:
        current_method: str | None = None
        for line in table_path.read_text().splitlines():
            match = row_pattern.search(line)
            if not match:
                continue

            method = match.group("multirow_method") or match.group("single_method")
            if method:
                current_method = method
            if current_method is None or current_method not in INCLUDED_METHODS:
                continue

            approach = match.group("approach").strip()
            if approach not in APPROACH_ORDER:
                continue

            cells = [cell.strip() for cell in match.group("rest").split("&")]
            if len(cells) < 12:
                continue

            # The table stores each language as LC, WR pairs in this order:
            # es, ru, en, de, fr, then Avg LC and Avg WR.
            for lang_index, language in enumerate(LANGUAGE_TABLE_ORDER):
                lc_cell_index = lang_index * 2
                values[approach][language].append(float(cells[lc_cell_index]))

    return values


def average_lc_values(
    values: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    return {
        approach: {
            language: sum(language_values) / len(language_values)
            for language, language_values in approach_values.items()
            if language_values
        }
        for approach, approach_values in values.items()
    }


def plot_barchart(values: dict[str, dict[str, float]], output_path: Path) -> None:
    x = np.arange(len(LANGUAGE_PLOT_ORDER))
    width = 0.28
    colors = ["#F0997B", "#8E44AD"]

    with plt.rc_context({"font.size": plt.rcParams["font.size"] * FONT_SCALE}):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=200)

        for index, approach in enumerate(APPROACH_ORDER):
            offsets = x + (index - (len(APPROACH_ORDER) - 1) / 2) * width
            heights = [
                values.get(approach, {}).get(language, np.nan)
                for language in LANGUAGE_PLOT_ORDER
            ]
            ax.bar(
                offsets,
                heights,
                width=width,
                label=APPROACH_LABELS[approach],
                color=colors[index],
                edgecolor="#4A4A4A",
                linewidth=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([LANGUAGE_LABELS[language] for language in LANGUAGE_PLOT_ORDER])
        ax.set_xlabel("Language")
        ax.set_ylabel("Average\nLength-Controlled\nWin Rate")
        ax.set_title("Average Length-Controlled Win Rate for PPO and NPO")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(
            title="Model",
            frameon=True,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0.24, 1, 1])
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average LC win rate by language for PPO and NPO best checkpoint."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input LaTeX table path. Defaults to table_methods_comparison.tex.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/figures/ppo_npo_lc_barchart.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = average_lc_values(extract_lc_values(args.inputs))
    plot_barchart(values, args.output)
    print(f"Saved chart to: {args.output}")
    for approach in APPROACH_ORDER:
        print(
            approach,
            ", ".join(
                f"{language}: {values.get(approach, {}).get(language, float('nan')):.2f}"
                for language in LANGUAGE_PLOT_ORDER
            ),
        )


if __name__ == "__main__":
    main()
