#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAMBDA_ORDER = ["0.1", "1", "10"]
LANGUAGE_ORDER = ["de", "en", "es", "fr", "ru"]
LANGUAGE_LABELS = {
    "es": "ES",
    "ru": "RU",
    "en": "EN",
    "de": "DE",
    "fr": "FR",
}


def extract_language_winrates(table_path: Path) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    current_model: str | None = None

    row_pattern = re.compile(
        r"(?:\\multirow\{\d+\}\{\*\}\{\\textbf\{(?P<multirow_model>[^}]+)\}\}|\\textbf\{(?P<single_model>[^}]+)\}|)\s*&\s*"
        r"(?P<lambda>0\.1|1|10)\s*&(?P<rest>.*?)\\\\"
    )

    for line in table_path.read_text().splitlines():
        match = row_pattern.search(line)
        if not match:
            continue

        model = match.group("multirow_model") or match.group("single_model")
        if model:
            current_model = model
        if current_model is None:
            continue

        cells = [cell.strip() for cell in match.group("rest").split("&")]
        if len(cells) < 12:
            continue

        lambda_value = match.group("lambda")
        # The table stores each language as LC, WR pairs in this order:
        # es, ru, en, de, fr, then Avg LC and Avg WR.
        for lang_index, language in enumerate(LANGUAGE_ORDER):
            wr_cell_index = (lang_index * 2) + 1
            values[lambda_value][language].append(float(cells[wr_cell_index]))

    return values


def average_language_winrates(
    values: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    return {
        lambda_value: {
            language: sum(winrates) / len(winrates)
            for language, winrates in language_values.items()
            if winrates
        }
        for lambda_value, language_values in values.items()
    }


def plot_barchart(values: dict[str, dict[str, float]], output_path: Path) -> None:
    x = np.arange(len(LANGUAGE_ORDER))
    width = 0.22
    colors = ["#55BDEB", "#BDBDBD", "#8F8F8F"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)

    for index, lambda_value in enumerate(LAMBDA_ORDER):
        offsets = x + (index - (len(LAMBDA_ORDER) - 1) / 2) * width
        heights = [
            values.get(lambda_value, {}).get(language, np.nan)
            for language in LANGUAGE_ORDER
        ]
        ax.bar(
            offsets,
            heights,
            width=width,
            label=f"$\\lambda={lambda_value}$",
            color=colors[index],
            edgecolor="#4A4A4A",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LANGUAGE_LABELS[language] for language in LANGUAGE_ORDER])
    ax.set_xlabel("Language")
    ax.set_ylabel("Average Win Rate")
    ax.set_title("Average Win Rate by Language and $\\lambda$")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(
        title="$\\lambda$",
        frameon=True,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average WR by lambda from table_lambda_reorganized.tex."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("table_lambda_reorganized.tex"),
        help="Input LaTeX table path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/figures/lambda_avg_wr_barchart.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = average_language_winrates(extract_language_winrates(args.input))
    plot_barchart(values, args.output)
    print(f"Saved chart to: {args.output}")
    for lambda_value in LAMBDA_ORDER:
        print(
            f"lambda={lambda_value}",
            ", ".join(
                f"{language}: {values.get(lambda_value, {}).get(language, float('nan')):.2f}"
                for language in LANGUAGE_ORDER
            ),
        )


if __name__ == "__main__":
    main()
