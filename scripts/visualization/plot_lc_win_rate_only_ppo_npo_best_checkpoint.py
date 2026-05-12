#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt


LANGUAGES = ["de", "en", "es", "fr", "ru"]
TARGET_METHOD = "lacomsa"
TARGET_METHOD_LABEL = "LACOMSA"
HIGHLIGHT_MODEL = "w-reinforce_0.1_1.0"
BRIGHT_HIGHLIGHT_COLOR = "#4FC3F7"


def parse_method_and_model(model_name: str) -> tuple[str | None, str | None]:
    normalized_model = model_name.strip().lower()
    method_match = re.match(r"^(lacomsa)(?:_|-)", normalized_model)
    if not method_match or "_8b_" not in normalized_model:
        return None, None

    model_tail = normalized_model.split("_8b_", 1)[1]
    if "w-reinforce" not in model_tail and "dpo_250426" not in model_tail:
        return None, None

    return method_match.group(1), model_tail


def load_length_controlled_winrate(csv_path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source", "") or "").strip().lower()
            model_name = (row.get("model", "") or "").strip()

            if source not in LANGUAGES:
                continue

            method, model_key = parse_method_and_model(model_name)
            if method != TARGET_METHOD or model_key is None:
                continue

            raw_value = row.get("length_controlled_winrate", "") or row.get(
                "win_rate", ""
            )
            if raw_value in ("", None):
                continue

            try:
                winrate = float(raw_value)
            except (TypeError, ValueError):
                continue

            prev = values[model_key].get(source)
            if prev is None or winrate > prev:
                values[model_key][source] = winrate

    return values


def collect_models(values: dict[str, dict[str, float]]) -> list[str]:
    model_names = set(values.keys())
    if not model_names:
        return []

    def sort_key(model_name: str) -> tuple[int, str]:
        if model_name == HIGHLIGHT_MODEL:
            return (0, model_name)
        if "w-reinforce" in model_name:
            return (1, model_name)
        return (2, model_name)

    return sorted(model_names, key=sort_key)


def build_model_colors(model_order: list[str]) -> dict[str, tuple[float, float, float, float] | str]:
    non_highlight = [m for m in model_order if m != HIGHLIGHT_MODEL]
    color_map: dict[str, tuple[float, float, float, float] | str] = {}

    if HIGHLIGHT_MODEL in model_order:
        color_map[HIGHLIGHT_MODEL] = BRIGHT_HIGHLIGHT_COLOR

    total = max(len(non_highlight), 1)
    for idx, model_name in enumerate(non_highlight):
        # Greys(0.45 -> 0.8): medium-dark gray tones to keep non-highlight models muted.
        shade = 0.45 + (0.35 * idx / total)
        color_map[model_name] = plt.cm.Greys(shade)

    return color_map


def plot_grouped_barchart(values: dict[str, dict[str, float]], output_path: Path) -> None:
    model_order = collect_models(values)
    if not model_order:
        raise ValueError("No matching model containing 'w-reinforce' or 'dpo' found.")

    x = list(range(len(LANGUAGES)))
    width = min(0.75 / max(len(model_order), 1), 0.22)
    colors = build_model_colors(model_order)

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))

    max_bar = max(
        values.get(model, {}).get(lang, 0.0)
        for model in model_order
        for lang in LANGUAGES
    )
    y_max = max(5.0, max_bar) * 1.15

    for model_idx, model_name in enumerate(model_order):
        offset = (model_idx - (len(model_order) - 1) / 2) * width
        heights = [values.get(model_name, {}).get(lang, 0.0) for lang in LANGUAGES]
        positions = [xi + offset for xi in x]
        ax.bar(
            positions,
            heights,
            width=width,
            color=colors[model_name],
            label=model_name,
            edgecolor="black" if model_name == HIGHLIGHT_MODEL else None,
            linewidth=0.9 if model_name == HIGHLIGHT_MODEL else 0.0,
            alpha=0.95 if model_name == HIGHLIGHT_MODEL else 0.9,
        )

    ax.set_title(TARGET_METHOD_LABEL)
    ax.set_xlabel("Language")
    ax.set_xticks(x)
    ax.set_xticklabels([lang.upper() for lang in LANGUAGES])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("length_controlled_winrate")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        ncol=min(4, max(len(model_order), 1)),
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "LACOMSA Length-Controlled Winrate for W-REINFORCE and DPO"
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.subplots_adjust(bottom=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped barcharts of length_controlled_winrate for models containing "
            "'w-reinforce' or 'dpo'. Highlight w-reinforce_0.1_1.0 with a bright color."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/gangstat/NeMA_result/results/lacomsa_with_w-reinforce/lacomsa.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/gangstat/NeMA_result/visualisation/"
            "length_controlled_winrate_lacomsa_w-reinforce_dpo.png"
        ),
        help="Path to output image (PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_length_controlled_winrate(args.input)
    plot_grouped_barchart(values, args.output)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()
