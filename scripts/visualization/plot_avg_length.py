#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCE_ORDER = ["de", "en", "es", "fr", "ru"]


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

    checkpoint_match = re.search(r"npo_checkpoint-(\d+)", tail)
    if checkpoint_match:
        return f"npo_checkpoint-{int(checkpoint_match.group(1))}"

    base_match = re.search(r"(?:^|_)(dpo|ppo|sft|npo)(?:_|$)", tail)
    if base_match:
        return base_match.group(1)

    return tail


def sort_model_types(model_types: list[str]) -> list[str]:
    priority = {
        "w-reinforce": 0,
        "w-reinforce_0.1": 1,
        "w-reinforce_10": 2,
        "dpo": 3,
        "ppo": 4,
        "sft": 5,
    }

    checkpoints: list[tuple[int, str]] = []
    others: list[str] = []
    for model in model_types:
        m = re.match(r"^npo_checkpoint-(\d+)$", model)
        if m:
            checkpoints.append((int(m.group(1)), model))
        else:
            others.append(model)

    ordered = [m for m in priority if m in model_types]
    ordered.extend(model for _idx, model in sorted(checkpoints))

    if "npo" in model_types:
        ordered.append("npo")

    for model in sorted(others):
        if model not in ordered:
            ordered.append(model)

    return ordered


def load_avg_lengths(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    values: dict[tuple[str, str], float] = {}
    model_types: set[str] = set()

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
            model_name = (row.get("model", "") or "").strip()
            raw_avg_length = row.get("avg_length", "")

            if source not in SOURCE_ORDER or raw_avg_length in ("", None):
                continue

            model_type = parse_model_type(model_name)
            if not model_type:
                continue

            try:
                avg_length = float(raw_avg_length)
            except (TypeError, ValueError):
                continue

            values[(source, model_type)] = avg_length
            model_types.add(model_type)

    if not values:
        raise ValueError(f"No avg_length values loaded from {csv_path}")

    ordered_models = sort_model_types(list(model_types))
    matrix = np.full((len(SOURCE_ORDER), len(ordered_models)), np.nan)

    for r, source in enumerate(SOURCE_ORDER):
        for c, model in enumerate(ordered_models):
            if (source, model) in values:
                matrix[r, c] = values[(source, model)]

    return SOURCE_ORDER, ordered_models, matrix


def plot_heatmap(sources: list[str], models: list[str], matrix: np.ndarray, output_path: Path) -> None:
    fig_width = max(12, 0.8 * len(models) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#f5f5f5")
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels(sources)
    ax.set_xlabel("model")
    ax.set_ylabel("source")
    ax.set_title("Average length by source and model")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            if np.isnan(value):
                continue
            ax.text(c, r, f"{int(round(value))}", ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("avg_length")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot avg_length heatmap for all model types from LACOMSA full_npo results."
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
        default=Path("visualisation/average_length_lacomsa_heatmap.png"),
        help="Path to output PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources, models, matrix = load_avg_lengths(args.input)
    plot_heatmap(sources, models, matrix, args.output)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()