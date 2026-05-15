#!/usr/bin/env python3
"""Average layerwise sil_score metric CSVs across layers.

Reads outputs/sil_score/{metric}.csv and writes
outputs/sil_score/avg_{metric}.csv with one row per model and language.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_METRICS = [
    "semantic_silhouette",
    "language_silhouette",
    "hub_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average metric values across all layers from sil_score CSVs."
    )
    parser.add_argument(
        "--sil-score-dir",
        type=Path,
        default=Path("outputs/sil_score"),
        help="Directory containing semantic_silhouette.csv, language_silhouette.csv, and hub_gap.csv.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric CSV basenames to average.",
    )
    return parser.parse_args()


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def average_metric_csv(input_path: Path) -> list[dict[str, object]]:
    values = defaultdict(list)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"model", "language", "layer", "mean"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_path} is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            value = finite_float(row.get("mean", ""))
            if value is None:
                raise ValueError(f"Invalid mean value in {input_path}:{row_number}")
            values[(row["model"], row["language"])].append(value)

    rows = []
    for (model, language), layer_means in sorted(
        values.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        avg_mean = mean(layer_means)
        layer_std = stdev(layer_means) if len(layer_means) > 1 else 0.0
        rows.append(
            {
                "model": model,
                "language": language,
                "avg_mean": avg_mean,
                "std_across_layers": layer_std,
                "avg_mean_pm_layer_std": f"{avg_mean:.6f} +- {layer_std:.6f}",
                "n_layers": len(layer_means),
            }
        )
    return rows


def write_avg_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "language",
                "avg_mean",
                "std_across_layers",
                "avg_mean_pm_layer_std",
                "n_layers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    for metric in args.metrics:
        input_path = args.sil_score_dir / f"{metric}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Metric CSV does not exist: {input_path}")

        rows = average_metric_csv(input_path)
        output_path = args.sil_score_dir / f"avg_{metric}.csv"
        write_avg_csv(output_path, rows)
        print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
