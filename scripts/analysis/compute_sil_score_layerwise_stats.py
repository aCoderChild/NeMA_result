#!/usr/bin/env python3
"""Compute layerwise silhouette-score statistics by language.

The script reads JSONL files under outputs/sil_score/{model}/seed_*/*.jsonl,
groups rows by model, language, layer, and metric, and writes one CSV per metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import DefaultDict


DEFAULT_MODELS = [
    "base-sft-dpo",
    "icr_dpo_ckpt36",
    "icr_npo_ckpt10",
    "icr_ppo_ckpt36",
    "icr_reinforce",
]
MODEL_ALIASES = {
    "bast-sft-dpo": "base-sft-dpo",
}
DEFAULT_METRICS = [
    "semantic_silhouette",
    "language_silhouette",
    "hub_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean/std layerwise sil_score metrics per language and model."
    )
    parser.add_argument(
        "--sil-score-dir",
        type=Path,
        default=Path("outputs/sil_score"),
        help="Root directory containing model sil_score folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model folders to process. 'bast-sft-dpo' is accepted as an alias for 'base-sft-dpo'.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Numeric JSONL fields to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metric CSV files. Defaults to --sil-score-dir.",
    )
    return parser.parse_args()


def normalize_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def language_from_path(path: Path) -> str:
    """Return language code from names such as 'fr.jsonl' or 'fr (1).jsonl'."""
    return re.sub(r"\s+\(\d+\)$", "", path.stem)


def finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def collect_values(
    model_dir: Path, metrics: list[str]
) -> DefaultDict[tuple[str, int, str], list[float]]:
    values: DefaultDict[tuple[str, int, str], list[float]] = defaultdict(list)

    for jsonl_path in sorted(model_dir.glob("seed_*/*.jsonl")):
        language = language_from_path(jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {jsonl_path}:{line_number}: {exc}") from exc

                layer = record.get("layer")
                if not isinstance(layer, int):
                    raise ValueError(f"Missing integer layer in {jsonl_path}:{line_number}")

                for metric in metrics:
                    value = finite_number(record.get(metric))
                    if value is not None:
                        values[(language, layer, metric)].append(value)

    return values


def build_stats_rows(
    model: str,
    values: DefaultDict[tuple[str, int, str], list[float]],
) -> list[dict[str, object]]:
    rows = []
    for (language, layer, metric), metric_values in sorted(values.items()):
        metric_mean = mean(metric_values)
        metric_std = stdev(metric_values) if len(metric_values) > 1 else 0.0
        rows.append(
            {
                "model": model,
                "language": language,
                "layer": layer,
                "metric": metric,
                "mean": metric_mean,
                "std": metric_std,
                "mean_pm_std": f"{metric_mean:.6f} +- {metric_std:.6f}",
                "n": len(metric_values),
            }
        )
    return rows


def write_stats_csv(csv_path: Path, rows: list[dict[str, object]], include_metric: bool = False) -> None:
    fieldnames = ["model", "language", "layer"]
    if include_metric:
        fieldnames.append("metric")
    fieldnames.extend(["mean", "std", "mean_pm_std", "n"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows = []
    output_dir = args.output_dir or args.sil_score_dir

    for raw_model in args.models:
        model = normalize_model_name(raw_model)
        model_dir = args.sil_score_dir / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        values = collect_values(model_dir=model_dir, metrics=args.metrics)
        if not values:
            raise ValueError(f"No metric values found under {model_dir}")

        rows = build_stats_rows(model=model, values=values)
        all_rows.extend(rows)
        print(f"Computed {len(rows)} rows for {model}")

    for metric in args.metrics:
        metric_rows = [row for row in all_rows if row["metric"] == metric]
        csv_path = output_dir / f"{metric}.csv"
        write_stats_csv(csv_path=csv_path, rows=metric_rows)
        print(f"Saved {len(metric_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
