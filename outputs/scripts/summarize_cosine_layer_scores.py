#!/usr/bin/env python3
"""Summarize raw cosine similarity scores by model and language.

Reads:
  outputs/results/cosine_sim/{model}/seed_*/*.csv

Writes one CSV by default:
  outputs/results/aggregated/cosine_sim/layer_12_18_mean_and_layer_16_scores.csv

For each model/language pair, the output includes:
  - mean cosine_sim over layers 12-18 across seeds
  - mean cosine_sim_adjusted over layers 12-18 across seeds
  - layer 16 cosine_sim averaged across seeds
  - layer 16 cosine_sim_adjusted averaged across seeds
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


INPUT_ROOT = Path("outputs/results/cosine_sim")
OUTPUT_PATH = Path(
    "outputs/results/aggregated/cosine_sim/layer_12_18_mean_and_layer_16_scores.csv"
)
METRICS = ("cosine_sim", "cosine_sim_adjusted")
MODEL_ORDER = [
    "base-sft-dpo",
    "icr-dpo-ckpt36",
    "icr-npo-ckpt10",
    "icr-npo-ckpt36",
    "icr-ppo-ckpt36",
    "icr-w-reinforce",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cosine scores over a layer range and at one target layer."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help=f"Raw cosine_sim results root. Default: {INPUT_ROOT}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path. Default: {OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=12,
        help="First layer for range means. Default: 12.",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=18,
        help="Last layer for range means, inclusive. Default: 18.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=16,
        help="Layer to extract separately. Default: 16.",
    )
    return parser.parse_args()


def finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value.strip())
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def seed_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seed_"):
            return part.removeprefix("seed_")
    raise ValueError(f"Could not find seed_* directory in path: {path}")


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model), model
    except ValueError:
        return len(MODEL_ORDER), model


def collect_rows(
    input_root: Path,
    *,
    start_layer: int,
    end_layer: int,
    target_layer: int,
) -> list[dict[str, object]]:
    range_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    target_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    range_seeds: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    target_seeds: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for model_dir in sorted((path for path in input_root.iterdir() if path.is_dir()), key=lambda path: model_sort_key(path.name)):
        model = model_dir.name
        for csv_path in sorted(model_dir.glob("seed_*/*.csv")):
            language = csv_path.stem
            seed = seed_from_path(csv_path)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                reader.fieldnames = [field.strip() for field in reader.fieldnames or []]
                required = {"layer", *METRICS}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

                for row_number, raw_row in enumerate(reader, start=2):
                    row = {
                        (key.strip() if key is not None else key): (
                            value.strip() if isinstance(value, str) else value
                        )
                        for key, value in raw_row.items()
                    }
                    try:
                        layer = int(row.get("layer", ""))
                    except ValueError as exc:
                        raise ValueError(f"Invalid layer in {csv_path}:{row_number}") from exc

                    for metric in METRICS:
                        value = finite_float(row.get(metric))
                        if value is None:
                            continue

                        key = (language, model, metric)
                        if start_layer <= layer <= end_layer:
                            range_values[key].append(value)
                            range_seeds[key].add(seed)
                        if layer == target_layer:
                            target_values[key].append(value)
                            target_seeds[key].add(seed)

    language_models = sorted(
        {(language, model) for language, model, _metric in set(range_values) | set(target_values)},
        key=lambda item: (item[0], model_sort_key(item[1])),
    )
    rows = []
    for language, model in language_models:
        row: dict[str, object] = {"language": language, "model": model}
        for metric in METRICS:
            metric_range_values = range_values.get((language, model, metric), [])
            metric_target_values = target_values.get((language, model, metric), [])
            row[f"{metric}_layers_{start_layer}_{end_layer}_mean"] = (
                mean(metric_range_values) if metric_range_values else ""
            )
            row[f"{metric}_layers_{start_layer}_{end_layer}_n_values"] = len(metric_range_values)
            row[f"{metric}_layers_{start_layer}_{end_layer}_n_seeds"] = len(
                range_seeds.get((language, model, metric), set())
            )
            row[f"{metric}_layer_{target_layer}_mean"] = (
                mean(metric_target_values) if metric_target_values else ""
            )
            row[f"{metric}_layer_{target_layer}_n_seeds"] = len(
                target_seeds.get((language, model, metric), set())
            )
        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    if args.end_layer < args.start_layer:
        raise ValueError("--end-layer must be greater than or equal to --start-layer")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    rows = collect_rows(
        args.input_root,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        target_layer=args.target_layer,
    )

    fieldnames = ["language", "model"]
    for metric in METRICS:
        fieldnames.extend(
            [
                f"{metric}_layers_{args.start_layer}_{args.end_layer}_mean",
                f"{metric}_layers_{args.start_layer}_{args.end_layer}_n_values",
                f"{metric}_layers_{args.start_layer}_{args.end_layer}_n_seeds",
                f"{metric}_layer_{args.target_layer}_mean",
                f"{metric}_layer_{args.target_layer}_n_seeds",
            ]
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
