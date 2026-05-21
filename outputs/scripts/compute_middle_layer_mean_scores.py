#!/usr/bin/env python3
"""Compute language/model mean scores over a selected layer range.

By default this writes the same combined columns as:
  outputs/results/aggregated/mean_scores_by_language.csv

using only layers 12 through 18, inclusive.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


AGGREGATED_ROOT = Path("outputs/results/aggregated")
OUTPUT_PATH = Path("outputs/results/aggregated/mean_scores_by_language_layers_12_18.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine language/model mean ID and sil-score values over a layer range."
    )
    parser.add_argument(
        "--aggregated-root",
        type=Path,
        default=AGGREGATED_ROOT,
        help=f"Input aggregated root. Default: {AGGREGATED_ROOT}.",
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
        help="First layer to include. Default: 12.",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=18,
        help="Last layer to include, inclusive. Default: 18.",
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


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {
        (key.strip() if key is not None else key): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
    }


def layer_in_range(value: str | None, start_layer: int, end_layer: int) -> bool:
    try:
        layer = int((value or "").strip())
    except ValueError:
        return False
    return start_layer <= layer <= end_layer


def read_layer_mean_values(
    csv_path: Path,
    *,
    mean_column: str,
    start_layer: int,
    end_layer: int,
) -> list[float]:
    values = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip() for field in reader.fieldnames or []]
        required_columns = {"layer", mean_column}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

        for raw_row in reader:
            row = clean_row(raw_row)
            if not layer_in_range(row.get("layer"), start_layer, end_layer):
                continue
            value = finite_float(row.get(mean_column))
            if value is not None:
                values.append(value)

    return values


def collect_metric_rows(
    metric_root: Path,
    *,
    file_suffix: str,
    mean_column: str,
    output_column: str,
    n_layers_column: str,
    start_layer: int,
    end_layer: int,
) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}

    for model_dir in sorted(path for path in metric_root.iterdir() if path.is_dir()):
        for csv_path in sorted(model_dir.glob(f"*{file_suffix}")):
            if csv_path.name.startswith("all_languages"):
                continue

            language = (
                csv_path.name.removesuffix("_layers.csv")
                if file_suffix == "_layers.csv"
                else csv_path.stem
            )
            values = read_layer_mean_values(
                csv_path,
                mean_column=mean_column,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            if not values:
                continue

            key = (language, model_dir.name)
            rows[key] = {
                "language": language,
                "model": model_dir.name,
                output_column: mean(values),
                n_layers_column: len(values),
            }

    return rows


def main() -> None:
    args = parse_args()
    if args.end_layer < args.start_layer:
        raise ValueError("--end-layer must be greater than or equal to --start-layer")

    id_rows = collect_metric_rows(
        args.aggregated_root / "id",
        file_suffix="_layers.csv",
        mean_column="id_mean",
        output_column="mean_id",
        n_layers_column="id_n_layers",
        start_layer=args.start_layer,
        end_layer=args.end_layer,
    )
    sil_rows = collect_metric_rows(
        args.aggregated_root / "sil_score",
        file_suffix=".csv",
        mean_column="silhouette_score_mean",
        output_column="mean_silhouette_score",
        n_layers_column="sil_score_n_layers",
        start_layer=args.start_layer,
        end_layer=args.end_layer,
    )

    combined = {}
    for key, row in id_rows.items():
        combined.setdefault(key, {"language": key[0], "model": key[1]}).update(row)
    for key, row in sil_rows.items():
        combined.setdefault(key, {"language": key[0], "model": key[1]}).update(row)

    fieldnames = [
        "language",
        "model",
        "mean_id",
        "id_n_layers",
        "mean_silhouette_score",
        "sil_score_n_layers",
    ]
    rows = sorted(combined.values(), key=lambda row: (row["language"], row["model"]))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} rows for layers {args.start_layer}-{args.end_layer} "
        f"to {args.output_path}"
    )


if __name__ == "__main__":
    main()
