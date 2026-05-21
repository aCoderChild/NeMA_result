#!/usr/bin/env python3
"""Average aggregated layer scores into one mean score per model and language.

Reads:
  outputs/results/aggregated/sil_score/{model}/{lang}.csv
  outputs/results/aggregated/id/{model}/{lang}_layers.csv

Writes:
  outputs/results/sil_score/{lang}.csv
  outputs/results/id/{lang}.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


AGGREGATED_ROOT = Path("outputs/results/aggregated")
OUTPUT_ROOT = Path("outputs/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-language mean sil_score and ID scores for each model."
    )
    parser.add_argument(
        "--aggregated-root",
        type=Path,
        default=AGGREGATED_ROOT,
        help=f"Input aggregated root. Default: {AGGREGATED_ROOT}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output root. Default: {OUTPUT_ROOT}.",
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


def read_mean_values(csv_path: Path, mean_column: str) -> list[float]:
    values = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip() for field in reader.fieldnames or []]
        if mean_column not in set(reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing column: {mean_column}")

        for row in reader:
            clean_row = {
                (key.strip() if key is not None else key): (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
            }
            value = finite_float(clean_row.get(mean_column))
            if value is not None:
                values.append(value)

    return values


def collect_language_means(
    kind_root: Path,
    *,
    file_suffix: str,
    output_metric: str,
    mean_column: str,
) -> dict[str, list[dict[str, object]]]:
    rows_by_language: dict[str, list[dict[str, object]]] = defaultdict(list)

    for model_dir in sorted(path for path in kind_root.iterdir() if path.is_dir()):
        for csv_path in sorted(model_dir.glob(f"*{file_suffix}")):
            if csv_path.name.startswith("all_languages"):
                continue

            if file_suffix == "_layers.csv":
                language = csv_path.name.removesuffix("_layers.csv")
            else:
                language = csv_path.stem

            values = read_mean_values(csv_path, mean_column)
            if not values:
                continue

            rows_by_language[language].append(
                {
                    "model": model_dir.name,
                    "language": language,
                    output_metric: mean(values),
                    "n_layers": len(values),
                }
            )

    return rows_by_language


def write_language_files(
    output_dir: Path,
    rows_by_language: dict[str, list[dict[str, object]]],
    fieldnames: list[str],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for language, rows in sorted(rows_by_language.items()):
        output_path = output_dir / f"{language}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["model"]))
        saved += 1
        print(f"Saved {output_path}")

    return saved


def main() -> None:
    args = parse_args()

    sil_score_rows = collect_language_means(
        args.aggregated_root / "sil_score",
        file_suffix=".csv",
        output_metric="mean_silhouette_score",
        mean_column="silhouette_score_mean",
    )
    id_rows = collect_language_means(
        args.aggregated_root / "id",
        file_suffix="_layers.csv",
        output_metric="mean_id",
        mean_column="id_mean",
    )

    n_sil = write_language_files(
        args.output_root / "sil_score",
        sil_score_rows,
        ["model", "language", "mean_silhouette_score", "n_layers"],
    )
    n_id = write_language_files(
        args.output_root / "id",
        id_rows,
        ["model", "language", "mean_id", "n_layers"],
    )

    print(f"Done. Saved {n_sil} sil_score CSVs and {n_id} ID CSVs.")


if __name__ == "__main__":
    main()
