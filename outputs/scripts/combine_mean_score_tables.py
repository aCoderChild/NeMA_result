#!/usr/bin/env python3
"""Combine per-language mean ID and sil-score CSVs into one sorted table.

Reads:
  outputs/results/id/{lang}.csv
  outputs/results/sil_score/{lang}.csv

Writes:
  outputs/results/aggregated/mean_scores_by_language.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RESULTS_ROOT = Path("outputs/results")
OUTPUT_PATH = Path("outputs/results/aggregated/mean_scores_by_language.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine language-level mean ID and sil-score CSVs into one file."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=f"Root containing id/ and sil_score/ CSVs. Default: {RESULTS_ROOT}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Combined CSV path. Default: {OUTPUT_PATH}.",
    )
    return parser.parse_args()


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {
        (key.strip() if key is not None else key): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
    }


def read_metric_rows(
    metric_dir: Path,
    *,
    metric_column: str,
    output_column: str,
    n_layers_column: str,
) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}

    for csv_path in sorted(metric_dir.glob("*.csv")):
        language_from_path = csv_path.stem
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                row = clean_row(raw_row)
                model = row.get("model", "")
                language = row.get("language", language_from_path) or language_from_path
                value = row.get(metric_column, "")
                n_layers = row.get("n_layers", "")
                if not model:
                    continue

                key = (language, model)
                rows.setdefault(key, {"language": language, "model": model})
                rows[key][output_column] = value
                rows[key][n_layers_column] = n_layers

    return rows


def main() -> None:
    args = parse_args()
    id_rows = read_metric_rows(
        args.results_root / "id",
        metric_column="mean_id",
        output_column="mean_id",
        n_layers_column="id_n_layers",
    )
    sil_rows = read_metric_rows(
        args.results_root / "sil_score",
        metric_column="mean_silhouette_score",
        output_column="mean_silhouette_score",
        n_layers_column="sil_score_n_layers",
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

    print(f"Saved {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
