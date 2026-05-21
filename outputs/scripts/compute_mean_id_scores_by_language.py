#!/usr/bin/env python3
"""Compute mean intrinsic dimension scores across layers per language.

Input model folders should contain files like:
  outputs/results/aggregated/id/base-sft-dpo/de_layers.csv

Outputs are written as one CSV per language under:
  outputs/results/aggregated/id/all_models
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


DEFAULT_OUTPUT_DIR = Path("outputs/results/aggregated/id/all_models")
FIELDNAMES = [
    "language",
    "model",
    "n_layers",
    "first_layer",
    "last_layer",
    "mean_id_across_layers",
    "std_id_across_layers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-language mean ID scores across all layers for model folders."
    )
    parser.add_argument(
        "positional_model_dirs",
        nargs="*",
        help="Aggregated ID model folders, e.g. outputs/results/aggregated/id/base-sft-dpo.",
    )
    parser.add_argument(
        "--model_dirs",
        "--model-dirs",
        nargs="+",
        default=None,
        help="Aggregated ID model folders. This is equivalent to passing them positionally.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-language CSV outputs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    args = parser.parse_args()
    raw_model_dirs = args.model_dirs or args.positional_model_dirs
    args.model_dirs = [Path(model_dir.strip()) for model_dir in raw_model_dirs]
    if not args.model_dirs:
        parser.error("provide at least one model folder")
    return args


def clean_column_names(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def summarize_language_file(path: Path, model_name: str) -> dict[str, object] | None:
    language = path.name.removesuffix("_layers.csv")
    id_values = []
    layers = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty")

        fieldnames = {field.strip() for field in reader.fieldnames}
        missing = {"layer", "id_mean"} - fieldnames
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            row = clean_column_names(row)
            if not row.get("id_mean"):
                continue
            id_values.append(float(row["id_mean"]))
            layers.append(int(row["layer"]))

    if not id_values:
        return None

    return {
        "language": language,
        "model": model_name,
        "n_layers": len(id_values),
        "first_layer": min(layers),
        "last_layer": max(layers),
        "mean_id_across_layers": mean(id_values),
        "std_id_across_layers": stdev(id_values) if len(id_values) > 1 else 0.0,
    }


def collect_rows(model_dirs: list[Path]) -> dict[str, list[dict[str, object]]]:
    rows_by_language: dict[str, list[dict[str, object]]] = {}

    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model folder does not exist: {model_dir}")
        if not model_dir.is_dir():
            raise NotADirectoryError(f"Model path is not a folder: {model_dir}")

        language_files = sorted(model_dir.glob("*_layers.csv"))
        language_files = [path for path in language_files if path.name != "all_languages_layers.csv"]
        if not language_files:
            raise ValueError(f"No per-language *_layers.csv files found in {model_dir}")

        for path in language_files:
            row = summarize_language_file(path, model_dir.name)
            if row is not None:
                rows_by_language.setdefault(str(row["language"]), []).append(row)

    return rows_by_language


def write_outputs(output_dir: Path, rows_by_language: dict[str, list[dict[str, object]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for language, rows in sorted(rows_by_language.items()):
        rows = sorted(rows, key=lambda row: str(row["model"]))
        all_rows.extend(rows)
        output_path = output_dir / f"{language}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    all_languages_path = output_dir / "all_languages.csv"
    with all_languages_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (str(row["language"]), str(row["model"]))))


def main() -> None:
    args = parse_args()
    rows_by_language = collect_rows(args.model_dirs)
    write_outputs(args.output_dir, rows_by_language)
    print(f"Wrote {len(rows_by_language)} language files to {args.output_dir}")


if __name__ == "__main__":
    main()
