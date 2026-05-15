#!/usr/bin/env python3
"""Extract a target layer from semantic_silhouette.csv.

By default this writes outputs/sil_score/middle_layer_sil.csv with layer 16
rows sorted by language first, then model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one layer from a silhouette metric CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/sil_score/semantic_silhouette.csv"),
        help="Input semantic_silhouette.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sil_score/middle_layer_sil.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Layer index to extract.",
    )
    return parser.parse_args()


def extract_layer(input_path: Path, layer: int) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"model", "language", "layer", "mean", "std", "mean_pm_std", "n"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_path} is missing required columns: {missing}")

        rows = [row for row in reader if int(row["layer"]) == layer]

    rows.sort(key=lambda row: (row["language"], row["model"]))
    return rows


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["language", "model", "layer", "mean", "std", "mean_pm_std", "n"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = extract_layer(args.input, args.layer)
    if not rows:
        raise ValueError(f"No rows found for layer {args.layer} in {args.input}")

    write_csv(args.output, rows)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
