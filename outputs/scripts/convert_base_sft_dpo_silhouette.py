#!/usr/bin/env python3
"""Rewrite base-sft-dpo sil-score JSONL files to the standard schema.

For every JSONL row under outputs/results/sil_score/base-sft-dpo/seed_*/*.jsonl,
this keeps only:
  {"layer": ..., "silhouette_score": semantic_silhouette}

The files are overwritten in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_BASE_DIR = Path("outputs/results/sil_score/base-sft-dpo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert base-sft-dpo semantic_silhouette rows to silhouette_score rows."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Folder to rewrite in place. Default: {DEFAULT_BASE_DIR}.",
    )
    return parser.parse_args()


def convert_file(path: Path) -> int:
    converted_rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

            if "layer" not in record:
                raise ValueError(f"Missing layer in {path}:{line_number}")
            if "semantic_silhouette" not in record:
                raise ValueError(f"Missing semantic_silhouette in {path}:{line_number}")

            converted_rows.append(
                {
                    "layer": record["layer"],
                    "silhouette_score": record["semantic_silhouette"],
                }
            )

    with path.open("w", encoding="utf-8") as handle:
        for record in converted_rows:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return len(converted_rows)


def main() -> None:
    args = parse_args()
    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {args.base_dir}")

    jsonl_paths = sorted(args.base_dir.glob("seed_*/*.jsonl"))
    if not jsonl_paths:
        raise ValueError(f"No JSONL files found under {args.base_dir}")

    total_rows = 0
    for path in jsonl_paths:
        row_count = convert_file(path)
        total_rows += row_count
        print(f"Converted {row_count} rows in {path}")

    print(f"Done. Converted {len(jsonl_paths)} files and {total_rows} rows.")


if __name__ == "__main__":
    main()
