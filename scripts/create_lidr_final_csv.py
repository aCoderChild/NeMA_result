#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LANG_ORDER = ["de", "en", "es", "fr", "ru"]
MODEL_PREFIX = "lidr_enesru_llama3_8b_"
MODEL_ORDER = [
    "w-reinforce_0.1_1.0",
    "dpo",
    "ppo",
    "sft",
    "npo",
]
MODEL_SET = set(MODEL_ORDER)
OUTPUT_COLUMNS = [
    "source",
    "model",
    "win_rate",
    "length_controlled_winrate",
    "avg_length",
]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for key, value in row.items()
        if key is not None and value is not None
    }


def compact_model_name(model_name: str) -> str | None:
    if not model_name.startswith(MODEL_PREFIX):
        return None
    return model_name.removeprefix(MODEL_PREFIX)


def model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model_name), model_name
    except ValueError:
        return len(MODEL_ORDER), model_name


def load_lidr_rows(input_dir: Path) -> list[dict[str, str]]:
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}

    for leaderboard_path in sorted(input_dir.glob("*/leaderboard.csv")):
        source = leaderboard_path.parent.name.split("-", 1)[0]
        if source not in LANG_ORDER:
            continue

        with leaderboard_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            if reader.fieldnames:
                reader.fieldnames = [field.strip() for field in reader.fieldnames]

            for raw_row in reader:
                row = normalize_row(raw_row)
                full_model_name = row.get("model", "") or row.get("", "")
                model = compact_model_name(full_model_name)
                if model is None or model not in MODEL_SET:
                    continue

                output_row = {
                    "source": source,
                    "model": model,
                    "win_rate": row.get("win_rate", ""),
                    "length_controlled_winrate": row.get(
                        "length_controlled_winrate", ""
                    ),
                    "avg_length": row.get("avg_length", ""),
                }
                key = (source, model)

                existing = rows_by_key.get(key)
                if existing is not None and existing != output_row:
                    raise ValueError(
                        f"Conflicting rows for {source}/{model} in {leaderboard_path}"
                    )
                rows_by_key[key] = output_row

    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            LANG_ORDER.index(row["source"]),
            model_sort_key(row["model"]),
        ),
    )


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create results/final/lidr.csv from responses/lidr leaderboards."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("responses/lidr"),
        help="Directory containing LIDR response result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/lidr.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_lidr_rows(args.input_dir)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
