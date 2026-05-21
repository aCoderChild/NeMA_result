#!/usr/bin/env python3
"""Aggregate layerwise silhouette, ID, and cosine similarity scores across seeds.

Inputs:
  outputs/results/sil_score/{model_name}/seed_*/*.jsonl
  outputs/results/id/{model_name}/seed_*/*_layers.csv
  outputs/results/cosine_sim/{model_name}/seed_*/*.csv

By default, outputs are written under:
  outputs/results/aggregated/sil_score/{model_name}
  outputs/results/aggregated/id/{model_name}
  outputs/results/aggregated/cosine_sim/{model_name}

Each language gets one CSV with mean, std, "mean +- std", and n_seeds per layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import DefaultDict


SIL_SCORE_FIELD = "silhouette_score"
ID_FIELDS = ("id", "r")
COSINE_FIELDS = ("cosine_sim", "cosine_sim_adjusted")
SIL_SCORE_ROOT = Path("outputs/results/sil_score")
ID_ROOT = Path("outputs/results/id")
COSINE_ROOT = Path("outputs/results/cosine_sim")
OUTPUT_ROOT = Path("outputs/results/aggregated")

DEFAULT_MODELS = [
    "base-sft-dpo",
    "icr_dpo_ckpt36",
    "icr_npo_ckpt10",
    "icr_npo_ckpt36",
    "icr_ppo_ckpt36",
    "icr_w_reinforce",
]
MODEL_ID_DIRS = {
    "base-sft-dpo": "llama_base_sft_dpo",
    "icr_dpo_ckpt36": "icr_dpo_chkpt36",
    "icr_npo_ckpt10": "icr_npo_chkpt10",
    "icr_npo_ckpt36": "icr_npo_chkpt36",
    "icr_ppo_ckpt36": "icr_ppo_chkpt36",
    "icr_w_reinforce": "icr_w_reinforce",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute layerwise mean +- std for sil-score and ID scores across seeds."
    )
    parser.add_argument(
        "--sil-score-root",
        type=Path,
        default=SIL_SCORE_ROOT,
        help=f"Root directory containing sil-score model folders. Default: {SIL_SCORE_ROOT}.",
    )
    parser.add_argument(
        "--id-root",
        type=Path,
        default=ID_ROOT,
        help=f"Root directory containing ID model folders. Default: {ID_ROOT}.",
    )
    parser.add_argument(
        "--cosine-root",
        type=Path,
        default=COSINE_ROOT,
        help=f"Root directory containing cosine-sim model folders. Default: {COSINE_ROOT}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output root that mirrors outputs/results. Default: {OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Sil-score model folder names to process. Defaults to the current hard-coded model list.",
    )
    parser.add_argument(
        "--sil-score-field",
        default=SIL_SCORE_FIELD,
        help=f"Silhouette JSONL score field to aggregate. Default: {SIL_SCORE_FIELD}.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "sil-score", "id", "cosine-sim"),
        default="all",
        help="Aggregate only one score type, or sil-score and ID by default.",
    )
    return parser.parse_args()


def id_model_name_for(sil_model_name: str) -> str:
    return MODEL_ID_DIRS.get(sil_model_name, sil_model_name.replace("_ckpt", "_chkpt"))


def clean_language_name(path: Path, suffix: str = "") -> str:
    """Return language from names like 'de.jsonl' or 'de_layers.csv'."""
    stem = path.stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def seed_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seed_"):
            return part.removeprefix("seed_")
    raise ValueError(f"Could not find seed_* directory in path: {path}")


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def layer_int(value: object, source: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid layer value in {source}: {value!r}") from exc


def collect_sil_scores(
    model_dir: Path, score_field: str
) -> DefaultDict[tuple[str, int], list[float]]:
    values: DefaultDict[tuple[str, int], list[float]] = defaultdict(list)
    seen_seed_layers: set[tuple[str, str, int]] = set()

    for jsonl_path in sorted(model_dir.glob("seed_*/*.jsonl")):
        language = clean_language_name(jsonl_path)
        seed = seed_from_path(jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {jsonl_path}:{line_number}: {exc}") from exc

                layer = layer_int(record.get("layer"), f"{jsonl_path}:{line_number}")
                value = finite_float(record.get(score_field))
                if value is None:
                    continue
                seed_layer = (language, seed, layer)
                if seed_layer in seen_seed_layers:
                    raise ValueError(
                        f"Duplicate {score_field} value for language={language}, seed={seed}, "
                        f"layer={layer} in {jsonl_path}:{line_number}"
                    )
                seen_seed_layers.add(seed_layer)
                values[(language, layer)].append(value)

    return values


def collect_id_scores(
    model_dir: Path,
) -> dict[str, DefaultDict[tuple[str, int], list[float]]]:
    values: dict[str, DefaultDict[tuple[str, int], list[float]]] = {
        metric: defaultdict(list) for metric in ID_FIELDS
    }
    seen_seed_layers: dict[str, set[tuple[str, str, int]]] = {metric: set() for metric in ID_FIELDS}

    for csv_path in sorted(model_dir.glob("seed_*/*_layers.csv")):
        language = clean_language_name(csv_path, suffix="_layers")
        seed = seed_from_path(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"layer", *ID_FIELDS}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

            for row_number, row in enumerate(reader, start=2):
                layer = layer_int(row.get("layer"), f"{csv_path}:{row_number}")
                for metric in ID_FIELDS:
                    value = finite_float(row.get(metric))
                    if value is not None:
                        seed_layer = (language, seed, layer)
                        if seed_layer in seen_seed_layers[metric]:
                            raise ValueError(
                                f"Duplicate {metric} value for language={language}, seed={seed}, "
                                f"layer={layer} in {csv_path}:{row_number}"
                            )
                        seen_seed_layers[metric].add(seed_layer)
                        values[metric][(language, layer)].append(value)

    return values


def collect_csv_layer_scores(
    model_dir: Path,
    metric_fields: tuple[str, ...],
    filename_glob: str = "*.csv",
    language_suffix: str = "",
) -> dict[str, DefaultDict[tuple[str, int], list[float]]]:
    values: dict[str, DefaultDict[tuple[str, int], list[float]]] = {
        metric: defaultdict(list) for metric in metric_fields
    }
    seen_seed_layers: dict[str, set[tuple[str, str, int]]] = {
        metric: set() for metric in metric_fields
    }

    for csv_path in sorted(model_dir.glob(f"seed_*/{filename_glob}")):
        language = clean_language_name(csv_path, suffix=language_suffix)
        seed = seed_from_path(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"layer", *metric_fields}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

            for row_number, row in enumerate(reader, start=2):
                layer = layer_int(row.get("layer"), f"{csv_path}:{row_number}")
                for metric in metric_fields:
                    value = finite_float(row.get(metric))
                    if value is None:
                        continue
                    seed_layer = (language, seed, layer)
                    if seed_layer in seen_seed_layers[metric]:
                        raise ValueError(
                            f"Duplicate {metric} value for language={language}, seed={seed}, "
                            f"layer={layer} in {csv_path}:{row_number}"
                        )
                    seen_seed_layers[metric].add(seed_layer)
                    values[metric][(language, layer)].append(value)

    return values


def stats(values: list[float]) -> tuple[float, float]:
    value_mean = mean(values)
    value_std = stdev(values) if len(values) > 1 else 0.0
    return value_mean, value_std


def write_sil_outputs(
    output_dir: Path, values: DefaultDict[tuple[str, int], list[float]], score_field: str
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_language: DefaultDict[str, list[dict[str, object]]] = defaultdict(list)

    for (language, layer), layer_values in sorted(values.items()):
        value_mean, value_std = stats(layer_values)
        by_language[language].append(
            {
                "language": language,
                "layer": layer,
                f"{score_field}_mean": value_mean,
                f"{score_field}_std": value_std,
                f"{score_field}_mean_pm_std": f"{value_mean:.6f} +- {value_std:.6f}",
                f"{score_field}_n_seeds": len(layer_values),
            }
        )

    fieldnames = [
        "language",
        "layer",
        f"{score_field}_mean",
        f"{score_field}_std",
        f"{score_field}_mean_pm_std",
        f"{score_field}_n_seeds",
    ]
    write_language_csvs(output_dir, by_language, fieldnames, suffix=".csv")
    return sum(len(rows) for rows in by_language.values())


def write_id_outputs(
    output_dir: Path, metric_values: dict[str, DefaultDict[tuple[str, int], list[float]]]
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = sorted({language for values in metric_values.values() for language, _layer in values})
    by_language: DefaultDict[str, list[dict[str, object]]] = defaultdict(list)

    for language in languages:
        layers = sorted({layer for values in metric_values.values() for lang, layer in values if lang == language})
        for layer in layers:
            row: dict[str, object] = {"language": language, "layer": layer}
            for metric in ID_FIELDS:
                values = metric_values[metric].get((language, layer), [])
                if not values:
                    continue
                value_mean, value_std = stats(values)
                row[f"{metric}_mean"] = value_mean
                row[f"{metric}_std"] = value_std
                row[f"{metric}_mean_pm_std"] = f"{value_mean:.6f} +- {value_std:.6f}"
                row[f"{metric}_n_seeds"] = len(values)
            by_language[language].append(row)

    fieldnames = [
        "language",
        "layer",
        "id_mean",
        "id_std",
        "id_mean_pm_std",
        "id_n_seeds",
        "r_mean",
        "r_std",
        "r_mean_pm_std",
        "r_n_seeds",
    ]
    write_language_csvs(output_dir, by_language, fieldnames, suffix="_layers.csv")
    return sum(len(rows) for rows in by_language.values())


def write_multi_metric_outputs(
    output_dir: Path,
    metric_values: dict[str, DefaultDict[tuple[str, int], list[float]]],
    metrics: tuple[str, ...],
    suffix: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = sorted({language for values in metric_values.values() for language, _layer in values})
    by_language: DefaultDict[str, list[dict[str, object]]] = defaultdict(list)

    for language in languages:
        layers = sorted({layer for values in metric_values.values() for lang, layer in values if lang == language})
        for layer in layers:
            row: dict[str, object] = {"language": language, "layer": layer}
            for metric in metrics:
                values = metric_values[metric].get((language, layer), [])
                if not values:
                    continue
                value_mean, value_std = stats(values)
                row[f"{metric}_mean"] = value_mean
                row[f"{metric}_std"] = value_std
                row[f"{metric}_mean_pm_std"] = f"{value_mean:.6f} +- {value_std:.6f}"
                row[f"{metric}_n_seeds"] = len(values)
            by_language[language].append(row)

    fieldnames = ["language", "layer"]
    for metric in metrics:
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_mean_pm_std",
                f"{metric}_n_seeds",
            ]
        )
    write_language_csvs(output_dir, by_language, fieldnames, suffix=suffix)
    return sum(len(rows) for rows in by_language.values())


def write_language_csvs(
    output_dir: Path,
    by_language: dict[str, list[dict[str, object]]],
    fieldnames: list[str],
    suffix: str,
) -> None:
    all_rows = []
    for language, rows in sorted(by_language.items()):
        all_rows.extend(rows)
        output_path = output_dir / f"{language}{suffix}"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    combined_path = output_dir / f"all_languages{suffix}"
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (row["language"], row["layer"])))


def process_model(
    model_name: str,
    sil_score_root: Path,
    id_root: Path,
    cosine_root: Path,
    output_root: Path,
    score_field: str,
    only: str,
) -> tuple[int, int, int, list[str]]:
    sil_score_dir = sil_score_root / model_name
    mapped_id_dir = id_root / id_model_name_for(model_name)
    direct_id_dir = id_root / model_name
    id_dir = mapped_id_dir if mapped_id_dir.exists() or not direct_id_dir.exists() else direct_id_dir
    cosine_dir = cosine_root / model_name
    warnings = []

    n_sil_rows = 0
    if only in {"id", "cosine-sim"}:
        pass
    elif not sil_score_dir.exists():
        warnings.append(f"Sil-score directory does not exist: {sil_score_dir}")
    else:
        sil_values = collect_sil_scores(sil_score_dir, score_field)
        if sil_values:
            sil_output_dir = output_root / "sil_score" / model_name
            n_sil_rows = write_sil_outputs(sil_output_dir, sil_values, score_field)
        else:
            warnings.append(f"No sil-score values found under {sil_score_dir}")

    n_id_rows = 0
    if only in {"sil-score", "cosine-sim"}:
        pass
    elif not id_dir.exists():
        warnings.append(
            f"ID directory does not exist for {model_name}: {id_dir}. "
            "Add it to MODEL_ID_DIRS if the folder uses a different name."
        )
    else:
        id_values = collect_id_scores(id_dir)
        if any(id_values.values()):
            id_output_dir = output_root / "id" / model_name
            n_id_rows = write_id_outputs(id_output_dir, id_values)
        else:
            warnings.append(f"No ID values found under {id_dir}")

    n_cosine_rows = 0
    if only != "cosine-sim":
        pass
    elif not cosine_dir.exists():
        warnings.append(f"Cosine-sim directory does not exist: {cosine_dir}")
    else:
        cosine_values = collect_csv_layer_scores(cosine_dir, COSINE_FIELDS)
        if any(cosine_values.values()):
            cosine_output_dir = output_root / "cosine_sim" / model_name
            n_cosine_rows = write_multi_metric_outputs(
                cosine_output_dir, cosine_values, COSINE_FIELDS, suffix=".csv"
            )
        else:
            warnings.append(f"No cosine-sim values found under {cosine_dir}")

    if n_sil_rows == 0 and n_id_rows == 0 and n_cosine_rows == 0:
        raise ValueError(f"No usable layerwise data found for {model_name}")

    return n_sil_rows, n_id_rows, n_cosine_rows, warnings


def main() -> None:
    args = parse_args()
    if args.only in {"all", "sil-score"} and not args.sil_score_root.exists():
        raise FileNotFoundError(f"Sil-score root does not exist: {args.sil_score_root}")
    if args.only in {"all", "id"} and not args.id_root.exists():
        raise FileNotFoundError(f"ID root does not exist: {args.id_root}")
    if args.only == "cosine-sim" and not args.cosine_root.exists():
        raise FileNotFoundError(f"Cosine-sim root does not exist: {args.cosine_root}")

    failures = []
    warnings = []
    for model_name in args.models:
        print(f"Processing {model_name}")
        try:
            n_sil_rows, n_id_rows, n_cosine_rows, model_warnings = process_model(
                model_name=model_name,
                sil_score_root=args.sil_score_root,
                id_root=args.id_root,
                cosine_root=args.cosine_root,
                output_root=args.output_root,
                score_field=args.sil_score_field,
                only=args.only,
            )
        except (FileNotFoundError, ValueError) as exc:
            failures.append((model_name, exc))
            print(f"Skipped {model_name}: {exc}")
            continue

        print(
            f"Saved {n_sil_rows} sil-score rows, {n_id_rows} ID rows, "
            f"and {n_cosine_rows} cosine-sim rows for {model_name}"
        )
        warnings.extend((model_name, warning) for warning in model_warnings)

    if warnings:
        print("\nWarnings:")
        for model_name, warning in warnings:
            print(f"- {model_name}: {warning}")

    if failures:
        print("\nModels skipped:")
        for model_name, exc in failures:
            print(f"- {model_name}: {exc}")
        raise SystemExit(1)

    print(f"\nDone. Aggregated {len(args.models)} models into {args.output_root}")


if __name__ == "__main__":
    main()
