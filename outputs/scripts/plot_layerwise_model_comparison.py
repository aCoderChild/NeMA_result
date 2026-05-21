#!/usr/bin/env python3
"""Draw layerwise model-comparison line plots per language.

Reads:
  outputs/results/aggregated/sil_score/{model}/{lang}.csv
  outputs/results/aggregated/id/{model}/{lang}_layers.csv
  outputs/results/aggregated/cosine_sim/{model}/{lang}.csv

Writes:
  outputs/figures/sil_score/{lang}.png
  outputs/figures/id/{lang}.png
  outputs/figures/similarity/cosine_sim/{lang}.png
  outputs/figures/similarity/cosine_sim_adjusted_mean/{lang}.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


CACHE_ROOT = Path("outputs/.cache").resolve()
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGGREGATED_ROOT = Path("outputs/results/aggregated")
OUTPUT_ROOT = Path("outputs/figures")
MODEL_ORDER = [
    "base-sft-dpo",
    "icr-dpo-ckpt36",
    "icr-npo-ckpt10",
    "icr-npo-ckpt36",
    "icr-ppo-ckpt36",
    "icr-w-reinforce",
    "icr_dpo_ckpt36",
    "icr_dpo_ckpt36_old_ver",
    "icr_npo_ckpt10",
    "icr_npo_ckpt36",
    "icr_ppo_ckpt36",
    "icr_w_reinforce",
    "icr_w_reinforce_old_ver",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layerwise model comparisons per language."
    )
    parser.add_argument(
        "--aggregated-root",
        type=Path,
        default=AGGREGATED_ROOT,
        help=f"Root containing aggregated score directories. Default: {AGGREGATED_ROOT}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Figure output root. Default: {OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["sil_score", "id", "cosine_sim"],
        default=["sil_score", "id"],
        help="Which plot groups to generate.",
    )
    parser.add_argument(
        "--id-metric",
        choices=["id", "r"],
        default="id",
        help="Metric to plot for ID CSVs. Default: id.",
    )
    parser.add_argument(
        "--cosine-metric",
        choices=["cosine_sim", "cosine_sim_adjusted"],
        default="cosine_sim",
        help="Metric to plot for cosine_sim CSVs. Default: cosine_sim.",
    )
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Disable translucent mean +- std bands.",
    )
    return parser.parse_args()


def finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def available_models(kind_root: Path) -> list[str]:
    models = [path.name for path in kind_root.iterdir() if path.is_dir() and path.name != "all_models"]
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered.extend(sorted(model for model in models if model not in MODEL_ORDER))
    return ordered


def available_languages(kind_root: Path, suffix: str) -> list[str]:
    languages = set()
    for model_dir in kind_root.iterdir():
        if not model_dir.is_dir() or model_dir.name == "all_models":
            continue
        for csv_path in model_dir.glob(f"*{suffix}"):
            if csv_path.name.startswith("all_languages"):
                continue
            if suffix == "_layers.csv":
                languages.add(csv_path.name.removesuffix("_layers.csv"))
            else:
                languages.add(csv_path.stem)
    return sorted(languages)


def read_series(csv_path: Path, mean_column: str, std_column: str) -> tuple[list[int], list[float], list[float]]:
    layers = []
    means = []
    stds = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip() for field in reader.fieldnames or []]
        required = {"layer", mean_column, std_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            return layers, means, stds

        for row in reader:
            clean_row = {
                (key.strip() if key is not None else key): (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
            }
            mean_value = finite_float(clean_row.get(mean_column))
            std_value = finite_float(clean_row.get(std_column))
            layer_value = finite_float(clean_row.get("layer"))
            if mean_value is None or std_value is None or layer_value is None:
                continue
            layers.append(int(layer_value))
            means.append(mean_value)
            stds.append(std_value)

    ordered = sorted(zip(layers, means, stds), key=lambda item: item[0])
    if not ordered:
        return [], [], []
    sorted_layers, sorted_means, sorted_stds = zip(*ordered)
    return list(sorted_layers), list(sorted_means), list(sorted_stds)


def plot_language(
    *,
    kind_root: Path,
    output_path: Path,
    language: str,
    models: list[str],
    file_suffix: str,
    mean_column: str,
    std_column: str,
    title: str,
    ylabel: str,
    show_std_band: bool,
) -> bool:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = 0

    for model in models:
        csv_name = f"{language}{file_suffix}"
        csv_path = kind_root / model / csv_name
        if not csv_path.exists():
            continue

        layers, means, stds = read_series(csv_path, mean_column, std_column)
        if not layers:
            continue

        line = ax.plot(layers, means, marker="o", linewidth=2, markersize=4, label=model)[0]
        if show_std_band:
            lower = [value - std for value, std in zip(means, stds)]
            upper = [value + std for value, std in zip(means, stds)]
            ax.fill_between(layers, lower, upper, color=line.get_color(), alpha=0.12, linewidth=0)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def plot_kind(
    *,
    aggregated_root: Path,
    output_root: Path,
    kind: str,
    id_metric: str,
    cosine_metric: str,
    show_std_band: bool,
) -> int:
    kind_root = aggregated_root / kind
    if not kind_root.exists():
        raise FileNotFoundError(f"Aggregated {kind} directory does not exist: {kind_root}")

    if kind == "sil_score":
        file_suffix = ".csv"
        mean_column = "silhouette_score_mean"
        std_column = "silhouette_score_std"
        ylabel = "Silhouette score"
        output_dir = output_root / kind
    elif kind == "id":
        file_suffix = "_layers.csv"
        mean_column = f"{id_metric}_mean"
        std_column = f"{id_metric}_std"
        ylabel = id_metric
        output_dir = output_root / kind
    else:
        file_suffix = ".csv"
        mean_column = f"{cosine_metric}_mean"
        std_column = f"{cosine_metric}_std"
        ylabel = cosine_metric.replace("_", " ")
        cosine_output_name = (
            "cosine_sim_adjusted_mean"
            if cosine_metric == "cosine_sim_adjusted"
            else "cosine_sim"
        )
        output_dir = output_root / "similarity" / cosine_output_name

    models = available_models(kind_root)
    languages = available_languages(kind_root, file_suffix)
    saved = 0

    for language in languages:
        title = f"{ylabel} by layer - {language}"
        output_path = output_dir / f"{language}.png"
        if plot_language(
            kind_root=kind_root,
            output_path=output_path,
            language=language,
            models=models,
            file_suffix=file_suffix,
            mean_column=mean_column,
            std_column=std_column,
            title=title,
            ylabel=ylabel,
            show_std_band=show_std_band,
        ):
            saved += 1
            print(f"Saved {output_path}")

    return saved


def main() -> None:
    args = parse_args()
    total = 0
    for kind in args.kinds:
        total += plot_kind(
            aggregated_root=args.aggregated_root,
            output_root=args.output_root,
            kind=kind,
            id_metric=args.id_metric,
            cosine_metric=args.cosine_metric,
            show_std_band=not args.no_std_band,
        )

    print(f"Done. Saved {total} figures under {args.output_root}.")


if __name__ == "__main__":
    main()
