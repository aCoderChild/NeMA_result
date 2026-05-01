#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt


LANGUAGES = ["de", "en", "es", "fr", "ru"]
MODEL_ORDER = ["w-reinforce", "dpo", "ppo", "sft", "npo", "npo_checkpoint_best"]
MODEL_LABELS = ["W-Reinforce (Best)", "DPO", "PPO", "SFT", "NPO", "NPO-Checkpoint (Best)"]
METHOD_ORDER = ["icr", "lacomsa", "mapo"]
METHOD_LABELS = {
    "icr": "ICR",
    "lacomsa": "LACOMSA",
    "mapo": "MAPO",
}
BASELINE_LC = {
    "en": 23.80,
    "es": 15.33,
    "ru": 14.77,
    "de": 12.86,
    "fr": 16.39,
}


def parse_model(model_name: str) -> tuple[str | None, str | None, int | None]:
    normalized_model = model_name.strip().lower()
    method_match = re.match(r"^(icr|lacomsa|mapo)(?:_|-)", normalized_model)
    if not method_match:
        return None, None, None

    if "_8b_" not in normalized_model:
        return None, None, None

    model_tail = normalized_model.split("_8b_", 1)[1]
    if "w-reinforce" in model_tail:
        return method_match.group(1), "w-reinforce", None

    checkpoint_match = re.search(r"npo_checkpoint-(\d+)", model_tail)
    if checkpoint_match:
        return method_match.group(1), "npo_checkpoint", int(checkpoint_match.group(1))

    base_match = re.search(r"(?:^|_)(dpo|ppo|sft|npo)(?:_|$)", model_tail)
    if base_match:
        return method_match.group(1), base_match.group(1), None

    return None, None, None


def load_length_controlled_winrate(
    csv_paths: list[Path],
) -> dict[str, dict[str, dict[str, float]]]:
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    checkpoint_bucket: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for csv_path in csv_paths:
        # utf-8-sig helps when CSV header starts with BOM (e.g. "\ufeffsource").
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = (row.get("source", "") or "").strip().lower()
                model_name = (row.get("model", "") or "").strip()

                if source not in LANGUAGES:
                    continue

                method, model_type, checkpoint_id = parse_model(model_name)
                if method not in METHOD_ORDER or model_type is None:
                    continue

                # Primary metric for this figure is length-controlled win rate.
                raw_value = row.get("length_controlled_winrate", "") or row.get(
                    "win_rate", ""
                )
                if raw_value in ("", None):
                    continue
                try:
                    winrate = float(raw_value)
                except (TypeError, ValueError):
                    continue

                if model_type == "npo_checkpoint":
                    if checkpoint_id is None:
                        continue
                    prev = checkpoint_bucket[method][checkpoint_id].get(source)
                    if prev is None or winrate > prev:
                        checkpoint_bucket[method][checkpoint_id][source] = winrate
                    continue

                if model_type not in MODEL_ORDER:
                    continue

                prev = values[method][model_type].get(source)
                if prev is None or winrate > prev:
                    values[method][model_type][source] = winrate

    for method, by_checkpoint in checkpoint_bucket.items():
        best_checkpoint_langs: dict[str, float] = {}
        best_checkpoint_avg = float("-inf")
        for _checkpoint_id, by_lang in by_checkpoint.items():
            if not by_lang:
                continue
            avg_score = sum(by_lang.values()) / len(by_lang)
            if avg_score > best_checkpoint_avg:
                best_checkpoint_avg = avg_score
                best_checkpoint_langs = by_lang
        if best_checkpoint_langs:
            values[method]["npo_checkpoint_best"] = best_checkpoint_langs

    return values


def plot_grouped_barchart(
    values: dict[str, dict[str, dict[str, float]]], output_path: Path
) -> None:
    x = list(range(len(MODEL_ORDER)))
    width = 0.13

    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(20, 7), sharey=True)
    if len(METHOD_ORDER) == 1:
        axes = [axes]

    max_bar = max(
        values.get(method, {}).get(model, {}).get(lang, 0.0)
        for method in METHOD_ORDER
        for model in MODEL_ORDER
        for lang in LANGUAGES
    )
    y_max = max(max_bar, max(BASELINE_LC.values())) * 1.12

    for idx, method in enumerate(METHOD_ORDER):
        ax = axes[idx]
        bar_colors = {}
        for i, lang in enumerate(LANGUAGES):
            offset = (i - (len(LANGUAGES) - 1) / 2) * width
            heights = [
                values.get(method, {}).get(model, {}).get(lang, 0.0)
                for model in MODEL_ORDER
            ]
            positions = [xi + offset for xi in x]
            bars = ax.bar(positions, heights, width=width, label=lang.upper())
            bar_colors[lang] = bars[0].get_facecolor()

        for lang in LANGUAGES:
            baseline = BASELINE_LC.get(lang)
            if baseline is None:
                continue
            ax.axhline(
                y=baseline,
                color=bar_colors.get(lang),
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label="_nolegend_",
            )

        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(0, y_max)
        if idx == 0:
            ax.set_ylabel("win_rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Language",
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Length Controlled Winrate by Method, Model, and Language")

    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    fig.subplots_adjust(bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped barcharts of length_controlled_winrate for ICR/LACOMSA/MAPO "
            "with 5 models x 5 languages and per-language baseline dashed lines."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[
            Path(
                "/mnt/d/rail-cross-lingual-transfer/_test_local/data/"
                "RAIL CrossLingual Transfer - ICR_150426_Run_Only.csv"
            ),
            Path(
                "/mnt/d/rail-cross-lingual-transfer/_test_local/data/"
                "RAIL CrossLingual Transfer - LACOMSA_Results_Run_01.csv"
            ),
            Path(
                "/mnt/d/rail-cross-lingual-transfer/_test_local/data/"
                "RAIL CrossLingual Transfer - MAPO_Results_Run_01.csv"
            ),
        ],
        help="One or more input CSV paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/rail-cross-lingual-transfer/_test_local/visualization/"
            "winrate_icr_lacomsa_mapo.png"
        ),
        help="Path to output image (PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_length_controlled_winrate(args.input)
    plot_grouped_barchart(values, args.output)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()
