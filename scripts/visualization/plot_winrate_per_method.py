#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LANGUAGES = ["es", "ru", "en", "fr", "de"]
MODEL_ORDER = [
    "w-reinforce_0.1",
    "w-reinforce_10",
    "w-reinforce",
    "dpo",
    "ppo",
    "sft",
    "npo",
    "npo_checkpoint_best",
]
MODEL_LABELS = [
    "W-Reinforce\n(0.1)",
    "W-Reinforce\n(10)",
    "W-Reinforce",
    "DPO",
    "PPO",
    "SFT",
    "NPO",
    "NPO-Checkpoint\n(Best)",
]
DEFAULT_INPUT = Path("analysis/lc_winrate_lacomsa_metrics.csv")
DEFAULT_OUTPUT = Path("visualisation/winrate_per_method_lacomsa.png")
SAVE_DPI = 100


def load_winrate(csv_path: Path, method: str) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    checkpoint_bucket: dict[int, dict[str, float]] = defaultdict(dict)
    baselines: dict[str, float] = {}

    def normalize_model_name(model_name: str) -> str | None:
        normalized = model_name.strip().lower()
        if "w-reinforce_0.1" in normalized or "w-reinforce-0.1" in normalized:
            return "w-reinforce_0.1"
        if "w-reinforce_10" in normalized or "w-reinforce-10" in normalized:
            return "w-reinforce_10"
        if "w-reinforce" in normalized:
            return "w-reinforce"
        if normalized.startswith("npo_checkpoint"):
            return "npo_checkpoint_best"
        if normalized in {"dpo", "ppo", "sft", "npo"}:
            return normalized
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for raw_row in reader:
            row = {
                (key.strip() if isinstance(key, str) else key): (
                    value.strip() if isinstance(value, str) else value
                )
                for key, value in raw_row.items()
            }

            row_method = (row.get("method", "") or "").strip().lower()
            if row_method != method:
                continue

            language = (row.get("language", "") or "").strip().lower()
            if language not in LANGUAGES:
                continue

            model_name = (row.get("model", "") or "").strip()
            model_key = normalize_model_name(model_name)
            checkpoint_id = None

            if model_key not in MODEL_ORDER:
                continue

            raw_win_rate = row.get("win_rate", "")
            if raw_win_rate in ("", None):
                continue
            try:
                win_rate = float(raw_win_rate)
            except (TypeError, ValueError):
                continue

            raw_improvement = row.get("winrate_improvement", "")
            if raw_improvement not in ("", None):
                try:
                    baselines.setdefault(language, win_rate - float(raw_improvement))
                except (TypeError, ValueError):
                    pass

            if model_key == "npo_checkpoint_best":
                # The metrics CSV already stores the best checkpoint selection.
                prev = checkpoint_bucket[0].get(language)
                if prev is None or win_rate > prev:
                    checkpoint_bucket[0][language] = win_rate
                continue

            prev = values[model_key].get(language)
            if prev is None or win_rate > prev:
                values[model_key][language] = win_rate

    if checkpoint_bucket:
        best_checkpoint_langs = checkpoint_bucket[0]
        if best_checkpoint_langs:
            values["npo_checkpoint_best"] = best_checkpoint_langs

    values["_baselines"] = baselines
    return values


def plot_winrate(values: dict[str, dict[str, float]], output_path: Path, method: str) -> None:
    has_data = any(
        values.get(model, {}).get(language, 0.0) > 0.0
        for model in MODEL_ORDER
        for language in LANGUAGES
    )
    if not has_data:
        raise ValueError(
            f"No win_rate data found for method '{method}'. Check the input CSV and --method."
        )

    fig, ax = plt.subplots()
    x = list(range(len(MODEL_ORDER)))
    width = 0.13

    bar_colors: dict[str, tuple[float, float, float, float]] = {}
    for i, language in enumerate(LANGUAGES):
        offset = (i - (len(LANGUAGES) - 1) / 2) * width
        heights = [values.get(model, {}).get(language, 0.0) for model in MODEL_ORDER]
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, heights, width=width, label=language.upper())
        bar_colors[language] = bars[0].get_facecolor()

    baselines = values.get("_baselines", {})
    for language in LANGUAGES:
        baseline = baselines.get(language)
        if baseline is None:
            continue
        ax.axhline(
            y=baseline,
            color=bar_colors.get(language),
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label="_nolegend_",
        )

    max_bar = max(
        values.get(model, {}).get(language, 0.0)
        for model in MODEL_ORDER
        for language in LANGUAGES
    )
    max_baseline = max(baselines.values()) if baselines else 0.0
    ax.set_ylim(0, max(max_bar, max_baseline) * 1.12)

    ax.set_title(method.upper())
    ax.set_xlabel("Model")
    ax.set_ylabel("win_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, rotation=0, ha="center", fontsize=7)
    ax.tick_params(axis="x", pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Language",
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.08),
        fontsize=6,
        title_fontsize=7,
        frameon=False,
    )
    fig.suptitle("Win Rate by Model and Language")
    fig.tight_layout(rect=[0, 0.26, 1, 0.92])
    fig.subplots_adjust(bottom=0.34)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot win_rate from the ICR metrics CSV as a compact grouped bar chart."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", default="lacomsa", help="Method name to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_winrate(args.input, args.method)
    plot_winrate(values, args.output, args.method)
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()