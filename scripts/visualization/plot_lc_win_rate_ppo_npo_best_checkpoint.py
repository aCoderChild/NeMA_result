#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt


LANGUAGES = ["de", "en", "es", "fr", "ru"]
TARGET_METHOD = "lacomsa"
TARGET_METHOD_LABEL = "LACOMSA"
PPO_KEY = "ppo"
NPO_BEST_KEY = "npo_best_checkpoint"


def parse_method_and_model(model_name: str) -> tuple[str | None, str | None, int | None]:
    normalized_model = model_name.strip().lower()
    method_match = re.match(r"^(lacomsa)(?:_|-)", normalized_model)
    if not method_match or "_8b_" not in normalized_model:
        return None, None, None

    model_tail = normalized_model.split("_8b_", 1)[1]

    if re.search(r"(?:^|_)ppo(?:_|$)", model_tail):
        return method_match.group(1), PPO_KEY, None

    checkpoint_match = re.search(r"npo_checkpoint-(\d+)", model_tail)
    if checkpoint_match:
        return method_match.group(1), "npo_checkpoint", int(checkpoint_match.group(1))

    return None, None, None


def load_length_controlled_winrate(csv_path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    npo_best_by_lang: dict[str, float] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source", "") or "").strip().lower()
            model_name = (row.get("model", "") or "").strip()

            if source not in LANGUAGES:
                continue

            method, model_type, _checkpoint_id = parse_method_and_model(model_name)
            if method != TARGET_METHOD or model_type is None:
                continue

            raw_value = row.get("length_controlled_winrate", "") or row.get(
                "win_rate", ""
            )
            if raw_value in ("", None):
                continue

            try:
                winrate = float(raw_value)
            except (TypeError, ValueError):
                continue

            if model_type == PPO_KEY:
                prev = values[PPO_KEY].get(source)
                if prev is None or winrate > prev:
                    values[PPO_KEY][source] = winrate
                continue

            # For NPO, keep only the best checkpoint per language.
            prev_npo = npo_best_by_lang.get(source)
            if prev_npo is None or winrate > prev_npo:
                npo_best_by_lang[source] = winrate

    values[NPO_BEST_KEY] = npo_best_by_lang
    return values


def plot_grouped_barchart(values: dict[str, dict[str, float]], output_path: Path) -> None:
    model_order = [PPO_KEY, NPO_BEST_KEY]
    model_labels = {
        PPO_KEY: "PPO",
        NPO_BEST_KEY: "NPO (Best Checkpoint)",
    }
    model_colors = {
        PPO_KEY: "#EC4899",
        NPO_BEST_KEY: "#8B5CF6",
    }

    x = list(range(len(LANGUAGES)))
    width = 0.34
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    max_bar = max(
        values.get(model, {}).get(lang, 0.0)
        for model in model_order
        for lang in LANGUAGES
    )
    y_max = max(5.0, max_bar) * 1.15

    for model_idx, model_name in enumerate(model_order):
        offset = (model_idx - (len(model_order) - 1) / 2) * width
        heights = [values.get(model_name, {}).get(lang, 0.0) for lang in LANGUAGES]
        positions = [xi + offset for xi in x]
        ax.bar(
            positions,
            heights,
            width=width,
            color=model_colors[model_name],
            label=model_labels[model_name],
            alpha=0.92,
        )

    ax.set_title(TARGET_METHOD_LABEL)
    ax.set_xlabel("Language")
    ax.set_xticks(x)
    ax.set_xticklabels([lang.upper() for lang in LANGUAGES])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("length_controlled_winrate")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("LACOMSA Length-Controlled Winrate: PPO vs NPO Best Checkpoint")

    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.subplots_adjust(bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot LACOMSA length_controlled_winrate for PPO and NPO best checkpoint "
            "(best selected per language)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/gangstat/NeMA_result/results/lacomsa_with_w-reinforce/lacomsa.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/gangstat/NeMA_result/visualisation/"
            "length_controlled_winrate_lacomsa_ppo_npo_best_checkpoint.png"
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
