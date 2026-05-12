#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LANGUAGES = ["en", "es", "ru", "de", "fr"]
CSV_FILES = ["icr.csv", "lacomsa.csv", "mapo.csv"]
BASELINE_LC = {
    "en": 23.80,
    "es": 15.33,
    "ru": 14.77,
    "de": 12.86,
    "fr": 16.39,
}
BASELINE_WINRATE = {
    "en": 22.61,
    "ru": 19.32,
    "fr": 18.88,
    "es": 16.21,
    "de": 15.28,
}
DEFAULT_INPUT_DIR = Path("results/100526")
DEFAULT_OUTPUT = Path("analysis/100526/lc_winrate_w_reinforce_all.png")
SAVE_DPI = 160
ICR_EXCLUDED_VARIANTS = {"w-reinforce", "w-reinforce_clone_150426"}


def normalize_variant(model_name: str) -> str:
    """Keep only the model suffix after 'w-reinforce' for concise labels."""
    lower = model_name.lower()
    idx = lower.find("w-reinforce")
    if idx == -1:
        return model_name
    return model_name[idx:]


def sort_key(variant: str) -> tuple[int, str]:
    if variant == "w-reinforce":
        return (0, variant)
    if "0.1" in variant:
        return (1, variant)
    if "10.0" in variant or "_10" in variant:
        return (2, variant)
    if "clone" in variant:
        return (3, variant)
    return (4, variant)


def load_w_reinforce_metrics(csv_path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Return: {variant: {language: (win_rate, lc_winrate)}}
    Keeps the best lc_winrate row if duplicates exist for variant+language.
    """
    values: dict[str, dict[str, tuple[float, float]]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for raw_row in reader:
            row = {
                (k.strip() if isinstance(k, str) else k): (
                    v.strip() if isinstance(v, str) else v
                )
                for k, v in raw_row.items()
            }
            model = (row.get("model") or "").strip()
            if "w-reinforce" not in model.lower():
                continue

            language = (row.get("source") or "").strip().lower()
            if language not in LANGUAGES:
                continue

            win_raw = row.get("win_rate")
            lc_raw = row.get("length_controlled_winrate")
            try:
                win_rate = float(win_raw) if win_raw not in (None, "") else np.nan
                lc_winrate = float(lc_raw) if lc_raw not in (None, "") else np.nan
            except (TypeError, ValueError):
                continue

            variant = normalize_variant(model)
            if csv_path.stem.lower() == "icr" and variant in ICR_EXCLUDED_VARIANTS:
                continue
            values.setdefault(variant, {})
            prev = values[variant].get(language)
            if prev is None or (not np.isnan(lc_winrate) and lc_winrate > prev[1]):
                values[variant][language] = (win_rate, lc_winrate)

    return values


def plot_file(ax: plt.Axes, csv_path: Path, plot_mode: str) -> None:
    data = load_w_reinforce_metrics(csv_path)
    if not data:
        ax.set_title(csv_path.stem.upper())
        ax.text(0.5, 0.5, "No w-reinforce data", ha="center", va="center")
        ax.set_axis_off()
        return

    variants = sorted(data.keys(), key=sort_key)
    x = np.arange(len(variants))
    lang_colors = {
        "en": "#1f77b4",
        "es": "#ff7f0e",
        "ru": "#2ca02c",
        "de": "#d62728",
        "fr": "#9467bd",
    }

    # 5 languages x 1 metric per variant group.
    total_bars = len(LANGUAGES)
    group_width = 0.88
    bar_w = group_width / total_bars
    start = -group_width / 2 + bar_w / 2

    for lang_idx, lang in enumerate(LANGUAGES):
        color = lang_colors[lang]
        win_vals = []
        lc_vals = []
        for variant in variants:
            pair = data.get(variant, {}).get(lang, (np.nan, np.nan))
            win_vals.append(pair[0])
            lc_vals.append(pair[1])

        base_offset = start + lang_idx * bar_w
        if plot_mode == "winrate":
            y_vals = win_vals
            bar_label = f"{lang.upper()} win"
        else:
            y_vals = lc_vals
            bar_label = f"{lang.upper()} lc"
        ax.bar(
            x + base_offset,
            y_vals,
            width=bar_w,
            color=color,
            alpha=0.85,
            label=bar_label,
        )

    # Baseline reference line per language depending on mode.
    baseline_map = BASELINE_WINRATE if plot_mode == "winrate" else BASELINE_LC
    baseline_suffix = "win" if plot_mode == "winrate" else "LC"
    for lang in LANGUAGES:
        ax.axhline(
            baseline_map[lang],
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            color=lang_colors[lang],
            label=f"{lang.upper()} baseline {baseline_suffix}",
        )

    ax.set_title(csv_path.stem.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot win_rate and length_controlled_winrate for all w-reinforce models "
            "from icr/lacomsa/mapo CSVs, with baseline LC lines."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plot-mode",
        choices=["winrate", "lc+winrate"],
        default="winrate",
        help="Choose metric to plot: winrate, or LC winrate (lc+winrate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output: Path = args.output
    plot_mode: str = args.plot_mode

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

    for ax, filename in zip(axes, CSV_FILES):
        plot_file(ax, input_dir / filename, plot_mode)

    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    if plot_mode == "winrate":
        title = "W-Reinforce Variants: Win Rate (5 Languages)"
    else:
        title = "W-Reinforce Variants: LC Win Rate (5 Languages)"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved chart to: {output}")


if __name__ == "__main__":
    main()
