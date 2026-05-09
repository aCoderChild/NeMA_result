import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LANG_ORDER = ["de", "en", "es", "fr", "ru"]
COLOR_MAP = {"de": "#1f77b4", "en": "#ff7f0e", "es": "#2ca02c", "fr": "#d62728", "ru": "#9467bd", "others": "#8c564b"}


def model_sort_key(model_type: str) -> tuple:
    """Sort models in order: dpo → npo → npo_checkpoint-* → ppo → sft → w-reinforce"""
    normalized = model_type.strip().lower()
    if normalized == "dpo":
        return (0, 0)
    if normalized == "npo":
        return (1, 0)
    if normalized.startswith("npo_checkpoint-"):
        try:
            checkpoint = int(re.search(r"npo_checkpoint-(\d+)", normalized).group(1))
        except (IndexError, ValueError):
            checkpoint = 999
        return (2, checkpoint)
    if normalized == "ppo":
        return (3, 0)
    if normalized == "sft":
        return (4, 0)
    if normalized == "w-reinforce":
        return (5, 0)
    return (99, normalized)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and normalize CSV data"""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]

    for column in ["model_type", "expected_lang", "method"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    for column in ["de", "fr", "en", "es", "ru", "others"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    required_columns = {"model_type", "expected_lang", "de", "fr", "en", "es", "ru", "others"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def plot_mismatch_by_lang(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create 5 separate visualizations (one per expected_lang), each showing predicted language 
    distribution excluding the expected_lang itself. Each PNG shows all model_types.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lang_columns = ["de", "en", "es", "fr", "ru", "others"]

    for expected_lang in LANG_ORDER:
        # Filter data for this expected language
        data_subset = df[df["expected_lang"] == expected_lang].copy()
        data_subset = data_subset.sort_values("model_type", key=lambda x: x.map(lambda v: model_sort_key(v)))

        if len(data_subset) == 0:
            print(f"No data for {expected_lang}, skipping")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create stacked bar chart
        x_pos = range(len(data_subset))
        model_labels = data_subset["model_type"].values

        # Languages to show (all except expected_lang)
        langs_to_show = [lang for lang in lang_columns if lang != expected_lang]

        # Stack the bars
        bottom = None
        for lang in langs_to_show:
            values = data_subset[lang].values
            color = COLOR_MAP.get(lang, "#cccccc")
            if bottom is None:
                ax.bar(x_pos, values, label=lang, color=color, width=0.7)
                bottom = values
            else:
                ax.bar(x_pos, values, bottom=bottom, label=lang, color=color, width=0.7)
                bottom = bottom + values

        ax.set_title(f"FastText Language Predictions (Ground-Truth: {expected_lang.upper()})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model Type", fontsize=12)
        ax.set_ylabel("Prediction Count", fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"mislang_mismatch_{expected_lang}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot FastText language mismatch predictions")
    parser.add_argument("--input", type=str, default="analysis/mislang_multilangs_model_lacomsa.csv", help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, default="analysis/figures/mislang_multilangs_lacomsa", help="Path to output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows")

    print("Creating visualizations...")
    plot_mismatch_by_lang(df, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
