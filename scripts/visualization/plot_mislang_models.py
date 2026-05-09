import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = "analysis/lang_correct_incorrect_distribution_model_lacomsa.csv"
DEFAULT_OUTPUT_DIR = "analysis/figures/mislang_models_lacomsa"
LANG_ORDER = ["de", "en", "es", "fr", "ru"]


def model_sort_key(model_type: str) -> tuple:
    normalized = model_type.strip().lower()
    if normalized == "dpo_250426":
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
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]

    for column in ["model_type", "lang"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    for column in ["total_samples", "correct_count", "incorrect_count", "no_responses"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    required_columns = {"model_type", "lang", "total_samples", "correct_count", "incorrect_count", "no_responses"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    return df


def plot_language_breakdown(df: pd.DataFrame, lang: str, output_path: Path) -> None:
    lang_df = df[df["lang"] == lang].copy()
    if lang_df.empty:
        print(f"Skipping {lang}: no rows found.")
        return

    lang_df["model_sort"] = lang_df["model_type"].map(model_sort_key)
    lang_df = lang_df.sort_values("model_sort", kind="stable")

    x = range(len(lang_df))
    correct = lang_df["correct_count"].to_numpy()
    incorrect = lang_df["incorrect_count"].to_numpy()
    no_responses = lang_df["no_responses"].to_numpy()

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x, correct, label="Correct", color="#2b8cbe")
    ax.bar(x, incorrect, bottom=correct, label="Incorrect", color="#fdae61")
    ax.bar(x, no_responses, bottom=correct + incorrect, label="No responses", color="#d73027")

    ax.set_xticks(list(x))
    ax.set_xticklabels(lang_df["model_type"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Mislang breakdown for {lang.upper()}")
    ax.set_ylim(0, max(lang_df["total_samples"].max(), (correct + incorrect + no_responses).max()) * 1.08)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one mislang visualization PNG per language."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the five PNG files will be written.",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        default=LANG_ORDER,
        help="Languages to plot, in order.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)

    for lang in args.langs:
        output_path = output_dir / f"mislang_models_{lang}.png"
        plot_language_breakdown(df, lang, output_path)


if __name__ == "__main__":
    main()
