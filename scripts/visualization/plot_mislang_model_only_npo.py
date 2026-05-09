import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANG_ORDER = ["de", "en", "es", "fr", "ru"]


def model_sort_key(model_type: str) -> tuple:
    normalized = str(model_type).strip().lower()

    checkpoint_match = re.match(r"npo_checkpoint-(\d+)$", normalized)
    if checkpoint_match:
        return (0, int(checkpoint_match.group(1)))

    if normalized.startswith("npo_"):
        return (1, normalized)

    if normalized == "npo":
        return (2, 0)

    return (99, normalized)


def _plot_heatmap(ax, values_df: pd.DataFrame, title: str, cmap: str, vmin=None, vmax=None) -> None:
    matrix = values_df.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(values_df.shape[1]))
    ax.set_xticklabels(values_df.columns.tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(values_df.shape[0]))
    ax.set_yticklabels(values_df.index.tolist(), fontsize=9)

    for row in range(values_df.shape[0]):
        for col in range(values_df.shape[1]):
            value = matrix[row, col]
            label = f"{value:.2f}" if np.isfinite(value) else ""
            ax.text(col, row, label, ha="center", va="center", fontsize=6)


def _build_top_share_df(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby("model_type", as_index=False)[["top1_count", "top2_count", "top3_count", "others_count"]]
        .sum()
        .set_index("model_type")
    )

    grouped = grouped.reindex(model_order).fillna(0)
    totals = grouped.sum(axis=1)
    pct_df = grouped.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
    return pct_df


def create_plot(input_csv: str, output_png: str, title: str) -> None:
    df = pd.read_csv(input_csv, skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]

    string_columns = ["lang_prefix", "model_type", "top1_lang", "top2_lang", "top3_lang"]
    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    numeric_columns = [
        "mislang_pct",
        "avg_length",
        "top1_count",
        "top2_count",
        "top3_count",
        "others_count",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    required = {
        "lang_prefix",
        "model_type",
        "mislang_pct",
        "avg_length",
        "top1_count",
        "top2_count",
        "top3_count",
        "others_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    model_order = sorted(df["model_type"].dropna().unique().tolist(), key=model_sort_key)

    mislang_matrix = (
        df.pivot_table(index="lang_prefix", columns="model_type", values="mislang_pct", aggfunc="mean")
        .reindex(index=LANG_ORDER)
        .reindex(columns=model_order)
    )

    avglen_matrix = (
        df.pivot_table(index="lang_prefix", columns="model_type", values="avg_length", aggfunc="mean")
        .reindex(index=LANG_ORDER)
        .reindex(columns=model_order)
    )

    top_share_df = _build_top_share_df(df, model_order)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.28)
    ax_mislang = fig.add_subplot(grid[0, 0])
    ax_avglen = fig.add_subplot(grid[0, 1])
    ax_stack = fig.add_subplot(grid[1, :])

    _plot_heatmap(
        ax=ax_mislang,
        values_df=mislang_matrix,
        title="Mislang % (Lang x Model Type)",
        cmap="YlOrRd",
        vmin=0,
        vmax=float(np.nanmax(mislang_matrix.to_numpy(dtype=float))) if mislang_matrix.size else None,
    )

    _plot_heatmap(
        ax=ax_avglen,
        values_df=avglen_matrix,
        title="Avg Length (Lang x Model Type)",
        cmap="YlGnBu",
        vmin=float(np.nanmin(avglen_matrix.to_numpy(dtype=float))) if avglen_matrix.size else None,
        vmax=float(np.nanmax(avglen_matrix.to_numpy(dtype=float))) if avglen_matrix.size else None,
    )

    x = np.arange(len(model_order))
    bottom = np.zeros(len(model_order), dtype=float)
    stack_colors = {
        "top1_count": "#1f77b4",
        "top2_count": "#ff7f0e",
        "top3_count": "#2ca02c",
        "others_count": "#d62728",
    }
    stack_labels = {
        "top1_count": "Top1 %",
        "top2_count": "Top2 %",
        "top3_count": "Top3 %",
        "others_count": "Others %",
    }

    for column in ["top1_count", "top2_count", "top3_count", "others_count"]:
        values = top_share_df[column].to_numpy(dtype=float)
        ax_stack.bar(x, values, bottom=bottom, color=stack_colors[column], label=stack_labels[column])
        bottom += values

    ax_stack.set_title("Mis-used Language Composition by Model Type", fontsize=11)
    ax_stack.set_ylabel("Share within mislang (%)", fontsize=9)
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
    ax_stack.set_ylim(0, 105)
    ax_stack.grid(axis="y", alpha=0.25)
    ax_stack.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPO-only-NPO mislang overview figure.")
    parser.add_argument(
        "--input",
        default="analysis/mislang_model_mapo_only_npo.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        default="analysis/figures/mislang_models_mapo/mislang_model_mapo_only_npo_overview.png",
        help="Path to output PNG.",
    )
    parser.add_argument(
        "--title",
        default="MAPO with Only NPO",
        help="Figure title.",
    )
    args = parser.parse_args()

    create_plot(args.input, args.output, args.title)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()