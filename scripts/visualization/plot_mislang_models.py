import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = "/home/gangstat/NeMA_result/analysis/mislang_model_lacomsa.csv"
DEFAULT_OUTPUT = "/home/gangstat/NeMA_result/analysis/figures/mislang_models_overview.png"
DEFAULT_NAME = "LaCoMSA"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "mislang_pct",
        "avg_length",
        "top1_pct",
        "top2_pct",
        "top3_pct",
        "others_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def aggregate_by_model_and_lang(df):
    grouped = (
        df.groupby(["lang_prefix", "model_type"], as_index=False)
        .agg(
            mislang_pct=("mislang_pct", "mean"),
            avg_length=("avg_length", "mean"),
            top1_pct=("top1_pct", "mean"),
            top2_pct=("top2_pct", "mean"),
            top3_pct=("top3_pct", "mean"),
            others_pct=("others_pct", "mean"),
        )
        .sort_values(["lang_prefix", "model_type"])
    )
    return grouped


def plot_heatmap(ax, matrix, title, cmap):
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
    ax.set_title(title, pad=14)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            val = matrix.iat[y, x]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7)
    return im


def plot_stacked_mix(ax, grouped):
    by_model = (
        grouped.groupby("model_type", as_index=False)
        .agg(
            top1_pct=("top1_pct", "mean"),
            top2_pct=("top2_pct", "mean"),
            top3_pct=("top3_pct", "mean"),
            others_pct=("others_pct", "mean"),
        )
        .sort_values("model_type")
    )

    x = range(len(by_model))
    b1 = by_model["top1_pct"].values
    b2 = by_model["top2_pct"].values
    b3 = by_model["top3_pct"].values
    b4 = by_model["others_pct"].values

    ax.bar(x, b1, label="Top1 %")
    ax.bar(x, b2, bottom=b1, label="Top2 %")
    ax.bar(x, b3, bottom=b1 + b2, label="Top3 %")
    ax.bar(x, b4, bottom=b1 + b2 + b3, label="Others %")
    ax.set_xticks(list(x))
    ax.set_xticklabels(by_model["model_type"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Share within mislang (%)")
    ax.set_title("Mis-used Language Composition by Model Type", pad=10)
    ax.legend(loc="upper right", fontsize=8)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mislang model analysis CSV efficiently."
    )
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument(
        "--skip-composition",
        action="store_true",
        help="Skip the mis-used language composition subplot.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = load_data(args.input)
    grouped = aggregate_by_model_and_lang(df)

    mislang_matrix = grouped.pivot(
        index="lang_prefix", columns="model_type", values="mislang_pct"
    ).fillna(0.0)
    length_matrix = grouped.pivot(
        index="lang_prefix", columns="model_type", values="avg_length"
    ).fillna(0.0)

    if args.skip_composition:
        fig = plt.figure(figsize=(24, 7))
        grid = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = None
    else:
        fig = plt.figure(figsize=(21, 12))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.35)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, :])

    im1 = plot_heatmap(ax1, mislang_matrix, "Mislang % (Lang x Model Type)", "YlOrRd")
    im2 = plot_heatmap(ax2, length_matrix, "Avg Length (Lang x Model Type)", "YlGnBu")
    if ax3 is not None:
        plot_stacked_mix(ax3, grouped)

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.suptitle(args.name, fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.16)

    plt.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()
