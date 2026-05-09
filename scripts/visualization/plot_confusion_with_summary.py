import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None


DEFAULT_CM_INPUTS = [
    "analysis/cm_icr.csv",
    "analysis/cm_lacomsa.csv",
    "analysis/cm_mapo.csv",
]
DEFAULT_MISLANG_CSV = "analysis/mislang_data.csv"
DEFAULT_OUTPUT = "analysis/figures/confusion_with_summary.png"


def load_confusion_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "expected_lang" not in df.columns:
        raise ValueError(f"Missing 'expected_lang' in: {csv_path}")
    df = df.set_index("expected_lang")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.fillna(0.0)


def dataset_key_from_cm_path(cm_path):
    base = os.path.splitext(os.path.basename(cm_path))[0]  # e.g. cm_icr
    return base.replace("cm_", "")


def load_summary_metrics(mislang_csv):
    df = pd.read_csv(mislang_csv)
    if "filename" not in df.columns:
        raise ValueError(f"Missing 'filename' in: {mislang_csv}")
    df["dataset_key"] = (
        df["filename"].astype(str).str.replace("train_", "", regex=False).str.replace(".jsonl", "", regex=False)
    )
    df["mislang_pct"] = pd.to_numeric(df["mislang_pct"], errors="coerce")
    df["avg_response_length"] = pd.to_numeric(df["avg_response_length"], errors="coerce")
    return df


def draw_heatmap(ax, df, title):
    if sns is not None:
        sns.heatmap(
            df,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar=False,
            ax=ax,
        )
    else:
        im = ax.imshow(df.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                val = df.iat[y, x]
                txt = f"{val:.2f}" if val > 0 else ""
                ax.text(x, y, txt, ha="center", va="center", fontsize=8)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Detected")
    ax.set_ylabel("Expected")
    ax.set_aspect("auto")


def get_summary_values(summary_df, dataset_keys):
    labels = dataset_keys
    mislang_vals = []
    avg_len_vals = []

    for key in dataset_keys:
        row = summary_df[summary_df["dataset_key"] == key]
        if row.empty:
            mislang_vals.append(0.0)
            avg_len_vals.append(0.0)
        else:
            mislang_vals.append(float(row.iloc[-1]["mislang_pct"]))
            avg_len_vals.append(float(row.iloc[-1]["avg_response_length"]))

    return labels, mislang_vals, avg_len_vals


def draw_mislang_panel(ax, labels, mislang_vals):
    x = list(range(len(labels)))
    bars = ax.bar(x, mislang_vals, color="#4C78A8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mislang %")
    ax.set_ylim(0, max(mislang_vals + [1]) * 1.25)
    ax.set_title("Mislang Percentage")

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{mislang_vals[i]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def draw_avg_len_panel(ax, labels, avg_len_vals):
    x = list(range(len(labels)))
    bars = ax.bar(x, avg_len_vals, color="#F58518", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Avg response length")
    ax.set_title("Average Response Length", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.08, y=0.22)

    y_offset = max(avg_len_vals + [1]) * 0.015
    for i, b in enumerate(bars):
        val = avg_len_vals[i]
        ax.text(
            b.get_x() + b.get_width() / 2,
            val + y_offset,
            f"{val:.0f}",
            color="#F58518",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot 3 confusion heatmaps + summary panel (mislang % and avg length)."
    )
    parser.add_argument("--cm-inputs", nargs="+", default=DEFAULT_CM_INPUTS)
    parser.add_argument("--mislang-csv", default=DEFAULT_MISLANG_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for p in args.cm_inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing confusion CSV: {p}")
    if not os.path.exists(args.mislang_csv):
        raise FileNotFoundError(f"Missing summary CSV: {args.mislang_csv}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cm_dfs = [load_confusion_csv(p) for p in args.cm_inputs]
    cm_names = [os.path.splitext(os.path.basename(p))[0] for p in args.cm_inputs]
    dataset_keys = [dataset_key_from_cm_path(p) for p in args.cm_inputs]
    summary_df = load_summary_metrics(args.mislang_csv)

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.45, 0.75, 0.75],
        hspace=0.45,
        wspace=0.35,
    )

    top_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    mid_axis = fig.add_subplot(grid[1, :])
    bottom_axis = fig.add_subplot(grid[2, :])

    for i in range(3):
        draw_heatmap(top_axes[i], cm_dfs[i], cm_names[i])

    labels, mislang_vals, avg_len_vals = get_summary_values(summary_df, dataset_keys)
    draw_mislang_panel(mid_axis, labels, mislang_vals)
    draw_avg_len_panel(bottom_axis, labels, avg_len_vals)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)
    plt.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
