import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None


DEFAULT_INPUTS = [
    # "/home/gangstat/NeMA_result/analysis/cm_icr.csv",
    # "/home/gangstat/NeMA_result/analysis/cm_lacomsa.csv",
    # "/home/gangstat/NeMA_result/analysis/cm_mapo.csv",
    "/home/gangstat/NeMA_result/analysis/confusion_train_mapo.csv",
]
DEFAULT_OUTPUT_DIR = "/home/gangstat/NeMA_result/analysis/figures"


def load_confusion_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "expected_lang" not in df.columns:
        raise ValueError(f"Missing 'expected_lang' column in {csv_path}")

    df = df.set_index("expected_lang")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0.0)
    return df


def plot_single_heatmap(df, title, output_path):
    plt.figure(figsize=(10, 3.2))
    if sns is not None:
        sns.heatmap(
            df,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar=True,
        )
    else:
        ax = plt.gca()
        im = ax.imshow(df.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                val = df.iat[y, x]
                text = f"{val:.2f}" if val > 0 else ""
                ax.text(x, y, text, ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
    plt.title(title)
    plt.xlabel("Detected language")
    plt.ylabel("Expected language")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_heatmap(dfs, names, output_path):
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 3.8), squeeze=False)
    axes = axes[0]

    for i, (df, name) in enumerate(zip(dfs, names)):
        if sns is not None:
            sns.heatmap(
                df,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                vmin=0,
                vmax=100,
                linewidths=0.5,
                cbar=(i == len(dfs) - 1),
                ax=axes[i],
            )
        else:
            im = axes[i].imshow(
                df.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto"
            )
            axes[i].set_xticks(range(len(df.columns)))
            axes[i].set_xticklabels(df.columns, rotation=45, ha="right")
            axes[i].set_yticks(range(len(df.index)))
            axes[i].set_yticklabels(df.index)
            for y in range(df.shape[0]):
                for x in range(df.shape[1]):
                    val = df.iat[y, x]
                    text = f"{val:.2f}" if val > 0 else ""
                    axes[i].text(x, y, text, ha="center", va="center", fontsize=8)
            if i == len(dfs) - 1:
                fig.colorbar(im, ax=axes[i])
        axes[i].set_title(name)
        axes[i].set_xlabel("Detected language")
        axes[i].set_ylabel("Expected language")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot heatmaps from confusion-matrix CSV files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input confusion CSV paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dfs = []
    names = []
    for csv_path in args.inputs:
        if not os.path.exists(csv_path):
            print(f"Skip missing file: {csv_path}")
            continue

        name = os.path.splitext(os.path.basename(csv_path))[0]
        df = load_confusion_csv(csv_path)
        dfs.append(df)
        names.append(name)

        single_out = os.path.join(args.output_dir, f"{name}_heatmap.png")
        plot_single_heatmap(df, f"Confusion Heatmap - {name}", single_out)
        print(f"Saved: {single_out}")

    if not dfs:
        print("No valid confusion CSV files found.")
        return

    combined_out = os.path.join(args.output_dir, "confusion_heatmaps_combined_positive_mapo.png")
    plot_combined_heatmap(dfs, names, combined_out)
    print(f"Saved: {combined_out}")


if __name__ == "__main__":
    main()
