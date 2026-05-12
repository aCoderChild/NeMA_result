import argparse
import json
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUTS = {
    "ICR": "/home/gangstat/NeMA_result/_pairs/train_icr.jsonl",
    "LaCoMSA": "/home/gangstat/NeMA_result/_pairs/train_lacomsa_relabeled.jsonl",
    "MAPO": "/home/gangstat/NeMA_result/_pairs/train_mapo.jsonl",
}
DEFAULT_OUTPUT_DIR = "/home/gangstat/NeMA_result/analysis"
DEFAULT_FIGURE_NAME = "score_diff_distribution_comparison.png"
SMALL_METHOD_X_MIN = -0.02
SMALL_METHOD_X_MAX = 0.20


def load_score_diff(jsonl_path: str) -> List[float]:
    score_diffs: List[float] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skip malformed JSON at {jsonl_path}:{line_number}")
                continue

            value = row.get("score_diff")
            if value is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(numeric_value):
                score_diffs.append(numeric_value)

    return score_diffs


def summarize(score_diffs: List[float]) -> Dict[str, float]:
    arr = np.asarray(score_diffs, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def adaptive_bins(arr: np.ndarray) -> int:
    n = arr.size
    if n <= 1:
        return 10
    # Finer bins so small score_diff ranges are visible.
    return int(np.clip(np.sqrt(n) * 5, 120, 700))


def get_plot_range(method: str, arr: np.ndarray) -> tuple[float, float, str]:
    if method == "ICR":
        # ICR has much wider spread; give it its own axis.
        x_min = float(np.percentile(arr, 0.5))
        x_max = float(np.percentile(arr, 99.5))
        label = "ICR-only axis (p0.5-p99.5)"
    else:
        # MAPO + LaCoMSA share the same narrow axis for direct comparison.
        x_min = SMALL_METHOD_X_MIN
        x_max = SMALL_METHOD_X_MAX
        label = f"shared small-axis ({SMALL_METHOD_X_MIN:.2f}-{SMALL_METHOD_X_MAX:.2f})"

    if x_max <= x_min:
        x_min = float(np.min(arr))
        x_max = float(np.max(arr))
        label = "fallback min-max axis"
    return x_min, x_max, label


def plot_score_diff(data: Dict[str, List[float]], output_path: str) -> None:
    methods = [m for m, values in data.items() if values]
    if not methods:
        raise ValueError("No valid score_diff found in all input files.")

    fig, axes = plt.subplots(
        nrows=len(methods) + 1,
        ncols=1,
        figsize=(13, 4.2 + 2.2 * len(methods)),
        gridspec_kw={"height_ratios": [1.8] * len(methods) + [2.0]},
    )
    dist_axes = axes[:-1]
    compare_ax = axes[-1]

    # Distribution panel per method (separate axis so tiny ranges are visible)
    for ax, method in zip(dist_axes, methods):
        arr = np.asarray(data[method], dtype=float)

        x_min, x_max, axis_label = get_plot_range(method, arr)
        visible = arr[(arr >= x_min) & (arr <= x_max)]
        common_bins = 500
        ax.hist(
            visible,
            bins=np.linspace(x_min, x_max, common_bins + 1),
            alpha=0.75,
            density=True,
            color="#4c78a8",
            edgecolor="none",
        )

        mean_value = float(np.mean(arr))
        ax.axvline(mean_value, color="#d62728", linestyle="--", linewidth=1.8)
        ax.text(
            mean_value,
            ax.get_ylim()[1] * 0.92,
            f"mean={mean_value:.4f}",
            color="#d62728",
            fontsize=9,
            ha="left",
            va="top",
        )

        ax.set_xlim(x_min, x_max)
        coverage = 100.0 * (visible.size / arr.size) if arr.size else 0.0
        ax.text(
            0.99,
            0.98,
            f"{axis_label}, bins=500, in-range={coverage:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#444444",
        )

        ax.set_title(f"{method} score_diff distribution (bins=500, n={arr.size})")
        ax.set_xlabel("score_diff")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25, linestyle="--")

    # Bottom panel: boxplot + mean marker for clear average comparison
    box_data = [data[m] for m in methods]
    compare_ax.boxplot(
        box_data,
        tick_labels=methods,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#cfe8ff", "alpha": 0.75},
        medianprops={"color": "#1d4f91", "linewidth": 2},
    )

    means = [np.mean(np.asarray(data[m], dtype=float)) for m in methods]
    x = np.arange(1, len(methods) + 1)
    compare_ax.plot(
        x, means, "o-", color="#d62728", linewidth=2, markersize=7, label="Mean"
    )
    for idx, mean_value in enumerate(means, start=1):
        compare_ax.text(
            idx,
            mean_value,
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#d62728",
        )

    compare_ax.set_title("Average and Spread of Score Diff (symlog scale)")
    compare_ax.set_ylabel("score_diff")
    compare_ax.set_yscale("symlog", linthresh=0.05)
    compare_ax.grid(alpha=0.25, linestyle="--")
    compare_ax.legend()

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06, hspace=0.42)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot score_diff distributions and average differences for ICR, "
            "LaCoMSA, and MAPO from JSONL pair files."
        )
    )
    parser.add_argument("--icr", default=DEFAULT_INPUTS["ICR"], help="Path to ICR JSONL")
    parser.add_argument(
        "--lacomsa",
        default=DEFAULT_INPUTS["LaCoMSA"],
        help="Path to LaCoMSA JSONL",
    )
    parser.add_argument("--mapo", default=DEFAULT_INPUTS["MAPO"], help="Path to MAPO JSONL")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figure",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_FIGURE_NAME,
        help="Output figure filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    method_paths = {
        "ICR": args.icr,
        "LaCoMSA": args.lacomsa,
        "MAPO": args.mapo,
    }

    all_data: Dict[str, List[float]] = {}
    for method, path in method_paths.items():
        if not os.path.exists(path):
            print(f"Skip missing file for {method}: {path}")
            continue
        values = load_score_diff(path)
        all_data[method] = values
        if values:
            stats = summarize(values)
            print(
                f"{method}: count={stats['count']}, mean={stats['mean']:.4f}, "
                f"median={stats['median']:.4f}, std={stats['std']:.4f}, "
                f"q1={stats['q1']:.4f}, q3={stats['q3']:.4f}"
            )
        else:
            print(f"{method}: no valid score_diff values.")

    output_path = os.path.join(args.output_dir, args.output_name)
    plot_score_diff(all_data, output_path)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
