#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LANGUAGES = ["es", "ru", "en", "de", "fr"]
LAMBDA_MODELS = {
    0.1: "w-reinforce_0.1_1.0",
    1.0: "w-reinforce",
    10.0: "w-reinforce_10.0_1.0",
}
METHOD_FILES = {
    "ICR": Path("results/final/icr.csv"),
    "LaCoMSA": Path("results/final/lacomsa.csv"),
    "LIDR": Path("results/final/lidr.csv"),
}
METHOD_COLORS = {
    "ICR": "#1f77b4",
    "LaCoMSA": "#ff7f0e",
    "LIDR": "#2ca02c",
}


def load_avg_winrates(csv_path: Path) -> dict[float, float]:
    values: dict[float, list[float]] = {}

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [field.strip() for field in reader.fieldnames]

        for raw_row in reader:
            row = {
                key.strip(): value.strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            source = row.get("source", "").lower()
            model = row.get("model", "")
            if source not in LANGUAGES:
                continue

            for lambda_value, expected_model in LAMBDA_MODELS.items():
                if model == expected_model:
                    values.setdefault(lambda_value, []).append(float(row["win_rate"]))

    return {
        lambda_value: sum(winrates) / len(winrates)
        for lambda_value, winrates in values.items()
        if winrates
    }


def plot_lambda_avg_wr(method_values: dict[str, dict[float, float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

    for method, values in method_values.items():
        xs = sorted(values)
        ys = [values[x] for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5,
            linewidth=2,
            color=METHOD_COLORS.get(method),
            label=method,
        )

    ax.set_xscale("log", base=10)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_xlim(0.08, 12)

    ax.set_title("Average Win Rate Across $\\lambda$")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Average Win Rate")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average win rate against lambda for W-REINFORCE estimators."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/figures/lambda_avg_wr_lineplot.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_values = {
        method: load_avg_winrates(csv_path)
        for method, csv_path in METHOD_FILES.items()
        if csv_path.exists()
    }
    plot_lambda_avg_wr(method_values, args.output)
    print(f"Saved chart to: {args.output}")
    for method, values in method_values.items():
        formatted = ", ".join(f"{lambda_value:g}: {avg_wr:.2f}" for lambda_value, avg_wr in sorted(values.items()))
        print(f"{method}: {formatted}")


if __name__ == "__main__":
    main()
