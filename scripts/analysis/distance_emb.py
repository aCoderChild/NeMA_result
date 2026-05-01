import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANG_ORDER = ["de", "en", "es", "fr", "ru"]
NPO_VARIANT_ORDER = [
    "npo_150426",
    "npo_checkpoint-1",
    "npo_checkpoint-2",
    "npo_checkpoint-3",
    "npo_checkpoint-4",
    "npo_checkpoint-5",
    "npo_checkpoint-10",
    "npo_checkpoint-20",
    "npo_checkpoint-30",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute distance of model responses (baseline + NPO) to GPT-4 "
            "from embedding cache, export CSV, and render stacked bar chart."
        )
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_id2_embeddings_cache_npo_only_v1.json",
        help="Path to embedding cache JSON.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=2,
        help="Sample id filter. Set -1 to disable filtering.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/distance_to_gpt4_id2.csv",
        help="Output CSV for long-form distances.",
    )
    parser.add_argument(
        "--csv-pivot-out",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/distance_to_gpt4_id2_pivot.csv",
        help="Output CSV for pivoted stacked-bar values.",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/figures/distance_to_gpt4_id2_stacked.png",
        help="Output stacked bar plot path.",
    )
    return parser.parse_args()


def load_cache(cache_path):
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def distance(a, b, metric):
    if metric == "cosine":
        return cosine_distance(a, b)
    return euclidean_distance(a, b)


def model_label(item):
    group = item.get("group")
    if group == "baseline_output":
        return "baseline_sft_dpo"
    if group == "npo_output_2":
        return item.get("variant", "npo_unknown")
    return None


def model_sort_key(name):
    if name == "baseline_sft_dpo":
        return (0, name)
    if name in NPO_VARIANT_ORDER:
        return (1 + NPO_VARIANT_ORDER.index(name), name)
    return (999, name)


def compute_distances(items, metric):
    # GPT-4 anchors by (sample_id, lang)
    gpt_by_key = {}
    for item in items:
        if item.get("group") == "base_output_1":
            key = (item.get("sample_id"), item.get("lang"))
            gpt_by_key[key] = np.array(item["embedding"], dtype=np.float64)

    rows = []
    for item in items:
        mlabel = model_label(item)
        if mlabel is None:
            continue
        key = (item.get("sample_id"), item.get("lang"))
        gpt_vec = gpt_by_key.get(key)
        if gpt_vec is None:
            continue
        model_vec = np.array(item["embedding"], dtype=np.float64)
        d = distance(model_vec, gpt_vec, metric)
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "lang": item.get("lang"),
                "model": mlabel,
                "model_type": item.get("model_type"),
                "group": item.get("group"),
                "variant": item.get("variant"),
                "distance": d,
            }
        )
    return pd.DataFrame(rows)


def build_pivot(df):
    if df.empty:
        return pd.DataFrame()
    pivot = (
        df.groupby(["model", "lang"], as_index=False)["distance"]
        .mean()
        .pivot(index="model", columns="lang", values="distance")
        .fillna(0.0)
    )

    # Stable column order for languages
    cols = [c for c in LANG_ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in LANG_ORDER]
    pivot = pivot[cols]

    # Stable row order for models
    ordered_index = sorted(pivot.index.tolist(), key=model_sort_key)
    pivot = pivot.loc[ordered_index]
    return pivot


def plot_stacked_bar(pivot, metric, plot_path):
    if pivot.empty:
        raise ValueError("Pivot is empty, nothing to plot.")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index), dtype=np.float64)

    cmap = plt.get_cmap("tab10")
    langs = list(pivot.columns)
    for i, lang in enumerate(langs):
        values = pivot[lang].values
        ax.bar(x, values, bottom=bottom, label=lang, color=cmap(i % 10), alpha=0.9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel(f"Distance to GPT-4 ({metric})")
    ax.set_xlabel("Model")
    ax.set_title("Distance to GPT-4 by model (stacked by language)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Language", loc="upper center", ncol=min(5, len(langs)), bbox_to_anchor=(0.5, -0.14))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    csv_out = Path(args.csv_out)
    csv_pivot_out = Path(args.csv_pivot_out)
    plot_out = Path(args.plot_out)

    cache = load_cache(cache_path)
    items = [it for it in cache.get("items", []) if "embedding" in it]
    if args.sample_id != -1:
        items = [it for it in items if str(it.get("sample_id")) == str(args.sample_id)]

    df = compute_distances(items, metric=args.metric)
    if df.empty:
        raise ValueError("No distances computed. Check cache content and sample_id.")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    pivot = build_pivot(df)
    pivot.to_csv(csv_pivot_out)
    plot_stacked_bar(pivot, metric=args.metric, plot_path=plot_out)

    print(f"Saved long CSV : {csv_out}")
    print(f"Saved pivot CSV: {csv_pivot_out}")
    print(f"Saved plot     : {plot_out}")
    print(f"Rows computed  : {len(df)}")


if __name__ == "__main__":
    main()
