import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot embeddings from cache file only (no API calls)."
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_id1_embeddings_cache.json",
        help="Path to embedding cache JSON file.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/figures/icr_id1_embeddings_dr.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="",
        help="Comma-separated language filter, e.g. 'ru' or 'ru,en'. Empty means all.",
    )
    parser.add_argument(
        "--show-lang-labels",
        action="store_true",
        help="If set, show language text labels on each point.",
    )
    return parser.parse_args()


def load_cache(cache_path):
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reduce_embeddings(vectors):
    return {"PCA": PCA(n_components=2).fit_transform(vectors)}


def build_npo_color_map(items):
    variants = sorted(
        {
            it.get("variant")
            for it in items
            if it.get("group") == "npo_output_2" and isinstance(it.get("variant"), str)
        }
    )
    cmap = plt.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(variants)}


def marker_style(item, npo_color_map):
    group = item.get("group")
    if group == "base_output_1":
        return "green", "o", "base_output_1"
    if group == "baseline_output":
        return "black", "^", "baseline_output"
    if group == "npo_output_2":
        return npo_color_map.get(item.get("variant"), "gray"), "x", item.get("variant", "npo")
    return "gray", ".", "other"


def plot_from_cache(cache, plot_path, lang_filter, show_lang_labels):
    items = [it for it in cache.get("items", []) if "embedding" in it]
    if lang_filter:
        items = [it for it in items if it.get("lang") in lang_filter]
    if not items:
        raise ValueError("No embeddings found in cache.")

    vectors = np.array([it["embedding"] for it in items], dtype=np.float64)
    dr = reduce_embeddings(vectors)
    npo_color_map = build_npo_color_map(items)

    fig, axes = plt.subplots(1, len(dr), figsize=(7 * len(dr), 6), squeeze=False)
    axes = axes.ravel()

    for ax, (name, coords) in zip(axes, dr.items()):
        for i, item in enumerate(items):
            color, marker, _ = marker_style(item, npo_color_map)
            ax.scatter(coords[i, 0], coords[i, 1], c=[color], marker=marker, s=60, alpha=0.9)
            if show_lang_labels:
                ax.annotate(
                    item.get("lang", ""),
                    (coords[i, 0], coords[i, 1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.9,
                )
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=8),
    ]
    labels = ["gpt-4-turbo", "Llama3-8B-Base-SFT-DPO"]

    for variant, color in sorted(npo_color_map.items()):
        handles.append(
            plt.Line2D([0], [0], marker="x", color=color, linestyle="None", markersize=8)
        )
        labels.append(variant)

    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    title_suffix = "with lang labels" if show_lang_labels else "no point labels"
    fig.suptitle(f"ICR id=1 Embeddings ({title_suffix})", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    plot_path = Path(args.plot_path)
    lang_filter = {x.strip() for x in args.langs.split(",") if x.strip()}
    cache = load_cache(cache_path)
    plot_from_cache(cache, plot_path, lang_filter, args.show_lang_labels)
    print(f"Rendered plot from cache: {plot_path}")


if __name__ == "__main__":
    main()
