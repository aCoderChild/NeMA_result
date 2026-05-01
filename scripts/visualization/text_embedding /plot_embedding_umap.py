import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import umap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot embeddings from cache using UMAP (no API calls)."
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
        default="/home/gangstat/NeMA_result/analysis/figures/icr_id1_embeddings_umap.png",
        help="Output UMAP plot path.",
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
    parser.add_argument(
        "--show-lang-regions",
        action="store_true",
        help="If set, draw faint ellipse borders to indicate language regions.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP.",
    )
    return parser.parse_args()


def load_cache(cache_path):
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        return "green", "o"
    if group == "baseline_output":
        return "black", "^"
    if group == "npo_output_2":
        return npo_color_map.get(item.get("variant"), "gray"), "x"
    return "gray", "."


def _draw_language_regions(ax, coords, items):
    lang_to_idx = {}
    for i, item in enumerate(items):
        lang = item.get("lang", "unknown")
        lang_to_idx.setdefault(lang, []).append(i)

    lang_colors = plt.get_cmap("tab10")
    x_span = max(1e-6, float(np.ptp(coords[:, 0])))
    y_span = max(1e-6, float(np.ptp(coords[:, 1])))
    min_w = 0.08 * x_span
    min_h = 0.08 * y_span

    for j, (lang, idxs) in enumerate(sorted(lang_to_idx.items())):
        pts = coords[idxs]
        center = pts.mean(axis=0)
        edge = lang_colors(j % 10)

        if len(pts) >= 3:
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-9)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            # 2-sigma envelope for a soft region boundary
            width = max(2.0 * 2.0 * np.sqrt(vals[0]), min_w)
            height = max(2.0 * 2.0 * np.sqrt(vals[1]), min_h)
        elif len(pts) == 2:
            diff = pts[1] - pts[0]
            angle = np.degrees(np.arctan2(diff[1], diff[0]))
            width = max(1.5 * np.linalg.norm(diff), min_w)
            height = min_h
        else:
            angle = 0.0
            width = min_w
            height = min_h

        ellipse = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor=edge,
            linewidth=1.8,
            linestyle="--",
            alpha=0.4,
            zorder=1,
        )
        ax.add_patch(ellipse)
        ax.annotate(
            lang,
            (center[0], center[1]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=9,
            color=edge,
            alpha=0.8,
            ha="center",
            va="center",
            zorder=2,
        )


def plot_umap(
    cache,
    plot_path,
    lang_filter,
    show_lang_labels,
    show_lang_regions,
    n_neighbors,
    min_dist,
    random_state,
):
    items = [it for it in cache.get("items", []) if "embedding" in it]
    if lang_filter:
        items = [it for it in items if it.get("lang") in lang_filter]
    if not items:
        raise ValueError("No embeddings found in cache after filtering.")

    vectors = np.array([it["embedding"] for it in items], dtype=np.float64)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(n_neighbors, len(vectors) - 1)),
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(vectors)
    npo_color_map = build_npo_color_map(items)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    if show_lang_regions:
        _draw_language_regions(ax, coords, items)

    for i, item in enumerate(items):
        color, marker = marker_style(item, npo_color_map)
        ax.scatter(coords[i, 0], coords[i, 1], c=[color], marker=marker, s=60, alpha=0.9, zorder=3)
        if show_lang_labels:
            ax.annotate(
                item.get("lang", ""),
                (coords[i, 0], coords[i, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.9,
            )

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

    ax.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("ICR Embeddings (UMAP)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    plot_path = Path(args.plot_path)
    lang_filter = {x.strip() for x in args.langs.split(",") if x.strip()}

    cache = load_cache(cache_path)
    plot_umap(
        cache=cache,
        plot_path=plot_path,
        lang_filter=lang_filter,
        show_lang_labels=args.show_lang_labels,
        show_lang_regions=args.show_lang_regions,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    print(f"Rendered UMAP plot from cache: {plot_path}")


if __name__ == "__main__":
    main()
