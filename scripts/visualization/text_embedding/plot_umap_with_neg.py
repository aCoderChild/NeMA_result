import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap


NEGATIVE_GROUPS = frozenset({"negative_rejected"})


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UMAP on merged main embedding cache + negative (rejected) cache; "
            "negative points plotted as red dots."
        )
    )
    parser.add_argument(
        "--main-cache",
        type=str,
        required=True,
        help="Primary cache (e.g. icr instruction+response).",
    )
    parser.add_argument(
        "--neg-cache",
        type=str,
        required=True,
        help="Negative rejected cache from embedding_only_neg.py.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/figures/icr_umap_main_plus_neg.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--html-path",
        type=str,
        default="",
        help="Optional interactive HTML output path (zoom/pan/hover).",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="",
        help="Comma-separated lang filter for main-cache only (neg-cache always plotted in full).",
    )
    parser.add_argument(
        "--show-lang-labels",
        action="store_true",
        help="Annotate lang on each point (can be busy).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="UMAP random seed.",
    )
    return parser.parse_args()


def load_cache(path: Path):
    with path.open("r", encoding="utf-8") as f:
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


def main_style(item, npo_color_map):
    group = item.get("group")
    if group in NEGATIVE_GROUPS:
        return None
    if group == "base_output_1":
        return "green", "o"
    if group == "baseline_output":
        return "black", "^"
    if group == "npo_output_2":
        return npo_color_map.get(item.get("variant"), "gray"), "x"
    return "gray", "."


def compute_umap_coords(items, n_neighbors, min_dist, random_state):
    items = [it for it in items if "embedding" in it]
    if not items:
        raise ValueError("No items with embeddings after filter.")

    vectors = np.array([it["embedding"] for it in items], dtype=np.float64)
    dim = vectors.shape[1]
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite values in embedding matrix.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(n_neighbors, len(vectors) - 1)),
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(vectors)
    return items, coords, dim


def plot_png(items, coords, dim, plot_path: Path, show_lang_labels: bool):
    npo_color_map = build_npo_color_map(items)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    neg_idx = [i for i, it in enumerate(items) if it.get("group") in NEGATIVE_GROUPS]
    main_idx = [i for i in range(len(items)) if i not in set(neg_idx)]

    for i in main_idx:
        item = items[i]
        style = main_style(item, npo_color_map)
        if style is None:
            continue
        color, marker = style
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            c=[color],
            marker=marker,
            s=55,
            alpha=0.85,
            zorder=2,
        )
        if show_lang_labels:
            ax.annotate(
                item.get("lang", ""),
                (coords[i, 0], coords[i, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )

    for i in neg_idx:
        item = items[i]
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            c="red",
            marker="o",
            s=36,
            alpha=0.9,
            edgecolors="darkred",
            linewidths=0.4,
            zorder=4,
        )
        if show_lang_labels:
            ax.annotate(
                item.get("lang", ""),
                (coords[i, 0], coords[i, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="darkred",
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
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=8,
        )
    )
    labels.append("rejected (train_icr)")

    ax.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"UMAP: main + negative (n={len(items)}, dim={dim})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_html(items, coords, dim, html_path: Path):
    rows = []
    for i, item in enumerate(items):
        group = item.get("group")
        if group in NEGATIVE_GROUPS:
            label = "rejected (train_icr)"
            symbol = "negative"
        elif group == "base_output_1":
            label = "gpt-4-turbo"
            symbol = "gpt4"
        elif group == "baseline_output":
            label = "Llama3-8B-Base-SFT-DPO"
            symbol = "baseline"
        elif group == "npo_output_2":
            label = str(item.get("variant", "npo_unknown"))
            symbol = "npo"
        else:
            label = str(group or "other")
            symbol = "other"

        rows.append(
            {
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "label": label,
                "lang": item.get("lang", ""),
                "group": group,
                "model_type": item.get("model_type", ""),
                "symbol": symbol,
            }
        )

    color_map = {"gpt-4-turbo": "green", "Llama3-8B-Base-SFT-DPO": "black", "rejected (train_icr)": "red"}
    fig = px.scatter(
        rows,
        x="x",
        y="y",
        color="label",
        symbol="symbol",
        color_discrete_map=color_map,
        hover_data=["lang", "group", "model_type"],
        title=f"UMAP: main + negative (n={len(items)}, dim={dim})",
    )
    fig.update_traces(marker={"size": 9, "opacity": 0.9})
    fig.update_layout(xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend_title="Series")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")


def main():
    args = parse_args()
    main_path = Path(args.main_cache)
    neg_path = Path(args.neg_cache)
    plot_path = Path(args.plot_path)
    lang_filter = {x.strip() for x in args.langs.split(",") if x.strip()}

    mc = load_cache(main_path)
    nc = load_cache(neg_path)
    m_model = mc.get("embedding_model")
    n_model = nc.get("embedding_model")
    if m_model != n_model:
        raise ValueError(
            f"embedding_model mismatch: main={m_model!r} neg={n_model!r}"
        )

    main_items = list(mc.get("items", []))
    neg_items = list(nc.get("items", []))
    if not main_items or not neg_items:
        raise ValueError("One of the caches has no items.")

    def dim_of(it):
        return len(it["embedding"])

    all_with_emb = [it for it in main_items + neg_items if "embedding" in it]
    d0 = dim_of(all_with_emb[0])
    for it in all_with_emb:
        if dim_of(it) != d0:
            raise ValueError(
                f"Inconsistent embedding dim: expected {d0}, got {dim_of(it)}"
            )

    if lang_filter:
        main_items = [it for it in main_items if it.get("lang") in lang_filter]

    merged = main_items + neg_items
    merged_items, coords, dim = compute_umap_coords(
        merged,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    plot_png(merged_items, coords, dim, plot_path, show_lang_labels=args.show_lang_labels)
    if args.html_path:
        plot_html(merged_items, coords, dim, Path(args.html_path))
    print(
        f"Wrote {plot_path} (main_points={len(main_items)}, neg_points={len(neg_items)}, total={len(merged)})"
    )


if __name__ == "__main__":
    main()
