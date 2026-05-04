import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.patches import Ellipse, Polygon
from openai import OpenAI
from scipy.spatial import ConvexHull, QhullError
from sklearn.manifold import TSNE


CACHE_SCHEMA = "same_instruction_es_ru_first_v1"


def format_embedding_text(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction} ### Response: {response}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "First es + first ru sample per JSONL: embed instruction+chosen / instruction+rejected, "
            "cache, UMAP + t-SNE side-by-side plot (chosen=green, rejected=red); "
            "optional convex hull around chosen points."
        )
    )
    parser.add_argument(
        "--lacomsa-jsonl",
        type=str,
        default="/home/gangstat/NeMA_result/_pairs/train_lacomsa_relabeled.jsonl",
    )
    parser.add_argument(
        "--icr-jsonl",
        type=str,
        default="/home/gangstat/NeMA_result/_pairs/train_icr.jsonl",
    )
    parser.add_argument(
        "--mapo-jsonl",
        type=str,
        default="/home/gangstat/NeMA_result/_pairs/train_mapo.jsonl",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="es,ru",
        help="Languages to grab (first occurrence each per file). Comma-separated.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/plot_same_instruction_es_ru_cache.json",
        help="Read/write embedding cache.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Never call OpenAI; all embedding_text must hit cache.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached embeddings and re-call API for every point.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/figures/plot_embedding_same_instruction_umap_tsne.png",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (clamped to dataset size).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=42,
        help="UMAP random seed.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=5.0,
        help="t-SNE perplexity (clamped below n_samples).",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="t-SNE random seed.",
    )
    parser.add_argument(
        "--show-lang-label",
        action="store_true",
        help="Annotate lang near each point.",
    )
    parser.add_argument(
        "--show-file-label",
        action="store_true",
        help="Annotate source filename near each point.",
    )
    parser.add_argument(
        "--show-chosen-hull",
        action="store_true",
        help="Draw a convex hull (+ light fill) around all chosen / positive embeddings per panel.",
    )
    parser.add_argument(
        "--chosen-hull-alpha",
        type=float,
        default=0.22,
        help="Face alpha for the chosen-region polygon.",
    )
    parser.add_argument(
        "--chosen-hull-expand",
        type=float,
        default=0.08,
        help="Grow hull away from centroid by this fraction (0 = tight hull).",
    )
    return parser.parse_args()


def load_cache(cache_path: Path):
    if not cache_path.exists():
        return {"embedding_model": None, "cache_schema": CACHE_SCHEMA, "items": []}
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("cache_schema", CACHE_SCHEMA)
    data.setdefault("items", [])
    return data


def save_cache(cache_path: Path, cache: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache["cache_schema"] = CACHE_SCHEMA
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def text_key(embedding_text: str, embedding_model: str) -> str:
    return hashlib.sha256(f"{embedding_model}\n{embedding_text}".encode("utf-8")).hexdigest()


def upsert_cache_item(cache: dict, item: dict):
    for i, old in enumerate(cache["items"]):
        if old.get("cache_key") == item.get("cache_key"):
            cache["items"][i] = item
            return
    cache["items"].append(item)


def collect_first_per_lang(jsonl_path: Path, target_langs: set):
    seen = set()
    out = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            lang = row.get("lang")
            if lang not in target_langs or lang in seen:
                continue
            seen.add(lang)
            out.append((line_no, row))
            if seen == target_langs:
                break
    missing = target_langs - seen
    if missing:
        raise ValueError(f"{jsonl_path}: no rows for langs {sorted(missing)}")
    return out


def build_points(jsonl_paths: dict, target_langs: set):
    points = []
    file_markers = {
        "lacomsa": "o",
        "icr": "s",
        "mapo": "D",
    }
    for key, path in jsonl_paths.items():
        path = Path(path)
        for line_no, row in collect_first_per_lang(path, target_langs):
            lang = row.get("lang", "")
            instruction = row.get("instruction") or ""
            pair_id = row.get("id")
            for side, field in (("chosen", "chosen"), ("rejected", "rejected")):
                response = row.get(field) or ""
                if not str(response).strip():
                    raise ValueError(f"{path} line {line_no} empty {field}")
                et = format_embedding_text(instruction, response)
                points.append(
                    {
                        "source_key": key,
                        "source_path": path.name,
                        "jsonl_line": line_no,
                        "lang": lang,
                        "pair_id": pair_id,
                        "side": side,
                        "instruction": instruction,
                        "response": response,
                        "embedding_text": et,
                        "marker": file_markers[key],
                    }
                )
    return points


def resolve_embeddings(points, embedding_model: str, cache_path: Path, cache_only: bool, force_refresh: bool):
    cache = load_cache(cache_path)
    if cache.get("embedding_model") not in (None, embedding_model):
        raise ValueError(
            f"Cache model {cache.get('embedding_model')!r} != {embedding_model!r}"
        )
    cache["embedding_model"] = embedding_model

    by_key = {it["cache_key"]: it for it in cache["items"] if "cache_key" in it}
    to_embed = []

    for p in points:
        ck = text_key(p["embedding_text"], embedding_model)
        p["cache_key"] = ck
        if not force_refresh:
            hit = by_key.get(ck)
            if hit and "embedding" in hit:
                p["embedding"] = hit["embedding"]
                continue
        to_embed.append(p)

    if to_embed and cache_only:
        raise RuntimeError(
            f"--cache-only: missing {len(to_embed)} embeddings in cache. "
            "Run once without --cache-only to fill."
        )

    if to_embed:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (needed for new embeddings).")
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(
            model=embedding_model,
            input=[x["embedding_text"] for x in to_embed],
        )
        for p, row in zip(to_embed, resp.data):
            p["embedding"] = row.embedding

    for p in points:
        item = {
            "cache_key": p["cache_key"],
            "source_key": p["source_key"],
            "source_path": p["source_path"],
            "jsonl_line": p["jsonl_line"],
            "lang": p["lang"],
            "pair_id": p["pair_id"],
            "side": p["side"],
            "embedding_model": embedding_model,
            "instruction": p["instruction"],
            "response": p["response"],
            "embedding_text": p["embedding_text"],
            "text": p["embedding_text"],
            "embedding": p["embedding"],
        }
        upsert_cache_item(cache, item)

    save_cache(cache_path, cache)
    return points


def _data_span(xy: np.ndarray) -> float:
    return float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9))


def _expand_about_centroid(pts_xy: np.ndarray, pad_frac: float) -> np.ndarray:
    c = pts_xy.mean(axis=0)
    return c + (1.0 + pad_frac) * (pts_xy - c)


def _draw_chosen_region(
    ax,
    xy: np.ndarray,
    points: list,
    *,
    alpha: float,
    pad_frac: float,
):
    idx = [i for i, p in enumerate(points) if p["side"] == "chosen"]
    if not idx:
        return
    P = xy[idx].astype(np.float64)
    U = np.unique(np.round(P, decimals=10), axis=0)
    span = _data_span(xy)
    stripe_w = max(span * 0.045, 1e-6)

    edge = "#1b7837"
    face = "#33a02c"

    def add_polygon(vertices: np.ndarray):
        vert = np.asarray(vertices, dtype=np.float64)
        if pad_frac != 0.0 and len(vert) >= 1:
            vert = _expand_about_centroid(vert, pad_frac)
        ax.add_patch(
            Polygon(
                vert,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.8,
                alpha=alpha,
                zorder=0,
            )
        )

    def stripe_between(a: np.ndarray, b: np.ndarray):
        aa, bb = np.asarray(a), np.asarray(b)
        if pad_frac != 0.0:
            ctr = np.stack([aa, bb]).mean(axis=0)
            m = ctr + (1.0 + pad_frac) * (np.stack([aa, bb]) - ctr)
            aa, bb = m[0], m[1]
        d = bb - aa
        L = float(np.linalg.norm(d)) + 1e-12
        nrm = np.array([-d[1], d[0]]) / L
        thickness = stripe_w * 3.0 + abs(pad_frac) * stripe_w * 12.0
        quad = np.stack(
            [
                aa + nrm * thickness,
                bb + nrm * thickness,
                bb - nrm * thickness,
                aa - nrm * thickness,
            ]
        )
        add_polygon(quad)

    def blob(center: np.ndarray, w: float, h: float, angle_deg: float = 0.0):
        ax.add_patch(
            Ellipse(
                tuple(center),
                width=w,
                height=h,
                angle=angle_deg,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.8,
                alpha=alpha,
                zorder=0,
            )
        )

    if len(U) == 1:
        w = span * 0.07 + stripe_w * 2.0
        blob(U[0], w, w)
        return
    if len(U) == 2:
        stripe_between(U[0], U[1])
        return

    verts = np.empty((0, 2), dtype=np.float64)
    try:
        hull = ConvexHull(U)
        verts = hull.points[hull.vertices]
    except QhullError:
        verts = np.empty((0, 2), dtype=np.float64)

    if verts.shape[0] >= 3:
        add_polygon(verts)
        return

    if verts.shape[0] == 2:
        stripe_between(verts[0], verts[1])
        return

    ctr = U.mean(axis=0)
    blob(ctr, stripe_w * 10.0, stripe_w * 9.0)


def _scatter_projection(ax, xy, points, title, xlabel, ylabel, show_lang: bool, show_file: bool):
    for i, p in enumerate(points):
        color = "tab:green" if p["side"] == "chosen" else "tab:red"
        ax.scatter(
            xy[i, 0],
            xy[i, 1],
            c=[color],
            marker=p["marker"],
            s=120,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )
        parts = []
        if show_lang:
            parts.append(p["lang"])
        if show_file:
            parts.append(p["source_path"])
        if parts:
            ax.annotate(
                " | ".join(parts),
                (xy[i, 0], xy[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9,
                zorder=5,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_umap_tsne(
    points,
    plot_path: Path,
    show_lang: bool,
    show_file: bool,
    *,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_random_state: int,
    tsne_perplexity: float,
    tsne_random_state: int,
    show_chosen_hull: bool,
    chosen_hull_alpha: float,
    chosen_hull_expand: float,
):
    X = np.array([p["embedding"] for p in points], dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in embedding matrix.")
    n = len(points)
    if n < 3:
        raise ValueError(f"Need at least 3 points for UMAP/t-SNE, got {n}")

    n_neighbors = max(2, min(int(umap_n_neighbors), n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
    )
    xy_umap = reducer.fit_transform(X)

    max_perp = float(n - 1)
    # sklearn requires perplexity < n_samples
    perp = min(float(tsne_perplexity), max_perp - 1e-6 if max_perp > 1 else max_perp)
    perp = max(perp, 2.0) if max_perp >= 3 else max(perp, 1.0)

    xy_tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=tsne_random_state,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
    ).fit_transform(X)

    fig, (ax_u, ax_t) = plt.subplots(1, 2, figsize=(14, 6.5))

    nn_note = n_neighbors if n_neighbors == int(umap_n_neighbors) else f"{n_neighbors} (clamped)"
    if show_chosen_hull:
        _draw_chosen_region(
            ax_u,
            xy_umap,
            points,
            alpha=chosen_hull_alpha,
            pad_frac=chosen_hull_expand,
        )

    _scatter_projection(
        ax_u,
        xy_umap,
        points,
        f"UMAP (n_neighbors={nn_note}, instruction + response)",
        "UMAP-1",
        "UMAP-2",
        show_lang,
        show_file,
    )

    pp_note = tsne_perplexity if abs(perp - tsne_perplexity) < 1e-4 else f"{perp:.2f} (clamped)"
    if show_chosen_hull:
        _draw_chosen_region(
            ax_t,
            xy_tsne,
            points,
            alpha=chosen_hull_alpha,
            pad_frac=chosen_hull_expand,
        )

    _scatter_projection(
        ax_t,
        xy_tsne,
        points,
        f"t-SNE (perplexity={pp_note}, instruction + response)",
        "t-SNE 1",
        "t-SNE 2",
        show_lang,
        show_file,
    )

    leg = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green", markersize=10, label="chosen"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=10, label="rejected"),
        plt.Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=10, label="lacomsa"),
        plt.Line2D([0], [0], marker="s", color="k", linestyle="None", markersize=10, label="icr"),
        plt.Line2D([0], [0], marker="D", color="k", linestyle="None", markersize=10, label="mapo"),
    ]
    if show_chosen_hull:
        leg.append(
            mpatches.Patch(
                facecolor="#33a02c",
                edgecolor="#1b7837",
                linewidth=1.8,
                alpha=chosen_hull_alpha,
                label="chosen convex hull",
            )
        )

    ncol = len(leg)
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.085), ncol=ncol, fontsize=9)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.86)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    target_langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    if not target_langs:
        raise ValueError("--langs empty")

    jsonl_paths = {
        "lacomsa": Path(args.lacomsa_jsonl),
        "icr": Path(args.icr_jsonl),
        "mapo": Path(args.mapo_jsonl),
    }
    cache_path = Path(args.cache_path)
    plot_path = Path(args.plot_path)

    points = build_points(jsonl_paths, target_langs)
    resolve_embeddings(
        points,
        embedding_model=args.embedding_model,
        cache_path=cache_path,
        cache_only=args.cache_only,
        force_refresh=args.force_refresh,
    )
    plot_umap_tsne(
        points,
        plot_path,
        show_lang=args.show_lang_label,
        show_file=args.show_file_label,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_random_state=args.umap_random_state,
        tsne_perplexity=args.tsne_perplexity,
        tsne_random_state=args.tsne_random_state,
        show_chosen_hull=args.show_chosen_hull,
        chosen_hull_alpha=args.chosen_hull_alpha,
        chosen_hull_expand=args.chosen_hull_expand,
    )
    print(f"Points: {len(points)}, cache: {cache_path}, plot: {plot_path}")


if __name__ == "__main__":
    main()
