"""
UMAP+t-SNE from precomputed embedding JSON files only.

This script does not call OpenAI or any embedding HTTP API — it reads vectors already
stored in caches (populate them with embedding_*.py / plot_embedding_same_instruction.py flow).
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial import ConvexHull, QhullError
from sklearn.manifold import TSNE


_ALLOWED_METHODS = ("lacomsa", "icr", "mapo")


def _parse_methods_csv(s: str) -> list[str]:
    raw = [x.strip().lower() for x in s.split(",") if x.strip()]
    if not raw:
        return []
    bad = [x for x in raw if x not in _ALLOWED_METHODS]
    if bad:
        raise ValueError(
            f"Unknown method(s): {bad}. Use only: {', '.join(_ALLOWED_METHODS)} "
            "(comma-separated, duplicates ignored)."
        )
    seen = set()
    order = []
    for x in raw:
        if x not in seen:
            seen.add(x)
            order.append(x)
    return order


def _parse_methods_csv_or_empty(s: str) -> list[str]:
    raw = (s or "").strip()
    if not raw:
        return []
    return _parse_methods_csv(raw)


def _load_same_instruction_module():
    here = Path(__file__).resolve().parent
    path = here / "plot_embedding_same_instruction.py"
    spec = importlib.util.spec_from_file_location("plot_embedding_same_instruction", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "UMAP+t-SNE: chosen/rejected from main cache "
            "(plot_embedding_same_instruction); optional Gemini (one point per --langs, default es/ru) "
            "from --gemini-cache-path; optional extra training-checkpoint embeddings from "
            "--extra-checkpoint-cache-paths (e.g. ICR+NPO Ultrafeedback per-checkpoint JSON cache); "
            "optional --additional-chosen-cache-path / --additional-rejected-cache-path (pre-embedded batches); "
            "optional --policy-rollout-cache-path (embedding_policy_rollout_responses.py output, same markers as add. rejected). "
            "Use --skip-gemini for main + extras only. "
            "No embedding API calls are made — only disk cache reads."
        )
    )
    p.add_argument("--lacomsa-jsonl", default="/home/gangstat/NeMA_result/_pairs/train_lacomsa_relabeled.jsonl")
    p.add_argument("--icr-jsonl", default="/home/gangstat/NeMA_result/_pairs/train_icr.jsonl")
    p.add_argument("--mapo-jsonl", default="/home/gangstat/NeMA_result/_pairs/train_mapo.jsonl")
    p.add_argument("--langs", default="es,ru")
    p.add_argument(
        "-M",
        "--methods",
        "--sources",
        dest="methods",
        default="lacomsa,icr,mapo",
        metavar="NAMES",
        help=(
            "Comma-separated subset of lacomsa, icr, mapo to plot (main JSONL + main cache points). "
            "Order is kept for Gemini cache tie-break (first match per lang). "
            "--sources is a backward-compatible alias."
        ),
    )
    p.add_argument(
        "--main-chosen-methods",
        default="",
        metavar="NAMES",
        help=(
            "Optional override for chosen-side main points only (subset of lacomsa, icr, mapo). "
            "Empty => fallback to --methods."
        ),
    )
    p.add_argument(
        "--main-rejected-methods",
        default="",
        metavar="NAMES",
        help=(
            "Optional override for rejected-side main points only (subset of lacomsa, icr, mapo). "
            "Empty => fallback to --methods."
        ),
    )
    p.add_argument(
        "--main-cache-path",
        default="/home/gangstat/NeMA_result/analysis/embeddings/plot_same_instruction_es_ru_cache.json",
        help="Cache with chosen/rejected embeddings.",
    )
    p.add_argument(
        "--gemini-cache-path",
        default="/home/gangstat/NeMA_result/analysis/embeddings/gemini_same_instruction_es_ru_cache.json",
        help="Cache written by embedding_gemini.py.",
    )
    p.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help=(
            "Must match main cache, Gemini (unless --skip-gemini), UFB caches, "
            "and optional additional chosen/rejected caches."
        ),
    )
    p.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Do not load or plot Gemini embeddings (main cache + extras only).",
    )
    p.add_argument(
        "--extra-checkpoint-cache-paths",
        default=(
            "/home/gangstat/NeMA_result/analysis/embeddings/"
            "icr_npo_ultrafeedback_first_es_ru_cache.json"
        ),
        help=(
            "Comma-separated JSON caches whose items include checkpoint_dir + lang + embedding "
            "(e.g. embedding_icr_npo_ufb.py output). Use empty string to disable."
        ),
    )
    p.add_argument(
        "--additional-chosen-cache-path",
        default="",
        help=(
            "JSON cache with pre-computed embeddings for extra chosen-side points "
            "(e.g. icr_*_pos_chosen_cache.json). Filtered by --langs. Empty to disable."
        ),
    )
    p.add_argument(
        "--additional-rejected-cache-path",
        default="",
        help=(
            "JSON cache with pre-computed embeddings for extra rejected-side points "
            "(e.g. icr_*_neg_rejected_cache.json). Filtered by --langs. Empty to disable."
        ),
    )
    p.add_argument(
        "--policy-rollout-cache-path",
        default="",
        help=(
            "JSON cache from embedding_policy_rollout_responses.py (policy instruction+response rollouts). "
            "Plotted like --additional-rejected-cache-path (red X, variant label). Filtered by --langs. Empty to disable."
        ),
    )
    p.add_argument(
        "--plot-path",
        default="/home/gangstat/NeMA_result/analysis/figures/plot_embedding_with_gemini_umap_tsne.png",
    )
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-random-state", type=int, default=42)
    p.add_argument("--tsne-perplexity", type=float, default=5.0)
    p.add_argument("--tsne-random-state", type=int, default=42)
    p.add_argument("--show-lang-label", action="store_true")
    p.add_argument("--show-file-label", action="store_true")
    p.add_argument(
        "--show-chosen-hull",
        action="store_true",
        help="Per-language convex hulls around chosen (green) plus Gemini stars (same lang only).",
    )
    p.add_argument("--chosen-hull-alpha", type=float, default=0.22)
    p.add_argument("--chosen-hull-expand", type=float, default=0.08)
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print which JSON caches were read and point counts (still no embedding API calls).",
    )
    return p.parse_args()


def hydrate_from_main_cache(sess, points, embedding_model: str, cache_path: Path):
    cache = sess.load_cache(cache_path)
    if cache.get("embedding_model") and cache["embedding_model"] != embedding_model:
        raise ValueError(
            f"Main cache embedding_model={cache['embedding_model']!r} != CLI {embedding_model!r}"
        )
    by_key = {it["cache_key"]: it["embedding"] for it in cache["items"] if "cache_key" in it}
    for pt in points:
        ck = sess.text_key(pt["embedding_text"], embedding_model)
        if ck not in by_key:
            raise RuntimeError(
                f"Missing main-cache embedding for {pt['source_key']} {pt['side']} {pt['lang']}. "
                f"Run plot_embedding_same_instruction.py with the same paths/model first."
            )
        pt["embedding"] = by_key[ck]


def select_jsonl_paths(keys, paths):
    bad = [k for k in keys if k not in paths]
    if bad:
        raise ValueError(f"Unknown method: {bad}. Allowed: {list(paths)}.")
    return {k: paths[k] for k in keys}


def filter_main_points_by_side_methods(
    points: list[dict],
    chosen_methods: set[str],
    rejected_methods: set[str],
) -> list[dict]:
    out = []
    for p in points:
        side = p.get("side")
        src = p.get("source_key")
        if side == "chosen" and src in chosen_methods:
            out.append(p)
        elif side == "rejected" and src in rejected_methods:
            out.append(p)
    return out


def pick_gemini_one_per_lang(
    gemini_cache_path: Path,
    embedding_model: str,
    *,
    source_order,
    source_keys,
    target_langs,
):
    """
    From Gemini cache rows whose source_key is in ``source_keys``,
    sort by (index in ``source_order``, jsonl_line) and take the first row seen for each lang in ``target_langs``.
    Returns len(target_langs) plot-style dicts (typically 2: es + ru).
    """
    raw = gemini_cache_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("embedding_model") and data["embedding_model"] != embedding_model:
        raise ValueError(
            f"Gemini cache embedding_model={data['embedding_model']!r} != CLI {embedding_model!r}"
        )

    prio = {k: i for i, k in enumerate(source_order)}
    rows = [it for it in (data.get("items") or []) if it.get("source_key") in source_keys]
    if not rows:
        raise RuntimeError(
            f"No Gemini cache rows for source_keys={sorted(source_keys)} in {gemini_cache_path}"
        )
    rows.sort(key=lambda it: (prio.get(it.get("source_key"), 99), it.get("jsonl_line", 0)))

    found = {}
    for it in rows:
        lg = it.get("lang")
        if lg not in target_langs or lg in found:
            continue
        emb = it.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError(
                f"Gemini cache item missing embedding: {it.get('source_key')} {it.get('lang')}"
            )
        found[lg] = it
        if len(found) == len(target_langs):
            break

    missing = target_langs - set(found.keys())
    if missing:
        raise RuntimeError(
            f"Need exactly one Gemini cache row per lang in {sorted(target_langs)} "
            f"(after filtering source_key in {sorted(source_keys)}, ordered by --methods then line). "
            f"Still missing langs: {sorted(missing)}."
        )

    out = []
    for lg in sorted(target_langs):
        it = found[lg]
        out.append(
            {
                "side": "gemini",
                "marker": "*",
                "lang": it.get("lang", ""),
                "source_key": it.get("source_key", ""),
                "source_path": it.get("source_path", ""),
                "jsonl_line": it.get("jsonl_line"),
                "pair_id": it.get("pair_id"),
                "instruction": it.get("instruction"),
                "response": it.get("gemini_response"),
                "embedding_text": it.get("embedding_text"),
                "embedding": it.get("embedding"),
            }
        )
    return out


def parse_extra_checkpoint_cache_paths(s: str) -> list[Path]:
    if not s or not str(s).strip():
        return []
    return [Path(p.strip()) for p in str(s).split(",") if p.strip()]


def optional_cache_path(cli_value: str) -> Path | None:
    raw = (cli_value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_file():
        raise FileNotFoundError(f"Cache file not found: {p}")
    return p


def load_additional_side_cache(
    cache_path: Path,
    side: str,
    embedding_model: str,
    target_langs: set[str],
) -> list[dict]:
    """
    Loads items with embeddings from a supplementary cache (e.g. icr_train_* lines batch).
    Markers: pentagon ``p`` (chosen), ``X`` (rejected), same face colors as main chosen/rejected.
    """
    if side not in ("chosen", "rejected"):
        raise ValueError(f"side must be chosen or rejected, got {side!r}")

    raw = cache_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("embedding_model") and data["embedding_model"] != embedding_model:
        raise ValueError(
            f"{cache_path.name}: embedding_model={data['embedding_model']!r} "
            f"!= CLI {embedding_model!r}"
        )
    stem = cache_path.stem
    marker = "p" if side == "chosen" else "X"
    out: list[dict] = []
    for it in data.get("items") or []:
        lg = it.get("lang")
        if lg not in target_langs:
            continue
        emb = it.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError(
                f"{cache_path}: missing embedding (lang={lg}, jsonl_line={it.get('jsonl_line')})"
            )
        out.append(
            {
                "side": side,
                "marker": marker,
                "lang": lg,
                "source_key": f"additional_{side}",
                "source_path": str(it.get("jsonl_path") or cache_path),
                "embedding": emb,
                "embedding_text": it.get("embedding_text") or it.get("text") or "",
                "series": "additional_cache",
                "additional_label": it.get("variant") or stem,
                "pair_id": it.get("pair_id"),
                "jsonl_line": it.get("jsonl_line"),
                "instruction": it.get("instruction"),
            }
        )
    out.sort(key=lambda r: (r.get("lang", ""), int(r.get("jsonl_line") or 0)))
    return out


def _policy_rollout_style(model_type: str) -> tuple[str, str]:
    """(matplotlib marker, face color hex) for policy rollout cache rows."""
    mt = (model_type or "").strip()
    known = {
        "icr_ppo": ("P", "#ff7f0e"),
        "icr-w-reinforce-0.1-checkpoint10": ("D", "#756bb1"),
        "icr-w-reinforce-0.1-checkpoint36": ("s", "#31a354"),
    }
    if mt in known:
        return known[mt]
    i = int(hashlib.sha256(mt.encode("utf-8")).hexdigest()[:8], 16)
    markers = ("P", "D", "s", "v", "^", "<", ">", "h")
    return markers[i % len(markers)], _EXTRA_CKPT_COLORS_HEX[i % len(_EXTRA_CKPT_COLORS_HEX)]


def load_policy_rollout_cache(
    cache_path: Path,
    embedding_model: str,
    target_langs: set[str],
) -> list[dict]:
    """
    Same JSON shape as embedding_policy_rollout_responses.py output, but plotted with
    distinct markers/colors per ``model_type`` (not the same red X as human rejected).
    """
    raw = cache_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("embedding_model") and data["embedding_model"] != embedding_model:
        raise ValueError(
            f"{cache_path.name}: embedding_model={data['embedding_model']!r} "
            f"!= CLI {embedding_model!r}"
        )
    stem = cache_path.stem
    out: list[dict] = []
    for it in data.get("items") or []:
        lg = it.get("lang")
        if lg not in target_langs:
            continue
        emb = it.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError(
                f"{cache_path}: missing embedding (lang={lg}, jsonl_line={it.get('jsonl_line')})"
            )
        mt = it.get("model_type") or ""
        if not mt and it.get("jsonl_path"):
            mt = Path(str(it["jsonl_path"])).parent.name
        marker, plot_color = _policy_rollout_style(mt)
        out.append(
            {
                "side": "rejected",
                "marker": marker,
                "plot_color": plot_color,
                "lang": lg,
                "source_key": "policy_rollout",
                "source_path": str(it.get("jsonl_path") or cache_path),
                "embedding": emb,
                "embedding_text": it.get("embedding_text") or it.get("text") or "",
                "series": "policy_rollout",
                "model_type": mt,
                "additional_label": it.get("variant") or stem,
                "pair_id": it.get("pair_id"),
                "jsonl_line": it.get("jsonl_line"),
                "instruction": it.get("instruction"),
            }
        )
    out.sort(key=lambda r: (r.get("lang", ""), str(r.get("model_type", "")), int(r.get("jsonl_line") or 0)))
    return out


def _checkpoint_sort_key(dirname: str) -> tuple[int, str]:
    m = re.match(r"^checkpoint-(\d+)$", dirname)
    if m:
        return (0, f"{int(m.group(1)):08d}")
    return (1, dirname)


_EXTRA_CKPT_COLORS_HEX = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _color_for_ckpt_dirs(ck_dirs: list[str]) -> dict[str, str]:
    uniq = sorted(set(ck_dirs), key=_checkpoint_sort_key)
    pal = _EXTRA_CKPT_COLORS_HEX
    return {name: pal[i % len(pal)] for i, name in enumerate(uniq)}


def load_extra_checkpoint_model_points(
    cache_paths: list[Path],
    embedding_model: str,
    target_langs: set[str],
) -> list[dict]:
    """
    Load cache rows shaped like embedding_icr_npo_ufb.py (checkpoint_dir, lang, embedding, ...).
    Multiple cache files merged; marker ^ for es / v for ru; color cycles per checkpoint_* dir.
    """
    loaded: list[dict] = []
    for cp in cache_paths:
        if not cp.is_file():
            raise FileNotFoundError(f"Extra checkpoint cache not found: {cp}")
        data = json.loads(cp.read_text(encoding="utf-8"))
        if data.get("embedding_model") and data["embedding_model"] != embedding_model:
            raise ValueError(
                f"{cp.name}: embedding_model={data['embedding_model']!r} != CLI {embedding_model!r}"
            )
        for it in data.get("items") or []:
            cdir = it.get("checkpoint_dir")
            lg = it.get("lang")
            emb = it.get("embedding")
            if not cdir or lg not in target_langs:
                continue
            if not isinstance(emb, list) or not emb:
                raise RuntimeError(
                    f"{cp}: item missing embedding for {cdir} {lg} (populate cache first)."
                )
            loaded.append(
                {
                    **it,
                    "cache_file": cp.name,
                    "side": "ufb_ckpt",
                    "marker": "^" if lg == "es" else "v",
                    "checkpoint_dir": cdir,
                    "lang": lg,
                    "embedding": emb,
                    "embedding_text": it.get("embedding_text") or it.get("text", ""),
                    "source_path": it.get("jsonl_path", ""),
                }
            )

    if not loaded:
        return []

    colors = _color_for_ckpt_dirs([p["checkpoint_dir"] for p in loaded])
    for p in loaded:
        p["plot_color"] = colors[p["checkpoint_dir"]]

    loaded.sort(key=lambda p: (_checkpoint_sort_key(p["checkpoint_dir"]), p["lang"], p["cache_file"]))
    return loaded


def _data_span(xy: np.ndarray) -> float:
    return float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9))


def _expand_about_centroid(pts_xy: np.ndarray, pad_frac: float) -> np.ndarray:
    c = pts_xy.mean(axis=0)
    return c + (1.0 + pad_frac) * (pts_xy - c)


def _draw_single_lang_hull_cluster(
    ax, xy: np.ndarray, idx: list[int], *, alpha: float, pad_frac: float, span: float, stripe_w: float
):
    """Hull/stripe/blob for chosen+Gemini points that share one language."""
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

    P = xy[np.asarray(idx, dtype=int)].astype(np.float64)
    U = np.unique(np.round(P, decimals=10), axis=0)

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


def _draw_positive_hull_region(ax, xy, points, *, alpha: float, pad_frac: float):
    """
    One convex hull (or degenerate stripe/blob) per language for chosen + Gemini points only.
    Different langs are never connected in one polygon.
    """
    by_lang: dict[str, list[int]] = {}
    for i, p in enumerate(points):
        if p.get("side") not in ("chosen", "gemini"):
            continue
        lg = str(p.get("lang") or "").strip() or "__na__"
        by_lang.setdefault(lg, []).append(i)
    if not by_lang:
        return

    span = _data_span(xy)
    stripe_w = max(span * 0.045, 1e-6)
    for _lg, idx in sorted(by_lang.items(), key=lambda kv: kv[0]):
        if not idx:
            continue
        _draw_single_lang_hull_cluster(
            ax, xy, idx, alpha=alpha, pad_frac=pad_frac, span=span, stripe_w=stripe_w
        )


def _scatter_projection(ax, xy, points, title, xlabel, ylabel, show_lang: bool, show_file: bool):
    for i, p in enumerate(points):
        side = p.get("side", "")
        series = p.get("series", "")
        if series == "policy_rollout":
            color = p.get("plot_color", "#ff7f0e")
        elif side == "rejected":
            color = "tab:red"
        elif side in ("chosen", "gemini"):
            color = "tab:green"
        elif side == "ufb_ckpt":
            color = p.get("plot_color", "tab:gray")
        else:
            color = "tab:gray"
        if side == "gemini":
            size = 185
        elif side == "ufb_ckpt":
            size = 132
        elif series == "policy_rollout":
            size = 118
        elif p.get("series") == "additional_cache":
            size = 95
        else:
            size = 120
        ax.scatter(
            xy[i, 0],
            xy[i, 1],
            c=[color],
            marker=p["marker"],
            s=size,
            alpha=0.85 if side != "ufb_ckpt" else 0.88,
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
        )
        parts = []
        if side == "gemini":
            parts.append("gemini")
        if series == "policy_rollout":
            mt = str(p.get("model_type") or "").strip()
            if mt:
                parts.append(mt if len(mt) <= 40 else mt[:37] + "...")
            if (show_lang or show_file) and p.get("additional_label"):
                parts.append(str(p["additional_label"])[:48])
        if p.get("series") == "additional_cache" and (show_lang or show_file):
            jl = p.get("jsonl_line")
            lbl = str(p.get("additional_label") or "")[:42]
            if lbl:
                parts.append(lbl)
            if jl is not None:
                parts.append(f"L{jl}")
        if side == "ufb_ckpt":
            ck = p.get("checkpoint_dir", "")
            if ck:
                parts.append(ck)
        if show_lang:
            parts.append(p.get("lang", ""))
        if show_file:
            parts.append(p.get("source_path", ""))
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
    plot_has_gemini: bool,
    plot_has_extra_ckpts: bool,
    methods_plotted: list[str],
    plot_has_additional_chosen: bool,
    plot_has_additional_rejected: bool,
    plot_has_policy_rollout: bool,
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

    if show_chosen_hull:
        _draw_positive_hull_region(
            ax_u, xy_umap, points, alpha=chosen_hull_alpha, pad_frac=chosen_hull_expand
        )
    _scatter_projection(
        ax_u,
        xy_umap,
        points,
        "UMAP",
        "UMAP-1",
        "UMAP-2",
        show_lang,
        show_file,
    )

    if show_chosen_hull:
        _draw_positive_hull_region(
            ax_t, xy_tsne, points, alpha=chosen_hull_alpha, pad_frac=chosen_hull_expand
        )
    _scatter_projection(
        ax_t,
        xy_tsne,
        points,
        "T-SNE",
        "T-SNE 1",
        "T-SNE 2",
        show_lang,
        show_file,
    )

    leg = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:green",
            markersize=10,
            label="chosen",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:red",
            markersize=10,
            label="rejected",
        ),
    ]
    if plot_has_additional_chosen:
        leg.append(
            plt.Line2D(
                [0],
                [0],
                marker="p",
                color="w",
                markerfacecolor="tab:green",
                markersize=10,
                linestyle="None",
                label="add. chosen cache (p)",
            )
        )
    if plot_has_additional_rejected:
        leg.append(
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="tab:red",
                markersize=10,
                linestyle="None",
                label="add. rejected cache (X)",
            )
        )
    if plot_has_gemini:
        leg.append(
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:green", markersize=12, label="Gemini"),
        )
    if plot_has_extra_ckpts:
        leg.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#9467bd",
                markersize=10,
                label="UFB checkpoints (^ es / v ru; color = ckpt)",
            )
        )
    if plot_has_policy_rollout:
        seen = set()
        for p in points:
            if p.get("series") != "policy_rollout":
                continue
            mt = str(p.get("model_type") or "").strip() or "policy"
            mk = p.get("marker") or "P"
            col = p.get("plot_color") or "#ff7f0e"
            key = (mt, mk, col)
            if key in seen:
                continue
            seen.add(key)
            lbl = f"rollout {mt}" if len(mt) <= 36 else f"rollout {mt[:33]}..."
            leg.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=mk,
                    color="w",
                    markerfacecolor=col,
                    markersize=10,
                    linestyle="None",
                    label=lbl,
                )
            )
    method_legend = {
        "lacomsa": plt.Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=10, label="lacomsa"),
        "icr": plt.Line2D([0], [0], marker="s", color="k", linestyle="None", markersize=10, label="icr"),
        "mapo": plt.Line2D([0], [0], marker="D", color="k", linestyle="None", markersize=10, label="mapo"),
    }
    for m in methods_plotted:
        if m in method_legend:
            leg.append(method_legend[m])
    hull_label = (
        "convex hull / lang (chosen + Gemini)"
        if plot_has_gemini
        else "convex hull / lang (chosen)"
    )
    if show_chosen_hull:
        leg.append(
            mpatches.Patch(
                facecolor="#33a02c",
                edgecolor="#1b7837",
                linewidth=1.8,
                alpha=chosen_hull_alpha,
                label=hull_label,
            )
        )

    ncol = len(leg)
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.085), ncol=ncol, fontsize=9)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.84)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    print(
        "[plot_embedding_with_full] Embedding API: not used "
        "(read-only: vectors loaded from JSON caches on disk)."
    )
    sess = _load_same_instruction_module()
    target_langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    if not target_langs:
        raise ValueError("--langs empty")

    sk = _parse_methods_csv(args.methods)
    if not sk:
        raise ValueError("--methods empty (use lacomsa, icr, mapo)")
    chosen_sk = _parse_methods_csv_or_empty(args.main_chosen_methods) or sk
    rejected_sk = _parse_methods_csv_or_empty(args.main_rejected_methods) or sk
    if not chosen_sk and not rejected_sk:
        raise ValueError("At least one of chosen/rejected method lists must be non-empty.")
    source_keys = set(sk)
    all_paths = {
        "lacomsa": Path(args.lacomsa_jsonl),
        "icr": Path(args.icr_jsonl),
        "mapo": Path(args.mapo_jsonl),
    }
    jsonl_paths = select_jsonl_paths(sk, all_paths)
    baseline = sess.build_points(jsonl_paths, target_langs)
    baseline = filter_main_points_by_side_methods(
        baseline,
        set(chosen_sk),
        set(rejected_sk),
    )
    if not baseline:
        raise ValueError(
            "No main baseline points after side-specific filtering. "
            "Check --main-chosen-methods/--main-rejected-methods with --methods/--langs."
        )
    hydrate_from_main_cache(sess, baseline, args.embedding_model, Path(args.main_cache_path))

    add_chosen_path = optional_cache_path(args.additional_chosen_cache_path)
    add_chosen_pts = (
        load_additional_side_cache(
            add_chosen_path,
            "chosen",
            args.embedding_model,
            target_langs,
        )
        if add_chosen_path
        else []
    )
    add_rejected_path = optional_cache_path(args.additional_rejected_cache_path)
    add_rejected_pts = (
        load_additional_side_cache(
            add_rejected_path,
            "rejected",
            args.embedding_model,
            target_langs,
        )
        if add_rejected_path
        else []
    )

    policy_rollout_path = optional_cache_path(args.policy_rollout_cache_path)
    policy_rollout_pts = (
        load_policy_rollout_cache(
            policy_rollout_path,
            args.embedding_model,
            target_langs,
        )
        if policy_rollout_path
        else []
    )

    extra_paths = parse_extra_checkpoint_cache_paths(args.extra_checkpoint_cache_paths)
    extras = (
        load_extra_checkpoint_model_points(extra_paths, args.embedding_model, target_langs)
        if extra_paths
        else []
    )

    gemini_pts = []
    if not args.skip_gemini:
        gemini_pts = pick_gemini_one_per_lang(
            Path(args.gemini_cache_path),
            args.embedding_model,
            source_order=sk,
            source_keys=source_keys,
            target_langs=target_langs,
        )

    merged = baseline + gemini_pts + extras + add_chosen_pts + add_rejected_pts + policy_rollout_pts

    if args.verbose:
        print("[plot_embedding_with_full] Cached sources (vectors from disk):")
        print(f"  main chosen methods:   {','.join(chosen_sk)}")
        print(f"  main rejected methods: {','.join(rejected_sk)}")
        print(f"  main:              {args.main_cache_path}")
        print(f"    baseline points: {len(baseline)} chosen/rejected")
        if gemini_pts:
            print(f"  gemini:            {args.gemini_cache_path}")
            print(f"    Gemini points:   {len(gemini_pts)}")
        elif args.skip_gemini:
            print("  gemini:            (--skip-gemini)")
        if extra_paths:
            for ep in extra_paths:
                print(f"  extra UFB:         {ep}")
            print(f"    UFB points:      {len(extras)}")
        else:
            print("  extra UFB:         (disabled)")
        if add_chosen_path:
            print(f"  add-chosen cache: {add_chosen_path}  ({len(add_chosen_pts)} pts)")
        else:
            print("  add-chosen cache: (none)")
        if add_rejected_path:
            print(f"  add-rejected cache: {add_rejected_path}  ({len(add_rejected_pts)} pts)")
        else:
            print("  add-rejected cache: (none)")
        if policy_rollout_path:
            print(f"  policy-rollout cache: {policy_rollout_path}  ({len(policy_rollout_pts)} pts)")
        else:
            print("  policy-rollout cache: (none)")
        print(f"[plot_embedding_with_full] Total plotted vectors: {len(merged)} (still no remote embed calls).")

    subt = [
        f"methods={','.join(sk)}",
        f"main(ch={','.join(chosen_sk)}|rj={','.join(rejected_sk)})",
    ]
    if gemini_pts:
        subt.append("Gemini 1/lang")
    if extras:
        subt.append("UFB ckpts")
    if add_chosen_pts or add_rejected_pts or policy_rollout_pts:
        subt.append(
            f"add ch={len(add_chosen_pts)} rj={len(add_rejected_pts)} pol={len(policy_rollout_pts)}"
        )
    run_summary = " + ".join(subt)

    plot_umap_tsne(
        merged,
        Path(args.plot_path),
        args.show_lang_label,
        args.show_file_label,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_random_state=args.umap_random_state,
        tsne_perplexity=args.tsne_perplexity,
        tsne_random_state=args.tsne_random_state,
        show_chosen_hull=args.show_chosen_hull,
        chosen_hull_alpha=args.chosen_hull_alpha,
        chosen_hull_expand=args.chosen_hull_expand,
        plot_has_gemini=bool(gemini_pts),
        plot_has_extra_ckpts=bool(extras),
        methods_plotted=sk,
        plot_has_additional_chosen=bool(add_chosen_pts),
        plot_has_additional_rejected=bool(add_rejected_pts),
        plot_has_policy_rollout=bool(policy_rollout_pts),
    )
    print(
        f"[methods={','.join(sk)}] "
        f"Plotted {len(baseline)} main + {len(gemini_pts)} Gemini + {len(extras)} UFB "
        f"+ {len(add_chosen_pts)} add-chosen + {len(add_rejected_pts)} add-rejected "
        f"+ {len(policy_rollout_pts)} policy-rollout "
        f"= {len(merged)} pts -> {args.plot_path}"
    )
    print(f"[run] {run_summary}")


if __name__ == "__main__":
    main()
