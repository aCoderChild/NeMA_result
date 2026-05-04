import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


EXPECTED_LANGS = ["de", "en", "es", "fr", "ru"]
EXPECTED_NPO_VARIANTS = [
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
            "Collect id=1 outputs from responses/icr annotations, cache embeddings, "
            "and plot dimensionality-reduced visualization."
        )
    )
    parser.add_argument(
        "--icr-root",
        type=str,
        default="/home/gangstat/NeMA_result/responses/icr",
        help="Root directory containing icr result folders.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=1,
        help="Response id to retrieve from each annotations.json.",
    )
    parser.add_argument(
        "--base-from-variant",
        type=str,
        default="npo_150426",
        help=(
            "Which NPO variant folder to use when collecting base output_1 "
            "for each language (example: npo_150426)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_id1_embeddings_cache.json",
        help="Path to save/reuse extracted outputs + model types + embeddings.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/figures/icr_id1_embeddings_dr.png",
        help="Path to save the DR plot image.",
    )
    return parser.parse_args()


def extract_lang(folder_name):
    m = re.match(r"^([a-z]{2})-results-", folder_name)
    return m.group(1) if m else None


def extract_npo_variant(folder_name):
    if "npo_150426" in folder_name:
        return "npo_150426"
    m = re.search(r"npo_checkpoint-(\d+)", folder_name)
    if m:
        return f"npo_checkpoint-{m.group(1)}"
    return None


def load_annotations_sample(annotations_path, sample_id):
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        row_id = row.get("id")
        if row_id == sample_id or str(row_id) == str(sample_id):
            return row
    raise ValueError(f"id={sample_id} not found in {annotations_path}")


def find_npo_folders(icr_root):
    # Map: variant -> lang -> folder path
    variant_lang_folder = {}
    for p in icr_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        lang = extract_lang(name)
        if not lang:
            continue
        variant = extract_npo_variant(name)
        if not variant:
            continue
        if not (p / "annotations.json").exists():
            continue
        variant_lang_folder.setdefault(variant, {})[lang] = p
    return variant_lang_folder


def text_key(text, embedding_model):
    digest = hashlib.sha256(f"{embedding_model}\n{text}".encode("utf-8")).hexdigest()
    return digest


def load_cache(cache_path):
    if not cache_path.exists():
        return {"embedding_model": None, "items": []}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def upsert_cache_item(cache, item):
    items = cache["items"]
    for i, old in enumerate(items):
        if old["cache_key"] == item["cache_key"]:
            items[i] = item
            return
    items.append(item)


def build_records(icr_root, sample_id, base_from_variant):
    records = []
    npo_map = find_npo_folders(icr_root)

    # 1) Base: output_1, one folder per language (sourced from chosen NPO variant)
    if base_from_variant not in npo_map:
        raise ValueError(
            f"Base variant '{base_from_variant}' not found. "
            f"Available variants: {sorted(npo_map.keys())}"
        )
    for lang in EXPECTED_LANGS:
        folder = npo_map[base_from_variant].get(lang)
        if folder is None:
            raise ValueError(
                f"Missing lang={lang} in base variant={base_from_variant}"
            )
        sample = load_annotations_sample(folder / "annotations.json", sample_id)
        records.append(
            {
                "group": "base_output_1",
                "lang": lang,
                "variant": "base_output_1",
                "folder": folder.name,
                "output_field": "output_1",
                "text": sample.get("output_1", ""),
                "model_type": sample.get("generator_1", "unknown"),
                "sample_id": sample_id,
            }
        )

    # 2) NPO: output_2, each variant for all languages
    missing_variants = [v for v in EXPECTED_NPO_VARIANTS if v not in npo_map]
    if missing_variants:
        raise ValueError(f"Missing NPO variants: {missing_variants}")

    for variant in EXPECTED_NPO_VARIANTS:
        lang_folder_map = npo_map[variant]
        for lang in EXPECTED_LANGS:
            folder = lang_folder_map.get(lang)
            if folder is None:
                raise ValueError(f"Missing lang={lang} for NPO variant={variant}")
            sample = load_annotations_sample(folder / "annotations.json", sample_id)
            records.append(
                {
                    "group": "npo_output_2",
                    "lang": lang,
                    "variant": variant,
                    "folder": folder.name,
                    "output_field": "output_2",
                    "text": sample.get("output_2", ""),
                    "model_type": sample.get("generator_2", "unknown"),
                    "sample_id": sample_id,
                }
            )

    for r in records:
        if not r["text"]:
            raise ValueError(
                f"Empty text detected ({r['folder']}, {r['output_field']}, id={sample_id})"
            )
    return records


def fill_embeddings(records, embedding_model, cache_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    cache = load_cache(cache_path)
    if cache.get("embedding_model") != embedding_model:
        cache = {"embedding_model": embedding_model, "items": []}

    cached_by_key = {item["cache_key"]: item for item in cache["items"]}
    to_embed = []

    for rec in records:
        key = text_key(rec["text"], embedding_model)
        rec["cache_key"] = key
        cached = cached_by_key.get(key)
        if cached:
            rec["embedding"] = cached["embedding"]
        else:
            to_embed.append(rec)

    if to_embed:
        client = OpenAI(api_key=api_key)
        inputs = [r["text"] for r in to_embed]
        response = client.embeddings.create(model=embedding_model, input=inputs)
        for rec, item in zip(to_embed, response.data):
            rec["embedding"] = item.embedding

    for rec in records:
        cache_item = {
            "cache_key": rec["cache_key"],
            "group": rec["group"],
            "lang": rec["lang"],
            "variant": rec["variant"],
            "folder": rec["folder"],
            "output_field": rec["output_field"],
            "sample_id": rec["sample_id"],
            "model_type": rec["model_type"],
            "embedding_model": embedding_model,
            "text": rec["text"],
            "embedding": rec["embedding"],
        }
        upsert_cache_item(cache, cache_item)

    save_cache(cache_path, cache)
    return records


def reduce_embeddings(embeddings):
    out = {}
    out["PCA"] = PCA(n_components=2).fit_transform(embeddings)
    out["MDS"] = MDS(
        n_components=2,
        random_state=42,
        dissimilarity="euclidean",
        normalized_stress="auto",
    ).fit_transform(embeddings)
    perplexity = max(2, min(30, embeddings.shape[0] // 3))
    out["t-SNE"] = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    ).fit_transform(embeddings)
    return out


def color_for_record(rec, npo_color_map):
    if rec["group"] == "base_output_1":
        return "green"
    return npo_color_map[rec["variant"]]


def plot_records(records, plot_path):
    vectors = np.array([r["embedding"] for r in records], dtype=np.float64)
    dr = reduce_embeddings(vectors)

    cmap = plt.get_cmap("tab10")
    npo_variants = [v for v in EXPECTED_NPO_VARIANTS]
    npo_color_map = {v: cmap(i % 10) for i, v in enumerate(npo_variants)}

    fig, axes = plt.subplots(1, len(dr), figsize=(7 * len(dr), 6), squeeze=False)
    axes = axes.ravel()

    for ax, (name, coords) in zip(axes, dr.items()):
        for i, rec in enumerate(records):
            color = color_for_record(rec, npo_color_map)
            marker = "o" if rec["group"] == "base_output_1" else "x"
            ax.scatter(coords[i, 0], coords[i, 1], c=[color], marker=marker, s=60, alpha=0.9)
            ax.annotate(
                rec["lang"],
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

    legend_handles = []
    legend_labels = []

    base_handle = plt.Line2D(
        [0], [0], marker="o", color="w", label="base_output_1", markerfacecolor="green", markersize=8
    )
    legend_handles.append(base_handle)
    legend_labels.append("base_output_1 (5 langs)")

    for variant in npo_variants:
        handle = plt.Line2D(
            [0],
            [0],
            marker="x",
            color=npo_color_map[variant],
            linestyle="None",
            markersize=8,
            label=variant,
        )
        legend_handles.append(handle)
        legend_labels.append(f"{variant} (5 langs)")

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("ICR id=1 Output Embeddings (base output_1 vs NPO output_2)", fontsize=14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    icr_root = Path(args.icr_root)
    cache_path = Path(args.cache_path)
    plot_path = Path(args.plot_path)

    records = build_records(
        icr_root=icr_root,
        sample_id=args.sample_id,
        base_from_variant=args.base_from_variant,
    )
    records = fill_embeddings(
        records=records,
        embedding_model=args.embedding_model,
        cache_path=cache_path,
    )
    plot_records(records, plot_path)

    print(f"Saved cache: {cache_path}")
    print(f"Saved plot : {plot_path}")
    print(f"Total points plotted: {len(records)}")


if __name__ == "__main__":
    main()
