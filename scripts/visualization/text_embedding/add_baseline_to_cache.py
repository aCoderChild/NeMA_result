import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


LANGS = ["de", "en", "es", "fr", "ru"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read baseline id=1 responses, append embeddings to cache, and re-render DR plot."
        )
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=1,
        help="Sample id to retrieve from each baseline JSONL.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="/home/gangstat/NeMA_result/responses/baseline",
        help="Directory containing baseline JSONL files.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_id1_embeddings_cache.json",
        help="Existing cache path to update with baseline entries.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/figures/icr_id1_embeddings_dr.png",
        help="Output plot path (overwrite).",
    )
    return parser.parse_args()


def baseline_file_for_lang(baseline_dir, lang):
    matches = sorted(
        baseline_dir.glob(
            f"{lang}.json.prediction.with_Llama-3-Base-8B-SFT-DPO.to_{lang}.jsonl"
        )
    )
    if not matches:
        raise FileNotFoundError(f"Baseline JSONL not found for lang={lang} in {baseline_dir}")
    return matches[0]


def read_jsonl_record_by_id(path, sample_id):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_id = row.get("id")
            if row_id == sample_id or str(row_id) == str(sample_id):
                return row
    raise ValueError(f"id={sample_id} not found in {path}")


def text_key(text, embedding_model):
    return hashlib.sha256(f"{embedding_model}\n{text}".encode("utf-8")).hexdigest()


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
    for i, old in enumerate(cache["items"]):
        if old.get("cache_key") == item.get("cache_key"):
            cache["items"][i] = item
            return
    cache["items"].append(item)


def collect_baseline_records(baseline_dir, sample_id):
    records = []
    for lang in LANGS:
        fpath = baseline_file_for_lang(baseline_dir, lang)
        row = read_jsonl_record_by_id(fpath, sample_id)
        responses = row.get("response", [])
        if not responses:
            raise ValueError(f"Empty response list in {fpath} for id={sample_id}")
        text = responses[0]
        records.append(
            {
                "group": "baseline_output",
                "lang": lang,
                "variant": "baseline_sft_dpo",
                "folder": fpath.name,
                "output_field": "response[0]",
                "sample_id": sample_id,
                "model_type": "Llama-3-Base-8B-SFT-DPO",
                "text": text,
            }
        )
    return records


def embed_missing(records, embedding_model, cache):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    if cache.get("embedding_model") not in (None, embedding_model):
        raise ValueError(
            f"Cache embedding_model={cache.get('embedding_model')} differs from requested "
            f"{embedding_model}. Use same model or clear cache."
        )
    cache["embedding_model"] = embedding_model

    cached_by_key = {it["cache_key"]: it for it in cache["items"] if "cache_key" in it}
    to_embed = []
    for rec in records:
        key = text_key(rec["text"], embedding_model)
        rec["cache_key"] = key
        cached = cached_by_key.get(key)
        if cached and "embedding" in cached:
            rec["embedding"] = cached["embedding"]
        else:
            to_embed.append(rec)

    if to_embed:
        client = OpenAI(api_key=api_key)
        inputs = [r["text"] for r in to_embed]
        response = client.embeddings.create(model=embedding_model, input=inputs)
        for rec, emb_row in zip(to_embed, response.data):
            rec["embedding"] = emb_row.embedding

    for rec in records:
        upsert_cache_item(
            cache,
            {
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
            },
        )


def reduce_embeddings(vectors):
    out = {}
    out["PCA"] = PCA(n_components=2).fit_transform(vectors)
    out["MDS"] = MDS(
        n_components=2,
        random_state=42,
        dissimilarity="euclidean",
        normalized_stress="auto",
    ).fit_transform(vectors)
    perplexity = max(2, min(30, vectors.shape[0] // 3))
    out["t-SNE"] = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    ).fit_transform(vectors)
    return out


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


def color_marker(item, npo_color_map):
    group = item.get("group")
    if group == "base_output_1":
        return "green", "o", "base_output_1"
    if group == "baseline_output":
        return "black", "^", "baseline_output"
    if group == "npo_output_2":
        return npo_color_map.get(item.get("variant"), "gray"), "x", item.get("variant", "npo")
    return "gray", ".", "other"


def plot_from_cache(cache, plot_path):
    items = [it for it in cache.get("items", []) if "embedding" in it]
    if not items:
        raise ValueError("No embeddings in cache to plot.")

    vectors = np.array([it["embedding"] for it in items], dtype=np.float64)
    dr = reduce_embeddings(vectors)
    npo_color_map = build_npo_color_map(items)

    fig, axes = plt.subplots(1, len(dr), figsize=(7 * len(dr), 6), squeeze=False)
    axes = axes.ravel()

    for ax, (name, coords) in zip(axes, dr.items()):
        for i, item in enumerate(items):
            color, marker, _ = color_marker(item, npo_color_map)
            ax.scatter(coords[i, 0], coords[i, 1], c=[color], marker=marker, s=60, alpha=0.9)
            lang = item.get("lang", "na")
            ax.annotate(lang, (coords[i, 0], coords[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)

    handles = []
    labels = []

    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8))
    labels.append("base_output_1")
    handles.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=8))
    labels.append("baseline_output")
    for variant, color in sorted(npo_color_map.items()):
        handles.append(plt.Line2D([0], [0], marker="x", color=color, linestyle="None", markersize=8))
        labels.append(variant)

    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("ICR id=1 Embeddings (base + NPO + baseline)", fontsize=14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    cache_path = Path(args.cache_path)
    plot_path = Path(args.plot_path)

    records = collect_baseline_records(baseline_dir, args.sample_id)
    cache = load_cache(cache_path)
    embed_missing(records, args.embedding_model, cache)
    save_cache(cache_path, cache)
    plot_from_cache(cache, plot_path)

    print(f"Updated cache: {cache_path}")
    print(f"Rendered plot: {plot_path}")
    print(f"Added baseline points: {len(records)}")


if __name__ == "__main__":
    main()
