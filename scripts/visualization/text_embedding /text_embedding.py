import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


def get_text_embeddings(texts, model="text-embedding-3-small"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts)
    return np.array([row.embedding for row in response.data], dtype=np.float64)


def load_first_n_pairs(jsonl_path, n_samples):
    texts = []
    response_types = []
    labels = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= n_samples:
                break

            row = json.loads(line)
            chosen = row.get("chosen", "").strip()
            rejected = row.get("rejected", "").strip()

            if chosen:
                texts.append(chosen)
                response_types.append("chosen")
                labels.append(f"{idx}_chosen")

            if rejected:
                texts.append(rejected)
                response_types.append("rejected")
                labels.append(f"{idx}_rejected")

    if not texts:
        raise ValueError(f"No chosen/rejected texts found in: {jsonl_path}")

    return texts, response_types, labels


def reduce_embeddings(embeddings):
    reductions = {}
    n_samples = embeddings.shape[0]

    # PCA is deterministic and stable even for tiny sample counts.
    reductions["PCA"] = PCA(n_components=2).fit_transform(embeddings)

    # MDS works nicely from pairwise distances; with 2 points it still produces 2D output.
    reductions["MDS"] = MDS(
        n_components=2,
        random_state=42,
        dissimilarity="euclidean",
        normalized_stress="auto",
    ).fit_transform(embeddings)

    # t-SNE requires perplexity < n_samples; guard small sample sizes.
    if n_samples >= 2:
        perplexity = max(1, min(5, n_samples - 1))
        reductions["t-SNE"] = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="random",
            learning_rate="auto",
        ).fit_transform(embeddings)

    return reductions


def plot_reductions(reductions, response_types, output_path, chosen_color, rejected_color):
    n_plots = len(reductions)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes.ravel()

    color_map = {"chosen": chosen_color, "rejected": rejected_color}

    for idx, (name, points) in enumerate(reductions.items()):
        ax = axes[idx]
        colors = [color_map.get(t, "gray") for t in response_types]
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=40, alpha=0.8)

        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)
        ax.scatter([], [], c=chosen_color, label="chosen")
        ax.scatter([], [], c=rejected_color, label="rejected")
        ax.legend(loc="best")

    fig.suptitle("Chosen vs Rejected Response Embeddings", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed chosen/rejected responses and plot DR projections."
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="/home/gangstat/NeMA_result/_pairs/train_lacomsa_relabeled.jsonl",
        help="Path to JSONL file with chosen/rejected fields.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=40,
        help="Number of first rows to read from JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/figures/text_embedding_dr.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--chosen-color",
        type=str,
        default="green",
        help="Color for chosen responses.",
    )
    parser.add_argument(
        "--rejected-color",
        type=str,
        default="red",
        help="Color for rejected responses.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl_path)
    texts, response_types, _labels = load_first_n_pairs(jsonl_path, args.n_samples)

    embeddings = get_text_embeddings(texts)
    reductions = reduce_embeddings(embeddings)
    output_path = Path(args.output)
    plot_reductions(
        reductions,
        response_types,
        output_path,
        chosen_color=args.chosen_color,
        rejected_color=args.rejected_color,
    )

    print(f"Saved plot to: {output_path} (samples={args.n_samples}, points={len(texts)})")


if __name__ == "__main__":
    main()

