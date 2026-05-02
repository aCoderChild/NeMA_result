#!/usr/bin/env bash
# Usage:
#   export OPENAI_API_KEY=...
#   bash run_instruction_embed_and_umap.sh 1
#   bash run_instruction_embed_and_umap.sh 2

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ID="${1:?usage: $0 <sample_id e.g. 1 or 2>}"
CACHE="${IC_RESULT_CACHE:-/home/gangstat/NeMA_result/analysis/embeddings/icr_id${SAMPLE_ID}_embeddings_with_instruction_cache.json}"
PLOT="${IC_RESULT_UMAP:-/home/gangstat/NeMA_result/analysis/figures/icr_id${SAMPLE_ID}_embeddings_umap_instruction.png}"

: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"

python "$DIR/embedding_with_instruction.py" \
  --sample-id "$SAMPLE_ID" \
  --cache-path "$CACHE"

python "$DIR/plot_embedding_umap.py" \
  --cache-path "$CACHE" \
  --plot-path "$PLOT"

echo "Done. cache=$CACHE plot=$PLOT"
