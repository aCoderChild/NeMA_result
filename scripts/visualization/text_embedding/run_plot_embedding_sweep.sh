#!/usr/bin/env bash
#
# Batch-run plot_embedding_with_full.py over several UMAP/t-SNE random seeds.
#
# Rules baked in (per your request):
#   - Main chosen always includes icr; lacomsa is optional (no mapo in main sides).
#   - --methods is lacomsa,icr or icr only (never mapo) so baseline + Gemini tie-break stay consistent.
#
# Configuration: set env vars before calling, e.g.
#   NUM_SEEDS=5 SEED_START=42 INCLUDE_LACOMSA=1 USE_GEMINI=0 ./run_plot_embedding_sweep.sh
#   SEEDS_CSV="13,19,42,51,99" INCLUDE_LACOMSA=0 ./run_plot_embedding_sweep.sh   # overrides NUM_SEEDS/SEED_START
#
set -euo pipefail

ROOT="${ROOT:-/home/gangstat/NeMA_result}"
PY="${PLOT_SCRIPT:-${ROOT}/scripts/visualization/text_embedding/plot_embedding_with_full.py}"

# --- sweep knobs (override via environment) ---
NUM_SEEDS="${NUM_SEEDS:-5}"
SEED_START="${SEED_START:-42}"
# If non-empty, comma-separated ints; overrides NUM_SEEDS / SEED_START length and values.
SEEDS_CSV="${SEEDS_CSV:-}"

# 0 = main chosen/rejected only icr; 1 = lacomsa,icr (still no mapo)
INCLUDE_LACOMSA="${INCLUDE_LACOMSA:-0}"
# 1 = load Gemini cache; 0 = --skip-gemini
USE_GEMINI="${USE_GEMINI:-1}"
# If 1, run each seed twice (with and without Gemini); filenames get _g1 / _g0.
RUN_BOTH_GEMINI="${RUN_BOTH_GEMINI:-0}"
# If 1, run each (seed × gemini) combo twice: icr-only and icr+lacomsa mains; filenames get _mainicr / _mainicr_lacomsa
RUN_BOTH_LACOMSA="${RUN_BOTH_LACOMSA:-0}"

LANGS="${LANGS:-ru}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
MAIN_CACHE="${MAIN_CACHE:-${ROOT}/analysis/embeddings/plot_same_instruction_es_ru_cache.json}"
GEMINI_CACHE="${GEMINI_CACHE:-${ROOT}/analysis/embeddings/gemini_same_instruction_es_ru_cache.json}"
EXTRA_CKPTS="${EXTRA_CKPTS:-${ROOT}/analysis/embeddings/icr_npo_ultrafeedback_first_es_ru_cache.json}"
POLICY_CACHE="${POLICY_CACHE:-${ROOT}/analysis/embeddings/icr_policy_rollouts_ppo_reinforce_ru_cache.json}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/analysis/figures/12_05_2026}"

# Extra CLI flags passed to every run (space-separated tokens).
EXTRA_FLAGS="${EXTRA_FLAGS:---show-chosen-hull}"

UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"
TSNE_PERPLEXITY="${TSNE_PERPLEXITY:-5.0}"

# --- build seed list ---
SEEDS=()
if [[ -n "${SEEDS_CSV// /}" ]]; then
  IFS=',' read -ra _parts <<< "${SEEDS_CSV// /}"
  for x in "${_parts[@]}"; do
    [[ -z "${x// /}" ]] && continue
    SEEDS+=("${x// /}")
  done
else
  for ((i = 0; i < NUM_SEEDS; i++)); do
    SEEDS+=("$((SEED_START + i))")
  done
fi

if ((${#SEEDS[@]} == 0)); then
  echo "error: no seeds (set SEEDS_CSV or NUM_SEEDS/SEED_START)" >&2
  exit 1
fi

lang_tag="${LANGS//,/_}"

run_one() {
  local seed="$1"
  local with_gemini="$2"   # 1 or 0
  local with_lacomsa="$3" # 1 or 0

  local methods main_ch main_rj gem_flag gem_slug main_slug out

  if [[ "$with_lacomsa" -eq 1 ]]; then
    methods="lacomsa,icr"
    main_ch="lacomsa,icr"
    main_rj="lacomsa,icr"
    main_slug="icr_lacomsa"
  else
    methods="icr"
    main_ch="icr"
    main_rj="icr"
    main_slug="icr"
  fi

  if [[ "$with_gemini" -eq 1 ]]; then
    gem_flag=()
    gem_slug="g1"
  else
    gem_flag=(--skip-gemini)
    gem_slug="g0"
  fi

  out="${OUTPUT_DIR}/seed_${seed}_${gem_slug}_${main_slug}_plot_embedding_with_full_umap_tsne_${lang_tag}.png"

  mkdir -p "${OUTPUT_DIR}"

  echo "==> seed=${seed} gemini=${with_gemini} lacomsa_in_main=${with_lacomsa} -> ${out}"

  # shellcheck disable=SC2086
  python3 "${PY}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --main-cache-path "${MAIN_CACHE}" \
    --gemini-cache-path "${GEMINI_CACHE}" \
    --extra-checkpoint-cache-paths "${EXTRA_CKPTS}" \
    --policy-rollout-cache-path "${POLICY_CACHE}" \
    --methods "${methods}" \
    --main-chosen-methods "${main_ch}" \
    --main-rejected-methods "${main_rj}" \
    --langs "${LANGS}" \
    --plot-path "${out}" \
    --umap-n-neighbors "${UMAP_N_NEIGHBORS}" \
    --umap-min-dist "${UMAP_MIN_DIST}" \
    --umap-random-state "${seed}" \
    --tsne-perplexity "${TSNE_PERPLEXITY}" \
    --tsne-random-state "${seed}" \
    "${gem_flag[@]}" \
    ${EXTRA_FLAGS}
}

echo "Sweep: ${#SEEDS[@]} seeds -> ${SEEDS[*]}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

for seed in "${SEEDS[@]}"; do
  gem_modes=()
  if [[ "${RUN_BOTH_GEMINI}" -eq 1 ]]; then
    gem_modes=(1 0)
  else
    gem_modes=("${USE_GEMINI}")
  fi

  lac_modes=()
  if [[ "${RUN_BOTH_LACOMSA}" -eq 1 ]]; then
    lac_modes=(0 1)
  else
    lac_modes=("${INCLUDE_LACOMSA}")
  fi

  for g in "${gem_modes[@]}"; do
    for l in "${lac_modes[@]}"; do
      run_one "${seed}" "${g}" "${l}"
    done
  done
done

echo "Done."
