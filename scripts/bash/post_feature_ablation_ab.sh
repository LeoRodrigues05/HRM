#!/bin/bash
#SBATCH --job-name=hrm_sae_ablate
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/hrm_sae_ablate_%j.out
#SBATCH --error=logs/hrm_sae_ablate_%j.err

# NOTE on scheduling: keep mem/cpu SMALL so the job stays backfillable. The work
# only touches a ~2.7 GB activation tensor + a ~110 MB model, so 48G/16cpu is ample;
# a large request (e.g. mem=180G) cannot backfill into the partial-node gaps that
# are usually all that's free on cscc-gpu-p, leaving the job stuck at "(Resources)"
# behind higher-priority reservations even when GPUs are physically idle.
# TIME: the causal ablation alone is ~70 min at N_PUZZLES=200 (~20 s/puzzle), so the
# old 1h wall killed it mid-run. We ask for 2h: both causal tasks run concurrently
# (one per GPU, see task ordering below), finishing in ~75 min wall with margin.

# Follow-up to the SAE A/B sweep: the mechanistic stock-vs-bptt comparison.
# For BOTH trained HRMs, on the Pareto-best SAE from the sweep, run:
#   1) feature analysis  (specialization, step profiles, geometry)  -- no model
#   2) causal ablation   (does zeroing top SAE features hurt solving vs random?)
#      -- runs the SAE's OWN HRM forward (patched --checkpoint_dir), so stock is
#         tested on the stock model and bptt on the bptt model.
#   3) plots              (per model)
#
# PARALLELISM: the 4 independent tasks (2 models x {analyze, causal}) are scheduled
# in waves sized to however many GPUs Slurm actually gave us (SLURM_GPUS_ON_NODE).
# We request gpu:2 by default because cscc-gpu-p is often fragmented and a whole
# free 4-GPU node can be unschedulable for a long time (reason "(Resources)"),
# whereas 2 free GPUs on a node are usually available immediately. With 2 GPUs the
# job runs 2 tasks at a time; with 4 (sbatch --gres=gpu:4) it runs all at once;
# with 1 it runs serially. Either way it adapts — no edit needed.
#   To use more/fewer GPUs:  sbatch --gres=gpu:4 scripts/bash/post_feature_ablation_ab.sh
#
# Knobs (env overrides):
#   SAE_TAG=d2048_l10.003   # which SAE to analyze (Pareto-best baseline from sweep)
#   N_PUZZLES=200           # causal-ablation puzzles (smaller = faster)
#   TOP_K=50                # top SAE features to ablate

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
source scripts/bash/_activate_env.sh
export DISABLE_COMPILE=1

SAE_TAG="${SAE_TAG:-d2048_l10.003}"
N_PUZZLES="${N_PUZZLES:-200}"
TOP_K="${TOP_K:-50}"
ROOT="results/sae_study/bptt_study"

# Models with a trained SAE present.
MODELS=()
for name in hrm_stock hrm_bptt; do
    SAE="${ROOT}/${name}/sae_${SAE_TAG}.pt"
    ACT="${ROOT}/${name}/activations_zH.pt"
    if [ -f "${SAE}" ] && [ -f "${ACT}" ]; then
        MODELS+=("${name}")
    else
        echo "WARN: missing ${SAE} or ${ACT} — skipping ${name}"
    fi
done
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "ERROR: no (SAE, activations) pairs found under ${ROOT} for tag ${SAE_TAG}." >&2
    echo "       Run scripts/bash/post_collect_and_sae_ab.sh first." >&2
    exit 1
fi

JID="${SLURM_JOB_ID:-local}"

# How many GPUs did we actually get? Adapt the wave size to it.
NGPU="${SLURM_GPUS_ON_NODE:-}"
[ -z "${NGPU}" ] && NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ -z "${NGPU}" ] || [ "${NGPU}" -lt 1 ] && NGPU=1
echo "Detected ${NGPU} GPU(s); scheduling tasks in waves of ${NGPU}."

# Run one task (analyze|causal) for one model, pinned to a GPU index.
run_task() {
    local name="$1" kind="$2" gpu="$3"
    local SAE="${ROOT}/${name}/sae_${SAE_TAG}.pt"
    local ACT="${ROOT}/${name}/activations_zH.pt"
    if [ "${kind}" = "analyze" ]; then
        python scripts/sae/sae_analyze_features.py \
            --sae_path "${SAE}" \
            --activations_path "${ACT}" \
            --output_dir "${ROOT}/${name}/feature_analysis" \
            --device "cuda:${gpu}" \
            > "logs/sae_analyze_${name}_${JID}.log" 2>&1
    else
        python scripts/sae/sae_causal_ablation.py \
            --sae_path "${SAE}" \
            --checkpoint_dir "checkpoints/bptt_study/${name}" \
            --activations_path "${ACT}" \
            --n_puzzles "${N_PUZZLES}" \
            --top_k "${TOP_K}" \
            --output_dir "${ROOT}/${name}/causal_ablation" \
            --save_puzzle_indices "${ROOT}/${name}/causal_ablation/puzzle_indices.json" \
            --device "cuda:${gpu}" \
            > "logs/sae_causal_${name}_${JID}.log" 2>&1
    fi
}

# Build the task list grouped by KIND: all the light analyze tasks first, then all
# the heavy causal tasks. With a 2-GPU wave this runs both analyses together (~2 min)
# and THEN both causal ablations together (~70 min each, in parallel), so the job
# finishes in ~75 min wall instead of ~140 min when a slow causal task would share
# its wave with an idle GPU. (At gpu:1 it falls back to serial; at gpu:4, all at once.)
TASKS=()
for name in "${MODELS[@]}"; do TASKS+=("${name}:analyze"); done
for name in "${MODELS[@]}"; do TASKS+=("${name}:causal"); done

# ── Phase 1: analyze + causal, scheduled in waves of NGPU ────────────────────
echo "=== Phase 1: feature analysis + causal ablation on $(date) ==="
fail=0
i=0
while [ "${i}" -lt "${#TASKS[@]}" ]; do
    PIDS=(); LABELS=()
    g=0
    while [ "${g}" -lt "${NGPU}" ] && [ "${i}" -lt "${#TASKS[@]}" ]; do
        name="${TASKS[$i]%%:*}"; kind="${TASKS[$i]##*:}"
        echo "  [${kind}] ${name} -> cuda:${g}"
        run_task "${name}" "${kind}" "${g}" &
        PIDS+=($!); LABELS+=("${TASKS[$i]}")
        g=$((g + 1)); i=$((i + 1))
    done
    for j in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$j]}"; then
            echo "ERROR: ${LABELS[$j]} failed (see logs/sae_*_${JID}.log)" >&2
            fail=1
        fi
    done
done
[ "${fail}" -ne 0 ] && exit 1
echo "=== Phase 1 done on $(date) ==="

# ── Phase 2: plots per model (cheap; serial) ─────────────────────────────────
echo "=== Phase 2: plots on $(date) ==="
for name in "${MODELS[@]}"; do
    python scripts/sae/sae_plot.py \
        --sae_path "${ROOT}/${name}/sae_${SAE_TAG}.pt" \
        --activations_path "${ROOT}/${name}/activations_zH.pt" \
        --analysis_dir "${ROOT}/${name}/feature_analysis" \
        --causal_dir "${ROOT}/${name}/causal_ablation" \
        --sweep_csv "${ROOT}/${name}/sweep_results.csv" \
        --output_dir "${ROOT}/${name}/plots" \
        --device cuda:0 \
        > "logs/sae_plot_${name}_${JID}.log" 2>&1 \
        || echo "WARN: plotting failed for ${name} (non-fatal; see logs/sae_plot_${name}_${JID}.log)"
done

echo "Ablation follow-up complete at $(date)."
echo "Compare causal effect:  ${ROOT}/{hrm_stock,hrm_bptt}/causal_ablation/aggregate.json"
echo "Compare features:       ${ROOT}/{hrm_stock,hrm_bptt}/feature_analysis/specialization_summary.json"
