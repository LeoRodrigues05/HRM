#!/bin/bash
#SBATCH --job-name=hrm_sae_ab
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/hrm_sae_ab_%j.out
#SBATCH --error=logs/hrm_sae_ab_%j.err

# Post step (recommended order #3): for BOTH trained HRMs (stock + bptt),
# collect z_H activations and run the SAE A/B sweep (baseline vs mean-centered)
# across dict_size x l1. Produces the (stock|bptt) x (baseline|mean-centered)
# table on dead_count / L0 / recon / gamma.
#
# PARALLELISM (this node has 4x A100-40GB; cscc-gpu-qos caps the user at 4 GPUs):
#   Phase 1 (collect): stock on cuda:0, bptt on cuda:1 — concurrently (2 GPUs).
#   Phase 2 (sweep):   the A/B sweep is 2 models x 2 centering modes x (3 dict x
#                      3 l1) = 36 independent SAE trainings. We split it into 4
#                      quadrants (model x centering), one GPU each = 9 trainings
#                      per GPU, all 4 GPUs busy. This is what turns the ~4h serial
#                      job into ~1h.
#   Phase 3 (merge):   recombine each model's baseline+centered shards into the
#                      original layout (results/sae_study/bptt_study/<name>/).
#
# Per-SAE output files carry a "_mc" suffix when centered, so baseline/centered
# never collide; only sweep_results.csv/_meta.json would, hence the shard dirs.

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
source scripts/bash/_activate_env.sh
export DISABLE_COMPILE=1

N_PUZZLES="${N_PUZZLES:-1000}"
DICT_SIZES="${DICT_SIZES:-1024,2048,4096}"
L1_COEFFS="${L1_COEFFS:-0.003,0.01,0.03}"
ROOT="results/sae_study/bptt_study"

# Models present? Build the active list (skip any missing checkpoint dir).
MODELS=()
for name in hrm_stock hrm_bptt; do
    if [ -d "checkpoints/bptt_study/${name}" ]; then
        MODELS+=("${name}")
    else
        echo "WARN: checkpoints/bptt_study/${name} not found — skipping ${name}"
    fi
done
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "ERROR: no trained HRM checkpoints under checkpoints/bptt_study — nothing to do." >&2
    exit 1
fi

# ── Phase 1: collect z_H activations (one GPU per model, in parallel) ─────────
echo "=== Phase 1: collecting activations on $(date) ==="
gpu=0
declare -A COLLECT_PID
for name in "${MODELS[@]}"; do
    OUT="${ROOT}/${name}"
    mkdir -p "${OUT}"
    echo "  [collect] ${name} -> cuda:${gpu} -> ${OUT}/activations_zH.pt"
    python scripts/sae/sae_collect_activations.py \
        --n_puzzles "${N_PUZZLES}" \
        --checkpoint_dir "checkpoints/bptt_study/${name}" \
        --output_dir "${OUT}" \
        --device "cuda:${gpu}" \
        > "logs/sae_collect_${name}_${SLURM_JOB_ID:-local}.log" 2>&1 &
    COLLECT_PID[$name]=$!
    gpu=$((gpu + 1))
done
fail=0
for name in "${MODELS[@]}"; do
    if ! wait "${COLLECT_PID[$name]}"; then
        echo "ERROR: activation collection failed for ${name} (see logs/sae_collect_${name}_*.log)" >&2
        fail=1
    fi
done
[ "${fail}" -ne 0 ] && exit 1
echo "=== Phase 1 done on $(date) ==="

# ── Phase 2: SAE sweep, 4 quadrants (model x centering), one GPU each ─────────
echo "=== Phase 2: SAE A/B sweep (4-way parallel) on $(date) ==="
gpu=0
PIDS=()
LABELS=()
for name in "${MODELS[@]}"; do
    for mode in baseline centered; do
        OUT="${ROOT}/${name}"
        SHARD="${OUT}/_shard_${mode}"
        mkdir -p "${SHARD}"
        CENTER_FLAG=""
        [ "${mode}" = "centered" ] && CENTER_FLAG="--center_mean"
        echo "  [sweep] ${name}/${mode} -> cuda:${gpu} -> ${SHARD}"
        python scripts/sae/sae_sweep.py \
            ${CENTER_FLAG} \
            --activations_path "${OUT}/activations_zH.pt" \
            --output_dir "${SHARD}" \
            --dict_sizes "${DICT_SIZES}" \
            --l1_coeffs "${L1_COEFFS}" \
            --device "cuda:${gpu}" \
            > "logs/sae_sweep_${name}_${mode}_${SLURM_JOB_ID:-local}.log" 2>&1 &
        PIDS+=($!)
        LABELS+=("${name}/${mode}")
        gpu=$((gpu + 1))
    done
done
fail=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: sweep failed for ${LABELS[$i]} (see logs/sae_sweep_*_${SLURM_JOB_ID:-local}.log)" >&2
        fail=1
    fi
done
[ "${fail}" -ne 0 ] && exit 1
echo "=== Phase 2 done on $(date) ==="

# ── Phase 3: merge each model's baseline+centered shards into <name>/ ─────────
echo "=== Phase 3: merging shards on $(date) ==="
for name in "${MODELS[@]}"; do
    OUT="${ROOT}/${name}"
    # Move per-SAE artifacts up (names carry _mc for centered, so no collision).
    for mode in baseline centered; do
        SHARD="${OUT}/_shard_${mode}"
        [ -d "${SHARD}" ] || continue
        find "${SHARD}" -maxdepth 1 -type f ! -name "sweep_results.csv" ! -name "_meta.json" \
            -exec mv -f {} "${OUT}/" \;
    done
    # Concatenate the two sweep CSVs (header once) into the model's combined table.
    python - "$OUT" <<'PY'
import csv, os, sys
out = sys.argv[1]
rows, fieldnames = [], None
for mode in ("baseline", "centered"):
    p = os.path.join(out, f"_shard_{mode}", "sweep_results.csv")
    if not os.path.exists(p):
        continue
    with open(p) as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        rows.extend(r)
if rows:
    with open(os.path.join(out, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"  merged {len(rows)} rows -> {os.path.join(out,'sweep_results.csv')}")
PY
    rm -rf "${OUT}/_shard_baseline" "${OUT}/_shard_centered"
done

echo "Post step complete at $(date). Compare ${ROOT}/*/sweep_results.csv"
