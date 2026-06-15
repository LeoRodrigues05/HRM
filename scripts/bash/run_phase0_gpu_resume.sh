#!/usr/bin/env bash
# Phase 0 GPU driver — RESUME from controlled ablation onward.
# Skips E8 probes and controlled directed ablation (already completed fresh today).
#
# Run this INSIDE a GPU allocation:
#     source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrm
#     nohup bash scripts/bash/run_phase0_gpu_resume.sh > logs/phase0/driver_resume.log 2>&1 &
#
# Each step logs to logs/phase0/<step>.log and writes _meta.json provenance.
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/phase0
LOG=logs/phase0
DEV=cuda

run() {  # run <label> <logfile> <cmd...>
  local label="$1"; local logf="$2"; shift 2
  echo "============================================================"
  echo ">>> $label"
  echo ">>> $*"
  echo "============================================================"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?
  echo "<<< $label finished rc=$rc in $((SECONDS - t0))s (log: $logf)"
  return $rc
}

# 3) Controlled ablation / freeze / time-shift (provenance + CIs)
run "Controlled z_H/z_L ablation (N=5000)" "$LOG/controlled_ablation.log" \
  python scripts/controlled/controlled_ablation.py \
    --num_puzzles 5000 --device $DEV \
    --output_dir results/controlled/ablation

run "Controlled freeze (N=1000)" "$LOG/controlled_freeze.log" \
  python scripts/controlled/controlled_freeze.py \
    --num_puzzles 1000 --device $DEV \
    --output_dir results/controlled/freeze

run "Controlled time-shift (N=1000, fixed)" "$LOG/controlled_timeshift.log" \
  python scripts/controlled/controlled_time_shift.py \
    --mode fixed --num_puzzles 1000 --device $DEV \
    --output_dir results/controlled/time_shift

# 4) SAE d x l1 sweep (Goldilocks search)
run "SAE sweep d x l1" "$LOG/sae_sweep.log" \
  python scripts/sae/sae_sweep.py \
    --dict_sizes 1024,2048,4096,8192 \
    --l1_coeffs 0.003,0.01,0.03 \
    --activations_path results/sae_study/activations_zH.pt \
    --device $DEV --output_dir results/sae_study

# 5) SAE causal ablation on the canonical config (d=2048, l1=0.01)
run "SAE causal ablation (d2048 l0.01, N=300)" "$LOG/sae_causal.log" \
  python scripts/sae/sae_causal_ablation.py \
    --sae_path results/sae_study/sae_d2048_l10.01.pt \
    --probe_weights_path results/probes/e8_constraint_probes/probe_weights.pt \
    --n_puzzles 300 --top_k 50 --device $DEV \
    --output_dir results/sae_study/causal_ablation

# 6) Baseline recurrence eval — re-run HRM with bootstrap CIs
declare -A CKPTS=(
  [HRM]="checkpoints/sapientinc-sudoku-extreme/checkpoint.pt"
)
for name in "${!CKPTS[@]}"; do
  ck="${CKPTS[$name]}"
  [ -f "$ck" ] || { echo "skip $name (no ckpt $ck)"; continue; }
  run "Baseline eval $name" "$LOG/baseline_${name}.log" \
    python scripts/analysis/evaluate_baselines.py \
      --checkpoint "$ck" --model_name "$name" \
      --n_puzzles 500 --device $DEV \
      --output_dir results/baseline_comparison
done

echo "ALL PHASE 0 RESUME STEPS DONE"
