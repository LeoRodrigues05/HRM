#!/usr/bin/env bash
# Phase 0 GPU driver — SAE-priority batch (fits a short allocation).
# Runs the genuinely-missing / stale-cheap steps only:
#   - SAE d x l1 sweep (THE missing gap; trains on cached activations)
#   - SAE causal ablation (canonical d2048 l0.01, N=300)
#   - Baseline HRM eval (N=500, bootstrap CIs)
# Skips controlled ablation/freeze/time-shift (already adequate w/ CIs).
#
# Run INSIDE a GPU allocation:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrm
#   nohup bash scripts/bash/run_phase0_sae.sh > logs/phase0/driver_sae.log 2>&1 &
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

# 1) SAE d x l1 sweep (Goldilocks search) — the genuine missing gap
run "SAE sweep d x l1" "$LOG/sae_sweep.log" \
  python scripts/sae/sae_sweep.py \
    --dict_sizes 1024,2048,4096,8192 \
    --l1_coeffs 0.003,0.01,0.03 \
    --activations_path results/sae_study/activations_zH.pt \
    --device $DEV --output_dir results/sae_study

# 2) SAE causal ablation on the canonical config (d=2048, l1=0.01)
run "SAE causal ablation (d2048 l0.01, N=300)" "$LOG/sae_causal.log" \
  python scripts/sae/sae_causal_ablation.py \
    --sae_path results/sae_study/sae_d2048_l10.01.pt \
    --probe_weights_path results/probes/e8_constraint_probes/probe_weights.pt \
    --n_puzzles 300 --top_k 50 --device $DEV \
    --output_dir results/sae_study/causal_ablation

# 3) Baseline recurrence eval — re-run HRM with bootstrap CIs
run "Baseline eval HRM (N=500)" "$LOG/baseline_HRM.log" \
  python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/sapientinc-sudoku-extreme/checkpoint.pt \
    --model_name HRM --n_puzzles 500 --device $DEV \
    --output_dir results/baseline_comparison

echo "ALL PHASE 0 SAE STEPS DONE"
