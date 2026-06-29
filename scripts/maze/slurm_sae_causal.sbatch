#!/usr/bin/env bash
# Finding 3 replication on maze: SAE-feature causal ablation (path-validity).
# Conditions: top-50 SAE features vs random-50, probe directions vs random dirs.
# Uses the d=2048, l1=0.01 SAE to match the Sudoku config for cross-task parity.
# Submit:  sbatch -p gpu --gres=gpu:1 scripts/maze/slurm_sae_causal.sbatch
#SBATCH -J maze_sae_causal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/maze/%x_%j.out
#SBATCH -e logs/maze/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/maze results/maze/sae_study/causal_ablation

CHECKPOINT="${CHECKPOINT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
SAE_PATH="${SAE_PATH:-results/maze/sae_study/sae_d2048_l10.01.pt}"
ACTS="${ACTS:-results/maze/sae_study/activations_zH.pt}"
PROBE_W="${PROBE_W:-results/maze/hardened/probe_geometry/probe_weights.pt}"
N_PUZZLES="${N_PUZZLES:-300}"
TOP_K="${TOP_K:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-results/maze/sae_study/causal_ablation}"

echo "[maze_sae_causal] sae=$SAE_PATH n_puzzles=$N_PUZZLES top_k=$TOP_K"
conda run --no-capture-output -n hrm python -u scripts/maze/sae_causal_ablation_maze.py \
  --checkpoint "$CHECKPOINT" \
  --sae_path "$SAE_PATH" \
  --activations_path "$ACTS" \
  --probe_weights "$PROBE_W" \
  --n_puzzles "$N_PUZZLES" \
  --top_k "$TOP_K" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda
echo "[maze_sae_causal] done"
