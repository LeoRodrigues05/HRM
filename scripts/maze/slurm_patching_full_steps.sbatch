#!/usr/bin/env bash
# Issue #2 fix: full cross-puzzle patch INCLUDING the readout step (15).
# The hardened patching_spatial run only covered steps 4,8,12, so the claim
# "mid-step foreign z_H is rebuildable but the FINAL z_H is load-bearing" was
# never tested by patching. This re-runs the same validated engine over
# steps {0,4,8,12,14,15}; read off the `all` group (full-grid swap) for H/L.
# Submit:  sbatch -p gpu --gres=gpu:1 scripts/maze/slurm_patching_full_steps.sbatch
#SBATCH -J maze_patch_full
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/maze/%x_%j.out
#SBATCH -e logs/maze/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/maze results/maze/hardened/patching_full_steps

CHECKPOINT="${CHECKPOINT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
NUM_PAIRS="${NUM_PAIRS:-100}"
PATCH_STEPS="${PATCH_STEPS:-0,4,8,12,14,15}"
OUTPUT_DIR="${OUTPUT_DIR:-results/maze/hardened/patching_full_steps}"

echo "[maze_patch_full] checkpoint=$CHECKPOINT num_pairs=$NUM_PAIRS patch_steps=$PATCH_STEPS"
conda run --no-capture-output -n hrm python -u scripts/maze/controlled_activation_patching_maze.py \
  --checkpoint "$CHECKPOINT" \
  --num_pairs "$NUM_PAIRS" \
  --patch_steps "$PATCH_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda
echo "[maze_patch_full] done"
