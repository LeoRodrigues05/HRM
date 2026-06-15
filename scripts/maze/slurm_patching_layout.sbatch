#!/usr/bin/env bash
#SBATCH -J maze_patch_layout
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -w ws-l5-029
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/maze/%x_%j.out
#SBATCH -e logs/maze/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/maze results/maze/patching_spatial_layout

CHECKPOINT="${CHECKPOINT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
DEVICE="${DEVICE:-cuda}"
NUM_PAIRS="${NUM_PAIRS:-10}"
MAX_STEPS="${MAX_STEPS:-16}"
PATCH_STEPS="${PATCH_STEPS:-1,2,4,8,12}"

echo "[maze_patch_layout] checkpoint=$CHECKPOINT"
echo "[maze_patch_layout] device=$DEVICE num_pairs=$NUM_PAIRS max_steps=$MAX_STEPS patch_steps=$PATCH_STEPS"

conda run --no-capture-output -n hrm python -u scripts/maze/controlled_activation_patching_maze.py \
  --checkpoint "$CHECKPOINT" \
  --num_pairs "$NUM_PAIRS" \
  --patch_steps "$PATCH_STEPS" \
  --max_steps "$MAX_STEPS" \
  --output_dir results/maze/patching_spatial_layout \
  --device "$DEVICE"

echo "[maze_patch_layout] done"
