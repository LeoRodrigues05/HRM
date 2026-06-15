#!/usr/bin/env bash
#SBATCH -J maze_probes_hard
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
mkdir -p logs/maze results/maze/hardened/linear_probes

CHECKPOINT="${CHECKPOINT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
DEVICE="${DEVICE:-cuda}"
NUM_PUZZLES="${NUM_PUZZLES:-32}"
MAX_STEPS="${MAX_STEPS:-4}"
EPOCHS="${EPOCHS:-30}"
POSITIONS_PER_SAMPLE="${POSITIONS_PER_SAMPLE:-128}"
SEEDS="${SEEDS:-0,1,2,3,4}"
OUTPUT_DIR="${OUTPUT_DIR:-results/maze/hardened/linear_probes}"

echo "[maze_probes_hardened] checkpoint=$CHECKPOINT"
echo "[maze_probes_hardened] device=$DEVICE num_puzzles=$NUM_PUZZLES max_steps=$MAX_STEPS epochs=$EPOCHS"
echo "[maze_probes_hardened] positions_per_sample=$POSITIONS_PER_SAMPLE seeds=$SEEDS"
echo "[maze_probes_hardened] output_dir=$OUTPUT_DIR"

conda run --no-capture-output -n hrm python -u scripts/maze/linear_probes_maze.py \
  --checkpoint "$CHECKPOINT" \
  --num_puzzles "$NUM_PUZZLES" \
  --max_steps "$MAX_STEPS" \
  --epochs "$EPOCHS" \
  --positions_per_sample "$POSITIONS_PER_SAMPLE" \
  --seeds "$SEEDS" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --train_device auto

echo "[maze_probes_hardened] done"
