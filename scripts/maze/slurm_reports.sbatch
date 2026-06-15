#!/usr/bin/env bash
#SBATCH -J maze_reports
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
mkdir -p logs/maze results/maze/reports

CHECKPOINT="${CHECKPOINT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
DEVICE="${DEVICE:-cuda}"
MAX_STEPS="${MAX_STEPS:-16}"

echo "[maze_reports] checkpoint=$CHECKPOINT"
echo "[maze_reports] device=$DEVICE max_steps=$MAX_STEPS"

# Success report was generated during the interactive audit, but keep it here
# so the full report set is reproducible from SLURM.
conda run --no-capture-output -n hrm python -u scripts/maze/intermediate_substeps_maze.py \
  --checkpoint "$CHECKPOINT" \
  --puzzle_idx 0 \
  --num_pool 1 \
  --max_steps "$MAX_STEPS" \
  --device "$DEVICE" \
  --output results/maze/reports/intermediate_substeps_success_p0.html

conda run --no-capture-output -n hrm python -u scripts/maze/intermediate_substeps_maze.py \
  --checkpoint "$CHECKPOINT" \
  --puzzle_idx 4 \
  --num_pool 5 \
  --max_steps "$MAX_STEPS" \
  --device "$DEVICE" \
  --output results/maze/reports/intermediate_substeps_failed_p4.html

conda run --no-capture-output -n hrm python -u scripts/maze/layout_patching_report_maze.py \
  --checkpoint "$CHECKPOINT" \
  --src_idx 2 \
  --tgt_idx 3 \
  --patch_level H \
  --patch_steps 1 \
  --patch_group all \
  --max_steps "$MAX_STEPS" \
  --device "$DEVICE" \
  --output results/maze/reports/layout_patching_p2_to_p3_H_step1.html

conda run --no-capture-output -n hrm python -u scripts/maze/patching_report_maze.py \
  --checkpoint "$CHECKPOINT" \
  --src_idx 2 \
  --tgt_idx 3 \
  --patch_level H \
  --patch_steps 1 \
  --patch_group all \
  --max_steps "$MAX_STEPS" \
  --device "$DEVICE" \
  --output results/maze/reports/patching_p2_to_p3_H_step1.html

echo "[maze_reports] done"
