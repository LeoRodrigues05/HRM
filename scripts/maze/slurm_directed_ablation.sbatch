#!/usr/bin/env bash
# Finding 3 replication on maze: directed ablation of probe directions vs random
# controls, scored on path-validity. Runs BOTH streams (z_H then z_L) so we can
# state whether ANY maze probe direction is causally used.
# Submit:  sbatch -p gpu --gres=gpu:1 scripts/maze/slurm_directed_ablation.sbatch
#SBATCH -J maze_directed
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/maze/%x_%j.out
#SBATCH -e logs/maze/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/maze results/maze/hardened/directed_ablation

CHECKPOINT="${CHECKPOINT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
PROBE_W="${PROBE_W:-results/maze/hardened/probe_geometry/probe_weights.pt}"
NUM_PUZZLES="${NUM_PUZZLES:-500}"

for LVL in H L; do
  echo "[maze_directed] z_level=$LVL"
  conda run --no-capture-output -n hrm python -u scripts/maze/directed_ablation_maze.py \
    --checkpoint "$CHECKPOINT" \
    --num_puzzles "$NUM_PUZZLES" \
    --z_level "$LVL" \
    --probe_weights "$PROBE_W" \
    --output_dir "results/maze/hardened/directed_ablation/z${LVL}" \
    --device cuda
done
echo "[maze_directed] done"
