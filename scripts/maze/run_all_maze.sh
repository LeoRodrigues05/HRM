#!/usr/bin/env bash
# Orchestrator for maze eval-only experiments (eval-only, RTX 5000 Ada budget).
# Run from repo root: bash scripts/maze/run_all_maze.sh

set -e
cd "$(dirname "$0")/../.."

CKPT="checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs/maze results/maze

echo "================================================================"
echo "[1/5] Baseline eval + step dynamics + z_H trajectory  (~3 min)"
echo "================================================================"
python scripts/maze/eval_step_dynamics_maze.py \
    --checkpoint "$CKPT" \
    --num_puzzles 200 \
    --max_steps 16 \
    --output_dir results/maze/step_dynamics 2>&1 | tee logs/maze/step_dynamics.log

echo "================================================================"
echo "[2/5] Controlled activation ablation z_H + z_L  (~10 min)"
echo "================================================================"
python scripts/controlled/controlled_ablation.py \
    --checkpoint "$CKPT" \
    --num_puzzles 30 \
    --max_steps 16 \
    --output_dir results/maze/ablation_controlled 2>&1 | tee logs/maze/ablation.log

echo "================================================================"
echo "[3/5] Activation patching with spatial groupings  (~5 min)"
echo "================================================================"
python scripts/maze/controlled_activation_patching_maze.py \
    --checkpoint "$CKPT" \
    --num_pairs 20 \
    --patch_steps 4,8,12 \
    --max_steps 16 \
    --output_dir results/maze/patching_spatial 2>&1 | tee logs/maze/patching.log

echo "================================================================"
echo "[4/5] Controlled time-shift  (~10 min)"
echo "================================================================"
python scripts/controlled/controlled_time_shift.py \
    --checkpoint "$CKPT" \
    --num_puzzles 15 \
    --max_steps 16 \
    --output_dir results/maze/time_shift_controlled 2>&1 | tee logs/maze/time_shift.log

echo "================================================================"
echo "[5/5] Controlled freeze  (~10 min)"
echo "================================================================"
python scripts/controlled/controlled_freeze.py \
    --checkpoint "$CKPT" \
    --num_puzzles 15 \
    --max_steps 16 \
    --output_dir results/maze/freeze_controlled 2>&1 | tee logs/maze/freeze.log

echo "================================================================"
echo "ALL DONE.  Aggregates:"
echo "  results/maze/step_dynamics/aggregate.json"
echo "  results/maze/ablation_controlled/aggregate.json"
echo "  results/maze/patching_spatial/aggregate.json"
echo "  results/maze/time_shift_controlled/aggregate.json"
echo "  results/maze/freeze_controlled/aggregate.json"
echo "================================================================"
