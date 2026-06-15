#!/usr/bin/env bash
# Hardened orchestrator for maze interpretability experiments.
# Raises N for publication-grade statistics and (critically) records
# maze PATH metrics (valid_sg_path, path_f1, ...) in every aggregate,
# not just wall-dominated token accuracy.
#
# Writes to results/maze/hardened/<exp>/ so the stale May-27 results
# under results/maze/<exp>/ are preserved for comparison.
#
# Run from repo root:  bash scripts/maze/run_all_maze_hardened.sh
set -e
cd "$(dirname "$0")/../.."

CKPT="checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT="results/maze/hardened"
LOG="logs/maze/hardened"
mkdir -p "$LOG" "$OUT"

echo "################################################################"
echo "# HARDENED MAZE SUITE  ($(date))"
echo "# checkpoint=$CKPT"
echo "# out=$OUT"
echo "################################################################"

echo "================================================================"
echo "[1/5] Step dynamics + z_H trajectory   (N=500)"
echo "================================================================"
python scripts/maze/eval_step_dynamics_maze.py \
    --checkpoint "$CKPT" \
    --num_puzzles 500 \
    --max_steps 16 \
    --output_dir "$OUT/step_dynamics" 2>&1 | tee "$LOG/step_dynamics.log"

echo "================================================================"
echo "[2/5] Controlled ablation z_H + z_L  (N=200, PATH metrics)"
echo "================================================================"
python scripts/controlled/controlled_ablation.py \
    --checkpoint "$CKPT" \
    --num_puzzles 200 \
    --max_steps 16 \
    --output_dir "$OUT/ablation_controlled" 2>&1 | tee "$LOG/ablation.log"

echo "================================================================"
echo "[3/5] Activation patching (spatial groupings)  (N=100 pairs)"
echo "================================================================"
python scripts/maze/controlled_activation_patching_maze.py \
    --checkpoint "$CKPT" \
    --num_pairs 100 \
    --patch_steps 4,8,12 \
    --max_steps 16 \
    --output_dir "$OUT/patching_spatial" 2>&1 | tee "$LOG/patching.log"

echo "================================================================"
echo "[4/5] Controlled time-shift  (N=80, PATH metrics)"
echo "================================================================"
python scripts/controlled/controlled_time_shift.py \
    --checkpoint "$CKPT" \
    --num_puzzles 80 \
    --max_steps 16 \
    --output_dir "$OUT/time_shift_controlled" 2>&1 | tee "$LOG/time_shift.log"

echo "================================================================"
echo "[5/5] Controlled freeze  (N=100, PATH metrics)"
echo "================================================================"
python scripts/controlled/controlled_freeze.py \
    --checkpoint "$CKPT" \
    --num_puzzles 100 \
    --max_steps 16 \
    --output_dir "$OUT/freeze_controlled" 2>&1 | tee "$LOG/freeze.log"

echo "################################################################"
echo "# HARDENED MAZE SUITE DONE  ($(date))"
echo "# Aggregates under $OUT/*/"
echo "################################################################"
