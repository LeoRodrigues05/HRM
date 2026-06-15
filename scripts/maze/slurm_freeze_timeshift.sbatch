#!/usr/bin/env bash
#SBATCH -J freeze_timeshift
# NOTE: partition is NOT hardcoded — pass it on submit, e.g.
#   sbatch -p <your_gpu_partition> --gres=gpu:1 scripts/maze/slurm_freeze_timeshift.sbatch
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o logs/maze/%x_%j.out
#SBATCH -e logs/maze/%x_%j.err
#
# Freeze (E2) + Time-shift (E5) at a CONSISTENT N for BOTH Sudoku and Maze.
# Sudoku (seq 81) runs first (cheap); Maze (seq 900) runs last (~3-4 h each).
# Each step is independent — a failure is logged and the rest still run.
# Override N with:  sbatch --export=ALL,NUM_PUZZLES=1000 ...
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # conda nvcc activate.d trips set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/maze

NUM_PUZZLES="${NUM_PUZZLES:-1000}"
MAZE="${MAZE:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"

python -c "import torch;print('[env] cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "############ START $label  ($(date)) ############"
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "############ END   $label  rc=$rc  elapsed=$((SECONDS-t0))s ############"
  return 0   # never abort the batch on a single failure
}

# --- Sudoku (cheap) ---
run "SUDOKU freeze N=$NUM_PUZZLES" \
  python -u scripts/controlled/controlled_freeze.py \
    --num_puzzles "$NUM_PUZZLES" --device cuda \
    --output_dir results/controlled/freeze

run "SUDOKU time-shift N=$NUM_PUZZLES" \
  python -u scripts/controlled/controlled_time_shift.py --mode fixed \
    --num_puzzles "$NUM_PUZZLES" --device cuda \
    --output_dir results/controlled/time_shift

# --- Maze (heavy, seq 900) ---
run "MAZE freeze N=$NUM_PUZZLES" \
  python -u scripts/controlled/controlled_freeze.py --checkpoint "$MAZE" \
    --num_puzzles "$NUM_PUZZLES" --max_steps 16 --device cuda \
    --output_dir results/maze/hardened/freeze_controlled

run "MAZE time-shift N=$NUM_PUZZLES" \
  python -u scripts/controlled/controlled_time_shift.py --checkpoint "$MAZE" --mode fixed \
    --num_puzzles "$NUM_PUZZLES" --max_steps 16 --device cuda \
    --output_dir results/maze/hardened/time_shift_controlled

echo "############ ALL FREEZE + TIME-SHIFT DONE  ($(date)) ############"
