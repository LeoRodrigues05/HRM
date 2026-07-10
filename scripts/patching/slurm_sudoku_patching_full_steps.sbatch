#!/usr/bin/env bash
# Sudoku per-step cross-puzzle patching with bootstrap CIs (Phase-0 hardening of the
# paper's -53.1/-61.7 point estimates + the Sudoku series for fig_crosstask_patching).
# Runs a 2-pair smoke first so a wiring bug fails in minutes, then the full sweep.
# Submit:  sbatch -p gpu --gres=gpu:1 scripts/patching/slurm_sudoku_patching_full_steps.sbatch
#SBATCH -J sudoku_patch_full
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/patching/%x_%j.out
#SBATCH -e logs/patching/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/patching

NUM_PAIRS="${NUM_PAIRS:-250}"
PATCH_STEPS="${PATCH_STEPS:-1,2,4,6,8,10,12,15}"
OUTPUT_DIR="${OUTPUT_DIR:-results/patching/patching_full_steps}"

echo "[sudoku_patch_full] smoke (2 pairs)"
conda run --no-capture-output -n hrm python -u scripts/patching/patching_sudoku_full_steps.py \
    --num_pairs 2 --patch_steps 1,15 \
    --output_dir results/patching/_smoke_full_steps

echo "[sudoku_patch_full] full run: pairs=$NUM_PAIRS steps=$PATCH_STEPS"
conda run --no-capture-output -n hrm python -u scripts/patching/patching_sudoku_full_steps.py \
    --num_pairs "$NUM_PAIRS" --patch_levels H,L --patch_steps "$PATCH_STEPS" \
    --output_dir "$OUTPUT_DIR"

echo "[sudoku_patch_full] regenerating cross-task figures"
conda run --no-capture-output -n hrm python -u scripts/plotting/plot_crosstask_figures.py \
    --output_dir results/reports/paper_aaai_figures
echo "[sudoku_patch_full] done"
