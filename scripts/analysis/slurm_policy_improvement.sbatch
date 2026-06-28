#!/usr/bin/env bash
# Experiments A & D: HRM recursion as a policy-improvement operator.
# Runs maze then sudoku sequentially on a single GPU.
# Expected runtime: ~1 h each task at N=500 (2 h total, well within -t 04:00:00).
#
# Submit:
#   sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_policy_improvement.sbatch
#
# Override puzzle count:
#   N=50 sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_policy_improvement.sbatch
#SBATCH -J hrm_polimp
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs \
         results/maze/policy_improvement \
         results/controlled/policy_improvement \
         results/reports/policy_improvement_figures

N="${N:-500}"
echo "[hrm_polimp] starting  N=${N}  $(date)"

echo "[hrm_polimp] === maze ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/policy_improvement.py \
    --task maze \
    --num_puzzles "$N" \
    --output_dir results/maze/policy_improvement \
    --device cuda

echo "[hrm_polimp] === sudoku ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/policy_improvement.py \
    --task sudoku \
    --num_puzzles "$N" \
    --output_dir results/controlled/policy_improvement \
    --device cuda

echo "[hrm_polimp] === figures ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/plot_policy_improvement.py \
    --outdir results/reports/policy_improvement_figures

echo "[hrm_polimp] done  $(date)"
