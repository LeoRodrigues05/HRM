#!/bin/bash
#SBATCH --job-name=hrm_prep
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:1
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/hrm_prep_%j.out
#SBATCH --error=logs/hrm_prep_%j.err

# Prep step (recommended order #1 + #2 prerequisite):
#   1. Sanity-check the new code (BPTT flag + SAE mean-centering correctness).
#   2. Build the Sudoku-Extreme dataset if it is missing.
# Fast (~minutes). Training jobs depend on this completing successfully.

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
source scripts/bash/_activate_env.sh
export DISABLE_COMPILE=1

echo "=== [1/2] Sanity checks ==="
python scripts/sanity_check_bptt_sae.py

echo "=== [2/2] Dataset ==="
if [ ! -d data/sudoku-extreme-1k-aug-1000/train ]; then
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 --num-aug 1000
else
    echo "Dataset already present — skipping build."
fi
echo "Prep complete at $(date)"
