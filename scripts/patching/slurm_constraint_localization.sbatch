#!/usr/bin/env bash
#SBATCH -J ccl_sudoku
# NOTE: partition is NOT hardcoded — pass it on submit, e.g.
#   sbatch -p <your_gpu_partition> --gres=gpu:1 scripts/patching/slurm_constraint_localization.sbatch
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Counterfactual constraint-localization (Sudoku) — staged validation + experiment:
#   [1] dry run        (build/validate, no patching)
#   [2] 1-sample smoke (validate the patching path end-to-end)
#   [3] N=60 config    (consistent >50-sample single config: z_H, all steps)
#   [4] full sweep     (patch_level{H,L,both} × patch_steps{all,early,late,final})
# Runs each independently so a failure is logged and the rest still run.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs results/constraint_localization
CCL=scripts/patching/counterfactual_constraint_sudoku.py
python -c "import torch;print('[env] cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

run() { local label="$1"; shift; echo "############ $label ($(date)) ############"; "$@"; echo "rc=$? for $label"; }

run "[1] DRY RUN" python -u $CCL --dry_run --n_puzzles 6 --num_samples 8 \
    --device cuda --output_dir results/constraint_localization/dry

run "[2] 1-SAMPLE SMOKE" python -u $CCL --num_samples 1 --num_donors 2 --n_puzzles 6 \
    --patch_level H --patch_steps late --device cuda \
    --output_dir results/constraint_localization/smoke

run "[3] N=60 (z_H, all steps)" python -u $CCL --num_samples 60 --num_donors 5 --n_puzzles 60 \
    --patch_level H --patch_steps all --device cuda \
    --output_dir results/constraint_localization/n60

run "[4] SWEEP (level x steps)" python -u $CCL --sweep --num_samples 60 --num_donors 5 --n_puzzles 60 \
    --device cuda --output_dir results/constraint_localization/sweep

echo "############ CONSTRAINT-LOCALIZATION SUITE DONE ($(date)) ############"
echo "Key files: results/constraint_localization/{n60/result.json, sweep/sweep_results.csv}"
