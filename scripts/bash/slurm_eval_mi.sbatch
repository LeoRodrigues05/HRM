#!/bin/bash
#SBATCH --job-name=eval_mi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=ws-ia
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Generic single-GPU submit wrapper for the eval-only MI jobs that run >4h on the
# RTX 5000. Pass the command to run via the CMD env var (sbatch --export=ALL
# propagates it). Override --job-name / --time / --partition on the sbatch line.
#
#   CMD='python scripts/...' sbatch --job-name=r2mean_sud scripts/bash/slurm_eval_mi.sbatch
#
# NOTE: adjust --partition (and add `-w ws-l3-020`) for your cluster if ws-ia is
# not the right eval partition.
set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${HRM_CONDA_ENV:-hrm}"; fi
export DISABLE_COMPILE=1
mkdir -p logs
: "${CMD:?set CMD='python ...' (the command to run)}"
echo "[eval_mi] $(date) on $(hostname)"
echo "[eval_mi] CMD: $CMD"
bash -c "$CMD"
echo "[eval_mi] done $(date)"
