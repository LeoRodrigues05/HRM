#!/bin/bash
#SBATCH --job-name=hrm_bptt
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/hrm_bptt_%j.out
#SBATCH --error=logs/hrm_bptt_%j.err

# BPTT HRM (within-step back-propagation through time) on Sudoku-Extreme.
# 2 nodes x 4 A100 = 8 GPUs (DDP). BPTT is the long pole, so it gets more GPUs
# than the stock run to land inside the deadline. The repo's manual cross-rank
# all-reduce (pretrain.py) is correct across nodes, so DDP scales here.
#
# A100s on CIAI are 40 GB. Within-step BPTT raises activation memory ~4x vs stock.
# If you hit CUDA OOM, lower the batch: export GBS=384 (DDP keeps the math
# equivalent; only throughput changes).
#
# Optional env overrides (export before sbatch, or via submit_all.sh):
#   EPOCHS=8000   # shrink to fit a deadline (apply the SAME value to BOTH jobs)
#   GBS=768       # global_batch_size (drop to 384 on OOM)
#   HRM_OVERRIDES="..."

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

if [ ! -d data/sudoku-extreme-1k-aug-1000/train ]; then
    echo "ERROR: dataset missing. Run the prep job first (submit_all.sh handles this)." >&2
    exit 1
fi

OV=""
[ -n "${EPOCHS:-}" ] && OV="$OV epochs=${EPOCHS}"
[ -n "${GBS:-}" ]    && OV="$OV global_batch_size=${GBS}"
export HRM_CONFIG=cfg_pretrain_hrm_bptt
export HRM_OVERRIDES="${OV} ${HRM_OVERRIDES:-}"
export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"

echo "Starting BPTT HRM at $(date) | nodes=${SLURM_JOB_NUM_NODES} master=${MASTER_ADDR}"
srun bash scripts/bash/_torchrun_launch.sh
echo "Finished at $(date)"
