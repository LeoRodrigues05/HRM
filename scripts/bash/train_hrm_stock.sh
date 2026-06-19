#!/bin/bash
#SBATCH --job-name=hrm_stock
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/hrm_stock_%j.out
#SBATCH --error=logs/hrm_stock_%j.err

# Stock HRM (one-step gradient) on Sudoku-Extreme.
# 1 node x 4 A100 (DDP). Pairs with train_hrm_bptt.sh (2 nodes) — together they
# use 4 + 8 = 12 GPUs, exactly the CIAI `long` partition cap, so both can run
# concurrently.
#
# Optional env overrides (export before sbatch, or via submit_all.sh):
#   EPOCHS=8000   # shrink to fit a deadline (apply the SAME value to BOTH jobs)
#   GBS=768       # global_batch_size
#   HRM_OVERRIDES="..."  # any extra hydra overrides

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
export HRM_CONFIG=cfg_pretrain_hrm_stock
export HRM_OVERRIDES="${OV} ${HRM_OVERRIDES:-}"
export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"

echo "Starting STOCK HRM at $(date) | nodes=${SLURM_JOB_NUM_NODES} master=${MASTER_ADDR}"
srun bash scripts/bash/_torchrun_launch.sh
echo "Finished at $(date)"
