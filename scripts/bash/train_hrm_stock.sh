#!/bin/bash
#SBATCH --job-name=hrm_stock
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/hrm_stock_%j.out
#SBATCH --error=logs/hrm_stock_%j.err

# Stock HRM (one-step gradient) on Sudoku-Extreme.
# 1 node x 4 A100 (DDP). The cscc-gpu-qos QOS caps a user at 4 GPUs total, so the
# stock and BPTT jobs run SEQUENTIALLY (not concurrently). submit_all.sh runs
# this one first, then BPTT after it finishes.
#
# Optional env overrides (export before sbatch, or via submit_all.sh):
#   EPOCHS=8000   # shrink to fit a deadline (apply the SAME value to BOTH jobs)
#   GBS=768       # global_batch_size
#   HRM_OVERRIDES="..."  # any extra hydra overrides
#
# Self-healing through transient GPU faults: --requeue only covers preemption /
# node failure, NOT an application-level CUDA crash (e.g. "unspecified launch
# failure"), which exits non-zero. Since we checkpoint every 500 steps, on such a
# crash this job resubmits ITSELF (up to STOCK_MAX_RETRIES) and the new job resumes
# from latest.pt. Knobs:
#   STOCK_MAX_RETRIES=5    # max self-resubmits before giving up
#   STOCK_ATTEMPT=1        # internal counter (do not set by hand)
#   STOCK_EXCLUDE=gpu-54   # comma-separated nodes to avoid (the gpu-54 fault node);
#                          # set to empty to disable exclusion

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

: "${STOCK_MAX_RETRIES:=5}"
: "${STOCK_ATTEMPT:=1}"
: "${STOCK_EXCLUDE:=gpu-54}"

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

echo "Starting STOCK HRM (attempt ${STOCK_ATTEMPT}/${STOCK_MAX_RETRIES}) at $(date) | nodes=${SLURM_JOB_NUM_NODES} master=${MASTER_ADDR}"

# Don't let a non-zero srun abort the script before we can resubmit (set -e).
rc=0
srun bash scripts/bash/_torchrun_launch.sh || rc=$?

if [ "${rc}" -ne 0 ]; then
    echo "srun FAILED (exit ${rc}) on attempt ${STOCK_ATTEMPT}/${STOCK_MAX_RETRIES} at $(date)" >&2
    if [ "${STOCK_ATTEMPT}" -lt "${STOCK_MAX_RETRIES}" ]; then
        EXCL_ARG=""
        [ -n "${STOCK_EXCLUDE}" ] && EXCL_ARG="--exclude=${STOCK_EXCLUDE}"
        echo "Resubmitting to resume from latest.pt (next attempt $((STOCK_ATTEMPT+1)))${EXCL_ARG:+ ${EXCL_ARG}}" >&2
        sbatch ${EXCL_ARG} \
            --export=ALL,STOCK_ATTEMPT=$((STOCK_ATTEMPT+1)),STOCK_MAX_RETRIES="${STOCK_MAX_RETRIES}",STOCK_EXCLUDE="${STOCK_EXCLUDE}" \
            scripts/bash/train_hrm_stock.sh
    else
        echo "Reached STOCK_MAX_RETRIES (${STOCK_MAX_RETRIES}); not resubmitting." >&2
    fi
    exit "${rc}"
fi

echo "Finished at $(date)"
