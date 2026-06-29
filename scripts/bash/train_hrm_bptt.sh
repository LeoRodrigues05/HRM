#!/bin/bash
#SBATCH --job-name=hrm_bptt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/hrm_bptt_%j.out
#SBATCH --error=logs/hrm_bptt_%j.err

# BPTT HRM (within-step back-propagation through time) on Sudoku-Extreme.
# 1 node x 4 A100 (DDP). The cscc-gpu-qos QOS caps a user at 4 GPUs total, so
# this CANNOT run concurrently with the stock job and CANNOT use 8 GPUs — run it
# sequentially after the stock job (submit_all.sh wires the dependency).
#
# A100s here are 40 GB. Within-step BPTT raises activation memory ~4x vs stock.
# If you hit CUDA OOM, lower the batch: export GBS=384 (DDP keeps the math
# equivalent; only throughput changes).
#
# Optional env overrides (export before sbatch, or via submit_all.sh):
#   EPOCHS=8000   # shrink to fit a deadline (apply the SAME value to BOTH jobs)
#   GBS=768       # global_batch_size (drop to 384 on OOM)
#   HRM_OVERRIDES="..."
#
# Self-healing through transient GPU faults: --requeue only covers preemption /
# node failure, NOT an application-level CUDA crash (e.g. "unspecified launch
# failure"), which exits non-zero. Since we checkpoint every 500 steps, on such a
# crash this job resubmits ITSELF (up to BPTT_MAX_RETRIES) and the new job resumes
# from latest.pt. Knobs:
#   BPTT_MAX_RETRIES=5     # max self-resubmits before giving up
#   BPTT_ATTEMPT=1         # internal counter (do not set by hand)
#   BPTT_EXCLUDE=gpu-54    # comma-separated nodes to avoid (the gpu-54 fault node);
#                          # set to empty to disable exclusion

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

: "${BPTT_MAX_RETRIES:=5}"
: "${BPTT_ATTEMPT:=1}"
: "${BPTT_EXCLUDE:=gpu-54}"

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

echo "Starting BPTT HRM (attempt ${BPTT_ATTEMPT}/${BPTT_MAX_RETRIES}) at $(date) | nodes=${SLURM_JOB_NUM_NODES} master=${MASTER_ADDR}"

# Don't let a non-zero srun abort the script before we can resubmit (set -e).
rc=0
srun bash scripts/bash/_torchrun_launch.sh || rc=$?

if [ "${rc}" -ne 0 ]; then
    echo "srun FAILED (exit ${rc}) on attempt ${BPTT_ATTEMPT}/${BPTT_MAX_RETRIES} at $(date)" >&2
    if [ "${BPTT_ATTEMPT}" -lt "${BPTT_MAX_RETRIES}" ]; then
        EXCL_ARG=""
        [ -n "${BPTT_EXCLUDE}" ] && EXCL_ARG="--exclude=${BPTT_EXCLUDE}"
        echo "Resubmitting to resume from latest.pt (next attempt $((BPTT_ATTEMPT+1)))${EXCL_ARG:+ ${EXCL_ARG}}" >&2
        sbatch ${EXCL_ARG} \
            --export=ALL,BPTT_ATTEMPT=$((BPTT_ATTEMPT+1)),BPTT_MAX_RETRIES="${BPTT_MAX_RETRIES}",BPTT_EXCLUDE="${BPTT_EXCLUDE}" \
            scripts/bash/train_hrm_bptt.sh
    else
        echo "Reached BPTT_MAX_RETRIES (${BPTT_MAX_RETRIES}); not resubmitting." >&2
    fi
    exit "${rc}"
fi

echo "Finished at $(date)"
