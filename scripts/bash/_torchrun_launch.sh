#!/bin/bash
# Launched once per node by srun (--ntasks-per-node=1). Each invocation starts a
# torchrun that spawns 4 workers (one per GPU). Across N nodes that is 4*N ranks,
# coordinated via the c10d rendezvous on MASTER_ADDR.
#
# Reads:
#   HRM_CONFIG     — pretrain.py --config-name value (required)
#   HRM_OVERRIDES  — extra hydra overrides (optional, space-separated)
#   MASTER_ADDR    — rendezvous host (exported by the job script)
#   SLURM_*        — provided by Slurm
set -eo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Activate Python env (fresh shell under srun)
source scripts/bash/_activate_env.sh

export DISABLE_COMPILE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# wandb defaults to online and will hang/error on compute nodes without internet
# or a login. Default to offline (metrics saved locally; `wandb sync` later).
# Override by exporting WANDB_MODE=online before sbatch if your nodes have egress.
export WANDB_MODE="${WANDB_MODE:-offline}"

NNODES="${SLURM_JOB_NUM_NODES:-1}"
MASTER="${MASTER_ADDR:-127.0.0.1}"

echo "[launch] node=$(hostname) nnodes=${NNODES} master=${MASTER} config=${HRM_CONFIG} overrides=${HRM_OVERRIDES:-<none>}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node=4 \
    --rdzv_id="${SLURM_JOB_ID:-0}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER}:29500" \
    pretrain.py --config-name "${HRM_CONFIG}" ${HRM_OVERRIDES:-}
