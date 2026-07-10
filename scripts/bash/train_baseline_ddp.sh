#!/bin/bash
# Train ONE baseline on ONE task using all GPUs on the node via DDP (torchrun).
# Use this for the heavy stragglers (universal_transformer, vanilla_rnn) or for
# the ARC runs, where you want a single model to finish as fast as possible
# rather than sharing the node. Cluster-agnostic: run inside an allocation you
# already hold (salloc / sbatch).
#
# Usage:
#   MODEL=universal_transformer TASK=maze bash scripts/bash/train_baseline_ddp.sh
#   MODEL=plain_transformer TASK=arc EPOCHS=20000 bash scripts/bash/train_baseline_ddp.sh
#
# Env:
#   MODEL   vanilla_rnn | standard_rnn | plain_transformer | universal_transformer  (required)
#   TASK    maze | arc                       (required)
#   NPROC   GPUs to use (default 4)
#   EPOCHS  override epoch budget            (optional)
#   GBS     override global_batch_size       (optional; must be divisible by NPROC)
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

MODEL="${MODEL:?set MODEL=vanilla_rnn|standard_rnn|plain_transformer|universal_transformer}"
TASK="${TASK:?set TASK=maze|arc}"
NPROC="${NPROC:-4}"
mkdir -p logs

cfg="cfg_pretrain_${MODEL}_${TASK}"
[ -f "config/${cfg}.yaml" ] || { echo "[error] missing config/${cfg}.yaml" >&2; exit 1; }

# --- activate environment (prefer .venv, fallback to conda env "hrm") ---
# Relax -u: conda activate.d cuda hooks reference unbound vars (NVCC_PREPEND_FLAGS).
set +u
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${HRM_CONDA_ENV:-hrm}"
fi
set -u

export DISABLE_COMPILE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

OV=""
[ -n "${EPOCHS:-}" ] && OV="$OV epochs=${EPOCHS}"
[ -n "${GBS:-}" ]    && OV="$OV global_batch_size=${GBS}"
[ -n "${SEED:-}" ]   && OV="$OV seed=${SEED}"   # multi-seed baselines (R6)

ts="$(date +%Y%m%d_%H%M%S)"
log="logs/baseline_${TASK}_${MODEL}_ddp_seed${SEED:-0}_${ts}.log"
echo "[launch] DDP nproc=${NPROC} cfg=${cfg} EPOCHS=${EPOCHS:-config} GBS=${GBS:-config}  (log: ${log})"

torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${MASTER_PORT:-29500}" \
    pretrain.py --config-name "${cfg}" ${OV} 2>&1 | tee "${log}"
