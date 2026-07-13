#!/bin/bash
# Train all four "initial" baselines (Vanilla RNN, Standard RNN/GRU, Plain
# Transformer, Universal Transformer) for ONE task CONCURRENTLY — one model per
# GPU — on a 4-GPU node. Cluster-agnostic: contains no Slurm directives, so run
# it once you already hold the node (via `salloc ... --gres=gpu:4` or from inside
# an sbatch allocation such as slurm_train_baselines.sbatch).
#
# Usage:
#   TASK=maze bash scripts/bash/train_baselines_concurrent.sh
#   TASK=arc EPOCHS=20000 bash scripts/bash/train_baselines_concurrent.sh
#
# Env:
#   TASK    maze | arc                      (required)
#   EPOCHS  override the epoch budget        (optional; recommend 20000 for arc)
#   GBS     override global_batch_size       (optional; lower if you OOM at 40GB)
#   MODELS  subset/reorder the model list    (default: all four, GPUs 0..3)
#
# Each model is pinned to one GPU with CUDA_VISIBLE_DEVICES and logged to
# logs/baseline_<task>_<model>_<ts>.log. The wall-clock for the wave is the
# SLOWEST model (typically universal_transformer) — for that straggler you may
# prefer the 4-GPU DDP runner (train_baseline_ddp.sh) instead.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

TASK="${TASK:?set TASK=maze or TASK=arc}"
MODELS="${MODELS:-vanilla_rnn standard_rnn plain_transformer universal_transformer}"
mkdir -p logs

# --- activate environment ---
# Relax -u: conda activate.d CUDA hooks reference unbound variables.
set +u
source scripts/bash/_activate_env.sh
set -u
if ! command -v python >/dev/null 2>&1; then
    echo "[error] Python environment activation failed; 'python' is not on PATH" >&2
    exit 127
fi

export DISABLE_COMPILE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# Common hydra overrides
OV=""
[ -n "${EPOCHS:-}" ] && OV="$OV epochs=${EPOCHS}"
[ -n "${GBS:-}" ]    && OV="$OV global_batch_size=${GBS}"
if [ -n "${SEED:-}" ]; then
    # Keep ensemble members independent. pretrain.py automatically resumes
    # latest.pt, so sharing the config's checkpoint_path would merge seeds.
    OV="$OV ++seed=${SEED}"
fi

echo "[info] TASK=$TASK  EPOCHS=${EPOCHS:-config}  GBS=${GBS:-config}  SEED=${SEED:-config}  models: $MODELS"
echo "[info] $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | tr '\n' '|')"

gpu=0
pids=()
names=()
for m in $MODELS; do
    cfg="cfg_pretrain_${m}_${TASK}"
    if [ ! -f "config/${cfg}.yaml" ]; then
        echo "[error] missing config/${cfg}.yaml — skipping $m" >&2
        continue
    fi
    ts="$(date +%Y%m%d_%H%M%S)"
    log="logs/baseline_${TASK}_${m}_seed${SEED:-0}_${ts}.log"
    model_ov="${OV}"
    if [ -n "${SEED:-}" ]; then
        model_ov="$model_ov ++checkpoint_path=checkpoints/baselines-${TASK}/${m}/seed${SEED} ++run_name=${TASK}-${m}-seed${SEED}"
    fi
    echo "[launch] GPU ${gpu} -> ${cfg}   (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" python pretrain.py --config-name "${cfg}" ${model_ov} > "${log}" 2>&1 &
    pids+=("$!")
    names+=("${m}")
    gpu=$((gpu + 1))
done

echo "[info] launched ${#pids[@]} job(s) across ${gpu} GPU(s) at $(date); waiting..."
rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[ok]   ${names[$i]} finished"
    else
        echo "[FAIL] ${names[$i]} exited non-zero" >&2
        rc=1
    fi
done
echo "[done] all baseline jobs for TASK=${TASK} finished (rc=${rc}) at $(date)"
exit "${rc}"
