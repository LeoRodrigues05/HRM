#!/usr/bin/env bash
# Orchestration: wait for the suite job to free the GPU, dump predictions over the
# full eval set (evaluate.py), then score with TTA voting. Logs to voting_eval.log.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PY="${PY:-python}"
CKPT=checkpoints/arc2-adapted-evalonly/step_7391
DATA=data/arc-2-evalonly

# 1) Wait until any prior suite job leaves the queue (no-op if already gone).
WAIT_JOB="${WAIT_JOB:-}"   # optional SLURM job id to wait on before starting
while [ -n "$WAIT_JOB" ] && squeue -u "$USER" 2>/dev/null | grep -q "$WAIT_JOB"; do sleep 60; done
echo "[voting] starting prediction dump $(date)"

# 2) Dump per-example predictions over the full eval set (GPU, the slow part).
# Needs ~30-40 GB host RAM: evaluate() accumulates inputs/labels/logits for all
# ~165k augmented examples then concats (a 32 GB cap OOM-killed the first attempt).
srun -N1 -n1 -w ws-l3-019 --mem=96G --gres=gpu:1 \
    bash -c "PYTHONPATH=\$PWD $PY evaluate.py checkpoint=$CKPT" \
    && echo "[voting] dump complete $(date)" || { echo "[voting] dump FAILED"; exit 2; }

# 3) Vote + score (CPU-light, on a GPU alloc to respect the no-login-node rule).
srun -N1 -n1 -w ws-l3-019 --mem=96G \
    bash -c "PYTHONPATH=\$PWD $PY scripts/arc/voting_eval_arc.py --checkpoint $CKPT --dataset_path $DATA"
echo "[voting] DONE $(date)"
