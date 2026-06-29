#!/usr/bin/env bash
# Orchestration: wait for the suite job to free the GPU, dump predictions over the
# full eval set (evaluate.py), then score with TTA voting. Logs to voting_eval.log.
set -uo pipefail
cd /home/leo.rodrigues/HRM
PY=/home/leo.rodrigues/miniconda3/envs/hrm/bin/python
CKPT=checkpoints/arc2-adapted-evalonly/step_7391
DATA=data/arc-2-evalonly

# 1) Wait until any prior suite job leaves the queue (no-op if already gone).
while squeue -u leo.rodrigues 2>/dev/null | grep -q 144820; do sleep 60; done
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
