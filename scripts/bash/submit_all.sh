#!/bin/bash
# Submit the full pipeline to the CIAI cluster, in the recommended order, with
# Slurm dependencies so each stage waits for the previous one.
#
#   prep  (1 GPU)  : sanity checks + build dataset
#     -> stock (1 node / 4 GPU)   \  run concurrently (4 + 8 = 12 GPUs, the cap)
#     -> bptt  (2 nodes / 8 GPU)  /
#       -> post (1 GPU) : collect activations + SAE A/B (baseline vs mean-centered)
#
# Run this on the LOGIN node (it only calls sbatch):
#   bash scripts/bash/submit_all.sh
#
# Deadline / OOM knobs — export before running and they propagate to the jobs:
#   export EPOCHS=8000      # fewer epochs (applied to BOTH training jobs)
#   export GBS=384          # smaller global batch (e.g. if BPTT OOMs on 40GB A100)
#   export HRM_ENV_SETUP="source /path/to/your/env/bin/activate"   # if not using ./.venv

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

echo "Submitting pipeline (EPOCHS=${EPOCHS:-config} GBS=${GBS:-config})"

PREP=$(sbatch --parsable scripts/bash/prep_sanity_and_data.sh)
echo "  prep   : ${PREP}"

STOCK=$(sbatch --parsable --dependency=afterok:${PREP} scripts/bash/train_hrm_stock.sh)
echo "  stock  : ${STOCK}  (after ${PREP})"

BPTT=$(sbatch --parsable --dependency=afterok:${PREP} scripts/bash/train_hrm_bptt.sh)
echo "  bptt   : ${BPTT}  (after ${PREP})"

POST=$(sbatch --parsable --dependency=afterok:${STOCK}:${BPTT} scripts/bash/post_collect_and_sae_ab.sh)
echo "  post   : ${POST}  (after ${STOCK} & ${BPTT})"

echo
echo "Monitor:  squeue --me        (or:  watch -n5 squeue --me)"
echo "Logs:     logs/hrm_*_<jobid>.out"
echo "Cancel:   scancel ${PREP} ${STOCK} ${BPTT} ${POST}"
