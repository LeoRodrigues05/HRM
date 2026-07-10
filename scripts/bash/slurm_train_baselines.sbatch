#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=48:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
# >>> EDIT THESE FOR YOUR CLUSTER (uncomment + fill in) <<<
###SBATCH --partition=YOUR_PARTITION
###SBATCH --account=YOUR_ACCOUNT
###SBATCH --qos=YOUR_QOS
#
# Submits ONE task's four baselines to a single 4-GPU node, running them
# concurrently (one model per GPU). TASK / EPOCHS / GBS propagate from the
# submitting shell because sbatch defaults to --export=ALL.
#
#   TASK=maze sbatch scripts/bash/slurm_train_baselines.sbatch
#   TASK=arc EPOCHS=20000 sbatch --time=96:00:00 scripts/bash/slurm_train_baselines.sbatch
#
# For a single heavy model on all 4 GPUs (DDP) instead, submit a wrapper around
# train_baseline_ddp.sh, or run it interactively after salloc.
set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

echo "Starting baselines wave: TASK=${TASK:-?} EPOCHS=${EPOCHS:-config} at $(date) on $(hostname)"
bash scripts/bash/train_baselines_concurrent.sh
echo "Finished at $(date)"
