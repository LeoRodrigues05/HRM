#!/bin/bash
#SBATCH --job-name=vanilla_rnn
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=logs/vanilla_rnn_%j.out
#SBATCH --error=logs/vanilla_rnn_%j.err

mkdir -p logs

# Activate environment
eval "$(conda shell.bash hook)"
conda activate hrm

cd /home/leo.rodrigues/HRM

# Disable torch.compile for stability (avoids graph break issues)
export DISABLE_COMPILE=1

echo "Starting VanillaRNN training at $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python pretrain.py --config-name cfg_pretrain_vanilla_rnn

echo "Finished at $(date)"
