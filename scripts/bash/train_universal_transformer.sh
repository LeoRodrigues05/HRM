#!/bin/bash
#SBATCH --job-name=univ_trans
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=logs/universal_transformer_%j.out
#SBATCH --error=logs/universal_transformer_%j.err

mkdir -p logs

# Activate environment
# Activate environment (prefer .venv, fallback to conda env named "hrm")
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${HRM_CONDA_ENV:-hrm}"
fi

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Disable torch.compile for stability (avoids graph break issues)
export DISABLE_COMPILE=1

echo "Starting UniversalTransformer training at $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python pretrain.py --config-name cfg_pretrain_universal_transformer

echo "Finished at $(date)"
