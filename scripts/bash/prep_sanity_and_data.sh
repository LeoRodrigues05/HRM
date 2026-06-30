#!/bin/bash
#SBATCH --job-name=hrm_prep
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/hrm_prep_%j.out
#SBATCH --error=logs/hrm_prep_%j.err

# Prep step (recommended order #1 + #2 prerequisite):
#   1. Sanity-check the new code (BPTT flag + SAE mean-centering correctness).
#   2. Build the Sudoku-Extreme dataset if it is missing.
# Fast (~minutes). Training jobs depend on this completing successfully.

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
source scripts/bash/_activate_env.sh
export DISABLE_COMPILE=1

echo "=== [1/3] Sanity checks ==="
python scripts/sanity_check_bptt_sae.py

echo "=== [2/3] GPU attention backend check ==="
python - <<'PY'
import torch
from models.layers import _HAS_FLASH_ATTN
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
print("backend:", "flash-attn" if _HAS_FLASH_ATTN else "SDPA fallback")
if torch.cuda.is_available() and _HAS_FLASH_ATTN:
    # Real flash-attn call on this A100 to confirm the wheel runs (sm_80, bf16).
    from flash_attn import flash_attn_func
    q = torch.randn(2, 81, 8, 64, device="cuda", dtype=torch.bfloat16)
    o = flash_attn_func(q=q, k=q, v=q, causal=False)
    o = o[0] if isinstance(o, tuple) else o
    assert o.shape == q.shape and torch.isfinite(o).all()
    print("flash-attn GPU forward OK:", tuple(o.shape))
elif torch.cuda.is_available():
    print("flash-attn not present -> training will use SDPA on GPU")
PY

echo "=== [3/3] Dataset ==="
if [ ! -d data/sudoku-extreme-1k-aug-1000/train ]; then
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 --num-aug 1000
else
    echo "Dataset already present — skipping build."
fi
echo "Prep complete at $(date)"

