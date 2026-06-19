#!/bin/bash
# One-shot environment setup for the CIAI cluster: creates a conda env named
# `hrm`, installs PyTorch (CUDA) + project deps + FlashAttention, and (optionally)
# builds the Sudoku-Extreme dataset.
#
# Run ONCE from the repo root, on a node with internet access (login node is fine):
#   bash scripts/bash/setup_conda_env.sh
#
# After this, every job script activates the env via scripts/bash/_activate_env.sh
# (which runs `conda activate hrm` by default).
#
# Knobs (env vars):
#   ENV_NAME=hrm     # conda env name
#   PYVER=3.10       # python version
#   CU=cu121         # PyTorch CUDA wheel tag. cu121 matches the cluster's
#                    # driver 535 / CUDA 12.2 (see the CIAI nvidia-smi output).
#   SKIP_FA=1        # skip FlashAttention install
#   SKIP_DATA=1      # skip dataset build
set -eo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

ENV_NAME="${ENV_NAME:-hrm}"
PYVER="${PYVER:-3.10}"
CU="${CU:-cu121}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Load/install miniconda first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

# 1) Create the env if it does not already exist
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda env '${ENV_NAME}' (python=${PYVER})"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "Conda env '${ENV_NAME}' already exists — reusing."
fi

conda activate "${ENV_NAME}"
echo "Using python: $(command -v python)"

python -m pip install --upgrade pip

# 2) PyTorch with CUDA wheels matched to the cluster driver (535 / CUDA 12.2)
echo "Installing torch (${CU}) ..."
python -m pip install --index-url "https://download.pytorch.org/whl/${CU}" \
    torch torchvision torchaudio

# 3) Build helpers (needed by flash-attn) + project requirements
python -m pip install packaging ninja wheel setuptools setuptools-scm psutil
python -m pip install -r requirements.txt

# 4) Correct AdamAtan2 package: the code imports `adam_atan2_pytorch`
#    (requirements.txt's `adam-atan2` is a DIFFERENT module and is not enough).
python -m pip install adam-atan2-pytorch

# 5) FlashAttention (imported by models/layers.py; not in requirements.txt)
if [ "${SKIP_FA:-0}" != "1" ]; then
    echo "Installing flash-attn (this can take a while) ..."
    python -m pip install flash-attn --no-build-isolation
fi

# 6) Quick import smoke test
python - <<'PY'
import torch, flash_attn, adam_atan2_pytorch, hydra, omegaconf, pydantic, wandb, coolname
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("flash_attn", getattr(flash_attn, "__version__", "?"))
print("imports OK")
PY

# 7) Build the small Sudoku-Extreme dataset used by training
if [ "${SKIP_DATA:-0}" != "1" ] && [ ! -d data/sudoku-extreme-1k-aug-1000/train ]; then
    echo "Building dataset data/sudoku-extreme-1k-aug-1000 ..."
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 --num-aug 1000
fi

echo
echo "Done. Env '${ENV_NAME}' is ready. Submit the pipeline with:"
echo "    bash scripts/bash/submit_all.sh"
