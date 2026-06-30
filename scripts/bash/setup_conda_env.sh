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
#   CU=cu124         # PyTorch CUDA wheel tag. cu124 runs on the cluster's
#                    # driver 535 via CUDA-12 minor-version compatibility, and
#                    # matches the prebuilt flash-attn wheel below.
#   SKIP_FA=1        # skip FlashAttention install (SDPA fallback is used instead)
#   SKIP_DATA=1      # skip dataset build
set -eo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

ENV_NAME="${ENV_NAME:-hrm}"
PYVER="${PYVER:-3.10}"
CU="${CU:-cu124}"
TORCH_VER="${TORCH_VER:-2.5.1}"
TV_VER="${TV_VER:-0.20.1}"
# Prebuilt flash-attn wheel matched to torch 2.5 / CUDA 12 / cp310 / cxx11abiFALSE.
# Building from source needs nvcc (not available on login nodes), so we use the
# official prebuilt wheel. Must stay consistent with TORCH_VER/CU/PYVER above.
FA_WHEEL_URL="${FA_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

# Make conda available even when it is not on PATH (CIAI hides it behind
# `module load anaconda3` at /apps/local/anaconda3).
if ! command -v conda >/dev/null 2>&1; then
    for _init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh \
                 /usr/share/Modules/init/bash; do
        [ -f "${_init}" ] && source "${_init}" 2>/dev/null && break
    done
    command -v module >/dev/null 2>&1 && { module load anaconda3 2>/dev/null || true; }
    if ! command -v conda >/dev/null 2>&1 && [ -x /apps/local/anaconda3/bin/conda ]; then
        export PATH="/apps/local/anaconda3/bin:$PATH"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda still not found. Try 'module load anaconda3' manually, then re-run." >&2
    exit 1
fi
echo "Using conda: $(command -v conda) ($(conda --version 2>&1))"

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

# 2) PyTorch (force the exact CUDA build). Pinning the +${CU} local version makes
#    pip switch builds even if a different torch ${TORCH_VER} is already present
#    (a plain `torch==2.5.1` is considered "already satisfied" by +cu118).
echo "Installing torch ${TORCH_VER}+${CU} ..."
python -m pip install --index-url "https://download.pytorch.org/whl/${CU}" \
    "torch==${TORCH_VER}+${CU}" "torchvision==${TV_VER}+${CU}" "torchaudio==${TORCH_VER}+${CU}"

# 3) Build helpers (needed by flash-attn) + project requirements
python -m pip install packaging ninja wheel setuptools setuptools-scm psutil
python -m pip install -r requirements.txt

# 4) Correct AdamAtan2 package: the code imports `adam_atan2_pytorch`
#    (requirements.txt's `adam-atan2` is a DIFFERENT module and is not enough).
python -m pip install adam-atan2-pytorch

# 5) FlashAttention — install the prebuilt wheel (no nvcc needed). If it fails
#    for any reason the model still runs via the built-in PyTorch SDPA fallback,
#    so this step never blocks setup. Set SKIP_FA=1 to skip entirely.
if [ "${SKIP_FA:-0}" != "1" ]; then
    echo "Installing prebuilt flash-attn wheel ..."
    python -m pip install "${FA_WHEEL_URL}" \
        || python -m pip install flash-attn --no-build-isolation \
        || echo "WARN: flash-attn install failed — continuing with SDPA fallback."
fi

# 6) Quick import smoke test (flash_attn is optional)
python - <<'PY'
import torch, adam_atan2_pytorch, hydra, omegaconf, pydantic, wandb, coolname
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
try:
    import flash_attn
    print("flash_attn", getattr(flash_attn, "__version__", "?"))
except Exception:
    print("flash_attn: NOT installed -> using PyTorch SDPA fallback")
# Verify the model imports and attention path resolves either way.
from models.layers import _HAS_FLASH_ATTN
print("Attention backend:", "flash-attn" if _HAS_FLASH_ATTN else "SDPA fallback")
import models.hrm.hrm_act_v1  # must import cleanly
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
