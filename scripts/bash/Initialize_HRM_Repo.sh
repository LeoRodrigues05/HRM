#!/usr/bin/env bash
# Bootstrap script: install Python env, CUDA-enabled PyTorch, FlashAttention,
# project requirements, and download/build the small Sudoku-Extreme dataset.
#
# Run from the repository root:
#   bash scripts/bash/Initialize_HRM_Repo.sh
#
# Skip optional steps via env vars:
#   SKIP_CUDA=1   # do not install CUDA toolkit (already on system)
#   SKIP_FA=1     # do not install FlashAttention
#   SKIP_DATA=1   # do not build Sudoku dataset
#   SKIP_APT=1    # do not install apt packages
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# 1) uv (fast Python package manager) + venv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv venv -p 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 2) System build tools (Linux only)
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt >/dev/null 2>&1; then
    sudo apt update && sudo apt install -y build-essential git ninja-build python3-dev
fi

# 3) Optional: install CUDA 12.6 toolkit side-by-side
if [ "${SKIP_CUDA:-0}" != "1" ]; then
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run"
    wget -q --show-progress -O cuda_12_6.run "$CUDA_URL"
    sudo sh cuda_12_6.run --silent --toolkit --override
    rm -f cuda_12_6.run
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# 4) PyTorch with CUDA 12.6 wheels + build helpers
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
uv pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
uv pip install packaging ninja wheel setuptools setuptools-scm psutil

# 5) FlashAttention (Ampere/Ada -> FA2 wheel; Hopper users see README)
if [ "${SKIP_FA:-0}" != "1" ]; then
    uv pip install flash-attn
fi

# 6) Project Python requirements + git submodules
git submodule update --init --recursive || true
uv pip install -r requirements.txt

# 7) Build the small Sudoku-Extreme dataset used by all replication scripts
if [ "${SKIP_DATA:-0}" != "1" ]; then
    uv run python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 --num-aug 1000
fi

echo "[Initialize_HRM_Repo] done. Activate the env with: source .venv/bin/activate"
