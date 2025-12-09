#!/usr/bin/env bash
set -euo pipefail

# Probe workflow commands (successful sequence)
# Usage: bash scripts/probe_commands.sh

# 1) Activate venv and set PYTHONPATH to repo root
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# 2) (Optional) Ensure CUDA toolkit path if using GPU
# export CUDA_HOME=/usr/local/cuda-12.6
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 3) Build Sudoku dataset used by probes
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# Verify dataset manifest exists
ls -lh data/sudoku-extreme-1k-aug-1000/test/dataset.json

# 4) Run probe collection driver
# Use conservative defaults; override via env if needed.
# Examples: PROBE_BATCH_SIZE=8 HRM_HALT_MAX_STEPS=4 MAX_PROBE_BATCHES=5 CPU_ONLY=1
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-8} \
HRM_HALT_MAX_STEPS=${HRM_HALT_MAX_STEPS:-8} \
MAX_PROBE_BATCHES=${MAX_PROBE_BATCHES:-10} \
python scripts/run_probes_driver.py

# 5) Inspect saved probes
ls -lh results/probes/

python - <<'PY'
import torch
g=torch.load("results/probes/probe_global.pt"); l=torch.load("results/probes/probe_local.pt")
print("global entries:",len(g),"local entries:",len(l))
if len(g)>0:
    z=g[0].get("z_H") or g[0].get("z_L")
    if z is not None:
        print("example pooled shape:", z.shape)
PY

# 6) Train linear probes (global is_solved, local per_cell_correct)
python scripts/train_linear_probes.py --probes_dir results/probes

# 7) Show trained probe files
ls -lh results/probes/global_probe_*.pt results/probes/local_probe_*.pt || true

echo "Probe workflow completed successfully."

#PROBE_BATCH_SIZE=8 HRM_HALT_MAX_STEPS=4 MAX_PROBE_BATCHES=5 CPU_ONLY=1 bash scripts/probe_commands.sh
