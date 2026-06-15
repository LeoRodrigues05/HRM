#!/usr/bin/env bash
# Tiny GPU smoke test for the maze parity suite. Validates that the rewritten
# probe script (linear + mlp) and the SAE collect+sweep run end-to-end on GPU
# at trivial N before the full sbatch job is submitted.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # conda nvcc activate.d trips set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

SMOKE=/tmp/maze_smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE"

echo "==================== [1/3] LINEAR probe smoke ===================="
python -u scripts/maze/linear_probes_maze.py \
  --num_puzzles 8 --steps 0,1 --epochs 5 --positions_per_sample 64 \
  --seeds 0,1 --device cuda --train_device cuda \
  --output_dir "$SMOKE/linear_probes" || { echo "LINEAR FAILED"; exit 1; }

echo "==================== [2/3] MLP probe smoke ===================="
python -u scripts/maze/linear_probes_maze.py \
  --num_puzzles 8 --steps 0,1 --epochs 5 --positions_per_sample 64 \
  --probe_type mlp --seeds 0,1 --device cuda --train_device cuda \
  --output_dir "$SMOKE/mlp_probes" || { echo "MLP FAILED"; exit 1; }

echo "==================== [3/3] SAE collect+sweep smoke ===================="
python -u scripts/sae/sae_collect_activations_maze.py \
  --n_puzzles 6 --store_dtype float32 --device cuda \
  --output_dir "$SMOKE/sae_study" || { echo "SAE COLLECT FAILED"; exit 1; }
python -u scripts/sae/sae_sweep.py \
  --activations_path "$SMOKE/sae_study/activations_zH.pt" \
  --dict_sizes 256,512 --l1_coeffs 0.01 --epochs 2 --batch_size 2048 \
  --device cuda --output_dir "$SMOKE/sae_study" || { echo "SAE SWEEP FAILED"; exit 1; }

echo "==================== SMOKE OK ===================="
echo "--- linear summary keys ---"
python - <<'PY'
import json
for name in ("linear_probes","mlp_probes"):
    d=json.load(open(f"/tmp/maze_smoke/{name}/probe_results.json"))
    g=d["global_probes"]; l=d["local_probes"]
    print(name, "global", len(g), "local", len(l),
          "| sample:", {k:g[0].get(k) for k in ("stream","step","target","score_mean","n_seeds","n_train","n_test")} if g else None)
import csv
rows=list(csv.DictReader(open("/tmp/maze_smoke/sae_study/sweep_results.csv")))
print("sae sweep rows:", [(r["dict_size"],r["l1_coeff"],r["L0"]) for r in rows])
PY
echo "==================== DONE ===================="
