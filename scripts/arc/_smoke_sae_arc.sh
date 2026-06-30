#!/usr/bin/env bash
# Tiny end-to-end smoke for the ARC SAE pipeline (collect -> train -> ablate).
# Validates wiring on minimal inputs before the multi-hour full run. ~3 min on GPU.
set -euo pipefail
PY=/home/leo.rodrigues/miniconda3/envs/hrm/bin/python
cd "$(dirname "$0")/../.."
D=results/arc_smoke/sae_study

echo "== SAE smoke [1/3] collect =="
$PY scripts/arc/sae_collect_activations_arc.py --n_puzzles 6 --output_dir "$D"

echo "== SAE smoke [2/3] train (d=256, 3 epochs) =="
$PY scripts/sae/sae_train.py --activations_path "$D/activations_zH.pt" \
    --activation_level z_H --dict_size 256 --l1_coeff 0.01 --epochs 3 --output_dir "$D"

echo "== SAE smoke [3/3] causal ablation =="
$PY scripts/arc/sae_causal_ablation_arc.py \
    --sae_path "$D/sae_d256_l10.01.pt" --activations_path "$D/activations_zH.pt" \
    --probe_weights results/arc/hardened/linear_probes/probe_weights.pt \
    --n_puzzles 3 --top_k 3 --n_random_features 3 --output_dir "$D/causal_ablation"

echo "== SAE smoke COMPLETE =="
