#!/usr/bin/env bash
# Orchestrate the remaining ARC interpretability experiments for the AAAI reframe.
# Runs sequentially inside ONE srun GPU allocation (no nested srun) so it never
# contends with itself under the per-user memory QOS. Aborts on first failure.
#
# Steps: cross-puzzle patching (full) -> SAE collect -> SAE train -> SAE ablate
#        -> regenerate cross-task + ARC paper figures.
#
# Launch (from repo root):
#   srun -N1 -n1 -w ws-l3-019 --mem=96G --gres=gpu:1 --time=05:00:00 \
#       bash scripts/arc/_run_remaining_arc_suite.sh
set -euo pipefail

PY=/home/leo.rodrigues/miniconda3/envs/hrm/bin/python
CKPT=checkpoints/arc2-adapted-evalonly/step_7391
SAE_DIR=results/arc/sae_study
cd "$(dirname "$0")/../.."

echo "================ [1/5] ARC cross-puzzle patching (full) ================"
# step 0 is a no-op in ACTV1 (post-reset carry = H_init/L_init for all puzzles), so
# the donor sweep starts at step 1.
$PY scripts/arc/patching_arc.py --checkpoint "$CKPT" \
    --num_pairs 100 --patch_levels H,L --patch_steps 1,2,4,6,8,10,12,15 \
    --output_dir results/arc/patching_full_steps

echo "================ [2/5] ARC SAE: collect z_H activations ================"
$PY scripts/arc/sae_collect_activations_arc.py --checkpoint "$CKPT" \
    --n_puzzles 300 --output_dir "$SAE_DIR"

echo "================ [3/5] ARC SAE: train (d=2048, l1=0.01) ================"
$PY scripts/sae/sae_train.py \
    --activations_path "$SAE_DIR/activations_zH.pt" --activation_level z_H \
    --dict_size 2048 --l1_coeff 0.01 --epochs 30 --output_dir "$SAE_DIR"

echo "================ [4/5] ARC SAE: causal ablation ========================"
$PY scripts/arc/sae_causal_ablation_arc.py --checkpoint "$CKPT" \
    --sae_path "$SAE_DIR/sae_d2048_l10.01.pt" \
    --activations_path "$SAE_DIR/activations_zH.pt" \
    --probe_weights results/arc/hardened/linear_probes/probe_weights.pt \
    --n_puzzles 150 --top_k 50 --n_random_features 50 \
    --output_dir "$SAE_DIR/causal_ablation"

echo "================ [5/5] Regenerate figures ==============================="
$PY scripts/plotting/plot_crosstask_figures.py --output_dir results/reports/paper_aaai_figures
$PY scripts/arc/plot_arc_paper_figures.py --output_dir results/reports/arc_paper_figures || true

echo "================ ARC remaining suite COMPLETE =========================="
