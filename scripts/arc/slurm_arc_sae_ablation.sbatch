#!/usr/bin/env bash
# Rerun of step [4/5]+[5/5] of _run_remaining_arc_suite.sh: the ARC SAE causal
# ablation was killed at the 5h srun time limit after steps 1-3 consumed ~4.5h.
# Steps 1-3's artifacts (patching_full_steps, activations_zH.pt, sae_d2048_l10.01.pt)
# are on disk, so this job runs only the ablation + figure regeneration.
# Submit:  sbatch -p gpu --gres=gpu:1 scripts/arc/slurm_arc_sae_ablation.sbatch
#SBATCH -J arc_sae_ablate
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o logs/arc/%x_%j.out
#SBATCH -e logs/arc/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/arc

CKPT="${CKPT:-checkpoints/arc2-adapted-evalonly/step_7391}"
SAE_DIR="${SAE_DIR:-results/arc/sae_study}"
N_PUZZLES="${N_PUZZLES:-150}"

echo "[arc_sae_ablate] ckpt=$CKPT n_puzzles=$N_PUZZLES"
conda run --no-capture-output -n hrm python -u scripts/arc/sae_causal_ablation_arc.py \
    --checkpoint "$CKPT" \
    --sae_path "$SAE_DIR/sae_d2048_l10.01.pt" \
    --activations_path "$SAE_DIR/activations_zH.pt" \
    --probe_weights results/arc/hardened/linear_probes/probe_weights.pt \
    --n_puzzles "$N_PUZZLES" --top_k 50 --n_random_features 50 \
    --output_dir "$SAE_DIR/causal_ablation"

echo "[arc_sae_ablate] regenerating figures"
conda run --no-capture-output -n hrm python -u scripts/plotting/plot_crosstask_figures.py \
    --output_dir results/reports/paper_aaai_figures
conda run --no-capture-output -n hrm python -u scripts/arc/plot_arc_paper_figures.py \
    --output_dir results/reports/arc_paper_figures || true
echo "[arc_sae_ablate] done"
