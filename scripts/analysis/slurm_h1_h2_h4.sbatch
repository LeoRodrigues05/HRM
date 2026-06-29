#!/usr/bin/env bash
# Experiments H1, H2, H4: Causal subspace + localizability scorecard + policy decomposition.
#
# Dependency order (required): H2 first (builds causal subspace tool consumed by H1),
# then H1, then H4 (independent of H1/H2), then all plots.
#
# Submit:
#   sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_h1_h2_h4.sbatch
#
# Override puzzle count:
#   N=50 sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_h1_h2_h4.sbatch
#
# Run H1 Track A on an HRM-BPTT checkpoint (once available):
#   HRM_BPTT_CKPT=/path/to/bptt/checkpoint.pt sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_h1_h2_h4.sbatch
#
#SBATCH -J hrm_h124
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- Configurable parameters ---
N="${N:-300}"                          # puzzle count for H1/H2
N_H4="${N_H4:-500}"                    # puzzle count for H4 (more for stable per-step stats)
HRM_BPTT_CKPT="${HRM_BPTT_CKPT:-}"   # set to run H1 Track A on BPTT checkpoint

echo "[hrm_h124] starting  N=${N}  N_H4=${N_H4}  $(date)"
echo "[hrm_h124] HRM_BPTT_CKPT='${HRM_BPTT_CKPT}'"

# Create output directories
mkdir -p logs \
         results/controlled/causal_subspace \
         results/maze/causal_subspace \
         results/localizability/hrm_1step \
         results/localizability/hrm_bptt \
         results/localizability/ut_1step \
         results/localizability/ut_bptt \
         results/controlled/policy_decomposition \
         results/maze/policy_decomposition \
         results/baseline_comparison/two_timescale \
         results/reports/causal_subspace_figures \
         results/reports/localizability_figures \
         results/reports/policy_decomposition_figures

# ═══════════════════════════════════════════════════════════
# STEP 1 — H2: Minimal causal subspace (Sudoku primary)
# ═══════════════════════════════════════════════════════════
echo "[hrm_h124] === H2: causal subspace (sudoku) ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/causal_subspace.py \
    --task sudoku \
    --num_puzzles "$N" \
    --n_pca "$N" \
    --output_dir results/controlled/causal_subspace \
    --device cuda

echo "[hrm_h124] === H2: causal subspace (maze, secondary) ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/causal_subspace.py \
    --task maze \
    --num_puzzles "$N" \
    --n_pca "$N" \
    --output_dir results/maze/causal_subspace \
    --device cuda || echo "[hrm_h124] maze causal subspace failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 2 — H1 Track A: HRM localizability scorecard
# ═══════════════════════════════════════════════════════════
echo "[hrm_h124] === H1 Track A: HRM 1-step scorecard ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/localizability_scorecard.py \
    --tag hrm_1step \
    --task sudoku \
    --num_puzzles "$N" \
    --output_dir results/localizability/hrm_1step \
    --device cuda

# Run HRM-BPTT scorecard only if the checkpoint path is provided
if [ -n "$HRM_BPTT_CKPT" ]; then
    echo "[hrm_h124] === H1 Track A: HRM BPTT scorecard ==="
    conda run --no-capture-output -n hrm python -u scripts/analysis/localizability_scorecard.py \
        --tag hrm_bptt \
        --checkpoint "$HRM_BPTT_CKPT" \
        --task sudoku \
        --num_puzzles "$N" \
        --output_dir results/localizability/hrm_bptt \
        --device cuda
else
    echo "[hrm_h124] HRM_BPTT_CKPT not set — skipping hrm_bptt scorecard"
    echo "[hrm_h124] (Set HRM_BPTT_CKPT=<path> to enable when checkpoint is available)"
fi

# ═══════════════════════════════════════════════════════════
# STEP 3 — H1 Track B: UT pair localizability scorecard
# ═══════════════════════════════════════════════════════════
echo "[hrm_h124] === H1 Track B: UT 1-step scorecard ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/baseline_localizability.py \
    --tag ut_1step \
    --num_puzzles "$N" \
    --output_dir results/localizability/ut_1step \
    --device cuda

echo "[hrm_h124] === H1 Track B: UT BPTT scorecard ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/baseline_localizability.py \
    --tag ut_bptt \
    --num_puzzles "$N" \
    --output_dir results/localizability/ut_bptt \
    --device cuda || echo "[hrm_h124] ut_bptt skipped (no checkpoint yet — expected)"

# ═══════════════════════════════════════════════════════════
# STEP 4 — H4: Policy decomposition (HRM)
# ═══════════════════════════════════════════════════════════
echo "[hrm_h124] === H4: policy decomposition (sudoku) ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/policy_decomposition.py \
    --task sudoku \
    --num_puzzles "$N_H4" \
    --output_dir results/controlled/policy_decomposition \
    --device cuda

echo "[hrm_h124] === H4: policy decomposition (maze) ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/policy_decomposition.py \
    --task maze \
    --num_puzzles "$N_H4" \
    --output_dir results/maze/policy_decomposition \
    --device cuda || echo "[hrm_h124] maze policy decomposition failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 5 — H4: Two-timescale baseline (UT vs HRM)
# ═══════════════════════════════════════════════════════════
echo "[hrm_h124] === H4: two-timescale baseline ==="
conda run --no-capture-output -n hrm python -u scripts/analysis/two_timescale_baseline.py \
    --tag ut_1step \
    --num_puzzles "$N" \
    --output_dir results/baseline_comparison/two_timescale \
    --device cuda || echo "[hrm_h124] two_timescale_baseline failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 6 — Figures
# ═══════════════════════════════════════════════════════════
echo "[hrm_h124] === Figures ==="

conda run --no-capture-output -n hrm python -u scripts/analysis/plot_causal_subspace.py \
    --task sudoku \
    --outdir results/reports/causal_subspace_figures

conda run --no-capture-output -n hrm python -u scripts/analysis/plot_causal_subspace.py \
    --task maze \
    --outdir results/reports/causal_subspace_figures || true

conda run --no-capture-output -n hrm python -u scripts/analysis/plot_localizability.py \
    --outdir results/reports/localizability_figures

conda run --no-capture-output -n hrm python -u scripts/analysis/plot_policy_decomposition.py \
    --outdir results/reports/policy_decomposition_figures

echo "[hrm_h124] done  $(date)"
echo "[hrm_h124] ========================================"
echo "[hrm_h124] Key outputs:"
echo "[hrm_h124]   results/controlled/causal_subspace/subspace_curve.json"
echo "[hrm_h124]   results/localizability/hrm_1step/scorecard.json"
echo "[hrm_h124]   results/localizability/ut_1step/scorecard.json"
echo "[hrm_h124]   results/controlled/policy_decomposition/aggregate.json"
echo "[hrm_h124]   results/baseline_comparison/two_timescale/two_timescale.json"
echo "[hrm_h124]   results/reports/*/  (all figures)"
echo "[hrm_h124] ========================================"
echo "[hrm_h124] To add HRM-BPTT (Track A airtight control):"
echo "[hrm_h124]   HRM_BPTT_CKPT=<path> sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_h1_h2_h4.sbatch"
