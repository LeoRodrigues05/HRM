#!/usr/bin/env bash
# ARC-AGI MI suite: end-to-end replication of the Sudoku/Maze experiments on the
# HRM ARC-2 checkpoint.
#
# Pipeline (each stage feeds the next):
#   0. (prerequisite) build data/arc-2-aug-1000  +  download ARC-2 checkpoint
#   1. Linear probes  (E4/E8 decodability) + export probe directions
#   2. MLP probes     (E9b non-linear decodability)
#   3. Directed ablation (E9 causal validation) using step-1 directions
#   4. Causal subspace   (H2) using step-1 directions
#   5. Policy improvement + decomposition (A/H4)
#   6. Figures
#
# Submit:
#   sbatch -p gpu --gres=gpu:1 scripts/arc/slurm_arc_suite.sbatch
# Override puzzle count:
#   N=50 sbatch -p gpu --gres=gpu:1 scripts/arc/slurm_arc_suite.sbatch
#
#SBATCH -J hrm_arc
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

N="${N:-500}"                          # puzzle count for probes / ablation
N_POLICY="${N_POLICY:-500}"            # puzzle count for policy A/H4
STEPS="${STEPS:-0,1,2,4,8,15}"
SEEDS="${SEEDS:-0,1,2,3,4}"
RUN="${RUN:-conda run --no-capture-output -n hrm python -u}"

LP=results/arc/hardened/linear_probes
MLP=results/arc/hardened/linear_probes_mlp
DA=results/arc/hardened/directed_ablation

echo "[hrm_arc] starting N=${N} steps=${STEPS} seeds=${SEEDS}  $(date)"
mkdir -p logs "$LP" "$MLP" "$DA" \
         results/arc/causal_subspace \
         results/arc/policy_improvement \
         results/arc/policy_decomposition \
         results/reports/arc_figures

# ═══════════════════════════════════════════════════════════
# STEP 1 — E4/E8: linear probes + export directions for E9/H2
# ═══════════════════════════════════════════════════════════
echo "[hrm_arc] === E4/E8: linear probes (+ probe_weights) ==="
$RUN scripts/arc/arc_linear_probes.py \
    --num_puzzles "$N" --steps "$STEPS" --seeds "$SEEDS" \
    --probe_type linear --save_probe_weights \
    --output_dir "$LP" --device cuda

# ═══════════════════════════════════════════════════════════
# STEP 2 — E9b: MLP (non-linear) probes + linear baseline + delta
# ═══════════════════════════════════════════════════════════
echo "[hrm_arc] === E9b: MLP probes ==="
$RUN scripts/arc/arc_linear_probes.py \
    --num_puzzles "$N" --steps "$STEPS" --seeds "$SEEDS" \
    --probe_type mlp \
    --output_dir "$MLP" --device cuda || echo "[hrm_arc] MLP probes failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 3 — E9: directed ablation along probe directions (z_H and z_L)
# ═══════════════════════════════════════════════════════════
echo "[hrm_arc] === E9: directed ablation (z_H) ==="
$RUN scripts/arc/directed_ablation_arc.py \
    --num_puzzles "$N" --z_level H \
    --probe_weights "$LP/probe_weights.pt" \
    --output_dir "$DA" --device cuda

echo "[hrm_arc] === E9: directed ablation (z_L) ==="
$RUN scripts/arc/directed_ablation_arc.py \
    --num_puzzles "$N" --z_level L \
    --probe_weights "$LP/probe_weights.pt" \
    --output_dir "${DA}_zL" --device cuda || echo "[hrm_arc] z_L ablation failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 4 — H2: minimal causal subspace (uses step-1 directions)
# ═══════════════════════════════════════════════════════════
echo "[hrm_arc] === H2: causal subspace ==="
$RUN scripts/analysis/causal_subspace.py \
    --task arc --num_puzzles "$N" --n_pca "$N" \
    --probe_weights "$LP/probe_weights.pt" \
    --output_dir results/arc/causal_subspace --device cuda \
    || echo "[hrm_arc] causal subspace failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 5 — A/H4: policy improvement + decomposition
# ═══════════════════════════════════════════════════════════
echo "[hrm_arc] === A: policy improvement ==="
$RUN scripts/analysis/policy_improvement.py \
    --task arc --num_puzzles "$N_POLICY" \
    --output_dir results/arc/policy_improvement --device cuda \
    || echo "[hrm_arc] policy improvement failed (non-fatal)"

echo "[hrm_arc] === H4: policy decomposition ==="
$RUN scripts/analysis/policy_decomposition.py \
    --task arc --num_puzzles "$N_POLICY" \
    --output_dir results/arc/policy_decomposition --device cuda \
    || echo "[hrm_arc] policy decomposition failed (non-fatal)"

# ═══════════════════════════════════════════════════════════
# STEP 6 — Figures
# ═══════════════════════════════════════════════════════════
echo "[hrm_arc] === Figures ==="
$RUN scripts/arc/plot_arc_figures.py \
    --results_dir results/arc \
    --output_dir results/reports/arc_figures || true

echo "[hrm_arc] done  $(date)"
echo "[hrm_arc] Key outputs:"
echo "[hrm_arc]   $LP/probe_summary.json   (linear decodability)"
echo "[hrm_arc]   $MLP/probe_summary.json  (MLP vs linear)"
echo "[hrm_arc]   $DA/analysis.json         (directed ablation)"
echo "[hrm_arc]   results/arc/causal_subspace/  results/arc/policy_*/"
echo "[hrm_arc]   results/reports/arc_figures/  (figures)"
