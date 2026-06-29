#!/usr/bin/env bash
# End-to-end ARC-AGI MI suite runner (designed to run ON A GPU/COMPUTE NODE).
#
# Does, in order:
#   0. Build data/arc-2-aug-1000 (seed=42, num_aug=1000) if missing
#   1. E4/E8 linear probes (+ export probe directions)
#   2. E9b MLP probes
#   3. E9  directed ablation (z_H, z_L)
#   4. H2  causal subspace
#   5. A/H4 policy improvement + decomposition
#   6. Figures
#
# Usage (from repo root, on the GPU node):
#   bash scripts/arc/run_arc_end_to_end.sh smoke   # tiny, fast validation (CPU ok)
#   bash scripts/arc/run_arc_end_to_end.sh full     # full run (GPU)
#
# Env overrides: N, N_POLICY, STEPS, SEEDS, DEVICE, FORCE_REBUILD=1, PY
set -uo pipefail

REPO=/home/leo.rodrigues/HRM
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY="${PY:-/home/leo.rodrigues/miniconda3/envs/hrm/bin/python}"
MODE="${1:-full}"

# ARC_CKPT: run the suite against this checkpoint instead of the default
# arc_common.ARC_CHECKPOINT (the published, misaligned one). Set this to the
# Path-A adapted checkpoint (checkpoints/arc2-adapted-evalonly/step_<N>). When
# set, the dataset for every script is taken from that checkpoint's
# all_config.yaml (data/arc-2-evalonly), so step 0's arc-2-aug-1000 build is
# skipped — never mix dataset builds with an adapted checkpoint's puzzle_emb.
ARC_CKPT="${ARC_CKPT:-}"
CKPT_ARG=(); [ -n "$ARC_CKPT" ] && CKPT_ARG=(--checkpoint "$ARC_CKPT")

# Default to GPU whenever torch can see one (override with DEVICE=cpu).
DEFAULT_DEVICE=$("$PY" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)

if [ "$MODE" = "smoke" ]; then
    N="${N:-6}"; N_POLICY="${N_POLICY:-4}"; STEPS="${STEPS:-0,1}"
    SEEDS="${SEEDS:-0,1}"; DEVICE="${DEVICE:-$DEFAULT_DEVICE}"; EPOCHS=4; POS=48
    ROOT=results/arc_smoke
else
    N="${N:-500}"; N_POLICY="${N_POLICY:-500}"; STEPS="${STEPS:-0,1,2,4,8,15}"
    SEEDS="${SEEDS:-0,1,2,3,4}"; DEVICE="${DEVICE:-$DEFAULT_DEVICE}"; EPOCHS=30; POS=128
    ROOT=results/arc
fi

LP="$ROOT/hardened/linear_probes"
MLP="$ROOT/hardened/linear_probes_mlp"
DA="$ROOT/hardened/directed_ablation"
LOG="logs/arc_run_${MODE}.log"
mkdir -p logs "$LP" "$MLP" "$DA" "$ROOT/causal_subspace" \
         "$ROOT/policy_improvement" "$ROOT/policy_decomposition" \
         results/reports/arc_figures
: > "$LOG"

run() {  # run <label> <cmd...>; non-fatal, logged
    local label="$1"; shift
    echo "[$(date +%T)] >>> $label" | tee -a "$LOG"
    if "$@" >>"$LOG" 2>&1; then
        echo "[$(date +%T)] <<< $label OK" | tee -a "$LOG"
    else
        echo "[$(date +%T)] <<< $label FAILED (exit $?) — see $LOG" | tee -a "$LOG"
    fi
}

echo "[arc_run] host=$(hostname) mode=$MODE device=$DEVICE N=$N steps=$STEPS seeds=$SEEDS" | tee -a "$LOG"
echo "[arc_run] checkpoint=${ARC_CKPT:-<default ARC_CHECKPOINT>}" | tee -a "$LOG"
"$PY" -c "import torch; print('[arc_run] torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

# ── 0. dataset ────────────────────────────────────────────────────────────
# When ARC_CKPT is set, the dataset is dictated by that checkpoint's
# all_config.yaml (the analysis scripts read it via load_model_and_dataloader),
# so we must NOT build/assume arc-2-aug-1000 — doing so would reintroduce the
# puzzle_emb index mismatch. Only build the full set for the default flow.
if [ -n "$ARC_CKPT" ]; then
    echo "[arc_run] ARC_CKPT set — dataset comes from the checkpoint config; skipping step 0 build" | tee -a "$LOG"
elif [ "${FORCE_REBUILD:-0}" = "1" ] || [ ! -f data/arc-2-aug-1000/test/dataset.json ]; then
    echo "[$(date +%T)] >>> build data/arc-2-aug-1000 (seed=42 num_aug=1000)" | tee -a "$LOG"
    "$PY" -u -c "
import sys; sys.path.insert(0, 'dataset')
import build_arc_dataset as b
cfg = b.DataProcessConfig(dataset_dirs=['dataset/raw-data/ARC-AGI-2/data'],
                          output_dir='data/arc-2-aug-1000', seed=42, num_aug=1000)
print('CONFIG', cfg.dataset_dirs, cfg.output_dir, cfg.num_aug, cfg.seed, flush=True)
b.convert_dataset(cfg); print('BUILD DONE', flush=True)
" >>"$LOG" 2>&1 && echo "[$(date +%T)] <<< build OK" | tee -a "$LOG" \
    || { echo "[$(date +%T)] build FAILED — aborting" | tee -a "$LOG"; exit 2; }
else
    echo "[arc_run] dataset already built — skipping" | tee -a "$LOG"
fi

# ── 1. linear probes (+ directions) ───────────────────────────────────────
run "E4/E8 linear probes" "$PY" -u scripts/arc/arc_linear_probes.py \
    --num_puzzles "$N" --steps "$STEPS" --seeds "$SEEDS" --epochs "$EPOCHS" \
    --positions_per_sample "$POS" --probe_type linear --save_probe_weights \
    --device "$DEVICE" --output_dir "$LP" "${CKPT_ARG[@]}"

# ── 2. MLP probes ─────────────────────────────────────────────────────────
run "E9b MLP probes" "$PY" -u scripts/arc/arc_linear_probes.py \
    --num_puzzles "$N" --steps "$STEPS" --seeds "$SEEDS" --epochs "$EPOCHS" \
    --positions_per_sample "$POS" --probe_type mlp \
    --device "$DEVICE" --output_dir "$MLP" "${CKPT_ARG[@]}"

# ── 3. directed ablation ──────────────────────────────────────────────────
run "E9 directed ablation z_H" "$PY" -u scripts/arc/directed_ablation_arc.py \
    --num_puzzles "$N" --z_level H --probe_weights "$LP/probe_weights.pt" \
    --device "$DEVICE" --output_dir "$DA" "${CKPT_ARG[@]}"
run "E9 directed ablation z_L" "$PY" -u scripts/arc/directed_ablation_arc.py \
    --num_puzzles "$N" --z_level L --probe_weights "$LP/probe_weights.pt" \
    --device "$DEVICE" --output_dir "${DA}_zL" "${CKPT_ARG[@]}"

# ── 4. causal subspace ────────────────────────────────────────────────────
run "H2 causal subspace" "$PY" -u scripts/analysis/causal_subspace.py \
    --task arc --num_puzzles "$N" --n_pca "$N" \
    --probe_weights "$LP/probe_weights.pt" \
    --device "$DEVICE" --output_dir "$ROOT/causal_subspace" "${CKPT_ARG[@]}"

# ── 5. policy A / H4 ──────────────────────────────────────────────────────
run "A policy improvement" "$PY" -u scripts/analysis/policy_improvement.py \
    --task arc --num_puzzles "$N_POLICY" \
    --device "$DEVICE" --output_dir "$ROOT/policy_improvement" "${CKPT_ARG[@]}"
run "H4 policy decomposition" "$PY" -u scripts/analysis/policy_decomposition.py \
    --task arc --num_puzzles "$N_POLICY" \
    --device "$DEVICE" --output_dir "$ROOT/policy_decomposition" "${CKPT_ARG[@]}"

# ── 6. figures ────────────────────────────────────────────────────────────
run "figures" "$PY" -u scripts/arc/plot_arc_figures.py \
    --results_dir "$ROOT" --output_dir results/reports/arc_figures

echo "[$(date +%T)] ARC SUITE COMPLETE (mode=$MODE)  outputs under $ROOT/" | tee -a "$LOG"
