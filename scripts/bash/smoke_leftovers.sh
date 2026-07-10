#!/bin/bash
# Fast (<5 min) end-to-end smoke of every leftover experiment whose code changed
# this session:
#   R2 — mean-ablation (engine + controlled_ablation driver)
#   R3 — TopK SAE training  +  reconstruction_only control on the 3 causal scripts
# Tiny N / dict / epochs: this validates that each code path RUNS end-to-end, it
# is NOT a scientific run. All output goes under results/_smoke/ (throwaway).
#
#   bash scripts/bash/smoke_leftovers.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# Activate env with -u relaxed: conda's activate.d cuda hooks reference unbound vars.
set +u
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${HRM_CONDA_ENV:-hrm}"; fi
set -u
export DISABLE_COMPILE=1

OUT=results/_smoke; mkdir -p "$OUT" logs
SUD=checkpoints/sapientinc-sudoku-extreme/checkpoint.pt
MAZE=checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint
ARC=checkpoints/arc2-adapted-evalonly/step_7391
SUD_ACT=results/Sudoku/sae_study/activations_zH.pt

rc=0
run () {  # run <name> <cmd...>
    local name="$1"; shift
    local t0; t0=$(date +%s)
    printf '=== [smoke] %-18s ' "$name"
    if "$@" > "logs/smoke_${name}.log" 2>&1; then
        echo "PASS ($(($(date +%s) - t0))s)"
    else
        echo "FAIL ($(($(date +%s) - t0))s)  -> logs/smoke_${name}.log"; rc=1
    fi
}

# R2 — mean-ablation, Sudoku, N=4 (writes zH_mean/ zL_mean/)
run r2_mean_sudoku python scripts/controlled/controlled_ablation.py \
    --checkpoint "$SUD" --ablation_mode mean \
    --num_puzzles 4 --mean_estimation_puzzles 4 --z_level both \
    --output_dir "$OUT/r2_mean_sudoku"

# R3 — TopK SAE training (tiny dict + 1 epoch)
run r3_topk_train python scripts/sae/sae_train.py \
    --activation topk --k 8 --dict_size 256 --epochs 1 \
    --activations_path "$SUD_ACT" --output_dir "$OUT/topk_train"

# R3 — Sudoku causal + reconstruction_only field (N=2; probe dirs auto-skipped)
run r3_causal_sudoku python scripts/sae/sae_causal_ablation.py \
    --sae_path "$OUT/topk_train/sae_d256_topk8.pt" --activations_path "$SUD_ACT" \
    --n_puzzles 2 --top_k 3 --n_random_features 3 \
    --output_dir "$OUT/r3_causal_sudoku"

# R3 — ARC causal + reconstruction_only condition (N=2)
# NB: explicit --activations_path — the script default (results/arc/...) is stale
# after the Sudoku/Maze/ARC results reorg (now results/ARC/arc/...).
run r3_recon_arc python scripts/arc/sae_causal_ablation_arc.py \
    --sae_path results/ARC/arc/sae_study/sae_d2048_l10.01.pt --checkpoint "$ARC" \
    --activations_path results/ARC/arc/sae_study/activations_zH.pt \
    --n_puzzles 2 --top_k 3 --n_random_features 3 \
    --output_dir "$OUT/r3_recon_arc"

# R3 — Maze causal + reconstruction_only condition (N=2)
run r3_recon_maze python scripts/maze/sae_causal_ablation_maze.py \
    --sae_path results/Maze/sae_study/sae_d2048_l10.01.pt --checkpoint "$MAZE" \
    --activations_path results/Maze/sae_study/activations_zH.pt \
    --n_puzzles 2 --top_k 3 --n_random_features 3 \
    --output_dir "$OUT/r3_recon_maze"

echo
echo "=== smoke summary: rc=$rc  (0 = all passed) ==="
echo "check the new fields:  grep -o reconstruction_only $OUT/r3_recon_arc/aggregate.json"
exit "$rc"
