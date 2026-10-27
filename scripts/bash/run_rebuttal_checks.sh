#!/usr/bin/env bash
# Rebuttal checks E0, E1 and E7.
#
#   E0  probe base rates and balanced metrics     (Sudoku, Maze)
#   E1  SAE reconstruction-only and error-preserving ablations
#       (Sudoku released model, Maze, BPTT-study stock and BPTT arms)
#   E7  clean accuracy and perturbation sensitivity of the BPTT-study arms
#
# Run from the repo root on a GPU machine:
#   bash scripts/bash/run_rebuttal_checks.sh            # all stages
#   bash scripts/bash/run_rebuttal_checks.sh e1         # one stage: e0 | e1 | e7
#   N_PUZZLES=5 bash scripts/bash/run_rebuttal_checks.sh e1   # smoke test
#
# Outputs go to new directories, so published results are not overwritten.
set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source scripts/bash/_activate_env.sh 2>/dev/null || true
export DISABLE_COMPILE=1
mkdir -p logs

STAGE="${1:-all}"
DEVICE="${DEVICE:-cuda}"
N="${N_PUZZLES:-}"
BPTT_ROOT="results/Sudoku/sae_study/bptt_study"

need() {
    local missing=0
    for f in "$@"; do
        if [ ! -e "$f" ]; then echo "  missing: $f"; missing=1; fi
    done
    return $missing
}

e0() {
    echo "=== E0: probe base rates and balanced metrics ==="
    # Same settings as the published Sudoku probes (results/Sudoku/probes/e8_constraint_probes/_meta.json)
    python scripts/probes/e8_constraint_probes.py \
        --n_puzzles "${N:-500}" --seeds 0,1,2,3,4 --steps 0,4,8,12,15 --epochs 60 \
        --z_level both --device "$DEVICE" \
        --output_dir results/Sudoku/probes/e8_constraint_probes_e0 \
        2>&1 | tee logs/e0_sudoku_probes.log
    # Maze already records majority baselines; this rerun adds balanced accuracy, AUROC and MCC.
    python scripts/maze/linear_probes_maze.py \
        --num_puzzles "${N:-1000}" --steps 0,1,2,4,8,15 --seeds 0,1,2,3,4 --device "$DEVICE" \
        --output_dir results/Maze/hardened/linear_probes_e0 \
        2>&1 | tee logs/e0_maze_probes.log
}

e1() {
    echo "=== E1: SAE reconstruction controls ==="
    if need results/Sudoku/sae_study/sae_d2048_l10.01.pt results/Sudoku/sae_study/activations_zH.pt; then
        python scripts/sae/sae_error_controls.py --task sudoku \
            --sae_path results/Sudoku/sae_study/sae_d2048_l10.01.pt \
            --activations_path results/Sudoku/sae_study/activations_zH.pt \
            --n_puzzles "${N:-300}" --norm_control --device "$DEVICE" \
            --output_dir results/Sudoku/sae_study/error_controls \
            2>&1 | tee logs/e1_sudoku.log
    else
        echo "  skipping Sudoku (run scripts/sae/sae_collect_activations.py and the SAE sweep first)"
    fi

    if need results/Maze/sae_study/sae_d2048_l10.01.pt results/Maze/sae_study/activations_zH.pt; then
        # Maze sequences are 11x longer; the replace mode was already run for the paper.
        python scripts/sae/sae_error_controls.py --task maze \
            --sae_path results/Maze/sae_study/sae_d2048_l10.01.pt \
            --activations_path results/Maze/sae_study/activations_zH.pt \
            --n_puzzles "${N:-300}" --single_modes error_preserving --joint_modes error_preserving \
            --n_random_draws 10 --norm_control --device "$DEVICE" \
            --output_dir results/Maze/sae_study/error_controls \
            2>&1 | tee logs/e1_maze.log
    else
        echo "  skipping Maze (missing SAE or activation bank)"
    fi

    for arm in hrm_stock hrm_bptt; do
        if need "checkpoints/bptt_study/$arm" "$BPTT_ROOT/$arm/sae_d2048_l10.003.pt" \
                "$BPTT_ROOT/$arm/activations_zH.pt"; then
            # Same 200 puzzles as the paper's Table 2.
            python scripts/sae/sae_error_controls.py --task sudoku \
                --checkpoint_dir "checkpoints/bptt_study/$arm" \
                --sae_path "$BPTT_ROOT/$arm/sae_d2048_l10.003.pt" \
                --activations_path "$BPTT_ROOT/$arm/activations_zH.pt" \
                --puzzle_indices "$BPTT_ROOT/$arm/causal_ablation/puzzle_indices.json" \
                --n_puzzles "${N:-200}" --norm_control --device "$DEVICE" \
                --output_dir "$BPTT_ROOT/$arm/error_controls" \
                2>&1 | tee "logs/e1_$arm.log"
        else
            echo "  skipping $arm (unpack bptt_study.tar.zst so the paths above exist)"
        fi
    done
}

e7() {
    echo "=== E7: BPTT-study arms, clean accuracy and noise sensitivity ==="
    need checkpoints/sapientinc-sudoku-extreme checkpoints/bptt_study/hrm_stock checkpoints/bptt_study/hrm_bptt \
        || echo "  arms with a missing checkpoint directory are skipped"
    python scripts/analysis/bptt_arm_eval.py \
        --n_puzzles "${N:-200}" --device "$DEVICE" \
        --output_dir "$BPTT_ROOT/arm_eval" \
        2>&1 | tee logs/e7_bptt_arms.log
}

case "$STAGE" in
    e0) e0 ;;
    e1) e1 ;;
    e7) e7 ;;
    # A failed stage is reported and the next one still runs.
    all) e7 || echo "E7 failed (see logs/)"
         e1 || echo "E1 failed (see logs/)"
         e0 || echo "E0 failed (see logs/)" ;;
    *) echo "usage: $0 [e0|e1|e7|all]" >&2; exit 1 ;;
esac
