#!/bin/bash

# Example Activation Patching Experiments for HRM
# This script demonstrates common activation patching scenarios
# 
# Features:
# - Forward experiment: Patch source activations into target puzzle
# - Inverse experiment: Patch target activations into source puzzle (automatic)
# - Multiple runs for stability verification
# - HTML reports with colored Sudoku grids

set -euo pipefail

# Set checkpoint path
CHECKPOINT="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt"
OUTPUT_DIR="results/activation_patching_examples"
NUM_RUNS=5  # Number of repeated runs for stability

echo "=============================================="
echo "Activation Patching Examples for HRM"
echo "=============================================="
echo ""

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

# Example 1: Basic patching - patch all levels at all steps (skipping step 0)
echo "Example 1: Basic full patching (both z_H and z_L, steps 1-7)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 1 \
    --patch_level both \
    --patch_steps 1,2,3,4,5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example1_full_patch"

echo ""
echo ""

# Example 2: H-level only patching (global/controller stream)
echo "Example 2: High-level reasoning patching (z_H only)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 2 \
    --patch_level H \
    --patch_steps 1,2,3,4,5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example2_h_level"

echo ""
echo ""

# Example 3: L-level only patching (local/per-cell stream)
echo "Example 3: Low-level reasoning patching (z_L only)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 2 \
    --patch_level L \
    --patch_steps 1,2,3,4,5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example3_l_level"

echo ""
echo ""

# Example 4: Early step patching (steps 1-3)
echo "Example 4: Early reasoning steps (steps 1-3)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 1 \
    --target_puzzle_idx 3 \
    --patch_level both \
    --patch_steps 1,2,3 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example4_early_steps"

echo ""
echo ""

# Example 5: Late step patching (steps 5-7)
echo "Example 5: Late reasoning steps (steps 5-7)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 1 \
    --target_puzzle_idx 3 \
    --patch_level both \
    --patch_steps 5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example5_late_steps"

echo ""
echo ""

# Example 6: Single step patching (step 3 only)
echo "Example 6: Single step patching (step 3 only)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 2 \
    --target_puzzle_idx 4 \
    --patch_level both \
    --patch_steps 3 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example6_single_step"

echo ""
echo ""

# Example 7: Position-specific patching (first row in Sudoku - positions 0-8)
echo "Example 7: Position-specific patching (first row, positions 0-8)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 3 \
    --target_puzzle_idx 5 \
    --patch_level both \
    --patch_steps 1,2,3,4,5,6,7 \
    --patch_positions 0,1,2,3,4,5,6,7,8 \
    --num_runs $NUM_RUNS \
    --report_html activation_patching_report.html \
    --output_dir "$OUTPUT_DIR/example7_position_specific"

echo ""
echo ""

echo "=============================================="
echo "All examples completed!"
echo "=============================================="
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Each example generates:"
echo "  - *_forward.yaml      : Forward experiment results (source→target)"
echo "  - *_inverse.yaml      : Inverse experiment results (target→source)"
echo "  - *_report.html       : Forward HTML report with colored grids"
echo "  - *_report_inverse.html : Inverse HTML report"
echo "  - activations_*.pt    : Cached activations for analysis"
echo ""
echo "To view HTML reports, open in browser:"
echo "  firefox $OUTPUT_DIR/example1_full_patch/activation_patching_report.html"
echo ""
echo "To run batch experiments on multiple puzzles:"
echo "  python scripts/patching/batch_activation_patching.py \\"
echo "      --checkpoint $CHECKPOINT \\"
echo "      --num_puzzles 10 \\"
echo "      --output_dir results/batch_experiments"
