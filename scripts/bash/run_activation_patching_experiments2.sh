#!/bin/bash

# Comprehensive Activation Patching Experiments
# Based on systematic exploration of z_H and z_L patching effects
#
# Experiments:
# 1. z_H only patching (global reasoning stream)
#    - Early steps (1,2,3)
#    - Late steps (5,6,7)
#    - Forward and inverse (automatic)
#
# 2. z_L only patching (local/per-cell stream)
#    - Early steps: first two rows, one subgrid, all positions
#    - Late steps: first two rows, one subgrid, all positions
#    - All steps
#    - Forward and inverse (automatic)
#    - Modified dataset experiment (3 rows missing from label as input)
#
# All experiments run both forward and inverse directions automatically

set -euo pipefail

# Configuration
CHECKPOINT="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt"
OUTPUT_BASE="results/activation_patching_experiments2"
NUM_RUNS=5

# Use consistent source/target puzzle pairs for comparability
SOURCE_IDX=111
TARGET_IDX=220

# Sudoku position mappings
FIRST_ROW="0,1,2,3,4,5,6,7,8"
SECOND_ROW="9,10,11,12,13,14,15,16,17"
FIRST_TWO_ROWS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
# Top-left 3x3 subgrid (box 0)
SUBGRID_0="0,1,2,9,10,11,18,19,20"
# First 3 rows (for modified dataset experiment)
FIRST_THREE_ROWS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"

echo "=============================================="
echo "Comprehensive Activation Patching Experiments"
echo "=============================================="
echo "Source puzzle: $SOURCE_IDX"
echo "Target puzzle: $TARGET_IDX"
echo "Output: $OUTPUT_BASE"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create base output directory
mkdir -p "$OUTPUT_BASE"

# ============================================================================
# PART 1: z_H ONLY PATCHING (Global Reasoning Stream)
# ============================================================================
echo ""
echo "########################################"
echo "# PART 1: z_H ONLY PATCHING"
echo "########################################"
echo ""

# 1.1: z_H at early steps (1,2,3)
echo "[1.1] z_H patching at EARLY steps (1,2,3)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level H \
    --patch_steps 1,2,3 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_H/early_steps_1_2_3"
echo ""

# 1.2: z_H at late steps (5,6,7)
echo "[1.2] z_H patching at LATE steps (5,6,7)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level H \
    --patch_steps 5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_H/late_steps_5_6_7"
echo ""

# ============================================================================
# PART 2: z_L ONLY PATCHING (Local/Per-Cell Stream)
# ============================================================================
echo ""
echo "########################################"
echo "# PART 2: z_L ONLY PATCHING"
echo "########################################"
echo ""

# ------------------------------------------------------------------------------
# 2.1: z_L at EARLY steps with position variations
# ------------------------------------------------------------------------------
echo ""
echo "=== z_L at EARLY steps (1,2,3) ==="
echo ""

# 2.1.1: z_L early steps - first two rows only
echo "[2.1.1] z_L EARLY steps (1,2,3) - FIRST TWO ROWS"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 1,2,3 \
    --patch_positions $FIRST_TWO_ROWS \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/early_steps/first_two_rows"
echo ""

# 2.1.2: z_L early steps - first subgrid only
echo "[2.1.2] z_L EARLY steps (1,2,3) - FIRST SUBGRID (top-left 3x3)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 1,2,3 \
    --patch_positions $SUBGRID_0 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/early_steps/first_subgrid"
echo ""

# 2.1.3: z_L early steps - ALL positions
echo "[2.1.3] z_L EARLY steps (1,2,3) - ALL POSITIONS"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 1,2,3 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/early_steps/all_positions"
echo ""

# ------------------------------------------------------------------------------
# 2.2: z_L at LATE steps with position variations
# ------------------------------------------------------------------------------
echo ""
echo "=== z_L at LATE steps (5,6,7) ==="
echo ""

# 2.2.1: z_L late steps - first two rows only
echo "[2.2.1] z_L LATE steps (5,6,7) - FIRST TWO ROWS"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 5,6,7 \
    --patch_positions $FIRST_TWO_ROWS \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/late_steps/first_two_rows"
echo ""

# 2.2.2: z_L late steps - first subgrid only
echo "[2.2.2] z_L LATE steps (5,6,7) - FIRST SUBGRID (top-left 3x3)"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 5,6,7 \
    --patch_positions $SUBGRID_0 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/late_steps/first_subgrid"
echo ""

# 2.2.3: z_L late steps - ALL positions
echo "[2.2.3] z_L LATE steps (5,6,7) - ALL POSITIONS"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/late_steps/all_positions"
echo ""

# ------------------------------------------------------------------------------
# 2.3: z_L at ALL steps (full temporal patching)
# ------------------------------------------------------------------------------
echo ""
echo "=== z_L at ALL steps (1-7) ==="
echo ""

echo "[2.3] z_L ALL steps (1,2,3,4,5,6,7) - ALL POSITIONS"
echo "----------------------------------------------"
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 1,2,3,4,5,6,7 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/all_steps/all_positions"
echo ""

# ------------------------------------------------------------------------------
# 2.4: z_L with MODIFIED INPUT (3 rows missing from label as puzzle input)
# This experiment uses a custom input where the solution has 3 rows blanked out
# to simulate a puzzle with more unknowns
# Uses --source_missing_rows and --target_missing_rows (1-indexed: 1-9)
# ------------------------------------------------------------------------------
echo ""
echo "=== z_L MODIFIED DATASET EXPERIMENT ==="
echo "(Using solution with first 3 rows masked as puzzle input)"
echo ""

echo "[2.4] z_L SINGLE step with MODIFIED INPUT (rows 1,2,3 missing from both puzzles)"
echo "----------------------------------------------"
echo "NOTE: Using --source_missing_rows and --target_missing_rows to create inputs"
echo "      from labels with first 3 rows blanked out."
python scripts/core/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx $SOURCE_IDX \
    --target_puzzle_idx $TARGET_IDX \
    --patch_level L \
    --patch_steps 4 \
    --source_missing_rows 1,2,3 \
    --target_missing_rows 1,2,3 \
    --num_runs $NUM_RUNS \
    --report_html report.html \
    --output_dir "$OUTPUT_BASE/z_L/modified_input/single_step_3rows_masked"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=============================================="
echo ""
echo "Results structure:"
echo "$OUTPUT_BASE/"
echo "├── z_H/"
echo "│   ├── early_steps_1_2_3/       # z_H at steps 1,2,3"
echo "│   └── late_steps_5_6_7/        # z_H at steps 5,6,7"
echo "└── z_L/"
echo "    ├── early_steps/"
echo "    │   ├── first_two_rows/      # z_L early, rows 0-1"
echo "    │   ├── first_subgrid/       # z_L early, top-left 3x3"
echo "    │   └── all_positions/       # z_L early, all cells"
echo "    ├── late_steps/"
echo "    │   ├── first_two_rows/      # z_L late, rows 0-1"
echo "    │   ├── first_subgrid/       # z_L late, top-left 3x3"
echo "    │   └── all_positions/       # z_L late, all cells"
echo "    ├── all_steps/"
echo "    │   └── all_positions/       # z_L all steps, all cells"
echo "    └── modified_input/"
echo "        └── single_step_3rows_masked/  # z_L single step, label with 3 rows masked"
echo ""
echo "Each folder contains:"
echo "  - *_forward.yaml        : Forward experiment (source→target)"
echo "  - *_inverse.yaml        : Inverse experiment (target→source)"
echo "  - report.html           : Forward HTML report"
echo "  - report_inverse.html   : Inverse HTML report"
echo ""
echo "To view reports:"
echo "  firefox $OUTPUT_BASE/z_H/early_steps_1_2_3/report.html"
