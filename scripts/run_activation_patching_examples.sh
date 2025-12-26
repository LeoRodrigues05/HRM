#!/bin/bash

# Example Activation Patching Experiments for HRM
# This script demonstrates common activation patching scenarios

# Set checkpoint path
CHECKPOINT="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt"
OUTPUT_DIR="results/activation_patching_examples"

echo "=============================================="
echo "Activation Patching Examples for HRM"
echo "=============================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

# Example 1: Basic patching - patch all levels at all steps
echo "Example 1: Basic full patching (all levels, all steps)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 1 \
    --patch_level both \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example1_full_patch"

echo ""
echo ""

# Example 2: H-level only patching
echo "Example 2: High-level reasoning patching (H-level only)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 2 \
    --patch_level H \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example2_h_level"

echo ""
echo ""

# Example 3: L-level only patching
echo "Example 3: Low-level reasoning patching (L-level only)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 2 \
    --patch_level L \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example3_l_level"

echo ""
echo ""

# Example 4: Early step patching
echo "Example 4: Early reasoning steps (steps 0-2)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 1 \
    --target_puzzle_idx 3 \
    --patch_level both \
    --patch_steps 0,1,2 \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example4_early_steps"

echo ""
echo ""

# Example 5: Late step patching
echo "Example 5: Late reasoning steps (steps 5-7)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 1 \
    --target_puzzle_idx 3 \
    --patch_level both \
    --patch_steps 5,6,7 \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example5_late_steps"

echo ""
echo ""

# Example 6: Single step patching
echo "Example 6: Single step patching (step 3 only)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 2 \
    --target_puzzle_idx 4 \
    --patch_level both \
    --patch_steps 3 \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example6_single_step"

echo ""
echo ""

# Example 7: Position-specific patching (first row in Sudoku - positions 0-8)
echo "Example 7: Position-specific patching (first 9 positions)"
echo "----------------------------------------------"
python scripts/activation_patching.py \
    --checkpoint "$CHECKPOINT" \
    --source_puzzle_idx 3 \
    --target_puzzle_idx 5 \
    --patch_level both \
    --patch_positions 0,1,2,3,4,5,6,7,8 \
    --max_steps 8 \
    --output_dir "$OUTPUT_DIR/example7_position_specific"

echo ""
echo ""

echo "=============================================="
echo "All examples completed!"
echo "=============================================="
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To view a specific result, run:"
echo "  cat $OUTPUT_DIR/example1_full_patch/patch_s0_t1_both.yaml"
echo ""
echo "To run batch experiments on multiple puzzles:"
echo "  python scripts/batch_activation_patching.py \\"
echo "      --checkpoint $CHECKPOINT \\"
echo "      --num_puzzles 10 \\"
echo "      --output_dir results/batch_experiments"
