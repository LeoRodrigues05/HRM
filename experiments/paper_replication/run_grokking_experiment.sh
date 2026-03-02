#!/bin/bash
# Grokking Experiment Runner
# This script trains an HRM model and analyzes grokking dynamics
#
# Usage:
#   ./run_grokking_experiment.sh                    # Default settings
#   ./run_grokking_experiment.sh --fast             # Quick test run
#   ./run_grokking_experiment.sh --full             # Full paper replication
#
# Run in background with:
#   nohup ./run_grokking_experiment.sh > grokking.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# Default parameters (using STEPS, not epochs!)
TOTAL_STEPS=50000
NUM_CHECKPOINTS=10
BATCH_SIZE=256
HIDDEN_SIZE=512
NUM_LAYERS=8
LEARNING_RATE=1e-4
TRAIN_SUBSET=100000
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
OUTPUT_DIR="experiments/paper_replication/results/grokking_checkpoints"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            # Quick test: smaller model, fewer steps (~15 min)
            TOTAL_STEPS=10000
            NUM_CHECKPOINTS=10
            BATCH_SIZE=256
            HIDDEN_SIZE=256
            NUM_LAYERS=4
            TRAIN_SUBSET=50000
            OUTPUT_DIR="experiments/paper_replication/results/grokking_checkpoints_fast"
            shift
            ;;
        --full)
            # Full run: more steps, full model (~2-4 hours)
            TOTAL_STEPS=100000
            NUM_CHECKPOINTS=20
            BATCH_SIZE=256
            HIDDEN_SIZE=512
            NUM_LAYERS=8
            TRAIN_SUBSET=200000
            OUTPUT_DIR="experiments/paper_replication/results/grokking_checkpoints_full"
            shift
            ;;
        --steps)
            TOTAL_STEPS="$2"
            shift 2
            ;;
        --checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "GROKKING DYNAMICS EXPERIMENT"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Total steps: $TOTAL_STEPS"
echo "  Num checkpoints: $NUM_CHECKPOINTS"
echo "  Batch size: $BATCH_SIZE"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Num layers: $NUM_LAYERS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Train subset: $TRAIN_SUBSET"
echo "  Data path: $DATA_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "Start time: $(date)"
echo ""

# Check if data exists
if [ ! -d "$DATA_PATH/train" ] || [ ! -d "$DATA_PATH/test" ]; then
    echo "ERROR: Data not found at $DATA_PATH"
    echo "Please ensure the Sudoku dataset is available."
    exit 1
fi

# Step 1: Training
echo "============================================================"
echo "STEP 1: TRAINING MODEL WITH CHECKPOINT SAVING"
echo "============================================================"
echo ""

python experiments/paper_replication/train_for_grokking.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --total_steps "$TOTAL_STEPS" \
    --num_checkpoints "$NUM_CHECKPOINTS" \
    --batch_size "$BATCH_SIZE" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --learning_rate "$LEARNING_RATE" \
    --train_subset "$TRAIN_SUBSET"

echo ""
echo "Training complete!"
echo ""

# Step 2: Analysis
echo "============================================================"
echo "STEP 2: ANALYZING GROKKING DYNAMICS"
echo "============================================================"
echo ""

python experiments/paper_replication/analyze_grokking_checkpoints.py \
    --checkpoint_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "GROKKING EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/snapshots.json          (training metrics)"
echo "  - $OUTPUT_DIR/grokking_results.json   (analysis results)"
echo "  - $OUTPUT_DIR/grokking_analysis.png   (visualization)"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/grokking_results.json | python -m json.tool | head -50"
