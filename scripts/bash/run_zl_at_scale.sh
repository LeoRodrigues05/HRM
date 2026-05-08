#!/bin/bash
# Gap 4: Scale z_L ablation from N=20 to N=1000
#
# Uses existing controlled_ablation.py with --num_puzzles 1000.
# No code changes needed — just running at scale.
#
# Usage:
#   bash scripts/bash/run_zl_at_scale.sh
#
# Expected runtime: ~10-15 min on GPU (1000 puzzles × 16+1 ablation conditions)

set -e

cd "$(dirname "$0")/../.."
echo "================================================"
echo "  Gap 4: z_L Ablation at Scale (N=1000)"
echo "  $(date)"
echo "================================================"

OUTPUT_DIR="results/controlled/ablation/zL_extended"
mkdir -p "$OUTPUT_DIR"

echo ""
echo ">>> Running z_L ablation (N=1000 puzzles)..."
python scripts/controlled/controlled_ablation.py \
    --num_puzzles 1000 \
    --z_level L \
    --max_steps 16 \
    --device cuda \
    --output_dir "$OUTPUT_DIR"

echo ""
echo ">>> Running z_H ablation at same scale for comparison (N=1000)..."
python scripts/controlled/controlled_ablation.py \
    --num_puzzles 1000 \
    --z_level H \
    --max_steps 16 \
    --device cuda \
    --output_dir "results/controlled/ablation/zH_1000"

echo ""
echo "================================================"
echo "  Done at $(date)"
echo "  Results in:"
echo "    z_L: $OUTPUT_DIR"
echo "    z_H: results/controlled/ablation/zH_1000"
echo "================================================"
