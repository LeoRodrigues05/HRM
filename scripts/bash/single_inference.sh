#!/bin/bash

# Single Inference Script for HRM
# Run inference on individual test cases and visualize results
# 
# Usage:
#   ./scripts/single_inference.sh                    # Run on first 10 puzzles
#   ./scripts/single_inference.sh --puzzle_idx 42    # Run on specific puzzle
#   ./scripts/single_inference.sh --num_puzzles 20   # Run on first 20 puzzles

set -euo pipefail

# Default configuration
CHECKPOINT="checkpoints/sapientinc-sudoku-extreme/checkpoint.pt"
DATA_DIR="data/sudoku-extreme-1k-aug-1000"
OUTPUT_DIR="results/single_inference"
NUM_PUZZLES=10
PUZZLE_IDX=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_puzzles)
            NUM_PUZZLES="$2"
            shift 2
            ;;
        --puzzle_idx)
            PUZZLE_IDX="$2"
            shift 2
            ;;
        -h|--help)
            echo "Single Inference Script for HRM"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH     Model checkpoint path (default: $CHECKPOINT)"
            echo "  --data_dir PATH       Dataset directory (default: $DATA_DIR)"
            echo "  --output_dir PATH     Output directory (default: $OUTPUT_DIR)"
            echo "  --num_puzzles N       Number of puzzles to run (default: $NUM_PUZZLES)"
            echo "  --puzzle_idx IDX      Specific puzzle index to run (overrides num_puzzles)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "HRM Single Inference"
echo "=============================================="
echo ""

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference using Python
python -c "
import torch
import numpy as np
import yaml
import sys
sys.path.insert(0, '.')

from dataset.build_sudoku_dataset import Sudoku_Dataset
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

# Load checkpoint
checkpoint_path = '$CHECKPOINT'
data_dir = '$DATA_DIR'
output_dir = '$OUTPUT_DIR'
puzzle_idx = '$PUZZLE_IDX'
num_puzzles = $NUM_PUZZLES

print(f'Loading checkpoint from {checkpoint_path}...')
checkpoint = torch.load(checkpoint_path, map_location='cuda')
cfg = checkpoint['cfg']

# Build model
model = HierarchicalReasoningModel_ACTV1(cfg).cuda()
model.load_state_dict(checkpoint['model_ema'])
model.eval()

# Load dataset
print(f'Loading dataset from {data_dir}...')
dataset = Sudoku_Dataset(data_dir, max_samples=None, split='test')

# Determine which puzzles to run
if puzzle_idx:
    indices = [int(puzzle_idx)]
else:
    indices = list(range(min(num_puzzles, len(dataset))))

print(f'Running inference on {len(indices)} puzzle(s)...')
print()

results = []

for idx in indices:
    sample = dataset[idx]
    x_puzzle = sample['puzzle'].cuda().unsqueeze(0)  # [1, 81]
    x_solution = sample['solution']  # [81]
    
    with torch.no_grad():
        output = model(x_puzzle)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        pred = logits.argmax(dim=-1).squeeze(0).cpu()  # [81]
    
    # Compute accuracy
    # Handle -1 (unknown) vs 0-8 (known) encoding
    mask = (x_puzzle.squeeze(0).cpu() == -1)  # Unknown cells
    correct = (pred == x_solution)
    
    # Overall accuracy
    overall_acc = correct.float().mean().item()
    
    # Unknown cell accuracy (the actual task)
    if mask.any():
        unknown_acc = correct[mask].float().mean().item()
    else:
        unknown_acc = 1.0
    
    # Known cell accuracy (should be very high)
    if (~mask).any():
        known_acc = correct[~mask].float().mean().item()
    else:
        known_acc = 1.0
    
    # Format puzzle for display
    puzzle_np = x_puzzle.squeeze(0).cpu().numpy()
    solution_np = x_solution.numpy()
    pred_np = pred.numpy()
    
    result = {
        'puzzle_idx': int(idx),
        'overall_accuracy': float(overall_acc),
        'unknown_cell_accuracy': float(unknown_acc),
        'known_cell_accuracy': float(known_acc),
        'num_unknown': int(mask.sum().item()),
        'num_correct_unknown': int(correct[mask].sum().item()) if mask.any() else 0,
    }
    results.append(result)
    
    print(f'Puzzle {idx}:')
    print(f'  Overall Accuracy: {overall_acc:.2%}')
    print(f'  Unknown Cell Accuracy: {unknown_acc:.2%} ({result[\"num_correct_unknown\"]}/{result[\"num_unknown\"]})')
    print(f'  Known Cell Accuracy: {known_acc:.2%}')
    
    # Print grids
    def print_sudoku(arr, title):
        print(f'  {title}:')
        for row in range(9):
            line = '    '
            for col in range(9):
                val = arr[row * 9 + col]
                if val == -1:
                    line += '. '
                else:
                    line += f'{val+1} '  # Convert 0-8 to 1-9
                if col in [2, 5]:
                    line += '| '
            print(line)
            if row in [2, 5]:
                print('    ------+-------+------')
    
    print_sudoku(puzzle_np, 'Input')
    print_sudoku(pred_np, 'Prediction')
    print_sudoku(solution_np, 'Ground Truth')
    print()

# Save results
results_file = f'{output_dir}/inference_results.yaml'
with open(results_file, 'w') as f:
    yaml.dump({'results': results}, f, default_flow_style=False)

print(f'Results saved to {results_file}')

# Summary statistics
if len(results) > 1:
    avg_overall = np.mean([r['overall_accuracy'] for r in results])
    avg_unknown = np.mean([r['unknown_cell_accuracy'] for r in results])
    print()
    print('Summary:')
    print(f'  Average Overall Accuracy: {avg_overall:.2%}')
    print(f'  Average Unknown Cell Accuracy: {avg_unknown:.2%}')
"

echo ""
echo "=============================================="
echo "Inference complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/inference_results.yaml"
