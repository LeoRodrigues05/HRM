#!/usr/bin/env python3
"""Experiment 1: Easy vs Hard Puzzle Performance Analysis.

This experiment tests the paper's claim that HRM struggles with easy puzzles.
We analyze model performance stratified by puzzle difficulty.

Difficulty metrics:
- Number of given cells (more givens = easier)
- Solution path complexity
- Number of naked singles vs hidden singles required
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@dataclass
class PuzzleDifficulty:
    """Difficulty metrics for a single puzzle."""
    puzzle_idx: int
    num_givens: int
    num_blanks: int
    constraint_difficulty: float  # Average candidates per blank cell
    
    
@dataclass
class DifficultyBinResult:
    """Results for a difficulty bin."""
    bin_id: int
    difficulty_range: Tuple[float, float]
    num_puzzles: int
    mean_accuracy: float
    std_accuracy: float
    mean_unknown_accuracy: float
    std_unknown_accuracy: float
    per_puzzle_accuracies: List[float]


def compute_puzzle_difficulty(puzzle: np.ndarray) -> PuzzleDifficulty:
    """Compute difficulty metrics for a puzzle.
    
    Args:
        puzzle: [81] array of token IDs (2-10 = digits 1-9, other = blank)
        
    Returns:
        PuzzleDifficulty with computed metrics
    """
    # Convert to digit representation (1-9 for digits, 0 for blank)
    digits = np.zeros(81, dtype=int)
    for i, tok in enumerate(puzzle):
        if 2 <= tok <= 10:
            digits[i] = tok - 1  # Token 2 -> digit 1, etc.
    
    grid = digits.reshape(9, 9)
    
    # Count givens and blanks
    num_givens = np.sum(grid > 0)
    num_blanks = 81 - num_givens
    
    # Compute constraint difficulty (average candidates per blank)
    total_candidates = 0
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                # Get used digits in row, col, box
                row_digits = set(grid[r, :]) - {0}
                col_digits = set(grid[:, c]) - {0}
                box_r, box_c = 3 * (r // 3), 3 * (c // 3)
                box_digits = set(grid[box_r:box_r+3, box_c:box_c+3].flatten()) - {0}
                
                used = row_digits | col_digits | box_digits
                candidates = 9 - len(used)
                total_candidates += candidates
    
    constraint_difficulty = total_candidates / max(num_blanks, 1)
    
    return PuzzleDifficulty(
        puzzle_idx=-1,  # Set later
        num_givens=int(num_givens),
        num_blanks=int(num_blanks),
        constraint_difficulty=float(constraint_difficulty)
    )


def load_model_and_data(checkpoint_path: str, data_path: str, device: str = "cuda"):
    """Load model checkpoint and dataset."""
    import yaml
    import json
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load config from YAML file
    config_path = Path(checkpoint_path).parent / "all_config.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    # Load dataset metadata
    dataset_json_path = Path(data_path) / "test" / "dataset.json"
    with open(dataset_json_path, "r") as f:
        dataset_metadata = json.load(f)
    
    # Extract arch config and add required fields from dataset metadata
    cfg = full_config['arch'].copy()
    cfg['batch_size'] = full_config.get('global_batch_size', 768)
    cfg['seq_len'] = dataset_metadata['seq_len']
    cfg['vocab_size'] = dataset_metadata['vocab_size']
    cfg['num_puzzle_identifiers'] = dataset_metadata['num_puzzle_identifiers']
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(cfg).to(device)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle compiled model prefix (_orig_mod.) and loss head wrapper (model.)
    # The checkpoint keys look like: _orig_mod.model.inner.X
    # We need: inner.X
    def fix_key(k):
        k = k.removeprefix("_orig_mod.")  # Remove torch.compile prefix
        k = k.removeprefix("model.")       # Remove ACTLossHead wrapper prefix
        return k
    
    fixed_state_dict = {fix_key(k): v for k, v in checkpoint.items()}
    model.load_state_dict(fixed_state_dict, assign=True)
    
    model.eval()
    
    print(f"Loading dataset from {data_path}...")
    # Load test inputs and labels directly
    test_dir = Path(data_path) / "test"
    inputs = np.load(test_dir / "all__inputs.npy")
    labels = np.load(test_dir / "all__labels.npy")
    
    return model, cfg, inputs, labels


def run_inference(model, inputs: np.ndarray, cfg: dict, device: str = "cuda", batch_size: int = 32):
    """Run model inference on all puzzles.
    
    Returns:
        predictions: [N, 81] predicted token IDs
        step_predictions: [max_steps, N, 81] predictions at each step
    """
    N = len(inputs)
    max_steps = cfg.get('halt_max_steps', 16)  # Default from config
    
    all_preds = []
    all_step_preds = []
    
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_inputs = torch.tensor(inputs[start:end], dtype=torch.long, device=device)
            batch_size_actual = batch_inputs.shape[0]
            
            # Create batch dict
            batch = {
                "inputs": batch_inputs,
                "labels": batch_inputs,  # Not used during inference
                "puzzle_identifiers": torch.zeros(batch_size_actual, dtype=torch.long, device=device),
            }
            
            # Initialize carry and move all tensors to device
            carry = model.initial_carry(batch)
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
            carry.steps = carry.steps.to(device)
            carry.halted = carry.halted.to(device)
            
            step_preds = []
            for step in range(max_steps):
                carry, outputs = model(carry, batch)
                logits = outputs["logits"]
                preds = logits.argmax(dim=-1)
                step_preds.append(preds.cpu().numpy())
            
            # Final prediction is from last step
            final_preds = step_preds[-1]
            all_preds.append(final_preds)
            
            # Stack step predictions: [steps, batch, 81]
            step_preds_arr = np.stack(step_preds, axis=0)
            all_step_preds.append(step_preds_arr)
            
            if (start // batch_size) % 10 == 0:
                print(f"  Processed {end}/{N} puzzles...")
    
    predictions = np.concatenate(all_preds, axis=0)
    # Concatenate step predictions along batch dimension
    step_predictions = np.concatenate(all_step_preds, axis=1)
    
    return predictions, step_predictions


def compute_accuracy_metrics(predictions: np.ndarray, labels: np.ndarray, inputs: np.ndarray) -> Dict:
    """Compute accuracy metrics.
    
    Returns dict with:
        - overall_accuracy: fraction of all cells correct
        - unknown_accuracy: fraction of blank cells correct
        - known_accuracy: fraction of given cells correct
    """
    N = len(predictions)
    
    overall_correct = (predictions == labels).sum()
    overall_total = N * 81
    
    # Find unknown cells (not givens)
    # Givens are tokens 2-10 (digits 1-9)
    is_given = (inputs >= 2) & (inputs <= 10)
    is_unknown = ~is_given
    
    unknown_correct = ((predictions == labels) & is_unknown).sum()
    unknown_total = is_unknown.sum()
    
    known_correct = ((predictions == labels) & is_given).sum()
    known_total = is_given.sum()
    
    return {
        "overall_accuracy": float(overall_correct) / overall_total,
        "unknown_accuracy": float(unknown_correct) / max(unknown_total, 1),
        "known_accuracy": float(known_correct) / max(known_total, 1),
        "overall_correct": int(overall_correct),
        "overall_total": int(overall_total),
        "unknown_correct": int(unknown_correct),
        "unknown_total": int(unknown_total),
    }


def analyze_by_difficulty(
    predictions: np.ndarray,
    labels: np.ndarray, 
    inputs: np.ndarray,
    num_bins: int = 5,
    difficulty_metric: str = "num_givens"
) -> List[DifficultyBinResult]:
    """Analyze accuracy stratified by puzzle difficulty.
    
    Args:
        predictions: [N, 81] model predictions
        labels: [N, 81] ground truth
        inputs: [N, 81] input puzzles
        num_bins: Number of difficulty bins
        difficulty_metric: "num_givens" or "constraint_difficulty"
        
    Returns:
        List of DifficultyBinResult for each bin
    """
    N = len(predictions)
    
    # Compute difficulty for each puzzle
    difficulties = []
    for i in range(N):
        diff = compute_puzzle_difficulty(inputs[i])
        diff.puzzle_idx = i
        difficulties.append(diff)
    
    # Get the metric values
    if difficulty_metric == "num_givens":
        metric_values = np.array([d.num_givens for d in difficulties])
    else:
        metric_values = np.array([d.constraint_difficulty for d in difficulties])
    
    # Create bins
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(metric_values, percentiles)
    
    # Assign puzzles to bins
    bin_assignments = np.digitize(metric_values, bin_edges[1:-1])
    
    results = []
    for bin_id in range(num_bins):
        bin_mask = bin_assignments == bin_id
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) == 0:
            continue
        
        # Compute per-puzzle accuracy for this bin
        per_puzzle_acc = []
        per_puzzle_unknown_acc = []
        
        for idx in bin_indices:
            pred = predictions[idx]
            label = labels[idx]
            inp = inputs[idx]
            
            # Overall accuracy for this puzzle
            acc = (pred == label).mean()
            per_puzzle_acc.append(acc)
            
            # Unknown cell accuracy
            is_unknown = ~((inp >= 2) & (inp <= 10))
            if is_unknown.sum() > 0:
                unknown_acc = ((pred == label) & is_unknown).sum() / is_unknown.sum()
            else:
                unknown_acc = 1.0
            per_puzzle_unknown_acc.append(unknown_acc)
        
        bin_result = DifficultyBinResult(
            bin_id=bin_id,
            difficulty_range=(float(bin_edges[bin_id]), float(bin_edges[bin_id + 1])),
            num_puzzles=len(bin_indices),
            mean_accuracy=float(np.mean(per_puzzle_acc)),
            std_accuracy=float(np.std(per_puzzle_acc)),
            mean_unknown_accuracy=float(np.mean(per_puzzle_unknown_acc)),
            std_unknown_accuracy=float(np.std(per_puzzle_unknown_acc)),
            per_puzzle_accuracies=per_puzzle_acc,
        )
        results.append(bin_result)
    
    return results


def plot_easy_hard_analysis(results: List[DifficultyBinResult], output_path: Path, metric_name: str):
    """Create visualization of easy vs hard performance."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy by difficulty bin
    ax1 = axes[0]
    bin_ids = [r.bin_id for r in results]
    means = [r.mean_accuracy for r in results]
    stds = [r.std_accuracy for r in results]
    
    x = np.arange(len(results))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    
    # Add difficulty range labels
    labels = [f"{r.difficulty_range[0]:.1f}-{r.difficulty_range[1]:.1f}" for r in results]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel(f"Difficulty Bin ({metric_name})")
    ax1.set_ylabel("Overall Accuracy")
    ax1.set_title("Model Accuracy by Puzzle Difficulty")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, r in zip(bars, results):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={r.num_puzzles}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Unknown cell accuracy by difficulty
    ax2 = axes[1]
    unknown_means = [r.mean_unknown_accuracy for r in results]
    unknown_stds = [r.std_unknown_accuracy for r in results]
    
    bars2 = ax2.bar(x, unknown_means, yerr=unknown_stds, capsize=5, color='coral', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_xlabel(f"Difficulty Bin ({metric_name})")
    ax2.set_ylabel("Unknown Cell Accuracy")
    ax2.set_title("Unknown Cell Accuracy by Puzzle Difficulty")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_scatter_difficulty_vs_accuracy(
    difficulties: List[PuzzleDifficulty],
    predictions: np.ndarray,
    labels: np.ndarray,
    inputs: np.ndarray,
    output_path: Path
):
    """Scatter plot of difficulty vs accuracy."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute per-puzzle accuracy
    accuracies = []
    unknown_accuracies = []
    num_givens = []
    
    for i, diff in enumerate(difficulties):
        pred = predictions[i]
        label = labels[i]
        inp = inputs[i]
        
        acc = (pred == label).mean()
        accuracies.append(acc)
        
        is_unknown = ~((inp >= 2) & (inp <= 10))
        if is_unknown.sum() > 0:
            unknown_acc = ((pred == label) & is_unknown).sum() / is_unknown.sum()
        else:
            unknown_acc = 1.0
        unknown_accuracies.append(unknown_acc)
        num_givens.append(diff.num_givens)
    
    # Plot 1: Num givens vs overall accuracy
    ax1 = axes[0]
    ax1.scatter(num_givens, accuracies, alpha=0.5, s=20)
    ax1.set_xlabel("Number of Given Cells")
    ax1.set_ylabel("Overall Accuracy")
    ax1.set_title("Puzzle Difficulty (Givens) vs Model Accuracy")
    ax1.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(num_givens, accuracies, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(num_givens), max(num_givens), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    ax1.legend()
    
    # Plot 2: Num givens vs unknown accuracy
    ax2 = axes[1]
    ax2.scatter(num_givens, unknown_accuracies, alpha=0.5, s=20, color='coral')
    ax2.set_xlabel("Number of Given Cells")
    ax2.set_ylabel("Unknown Cell Accuracy")
    ax2.set_title("Puzzle Difficulty (Givens) vs Unknown Cell Accuracy")
    ax2.grid(alpha=0.3)
    
    # Add trend line
    z2 = np.polyfit(num_givens, unknown_accuracies, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), 'r--', linewidth=2, label=f'Trend: y={z2[0]:.4f}x+{z2[1]:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Easy vs Hard Puzzle Analysis")
    parser.add_argument("--checkpoint", type=str, 
                       default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--output_dir", type=str, default="experiments/paper_replication/results/easy_hard_analysis")
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, cfg, inputs, labels = load_model_and_data(
        args.checkpoint, args.data_path, args.device
    )
    
    print(f"\nDataset size: {len(inputs)} puzzles")
    
    # Run inference
    print("\nRunning inference...")
    predictions, step_predictions = run_inference(
        model, inputs, cfg, args.device, args.batch_size
    )
    
    # Compute overall metrics
    print("\nComputing metrics...")
    overall_metrics = compute_accuracy_metrics(predictions, labels, inputs)
    print(f"Overall accuracy: {overall_metrics['overall_accuracy']:.4f}")
    print(f"Unknown cell accuracy: {overall_metrics['unknown_accuracy']:.4f}")
    print(f"Known cell accuracy: {overall_metrics['known_accuracy']:.4f}")
    
    # Analyze by difficulty
    print("\nAnalyzing by puzzle difficulty...")
    
    # By number of givens (more givens = easier)
    results_givens = analyze_by_difficulty(
        predictions, labels, inputs, 
        num_bins=args.num_bins, 
        difficulty_metric="num_givens"
    )
    
    print("\nResults by number of givens:")
    for r in results_givens:
        print(f"  Bin {r.bin_id} ({r.difficulty_range[0]:.0f}-{r.difficulty_range[1]:.0f} givens): "
              f"acc={r.mean_accuracy:.4f}±{r.std_accuracy:.4f}, "
              f"unknown_acc={r.mean_unknown_accuracy:.4f}±{r.std_unknown_accuracy:.4f}, "
              f"n={r.num_puzzles}")
    
    # Create plots
    plot_easy_hard_analysis(
        results_givens, 
        output_dir / "accuracy_by_givens.png",
        "Number of Givens"
    )
    
    # Compute difficulties for scatter plot
    difficulties = [compute_puzzle_difficulty(inputs[i]) for i in range(len(inputs))]
    for i, d in enumerate(difficulties):
        d.puzzle_idx = i
    
    plot_scatter_difficulty_vs_accuracy(
        difficulties, predictions, labels, inputs,
        output_dir / "scatter_difficulty_accuracy.png"
    )
    
    # Save results
    results_dict = {
        "overall_metrics": overall_metrics,
        "difficulty_bins": [asdict(r) for r in results_givens],
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "num_bins": args.num_bins,
            "num_puzzles": len(inputs),
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Summary analysis
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY: Does HRM struggle with easy puzzles?")
    print("="*60)
    
    # Check if there's a negative correlation between givens and accuracy
    accuracies = [r.mean_accuracy for r in results_givens]
    if len(accuracies) >= 2:
        # Compare easiest bin (most givens) vs hardest bin (fewest givens)
        easiest_acc = results_givens[-1].mean_accuracy  # Most givens
        hardest_acc = results_givens[0].mean_accuracy   # Fewest givens
        
        print(f"\nEasiest puzzles (most givens) accuracy: {easiest_acc:.4f}")
        print(f"Hardest puzzles (fewest givens) accuracy: {hardest_acc:.4f}")
        print(f"Difference: {easiest_acc - hardest_acc:.4f}")
        
        if easiest_acc < hardest_acc:
            print("\n⚠️  FINDING: HRM shows LOWER accuracy on easier puzzles!")
            print("   This supports the paper's claim that HRM struggles with easy puzzles.")
        else:
            print("\n✓ HRM shows expected behavior (higher accuracy on easier puzzles)")


if __name__ == "__main__":
    main()
