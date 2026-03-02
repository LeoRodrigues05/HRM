#!/usr/bin/env python3
"""Experiment 3: Step-wise Dynamics Analysis.

This experiment analyzes how the model refines its predictions across ACT steps:
- Hamming distance between consecutive steps
- KL divergence of logit distributions
- Per-step accuracy improvement
- Violation count reduction

Based on the paper's analysis of iterative refinement.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step: int
    accuracy: float
    unknown_accuracy: float
    hamming_from_prev: float  # Fraction of cells changed
    kl_from_prev: float       # KL divergence from previous step
    num_violations: float     # Average Sudoku constraint violations
    pct_filled: float         # Percentage of non-blank predictions


def count_sudoku_violations(grid: np.ndarray) -> int:
    """Count Sudoku constraint violations in a 9x9 grid.
    
    Args:
        grid: [81] or [9,9] array of digits 1-9 (0 for blank)
        
    Returns:
        Number of violated constraints
    """
    if grid.shape == (81,):
        grid = grid.reshape(9, 9)
    
    violations = 0
    
    # Row violations
    for r in range(9):
        row = grid[r, :]
        filled = row[row > 0]
        violations += len(filled) - len(set(filled))
    
    # Column violations
    for c in range(9):
        col = grid[:, c]
        filled = col[col > 0]
        violations += len(filled) - len(set(filled))
    
    # Box violations
    for box_r in range(3):
        for box_c in range(3):
            box = grid[box_r*3:(box_r+1)*3, box_c*3:(box_c+1)*3].flatten()
            filled = box[box > 0]
            violations += len(filled) - len(set(filled))
    
    return violations


def token_to_digit(tokens: np.ndarray) -> np.ndarray:
    """Convert token IDs to digits (1-9, 0 for blank)."""
    # Token 2 -> digit 1, token 3 -> digit 2, etc.
    digits = np.zeros_like(tokens)
    mask = (tokens >= 2) & (tokens <= 10)
    digits[mask] = tokens[mask] - 1
    return digits


def run_inference_with_steps(
    model, 
    inputs: np.ndarray, 
    labels: np.ndarray,
    cfg: dict, 
    device: str = "cuda",
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Run inference and collect predictions at each step.
    
    Returns:
        predictions: [N, 81] final predictions
        step_predictions: [max_steps, N, 81] predictions at each step
        step_logits: [max_steps, N, 81, V] logits at each step (if memory allows)
    """
    N = len(inputs)
    max_steps = cfg.get('halt_max_steps', 8)
    
    all_step_preds = [[] for _ in range(max_steps)]
    all_step_logits = [[] for _ in range(max_steps)]
    
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_inputs = torch.tensor(inputs[start:end], dtype=torch.long, device=device)
            batch_labels = torch.tensor(labels[start:end], dtype=torch.long, device=device)
            batch_size_actual = batch_inputs.shape[0]
            
            batch = {
                "inputs": batch_inputs,
                "labels": batch_labels,
                "puzzle_identifiers": torch.zeros(batch_size_actual, dtype=torch.long, device=device),
            }
            
            carry = model.initial_carry(batch)
            # Move carry to device
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
            carry.steps = carry.steps.to(device)
            carry.halted = carry.halted.to(device)
            
            for step in range(max_steps):
                carry, outputs = model(carry, batch)
                logits = outputs["logits"]  # [B, 81, V]
                preds = logits.argmax(dim=-1)
                
                all_step_preds[step].append(preds.cpu().numpy())
                # Only save logits if we have enough memory
                if N <= 1000:
                    all_step_logits[step].append(logits.cpu().numpy())
            
            if (start // batch_size) % 10 == 0:
                print(f"  Processed {end}/{N} puzzles...")
    
    # Concatenate
    step_preds = np.stack([np.concatenate(sp, axis=0) for sp in all_step_preds], axis=0)
    
    step_logits = None
    if all_step_logits[0]:
        step_logits = [np.concatenate(sl, axis=0) for sl in all_step_logits]
    
    return step_preds[-1], step_preds, step_logits


def compute_step_metrics(
    step_preds: np.ndarray,  # [max_steps, N, 81]
    step_logits: List[np.ndarray],  # [max_steps] list of [N, 81, V]
    labels: np.ndarray,      # [N, 81]
    inputs: np.ndarray,      # [N, 81]
) -> List[StepMetrics]:
    """Compute metrics for each step."""
    
    max_steps, N, _ = step_preds.shape
    
    # Identify unknown cells
    is_given = (inputs >= 2) & (inputs <= 10)
    is_unknown = ~is_given
    
    metrics = []
    
    for step in range(max_steps):
        preds = step_preds[step]  # [N, 81]
        
        # Accuracy
        correct = (preds == labels).sum()
        accuracy = correct / (N * 81)
        
        unknown_correct = ((preds == labels) & is_unknown).sum()
        unknown_total = is_unknown.sum()
        unknown_accuracy = unknown_correct / max(unknown_total, 1)
        
        # Hamming from previous step
        if step == 0:
            # Compare to input
            hamming = (preds != inputs).sum() / (N * 81)
        else:
            prev_preds = step_preds[step - 1]
            hamming = (preds != prev_preds).sum() / (N * 81)
        
        # KL divergence from previous step
        kl = 0.0
        if step_logits is not None and step > 0:
            curr_logits = step_logits[step]     # [N, 81, V]
            prev_logits = step_logits[step - 1]  # [N, 81, V]
            
            curr_probs = softmax(curr_logits, axis=-1)
            prev_probs = softmax(prev_logits, axis=-1)
            
            # KL(prev || curr)
            kl_per_cell = np.sum(prev_probs * (np.log(prev_probs + 1e-10) - np.log(curr_probs + 1e-10)), axis=-1)
            kl = np.mean(kl_per_cell)
        
        # Violations
        digits = token_to_digit(preds)
        violations = [count_sudoku_violations(digits[i]) for i in range(N)]
        avg_violations = np.mean(violations)
        
        # Percentage filled (non-blank predictions)
        non_blank = (preds >= 2) & (preds <= 10)
        pct_filled = non_blank.sum() / (N * 81)
        
        metrics.append(StepMetrics(
            step=step,
            accuracy=float(accuracy),
            unknown_accuracy=float(unknown_accuracy),
            hamming_from_prev=float(hamming),
            kl_from_prev=float(kl),
            num_violations=float(avg_violations),
            pct_filled=float(pct_filled),
        ))
    
    return metrics


def plot_step_dynamics(metrics: List[StepMetrics], output_dir: Path):
    """Create comprehensive step dynamics plots."""
    
    steps = [m.step for m in metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Accuracy over steps
    ax1 = axes[0, 0]
    ax1.plot(steps, [m.accuracy for m in metrics], 'b-o', label='Overall', linewidth=2)
    ax1.plot(steps, [m.unknown_accuracy for m in metrics], 'r-o', label='Unknown cells', linewidth=2)
    ax1.set_xlabel("ACT Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Progression Across Steps")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Hamming distance
    ax2 = axes[0, 1]
    ax2.plot(steps, [m.hamming_from_prev for m in metrics], 'g-o', linewidth=2)
    ax2.set_xlabel("ACT Step")
    ax2.set_ylabel("Hamming Distance (fraction)")
    ax2.set_title("Cells Changed from Previous Step")
    ax2.grid(alpha=0.3)
    
    # Plot 3: KL divergence
    ax3 = axes[0, 2]
    ax3.plot(steps[1:], [m.kl_from_prev for m in metrics[1:]], 'm-o', linewidth=2)
    ax3.set_xlabel("ACT Step")
    ax3.set_ylabel("KL Divergence")
    ax3.set_title("Logit Distribution Change (KL)")
    ax3.grid(alpha=0.3)
    
    # Plot 4: Violations
    ax4 = axes[1, 0]
    ax4.plot(steps, [m.num_violations for m in metrics], 'r-o', linewidth=2)
    ax4.set_xlabel("ACT Step")
    ax4.set_ylabel("Avg Violations")
    ax4.set_title("Sudoku Constraint Violations")
    ax4.grid(alpha=0.3)
    
    # Plot 5: Percentage filled
    ax5 = axes[1, 1]
    ax5.plot(steps, [m.pct_filled for m in metrics], 'c-o', linewidth=2)
    ax5.set_xlabel("ACT Step")
    ax5.set_ylabel("Percentage Filled")
    ax5.set_title("Non-blank Predictions")
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1.05)
    
    # Plot 6: Combined normalized metrics
    ax6 = axes[1, 2]
    
    # Normalize metrics to [0, 1] for comparison
    acc_norm = [m.accuracy for m in metrics]
    ham_norm = [m.hamming_from_prev for m in metrics]
    max_viol = max(m.num_violations for m in metrics)
    viol_norm = [m.num_violations / max(max_viol, 1) for m in metrics]
    
    ax6.plot(steps, acc_norm, 'b-o', label='Accuracy', linewidth=2)
    ax6.plot(steps, ham_norm, 'g-o', label='Hamming', linewidth=2)
    ax6.plot(steps, viol_norm, 'r-o', label='Violations (norm)', linewidth=2)
    ax6.set_xlabel("ACT Step")
    ax6.set_ylabel("Normalized Value")
    ax6.set_title("Combined Step Dynamics")
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "step_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved step dynamics plot to {output_dir / 'step_dynamics.png'}")


def plot_refinement_examples(
    step_preds: np.ndarray,
    labels: np.ndarray,
    inputs: np.ndarray,
    output_dir: Path,
    num_examples: int = 4
):
    """Visualize how predictions refine across steps for example puzzles."""
    
    max_steps, N, _ = step_preds.shape
    
    # Select puzzles with interesting refinement
    np.random.seed(42)
    example_indices = np.random.choice(N, min(num_examples, N), replace=False)
    
    fig, axes = plt.subplots(num_examples, max_steps + 2, figsize=(max_steps * 2 + 4, num_examples * 2))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(example_indices):
        # Input
        ax_in = axes[row, 0]
        inp_grid = token_to_digit(inputs[idx]).reshape(9, 9)
        ax_in.imshow(inp_grid, cmap='Blues', vmin=0, vmax=9)
        ax_in.set_title("Input" if row == 0 else "")
        ax_in.axis('off')
        
        # Each step
        for step in range(max_steps):
            ax = axes[row, step + 1]
            pred_grid = token_to_digit(step_preds[step, idx]).reshape(9, 9)
            label_grid = token_to_digit(labels[idx]).reshape(9, 9)
            
            # Color by correctness
            correct_mask = (pred_grid == label_grid).astype(float)
            ax.imshow(correct_mask, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f"Step {step}" if row == 0 else "")
            ax.axis('off')
        
        # Ground truth
        ax_gt = axes[row, -1]
        gt_grid = token_to_digit(labels[idx]).reshape(9, 9)
        ax_gt.imshow(gt_grid, cmap='Blues', vmin=0, vmax=9)
        ax_gt.set_title("Ground Truth" if row == 0 else "")
        ax_gt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "refinement_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved refinement examples to {output_dir / 'refinement_examples.png'}")


def main():
    parser = argparse.ArgumentParser(description="Step-wise Dynamics Analysis")
    parser.add_argument("--checkpoint", type=str,
                       default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/paper_replication/results/step_dynamics")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    import yaml
    import json
    print(f"Loading checkpoint from {args.checkpoint}...")
    
    # Load config from YAML file
    config_path = Path(args.checkpoint).parent / "all_config.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    # Load dataset metadata
    dataset_json_path = Path(args.data_path) / "test" / "dataset.json"
    with open(dataset_json_path, "r") as f:
        dataset_metadata = json.load(f)
    
    # Extract arch config and add required fields from dataset metadata
    cfg = full_config['arch'].copy()
    cfg['batch_size'] = full_config.get('global_batch_size', 768)
    cfg['seq_len'] = dataset_metadata['seq_len']
    cfg['vocab_size'] = dataset_metadata['vocab_size']
    cfg['num_puzzle_identifiers'] = dataset_metadata['num_puzzle_identifiers']
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(cfg).to(args.device)
    
    # Load checkpoint weights
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Handle compiled model prefix (_orig_mod.) and loss head wrapper (model.)
    def fix_key(k):
        k = k.removeprefix("_orig_mod.")
        k = k.removeprefix("model.")
        return k
    
    fixed_state_dict = {fix_key(k): v for k, v in checkpoint.items()}
    model.load_state_dict(fixed_state_dict, assign=True)
    
    model.eval()
    
    # Load data
    test_dir = Path(args.data_path) / "test"
    inputs = np.load(test_dir / "all__inputs.npy")
    labels = np.load(test_dir / "all__labels.npy")
    
    print(f"\nDataset size: {len(inputs)} puzzles")
    print(f"Max ACT steps: {cfg.get('halt_max_steps', 8)}")
    
    # Run inference
    print("\nRunning inference with step tracking...")
    final_preds, step_preds, step_logits = run_inference_with_steps(
        model, inputs, labels, cfg, args.device, args.batch_size
    )
    
    # Compute metrics
    print("\nComputing step-wise metrics...")
    metrics = compute_step_metrics(step_preds, step_logits, labels, inputs)
    
    # Print summary
    print("\nStep-wise Metrics:")
    print("-" * 80)
    print(f"{'Step':>4} | {'Accuracy':>8} | {'Unknown':>8} | {'Hamming':>8} | {'KL':>8} | {'Viols':>8}")
    print("-" * 80)
    for m in metrics:
        print(f"{m.step:>4} | {m.accuracy:>8.4f} | {m.unknown_accuracy:>8.4f} | "
              f"{m.hamming_from_prev:>8.4f} | {m.kl_from_prev:>8.4f} | {m.num_violations:>8.2f}")
    
    # Create plots
    plot_step_dynamics(metrics, output_dir)
    plot_refinement_examples(step_preds, labels, inputs, output_dir)
    
    # Save results
    results = {
        "step_metrics": [asdict(m) for m in metrics],
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "max_steps": cfg.get('halt_max_steps', 8),
            "num_puzzles": len(inputs),
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Analysis summary
    print("\n" + "="*60)
    print("STEP DYNAMICS ANALYSIS SUMMARY")
    print("="*60)
    
    # Check for iterative refinement
    acc_improvement = metrics[-1].accuracy - metrics[0].accuracy
    viol_reduction = metrics[0].num_violations - metrics[-1].num_violations
    
    print(f"\nAccuracy improvement (step 0 → final): {acc_improvement:.4f}")
    print(f"Violation reduction (step 0 → final): {viol_reduction:.2f}")
    
    # Find step with most improvement
    acc_changes = [metrics[i+1].accuracy - metrics[i].accuracy for i in range(len(metrics)-1)]
    max_change_step = np.argmax(acc_changes)
    
    print(f"\nMost improvement at step: {max_change_step} → {max_change_step+1} "
          f"(+{acc_changes[max_change_step]:.4f})")
    
    if acc_improvement > 0.1:
        print("\n✓ FINDING: Model shows significant iterative refinement across steps")
    else:
        print("\n⚠️  Limited iterative refinement observed")


if __name__ == "__main__":
    main()
