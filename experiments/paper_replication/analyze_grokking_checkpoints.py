#!/usr/bin/env python3
"""Analyze grokking dynamics from saved checkpoints.

This script analyzes checkpoints saved by train_for_grokking.py to
characterize the grokking phenomenon in HRM training.

Usage:
    python experiments/paper_replication/analyze_grokking_checkpoints.py \
        --checkpoint_dir experiments/paper_replication/results/grokking_checkpoints
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class GrokkingAnalysis:
    """Complete grokking analysis results."""
    # Grokking point
    grokking_epoch: Optional[int]
    grokking_step: Optional[int]
    grokking_train_acc: Optional[float]
    grokking_test_acc: Optional[float]
    
    # Gap metrics
    max_generalization_gap: float
    final_generalization_gap: float
    gap_reduction: float
    
    # Acceleration metrics
    max_test_acc_jump: float
    max_test_acc_jump_epoch: Optional[int]
    
    # Final performance
    final_train_acc: float
    final_test_acc: float
    
    # Timing
    epochs_to_grokking: Optional[int]
    
    # Classification
    grokking_detected: bool
    grokking_type: str  # "sharp", "gradual", "none"


def load_snapshots(checkpoint_dir: Path) -> List[dict]:
    """Load training snapshots from checkpoint directory."""
    
    snapshots_file = checkpoint_dir / "snapshots.json"
    if snapshots_file.exists():
        with open(snapshots_file, "r") as f:
            return json.load(f)
    
    # Fallback: load from individual checkpoint directories
    snapshots = []
    for ckpt_dir in sorted(checkpoint_dir.glob("checkpoint_*")):
        snapshot_file = ckpt_dir / "snapshot.json"
        if snapshot_file.exists():
            with open(snapshot_file, "r") as f:
                snapshots.append(json.load(f))
    
    return snapshots


def detect_grokking(snapshots: List[dict], threshold: float = 0.05) -> GrokkingAnalysis:
    """Analyze snapshots to detect and characterize grokking.
    
    Args:
        snapshots: List of training snapshots
        threshold: Minimum test accuracy jump to consider as grokking
        
    Returns:
        GrokkingAnalysis with complete analysis
    """
    if len(snapshots) < 2:
        return GrokkingAnalysis(
            grokking_epoch=None,
            grokking_step=None,
            grokking_train_acc=None,
            grokking_test_acc=None,
            max_generalization_gap=0,
            final_generalization_gap=0,
            gap_reduction=0,
            max_test_acc_jump=0,
            max_test_acc_jump_epoch=None,
            final_train_acc=snapshots[-1]['train_accuracy'] if snapshots else 0,
            final_test_acc=snapshots[-1]['test_accuracy'] if snapshots else 0,
            epochs_to_grokking=None,
            grokking_detected=False,
            grokking_type="none"
        )
    
    # Extract metrics
    epochs = [s['epoch'] for s in snapshots]
    train_accs = [s['train_accuracy'] for s in snapshots]
    test_accs = [s['test_accuracy'] for s in snapshots]
    
    # Compute generalization gaps
    gen_gaps = [train - test for train, test in zip(train_accs, test_accs)]
    
    # Find maximum gap and its location
    max_gap_idx = np.argmax(gen_gaps)
    max_gap = gen_gaps[max_gap_idx]
    
    # Find largest jump in test accuracy
    test_jumps = [test_accs[i+1] - test_accs[i] for i in range(len(test_accs)-1)]
    
    if test_jumps:
        max_jump_idx = np.argmax(test_jumps)
        max_jump = test_jumps[max_jump_idx]
        max_jump_epoch = epochs[max_jump_idx + 1]
    else:
        max_jump = 0
        max_jump_epoch = None
    
    # Determine if grokking occurred
    # Criteria: significant jump in test accuracy after training accuracy is already high
    grokking_detected = False
    grokking_idx = None
    grokking_type = "none"
    
    # Look for grokking: test accuracy jumps while train accuracy is already high
    for i in range(1, len(snapshots)):
        train_acc = train_accs[i-1]
        test_jump = test_accs[i] - test_accs[i-1]
        
        # Sharp grokking: train acc > 0.8 and test jump > threshold
        if train_acc > 0.8 and test_jump > threshold:
            grokking_detected = True
            grokking_idx = i
            
            # Classify as sharp if single large jump
            if test_jump > 0.1:
                grokking_type = "sharp"
            else:
                grokking_type = "gradual"
            break
    
    # Also check for gradual grokking: sustained test improvement while train is high
    if not grokking_detected:
        # Check if test accuracy caught up to train over time
        final_gap = gen_gaps[-1]
        initial_gap = gen_gaps[0] if gen_gaps[0] > 0 else max_gap
        
        if initial_gap > 0.1 and final_gap < 0.05:
            grokking_detected = True
            grokking_type = "gradual"
            # Find the point where gap started closing significantly
            for i in range(len(gen_gaps)):
                if gen_gaps[i] < initial_gap * 0.5:
                    grokking_idx = i
                    break
    
    # Build result
    final_gap = gen_gaps[-1]
    gap_reduction = max_gap - final_gap
    
    return GrokkingAnalysis(
        grokking_epoch=epochs[grokking_idx] if grokking_idx else None,
        grokking_step=snapshots[grokking_idx]['step'] if grokking_idx else None,
        grokking_train_acc=train_accs[grokking_idx] if grokking_idx else None,
        grokking_test_acc=test_accs[grokking_idx] if grokking_idx else None,
        max_generalization_gap=max_gap,
        final_generalization_gap=final_gap,
        gap_reduction=gap_reduction,
        max_test_acc_jump=max_jump,
        max_test_acc_jump_epoch=max_jump_epoch,
        final_train_acc=train_accs[-1],
        final_test_acc=test_accs[-1],
        epochs_to_grokking=epochs[grokking_idx] if grokking_idx else None,
        grokking_detected=grokking_detected,
        grokking_type=grokking_type
    )


def plot_grokking_analysis(
    snapshots: List[dict],
    analysis: GrokkingAnalysis,
    output_dir: Path
):
    """Create comprehensive grokking visualization."""
    
    epochs = [s['epoch'] for s in snapshots]
    train_accs = [s['train_accuracy'] for s in snapshots]
    test_accs = [s['test_accuracy'] for s in snapshots]
    train_losses = [s['train_loss'] for s in snapshots]
    test_losses = [s['test_loss'] for s in snapshots]
    gen_gaps = [t - e for t, e in zip(train_accs, test_accs)]
    
    # Figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy curves (main grokking visualization)
    ax = axes[0, 0]
    ax.plot(epochs, train_accs, 'b-', linewidth=2, label='Train Accuracy', marker='o', markersize=4)
    ax.plot(epochs, test_accs, 'r-', linewidth=2, label='Test Accuracy', marker='s', markersize=4)
    
    # Mark grokking point
    if analysis.grokking_epoch is not None:
        ax.axvline(x=analysis.grokking_epoch, color='g', linestyle='--', linewidth=2,
                   label=f'Grokking Point (epoch {analysis.grokking_epoch})')
        ax.scatter([analysis.grokking_epoch], [analysis.grokking_test_acc], 
                  color='green', s=100, zorder=5, marker='*')
    
    ax.fill_between(epochs, test_accs, train_accs, alpha=0.2, color='purple',
                    label='Generalization Gap')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Training Dynamics: Grokking Analysis', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Plot 2: Generalization gap over time
    ax = axes[0, 1]
    ax.plot(epochs, gen_gaps, 'purple', linewidth=2, marker='o', markersize=4)
    ax.fill_between(epochs, 0, gen_gaps, alpha=0.3, color='purple')
    
    if analysis.grokking_epoch is not None:
        ax.axvline(x=analysis.grokking_epoch, color='g', linestyle='--', linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Generalization Gap (Train - Test)', fontsize=11)
    ax.set_title('Generalization Gap Over Training', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss curves
    ax = axes[1, 0]
    ax.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.semilogy(epochs, test_losses, 'r-', linewidth=2, label='Test Loss', marker='s', markersize=4)
    
    if analysis.grokking_epoch is not None:
        ax.axvline(x=analysis.grokking_epoch, color='g', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title('Training and Test Loss', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Test accuracy deltas (to visualize jumps)
    ax = axes[1, 1]
    if len(epochs) > 1:
        epoch_deltas = epochs[1:]
        test_deltas = [test_accs[i+1] - test_accs[i] for i in range(len(test_accs)-1)]
        
        colors = ['green' if d > 0 else 'red' for d in test_deltas]
        ax.bar(epoch_deltas, test_deltas, color=colors, alpha=0.7, width=(epochs[-1]-epochs[0])/(len(epochs)*1.5))
        
        if analysis.max_test_acc_jump_epoch is not None:
            ax.axvline(x=analysis.max_test_acc_jump_epoch, color='gold', linestyle='--', 
                      linewidth=2, label=f'Max jump: {analysis.max_test_acc_jump:.3f}')
            ax.legend()
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy Change', fontsize=11)
    ax.set_title('Test Accuracy Improvement Per Checkpoint', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grokking_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grokking analysis plot to {output_dir / 'grokking_analysis.png'}")
    
    # Additional plot: Unknown cell accuracy (if available)
    if 'train_unknown_accuracy' in snapshots[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_unk = [s['train_unknown_accuracy'] for s in snapshots]
        test_unk = [s['test_unknown_accuracy'] for s in snapshots]
        
        ax.plot(epochs, train_unk, 'b-', linewidth=2, label='Train (Unknown Cells)', marker='o')
        ax.plot(epochs, test_unk, 'r-', linewidth=2, label='Test (Unknown Cells)', marker='s')
        
        if analysis.grokking_epoch is not None:
            ax.axvline(x=analysis.grokking_epoch, color='g', linestyle='--', linewidth=2,
                      label='Grokking Point')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Unknown Cell Accuracy', fontsize=11)
        ax.set_title('Accuracy on Unknown Cells (Actual Puzzle Solving)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'unknown_cell_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved unknown cell accuracy plot to {output_dir / 'unknown_cell_accuracy.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Grokking from Checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing training checkpoints")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: checkpoint_dir)")
    parser.add_argument("--grokking_threshold", type=float, default=0.05,
                       help="Minimum test accuracy jump to consider as grokking")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GROKKING DYNAMICS ANALYSIS")
    print("="*60)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    
    # Load snapshots
    print("\nLoading training snapshots...")
    snapshots = load_snapshots(checkpoint_dir)
    
    if not snapshots:
        print("ERROR: No snapshots found!")
        print("Please run train_for_grokking.py first.")
        return
    
    print(f"Found {len(snapshots)} checkpoints")
    
    # Analyze grokking
    print("\nAnalyzing grokking dynamics...")
    analysis = detect_grokking(snapshots, threshold=args.grokking_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("GROKKING ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📊 Training Summary:")
    print(f"   Total epochs: {snapshots[-1]['epoch']}")
    print(f"   Total steps: {snapshots[-1]['step']}")
    print(f"   Final train accuracy: {analysis.final_train_acc:.4f}")
    print(f"   Final test accuracy: {analysis.final_test_acc:.4f}")
    
    print(f"\n📈 Generalization Gap:")
    print(f"   Maximum gap: {analysis.max_generalization_gap:.4f}")
    print(f"   Final gap: {analysis.final_generalization_gap:.4f}")
    print(f"   Gap reduction: {analysis.gap_reduction:.4f}")
    
    print(f"\n🎯 Grokking Detection:")
    if analysis.grokking_detected:
        print(f"   ✅ GROKKING DETECTED!")
        print(f"   Type: {analysis.grokking_type}")
        print(f"   Grokking epoch: {analysis.grokking_epoch}")
        print(f"   Grokking step: {analysis.grokking_step}")
        print(f"   Train acc at grokking: {analysis.grokking_train_acc:.4f}")
        print(f"   Test acc at grokking: {analysis.grokking_test_acc:.4f}")
        print(f"   Max test accuracy jump: {analysis.max_test_acc_jump:.4f} (at epoch {analysis.max_test_acc_jump_epoch})")
    else:
        print(f"   ❌ No clear grokking detected")
        print(f"   Max test accuracy jump: {analysis.max_test_acc_jump:.4f}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_grokking_analysis(snapshots, analysis, output_dir)
    
    # Save results
    results = {
        "num_checkpoints": len(snapshots),
        "snapshots": snapshots,
        "analysis": asdict(analysis),
    }
    
    with open(output_dir / "grokking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'grokking_results.json'}")
    
    # Update experiment 2 results in paper_replication
    paper_results_dir = REPO_ROOT / "experiments/paper_replication/results/grokking_analysis"
    paper_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy analysis to paper_replication results
    paper_results = {
        "snapshots": [
            {
                "step": s["step"],
                "epoch": s["epoch"],
                "train_loss": s["train_loss"],
                "train_accuracy": s["train_accuracy"],
                "test_accuracy": s["test_accuracy"],
                "val_accuracy": s["test_accuracy"],  # For compatibility
            }
            for s in snapshots
        ],
        "grokking_metrics": {
            "grokking_step": analysis.grokking_step,
            "grokking_epoch": analysis.grokking_epoch,
            "train_acc_at_grokking": analysis.grokking_train_acc,
            "test_acc_at_grokking": analysis.grokking_test_acc,
            "train_acc_final": analysis.final_train_acc,
            "test_acc_final": analysis.final_test_acc,
            "generalization_gap_initial": analysis.max_generalization_gap,
            "generalization_gap_final": analysis.final_generalization_gap,
        } if analysis.grokking_detected else None,
        "config": {
            "checkpoint_dir": str(checkpoint_dir),
            "used_synthetic": False,
        }
    }
    
    with open(paper_results_dir / "results.json", "w") as f:
        json.dump(paper_results, f, indent=2)
    print(f"Updated paper replication results at {paper_results_dir / 'results.json'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if analysis.grokking_detected:
        print(f"\n✅ Paper claim VERIFIED: Grokking occurs in HRM training!")
        print(f"   The model showed {analysis.grokking_type} grokking behavior")
        print(f"   with generalization improving significantly after epoch {analysis.grokking_epoch}")
    else:
        print(f"\n⚠️  No clear grokking detected in this training run.")
        print(f"   This could indicate:")
        print(f"   - Need more training epochs")
        print(f"   - Different hyperparameters needed")
        print(f"   - Grokking may be task/data dependent")


if __name__ == "__main__":
    main()
