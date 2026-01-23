#!/usr/bin/env python3
"""Experiment 5: Activation Patching Ablation Study.

This experiment systematically tests the causal role of different activation streams
and steps using activation patching:
- Which stream (z_H vs z_L) is more important?
- Which steps are critical for reasoning?
- Does patching transfer useful information between puzzles?

Uses the existing activation_patching.py infrastructure.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@dataclass
class PatchingResult:
    """Result of a single patching experiment."""
    source_idx: int
    target_idx: int
    patch_level: str
    patch_steps: str
    baseline_accuracy: float
    patched_accuracy: float
    accuracy_change: float
    baseline_unknown_accuracy: float
    patched_unknown_accuracy: float


def run_patching_experiment(
    model,
    inputs: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    source_idx: int,
    target_idx: int,
    patch_level: str,  # "H", "L", or "both"
    patch_steps: List[int],
    device: str = "cuda"
) -> PatchingResult:
    """Run a single activation patching experiment.
    
    Args:
        model: HRM model
        inputs: All puzzle inputs
        labels: All puzzle labels
        cfg: Model config
        source_idx: Index of source puzzle
        target_idx: Index of target puzzle
        patch_level: Which activations to patch ("H", "L", or "both")
        patch_steps: List of steps to patch
        device: Device to run on
        
    Returns:
        PatchingResult with accuracies
    """
    max_steps = cfg.get('halt_max_steps', 8)
    
    # Get source and target
    source_input = torch.tensor(inputs[source_idx:source_idx+1], dtype=torch.long, device=device)
    target_input = torch.tensor(inputs[target_idx:target_idx+1], dtype=torch.long, device=device)
    target_label = torch.tensor(labels[target_idx:target_idx+1], dtype=torch.long, device=device)
    
    model.eval()
    
    # First, run source puzzle to collect activations
    source_activations = {}
    with torch.no_grad():
        batch_source = {
            "inputs": source_input,
            "labels": source_input,
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device),
        }
        carry = model.initial_carry(batch_source)
        # Move carry to device
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        
        for step in range(max_steps):
            carry, outputs = model(carry, batch_source)
            if step in patch_steps:
                source_activations[step] = {
                    'z_H': carry.inner_carry.z_H.clone(),
                    'z_L': carry.inner_carry.z_L.clone(),
                }
    
    # Run target puzzle without patching (baseline)
    with torch.no_grad():
        batch_target = {
            "inputs": target_input,
            "labels": target_label,
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device),
        }
        carry = model.initial_carry(batch_target)
        # Move carry to device
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        
        for step in range(max_steps):
            carry, outputs = model(carry, batch_target)
        
        baseline_preds = outputs["logits"].argmax(dim=-1)
        baseline_acc = (baseline_preds == target_label).float().mean().item()
        
        # Unknown cell accuracy
        is_given = (target_input >= 2) & (target_input <= 10)
        is_unknown = ~is_given
        if is_unknown.sum() > 0:
            baseline_unknown_acc = ((baseline_preds == target_label) & is_unknown).sum().float() / is_unknown.sum().float()
            baseline_unknown_acc = baseline_unknown_acc.item()
        else:
            baseline_unknown_acc = 1.0
    
    # Run target puzzle WITH patching
    with torch.no_grad():
        carry = model.initial_carry(batch_target)
        # Move carry to device
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        
        for step in range(max_steps):
            carry, outputs = model(carry, batch_target)
            
            # Apply patch if this step is in patch_steps
            if step in patch_steps and step in source_activations:
                if patch_level in ["H", "both"]:
                    carry.inner_carry.z_H.copy_(source_activations[step]['z_H'])
                if patch_level in ["L", "both"]:
                    carry.inner_carry.z_L.copy_(source_activations[step]['z_L'])
        
        patched_preds = outputs["logits"].argmax(dim=-1)
        patched_acc = (patched_preds == target_label).float().mean().item()
        
        if is_unknown.sum() > 0:
            patched_unknown_acc = ((patched_preds == target_label) & is_unknown).sum().float() / is_unknown.sum().float()
            patched_unknown_acc = patched_unknown_acc.item()
        else:
            patched_unknown_acc = 1.0
    
    return PatchingResult(
        source_idx=source_idx,
        target_idx=target_idx,
        patch_level=patch_level,
        patch_steps=",".join(map(str, patch_steps)),
        baseline_accuracy=baseline_acc,
        patched_accuracy=patched_acc,
        accuracy_change=patched_acc - baseline_acc,
        baseline_unknown_accuracy=baseline_unknown_acc,
        patched_unknown_accuracy=patched_unknown_acc,
    )


def run_patching_sweep(
    model,
    inputs: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    num_pairs: int = 20,
    device: str = "cuda"
) -> List[PatchingResult]:
    """Run sweep over multiple patching configurations."""
    
    max_steps = cfg.get('halt_max_steps', 8)
    N = len(inputs)
    
    # Define patch configurations
    patch_levels = ["H", "L", "both"]
    patch_step_configs = {
        "all": list(range(max_steps)),
        "early": [0, 1, 2],
        "late": list(range(3, max_steps)),
        "first": [0],
        "last": [max_steps - 1],
    }
    
    # Generate random puzzle pairs
    np.random.seed(42)
    pairs = []
    for _ in range(num_pairs):
        source = np.random.randint(0, N)
        target = np.random.randint(0, N)
        while target == source:
            target = np.random.randint(0, N)
        pairs.append((source, target))
    
    results = []
    total_exps = len(pairs) * len(patch_levels) * len(patch_step_configs)
    exp_count = 0
    
    for source_idx, target_idx in pairs:
        for patch_level in patch_levels:
            for config_name, patch_steps in patch_step_configs.items():
                exp_count += 1
                if exp_count % 20 == 0:
                    print(f"  Running experiment {exp_count}/{total_exps}...")
                
                result = run_patching_experiment(
                    model, inputs, labels, cfg,
                    source_idx, target_idx,
                    patch_level, patch_steps, device
                )
                result.patch_steps = config_name
                results.append(result)
    
    return results


def analyze_patching_results(results: List[PatchingResult]) -> Dict:
    """Analyze patching results to understand causal structure."""
    
    analysis = {
        "by_level": {},
        "by_steps": {},
        "overall": {},
    }
    
    # By patch level
    for level in ["H", "L", "both"]:
        level_results = [r for r in results if r.patch_level == level]
        if level_results:
            changes = [r.accuracy_change for r in level_results]
            unknown_changes = [r.patched_unknown_accuracy - r.baseline_unknown_accuracy 
                             for r in level_results]
            analysis["by_level"][level] = {
                "mean_change": float(np.mean(changes)),
                "std_change": float(np.std(changes)),
                "min_change": float(np.min(changes)),
                "max_change": float(np.max(changes)),
                "mean_unknown_change": float(np.mean(unknown_changes)),
                "num_positive": sum(1 for c in changes if c > 0.01),
                "num_negative": sum(1 for c in changes if c < -0.01),
                "count": len(changes),
            }
    
    # By step configuration
    step_configs = list(set(r.patch_steps for r in results))
    for config in step_configs:
        config_results = [r for r in results if r.patch_steps == config]
        if config_results:
            changes = [r.accuracy_change for r in config_results]
            analysis["by_steps"][config] = {
                "mean_change": float(np.mean(changes)),
                "std_change": float(np.std(changes)),
                "count": len(changes),
            }
    
    # Overall
    all_changes = [r.accuracy_change for r in results]
    analysis["overall"] = {
        "mean_change": float(np.mean(all_changes)),
        "std_change": float(np.std(all_changes)),
        "mean_baseline": float(np.mean([r.baseline_accuracy for r in results])),
        "mean_patched": float(np.mean([r.patched_accuracy for r in results])),
        "num_experiments": len(results),
    }
    
    return analysis


def plot_patching_analysis(results: List[PatchingResult], analysis: Dict, output_dir: Path):
    """Create visualization of patching results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy change by patch level
    ax1 = axes[0, 0]
    levels = list(analysis["by_level"].keys())
    means = [analysis["by_level"][l]["mean_change"] for l in levels]
    stds = [analysis["by_level"][l]["std_change"] for l in levels]
    
    colors = {'H': 'steelblue', 'L': 'coral', 'both': 'green'}
    x = np.arange(len(levels))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, 
                   color=[colors.get(l, 'gray') for l in levels])
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"z_{l}" if l != "both" else "z_H + z_L" for l in levels])
    ax1.set_xlabel("Patch Level")
    ax1.set_ylabel("Accuracy Change")
    ax1.set_title("Effect of Patching by Activation Stream")
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Accuracy change by step config
    ax2 = axes[0, 1]
    configs = list(analysis["by_steps"].keys())
    means = [analysis["by_steps"][c]["mean_change"] for c in configs]
    stds = [analysis["by_steps"][c]["std_change"] for c in configs]
    
    x = np.arange(len(configs))
    ax2.bar(x, means, yerr=stds, capsize=5, color='mediumpurple')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.set_xlabel("Patch Steps")
    ax2.set_ylabel("Accuracy Change")
    ax2.set_title("Effect of Patching by Step Configuration")
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Distribution of accuracy changes
    ax3 = axes[1, 0]
    changes = [r.accuracy_change for r in results]
    ax3.hist(changes, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(changes), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(changes):.4f}')
    ax3.set_xlabel("Accuracy Change")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Patching Effects")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Baseline vs Patched accuracy
    ax4 = axes[1, 1]
    baselines = [r.baseline_accuracy for r in results]
    patched = [r.patched_accuracy for r in results]
    
    # Color by patch level
    for level, color in colors.items():
        level_results = [r for r in results if r.patch_level == level]
        b = [r.baseline_accuracy for r in level_results]
        p = [r.patched_accuracy for r in level_results]
        ax4.scatter(b, p, alpha=0.4, s=20, color=color, label=f'z_{level}' if level != 'both' else 'z_H + z_L')
    
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No change')
    ax4.set_xlabel("Baseline Accuracy")
    ax4.set_ylabel("Patched Accuracy")
    ax4.set_title("Baseline vs Patched Performance")
    ax4.legend(loc='lower right')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 1.05)
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "patching_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved patching analysis plot to {output_dir / 'patching_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Activation Patching Ablation Study")
    parser.add_argument("--checkpoint", type=str,
                       default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/paper_replication/results/activation_patching")
    parser.add_argument("--num_pairs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    import yaml
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
    
    # Run patching sweep
    print(f"\nRunning activation patching sweep with {args.num_pairs} puzzle pairs...")
    results = run_patching_sweep(
        model, inputs, labels, cfg,
        num_pairs=args.num_pairs,
        device=args.device
    )
    
    print(f"\nCompleted {len(results)} patching experiments")
    
    # Analyze results
    analysis = analyze_patching_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("ACTIVATION PATCHING RESULTS")
    print("="*60)
    
    print("\nBy Patch Level:")
    print("-" * 50)
    for level, stats in analysis["by_level"].items():
        print(f"  z_{level}:")
        print(f"    Mean accuracy change: {stats['mean_change']:+.4f} ± {stats['std_change']:.4f}")
        print(f"    Range: [{stats['min_change']:.4f}, {stats['max_change']:.4f}]")
        print(f"    Improved/Degraded: {stats['num_positive']}/{stats['num_negative']}")
    
    print("\nBy Step Configuration:")
    print("-" * 50)
    for config, stats in analysis["by_steps"].items():
        print(f"  {config}: {stats['mean_change']:+.4f} ± {stats['std_change']:.4f}")
    
    # Create plots
    plot_patching_analysis(results, analysis, output_dir)
    
    # Save results
    results_dict = {
        "patching_results": [asdict(r) for r in results],
        "analysis": analysis,
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "num_pairs": args.num_pairs,
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Summary findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Which stream is more important?
    h_effect = abs(analysis["by_level"]["H"]["mean_change"])
    l_effect = abs(analysis["by_level"]["L"]["mean_change"])
    
    if h_effect > l_effect:
        print(f"\n✓ z_H (high-level) has LARGER causal effect ({h_effect:.4f} vs {l_effect:.4f})")
        print("  This suggests high-level reasoning is more important for solution transfer")
    else:
        print(f"\n✓ z_L (low-level) has LARGER causal effect ({l_effect:.4f} vs {h_effect:.4f})")
        print("  This suggests local computations are more important for solution transfer")
    
    # Which steps matter most?
    step_effects = [(k, abs(v["mean_change"])) for k, v in analysis["by_steps"].items()]
    step_effects.sort(key=lambda x: x[1], reverse=True)
    print(f"\nMost impactful step configuration: {step_effects[0][0]} ({step_effects[0][1]:.4f})")


if __name__ == "__main__":
    main()
