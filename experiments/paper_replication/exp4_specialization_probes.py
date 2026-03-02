#!/usr/bin/env python3
"""Experiment 4: Hierarchical Specialization Analysis.

This experiment tests whether z_H and z_L encode different types of information:
- z_H (high-level): Global puzzle state, search variables, overall progress
- z_L (low-level): Per-cell properties, local constraints, candidate values

We train linear probes to predict various targets from each activation stream.
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@dataclass
class ProbeResult:
    """Result of training a linear probe."""
    target: str
    feature_set: str  # "z_H", "z_L", "concat", "diff"
    scope: str        # "global" or "local"
    metric: str       # "accuracy" or "r2"
    score: float
    baseline: float   # Random or majority class baseline
    num_samples: int


def count_sudoku_violations(grid: np.ndarray) -> int:
    """Count Sudoku constraint violations in a 9x9 grid."""
    if grid.shape == (81,):
        grid = grid.reshape(9, 9)
    
    violations = 0
    for r in range(9):
        row = grid[r, :]
        filled = row[row > 0]
        violations += len(filled) - len(set(filled))
    for c in range(9):
        col = grid[:, c]
        filled = col[col > 0]
        violations += len(filled) - len(set(filled))
    for box_r in range(3):
        for box_c in range(3):
            box = grid[box_r*3:(box_r+1)*3, box_c*3:(box_c+1)*3].flatten()
            filled = box[box > 0]
            violations += len(filled) - len(set(filled))
    return violations


def collect_activations(
    model,
    inputs: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    device: str = "cuda",
    batch_size: int = 16,
    max_batches: int = 50
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Collect z_H and z_L activations for all puzzles.
    
    Returns:
        global_activations: Dict with z_H, z_L, labels (pooled per puzzle)
        local_activations: Dict with z_H, z_L, labels (per cell)
    """
    max_steps = cfg.get('halt_max_steps', 8)
    
    all_z_H_global = []
    all_z_L_global = []
    all_global_labels = {
        'is_solved': [],
        'num_violations': [],
        'pct_filled': [],
    }
    
    all_z_H_local = []
    all_z_L_local = []
    all_local_labels = {
        'is_correct': [],
        'cell_idx': [],
        'is_given': [],
    }
    
    N = len(inputs)
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            if num_batches >= max_batches:
                break
            
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
            
            # Run for all steps, collect at final step
            for step in range(max_steps):
                carry, outputs = model(carry, batch)
            
            # Extract activations from inner carry
            z_H = carry.inner_carry.z_H  # [B, seq_len, hidden]
            z_L = carry.inner_carry.z_L  # [B, seq_len, hidden]
            
            # Get predictions
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)  # [B, 81]
            
            # Global features (pooled)
            z_H_pooled = z_H.mean(dim=1)  # [B, hidden]
            z_L_pooled = z_L.mean(dim=1)  # [B, hidden]
            
            all_z_H_global.append(z_H_pooled.cpu())
            all_z_L_global.append(z_L_pooled.cpu())
            
            # Global labels
            for i in range(batch_size_actual):
                pred = preds[i].cpu().numpy()
                label = batch_labels[i].cpu().numpy()
                inp = batch_inputs[i].cpu().numpy()
                
                is_solved = (pred == label).all()
                all_global_labels['is_solved'].append(float(is_solved))
                
                # Count violations
                digits = np.zeros(81, dtype=int)
                for j, tok in enumerate(pred):
                    if 2 <= tok <= 10:
                        digits[j] = tok - 1
                grid = digits.reshape(9, 9)
                violations = count_sudoku_violations(grid)
                all_global_labels['num_violations'].append(float(violations))
                
                # Percent filled
                filled_count = ((pred >= 2) & (pred <= 10)).sum()
                all_global_labels['pct_filled'].append(float(filled_count) / 81)
            
            # Local features (per cell) - sample subset to manage memory
            # Take first puzzle in batch only
            z_H_local = z_H[0]  # [seq_len, hidden]
            z_L_local = z_L[0]  # [seq_len, hidden]
            
            # Skip puzzle embedding positions if present
            puzzle_emb_len = cfg.get('puzzle_emb_ndim', 0)
            if puzzle_emb_len > 0:
                hidden_size = cfg.get('hidden_size', 256)
                puzzle_emb_len = -(puzzle_emb_len // -hidden_size)
                z_H_local = z_H_local[puzzle_emb_len:]
                z_L_local = z_L_local[puzzle_emb_len:]
            
            if z_H_local.shape[0] >= 81:
                z_H_local = z_H_local[:81]
                z_L_local = z_L_local[:81]
                
                all_z_H_local.append(z_H_local.cpu())
                all_z_L_local.append(z_L_local.cpu())
                
                pred = preds[0].cpu().numpy()
                label = batch_labels[0].cpu().numpy()
                inp = batch_inputs[0].cpu().numpy()
                
                for cell in range(81):
                    all_local_labels['is_correct'].append(float(pred[cell] == label[cell]))
                    all_local_labels['cell_idx'].append(float(cell))
                    all_local_labels['is_given'].append(float((inp[cell] >= 2) and (inp[cell] <= 10)))
            
            num_batches += 1
            if num_batches % 10 == 0:
                print(f"  Collected {num_batches} batches...")
    
    # Stack tensors
    global_activations = {
        'z_H': torch.cat(all_z_H_global, dim=0),
        'z_L': torch.cat(all_z_L_global, dim=0),
        'labels': {k: torch.tensor(v) for k, v in all_global_labels.items()}
    }
    
    local_activations = {
        'z_H': torch.cat(all_z_H_local, dim=0),
        'z_L': torch.cat(all_z_L_local, dim=0),
        'labels': {k: torch.tensor(v) for k, v in all_local_labels.items()}
    }
    
    return global_activations, local_activations


def train_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    task: str = "classification",
    epochs: int = 100,
    lr: float = 0.01,
    val_split: float = 0.2
) -> Tuple[nn.Module, float, float]:
    """Train a linear probe.
    
    Args:
        X: Input features [N, D]
        y: Labels [N] (for classification) or [N] (for regression)
        task: "classification" or "regression"
        epochs: Number of training epochs
        lr: Learning rate
        val_split: Validation split ratio
        
    Returns:
        (model, train_score, val_score)
    """
    X = X.float()
    y = y.float() if task == "regression" else y.long()
    
    # Split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Model
    in_dim = X.shape[1]
    if task == "classification":
        out_dim = int(y.max().item()) + 1
        model = nn.Linear(in_dim, out_dim)
        criterion = nn.CrossEntropyLoss()
    else:
        model = nn.Linear(in_dim, 1)
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if task == "classification":
            logits = model(X_train)
            loss = criterion(logits, y_train)
        else:
            preds = model(X_train).squeeze()
            loss = criterion(preds, y_train)
        
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        if task == "classification":
            train_preds = model(X_train).argmax(dim=-1).numpy()
            val_preds = model(X_val).argmax(dim=-1).numpy()
            train_score = accuracy_score(y_train.numpy(), train_preds)
            val_score = accuracy_score(y_val.numpy(), val_preds)
        else:
            train_preds = model(X_train).squeeze().numpy()
            val_preds = model(X_val).squeeze().numpy()
            train_score = r2_score(y_train.numpy(), train_preds)
            val_score = r2_score(y_val.numpy(), val_preds)
    
    return model, train_score, val_score


def run_probe_sweep(
    activations: Dict[str, torch.Tensor],
    scope: str,
    targets: List[str],
    feature_sets: List[str] = ["z_H", "z_L", "concat", "diff"]
) -> List[ProbeResult]:
    """Run probes for all target/feature combinations."""
    
    results = []
    
    z_H = activations['z_H']
    z_L = activations['z_L']
    labels = activations['labels']
    
    for target in targets:
        if target not in labels:
            continue
        
        y = labels[target]
        
        # Determine task type
        unique_vals = len(torch.unique(y))
        if unique_vals <= 10 and y.dtype == torch.long:
            task = "classification"
            metric = "accuracy"
            # Majority class baseline
            _, counts = torch.unique(y, return_counts=True)
            baseline = float(counts.max()) / len(y)
        else:
            task = "regression"
            metric = "r2"
            baseline = 0.0  # R² baseline is 0
        
        for feat_set in feature_sets:
            if feat_set == "z_H":
                X = z_H
            elif feat_set == "z_L":
                X = z_L
            elif feat_set == "concat":
                X = torch.cat([z_H, z_L], dim=-1)
            elif feat_set == "diff":
                X = z_H - z_L
            else:
                continue
            
            try:
                _, train_score, val_score = train_linear_probe(X, y, task=task)
                
                results.append(ProbeResult(
                    target=target,
                    feature_set=feat_set,
                    scope=scope,
                    metric=metric,
                    score=val_score,
                    baseline=baseline,
                    num_samples=len(y),
                ))
            except Exception as e:
                print(f"  Error training probe for {target}/{feat_set}: {e}")
    
    return results


def plot_specialization_comparison(results: List[ProbeResult], output_path: Path):
    """Plot comparison of z_H vs z_L probe performance."""
    
    # Separate by scope
    global_results = [r for r in results if r.scope == "global"]
    local_results = [r for r in results if r.scope == "local"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Global probes
    ax1 = axes[0]
    targets_g = list(set(r.target for r in global_results))
    x = np.arange(len(targets_g))
    width = 0.35
    
    z_H_scores = []
    z_L_scores = []
    for t in targets_g:
        h_score = next((r.score for r in global_results if r.target == t and r.feature_set == "z_H"), 0)
        l_score = next((r.score for r in global_results if r.target == t and r.feature_set == "z_L"), 0)
        z_H_scores.append(h_score)
        z_L_scores.append(l_score)
    
    bars1 = ax1.bar(x - width/2, z_H_scores, width, label='z_H (High-level)', color='steelblue')
    bars2 = ax1.bar(x + width/2, z_L_scores, width, label='z_L (Low-level)', color='coral')
    
    ax1.set_xlabel("Target")
    ax1.set_ylabel("Probe Score")
    ax1.set_title("Global Probes: z_H vs z_L")
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets_g, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Local probes
    ax2 = axes[1]
    targets_l = list(set(r.target for r in local_results))
    x = np.arange(len(targets_l))
    
    z_H_scores = []
    z_L_scores = []
    for t in targets_l:
        h_score = next((r.score for r in local_results if r.target == t and r.feature_set == "z_H"), 0)
        l_score = next((r.score for r in local_results if r.target == t and r.feature_set == "z_L"), 0)
        z_H_scores.append(h_score)
        z_L_scores.append(l_score)
    
    bars1 = ax2.bar(x - width/2, z_H_scores, width, label='z_H (High-level)', color='steelblue')
    bars2 = ax2.bar(x + width/2, z_L_scores, width, label='z_L (Low-level)', color='coral')
    
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Probe Score")
    ax2.set_title("Local Probes: z_H vs z_L")
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets_l, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved specialization plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Specialization Analysis")
    parser.add_argument("--checkpoint", type=str,
                       default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/paper_replication/results/specialization")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=50)
    
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
    
    # Collect activations
    print("\nCollecting activations...")
    global_acts, local_acts = collect_activations(
        model, inputs, labels, cfg, args.device, args.batch_size, args.max_batches
    )
    
    print(f"Global activations: z_H={global_acts['z_H'].shape}, z_L={global_acts['z_L'].shape}")
    print(f"Local activations: z_H={local_acts['z_H'].shape}, z_L={local_acts['z_L'].shape}")
    
    # Run probe sweep
    print("\nTraining global probes...")
    global_targets = ['is_solved', 'num_violations', 'pct_filled']
    global_results = run_probe_sweep(global_acts, "global", global_targets)
    
    print("\nTraining local probes...")
    local_targets = ['is_correct', 'is_given']
    local_results = run_probe_sweep(local_acts, "local", local_targets)
    
    all_results = global_results + local_results
    
    # Print results
    print("\n" + "="*80)
    print("PROBE RESULTS")
    print("="*80)
    
    print("\nGlobal Probes:")
    print("-" * 60)
    for r in global_results:
        print(f"  {r.target:20} | {r.feature_set:10} | {r.metric}: {r.score:.4f} (baseline: {r.baseline:.4f})")
    
    print("\nLocal Probes:")
    print("-" * 60)
    for r in local_results:
        print(f"  {r.target:20} | {r.feature_set:10} | {r.metric}: {r.score:.4f} (baseline: {r.baseline:.4f})")
    
    # Create plots
    plot_specialization_comparison(all_results, output_dir / "specialization_comparison.png")
    
    # Save results
    results_dict = {
        "probe_results": [asdict(r) for r in all_results],
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Analysis
    print("\n" + "="*60)
    print("SPECIALIZATION ANALYSIS SUMMARY")
    print("="*60)
    
    # Compare z_H vs z_L for global targets
    print("\nGlobal targets - which stream encodes better:")
    for target in global_targets:
        h_score = next((r.score for r in global_results if r.target == target and r.feature_set == "z_H"), 0)
        l_score = next((r.score for r in global_results if r.target == target and r.feature_set == "z_L"), 0)
        
        if h_score > l_score:
            winner = "z_H (high-level)"
            diff = h_score - l_score
        else:
            winner = "z_L (low-level)"
            diff = l_score - h_score
        
        print(f"  {target}: {winner} wins by {diff:.4f}")
    
    # Compare z_H vs z_L for local targets
    print("\nLocal targets - which stream encodes better:")
    for target in local_targets:
        h_score = next((r.score for r in local_results if r.target == target and r.feature_set == "z_H"), 0)
        l_score = next((r.score for r in local_results if r.target == target and r.feature_set == "z_L"), 0)
        
        if h_score > l_score:
            winner = "z_H (high-level)"
            diff = h_score - l_score
        else:
            winner = "z_L (low-level)"
            diff = l_score - h_score
        
        print(f"  {target}: {winner} wins by {diff:.4f}")


if __name__ == "__main__":
    main()
