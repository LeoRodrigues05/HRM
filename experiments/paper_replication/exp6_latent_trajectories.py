#!/usr/bin/env python3
"""Experiment 6: Latent Reasoning Trajectory Analysis (Figure 3).

This experiment visualizes the latent reasoning trajectories using PCA:
- Project z_H and z_L activations onto first 2 principal components
- Track trajectory across reasoning steps for each puzzle
- Analyze stability and symmetry of trajectories
- Compare trajectories for puzzles with same solution structure
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@dataclass
class TrajectoryMetrics:
    """Metrics for trajectory analysis."""
    puzzle_idx: int
    is_solved: bool
    trajectory_length: float  # Total path length in PCA space
    final_drift: float        # Distance from convergence point
    symmetry_score: float     # How symmetric the trajectory is


def collect_step_activations(
    model,
    inputs: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    device: str = "cuda",
    num_puzzles: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect z_H and z_L activations at each step for multiple puzzles.
    
    Returns:
        z_H_steps: [num_puzzles, max_steps, hidden_size] - pooled z_H per step
        z_L_steps: [num_puzzles, max_steps, hidden_size] - pooled z_L per step
        is_correct_steps: [num_puzzles, max_steps] - whether prediction is correct at each step
        final_solved: [num_puzzles] - whether puzzle is fully solved
    """
    max_steps = cfg.get('halt_max_steps', 16)
    hidden_size = cfg.get('hidden_size', 512)
    
    # Select random puzzles
    N = len(inputs)
    indices = np.random.choice(N, min(num_puzzles, N), replace=False)
    
    z_H_all = []
    z_L_all = []
    correct_all = []
    solved_all = []
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            inp = torch.tensor(inputs[idx:idx+1], dtype=torch.long, device=device)
            lab = torch.tensor(labels[idx:idx+1], dtype=torch.long, device=device)
            
            batch = {
                "inputs": inp,
                "labels": lab,
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device),
            }
            
            carry = model.initial_carry(batch)
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
            carry.steps = carry.steps.to(device)
            carry.halted = carry.halted.to(device)
            
            z_H_steps = []
            z_L_steps = []
            correct_steps = []
            
            for step in range(max_steps):
                carry, outputs = model(carry, batch)
                
                # Get pooled activations (mean over sequence)
                # Convert from bfloat16 to float32 for numpy compatibility
                z_H = carry.inner_carry.z_H.float().mean(dim=1).cpu().numpy()  # [1, hidden]
                z_L = carry.inner_carry.z_L.float().mean(dim=1).cpu().numpy()  # [1, hidden]
                
                z_H_steps.append(z_H[0])
                z_L_steps.append(z_L[0])
                
                # Check if correct at this step
                preds = outputs["logits"].argmax(dim=-1)
                is_correct = (preds == lab).all().item()
                correct_steps.append(is_correct)
            
            z_H_all.append(np.stack(z_H_steps, axis=0))
            z_L_all.append(np.stack(z_L_steps, axis=0))
            correct_all.append(correct_steps)
            solved_all.append(correct_steps[-1])
    
    return (
        np.stack(z_H_all, axis=0),  # [num_puzzles, max_steps, hidden]
        np.stack(z_L_all, axis=0),  # [num_puzzles, max_steps, hidden]
        np.array(correct_all),      # [num_puzzles, max_steps]
        np.array(solved_all)        # [num_puzzles]
    )


def compute_trajectory_metrics(
    trajectories: np.ndarray,  # [num_puzzles, max_steps, 2] in PCA space
    is_solved: np.ndarray      # [num_puzzles]
) -> List[TrajectoryMetrics]:
    """Compute metrics for each trajectory."""
    metrics = []
    
    for i in range(len(trajectories)):
        traj = trajectories[i]  # [max_steps, 2]
        
        # Path length
        diffs = np.diff(traj, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        
        # Final drift (distance from last point to centroid of final 3 steps)
        final_centroid = traj[-3:].mean(axis=0)
        final_drift = np.linalg.norm(traj[-1] - final_centroid)
        
        # Symmetry score (variance of step sizes)
        step_sizes = np.linalg.norm(diffs, axis=1)
        symmetry = 1.0 / (1.0 + np.std(step_sizes))
        
        metrics.append(TrajectoryMetrics(
            puzzle_idx=i,
            is_solved=bool(is_solved[i]),
            trajectory_length=float(path_length),
            final_drift=float(final_drift),
            symmetry_score=float(symmetry)
        ))
    
    return metrics


def plot_trajectories(
    z_H_pca: np.ndarray,       # [num_puzzles, max_steps, 2]
    z_L_pca: np.ndarray,       # [num_puzzles, max_steps, 2]
    is_correct: np.ndarray,    # [num_puzzles, max_steps]
    is_solved: np.ndarray,     # [num_puzzles]
    output_dir: Path,
    max_display: int = 20
):
    """Plot latent trajectories similar to Figure 3."""
    
    num_puzzles = min(len(z_H_pca), max_display)
    
    # Figure 1: z_H trajectories colored by step
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # z_H trajectories
    ax = axes[0]
    cmap = plt.cm.viridis
    max_steps = z_H_pca.shape[1]
    
    for i in range(num_puzzles):
        traj = z_H_pca[i]
        color = 'green' if is_solved[i] else 'red'
        alpha = 0.7 if is_solved[i] else 0.3
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=alpha, linewidth=1)
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=30, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=50, marker='*', zorder=5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('z_H (High-Level) Trajectories\nGreen=Solved, Red=Unsolved')
    ax.grid(True, alpha=0.3)
    
    # z_L trajectories
    ax = axes[1]
    for i in range(num_puzzles):
        traj = z_L_pca[i]
        color = 'green' if is_solved[i] else 'red'
        alpha = 0.7 if is_solved[i] else 0.3
        
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=alpha, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=30, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=50, marker='*', zorder=5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('z_L (Low-Level) Trajectories\nGreen=Solved, Red=Unsolved')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot to {output_dir / 'latent_trajectories.png'}")
    
    # Figure 2: Trajectories colored by reasoning step
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, (data, title) in enumerate([(z_H_pca, 'z_H'), (z_L_pca, 'z_L')]):
        ax = axes[ax_idx]
        
        for i in range(num_puzzles):
            traj = data[i]
            # Color by step
            for step in range(max_steps - 1):
                color = cmap(step / max_steps)
                ax.plot(traj[step:step+2, 0], traj[step:step+2, 1], 
                       '-', color=color, alpha=0.5, linewidth=1.5)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_steps))
        sm.set_array([])
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{title} Trajectories (colored by reasoning step)')
        ax.grid(True, alpha=0.3)
    
    # Add shared colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Reasoning Step')
    
    plt.savefig(output_dir / 'trajectories_by_step.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved step-colored trajectory plot to {output_dir / 'trajectories_by_step.png'}")
    
    # Figure 3: Single puzzle detailed trajectory
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Find one solved and one unsolved puzzle
    solved_idx = np.where(is_solved)[0]
    unsolved_idx = np.where(~is_solved)[0]
    
    examples = []
    if len(solved_idx) > 0:
        examples.append(('Solved Puzzle', solved_idx[0]))
    if len(unsolved_idx) > 0:
        examples.append(('Unsolved Puzzle', unsolved_idx[0]))
    
    for row, (label, idx) in enumerate(examples[:2]):
        for col, (data, stream_name) in enumerate([(z_H_pca, 'z_H'), (z_L_pca, 'z_L')]):
            ax = axes[row, col]
            traj = data[idx]
            
            # Plot with arrows showing direction
            for step in range(max_steps - 1):
                color = cmap(step / max_steps)
                dx = traj[step+1, 0] - traj[step, 0]
                dy = traj[step+1, 1] - traj[step, 1]
                ax.annotate('', xy=(traj[step+1, 0], traj[step+1, 1]),
                           xytext=(traj[step, 0], traj[step, 1]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
            
            # Mark steps with numbers
            for step in range(0, max_steps, 2):
                ax.annotate(str(step), (traj[step, 0], traj[step, 1]), 
                           fontsize=8, ha='center', va='bottom')
            
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', 
                      zorder=10, label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='*', 
                      zorder=10, label='End')
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'{label} - {stream_name}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory examples to {output_dir / 'trajectory_examples.png'}")


def plot_trajectory_stats(
    metrics: List[TrajectoryMetrics],
    output_dir: Path
):
    """Plot trajectory statistics."""
    
    solved = [m for m in metrics if m.is_solved]
    unsolved = [m for m in metrics if not m.is_solved]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Path length comparison
    ax = axes[0]
    data = [
        [m.trajectory_length for m in solved],
        [m.trajectory_length for m in unsolved]
    ]
    ax.boxplot(data, labels=['Solved', 'Unsolved'])
    ax.set_ylabel('Path Length')
    ax.set_title('Trajectory Length by Outcome')
    
    # Final drift comparison
    ax = axes[1]
    data = [
        [m.final_drift for m in solved],
        [m.final_drift for m in unsolved]
    ]
    ax.boxplot(data, labels=['Solved', 'Unsolved'])
    ax.set_ylabel('Final Drift')
    ax.set_title('Convergence Stability by Outcome')
    
    # Symmetry comparison
    ax = axes[2]
    data = [
        [m.symmetry_score for m in solved],
        [m.symmetry_score for m in unsolved]
    ]
    ax.boxplot(data, labels=['Solved', 'Unsolved'])
    ax.set_ylabel('Symmetry Score')
    ax.set_title('Trajectory Symmetry by Outcome')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory stats to {output_dir / 'trajectory_stats.png'}")


def main():
    parser = argparse.ArgumentParser(description="Latent Trajectory Analysis")
    parser.add_argument("--checkpoint", type=str,
                       default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/paper_replication/results/latent_trajectories")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_puzzles", type=int, default=200)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    
    config_path = Path(args.checkpoint).parent / "all_config.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    dataset_json_path = Path(args.data_path) / "test" / "dataset.json"
    with open(dataset_json_path, "r") as f:
        dataset_metadata = json.load(f)
    
    cfg = full_config['arch'].copy()
    cfg['batch_size'] = full_config.get('global_batch_size', 768)
    cfg['seq_len'] = dataset_metadata['seq_len']
    cfg['vocab_size'] = dataset_metadata['vocab_size']
    cfg['num_puzzle_identifiers'] = dataset_metadata['num_puzzle_identifiers']
    
    model = HierarchicalReasoningModel_ACTV1(cfg).to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
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
    
    print(f"\nCollecting activations for {args.num_puzzles} puzzles...")
    z_H_steps, z_L_steps, is_correct, is_solved = collect_step_activations(
        model, inputs, labels, cfg, args.device, args.num_puzzles
    )
    
    print(f"z_H shape: {z_H_steps.shape}")
    print(f"z_L shape: {z_L_steps.shape}")
    print(f"Solved: {is_solved.sum()}/{len(is_solved)}")
    
    # Apply PCA
    print("\nApplying PCA...")
    num_puzzles, max_steps, hidden_size = z_H_steps.shape
    
    # Flatten for PCA fitting
    z_H_flat = z_H_steps.reshape(-1, hidden_size)
    z_L_flat = z_L_steps.reshape(-1, hidden_size)
    
    # Fit PCA on combined data
    pca_H = PCA(n_components=2)
    pca_L = PCA(n_components=2)
    
    z_H_pca_flat = pca_H.fit_transform(z_H_flat)
    z_L_pca_flat = pca_L.fit_transform(z_L_flat)
    
    # Reshape back
    z_H_pca = z_H_pca_flat.reshape(num_puzzles, max_steps, 2)
    z_L_pca = z_L_pca_flat.reshape(num_puzzles, max_steps, 2)
    
    print(f"PCA variance explained - z_H: {pca_H.explained_variance_ratio_.sum():.3f}")
    print(f"PCA variance explained - z_L: {pca_L.explained_variance_ratio_.sum():.3f}")
    
    # Plot trajectories
    print("\nPlotting trajectories...")
    plot_trajectories(z_H_pca, z_L_pca, is_correct, is_solved, output_dir)
    
    # Compute and plot metrics
    print("\nComputing trajectory metrics...")
    metrics_H = compute_trajectory_metrics(z_H_pca, is_solved)
    metrics_L = compute_trajectory_metrics(z_L_pca, is_solved)
    
    plot_trajectory_stats(metrics_H, output_dir)
    
    # Save results
    results = {
        "num_puzzles": int(num_puzzles),
        "num_solved": int(is_solved.sum()),
        "pca_variance_explained_H": pca_H.explained_variance_ratio_.tolist(),
        "pca_variance_explained_L": pca_L.explained_variance_ratio_.tolist(),
        "metrics_H": {
            "solved": {
                "mean_path_length": np.mean([m.trajectory_length for m in metrics_H if m.is_solved]),
                "mean_final_drift": np.mean([m.final_drift for m in metrics_H if m.is_solved]),
                "mean_symmetry": np.mean([m.symmetry_score for m in metrics_H if m.is_solved]),
            },
            "unsolved": {
                "mean_path_length": np.mean([m.trajectory_length for m in metrics_H if not m.is_solved]) if any(not m.is_solved for m in metrics_H) else 0,
                "mean_final_drift": np.mean([m.final_drift for m in metrics_H if not m.is_solved]) if any(not m.is_solved for m in metrics_H) else 0,
                "mean_symmetry": np.mean([m.symmetry_score for m in metrics_H if not m.is_solved]) if any(not m.is_solved for m in metrics_H) else 0,
            }
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("LATENT TRAJECTORY ANALYSIS SUMMARY")
    print("="*60)
    
    solved_H = [m for m in metrics_H if m.is_solved]
    unsolved_H = [m for m in metrics_H if not m.is_solved]
    
    if solved_H:
        print(f"\nSolved puzzles (n={len(solved_H)}):")
        print(f"  Mean path length: {np.mean([m.trajectory_length for m in solved_H]):.3f}")
        print(f"  Mean final drift: {np.mean([m.final_drift for m in solved_H]):.4f}")
        print(f"  Mean symmetry: {np.mean([m.symmetry_score for m in solved_H]):.3f}")
    
    if unsolved_H:
        print(f"\nUnsolved puzzles (n={len(unsolved_H)}):")
        print(f"  Mean path length: {np.mean([m.trajectory_length for m in unsolved_H]):.3f}")
        print(f"  Mean final drift: {np.mean([m.final_drift for m in unsolved_H]):.4f}")
        print(f"  Mean symmetry: {np.mean([m.symmetry_score for m in unsolved_H]):.3f}")


if __name__ == "__main__":
    main()
