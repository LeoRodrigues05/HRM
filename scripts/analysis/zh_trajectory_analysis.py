#!/usr/bin/env python3
"""z_H Trajectory Analysis — PCA visualization + directional metrics.

Collects z_H activations at each ACT step for N puzzles, then produces:
  1. PCA trajectory plots (2D) colored by step, separated by success/failure
  2. Cosine similarity between consecutive z_H states
  3. Norm evolution of z_H across steps
  4. Alignment with final state (cosine of z_H[step] vs z_H[15])
  5. Summary statistics and JSON output

Reuses the ActivationAblator infrastructure for z_H collection.

Usage:
    python scripts/analysis/zh_trajectory_analysis.py --n_puzzles 200 --device cuda
    python scripts/analysis/zh_trajectory_analysis.py --n_puzzles 50 --device cpu --quick
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _patch_attention_for_cpu,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_CELLS = 81
DIGIT_OFFSET = 1


# ═══════════════════════════════════════════════════════════════════════════
# Model loading (same pattern as sae_collect_activations.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data(checkpoint_path: str, device: torch.device):
    """Load HRM model and test dataloader."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_meta = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__, batch_size=1,
        vocab_size=test_meta.vocab_size, seq_len=test_meta.seq_len,
        num_puzzle_identifiers=test_meta.num_puzzle_identifiers, causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(
        k.startswith("_orig_mod.") for k in ck
    ):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(
        k.startswith("_orig_mod.") for k in mk
    ):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, "model"):
        m = m.model

    return m, test_loader


# ═══════════════════════════════════════════════════════════════════════════
# z_H collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_zh_trajectories(
    model: HierarchicalReasoningModel_ACTV1,
    test_loader,
    device: torch.device,
    n_puzzles: int,
    max_steps: int = 16,
) -> Dict[str, Any]:
    """Collect mean-pooled z_H at each ACT step for each puzzle.

    Returns dict with:
        'zh_mean': [N, steps, D] — mean-pooled z_H across 81 cells
        'per_step_accuracy': [N, steps] — cell accuracy at each step
        'final_accuracy': [N] — accuracy at last step
        'final_puzzle_acc': [N] — puzzle-level accuracy (1 if exact match)
        'inputs': [N, 81]
        'labels': [N, 81]
    """
    ablator = ActivationAblator(model, device=device)

    all_zh_mean = []      # [N, steps, D]
    all_step_acc = []     # [N, steps]
    all_final_acc = []    # [N]
    all_final_puzzle = [] # [N]
    all_inputs = []
    all_labels = []

    n_done = 0
    for data in test_loader:
        if n_done >= n_puzzles:
            break

        if isinstance(data, (tuple, list)):
            batch = data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        else:
            batch = data
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)

        targets = batch["labels"][:, -SUDOKU_CELLS:]  # [1, 81]
        inputs = batch["inputs"][:, -SUDOKU_CELLS:]

        zh_steps = []
        step_accs = []
        for step in range(max_steps):
            if step in cache:
                ac = cache[step]
                # z_H_out is post-forward z_H: [1, seq_len, D]
                # Take last 81 positions (puzzle cells, skip puzzle embedding prefix)
                z_H = ac.z_H_out[:, -SUDOKU_CELLS:, :].float()  # [1, 81, D]
                zh_mean = z_H.mean(dim=1).squeeze(0).cpu()  # [D]
                zh_steps.append(zh_mean)

                preds = ac.preds[:, -SUDOKU_CELLS:].cpu()
                target_cpu = targets.cpu()
                mask = target_cpu != -100
                correct = (preds == target_cpu) & mask
                acc = correct.float().sum() / mask.float().sum().clamp(min=1)
                step_accs.append(acc.item())
            else:
                zh_steps.append(torch.zeros(512))
                step_accs.append(0.0)

        all_zh_mean.append(torch.stack(zh_steps))  # [steps, D]
        all_step_acc.append(step_accs)
        all_final_acc.append(step_accs[-1])

        # Puzzle accuracy (exact match at final step)
        if max_steps - 1 in cache:
            final_preds = cache[max_steps - 1].preds[:, -SUDOKU_CELLS:].cpu()
            target_cpu = targets.cpu()
            mask = target_cpu != -100
            puzzle_correct = ((final_preds == target_cpu) | ~mask).all().item()
        else:
            puzzle_correct = False
        all_final_puzzle.append(float(puzzle_correct))

        all_inputs.append(inputs.squeeze(0).cpu())
        all_labels.append(targets.squeeze(0).cpu())
        n_done += 1

        if n_done % 50 == 0:
            logger.info(f"  {n_done}/{n_puzzles} puzzles processed")

    return {
        'zh_mean': torch.stack(all_zh_mean),              # [N, steps, D]
        'per_step_accuracy': torch.tensor(all_step_acc),   # [N, steps]
        'final_accuracy': torch.tensor(all_final_acc),     # [N]
        'final_puzzle_acc': torch.tensor(all_final_puzzle), # [N]
        'inputs': torch.stack(all_inputs),
        'labels': torch.stack(all_labels),
        'n_puzzles': n_done,
        'max_steps': max_steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Directional metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_directional_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute cosine similarities, norms, and alignment metrics.

    Args:
        data: Output of collect_zh_trajectories.

    Returns dict with per-puzzle and aggregate metrics.
    """
    zh = data['zh_mean']  # [N, steps, D]
    N, S, D = zh.shape

    # Normalize for cosine computations
    zh_norm = torch.nn.functional.normalize(zh, dim=-1)  # [N, S, D]

    # 1. Cosine similarity between consecutive steps
    # cos(z_H[t], z_H[t+1]) for t=0..S-2
    cos_consecutive = (zh_norm[:, :-1, :] * zh_norm[:, 1:, :]).sum(dim=-1)  # [N, S-1]

    # 2. L2 norm of z_H at each step
    norms = zh.norm(dim=-1)  # [N, S]

    # 3. Alignment with final state: cos(z_H[t], z_H[-1])
    zh_final = zh_norm[:, -1:, :]  # [N, 1, D]
    cos_final = (zh_norm * zh_final).sum(dim=-1)  # [N, S]

    # 4. Update magnitude: ||z_H[t+1] - z_H[t]||
    deltas = zh[:, 1:, :] - zh[:, :-1, :]  # [N, S-1, D]
    delta_norms = deltas.norm(dim=-1)  # [N, S-1]

    # 5. Cosine of consecutive updates: cos(Δz_H[t], Δz_H[t+1])
    delta_norm_safe = torch.nn.functional.normalize(deltas, dim=-1)
    cos_updates = (delta_norm_safe[:, :-1, :] * delta_norm_safe[:, 1:, :]).sum(dim=-1)  # [N, S-2]

    # Split by success/failure
    final_acc = data['final_accuracy']
    puzzle_acc = data['final_puzzle_acc']
    success_mask = puzzle_acc > 0.5  # Solved puzzles
    # Also define "high accuracy" for cell-level analysis
    high_acc_mask = final_acc > 0.9

    metrics = {
        'cos_consecutive': {
            'all_mean': cos_consecutive.mean(dim=0).tolist(),
            'all_std': cos_consecutive.std(dim=0).tolist(),
        },
        'norms': {
            'all_mean': norms.mean(dim=0).tolist(),
            'all_std': norms.std(dim=0).tolist(),
        },
        'cos_final_alignment': {
            'all_mean': cos_final.mean(dim=0).tolist(),
            'all_std': cos_final.std(dim=0).tolist(),
        },
        'delta_norms': {
            'all_mean': delta_norms.mean(dim=0).tolist(),
            'all_std': delta_norms.std(dim=0).tolist(),
        },
        'cos_updates': {
            'all_mean': cos_updates.mean(dim=0).tolist(),
            'all_std': cos_updates.std(dim=0).tolist(),
        },
        'n_success': int(success_mask.sum().item()),
        'n_failure': int((~success_mask).sum().item()),
        'n_high_acc': int(high_acc_mask.sum().item()),
    }

    # Split metrics by success/failure
    if success_mask.sum() > 0:
        metrics['cos_consecutive']['success_mean'] = cos_consecutive[success_mask].mean(dim=0).tolist()
        metrics['norms']['success_mean'] = norms[success_mask].mean(dim=0).tolist()
        metrics['cos_final_alignment']['success_mean'] = cos_final[success_mask].mean(dim=0).tolist()
        metrics['delta_norms']['success_mean'] = delta_norms[success_mask].mean(dim=0).tolist()

    if (~success_mask).sum() > 0:
        metrics['cos_consecutive']['failure_mean'] = cos_consecutive[~success_mask].mean(dim=0).tolist()
        metrics['norms']['failure_mean'] = norms[~success_mask].mean(dim=0).tolist()
        metrics['cos_final_alignment']['failure_mean'] = cos_final[~success_mask].mean(dim=0).tolist()
        metrics['delta_norms']['failure_mean'] = delta_norms[~success_mask].mean(dim=0).tolist()

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# PCA + Plotting
# ═══════════════════════════════════════════════════════════════════════════

def run_pca_and_plot(data: Dict[str, Any], metrics: Dict[str, Any],
                     output_dir: str):
    """Generate PCA trajectory plots and directional metric plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib or sklearn not available. Skipping plots.")
        return

    zh = data['zh_mean'].numpy()  # [N, steps, D]
    N, S, D = zh.shape
    puzzle_acc = data['final_puzzle_acc'].numpy()
    step_acc = data['per_step_accuracy'].numpy()  # [N, S]

    success_mask = puzzle_acc > 0.5
    failure_mask = ~success_mask

    # Fit PCA on all z_H across all steps and puzzles
    zh_flat = zh.reshape(-1, D)  # [N*S, D]
    pca = PCA(n_components=3)
    zh_pca = pca.fit_transform(zh_flat).reshape(N, S, 3)  # [N, S, 3]

    var_explained = pca.explained_variance_ratio_
    logger.info(f"PCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, PC3={var_explained[2]:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.cm.viridis

    # ─── Plot 1: All trajectories, colored by step ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Successful puzzles
    ax = axes[0]
    n_success = success_mask.sum()
    max_trajectories = min(30, n_success)
    success_indices = np.where(success_mask)[0][:max_trajectories]
    for idx in success_indices:
        traj = zh_pca[idx]  # [S, 3]
        for t in range(S - 1):
            ax.plot(traj[t:t+2, 0], traj[t:t+2, 1],
                    color=cmap(t / (S - 1)), alpha=0.5, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], color='green', s=30, zorder=5, marker='o')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=30, zorder=5, marker='x')
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_title(f'Solved Puzzles (n={n_success})')

    # Panel B: Failed puzzles
    ax = axes[1]
    n_failure = failure_mask.sum()
    max_trajectories_fail = min(30, n_failure)
    failure_indices = np.where(failure_mask)[0][:max_trajectories_fail]
    for idx in failure_indices:
        traj = zh_pca[idx]
        for t in range(S - 1):
            ax.plot(traj[t:t+2, 0], traj[t:t+2, 1],
                    color=cmap(t / (S - 1)), alpha=0.5, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], color='green', s=30, zorder=5, marker='o')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=30, zorder=5, marker='x')
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_title(f'Failed Puzzles (n={n_failure})')

    # Panel C: Overlay with color distinguishing success/failure
    ax = axes[2]
    for idx in success_indices[:15]:
        traj = zh_pca[idx]
        ax.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.3, linewidth=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], color='blue', s=20, zorder=5, marker='x')
    for idx in failure_indices[:15]:
        traj = zh_pca[idx]
        ax.plot(traj[:, 0], traj[:, 1], color='orange', alpha=0.3, linewidth=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], color='orange', s=20, zorder=5, marker='x')
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_title('Overlay: Solved (blue) vs Failed (orange)')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'zh_pca_trajectories.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'zh_pca_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved PCA trajectory plots.")

    # ─── Plot 2: Directional metrics ───
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    steps = np.arange(S)

    # (A) Cosine similarity between consecutive steps
    ax = axes[0, 0]
    cos_all = np.array(metrics['cos_consecutive']['all_mean'])
    cos_std = np.array(metrics['cos_consecutive']['all_std'])
    ax.plot(steps[:-1], cos_all, 'k-o', markersize=4, label='All')
    ax.fill_between(steps[:-1], cos_all - cos_std, cos_all + cos_std, alpha=0.2, color='gray')
    if 'success_mean' in metrics['cos_consecutive']:
        ax.plot(steps[:-1], metrics['cos_consecutive']['success_mean'], 'b-s', markersize=3, label='Solved')
    if 'failure_mean' in metrics['cos_consecutive']:
        ax.plot(steps[:-1], metrics['cos_consecutive']['failure_mean'], 'r-^', markersize=3, label='Failed')
    ax.set_xlabel('Step t')
    ax.set_ylabel('cos(z_H[t], z_H[t+1])')
    ax.set_title('Consecutive Step Similarity')
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)

    # (B) Norm evolution
    ax = axes[0, 1]
    norms_all = np.array(metrics['norms']['all_mean'])
    norms_std = np.array(metrics['norms']['all_std'])
    ax.plot(steps, norms_all, 'k-o', markersize=4, label='All')
    ax.fill_between(steps, norms_all - norms_std, norms_all + norms_std, alpha=0.2, color='gray')
    if 'success_mean' in metrics['norms']:
        ax.plot(steps, metrics['norms']['success_mean'], 'b-s', markersize=3, label='Solved')
    if 'failure_mean' in metrics['norms']:
        ax.plot(steps, metrics['norms']['failure_mean'], 'r-^', markersize=3, label='Failed')
    ax.set_xlabel('Step')
    ax.set_ylabel('||z_H||')
    ax.set_title('z_H Norm Evolution')
    ax.legend(fontsize=8)

    # (C) Alignment with final state
    ax = axes[1, 0]
    cos_final = np.array(metrics['cos_final_alignment']['all_mean'])
    cos_final_std = np.array(metrics['cos_final_alignment']['all_std'])
    ax.plot(steps, cos_final, 'k-o', markersize=4, label='All')
    ax.fill_between(steps, cos_final - cos_final_std, cos_final + cos_final_std, alpha=0.2, color='gray')
    if 'success_mean' in metrics['cos_final_alignment']:
        ax.plot(steps, metrics['cos_final_alignment']['success_mean'], 'b-s', markersize=3, label='Solved')
    if 'failure_mean' in metrics['cos_final_alignment']:
        ax.plot(steps, metrics['cos_final_alignment']['failure_mean'], 'r-^', markersize=3, label='Failed')
    ax.set_xlabel('Step')
    ax.set_ylabel('cos(z_H[t], z_H[final])')
    ax.set_title('Alignment with Final State')
    ax.legend(fontsize=8)

    # (D) Update magnitude (delta norms)
    ax = axes[1, 1]
    delta_all = np.array(metrics['delta_norms']['all_mean'])
    delta_std = np.array(metrics['delta_norms']['all_std'])
    ax.plot(steps[:-1], delta_all, 'k-o', markersize=4, label='All')
    ax.fill_between(steps[:-1], delta_all - delta_std, delta_all + delta_std, alpha=0.2, color='gray')
    if 'success_mean' in metrics['delta_norms']:
        ax.plot(steps[:-1], metrics['delta_norms']['success_mean'], 'b-s', markersize=3, label='Solved')
    if 'failure_mean' in metrics['delta_norms']:
        ax.plot(steps[:-1], metrics['delta_norms']['failure_mean'], 'r-^', markersize=3, label='Failed')
    ax.set_xlabel('Step t')
    ax.set_ylabel('||z_H[t+1] - z_H[t]||')
    ax.set_title('Update Magnitude')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'zh_directional_metrics.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'zh_directional_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved directional metric plots.")

    # ─── Plot 3: Step-wise accuracy curves (for paper) ───
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mean_acc = step_acc.mean(axis=0)
    std_acc = step_acc.std(axis=0)
    ax.plot(steps, mean_acc, 'k-o', markersize=5, label='Mean cell accuracy')
    ax.fill_between(steps, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color='gray')
    if success_mask.sum() > 0:
        ax.plot(steps, step_acc[success_mask].mean(axis=0), 'b-s', markersize=4, label='Solved puzzles')
    if failure_mask.sum() > 0:
        ax.plot(steps, step_acc[failure_mask].mean(axis=0), 'r-^', markersize=4, label='Failed puzzles')
    ax.set_xlabel('ACT Step')
    ax.set_ylabel('Cell Accuracy')
    ax.set_title('HRM Cell Accuracy Across ACT Steps')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(output_dir, 'zh_accuracy_curves.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'zh_accuracy_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved accuracy curve plot.")

    # Save PCA info
    pca_info = {
        'variance_explained': var_explained.tolist(),
        'n_components': 3,
        'n_puzzles': N,
        'n_steps': S,
    }
    with open(os.path.join(output_dir, 'pca_info.json'), 'w') as f:
        json.dump(pca_info, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="z_H Trajectory Analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt (auto-detects HRM if omitted)")
    parser.add_argument("--n_puzzles", type=int, default=200,
                        help="Number of puzzles to analyze")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/plots/zh_trajectories")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 20 puzzles")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.quick:
        args.n_puzzles = 20

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Default HRM checkpoint
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            REPO_ROOT, "checkpoints", "sapientinc-sudoku-extreme", "checkpoint.pt")

    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"N puzzles: {args.n_puzzles}")

    # Load model
    model, test_loader = load_model_and_data(args.checkpoint, device)
    logger.info("Model loaded.")

    # Collect z_H trajectories
    t0 = time.time()
    logger.info("Collecting z_H trajectories...")
    data = collect_zh_trajectories(model, test_loader, device,
                                   args.n_puzzles, args.max_steps)
    logger.info(f"Collection done in {time.time() - t0:.1f}s. "
                f"z_H shape: {data['zh_mean'].shape}")

    # Compute directional metrics
    logger.info("Computing directional metrics...")
    metrics = compute_directional_metrics(data)
    logger.info(f"Solved: {metrics['n_success']}, Failed: {metrics['n_failure']}")

    # Save raw data
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, 'zh_trajectory_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save torch data (for potential reuse)
    torch_path = os.path.join(args.output_dir, 'zh_trajectories.pt')
    torch.save({
        'zh_mean': data['zh_mean'],
        'per_step_accuracy': data['per_step_accuracy'],
        'final_accuracy': data['final_accuracy'],
        'final_puzzle_acc': data['final_puzzle_acc'],
        'n_puzzles': data['n_puzzles'],
        'max_steps': data['max_steps'],
    }, torch_path)
    logger.info(f"Trajectory data saved to {torch_path}")

    # Generate plots
    logger.info("Generating plots...")
    run_pca_and_plot(data, metrics, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("z_H TRAJECTORY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Puzzles analyzed: {data['n_puzzles']}")
    print(f"Solved: {metrics['n_success']} ({100*metrics['n_success']/data['n_puzzles']:.1f}%)")
    print(f"Failed: {metrics['n_failure']} ({100*metrics['n_failure']/data['n_puzzles']:.1f}%)")
    print()
    print("Cosine similarity (consecutive steps):")
    cos_vals = metrics['cos_consecutive']['all_mean']
    for i, v in enumerate(cos_vals):
        print(f"  Step {i:2d}→{i+1:2d}: {v:.4f}")
    print()
    print("z_H norm evolution:")
    norm_vals = metrics['norms']['all_mean']
    for i, v in enumerate(norm_vals):
        print(f"  Step {i:2d}: {v:.2f}")
    print()
    print("Alignment with final state:")
    final_vals = metrics['cos_final_alignment']['all_mean']
    for i, v in enumerate(final_vals):
        print(f"  Step {i:2d}: {v:.4f}")
    print()
    print(f"Output: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
