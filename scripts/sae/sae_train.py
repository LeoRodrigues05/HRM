#!/usr/bin/env python3
"""SAE E10: Train a Sparse Autoencoder on collected z_H activations.

Loads pre-collected activations and trains an SAE with configurable
hyperparameters. Supports filtering by ACT step, dead feature
reinitialization, and detailed logging.

Output
------
  results/sae_study/sae_d{dict_size}_l1{l1_coeff}.pt       — trained SAE
  results/sae_study/sae_d{dict_size}_l1{l1_coeff}_log.json  — training log
  results/sae_study/sae_d{dict_size}_l1{l1_coeff}_features.json — feature stats

Usage
-----
    python scripts/sae/sae_train.py
    python scripts/sae/sae_train.py --dict_size 4096 --l1_coeff 0.003
    python scripts/sae/sae_train.py --steps_filter "8,12,15" --epochs 100
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.sae import SparseAutoencoder, TopKSparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


def load_activations(
    activations_path: str,
    steps_filter: Optional[List[int]] = None,
    activation_level: str = "z_H",
) -> torch.Tensor:
    """Load and flatten activations for SAE training.

    Args:
        activations_path: Path to activations_zH.pt
        steps_filter: If set, only use activations from these steps
        activation_level: 'z_H' or 'z_L'

    Returns:
        flat_activations: [N_total, input_dim] tensor
    """
    logger.info(f"Loading activations from {activations_path}...")
    data = torch.load(activations_path, map_location="cpu", weights_only=False)

    activations = data[activation_level]  # [N_puzzles, steps, 81, D]
    logger.info(f"  Raw shape: {activations.shape}")

    if steps_filter is not None:
        logger.info(f"  Filtering to steps: {steps_filter}")
        activations = activations[:, steps_filter, :, :]

    # Flatten to [N_total, D]
    n_puzzles, n_steps, n_cells, dim = activations.shape
    flat = activations.reshape(n_puzzles * n_steps * n_cells, dim)
    logger.info(f"  Flattened shape: {flat.shape} ({flat.shape[0]:,} samples × {dim} dims)")
    return flat


def train_sae(
    activations: torch.Tensor,
    dict_size: int = 2048,
    l1_coeff: float = 0.01,
    lr: float = 3e-4,
    batch_size: int = 4096,
    epochs: int = 50,
    device: torch.device = torch.device("cpu"),
    reinit_interval: int = 5,
    dead_threshold: int = 1000,
    log_interval: int = 100,
    save_interval: int = 5,
    output_dir: str = "results/sae_study",
    activation: str = "relu",
    k: int = 64,
) -> Dict[str, Any]:
    """Train an SAE on flattened activations.

    Args:
        activation: 'relu' for L1-penalized SAE, 'topk' for TopK SAE.
        k: Number of top activations to keep (only used when activation='topk').

    Returns dict with trained model, training log, and feature statistics.
    """
    input_dim = activations.shape[1]
    n_samples = activations.shape[0]

    logger.info(f"Training SAE: input_dim={input_dim}, dict_size={dict_size}, "
                f"activation={activation}, l1_coeff={l1_coeff}, k={k}, "
                f"lr={lr}, batch_size={batch_size}, epochs={epochs}")

    # Create SAE
    if activation == "topk":
        sae = TopKSparseAutoencoder(input_dim=input_dim, dict_size=dict_size, k=k)
    else:
        sae = SparseAutoencoder(input_dim=input_dim, dict_size=dict_size, l1_coeff=l1_coeff)
    sae = sae.to(device)

    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Move activations to device
    activations = activations.to(device)

    training_log = []
    global_step = 0
    best_recon_loss = float('inf')

    os.makedirs(output_dir, exist_ok=True)
    tag = f"d{dict_size}_topk{k}" if activation == "topk" else f"d{dict_size}_l1{l1_coeff}"

    t0 = time.time()

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        activations_shuffled = activations[perm]

        epoch_recon_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        sae.train()
        for start in range(0, n_samples, batch_size):
            batch = activations_shuffled[start:start + batch_size]
            if batch.shape[0] < 2:
                continue

            x_hat, h, loss_dict = sae(batch)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon_loss += loss_dict['reconstruction_loss'].item()
            epoch_l1_loss += loss_dict['l1_loss'].item()
            epoch_total_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                stats = sae.get_feature_stats(dead_threshold)
                entry = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                    'l1_loss': loss_dict['l1_loss'].item(),
                    'total_loss': loss.item(),
                    **stats,
                }
                training_log.append(entry)

                logger.info(
                    f"  step {global_step:6d} | epoch {epoch:3d} | "
                    f"recon={loss_dict['reconstruction_loss'].item():.6f} | "
                    f"l1={loss_dict['l1_loss'].item():.6f} | "
                    f"alive={stats['alive_count']}/{dict_size} | "
                    f"L0={stats['L0']:.1f} | "
                    f"sparsity={stats['mean_sparsity']:.4f}"
                )

        # End-of-epoch stats
        avg_recon = epoch_recon_loss / max(n_batches, 1)
        avg_l1 = epoch_l1_loss / max(n_batches, 1)
        avg_total = epoch_total_loss / max(n_batches, 1)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | avg_recon={avg_recon:.6f} | "
            f"avg_l1={avg_l1:.6f} | avg_total={avg_total:.6f}"
        )

        if avg_recon < best_recon_loss:
            best_recon_loss = avg_recon

        # Dead feature reinitialization
        if (epoch + 1) % reinit_interval == 0 and epoch < epochs - 1:
            sae.eval()
            # Use a random sample of data for reinit
            reinit_sample_size = min(batch_size * 4, n_samples)
            reinit_indices = torch.randperm(n_samples, device=device)[:reinit_sample_size]
            reinit_data = activations[reinit_indices]
            n_reinit = sae.reinitialize_dead_features(reinit_data, dead_threshold)
            if n_reinit > 0:
                logger.info(f"  Reinitialized {n_reinit} dead features at epoch {epoch}")
            sae.reset_stats()

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"sae_{tag}_epoch{epoch+1}.pt")
            torch.save({
                'model_state_dict': sae.state_dict(),
                'config': {
                    'input_dim': input_dim,
                    'dict_size': dict_size,
                    'l1_coeff': l1_coeff,
                    'activation': activation,
                    'k': k,
                },
                'epoch': epoch,
                'global_step': global_step,
            }, ckpt_path)

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s. Best recon loss: {best_recon_loss:.6f}")

    # Final evaluation
    sae.eval()
    sae.reset_stats()
    with torch.no_grad():
        # Run through full data in batches to get final stats
        for start in range(0, n_samples, batch_size):
            batch = activations[start:start + batch_size]
            sae(batch)

    final_stats = sae.get_feature_stats(dead_threshold)
    logger.info(f"Final stats: {final_stats}")

    # Save final model
    final_path = os.path.join(output_dir, f"sae_{tag}.pt")
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': input_dim,
            'dict_size': dict_size,
            'l1_coeff': l1_coeff,
            'activation': activation,
            'k': k,
        },
        'epoch': epochs,
        'global_step': global_step,
        'final_stats': final_stats,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save training log
    log_path = os.path.join(output_dir, f"sae_{tag}_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Saved training log to {log_path}")

    # Save feature statistics
    features_path = os.path.join(output_dir, f"sae_{tag}_features.json")
    with open(features_path, 'w') as f:
        json.dump({
            'final_stats': final_stats,
            'training_time_sec': elapsed,
            'n_samples': n_samples,
            'best_recon_loss': best_recon_loss,
            'config': {
                'input_dim': input_dim,
                'dict_size': dict_size,
                'l1_coeff': l1_coeff,
                'activation': activation,
                'k': k,
                'lr': lr,
                'batch_size': batch_size,
                'epochs': epochs,
            },
        }, f, indent=2)
    logger.info(f"Saved feature stats to {features_path}")

    return {
        'sae': sae,
        'training_log': training_log,
        'final_stats': final_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="SAE E10: Train Sparse Autoencoder")
    parser.add_argument("--activations_path", type=str,
                        default="results/sae_study/activations_zH.pt",
                        help="Path to collected activations")
    parser.add_argument("--activation_level", type=str, default="z_H",
                        choices=["z_H", "z_L"],
                        help="Which activation level to train on")
    parser.add_argument("--dict_size", type=int, default=2048,
                        help="Dictionary size (default 2048 = 4× expansion)")
    parser.add_argument("--l1_coeff", type=float, default=0.01,
                        help="L1 sparsity penalty coefficient (ignored for topk)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "topk"],
                        help="Activation mode: 'relu' (L1 penalty) or 'topk'")
    parser.add_argument("--k", type=int, default=64,
                        help="Number of top activations to keep (only for topk)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_filter", type=str, default=None,
                        help="Comma-separated ACT steps to train on (e.g. '8,12,15')")
    parser.add_argument("--reinit_interval", type=int, default=5,
                        help="Reinitialize dead features every N epochs")
    parser.add_argument("--dead_threshold", type=int, default=1000,
                        help="Feature is dead if not activated in N consecutive batches")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/sae_study")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Parse steps filter
    steps_filter = None
    if args.steps_filter:
        steps_filter = [int(s) for s in args.steps_filter.split(",")]

    # Load activations
    activations = load_activations(
        args.activations_path,
        steps_filter=steps_filter,
        activation_level=args.activation_level,
    )

    # Train
    train_sae(
        activations=activations,
        dict_size=args.dict_size,
        l1_coeff=args.l1_coeff,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        reinit_interval=args.reinit_interval,
        dead_threshold=args.dead_threshold,
        output_dir=args.output_dir,
        activation=args.activation,
        k=args.k,
    )


if __name__ == "__main__":
    main()
