#!/usr/bin/env python3
"""SAE E10: Hyperparameter sweep over (dict_size, l1_coeff).

Trains SAEs across a grid of configurations, finds the Goldilocks zone
where reconstruction is good AND features are sparse.

Output
------
  results/sae_study/sweep_results.csv

Usage
-----
    python scripts/sae/sae_sweep.py
    python scripts/sae/sae_sweep.py --dict_sizes "1024,2048,4096" --l1_coeffs "0.001,0.003,0.01,0.03,0.1"
"""

import os
import sys
import csv
import argparse
import logging
import time
from typing import Any, Dict, List

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.sae.sae_train import load_activations, train_sae

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SAE E10: Hyperparameter sweep")
    parser.add_argument("--activations_path", type=str,
                        default="results/sae_study/activations_zH.pt")
    parser.add_argument("--activation_level", type=str, default="z_H",
                        choices=["z_H", "z_L"])
    parser.add_argument("--dict_sizes", type=str, default="1024,2048,4096",
                        help="Comma-separated dictionary sizes to sweep")
    parser.add_argument("--l1_coeffs", type=str, default="0.001,0.003,0.01,0.03,0.1",
                        help="Comma-separated L1 coefficients to sweep")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_filter", type=str, default=None)
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

    # Parse sweep grid
    dict_sizes = [int(x) for x in args.dict_sizes.split(",")]
    l1_coeffs = [float(x) for x in args.l1_coeffs.split(",")]
    steps_filter = [int(s) for s in args.steps_filter.split(",")] if args.steps_filter else None

    logger.info(f"Sweep grid: dict_sizes={dict_sizes} × l1_coeffs={l1_coeffs}")
    logger.info(f"Total configs: {len(dict_sizes) * len(l1_coeffs)}")

    # Load activations once
    activations = load_activations(
        args.activations_path,
        steps_filter=steps_filter,
        activation_level=args.activation_level,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    sweep_results = []

    for dict_size in dict_sizes:
        for l1_coeff in l1_coeffs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: dict_size={dict_size}, l1_coeff={l1_coeff}")
            logger.info(f"{'='*60}")

            torch.manual_seed(args.seed)

            t0 = time.time()
            result = train_sae(
                activations=activations,
                dict_size=dict_size,
                l1_coeff=l1_coeff,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=device,
                output_dir=args.output_dir,
            )
            elapsed = time.time() - t0

            stats = result['final_stats']
            row = {
                'dict_size': dict_size,
                'l1_coeff': l1_coeff,
                'final_recon_loss': result['training_log'][-1]['reconstruction_loss'] if result['training_log'] else float('nan'),
                'final_l1_loss': result['training_log'][-1]['l1_loss'] if result['training_log'] else float('nan'),
                'alive_count': stats['alive_count'],
                'alive_frac': stats['alive_frac'],
                'dead_count': stats['dead_count'],
                'mean_sparsity': stats['mean_sparsity'],
                'L0': stats['L0'],
                'mean_activation': stats['mean_activation'],
                'training_time_sec': elapsed,
            }
            sweep_results.append(row)
            logger.info(f"Result: {row}")

    # Save sweep results CSV
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_results)
    logger.info(f"\nSaved sweep results to {csv_path}")

    # Print summary table
    logger.info("\n" + "="*90)
    logger.info("SWEEP SUMMARY")
    logger.info("="*90)
    header = f"{'dict_size':>10} {'l1_coeff':>10} {'recon_loss':>12} {'alive':>8} {'L0':>8} {'sparsity':>10}"
    logger.info(header)
    logger.info("-" * 90)
    for row in sweep_results:
        line = (f"{row['dict_size']:>10} {row['l1_coeff']:>10.4f} "
                f"{row['final_recon_loss']:>12.6f} "
                f"{row['alive_count']:>8} {row['L0']:>8.1f} "
                f"{row['mean_sparsity']:>10.4f}")
        logger.info(line)

    # Identify Pareto frontier (non-dominated configs on recon_loss vs L0)
    # A config is Pareto-optimal if no other config is strictly better on
    # BOTH reconstruction loss AND sparsity (lower L0).
    logger.info("\n" + "-"*90)
    logger.info("PARETO FRONTIER (recon_loss vs L0, lower is better for both):")

    def _is_dominated(row, others):
        for o in others:
            if (o['final_recon_loss'] <= row['final_recon_loss'] and
                o['L0'] <= row['L0'] and
                (o['final_recon_loss'] < row['final_recon_loss'] or
                 o['L0'] < row['L0'])):
                return True
        return False

    pareto = [r for r in sweep_results if not _is_dominated(r, sweep_results)]
    pareto.sort(key=lambda r: r['final_recon_loss'])

    for i, row in enumerate(pareto):
        logger.info(f"  Pareto #{i+1}: dict={row['dict_size']}, l1={row['l1_coeff']}, "
                    f"recon={row['final_recon_loss']:.6f}, L0={row['L0']:.1f}, "
                    f"alive={row['alive_count']}")

    if not pareto:
        logger.info("  (no results)")


if __name__ == "__main__":
    main()
