#!/usr/bin/env python3
"""Gap 3: Non-Linear (MLP) Probes for Constraint Information.

Trains 2-layer MLP probes on the same z_H (and z_L) activations and targets
as E8, then compares accuracy to the existing linear probes. If MLP >> linear,
constraint info has non-linear structure. If MLP ≈ linear, the linear readout
captures the full story.

Architecture: 2-layer MLP (D → hidden_dim → 1 or n_classes), ReLU.

Output
------
  results/probes/nonlinear_probes/
    sweep_results.csv        – full accuracy table (linear + MLP side by side)
    comparison_summary.json  – per-target linear-vs-MLP accuracy deltas
    probe_weights.pt         – MLP weights for potential directed ablation

Usage
-----
    python scripts/probes/nonlinear_probes.py --n_puzzles 500 --device cuda
    python scripts/probes/nonlinear_probes.py --n_puzzles 20 --quick  # test
"""

import os
import sys
import json
import csv
import argparse
import time
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.probes.e8_constraint_probes import (
    load_model_and_data,
    collect_activations,
    derive_per_cell_labels,
    LinearProbe,
    train_binary, train_regression, train_multiclass,
    BINARY_TARGETS, REGRESSION_TARGETS, MULTICLASS_TARGETS, ALL_TARGETS,
    SUDOKU_CELLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MLP Probe
# ═══════════════════════════════════════════════════════════════════════════

class MLPProbe(nn.Module):
    """2-layer MLP probe: input_dim → hidden_dim → output_dim."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_binary_mlp(X_tr, y_tr, X_va, y_va, hidden_dim=256, epochs=50, lr=1e-3):
    device = X_tr.device
    model = MLPProbe(X_tr.shape[1], 1, hidden_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        logits = model(X_tr)
        loss = crit(logits, y_tr.float().view(-1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_acc = ((torch.sigmoid(model(X_tr)) > 0.5).long().view(-1) == y_tr.long()).float().mean().item()
        va_acc = ((torch.sigmoid(model(X_va)) > 0.5).long().view(-1) == y_va.long()).float().mean().item()
    return model, tr_acc, va_acc


def train_regression_mlp(X_tr, y_tr, X_va, y_va, hidden_dim=256, epochs=80, lr=1e-3):
    device = X_tr.device
    model = MLPProbe(X_tr.shape[1], 1, hidden_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        pred = model(X_tr)
        loss = crit(pred, y_tr.float().view(-1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    def _r2(X, y):
        with torch.no_grad():
            p = model(X).view(-1)
            ss_res = ((y.float() - p) ** 2).sum()
            ss_tot = ((y.float() - y.float().mean()) ** 2).sum().clamp(min=1e-8)
            return (1 - ss_res / ss_tot).item()
    return model, _r2(X_tr, y_tr), _r2(X_va, y_va)


def train_multiclass_mlp(X_tr, y_tr, X_va, y_va, n_classes=10, hidden_dim=256, epochs=50, lr=1e-3):
    device = X_tr.device
    model = MLPProbe(X_tr.shape[1], n_classes, hidden_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        logits = model(X_tr)
        loss = crit(logits, y_tr.long())
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_acc = (model(X_tr).argmax(1) == y_tr.long()).float().mean().item()
        va_acc = (model(X_va).argmax(1) == y_va.long()).float().mean().item()
    return model, tr_acc, va_acc


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gap 3: Non-Linear MLP Probes")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--steps", type=str, default="0,4,8,12,15")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr_linear", type=float, default=5e-3)
    parser.add_argument("--lr_mlp", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden", type=int, default=256,
                        help="Hidden dim for 2-layer MLP probe")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/probes/nonlinear_probes")
    parser.add_argument("--quick", action="store_true", help="Quick test with 20 puzzles")
    args = parser.parse_args()

    if args.quick:
        args.n_puzzles = 20
        args.steps = "0,15"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    steps_to_record = [int(s) for s in args.steps.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model and collect activations ────────────────────────────
    model, test_loader, test_meta = load_model_and_data(device)
    logger.info("Model loaded.")

    t0 = time.time()
    features_H, features_L, label_bank, hidden_dim = collect_activations(
        model, test_loader, device, args.n_puzzles, steps_to_record, args.max_steps
    )
    logger.info(f"Collection took {time.time()-t0:.1f}s")

    # ── Train both linear and MLP probes ──────────────────────────────
    probe_device = device
    torch.manual_seed(args.seed)
    n_steps_per_puzzle = len(steps_to_record)

    results = []
    mlp_weights = {}

    for step_str in sorted(features_H.keys(), key=int):
        feat_list = features_H[step_str]
        if not feat_list:
            continue

        X_all = torch.cat(feat_list, dim=0).to(probe_device)
        step_idx_within = sorted(features_H.keys(), key=int).index(step_str)

        for target in ALL_TARGETS:
            lbl_list = label_bank.get(target, [])
            if not lbl_list:
                continue

            step_labels = lbl_list[step_idx_within::n_steps_per_puzzle]
            if not step_labels:
                continue
            y_all = torch.cat([l.view(-1) for l in step_labels], dim=0).to(probe_device)

            # Align
            min_n = min(X_all.shape[0], y_all.shape[0])
            X_use, y_use = X_all[:min_n], y_all[:min_n]

            if y_use.float().std() < 1e-6:
                continue

            # Train/val split (same for both)
            n = X_use.shape[0]
            perm = torch.randperm(n, device=probe_device)
            n_val = max(1, int(n * args.val_frac))
            X_va, y_va = X_use[perm[:n_val]], y_use[perm[:n_val]]
            X_tr, y_tr = X_use[perm[n_val:]], y_use[perm[n_val:]]

            if target in BINARY_TARGETS:
                task = "binary"
                metric = "acc"
                # Linear
                _, lin_tr, lin_va = train_binary(X_tr, y_tr, X_va, y_va,
                                                  epochs=args.epochs, lr=args.lr_linear)
                # MLP
                mlp_model, mlp_tr, mlp_va = train_binary_mlp(
                    X_tr, y_tr, X_va, y_va,
                    hidden_dim=args.mlp_hidden, epochs=args.epochs, lr=args.lr_mlp)

            elif target in REGRESSION_TARGETS:
                task = "regression"
                metric = "r2"
                _, lin_tr, lin_va = train_regression(X_tr, y_tr, X_va, y_va,
                                                      epochs=args.epochs, lr=args.lr_linear)
                mlp_model, mlp_tr, mlp_va = train_regression_mlp(
                    X_tr, y_tr, X_va, y_va,
                    hidden_dim=args.mlp_hidden, epochs=args.epochs, lr=args.lr_mlp)

            elif target in MULTICLASS_TARGETS:
                task = "multiclass"
                metric = "acc"
                n_classes = 10
                _, lin_tr, lin_va = train_multiclass(X_tr, y_tr, X_va, y_va,
                                                      n_classes=n_classes,
                                                      epochs=args.epochs, lr=args.lr_linear)
                mlp_model, mlp_tr, mlp_va = train_multiclass_mlp(
                    X_tr, y_tr, X_va, y_va,
                    n_classes=n_classes, hidden_dim=args.mlp_hidden,
                    epochs=args.epochs, lr=args.lr_mlp)
            else:
                continue

            delta = mlp_va - lin_va

            results.append({
                "step": int(step_str),
                "target": target,
                "task": task,
                "metric": metric,
                "linear_train": round(lin_tr, 5),
                "linear_val": round(lin_va, 5),
                "mlp_train": round(mlp_tr, 5),
                "mlp_val": round(mlp_va, 5),
                "delta_val": round(delta, 5),
                "n_train": X_tr.shape[0],
                "n_val": X_va.shape[0],
                "in_dim": int(X_tr.shape[1]),
                "mlp_hidden": args.mlp_hidden,
            })

            # Save MLP weights
            key = f"z_H_step{step_str}_{target}"
            mlp_weights[key] = {
                "state_dict": mlp_model.state_dict(),
                "task": task,
                "metric": metric,
                "linear_val": lin_va,
                "mlp_val": mlp_va,
                "delta": delta,
                "step": int(step_str),
                "target": target,
                "hidden_dim": args.mlp_hidden,
            }

            logger.info(
                f"  step {step_str:>2s} {target:28s} "
                f"linear={lin_va:.4f}  MLP={mlp_va:.4f}  Δ={delta:+.4f}"
            )

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        logger.info(f"Saved {len(results)} rows to {csv_path}")

    # ── Save MLP weights ──────────────────────────────────────────────
    torch.save(mlp_weights, os.path.join(args.output_dir, "probe_weights.pt"))

    # ── Comparison summary ────────────────────────────────────────────
    summary = {}
    for r in results:
        tgt = r["target"]
        if tgt not in summary:
            summary[tgt] = {"steps": {}, "task": r["task"], "metric": r["metric"]}
        summary[tgt]["steps"][r["step"]] = {
            "linear": r["linear_val"],
            "mlp": r["mlp_val"],
            "delta": r["delta_val"],
        }

    # Aggregate per-target
    for tgt, info in summary.items():
        deltas = [v["delta"] for v in info["steps"].values()]
        info["mean_delta"] = round(float(np.mean(deltas)), 5) if deltas else 0.0
        info["max_delta"] = round(float(np.max(deltas)), 5) if deltas else 0.0

    # Sort by absolute mean delta
    summary_sorted = dict(sorted(summary.items(),
                                  key=lambda x: abs(x[1]["mean_delta"]),
                                  reverse=True))

    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary_sorted, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("NON-LINEAR PROBES — LINEAR vs MLP COMPARISON")
    print("=" * 90)
    print(f"{'target':28s}  {'task':12s}  {'linear':>8s}  {'MLP':>8s}  {'Δ(MLP-Lin)':>10s}")
    print("-" * 90)

    for tgt, info in summary_sorted.items():
        # Use step 15 if available, else last step
        best_step = max(info["steps"].keys())
        vals = info["steps"][best_step]
        print(f"{tgt:28s}  {info['task']:12s}  {vals['linear']:>8.4f}  "
              f"{vals['mlp']:>8.4f}  {vals['delta']:>+10.4f}")

    print("-" * 90)
    all_deltas = [r["delta_val"] for r in results]
    print(f"{'Mean Δ across all':28s}  {'':12s}  {'':>8s}  {'':>8s}  {np.mean(all_deltas):>+10.4f}")
    print(f"{'Max |Δ|':28s}  {'':12s}  {'':>8s}  {'':>8s}  {max(all_deltas, key=abs):>+10.4f}")

    interpretation = "MLP ≈ Linear" if abs(np.mean(all_deltas)) < 0.02 else "MLP > Linear"
    print(f"\nInterpretation: {interpretation}")
    if abs(np.mean(all_deltas)) < 0.02:
        print("  → Constraint info is encoded linearly in z_H. No hidden non-linear structure.")
    else:
        print("  → Constraint info has significant non-linear structure in z_H.")

    logger.info(f"All results in {args.output_dir}")


if __name__ == "__main__":
    main()
