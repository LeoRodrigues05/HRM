#!/usr/bin/env python3
"""Gap 3: Non-Linear (MLP) Probes for Constraint Information — HARDENED.

Trains 2-layer MLP probes on the same z_H (and z_L) activations and targets as
E8, then compares to a linear probe fit on the *same* split. If MLP >> linear,
constraint info has non-linear structure; if MLP ≈ linear, the linear readout
captures the story.

Hardening (parity with E8 + the maze MLP probes):
  - **Seed ensemble** (``--seeds 0,1,2,3,4``): mean ± 95% t-CI per (z, step, target).
  - **Puzzle-disjoint split** (E8's ``puzzle_disjoint_split``): no cell of a puzzle
    appears in both train and val (the old random-row split leaked).
  - **z_H AND z_L** (``--z_level both``): the old loop only probed z_H.

Architecture: 2-layer MLP (D → hidden_dim → out), ReLU.

Output
------
  results/probes/nonlinear_probes/
    sweep_results.csv        – per (z,step,target): linear vs MLP mean ± CI, delta
    comparison_summary.json  – per-target aggregate deltas
    _meta.json               – provenance

Usage
-----
    python scripts/probes/nonlinear_probes.py --n_puzzles 500 --seeds 0,1,2,3,4 \
        --z_level both --steps 0,4,8,12,15 --device cuda
    python scripts/probes/nonlinear_probes.py --quick   # 20 puzzles, fast
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
    LinearProbe,
    train_binary, train_regression, train_multiclass,
    mean_ci, puzzle_disjoint_split,
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
        opt.zero_grad(); loss.backward(); opt.step()
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
        opt.zero_grad(); loss.backward(); opt.step()
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
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(X_tr).argmax(1) == y_tr.long()).float().mean().item()
        va_acc = (model(X_va).argmax(1) == y_va.long()).float().mean().item()
    return model, tr_acc, va_acc


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gap 3: Non-Linear MLP Probes (hardened)")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--steps", type=str, default="0,4,8,12,15")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr_linear", type=float, default=5e-3)
    parser.add_argument("--lr_mlp", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Probe-ensemble seeds (each = independent puzzle-disjoint split)")
    parser.add_argument("--z_level", type=str, default="both", choices=["H", "L", "both"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/probes/nonlinear_probes")
    parser.add_argument("--quick", action="store_true", help="Quick test with 20 puzzles")
    args = parser.parse_args()

    if args.quick:
        args.n_puzzles = 20
        args.steps = "0,15"

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    logger.info(f"Device: {device}")

    steps_to_record = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()] or [args.seed]
    os.makedirs(args.output_dir, exist_ok=True)

    model, test_loader, test_meta = load_model_and_data(device)
    logger.info("Model loaded.")

    t0 = time.time()
    collected = collect_activations(
        model, test_loader, device, args.n_puzzles, steps_to_record, args.max_steps)
    features_H, features_L, label_bank, hidden_dim = collected[0], collected[1], collected[2], collected[3]
    logger.info(f"Collection took {time.time()-t0:.1f}s | seeds={seeds} (puzzle-disjoint, val_frac={args.val_frac})")

    probe_device = device
    n_steps_per_puzzle = len(steps_to_record)
    z_levels = ["H", "L"] if args.z_level == "both" else [args.z_level]

    results: List[Dict[str, Any]] = []

    for z_name in z_levels:
        features = features_H if z_name == "H" else features_L
        ordered_steps = sorted(features.keys(), key=int)
        for step_str in ordered_steps:
            feat_list = features[step_str]
            if not feat_list:
                continue
            X_all = torch.cat(feat_list, dim=0).to(probe_device)
            step_idx_within = ordered_steps.index(step_str)

            for target in ALL_TARGETS:
                lbl_list = label_bank.get(target, [])
                if not lbl_list:
                    continue
                step_labels = lbl_list[step_idx_within::n_steps_per_puzzle]
                if not step_labels:
                    continue
                y_all = torch.cat([l.view(-1) for l in step_labels], dim=0).to(probe_device)
                min_n = min(X_all.shape[0], y_all.shape[0])
                X_use, y_use = X_all[:min_n], y_all[:min_n]
                if y_use.float().std() < 1e-6:
                    continue

                if target in BINARY_TARGETS:
                    task, metric = "binary", "acc"
                elif target in REGRESSION_TARGETS:
                    task, metric = "regression", "r2"
                elif target in MULTICLASS_TARGETS:
                    task, metric = "multiclass", "acc"
                else:
                    continue

                lin_vals, mlp_vals = [], []
                for sd in seeds:
                    tr_idx, va_idx = puzzle_disjoint_split(
                        X_use.shape[0], args.val_frac, sd,
                        cells_per_puzzle=SUDOKU_CELLS, device=probe_device)
                    X_tr, y_tr = X_use[tr_idx], y_use[tr_idx]
                    X_va, y_va = X_use[va_idx], y_use[va_idx]
                    torch.manual_seed(sd)
                    if task == "binary":
                        _, _, lv = train_binary(X_tr, y_tr, X_va, y_va, epochs=args.epochs, lr=args.lr_linear)
                        _, _, mv = train_binary_mlp(X_tr, y_tr, X_va, y_va,
                                                    hidden_dim=args.mlp_hidden, epochs=args.epochs, lr=args.lr_mlp)
                    elif task == "regression":
                        _, _, lv = train_regression(X_tr, y_tr, X_va, y_va, epochs=args.epochs, lr=args.lr_linear)
                        _, _, mv = train_regression_mlp(X_tr, y_tr, X_va, y_va,
                                                        hidden_dim=args.mlp_hidden, epochs=args.epochs, lr=args.lr_mlp)
                    else:
                        _, _, lv = train_multiclass(X_tr, y_tr, X_va, y_va, n_classes=10,
                                                    epochs=args.epochs, lr=args.lr_linear)
                        _, _, mv = train_multiclass_mlp(X_tr, y_tr, X_va, y_va, n_classes=10,
                                                        hidden_dim=args.mlp_hidden, epochs=args.epochs, lr=args.lr_mlp)
                    lin_vals.append(lv); mlp_vals.append(mv)

                lm, llo, lhi, lsd = mean_ci(lin_vals)
                mm, mlo, mhi, msd = mean_ci(mlp_vals)
                delta = mm - lm
                results.append({
                    "z_level": z_name, "step": int(step_str), "target": target,
                    "task": task, "metric": metric,
                    "linear_val": round(lm, 5), "linear_ci_lower": round(llo, 5),
                    "linear_ci_upper": round(lhi, 5),
                    "mlp_val": round(mm, 5), "mlp_ci_lower": round(mlo, 5),
                    "mlp_ci_upper": round(mhi, 5),
                    "delta_val": round(delta, 5),
                    "n_seeds": len(seeds),
                    "linear_per_seed": ";".join(f"{v:.5f}" for v in lin_vals),
                    "mlp_per_seed": ";".join(f"{v:.5f}" for v in mlp_vals),
                })
                logger.info(f"  z_{z_name} step {step_str:>2s} {target:26s} "
                            f"lin={lm:.4f} MLP={mm:.4f} Δ={delta:+.4f} ({len(seeds)} seeds)")

    # CSV
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        logger.info(f"Saved {len(results)} rows to {csv_path}")

    # Per-target aggregate
    summary: Dict[str, Any] = {}
    for r in results:
        info = summary.setdefault(r["target"], {"task": r["task"], "metric": r["metric"], "by_z_step": {}})
        info["by_z_step"][f"z{r['z_level']}_s{r['step']}"] = {
            "linear": r["linear_val"], "mlp": r["mlp_val"], "delta": r["delta_val"]}
    for tgt, info in summary.items():
        deltas = [v["delta"] for v in info["by_z_step"].values()]
        info["mean_delta"] = round(float(np.mean(deltas)), 5) if deltas else 0.0
        info["max_abs_delta"] = round(float(max(deltas, key=abs)), 5) if deltas else 0.0
    summary = dict(sorted(summary.items(), key=lambda x: abs(x[1]["mean_delta"]), reverse=True))
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "nonlinear_probes", {
            "n_puzzles": args.n_puzzles, "steps": args.steps, "seeds": seeds,
            "z_level": args.z_level, "val_frac": args.val_frac, "epochs": args.epochs,
            "mlp_hidden": args.mlp_hidden, "split": "puzzle_disjoint",
        }, repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"Could not write _meta.json: {e}")

    all_deltas = [r["delta_val"] for r in results]
    if all_deltas:
        md = float(np.mean(all_deltas))
        print("\n" + "=" * 80)
        print(f"Mean Δ(MLP−linear) over {len(all_deltas)} probes = {md:+.4f}  "
              f"(max |Δ| = {max(all_deltas, key=abs):+.4f})")
        print("Interpretation:",
              "MLP ≈ Linear → constraint info is linearly encoded."
              if abs(md) < 0.02 else "MLP > Linear → non-linear structure present.")
    logger.info(f"All results in {args.output_dir}")


if __name__ == "__main__":
    main()
