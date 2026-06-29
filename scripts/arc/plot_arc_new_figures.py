#!/usr/bin/env python3
"""Extra ARC figures from data this project newly retrieved (Path-A adaptation run).

Complements scripts/arc/plot_arc_figures.py (decodability / MLP-linear / directed
ablation / value-by-step) with figures the standard plotter doesn't cover:

  fig_arc_accuracy_summary      single-shot metrics (token/colour/exact/shape/...)
  fig_arc_voting_passk          TTA-voting pass@K (the authors' metric scale)
  fig_arc_singleshot_vs_voting  exact_solved (single-shot) vs pass@1/pass@2 (voting)
  fig_arc_convergence_trend     token/colour/exact across adaptation checkpoints

All inputs are optional — missing files are skipped with a notice, so this is safe
to run whether or not the voting eval has finished.

Usage:
  python scripts/arc/plot_arc_new_figures.py \
      --diag_dir results/arc/diagnostics \
      --output_dir results/reports/arc_figures_step7391
"""
from __future__ import annotations
import os, json, argparse, glob
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot_arc_new] wrote {output_dir}/{name}.png|pdf")


def plot_accuracy_summary(acc: dict, output_dir: str):
    keys = ["token_acc", "colour_cell_acc", "colour_iou", "eos_acc",
            "exact_solved", "shape_correct", "height_correct", "width_correct"]
    keys = [k for k in keys if k in acc]
    vals = [acc[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(keys)), vals, color="#0074D9")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"ARC-2 adapted checkpoint — single-shot metrics (n={acc.get('n_puzzles','?')})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, "fig_arc_accuracy_summary")


def plot_voting_passk(vote: dict, output_dir: str):
    ks = sorted((int(k.split("@")[1]) for k in vote if k.startswith("pass@")))
    vals = [vote[f"pass@{k}"] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(k) for k in ks], vals, color="#2ECC40")
    ax.set_xlabel("K (number of allowed guesses)")
    ax.set_ylabel("fraction of puzzles solved")
    ax.set_title(f"ARC-2 TTA-voting pass@K (n={vote.get('n_puzzles','?')})")
    ax.set_ylim(0, max(0.05, max(vals) * 1.25) if vals else 1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=9)
    ax.axhline(0.403, ls="--", c="grey", lw=1)
    ax.text(len(ks) - 1, 0.41, "authors' pass@2 ≈ 40.3%", ha="right", fontsize=8, color="grey")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, "fig_arc_voting_passk")


def plot_singleshot_vs_voting(acc: dict, vote: dict, output_dir: str):
    labels, vals, colors = [], [], []
    if acc and "exact_solved" in acc:
        labels.append("single-shot\nexact_solved"); vals.append(acc["exact_solved"]); colors.append("#0074D9")
    if vote and "pass@1" in vote:
        labels.append("voting\npass@1"); vals.append(vote["pass@1"]); colors.append("#2ECC40")
    if vote and "pass@2" in vote:
        labels.append("voting\npass@2"); vals.append(vote["pass@2"]); colors.append("#2ECC40")
    if not vals:
        print("[plot_arc_new] singleshot_vs_voting: no data — skipped"); return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("fraction of puzzles solved")
    ax.set_title("ARC-2: single-shot vs TTA-voting (same checkpoint)")
    ax.set_ylim(0, max(0.05, max(vals) * 1.3))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, "fig_arc_singleshot_vs_voting")


def plot_convergence_trend(diag_dir: str, output_dir: str):
    rows = []
    for d in sorted(glob.glob(os.path.join(diag_dir, "trend_*"))):
        step = int(os.path.basename(d).split("_")[1])
        j = _load(os.path.join(d, "arc_accuracy.json"))
        if j:
            rows.append((step, j))
    if len(rows) < 2:
        print("[plot_arc_new] convergence_trend: <2 checkpoints — skipped"); return
    rows.sort()
    steps = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, c in [("token_acc", "#0074D9"), ("colour_cell_acc", "#FF851B"), ("exact_solved", "#FF4136")]:
        ax.plot(steps, [r[1].get(key, np.nan) for r in rows], "o-", color=c, label=key)
    ax.set_xlabel("adaptation step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("ARC-2 puzzle_emb adaptation — convergence (frozen core)")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.5, "token/colour plateau early;\nexact stays low (frozen-core ceiling)",
            transform=ax.transAxes, ha="center", fontsize=8, color="grey")
    _save(fig, output_dir, "fig_arc_convergence_trend")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_dir", default="results/arc/diagnostics")
    ap.add_argument("--output_dir", default="results/reports/arc_figures_new")
    args = ap.parse_args()

    acc = _load(os.path.join(args.diag_dir, "arc_accuracy.json"))
    vote = _load(os.path.join(args.diag_dir, "arc_voting_accuracy.json"))

    if acc:
        plot_accuracy_summary(acc, args.output_dir)
    else:
        print("[plot_arc_new] arc_accuracy.json missing — skipped accuracy summary")
    if vote:
        plot_voting_passk(vote, args.output_dir)
    else:
        print("[plot_arc_new] arc_voting_accuracy.json missing — skipped voting figures")
    if acc or vote:
        plot_singleshot_vs_voting(acc or {}, vote or {}, args.output_dir)
    plot_convergence_trend(args.diag_dir, args.output_dir)
    print(f"[plot_arc_new] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
