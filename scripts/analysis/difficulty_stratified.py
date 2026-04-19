#!/usr/bin/env python3
"""Gap 5: Difficulty-Stratified Analysis.

Stratifies all experiments by puzzle difficulty and reports separate metrics
for easy, medium, and hard puzzles. Difficulty is defined by baseline (step-0)
cell accuracy:
  - Easy:   step-0 cell acc >= 0.75
  - Medium: 0.65 <= step-0 cell acc < 0.75
  - Hard:   step-0 cell acc < 0.65

Loads per-puzzle data from existing experiment outputs and re-computes
metrics per difficulty stratum.

Output
------
  results/metrics/difficulty_stratified/
    stratified_summary.json  – per-stratum metrics for all experiments
    stratified_plots.pdf     – difficulty-stratified visualizations

Usage
-----
    python scripts/analysis/difficulty_stratified.py --device cuda --n_puzzles 500
    python scripts/analysis/difficulty_stratified.py --quick  # N=50 test
"""

import os
import sys
import json
import argparse
import time
import logging
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.core.activation_ablation import ActivationAblator, ActivationCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_CELLS = 81

# Difficulty thresholds based on step-0 cell accuracy
DIFFICULTY_BINS = {
    "easy":   (0.75, 1.01),   # >= 75% cells correct at step 0
    "medium": (0.65, 0.75),   # 65-75%
    "hard":   (0.00, 0.65),   # < 65%
}


def load_model_and_data(device):
    """Load HRM model and test data (reuses pattern from e8_constraint_probes)."""
    import yaml
    from pretrain import PretrainConfig, create_dataloader
    from utils.functions import load_model_class
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "sapientinc-sudoku-extreme")
    config_path = os.path.join(ckpt_dir, "all_config.yaml")
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

    ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"),
                       map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(k.startswith("_orig_mod.") for k in ck):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(k.startswith("_orig_mod.") for k in mk):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        from scripts.core.activation_ablation import _patch_attention_for_cpu
        _patch_attention_for_cpu(model_full)

    m = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, "model"):
        m = m.model

    return m, test_loader


def classify_difficulty(step0_cell_acc: float) -> str:
    """Classify puzzle difficulty based on step-0 cell accuracy."""
    for name, (lo, hi) in DIFFICULTY_BINS.items():
        if lo <= step0_cell_acc < hi:
            return name
    return "hard"


def collect_per_puzzle_data(model, test_loader, device, n_puzzles, max_steps=16):
    """Run inference on n_puzzles, return per-puzzle per-step metrics + difficulty."""
    ablator = ActivationAblator(model, device=device)
    puzzles = []

    for i, data in enumerate(test_loader):
        if i >= n_puzzles:
            break

        if isinstance(data, (tuple, list)):
            batch = data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        else:
            batch = data
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)

        targets = batch["labels"][:, -SUDOKU_CELLS:]
        inputs = batch["inputs"][:, -SUDOKU_CELLS:]

        step_metrics = {}
        for step in range(max_steps):
            if step not in cache:
                continue
            ac = cache[step]
            preds = ac.preds[:, -SUDOKU_CELLS:]

            correct = (preds == targets).float()
            unknown_mask = (inputs.squeeze(0) == 1)  # blank token = 1
            n_unknown = unknown_mask.sum().item()

            cell_acc = correct.mean().item()
            puzzle_acc = 1.0 if correct.all() else 0.0
            unk_acc = correct[0][unknown_mask].mean().item() if n_unknown > 0 else 1.0
            hamming = (preds != targets).sum().item()

            # Violations
            grid = preds.view(9, 9)
            row_viol = col_viol = box_viol = 0
            for r in range(9):
                for c in range(9):
                    d = grid[r, c].item()
                    if d <= 0:
                        continue
                    if (grid[r, :] == d).sum().item() > 1:
                        row_viol += 1
                    if (grid[:, c] == d).sum().item() > 1:
                        col_viol += 1
                    br, bc = (r // 3) * 3, (c // 3) * 3
                    if (grid[br:br+3, bc:bc+3] == d).sum().item() > 1:
                        box_viol += 1

            step_metrics[step] = {
                "cell_accuracy": cell_acc,
                "puzzle_accuracy": puzzle_acc,
                "unknown_cell_accuracy": unk_acc,
                "hamming_distance": hamming,
                "row_violations": row_viol,
                "col_violations": col_viol,
                "box_violations": box_viol,
            }

        # Classify difficulty by step-0 cell accuracy
        step0_acc = step_metrics.get(0, {}).get("cell_accuracy", 0.0)
        difficulty = classify_difficulty(step0_acc)

        puzzles.append({
            "puzzle_idx": i,
            "difficulty": difficulty,
            "step0_cell_acc": step0_acc,
            "step_metrics": step_metrics,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{n_puzzles} puzzles processed")

    return puzzles


def compute_stratified_metrics(puzzles, max_steps=16):
    """Compute per-difficulty-stratum metrics."""
    strata = defaultdict(list)
    for p in puzzles:
        strata[p["difficulty"]].append(p)

    results = {}
    for diff_name in ["easy", "medium", "hard", "all"]:
        if diff_name == "all":
            subset = puzzles
        else:
            subset = strata.get(diff_name, [])

        if not subset:
            results[diff_name] = {"n": 0}
            continue

        per_step = {}
        for step in range(max_steps):
            accs = [p["step_metrics"][step]["cell_accuracy"]
                    for p in subset if step in p["step_metrics"]]
            puzz_accs = [p["step_metrics"][step]["puzzle_accuracy"]
                         for p in subset if step in p["step_metrics"]]
            hammings = [p["step_metrics"][step]["hamming_distance"]
                        for p in subset if step in p["step_metrics"]]
            viols = [(p["step_metrics"][step]["row_violations"] +
                      p["step_metrics"][step]["col_violations"] +
                      p["step_metrics"][step]["box_violations"])
                     for p in subset if step in p["step_metrics"]]

            if accs:
                per_step[step] = {
                    "cell_accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
                    "puzzle_accuracy": {"mean": float(np.mean(puzz_accs)), "std": float(np.std(puzz_accs))},
                    "hamming_distance": {"mean": float(np.mean(hammings)), "std": float(np.std(hammings))},
                    "total_violations": {"mean": float(np.mean(viols)), "std": float(np.std(viols))},
                }

        # Improvement from step 0 → step 15
        if 0 in per_step and (max_steps - 1) in per_step:
            delta_cell = per_step[max_steps-1]["cell_accuracy"]["mean"] - per_step[0]["cell_accuracy"]["mean"]
            delta_puzzle = per_step[max_steps-1]["puzzle_accuracy"]["mean"] - per_step[0]["puzzle_accuracy"]["mean"]
        else:
            delta_cell = delta_puzzle = 0.0

        results[diff_name] = {
            "n": len(subset),
            "per_step": per_step,
            "delta_cell_0_to_15": round(delta_cell, 4),
            "delta_puzzle_0_to_15": round(delta_puzzle, 4),
        }

    return results, strata


def plot_stratified(results, output_dir, max_steps=16):
    """Generate difficulty-stratified plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336", "all": "#2196F3"}
    labels = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "all": "All"}

    steps = list(range(max_steps))

    # (a) Cell accuracy vs step
    ax = axes[0, 0]
    for diff in ["easy", "medium", "hard", "all"]:
        info = results.get(diff, {})
        if not info.get("per_step"):
            continue
        accs = [info["per_step"].get(s, {}).get("cell_accuracy", {}).get("mean", np.nan) for s in steps]
        ls = "--" if diff == "all" else "-"
        ax.plot(steps, accs, label=f"{labels[diff]} (N={info['n']})",
                color=colors[diff], linewidth=2, linestyle=ls)
    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("(a) Cell Accuracy by Difficulty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Puzzle accuracy vs step
    ax = axes[0, 1]
    for diff in ["easy", "medium", "hard", "all"]:
        info = results.get(diff, {})
        if not info.get("per_step"):
            continue
        accs = [info["per_step"].get(s, {}).get("puzzle_accuracy", {}).get("mean", np.nan) for s in steps]
        ls = "--" if diff == "all" else "-"
        ax.plot(steps, accs, label=f"{labels[diff]} (N={info['n']})",
                color=colors[diff], linewidth=2, linestyle=ls)
    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Puzzle Accuracy")
    ax.set_title("(b) Puzzle Accuracy by Difficulty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Hamming vs step
    ax = axes[1, 0]
    for diff in ["easy", "medium", "hard", "all"]:
        info = results.get(diff, {})
        if not info.get("per_step"):
            continue
        hamm = [info["per_step"].get(s, {}).get("hamming_distance", {}).get("mean", np.nan) for s in steps]
        ls = "--" if diff == "all" else "-"
        ax.plot(steps, hamm, label=f"{labels[diff]} (N={info['n']})",
                color=colors[diff], linewidth=2, linestyle=ls)
    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Hamming Distance")
    ax.set_title("(c) Hamming Distance by Difficulty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Improvement bar chart
    ax = axes[1, 1]
    diffs_to_plot = [d for d in ["easy", "medium", "hard"] if d in results and results[d]["n"] > 0]
    x = np.arange(len(diffs_to_plot))
    cell_deltas = [results[d]["delta_cell_0_to_15"] * 100 for d in diffs_to_plot]
    puzzle_deltas = [results[d]["delta_puzzle_0_to_15"] * 100 for d in diffs_to_plot]
    bar_width = 0.35

    bars1 = ax.bar(x - bar_width/2, cell_deltas, bar_width,
                    label="Δ Cell Acc (pp)", color=[colors[d] for d in diffs_to_plot], alpha=0.7)
    bars2 = ax.bar(x + bar_width/2, puzzle_deltas, bar_width,
                    label="Δ Puzzle Acc (pp)", color=[colors[d] for d in diffs_to_plot], alpha=0.4,
                    edgecolor=[colors[d] for d in diffs_to_plot], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[d]}\n(N={results[d]['n']})" for d in diffs_to_plot])
    ax.set_ylabel("Improvement (pp)")
    ax.set_title("(d) Step 0→15 Improvement by Difficulty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Difficulty-Stratified Analysis of HRM Reasoning", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"difficulty_stratified.{ext}"))
    plt.close(fig)
    logger.info("Saved difficulty_stratified plots")


def main():
    parser = argparse.ArgumentParser(description="Gap 5: Difficulty-Stratified Analysis")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/metrics/difficulty_stratified")
    parser.add_argument("--quick", action="store_true", help="Quick test (N=50)")
    args = parser.parse_args()

    if args.quick:
        args.n_puzzles = 50

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Device: {device}, N puzzles: {args.n_puzzles}")

    # Load model
    model, test_loader = load_model_and_data(device)
    logger.info("Model loaded.")

    # Collect per-puzzle data
    t0 = time.time()
    puzzles = collect_per_puzzle_data(model, test_loader, device, args.n_puzzles, args.max_steps)
    logger.info(f"Data collection took {time.time()-t0:.1f}s")

    # Stratify and compute
    results, strata = compute_stratified_metrics(puzzles, args.max_steps)

    # Save
    with open(os.path.join(args.output_dir, "stratified_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save per-puzzle data for potential re-analysis
    per_puzzle_path = os.path.join(args.output_dir, "per_puzzle.json")
    with open(per_puzzle_path, "w") as f:
        json.dump([{
            "puzzle_idx": p["puzzle_idx"],
            "difficulty": p["difficulty"],
            "step0_cell_acc": p["step0_cell_acc"],
            "final_cell_acc": p["step_metrics"].get(args.max_steps - 1, {}).get("cell_accuracy", 0),
            "final_puzzle_acc": p["step_metrics"].get(args.max_steps - 1, {}).get("puzzle_accuracy", 0),
        } for p in puzzles], f, indent=2)

    # Plot
    plot_stratified(results, args.output_dir, args.max_steps)

    # Print summary
    print("\n" + "=" * 80)
    print("DIFFICULTY-STRATIFIED ANALYSIS — SUMMARY")
    print("=" * 80)
    print(f"{'Stratum':>8s}  {'N':>5s}  {'Step0 Cell%':>12s}  {'Step15 Cell%':>13s}  "
          f"{'Δ Cell (pp)':>12s}  {'Step15 Puzz%':>13s}  {'Δ Puzz (pp)':>12s}")
    print("-" * 80)

    for diff in ["easy", "medium", "hard", "all"]:
        info = results.get(diff, {})
        n = info.get("n", 0)
        if n == 0:
            print(f"{diff:>8s}  {0:>5d}  {'N/A':>12s}  {'N/A':>13s}  {'N/A':>12s}  {'N/A':>13s}  {'N/A':>12s}")
            continue

        s0 = info["per_step"].get(0, {})
        s15 = info["per_step"].get(args.max_steps - 1, {})
        s0_cell = s0.get("cell_accuracy", {}).get("mean", 0) * 100
        s15_cell = s15.get("cell_accuracy", {}).get("mean", 0) * 100
        s15_puzz = s15.get("puzzle_accuracy", {}).get("mean", 0) * 100
        d_cell = info.get("delta_cell_0_to_15", 0) * 100
        d_puzz = info.get("delta_puzzle_0_to_15", 0) * 100

        print(f"{diff:>8s}  {n:>5d}  {s0_cell:>11.1f}%  {s15_cell:>12.1f}%  "
              f"{d_cell:>+11.1f}pp  {s15_puzz:>12.1f}%  {d_puzz:>+11.1f}pp")

    logger.info(f"All results in {args.output_dir}")


if __name__ == "__main__":
    main()
