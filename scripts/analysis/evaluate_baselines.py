#!/usr/bin/env python3
"""Evaluate baseline models (Vanilla RNN / Universal Transformer) on Sudoku-Extreme.

Runs inference on the test set and computes:
  - Cell accuracy, puzzle accuracy, unknown-cell accuracy per step
  - Constraint violations (row/col/box) per step
  - Hamming distance to solution per step
  - Activation patching (patch hidden state between puzzles, measure Δacc)

Outputs results in JSON format compatible with HRM evaluation for comparison.

Usage:
    python scripts/analysis/evaluate_baselines.py \\
        --checkpoint checkpoints/vanilla_rnn/checkpoint.pt \\
        --model_type vanilla_rnn \\
        --device cuda \\
        --output_dir results/baseline_comparison
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from scripts.core.sudoku_sample import (
    collect_indexed_batches,
    load_puzzle_indices,
    save_puzzle_indices,
)

SUDOKU_SIZE = 9
SUDOKU_CELLS = 81
IGNORE_LABEL_ID = -100


def bootstrap_ci(values, n_boot: int = 10000, confidence: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(mean, ci_lower, ci_upper)``. Resamples the per-puzzle values with
    replacement so the interval reflects sampling variability across puzzles.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + confidence) / 2))
    return mean, lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_sudoku_metrics(preds: torch.Tensor, targets: torch.Tensor, inputs: torch.Tensor):
    """Compute per-puzzle metrics from predictions.

    Args:
        preds:   [B, 81] predicted token ids
        targets: [B, 81] ground-truth token ids
        inputs:  [B, 81] input token ids (givens)

    Returns dict with scalar metrics.
    """
    B = preds.shape[0]
    mask = targets != IGNORE_LABEL_ID
    correct = (preds == targets) & mask

    # Cell accuracy
    cell_acc = correct.float().sum(-1) / mask.float().sum(-1).clamp(min=1)

    # Puzzle accuracy (all cells correct)
    puzzle_acc = (correct.sum(-1) == mask.sum(-1)).float()

    # Unknown-cell accuracy (cells that are NOT givens)
    digit_offset = 1
    input_digits = (inputs - digit_offset).clamp(0, SUDOKU_SIZE)
    unknown_mask = mask & (input_digits == 0)
    unknown_correct = correct & (input_digits == 0)
    unknown_acc = unknown_correct.float().sum(-1) / unknown_mask.float().sum(-1).clamp(min=1)

    # Hamming distance to solution
    hamming = (mask & (preds != targets)).float().sum(-1)

    # Constraint violations
    pred_digits = (preds - digit_offset).clamp(0, SUDOKU_SIZE)
    grid = pred_digits.view(B, SUDOKU_SIZE, SUDOKU_SIZE)

    row_violations = torch.zeros(B)
    col_violations = torch.zeros(B)
    box_violations = torch.zeros(B)

    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            d = grid[:, r, c]
            nonzero = d > 0

            # Row
            row_count = (grid[:, r, :] == d.unsqueeze(1)).sum(dim=1)
            row_violations += (nonzero & (row_count > 1)).float()

            # Col
            col_count = (grid[:, :, c] == d.unsqueeze(1)).sum(dim=1)
            col_violations += (nonzero & (col_count > 1)).float()

            # Box
            br, bc = (r // 3) * 3, (c // 3) * 3
            box = grid[:, br:br+3, bc:bc+3].reshape(B, 9)
            box_count = (box == d.unsqueeze(1)).sum(dim=1)
            box_violations += (nonzero & (box_count > 1)).float()

    return {
        "cell_accuracy": cell_acc.mean().item(),
        "puzzle_accuracy": puzzle_acc.mean().item(),
        "unknown_cell_accuracy": unknown_acc.mean().item(),
        "hamming_distance": hamming.mean().item(),
        "row_violations": row_violations.mean().item(),
        "col_violations": col_violations.mean().item(),
        "box_violations": box_violations.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_baseline_model(checkpoint_path: str, device: torch.device):
    """Load a baseline (or HRM) model from a checkpoint directory."""
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
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=test_meta.vocab_size,
        seq_len=test_meta.seq_len,
        num_puzzle_identifiers=test_meta.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(k.startswith("_orig_mod.") for k in ck):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(k.startswith("_orig_mod.") for k in mk):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    # Unwrap to inner model
    m = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if hasattr(m, "model"):
        m = m.model

    return m, model_full, test_loader, config


# ═══════════════════════════════════════════════════════════════════════════
# Per-step evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _extract_batch(data):
    if isinstance(data, (tuple, list)):
        return data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
    return data


def evaluate_per_step(
    model,
    test_loader,
    device,
    n_puzzles,
    max_steps=16,
    puzzle_indices_path=None,
):
    """Run model on test puzzles and collect per-step metrics."""
    model.eval()
    step_metrics: Dict[int, List[Dict]] = {s: [] for s in range(max_steps)}
    puzzle_indices = (
        load_puzzle_indices(puzzle_indices_path, limit=n_puzzles)
        if puzzle_indices_path else None
    )
    indexed_batches = collect_indexed_batches(
        test_loader,
        device,
        num_puzzles=n_puzzles,
        puzzle_indices=puzzle_indices,
        extract_batch=_extract_batch,
    )

    with torch.inference_mode():
        for _puzzle_idx, batch in indexed_batches:
            with torch.device(device):
                carry = model.initial_carry(batch)

            for step in range(max_steps):
                carry, outputs = model(carry=carry, batch=batch)

                preds = outputs["intermediate_preds_step"]  # [B, 81]
                targets = batch["labels"][:, -SUDOKU_CELLS:]
                inputs = batch["inputs"][:, -SUDOKU_CELLS:]

                metrics = compute_sudoku_metrics(
                    preds.cpu(), targets.cpu(), inputs.cpu(),
                )
                step_metrics[step].append(metrics)

    # Aggregate per step
    aggregated = {}
    for step in range(max_steps):
        if not step_metrics[step]:
            continue
        agg = {}
        for key in step_metrics[step][0]:
            vals = [m[key] for m in step_metrics[step]]
            mean, lo, hi = bootstrap_ci(vals)
            agg[key] = {
                "mean": mean,
                "std": float(np.std(vals)),
                "ci_lower": lo,
                "ci_upper": hi,
                "n": len(vals),
            }
        aggregated[step] = agg

    return aggregated, [idx for idx, _batch in indexed_batches]


# ═══════════════════════════════════════════════════════════════════════════
# Activation patching
# ═══════════════════════════════════════════════════════════════════════════

def activation_patching_experiment(
    model,
    test_loader,
    device,
    n_pairs=100,
    max_steps=16,
    puzzle_indices_path=None,
):
    """Patch hidden state from puzzle B into puzzle A at each step, measure Δacc.

    This measures causal importance of the hidden state representation.
    """
    model.eval()

    puzzle_indices = (
        load_puzzle_indices(puzzle_indices_path, limit=2 * n_pairs)
        if puzzle_indices_path else None
    )
    indexed_batches = collect_indexed_batches(
        test_loader,
        device,
        num_puzzles=2 * n_pairs,
        puzzle_indices=puzzle_indices,
        extract_batch=_extract_batch,
    )
    puzzles = [batch for _idx, batch in indexed_batches]

    results_by_step = {}

    with torch.inference_mode():
        for patch_step in range(0, max_steps, 2):  # Every 2 steps for efficiency
            deltas = []
            for i in range(min(n_pairs, len(puzzles) // 2)):
                batch_a = puzzles[2 * i]
                batch_b = puzzles[2 * i + 1]

                # Run A normally
                with torch.device(device):
                    carry_a = model.initial_carry(batch_a)
                for s in range(max_steps):
                    carry_a, outputs_a = model(carry=carry_a, batch=batch_a)
                preds_a = outputs_a["intermediate_preds_step"].cpu()
                targets_a = batch_a["labels"][:, -SUDOKU_CELLS:].cpu()
                inputs_a = batch_a["inputs"][:, -SUDOKU_CELLS:].cpu()
                base_acc = compute_sudoku_metrics(preds_a, targets_a, inputs_a)["cell_accuracy"]

                # Run B up to patch_step to get donor carry
                with torch.device(device):
                    carry_b = model.initial_carry(batch_b)
                for s in range(patch_step + 1):
                    carry_b, _ = model(carry=carry_b, batch=batch_b)

                # Run A with B's carry injected at patch_step
                with torch.device(device):
                    carry_patched = model.initial_carry(batch_a)
                for s in range(max_steps):
                    if s == patch_step:
                        # Inject B's hidden state into A's carry
                        carry_patched.inner_carry = carry_b.inner_carry
                    carry_patched, outputs_patched = model(carry=carry_patched, batch=batch_a)

                preds_patched = outputs_patched["intermediate_preds_step"].cpu()
                patched_acc = compute_sudoku_metrics(preds_patched, targets_a, inputs_a)["cell_accuracy"]

                deltas.append(patched_acc - base_acc)

            results_by_step[patch_step] = {
                "mean_delta_accuracy": float(np.mean(deltas)),
                "std_delta_accuracy": float(np.std(deltas)),
                "ci_lower_delta_accuracy": bootstrap_ci(deltas)[1],
                "ci_upper_delta_accuracy": bootstrap_ci(deltas)[2],
                "n_pairs": len(deltas),
            }

    return results_by_step


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on Sudoku-Extreme")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for this model in results (default: derived from checkpoint)")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--n_patching_pairs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/baseline_comparison")
    parser.add_argument("--puzzle_indices", type=str, default=None,
                        help="JSON manifest/list of dataloader puzzle indices to evaluate")
    parser.add_argument("--save_puzzle_indices", type=str, default=None,
                        help="Write the collected dataloader puzzle indices to this JSON manifest")
    parser.add_argument("--skip_patching", action="store_true",
                        help="Skip activation patching experiment")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_name = args.model_name
    if model_name is None:
        # Derive from checkpoint directory
        model_name = os.path.basename(os.path.dirname(args.checkpoint))

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  N puzzles:  {args.n_puzzles}")
    print(f"  Device:     {device}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 70)

    # Load model
    model, model_full, test_loader, config = load_baseline_model(args.checkpoint, device)

    # Per-step evaluation
    print("\nRunning per-step evaluation...")
    t0 = time.time()
    step_results, collected_indices = evaluate_per_step(
        model,
        test_loader,
        device,
        args.n_puzzles,
        args.max_steps,
        puzzle_indices_path=args.puzzle_indices,
    )
    if args.save_puzzle_indices:
        save_puzzle_indices(
            args.save_puzzle_indices,
            collected_indices,
            metadata={
                "experiment": "baseline_eval",
                "model_name": model_name,
                "num_puzzles": len(collected_indices),
                "source": args.puzzle_indices or "first_n_test_loader",
            },
        )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Activation patching
    patching_results = None
    if not args.skip_patching:
        print("\nRunning activation patching experiment...")
        t0 = time.time()
        _, _, test_loader2, _ = load_baseline_model(args.checkpoint, device)
        patching_results = activation_patching_experiment(
            model,
            test_loader2,
            device,
            args.n_patching_pairs,
            args.max_steps,
            puzzle_indices_path=args.puzzle_indices,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

    # Save results
    output = {
        "model_name": model_name,
        "checkpoint": args.checkpoint,
        "n_puzzles": args.n_puzzles,
        "max_steps": args.max_steps,
        "per_step_metrics": {str(k): v for k, v in step_results.items()},
    }
    if patching_results:
        output["activation_patching"] = {str(k): v for k, v in patching_results.items()}

    out_path = os.path.join(args.output_dir, f"{model_name}_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "baseline_eval", {
            "model_name": model_name, "checkpoint": args.checkpoint,
            "n_puzzles": args.n_puzzles, "max_steps": args.max_steps,
            "n_patching_pairs": args.n_patching_pairs,
            "puzzle_indices": args.puzzle_indices,
            "save_puzzle_indices": args.save_puzzle_indices,
            "num_puzzles_collected": len(collected_indices),
        }, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"  (could not write _meta.json: {e})")

    # Print summary table
    print(f"\n{'Step':>4s}  {'CellAcc':>8s}  {'PuzzAcc':>8s}  {'UnkAcc':>8s}  "
          f"{'Hamming':>8s}  {'RowViol':>8s}  {'ColViol':>8s}  {'BoxViol':>8s}")
    print("-" * 76)
    for step in sorted(step_results.keys()):
        m = step_results[step]
        print(f"  {step:3d}  {m['cell_accuracy']['mean']:>8.4f}  "
              f"{m['puzzle_accuracy']['mean']:>8.4f}  "
              f"{m['unknown_cell_accuracy']['mean']:>8.4f}  "
              f"{m['hamming_distance']['mean']:>8.1f}  "
              f"{m['row_violations']['mean']:>8.1f}  "
              f"{m['col_violations']['mean']:>8.1f}  "
              f"{m['box_violations']['mean']:>8.1f}")

    if patching_results:
        print(f"\nActivation Patching (Δacc when injecting foreign hidden state):")
        for step in sorted(patching_results.keys()):
            r = patching_results[step]
            print(f"  Step {step:2d}: Δacc = {r['mean_delta_accuracy']:+.4f} ± {r['std_delta_accuracy']:.4f}")


if __name__ == "__main__":
    main()
