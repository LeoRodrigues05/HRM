"""scripts/controlled_ablation.py

Controlled single-variable ablation of z_H and z_L.

For each puzzle:
  - 1 baseline (no ablation)
  - 16 single-step z_H ablations (ablate step k only, k=0..15)
  - 1 all-steps z_H ablation
  - 16 single-step z_L ablations (ablate step k only, k=0..15)
  - 1 all-steps z_L ablation

All results include per-step accuracy trajectories, bootstrap CIs,
and are saved as per_puzzle.jsonl + aggregate.json.

Usage:
    python scripts/controlled_ablation.py --num_puzzles 5000
    python scripts/controlled_ablation.py --quick  # N=20 test
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

from scripts.controlled.controlled_common import (
    add_common_args, resolve_args,
    load_model_and_dataloader, collect_puzzles,
    bootstrap_ci, extract_batch,
)
from scripts.core.sudoku_sample import save_puzzle_indices
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.core.activation_patching import compute_metrics
from scripts.maze.maze_common import MAZE_METRIC_KEYS, maybe_maze_batch_metrics


def run_ablation_for_puzzle(
    ablator: ActivationAblator,
    batch: Dict[str, torch.Tensor],
    puzzle_idx: int,
    ablate_level: str,
    max_steps: int,
) -> Dict[str, Any]:
    """Run baseline + all single-step ablations + all-steps ablation for one level."""
    labels = batch["labels"]
    num_steps = max_steps

    # Baseline
    base_cache: Dict[int, ActivationCache] = {}
    base_out = ablator.run_and_cache_activations(batch, base_cache, max_steps=max_steps)
    base_preds = base_out["logits"].argmax(-1)
    base_acc = compute_metrics(base_preds, labels)["accuracy"]
    base_maze_metrics = maybe_maze_batch_metrics(base_preds, labels, batch.get("inputs"))

    # Per-step baseline accuracy
    base_step_accs = []
    for s in range(num_steps):
        if s in base_cache:
            base_step_accs.append(compute_metrics(base_cache[s].preds, labels)["accuracy"])
        else:
            base_step_accs.append(0.0)

    # All-steps ablation
    all_out, all_cache, _ = ablator.run_with_ablation(
        batch, ablate_level=ablate_level, ablate_steps=None, max_steps=max_steps,
    )
    all_preds = all_out["logits"].argmax(-1)
    all_acc = compute_metrics(all_preds, labels)["accuracy"]
    all_maze_metrics = maybe_maze_batch_metrics(all_preds, labels, batch.get("inputs"))

    # Single-step ablations
    single_step_accs = []
    single_step_deltas = []
    single_step_trajectories = []
    single_step_maze_deltas: List[Optional[Dict[str, float]]] = []

    for step_k in range(num_steps):
        abl_out, abl_cache, _ = ablator.run_with_ablation(
            batch, ablate_level=ablate_level, ablate_steps=[step_k], max_steps=max_steps,
        )
        abl_preds = abl_out["logits"].argmax(-1)
        abl_acc = compute_metrics(abl_preds, labels)["accuracy"]
        single_step_accs.append(abl_acc)
        single_step_deltas.append(abl_acc - base_acc)

        if base_maze_metrics is not None:
            step_maze = maybe_maze_batch_metrics(abl_preds, labels, batch.get("inputs"))
            if step_maze is not None:
                single_step_maze_deltas.append({
                    metric: step_maze[metric] - base_maze_metrics[metric]
                    for metric in MAZE_METRIC_KEYS
                })
            else:
                single_step_maze_deltas.append(None)

        # Per-step trajectory for this ablation
        traj = []
        for s in range(num_steps):
            if s in abl_cache:
                traj.append(compute_metrics(abl_cache[s].preds, labels)["accuracy"])
            else:
                traj.append(0.0)
        single_step_trajectories.append(traj)

    result = {
        "puzzle_idx": puzzle_idx,
        "ablate_level": ablate_level,
        "baseline_accuracy": base_acc,
        "baseline_step_accuracies": base_step_accs,
        "all_steps_accuracy": all_acc,
        "all_steps_delta": all_acc - base_acc,
        "single_step_accuracies": single_step_accs,
        "single_step_deltas": single_step_deltas,
        "single_step_trajectories": single_step_trajectories,
    }
    if base_maze_metrics is not None:
        result["baseline_maze_metrics"] = base_maze_metrics
        if all_maze_metrics is not None:
            result["all_steps_maze_metrics"] = all_maze_metrics
            result["all_steps_maze_deltas"] = {
                metric: all_maze_metrics[metric] - base_maze_metrics[metric]
                for metric in MAZE_METRIC_KEYS
            }
        result["single_step_maze_deltas"] = single_step_maze_deltas
    return result


def compute_aggregates(
    all_results: List[Dict[str, Any]],
    max_steps: int,
) -> Dict[str, Any]:
    """Compute aggregate statistics with bootstrap CIs."""
    n = len(all_results)
    if n == 0:
        return {}

    base_accs = [r["baseline_accuracy"] for r in all_results]
    all_step_deltas = [r["all_steps_delta"] for r in all_results]

    agg = {
        "num_puzzles": n,
        "ablate_level": all_results[0]["ablate_level"],
        "baseline_accuracy": bootstrap_ci(base_accs),
        "all_steps_delta": bootstrap_ci(all_step_deltas),
        "per_step_ablation": {},
    }

    have_maze = all("baseline_maze_metrics" in r for r in all_results)
    if have_maze:
        agg["baseline_maze_metrics"] = {
            metric: bootstrap_ci([r["baseline_maze_metrics"][metric] for r in all_results])
            for metric in MAZE_METRIC_KEYS
        }
        if all("all_steps_maze_deltas" in r for r in all_results):
            agg["all_steps_maze_deltas"] = {
                metric: bootstrap_ci([r["all_steps_maze_deltas"][metric] for r in all_results])
                for metric in MAZE_METRIC_KEYS
            }
            agg["all_steps_maze_metrics"] = {
                metric: bootstrap_ci([r["all_steps_maze_metrics"][metric] for r in all_results])
                for metric in MAZE_METRIC_KEYS
            }

    for step_k in range(max_steps):
        deltas = [r["single_step_deltas"][step_k] for r in all_results
                  if step_k < len(r["single_step_deltas"])]
        accs = [r["single_step_accuracies"][step_k] for r in all_results
                if step_k < len(r["single_step_accuracies"])]
        if deltas:
            agg["per_step_ablation"][step_k] = {
                "delta_accuracy": bootstrap_ci(deltas),
                "ablated_accuracy": bootstrap_ci(accs),
            }
            if have_maze:
                metric_deltas = {}
                for metric in MAZE_METRIC_KEYS:
                    dvals = []
                    for r in all_results:
                        ssmd = r.get("single_step_maze_deltas")
                        if ssmd is None or step_k >= len(ssmd) or ssmd[step_k] is None:
                            continue
                        dvals.append(ssmd[step_k][metric])
                    if dvals:
                        metric_deltas[metric] = bootstrap_ci(dvals)
                if metric_deltas:
                    agg["per_step_ablation"][step_k]["maze_metric_deltas"] = metric_deltas

    return agg


def main():
    parser = argparse.ArgumentParser(description="Controlled z_H/z_L Ablation (Revised E1)")
    add_common_args(parser)
    parser.add_argument("--num_puzzles", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="results/controlled_ablation")
    parser.add_argument("--z_level", type=str, default="both",
                        choices=["H", "L", "both"],
                        help="Which level to ablate: H, L, or both (default)")
    args = resolve_args(parser.parse_args())

    if args.quick:
        args.num_puzzles = 20

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CONTROLLED ABLATION EXPERIMENT (Revised E1)")
    print("=" * 70)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  N puzzles:   {args.num_puzzles}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Device:      {device}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)

    # Load model
    model_obj, test_loader, config = load_model_and_dataloader(args.checkpoint, device)
    ablator = ActivationAblator(model_obj, device=device)

    # Collect puzzles
    print(f"Collecting {args.num_puzzles} puzzles...")
    puzzles = collect_puzzles(
        test_loader,
        device,
        args.num_puzzles,
        seed=args.seed,
        puzzle_indices_path=args.puzzle_indices,
    )
    if args.save_puzzle_indices:
        save_puzzle_indices(
            args.save_puzzle_indices,
            [idx for idx, _batch in puzzles],
            metadata={
                "experiment": "controlled_ablation",
                "seed": args.seed,
                "num_puzzles": len(puzzles),
                "source": args.puzzle_indices or "first_n_test_loader",
            },
        )
    print(f"Collected {len(puzzles)} puzzles")

    # Run for z_H and/or z_L
    levels = ["H", "L"] if args.z_level == "both" else [args.z_level]
    for level in levels:
        level_dir = os.path.join(args.output_dir, f"z{level}")
        os.makedirs(level_dir, exist_ok=True)

        jsonl_path = os.path.join(level_dir, "per_puzzle.jsonl")
        agg_path = os.path.join(level_dir, "aggregate.json")

        all_results = []
        t0 = time.time()

        with open(jsonl_path, "w") as f:
            for i, (puzzle_idx, batch) in enumerate(puzzles):
                result = run_ablation_for_puzzle(
                    ablator, batch, puzzle_idx,
                    ablate_level=level, max_steps=args.max_steps,
                )
                all_results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()

                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                if (i + 1) % 10 == 0 or (i + 1) <= 3:
                    worst_step = min(range(args.max_steps),
                                     key=lambda s: result["single_step_deltas"][s]
                                     if s < len(result["single_step_deltas"]) else 0)
                    print(
                        f"  [z_{level}] {i+1:4d}/{len(puzzles)} "
                        f"base={result['baseline_accuracy']:.3f} "
                        f"worst_step={worst_step}(Δ={result['single_step_deltas'][worst_step]:+.3f}) "
                        f"all_Δ={result['all_steps_delta']:+.3f} "
                        f"| {rate:.1f} puz/s"
                    )

        elapsed = time.time() - t0
        print(f"\n  z_{level} done: {len(all_results)} puzzles in {elapsed:.1f}s")

        # Aggregate
        agg = compute_aggregates(all_results, args.max_steps)
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  Aggregate saved to {agg_path}")

        try:
            from scripts.core.provenance import write_meta
            write_meta(level_dir, f"controlled_ablation_z{level}", {
                "num_puzzles": len(puzzles),
                "requested_num_puzzles": args.num_puzzles,
                "ablate_level": level,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "puzzle_indices": args.puzzle_indices,
                "save_puzzle_indices": args.save_puzzle_indices,
                "checkpoint": args.checkpoint,
            })
        except Exception as e:
            print(f"  (could not write _meta.json: {e})")

        # Summary
        print(f"\n  z_{level} SUMMARY:")
        print(f"  {'Step':>6} {'Mean Δacc':>12} {'95% CI':>20}")
        print(f"  {'-'*40}")
        for s in range(args.max_steps):
            if s in agg.get("per_step_ablation", {}):
                d = agg["per_step_ablation"][s]["delta_accuracy"]
                print(f"  {s:>6} {d['mean']:>+12.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]")
        d_all = agg.get("all_steps_delta", {})
        if d_all:
            print(f"  {'ALL':>6} {d_all['mean']:>+12.4f} [{d_all.get('ci_lower',0):+.4f}, {d_all.get('ci_upper',0):+.4f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
