"""scripts/controlled_freeze.py

Controlled freeze experiment: freeze z_H or z_L after step k.

For each puzzle:
  - 1 baseline (no freeze)
  - For k=0..15: freeze z_H after step k  (z_L continues evolving)
  - For k=0..15: freeze z_L after step k  (z_H continues evolving)

Tests whether each stream acts as a static plan (set early, read later)
or requires continuous dynamic refinement.

Usage:
    python scripts/controlled_freeze.py --num_puzzles 1000
    python scripts/controlled_freeze.py --quick
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

from scripts.controlled.controlled_common import (
    add_common_args, resolve_args,
    load_model_and_dataloader, collect_puzzles,
    bootstrap_ci, extract_batch,
)
from scripts.core.sudoku_sample import save_puzzle_indices
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _make_inner_carry,
)
from scripts.core.activation_patching import compute_metrics
from scripts.maze.maze_common import MAZE_METRIC_KEYS, maybe_maze_batch_metrics


class FreezeRunner(ActivationAblator):
    """Forward pass that freezes z_H or z_L after a specified step.

    At step ``freeze_at``, caches the specified stream's output.
    For all steps > freeze_at, injects the frozen snapshot.
    The other stream continues evolving normally.
    """

    def run_with_freeze(
        self,
        batch: Dict[str, torch.Tensor],
        freeze_at_step: int,
        freeze_level: str = "H",       # "H" or "L"
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache], Dict[str, Any]]:
        """Forward pass freezing one stream after ``freeze_at_step``."""
        frozen_cache: Dict[int, ActivationCache] = {}
        freeze_info: Dict[str, Any] = {
            "freeze_at_step": freeze_at_step,
            "freeze_level": freeze_level,
            "frozen_norm": None,
            "steps_frozen": [],
        }

        carry = self._init_carry(batch)
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps

        all_outputs: List[Dict[str, torch.Tensor]] = []
        step = 0
        frozen_activation: Optional[torch.Tensor] = None

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, _steps_in, _current_data = self._prepare_step_inputs(carry, batch)

                    # If past freeze step, inject frozen activation
                    if frozen_activation is not None and step > freeze_at_step:
                        if freeze_level == "H":
                            inner_in = _make_inner_carry(
                                self.model,
                                z_H=frozen_activation.clone(),
                                z_L=inner_in.z_L,
                            )
                        else:  # "L"
                            inner_in = _make_inner_carry(
                                self.model,
                                z_H=inner_in.z_H,
                                z_L=frozen_activation.clone(),
                            )
                        freeze_info["steps_frozen"].append(step)

                    # Forward
                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    # Capture activation at freeze step
                    if step == freeze_at_step:
                        if freeze_level == "H":
                            frozen_activation = new_carry.inner_carry.z_H.detach().clone()
                        else:
                            frozen_activation = new_carry.inner_carry.z_L.detach().clone()
                        freeze_info["frozen_norm"] = float(frozen_activation.norm().item())

                    frozen_cache[step] = ActivationCache(
                        z_H=inner_used.z_H.detach().clone(),
                        z_L=inner_used.z_L.detach().clone(),
                        step=step,
                        z_H_out=new_carry.inner_carry.z_H.detach().clone(),
                        z_L_out=new_carry.inner_carry.z_L.detach().clone(),
                        logits=outputs["logits"].detach().clone(),
                        preds=outputs["intermediate_preds_step"].detach().clone(),
                        q_halt_logits=outputs["q_halt_logits"].detach().clone(),
                        q_continue_logits=outputs["q_continue_logits"].detach().clone(),
                    )

                    all_outputs.append(outputs)
                    carry = new_carry
                    step += 1
                    if step >= (max_steps or original_max_steps):
                        break
        finally:
            if max_steps is not None:
                self.model.config.halt_max_steps = original_max_steps

        return all_outputs[-1] if all_outputs else {}, frozen_cache, freeze_info


def run_freeze_for_puzzle(
    runner: FreezeRunner,
    batch: Dict[str, torch.Tensor],
    puzzle_idx: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run baseline + all freeze conditions for one puzzle."""
    labels = batch["labels"]

    # Baseline
    base_cache: Dict[int, ActivationCache] = {}
    base_out = runner.run_and_cache_activations(batch, base_cache, max_steps=max_steps)
    base_preds = base_out["logits"].argmax(-1)
    base_acc = compute_metrics(base_preds, labels)["accuracy"]
    base_maze_metrics = maybe_maze_batch_metrics(base_preds, labels, batch.get("inputs"))

    result = {
        "puzzle_idx": puzzle_idx,
        "baseline_accuracy": base_acc,
        "freeze_H": {},
        "freeze_L": {},
    }
    if base_maze_metrics is not None:
        result["baseline_maze_metrics"] = base_maze_metrics

    # Freeze z_H at each step
    for k in range(max_steps):
        out, cache, info = runner.run_with_freeze(
            batch, freeze_at_step=k, freeze_level="H", max_steps=max_steps,
        )
        preds = out["logits"].argmax(-1) if out else base_preds
        acc = compute_metrics(preds, labels)["accuracy"]
        row = {
            "accuracy": acc,
            "delta": acc - base_acc,
            "steps_frozen": len(info["steps_frozen"]),
        }
        maze_metrics = maybe_maze_batch_metrics(preds, labels, batch.get("inputs"))
        if base_maze_metrics is not None and maze_metrics is not None:
            row["maze_metrics"] = maze_metrics
            row["maze_metric_deltas"] = {
                metric: maze_metrics[metric] - base_maze_metrics[metric]
                for metric in MAZE_METRIC_KEYS
            }
        result["freeze_H"][k] = row

    # Freeze z_L at each step
    for k in range(max_steps):
        out, cache, info = runner.run_with_freeze(
            batch, freeze_at_step=k, freeze_level="L", max_steps=max_steps,
        )
        preds = out["logits"].argmax(-1) if out else base_preds
        acc = compute_metrics(preds, labels)["accuracy"]
        row = {
            "accuracy": acc,
            "delta": acc - base_acc,
            "steps_frozen": len(info["steps_frozen"]),
        }
        maze_metrics = maybe_maze_batch_metrics(preds, labels, batch.get("inputs"))
        if base_maze_metrics is not None and maze_metrics is not None:
            row["maze_metrics"] = maze_metrics
            row["maze_metric_deltas"] = {
                metric: maze_metrics[metric] - base_maze_metrics[metric]
                for metric in MAZE_METRIC_KEYS
            }
        result["freeze_L"][k] = row

    return result


def compute_aggregates(all_results: List[Dict[str, Any]], max_steps: int) -> Dict[str, Any]:
    """Aggregate with bootstrap CIs."""
    n = len(all_results)
    base_accs = [r["baseline_accuracy"] for r in all_results]

    agg = {
        "num_puzzles": n,
        "baseline_accuracy": bootstrap_ci(base_accs),
        "freeze_H": {},
        "freeze_L": {},
    }
    if all("baseline_maze_metrics" in r for r in all_results):
        agg["baseline_maze_metrics"] = {
            metric: bootstrap_ci([r["baseline_maze_metrics"][metric] for r in all_results])
            for metric in MAZE_METRIC_KEYS
        }

    for level in ["freeze_H", "freeze_L"]:
        for k in range(max_steps):
            deltas = [r[level][str(k) if isinstance(list(r[level].keys())[0], str) else k]["delta"]
                      for r in all_results
                      if (str(k) if isinstance(list(r[level].keys())[0], str) else k) in r[level]]
            accs = [r[level][str(k) if isinstance(list(r[level].keys())[0], str) else k]["accuracy"]
                    for r in all_results
                    if (str(k) if isinstance(list(r[level].keys())[0], str) else k) in r[level]]
            if deltas:
                agg[level][k] = {
                    "delta_accuracy": bootstrap_ci(deltas),
                    "frozen_accuracy": bootstrap_ci(accs),
                }
                if all("baseline_maze_metrics" in r for r in all_results):
                    metric_deltas = {}
                    metric_values = {}
                    for metric in MAZE_METRIC_KEYS:
                        dvals = []
                        vals = []
                        for r in all_results:
                            rk = str(k) if isinstance(list(r[level].keys())[0], str) else k
                            row = r[level].get(rk)
                            if row is None or "maze_metric_deltas" not in row:
                                continue
                            dvals.append(row["maze_metric_deltas"][metric])
                            vals.append(row["maze_metrics"][metric])
                        if dvals:
                            metric_deltas[metric] = bootstrap_ci(dvals)
                            metric_values[metric] = bootstrap_ci(vals)
                    agg[level][k]["maze_metric_deltas"] = metric_deltas
                    agg[level][k]["maze_metrics"] = metric_values

    return agg


def main():
    parser = argparse.ArgumentParser(description="Controlled Freeze Experiment (Revised E2b)")
    add_common_args(parser)
    parser.add_argument("--num_puzzles", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/controlled_freeze")
    args = resolve_args(parser.parse_args())

    if args.quick:
        args.num_puzzles = 20

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CONTROLLED FREEZE EXPERIMENT (Revised E2b)")
    print("=" * 70)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  N puzzles:   {args.num_puzzles}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Device:      {device}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)

    model_obj, test_loader, config = load_model_and_dataloader(args.checkpoint, device)
    runner = FreezeRunner(model_obj, device=device)

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
                "experiment": "controlled_freeze",
                "seed": args.seed,
                "num_puzzles": len(puzzles),
                "source": args.puzzle_indices or "first_n_test_loader",
            },
        )
    print(f"Collected {len(puzzles)} puzzles")

    jsonl_path = os.path.join(args.output_dir, "per_puzzle.jsonl")
    agg_path = os.path.join(args.output_dir, "aggregate.json")

    all_results = []
    t0 = time.time()

    with open(jsonl_path, "w") as f:
        for i, (puzzle_idx, batch) in enumerate(puzzles):
            result = run_freeze_for_puzzle(runner, batch, puzzle_idx, args.max_steps)
            # Convert int keys to str for JSON
            result["freeze_H"] = {str(k): v for k, v in result["freeze_H"].items()}
            result["freeze_L"] = {str(k): v for k, v in result["freeze_L"].items()}
            all_results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            if (i + 1) % 10 == 0 or (i + 1) <= 3:
                # Find crossover: earliest freeze step with < 5% drop
                h_deltas = [result["freeze_H"][str(k)]["delta"] for k in range(args.max_steps)]
                crossover_H = next(
                    (k for k in range(args.max_steps) if h_deltas[k] > -0.05),
                    args.max_steps,
                )
                print(
                    f"  {i+1:4d}/{len(puzzles)} "
                    f"base={result['baseline_accuracy']:.3f} "
                    f"freezeH@0_Δ={h_deltas[0]:+.3f} "
                    f"crossover_H≈{crossover_H} "
                    f"| {rate:.1f} puz/s"
                )

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_results)} puzzles in {elapsed:.1f}s")

    # Aggregate
    agg = compute_aggregates(all_results, args.max_steps)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate saved to {agg_path}")

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "controlled_freeze", {
            "num_puzzles": len(puzzles),
            "requested_num_puzzles": args.num_puzzles,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "puzzle_indices": args.puzzle_indices,
            "save_puzzle_indices": args.save_puzzle_indices,
            "checkpoint": args.checkpoint,
        })
    except Exception as e:
        print(f"  (could not write _meta.json: {e})")

    # Summary table
    print(f"\n{'Step':>6} {'Freeze z_H Δacc':>20} {'Freeze z_L Δacc':>20}")
    print("-" * 50)
    for s in range(args.max_steps):
        h_d = agg["freeze_H"].get(s, {}).get("delta_accuracy", {})
        l_d = agg["freeze_L"].get(s, {}).get("delta_accuracy", {})
        h_str = f"{h_d.get('mean',0):+.4f}" if h_d else "   N/A"
        l_str = f"{l_d.get('mean',0):+.4f}" if l_d else "   N/A"
        print(f"  {s:>4}  {h_str:>20}  {l_str:>20}")


if __name__ == "__main__":
    main()
