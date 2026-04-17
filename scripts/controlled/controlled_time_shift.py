"""scripts/controlled_time_shift.py

Controlled time-shift (cross-step transfer) experiment.

Two modes:
  1. fixed  (default) — The user-requested "keep one step fixed" variant:
     (a) Fix recipient step, sweep donor step 0..15
     (b) Fix donor step, sweep recipient step 0..15
  2. matrix — Full 16×16 donor×recipient transfer matrix (expensive).

For each (donor, recipient) pair:
  - Run baseline, cache all step activations
  - Transplant z_H from donor_step → recipient_step in a second pass
  - Measure final accuracy delta

Includes same-puzzle transfer only (within-puzzle time poaching).

Usage:
    python scripts/controlled_time_shift.py --mode fixed --num_puzzles 500
    python scripts/controlled_time_shift.py --mode matrix --num_puzzles 100
    python scripts/controlled_time_shift.py --quick
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
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _make_inner_carry,
)
from scripts.core.activation_patching import compute_metrics


class TimeShiftRunner(ActivationAblator):
    """Forward pass that transplants z_H from a cached donor step
    into a specific recipient step of a second forward pass."""

    def run_with_time_shift(
        self,
        batch: Dict[str, torch.Tensor],
        donor_cache: Dict[int, ActivationCache],
        donor_step: int,
        recipient_step: int,
        transfer_level: str = "H",
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache]]:
        """Forward pass, replacing activations at recipient_step with
        those from donor_step in the donor cache.

        Returns: (final_outputs, transferred_cache)
        """
        if donor_step not in donor_cache:
            raise KeyError(
                f"donor_step={donor_step} not in cache "
                f"(available: {sorted(donor_cache.keys())})"
            )

        transferred_cache: Dict[int, ActivationCache] = {}
        carry = self._init_carry(batch)
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps

        all_outputs: List[Dict[str, torch.Tensor]] = []
        step = 0

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, _steps_in, _current_data = self._prepare_step_inputs(carry, batch)

                    # At recipient step, inject donor activations
                    if step == recipient_step:
                        donor_act = donor_cache[donor_step]
                        if transfer_level in ("H", "both"):
                            inner_in = _make_inner_carry(
                                self.model,
                                z_H=donor_act.z_H.detach().clone(),
                                z_L=inner_in.z_L,
                            )
                        if transfer_level in ("L", "both"):
                            inner_in = _make_inner_carry(
                                self.model,
                                z_H=inner_in.z_H,
                                z_L=donor_act.z_L.detach().clone(),
                            )

                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    transferred_cache[step] = ActivationCache(
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

        return all_outputs[-1] if all_outputs else {}, transferred_cache


def generate_pairs_fixed(max_steps: int, fixed_donor: int = 10, fixed_recipient: int = 2):
    """Generate (donor, recipient) pairs for the fixed-step mode.

    Two sweeps:
      1. Fix recipient=fixed_recipient, sweep donor=0..max_steps-1 (skip donor==recipient)
      2. Fix donor=fixed_donor, sweep recipient=0..max_steps-1 (skip donor==recipient)
    """
    pairs = set()
    # Sweep 1: fix recipient, vary donor
    for d in range(max_steps):
        if d != fixed_recipient:
            pairs.add((d, fixed_recipient))
    # Sweep 2: fix donor, vary recipient
    for r in range(max_steps):
        if r != fixed_donor:
            pairs.add((fixed_donor, r))
    return sorted(pairs)


def generate_pairs_matrix(max_steps: int):
    """Generate all (donor, recipient) pairs where donor != recipient."""
    pairs = []
    for d in range(max_steps):
        for r in range(max_steps):
            if d != r:
                pairs.append((d, r))
    return pairs


def run_time_shift_for_puzzle(
    runner: TimeShiftRunner,
    batch: Dict[str, torch.Tensor],
    puzzle_idx: int,
    pairs: List[Tuple[int, int]],
    max_steps: int,
) -> Dict[str, Any]:
    """Run baseline + all transfer pairs for one puzzle."""
    labels = batch["labels"]

    # Baseline: cache all steps
    base_cache: Dict[int, ActivationCache] = {}
    base_out = runner.run_and_cache_activations(batch, base_cache, max_steps=max_steps)
    base_preds = base_out["logits"].argmax(-1)
    base_acc = compute_metrics(base_preds, labels)["accuracy"]

    transfers = {}
    for donor_step, recipient_step in pairs:
        if donor_step not in base_cache:
            continue
        out, t_cache = runner.run_with_time_shift(
            batch, base_cache, donor_step, recipient_step,
            transfer_level="H", max_steps=max_steps,
        )
        preds = out["logits"].argmax(-1) if out else base_preds
        acc = compute_metrics(preds, labels)["accuracy"]
        transfers[f"{donor_step}->{recipient_step}"] = {
            "accuracy": acc,
            "delta": acc - base_acc,
        }

    return {
        "puzzle_idx": puzzle_idx,
        "baseline_accuracy": base_acc,
        "transfers": transfers,
    }


def compute_aggregates(
    all_results: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Aggregate with bootstrap CIs."""
    n = len(all_results)
    base_accs = [r["baseline_accuracy"] for r in all_results]

    agg = {
        "num_puzzles": n,
        "baseline_accuracy": bootstrap_ci(base_accs),
        "per_pair": {},
    }

    for d, r in pairs:
        key = f"{d}->{r}"
        deltas = [res["transfers"][key]["delta"]
                  for res in all_results if key in res["transfers"]]
        accs = [res["transfers"][key]["accuracy"]
                for res in all_results if key in res["transfers"]]
        if deltas:
            agg["per_pair"][key] = {
                "donor_step": d,
                "recipient_step": r,
                "delta_accuracy": bootstrap_ci(deltas),
                "transferred_accuracy": bootstrap_ci(accs),
            }

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Controlled Time-Shift Experiment (Revised E5)"
    )
    add_common_args(parser)
    parser.add_argument("--num_puzzles", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results/controlled_time_shift")
    parser.add_argument(
        "--mode", type=str, default="fixed", choices=["fixed", "matrix"],
        help="'fixed': fix one step, vary the other (fast). "
             "'matrix': full donor×recipient matrix (slow).",
    )
    parser.add_argument("--fixed_donor", type=int, default=10,
                        help="Donor step to fix in fixed mode (sweep 2)")
    parser.add_argument("--fixed_recipient", type=int, default=2,
                        help="Recipient step to fix in fixed mode (sweep 1)")
    args = resolve_args(parser.parse_args())

    if args.quick:
        args.num_puzzles = 20

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate pairs
    if args.mode == "fixed":
        pairs = generate_pairs_fixed(
            args.max_steps, args.fixed_donor, args.fixed_recipient,
        )
    else:
        pairs = generate_pairs_matrix(args.max_steps)

    print("=" * 70)
    print("CONTROLLED TIME-SHIFT EXPERIMENT (Revised E5)")
    print("=" * 70)
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  N puzzles:       {args.num_puzzles}")
    print(f"  Mode:            {args.mode}")
    if args.mode == "fixed":
        print(f"  Fixed donor:     {args.fixed_donor}")
        print(f"  Fixed recipient: {args.fixed_recipient}")
    print(f"  Transfer pairs:  {len(pairs)}")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Device:          {device}")
    print(f"  Output:          {args.output_dir}")
    print("=" * 70)

    model_obj, test_loader, config = load_model_and_dataloader(args.checkpoint, device)
    runner = TimeShiftRunner(model_obj, device=device)

    puzzles = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
    print(f"Collected {len(puzzles)} puzzles, {len(pairs)} pairs each")

    jsonl_path = os.path.join(args.output_dir, "per_puzzle.jsonl")
    agg_path = os.path.join(args.output_dir, "aggregate.json")
    pairs_path = os.path.join(args.output_dir, "pairs.json")

    # Save pairs for reference
    with open(pairs_path, "w") as f:
        json.dump({"mode": args.mode, "pairs": [[d, r] for d, r in pairs]}, f, indent=2)

    all_results = []
    t0 = time.time()

    with open(jsonl_path, "w") as f:
        for i, (puzzle_idx, batch) in enumerate(puzzles):
            result = run_time_shift_for_puzzle(
                runner, batch, puzzle_idx, pairs, args.max_steps,
            )
            all_results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            if (i + 1) % 10 == 0 or (i + 1) <= 3:
                # Best positive transfer
                best_pair = max(result["transfers"].items(),
                                key=lambda x: x[1]["delta"],
                                default=("N/A", {"delta": 0.0}))
                print(
                    f"  {i+1:4d}/{len(puzzles)} "
                    f"base={result['baseline_accuracy']:.3f} "
                    f"best={best_pair[0]}(Δ={best_pair[1]['delta']:+.3f}) "
                    f"| {rate:.1f} puz/s"
                )

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_results)} puzzles in {elapsed:.1f}s")

    agg = compute_aggregates(all_results, pairs)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate saved to {agg_path}")

    # Summary: top 10 best and worst pairs
    sorted_pairs = sorted(
        agg["per_pair"].items(),
        key=lambda x: x[1]["delta_accuracy"]["mean"],
        reverse=True,
    )

    print(f"\nTOP 10 BEST TRANSFERS:")
    print(f"  {'Pair':>12} {'Mean Δacc':>12} {'95% CI':>24}")
    print(f"  {'-'*50}")
    for key, v in sorted_pairs[:10]:
        d = v["delta_accuracy"]
        print(f"  {key:>12} {d['mean']:>+12.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]")

    print(f"\nTOP 10 WORST TRANSFERS:")
    for key, v in sorted_pairs[-10:]:
        d = v["delta_accuracy"]
        print(f"  {key:>12} {d['mean']:>+12.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]")

    # For matrix mode, also output a 2D heatmap-ready matrix
    if args.mode == "matrix":
        matrix = np.full((args.max_steps, args.max_steps), np.nan)
        for key, v in agg["per_pair"].items():
            d_s, r_s = key.split("->")
            matrix[int(d_s), int(r_s)] = v["delta_accuracy"]["mean"]
        matrix_path = os.path.join(args.output_dir, "transfer_matrix.npy")
        np.save(matrix_path, matrix)
        print(f"\nTransfer matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
