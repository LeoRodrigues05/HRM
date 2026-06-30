#!/usr/bin/env python3
"""ARC freezing experiment — freeze z_H (or z_L) after step k, score the damage.

ARC analog of ``scripts/controlled/controlled_freeze.py`` (Sudoku) and the maze
``freeze_controlled`` run. Reuses the *task-agnostic* intervention harness
``FreezeRunner.run_with_freeze`` (caches the stream's output at step k and injects
that frozen snapshot for every later step, while the other stream keeps evolving),
but scores the result on **ARC structure** (colour-cell accuracy etc.) instead of
Sudoku cell accuracy / maze path validity.

The point of the experiment (Finding 1, "task-dependent refinement depth"): a smooth
decay of freeze-after-k damage means z_H is *progressively* refined (Sudoku); a flat
~0 curve means z_H is a static plan that is only read out at the end (Maze). This run
places ARC on that spectrum.

Primary metric = ``colour_cell_acc`` (the ARC analog of the Sudoku/Maze task metric,
since exact-grid solving is ~0 for the frozen-core checkpoint). For cross-task figure
uniformity the aggregate also stores it under the key ``delta_accuracy`` (matching the
Sudoku/Maze freeze schema), with every ARC metric under ``arc_metric_deltas``.

Usage (GPU, via srun on ws-l3-019):
  python scripts/arc/freeze_arc.py \
      --checkpoint checkpoints/arc2-adapted-evalonly/step_7391 \
      --num_puzzles 500 --freeze_levels H,L \
      --output_dir results/arc/freeze
Smoke:
  python scripts/arc/freeze_arc.py --num_puzzles 8 --freeze_steps 0,8,15 \
      --output_dir results/arc_smoke/freeze
"""
from __future__ import annotations
import os, sys, json, time, argparse
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci,
)
from scripts.controlled.controlled_freeze import FreezeRunner
from scripts.core.activation_ablation import ActivationCache
from scripts.arc.arc_common import ARC_METRIC_KEYS, arc_prediction_metrics
from scripts.maze.linear_probes_maze import _flat, _slice_preds

# Default checkpoint = the Path-A adapted, frozen-core evalonly model that every
# other reportable ARC experiment used (see results/arc/*/_meta.json).
ARC_ADAPTED_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "arc2-adapted-evalonly", "step_7391")
PRIMARY = "colour_cell_acc"


def score(cache: Dict[int, ActivationCache], label, inp) -> Dict[str, float]:
    preds = _slice_preds(_flat(cache[max(cache.keys())].preds), label.size)
    return arc_prediction_metrics(preds, label, inp)


def main():
    p = argparse.ArgumentParser(description="ARC freezing (freeze z_H/z_L after step k)")
    p.add_argument("--checkpoint", default=ARC_ADAPTED_CKPT)
    p.add_argument("--num_puzzles", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--freeze_levels", default="H,L", help="comma list of H/L")
    p.add_argument("--freeze_steps", default="0,1,2,4,8,12,15",
                   help="comma list of k to freeze-after (subset of 0..max_steps-1)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/arc/freeze")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    levels = [s.strip().upper() for s in args.freeze_levels.split(",") if s.strip()]
    ks = [int(x) for x in args.freeze_steps.split(",") if x.strip() != ""]

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    runner = FreezeRunner(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
    print(f"[arc_freeze] {len(puzzles)} puzzles | levels={levels} | ks={ks} | metric={PRIMARY}")

    # per_puzzle[level][k] -> list of {delta_<metric>...}; baselines collected once.
    per_puzzle: Dict[str, Dict[int, List[Dict]]] = {f"freeze_{lv}": {k: [] for k in ks} for lv in levels}
    base_primary: List[float] = []
    jsonl = open(os.path.join(args.output_dir, "per_puzzle.jsonl"), "w")
    t0 = time.time()

    for i, (idx, batch) in enumerate(puzzles):
        inp, label = _flat(batch["inputs"]), _flat(batch["labels"])
        base_cache: Dict[int, ActivationCache] = {}
        runner.run_and_cache_activations(batch, base_cache, max_steps=args.max_steps)
        bm = score(base_cache, label, inp)
        base_primary.append(bm[PRIMARY])
        row_out = {"puzzle_idx": idx, "baseline": {k: bm[k] for k in ARC_METRIC_KEYS}}

        for lv in levels:
            row_out[f"freeze_{lv}"] = {}
            for k in ks:
                _, cache, _info = runner.run_with_freeze(
                    batch, freeze_at_step=k, freeze_level=lv, max_steps=args.max_steps)
                fm = score(cache, label, inp)
                deltas = {f"delta_{m}": fm[m] - bm[m] for m in ARC_METRIC_KEYS}
                per_puzzle[f"freeze_{lv}"][k].append(deltas)
                row_out[f"freeze_{lv}"][str(k)] = {**deltas, PRIMARY: fm[PRIMARY]}
        jsonl.write(json.dumps(row_out, default=float) + "\n"); jsonl.flush()

        if (i + 1) % 25 == 0 or (i + 1) <= 2:
            lv0 = f"freeze_{levels[0]}"
            d0 = np.mean([r[f"delta_{PRIMARY}"] for r in per_puzzle[lv0][ks[0]]])
            dN = np.mean([r[f"delta_{PRIMARY}"] for r in per_puzzle[lv0][ks[-1]]])
            print(f"  {i+1}/{len(puzzles)} base={np.mean(base_primary):.3f} "
                  f"{lv0} k{ks[0]}Δ={d0:+.4f} k{ks[-1]}Δ={dN:+.4f} | {(i+1)/(time.time()-t0):.1f} puz/s")
    jsonl.close()

    # Aggregate (bootstrap CIs); store PRIMARY both as delta_accuracy (cross-task
    # schema parity) and inside arc_metric_deltas.
    agg: Dict = {
        "task": "arc", "checkpoint": args.checkpoint, "num_puzzles": len(puzzles),
        "max_steps": args.max_steps, "primary_metric": PRIMARY,
        "freeze_steps": ks,
        "baseline_accuracy": bootstrap_ci(base_primary),
        "baseline_arc_metrics": {},  # filled below
    }
    for lv in levels:
        node = {}
        for k in ks:
            rows = per_puzzle[f"freeze_{lv}"][k]
            node[str(k)] = {
                "delta_accuracy": bootstrap_ci([r[f"delta_{PRIMARY}"] for r in rows]),
                "arc_metric_deltas": {
                    m: bootstrap_ci([r[f"delta_{m}"] for r in rows]) for m in ARC_METRIC_KEYS},
            }
        agg[f"freeze_{lv}"] = node

    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "arc_freeze", {
            "checkpoint": args.checkpoint, "num_puzzles": len(puzzles),
            "max_steps": args.max_steps, "freeze_levels": levels,
            "freeze_steps": ks, "seed": args.seed, "primary_metric": PRIMARY},
            repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[arc_freeze] WARN _meta: {e}")

    print("\n" + "=" * 60)
    print(f"ARC freeze-after-k  Δ{PRIMARY}  (base={agg['baseline_accuracy']['mean']:.3f}, "
          f"n={len(puzzles)})")
    print(f"{'k':>4} " + " ".join(f"{'freeze_'+lv:>14}" for lv in levels))
    for k in ks:
        cells = " ".join(f"{agg['freeze_'+lv][str(k)]['delta_accuracy']['mean']:>+14.4f}" for lv in levels)
        print(f"{k:>4} {cells}")
    print(f"[arc_freeze] wrote {args.output_dir} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
