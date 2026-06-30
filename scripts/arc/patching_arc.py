#!/usr/bin/env python3
"""ARC cross-puzzle activation patching — transplant a donor z_H/z_L, score the damage.

ARC analog of the Sudoku cross-puzzle patch (paper Fig. 4) and the maze
``patching_full_steps`` run. Reuses the task-agnostic ``ActivationPatcher`` harness:
for each (source, target) puzzle pair it caches the source's activations, then runs
the target forward pass while overwriting the target's z_H (or z_L) at **all grid
cell positions** at a single donor step k, sweeping k. Damage is scored on ARC
structure (``colour_cell_acc`` primary).

Interpretation (Finding 1): if a foreign z_H is catastrophic at every step, z_H is a
continuously-rewritten puzzle-specific state (Sudoku); if it only matters at the final
readout step, z_H is a static plan read out late (Maze). This run places ARC on that
spectrum, mirroring the freezing result.

Primary metric ``colour_cell_acc`` is also stored under ``delta_accuracy`` for
cross-task figure uniformity; all ARC metrics under ``arc_metric_deltas``.

Usage (GPU, via srun on ws-l3-019):
  python scripts/arc/patching_arc.py \
      --checkpoint checkpoints/arc2-adapted-evalonly/step_7391 \
      --num_pairs 100 --patch_levels H,L --output_dir results/arc/patching_full_steps
Smoke:
  python scripts/arc/patching_arc.py --num_pairs 4 --patch_steps 0,8,15 \
      --output_dir results/arc_smoke/patching_full_steps
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
from scripts.core.activation_patching import ActivationPatcher, ActivationCache
from scripts.arc.arc_common import (
    ARC_METRIC_KEYS, arc_prediction_metrics, cell_positions, get_puzzle_emb_len,
)
from scripts.maze.linear_probes_maze import _flat, _slice_preds

ARC_ADAPTED_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "arc2-adapted-evalonly", "step_7391")
PRIMARY = "colour_cell_acc"


def score(cache: Dict[int, ActivationCache], label, inp) -> Dict[str, float]:
    preds = _slice_preds(_flat(cache[max(cache.keys())].preds), label.size)
    return arc_prediction_metrics(preds, label, inp)


def main():
    p = argparse.ArgumentParser(description="ARC cross-puzzle activation patching")
    p.add_argument("--checkpoint", default=ARC_ADAPTED_CKPT)
    p.add_argument("--num_pairs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--patch_levels", default="H,L", help="comma list of H/L")
    p.add_argument("--patch_steps", default="0,2,4,6,8,10,12,15")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/arc/patching_full_steps")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    levels = [s.strip().upper() for s in args.patch_levels.split(",") if s.strip()]
    steps = [int(x) for x in args.patch_steps.split(",") if x.strip() != ""]

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    all_positions = cell_positions(pel).tolist()    # every 30x30 grid cell
    patcher = ActivationPatcher(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.num_pairs * 2, seed=args.seed)
    n_pairs = len(puzzles) // 2
    print(f"[arc_patch] {n_pairs} pairs | levels={levels} | steps={steps} | "
          f"pel={pel} | n_positions={len(all_positions)} | metric={PRIMARY}")

    # results[level][step] -> list of per-pair delta dicts
    results: Dict[str, Dict[int, List[Dict]]] = {f"patch_{lv}": {s: [] for s in steps} for lv in levels}
    base_primary: List[float] = []
    jsonl = open(os.path.join(args.output_dir, "per_pair.jsonl"), "w")
    t0 = time.time()

    for pair_i in range(n_pairs):
        idx_s, src_batch = puzzles[2 * pair_i]
        idx_t, tgt_batch = puzzles[2 * pair_i + 1]
        inp, label = _flat(tgt_batch["inputs"]), _flat(tgt_batch["labels"])

        src_cache: Dict[int, ActivationCache] = {}
        patcher.run_and_cache_activations(src_batch, src_cache, max_steps=args.max_steps)
        tgt_cache: Dict[int, ActivationCache] = {}
        patcher.run_and_cache_activations(tgt_batch, tgt_cache, max_steps=args.max_steps)
        bm = score(tgt_cache, label, inp)
        base_primary.append(bm[PRIMARY])
        rec = {"pair": pair_i, "src_idx": idx_s, "tgt_idx": idx_t,
               "baseline": {k: bm[k] for k in ARC_METRIC_KEYS}, "patches": []}

        for lv in levels:
            for s in steps:
                _, cache, _ = patcher.run_with_patching(
                    tgt_batch, src_cache, patch_level=lv, patch_steps=[s],
                    patch_positions=all_positions, max_steps=args.max_steps)
                pm = score(cache, label, inp)
                deltas = {f"delta_{m}": pm[m] - bm[m] for m in ARC_METRIC_KEYS}
                results[f"patch_{lv}"][s].append(deltas)
                rec["patches"].append({"level": lv, "step": s, PRIMARY: pm[PRIMARY], **deltas})
        jsonl.write(json.dumps(rec, default=float) + "\n"); jsonl.flush()

        if (pair_i + 1) % 10 == 0 or (pair_i + 1) <= 2:
            lv0 = f"patch_{levels[0]}"
            d0 = np.mean([r[f"delta_{PRIMARY}"] for r in results[lv0][steps[0]]])
            dN = np.mean([r[f"delta_{PRIMARY}"] for r in results[lv0][steps[-1]]])
            print(f"  {pair_i+1}/{n_pairs} base={np.mean(base_primary):.3f} "
                  f"{lv0} s{steps[0]}Δ={d0:+.4f} s{steps[-1]}Δ={dN:+.4f} | "
                  f"{(pair_i+1)/(time.time()-t0):.2f} pair/s")
    jsonl.close()

    agg: Dict = {
        "task": "arc", "checkpoint": args.checkpoint, "num_pairs": n_pairs,
        "max_steps": args.max_steps, "primary_metric": PRIMARY, "patch_steps": steps,
        "baseline_accuracy": bootstrap_ci(base_primary),
    }
    for lv in levels:
        node = {}
        for s in steps:
            rows = results[f"patch_{lv}"][s]
            node[str(s)] = {
                "delta_accuracy": bootstrap_ci([r[f"delta_{PRIMARY}"] for r in rows]),
                "arc_metric_deltas": {
                    m: bootstrap_ci([r[f"delta_{m}"] for r in rows]) for m in ARC_METRIC_KEYS},
            }
        agg[f"patch_{lv}"] = node

    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "arc_patching", {
            "checkpoint": args.checkpoint, "num_pairs": n_pairs, "max_steps": args.max_steps,
            "patch_levels": levels, "patch_steps": steps, "seed": args.seed,
            "primary_metric": PRIMARY, "n_positions": len(all_positions)},
            repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[arc_patch] WARN _meta: {e}")

    print("\n" + "=" * 60)
    print(f"ARC cross-puzzle patch  Δ{PRIMARY}  (base={agg['baseline_accuracy']['mean']:.3f}, "
          f"pairs={n_pairs})")
    print(f"{'step':>5} " + " ".join(f"{'patch_'+lv:>14}" for lv in levels))
    for s in steps:
        cells = " ".join(f"{agg['patch_'+lv][str(s)]['delta_accuracy']['mean']:>+14.4f}" for lv in levels)
        print(f"{s:>5} {cells}")
    print(f"[arc_patch] wrote {args.output_dir} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
