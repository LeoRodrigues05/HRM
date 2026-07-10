#!/usr/bin/env python3
"""Sudoku cross-puzzle activation patching — per-step donor sweep with bootstrap CIs.

Sudoku analog of ``scripts/maze/controlled_activation_patching_maze.py`` (the
``patching_full_steps`` run) and ``scripts/arc/patching_arc.py``. The paper's original
Sudoku patching numbers (steps 1-3: -53.1, steps 5-7: -61.7) are grouped point
estimates without CIs; this driver re-runs the experiment on the same validated
``ActivationPatcher`` engine as Maze/ARC, sweeping a single donor step k with
95%-bootstrap CIs, so Sudoku lands on the same per-step axis as the other two tasks
in the cross-task patching figure.

For each (source, target) pair: cache the source's activations, run the target while
overwriting its z_H (or z_L) at donor step k only, and score cell accuracy, constraint
violations, and Hamming distance to the unpatched baseline prediction.

Usage (GPU):
  python scripts/patching/patching_sudoku_full_steps.py \
      --num_pairs 250 --patch_levels H,L --patch_steps 1,2,4,6,8,10,12,15 \
      --output_dir results/patching/patching_full_steps
Smoke:
  python scripts/patching/patching_sudoku_full_steps.py --num_pairs 2 \
      --patch_steps 1,15 --output_dir results/patching/_smoke_full_steps
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
    find_checkpoint, load_model_and_dataloader, collect_puzzles, bootstrap_ci,
)
from scripts.core.activation_patching import ActivationPatcher, ActivationCache
from scripts.sae.sae_causal_ablation import count_violations, SUDOKU_CELLS

PRIMARY = "cell_acc"
METRIC_KEYS = [PRIMARY, "violated_rows", "violated_cols", "violated_boxes",
               "violated_total", "hamming_to_baseline"]


def _final_preds(cache: Dict[int, ActivationCache]) -> torch.Tensor:
    return cache[max(cache.keys())].preds[:, -SUDOKU_CELLS:]


def score(cache: Dict[int, ActivationCache], labels_tok: torch.Tensor,
          baseline_preds: torch.Tensor | None = None) -> Dict[str, float]:
    preds = _final_preds(cache)
    out = {PRIMARY: float((preds.view(-1) == labels_tok.view(-1)).float().mean().item())}
    out.update({k: float(v) for k, v in count_violations(preds).items()})
    out["hamming_to_baseline"] = (
        float((preds.view(-1) != baseline_preds.view(-1)).sum().item())
        if baseline_preds is not None else 0.0)
    return out


def main():
    p = argparse.ArgumentParser(description="Sudoku cross-puzzle activation patching (per-step)")
    p.add_argument("--checkpoint", default=None, help="default: auto-detect Sudoku checkpoint")
    p.add_argument("--num_pairs", type=int, default=250)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--patch_levels", default="H,L", help="comma list of H/L")
    p.add_argument("--patch_steps", default="1,2,4,6,8,10,12,15",
                   help="step 0 is a no-op in ACTV1 (post-reset carry is H_init/L_init)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/patching/patching_full_steps")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    ckpt = args.checkpoint or find_checkpoint()
    levels = [s.strip().upper() for s in args.patch_levels.split(",") if s.strip()]
    steps = [int(x) for x in args.patch_steps.split(",") if x.strip() != ""]

    model, test_loader, _ = load_model_and_dataloader(ckpt, device)
    patcher = ActivationPatcher(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.num_pairs * 2, seed=args.seed)
    n_pairs = len(puzzles) // 2
    print(f"[sudoku_patch] {n_pairs} pairs | levels={levels} | steps={steps} | metric={PRIMARY}")

    results: Dict[str, Dict[int, List[Dict]]] = {f"patch_{lv}": {s: [] for s in steps} for lv in levels}
    base_primary: List[float] = []
    jsonl = open(os.path.join(args.output_dir, "per_pair.jsonl"), "w")
    t0 = time.time()

    for pair_i in range(n_pairs):
        idx_s, src_batch = puzzles[2 * pair_i]
        idx_t, tgt_batch = puzzles[2 * pair_i + 1]
        labels_tok = tgt_batch["labels"][:, -SUDOKU_CELLS:]

        src_cache: Dict[int, ActivationCache] = {}
        patcher.run_and_cache_activations(src_batch, src_cache, max_steps=args.max_steps)
        tgt_cache: Dict[int, ActivationCache] = {}
        patcher.run_and_cache_activations(tgt_batch, tgt_cache, max_steps=args.max_steps)
        base_preds = _final_preds(tgt_cache)
        bm = score(tgt_cache, labels_tok)
        base_primary.append(bm[PRIMARY])
        rec = {"pair": pair_i, "src_idx": idx_s, "tgt_idx": idx_t,
               "baseline": bm, "patches": []}

        for lv in levels:
            for s in steps:
                _, cache, _ = patcher.run_with_patching(
                    tgt_batch, src_cache, patch_level=lv, patch_steps=[s],
                    max_steps=args.max_steps)
                pm = score(cache, labels_tok, baseline_preds=base_preds)
                deltas = {f"delta_{m}": pm[m] - bm[m] for m in METRIC_KEYS
                          if m != "hamming_to_baseline"}
                deltas["hamming_to_baseline"] = pm["hamming_to_baseline"]
                results[f"patch_{lv}"][s].append(deltas)
                rec["patches"].append({"level": lv, "step": s, PRIMARY: pm[PRIMARY], **deltas})
        jsonl.write(json.dumps(rec, default=float) + "\n"); jsonl.flush()

        if (pair_i + 1) % 10 == 0 or (pair_i + 1) <= 2:
            lv0 = f"patch_{levels[0]}"
            d0 = np.mean([r[f"delta_{PRIMARY}"] for r in results[lv0][steps[0]]])
            dN = np.mean([r[f"delta_{PRIMARY}"] for r in results[lv0][steps[-1]]])
            print(f"  {pair_i+1}/{n_pairs} base={np.mean(base_primary):.3f} "
                  f"{lv0} s{steps[0]}Δ={d0:+.4f} s{steps[-1]}Δ={dN:+.4f} | "
                  f"{(pair_i+1)/(time.time()-t0):.2f} pair/s", flush=True)
    jsonl.close()

    agg: Dict = {
        "task": "sudoku", "checkpoint": ckpt, "num_pairs": n_pairs,
        "max_steps": args.max_steps, "primary_metric": PRIMARY, "patch_steps": steps,
        "baseline_accuracy": bootstrap_ci(base_primary),
    }
    for lv in levels:
        node = {}
        for s in steps:
            rows = results[f"patch_{lv}"][s]
            node[str(s)] = {
                "delta_accuracy": bootstrap_ci([r[f"delta_{PRIMARY}"] for r in rows]),
                "sudoku_metric_deltas": {
                    **{m: bootstrap_ci([r[f"delta_{m}"] for r in rows])
                       for m in METRIC_KEYS if m not in (PRIMARY, "hamming_to_baseline")},
                    "hamming_to_baseline": bootstrap_ci(
                        [r["hamming_to_baseline"] for r in rows]),
                },
            }
        agg[f"patch_{lv}"] = node

    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "sudoku_patching_full_steps", {
            "checkpoint": ckpt, "num_pairs": n_pairs, "max_steps": args.max_steps,
            "patch_levels": levels, "patch_steps": steps, "seed": args.seed,
            "primary_metric": PRIMARY}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[sudoku_patch] WARN _meta: {e}")

    print("\n" + "=" * 60)
    print(f"Sudoku cross-puzzle patch  Δ{PRIMARY}  (base={agg['baseline_accuracy']['mean']:.3f}, "
          f"pairs={n_pairs})")
    print(f"{'step':>5} " + " ".join(f"{'patch_'+lv:>14}" for lv in levels))
    for s in steps:
        cells = " ".join(f"{agg['patch_'+lv][str(s)]['delta_accuracy']['mean']:>+14.4f}" for lv in levels)
        print(f"{s:>5} {cells}")
    print(f"[sudoku_patch] wrote {args.output_dir} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
