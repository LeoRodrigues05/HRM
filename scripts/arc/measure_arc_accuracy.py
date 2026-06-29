#!/usr/bin/env python3
"""Measure HRM ARC-2 test accuracy on the (rebuilt) dataset.

Diagnostic for the transductive puzzle-embedding alignment question: rebuilding
the ARC dataset can differ from the checkpoint by a few puzzle identifiers, which
shifts the learned per-puzzle embeddings. This script runs the model on N test
puzzles and reports exact-match / token / colour accuracy so we can see how much
that misalignment actually costs before running the full interpretability suite.

Usage:
  python scripts/arc/measure_arc_accuracy.py --num_puzzles 100 --device cuda
"""
from __future__ import annotations
import os, sys, json, argparse
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles,
)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.arc.arc_common import ARC_CHECKPOINT, arc_prediction_metrics, ARC_METRIC_KEYS
from scripts.maze.linear_probes_maze import _flat, _slice_preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=ARC_CHECKPOINT)
    p.add_argument("--num_puzzles", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="results/arc/diagnostics")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    ablator = ActivationAblator(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.num_puzzles)
    print(f"[measure_arc] collected {len(puzzles)} test puzzles, device={device}")

    rows: List[Dict[str, float]] = []
    for i, (idx, batch) in enumerate(puzzles):
        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=args.max_steps)
        if not cache:
            continue
        label = _flat(batch["labels"])
        preds = _slice_preds(_flat(cache[max(cache.keys())].preds), label.size)
        rows.append(arc_prediction_metrics(preds, label))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(puzzles)} | running exact={np.mean([r['exact_solved'] for r in rows]):.3f}")

    summary = {k: float(np.mean([r[k] for r in rows])) for k in ARC_METRIC_KEYS}
    summary["n_puzzles"] = len(rows)
    with open(os.path.join(args.output_dir, "arc_accuracy.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"ARC-2 test accuracy over {len(rows)} puzzles:")
    for k in ARC_METRIC_KEYS:
        print(f"  {k:22s} {summary[k]:.4f}")
    print("=" * 60)
    print("Interpretation: exact_solved near the published ~40% => embeddings align;")
    print("near 0 => the rebuilt-dataset identifier shift broke transductive solving.")
    print(f"[measure_arc] wrote {args.output_dir}/arc_accuracy.json")


if __name__ == "__main__":
    main()
