"""Maze-specific activation patching with spatial groupings.

For each (source, target) puzzle pair, patch z_H or z_L from source into target
ONLY at a spatial group of positions derived from the TARGET maze:

  - on_path  : target's optimal-path cells (PATH/START/GOAL)
  - off_path : passable target cells NOT on the optimal path
  - near_S   : passable target cells within BFS distance D of start
  - near_G   : passable target cells within BFS distance D of goal
  - all      : every maze cell (control)

For each group x patch_level (H/L) x patch_step in {early, mid, late}, measure
delta token accuracy plus maze-structural metrics vs the target baseline.

Outputs:
  results/maze/patching_spatial/per_pair.jsonl
  results/maze/patching_spatial/aggregate.json

Usage:
    python scripts/maze/controlled_activation_patching_maze.py \
        --num_pairs 20 --patch_steps 4,8,12
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci,
)
from scripts.core.activation_patching import ActivationPatcher, ActivationCache, compute_metrics
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, build_spatial_masks,
    cell_positions, maze_batch_metrics, MAZE_METRIC_KEYS,
    maze_layout_metrics, MAZE_LAYOUT_METRIC_KEYS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--num_pairs", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--patch_steps", type=str, default="4,8,12")
    p.add_argument("--near_dist", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="results/maze/patching_spatial")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    patch_steps = [int(x) for x in args.patch_steps.split(",") if x.strip()]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[patch_maze] device={device} patch_steps={patch_steps}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    print(f"[patch_maze] puzzle_emb_len={pel}")

    patcher = ActivationPatcher(model, device=device)

    n_needed = args.num_pairs * 2
    puzzles = collect_puzzles(test_loader, device, n_needed)
    print(f"[patch_maze] collected {len(puzzles)} puzzles ({args.num_pairs} pairs)")

    all_positions = cell_positions(pel).tolist()  # control: all maze cells

    per_pair_path = os.path.join(args.output_dir, "per_pair.jsonl")
    out_f = open(per_pair_path, "w")

    # results[group][level][step] -> list of delta_acc across pairs
    results: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    baseline_target_accs: List[float] = []
    baseline_maze_metrics: Dict[str, List[float]] = {k: [] for k in MAZE_METRIC_KEYS}
    baseline_layout_metrics: Dict[str, List[float]] = {k: [] for k in MAZE_LAYOUT_METRIC_KEYS}
    metric_deltas: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}
    layout_metric_deltas: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}

    t0 = time.time()
    for pair_i in range(args.num_pairs):
        idx_s, src_batch = puzzles[2 * pair_i]
        idx_t, tgt_batch = puzzles[2 * pair_i + 1]

        # Cache source activations
        src_cache: Dict[int, ActivationCache] = {}
        patcher.target_cache = {}  # reset
        # Use the cache helper from ActivationPatcher's parent semantics via run_and_cache_activations.
        # ActivationPatcher.run_and_cache_activations exists too.
        patcher.run_and_cache_activations(src_batch, src_cache, max_steps=args.max_steps)

        # Baseline target
        tgt_cache: Dict[int, ActivationCache] = {}
        tgt_out = patcher.run_and_cache_activations(tgt_batch, tgt_cache, max_steps=args.max_steps)
        tgt_preds = tgt_out["logits"].argmax(-1)
        base_acc = compute_metrics(tgt_preds, tgt_batch["labels"])["accuracy"]
        base_maze = maze_batch_metrics(tgt_preds, tgt_batch["labels"], tgt_batch["inputs"])
        base_layout = maze_layout_metrics(tgt_preds, tgt_batch["inputs"], src_batch["inputs"])
        baseline_target_accs.append(base_acc)
        for metric in MAZE_METRIC_KEYS:
            baseline_maze_metrics[metric].append(base_maze[metric])
        for metric in MAZE_LAYOUT_METRIC_KEYS:
            baseline_layout_metrics[metric].append(base_layout[metric])

        masks = build_spatial_masks(tgt_batch, pel, near_dist=args.near_dist)
        masks["all"] = all_positions

        pair_rec = {
            "pair": pair_i, "src_idx": idx_s, "tgt_idx": idx_t,
            "baseline_target_acc": base_acc,
            "baseline_target_maze_metrics": base_maze,
            "baseline_target_layout_metrics": base_layout,
            "group_sizes": {g: len(pos) for g, pos in masks.items()},
            "patches": [],
        }

        for group, positions in masks.items():
            if len(positions) == 0:
                continue
            for level in ("H", "L"):
                for step in patch_steps:
                    out, _, _ = patcher.run_with_patching(
                        tgt_batch, src_cache,
                        patch_level=level,
                        patch_steps=[step],
                        patch_positions=positions,
                        max_steps=args.max_steps,
                    )
                    preds = out["logits"].argmax(-1)
                    acc = compute_metrics(preds, tgt_batch["labels"])["accuracy"]
                    maze_metrics = maze_batch_metrics(preds, tgt_batch["labels"], tgt_batch["inputs"])
                    layout_metrics = maze_layout_metrics(preds, tgt_batch["inputs"], src_batch["inputs"])
                    delta = acc - base_acc
                    maze_delta = {
                        metric: maze_metrics[metric] - base_maze[metric]
                        for metric in MAZE_METRIC_KEYS
                    }
                    layout_delta = {
                        metric: layout_metrics[metric] - base_layout[metric]
                        for metric in MAZE_LAYOUT_METRIC_KEYS
                    }
                    pair_rec["patches"].append({
                        "group": group, "level": level, "step": step,
                        "acc": acc, "delta_acc": delta,
                        "maze_metrics": maze_metrics,
                        "maze_metric_deltas": maze_delta,
                        "layout_metrics": layout_metrics,
                        "layout_metric_deltas": layout_delta,
                    })
                    results.setdefault(group, {}).setdefault(level, {}).setdefault(step, []).append(delta)
                    for metric, dval in maze_delta.items():
                        metric_deltas.setdefault(metric, {}).setdefault(group, {}).setdefault(level, {}).setdefault(step, []).append(dval)
                    for metric, dval in layout_delta.items():
                        layout_metric_deltas.setdefault(metric, {}).setdefault(group, {}).setdefault(level, {}).setdefault(step, []).append(dval)

        out_f.write(json.dumps(pair_rec) + "\n")
        out_f.flush()
        print(f"  [{pair_i+1}/{args.num_pairs}] base_acc={base_acc:.3f}  elapsed={time.time()-t0:.1f}s")
    out_f.close()

    # Aggregate
    agg = {
        "num_pairs": args.num_pairs,
        "patch_steps": patch_steps,
        "near_dist": args.near_dist,
        "baseline_target_acc": bootstrap_ci(baseline_target_accs),
        "baseline_target_maze_metrics": {
            metric: bootstrap_ci(vals)
            for metric, vals in baseline_maze_metrics.items()
            if vals
        },
        "baseline_target_layout_metrics": {
            metric: bootstrap_ci(vals)
            for metric, vals in baseline_layout_metrics.items()
            if vals
        },
        "by_group_level_step": {},
        "maze_metric_deltas_by_group_level_step": {},
        "layout_metric_deltas_by_group_level_step": {},
    }
    for group, lv_dict in results.items():
        agg["by_group_level_step"][group] = {}
        for level, st_dict in lv_dict.items():
            agg["by_group_level_step"][group][level] = {}
            for step, deltas in st_dict.items():
                agg["by_group_level_step"][group][level][str(step)] = bootstrap_ci(deltas)

    for metric, group_dict in metric_deltas.items():
        agg["maze_metric_deltas_by_group_level_step"][metric] = {}
        for group, lv_dict in group_dict.items():
            agg["maze_metric_deltas_by_group_level_step"][metric][group] = {}
            for level, st_dict in lv_dict.items():
                agg["maze_metric_deltas_by_group_level_step"][metric][group][level] = {}
                for step, deltas in st_dict.items():
                    agg["maze_metric_deltas_by_group_level_step"][metric][group][level][str(step)] = bootstrap_ci(deltas)

    for metric, group_dict in layout_metric_deltas.items():
        agg["layout_metric_deltas_by_group_level_step"][metric] = {}
        for group, lv_dict in group_dict.items():
            agg["layout_metric_deltas_by_group_level_step"][metric][group] = {}
            for level, st_dict in lv_dict.items():
                agg["layout_metric_deltas_by_group_level_step"][metric][group][level] = {}
                for step, deltas in st_dict.items():
                    agg["layout_metric_deltas_by_group_level_step"][metric][group][level][str(step)] = bootstrap_ci(deltas)

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[patch_maze] wrote {agg_path}")


if __name__ == "__main__":
    main()
