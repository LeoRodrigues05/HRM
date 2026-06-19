#!/usr/bin/env python3
"""Maze directed ablation (E9) — project out maze probe directions, score path-validity.

Maze analog of scripts/controlled/controlled_directed_ablation.py. Reuses the
task-agnostic projection mechanism (DirectionalAblator.run_with_directional_ablation)
but (a) takes its directions from the **maze** probe weights produced by
scripts/maze/probe_geometry_maze.py, and (b) scores the damage on **path-validity**
(valid_sg_path / exact_solved / path_f1), not Sudoku cell accuracy.

For every probe-feature direction (+ N random-direction controls) and every puzzle
it records the Δ in each path metric, then reports per direction: bootstrap 95% CI,
paired t + Wilcoxon (probe vs per-puzzle mean random), and Cohen's d — exactly the
controls used on the Sudoku side.

Requires: results/maze/hardened/probe_geometry/probe_weights.pt  (run probe_geometry_maze.py)

Usage
  python scripts/maze/directed_ablation_maze.py --num_puzzles 500 --z_level H \
      --probe_weights results/maze/hardened/probe_geometry/probe_weights.pt \
      --device cuda --output_dir results/maze/hardened/directed_ablation
"""
from __future__ import annotations
import os, sys, json, time, argparse
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from scipy import stats as scipy_stats

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci,
)
from scripts.core.activation_ablation import ActivationCache
from scripts.directed_ablation.e9_directed_ablation import DirectionalAblator
from scripts.maze.maze_common import MAZE_CHECKPOINT, maze_prediction_metrics
from scripts.maze.linear_probes_maze import _flat, _slice_preds

PATH_METRICS = ["valid_sg_path", "exact_solved", "path_f1", "connects_start_goal"]
PRIMARY = "valid_sg_path"


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return 0.0 if sp < 1e-10 else float((a.mean()-b.mean())/sp)


def select_maze_directions(probe_weights: dict, stream: str) -> Dict[str, torch.Tensor]:
    """Per binary feature, take the highest-val_acc step's unit weight direction."""
    best = {}
    for key, info in probe_weights.items():
        if not isinstance(info, dict) or info.get("stream") != stream:
            continue
        W = info.get("W_mean")
        if W is None:
            continue
        w = W.float()
        w = w / w.norm().clamp(min=1e-8)
        feat, va = info.get("target", key), float(info.get("val_acc_mean", 0.0) or 0.0)
        if feat not in best or va > best[feat][0]:
            best[feat] = (va, w)
    return {feat: w for feat, (va, w) in best.items()}


def score(cache: Dict[int, ActivationCache], label, inp) -> Dict[str, float]:
    preds = _slice_preds(_flat(cache[max(cache.keys())].preds), label.size)
    return maze_prediction_metrics(preds, label, inp)


def main():
    p = argparse.ArgumentParser(description="Maze directed ablation (E9, path-validity)")
    p.add_argument("--checkpoint", default=MAZE_CHECKPOINT)
    p.add_argument("--num_puzzles", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--z_level", default="H", choices=["H", "L"])
    p.add_argument("--probe_weights", default="results/maze/hardened/probe_geometry/probe_weights.pt")
    p.add_argument("--n_random_controls", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/maze/hardened/directed_ablation")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    stream = "z_H" if args.z_level == "H" else "z_L"

    if not os.path.exists(args.probe_weights):
        sys.exit(f"ERROR: {args.probe_weights} not found — run scripts/maze/probe_geometry_maze.py first.")
    pw = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
    directions = select_maze_directions(pw, stream)
    if not directions:
        sys.exit(f"ERROR: no {stream} directions in probe weights.")
    hidden = next(iter(directions.values())).shape[0]
    print(f"[maze_e9] {len(directions)} probe directions ({stream}), hidden={hidden}")

    rng = torch.Generator().manual_seed(args.seed)
    randoms = {}
    for i in range(args.n_random_controls):
        rd = torch.randn(hidden, generator=rng)
        randoms[f"random_{i}"] = rd / rd.norm()

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    ablator = DirectionalAblator(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
    print(f"[maze_e9] collected {len(puzzles)} puzzles")

    all_dirs = {**directions, **randoms}
    results_by_dir: Dict[str, List[Dict]] = {name: [] for name in all_dirs}
    t0 = time.time()
    for i, (idx, batch) in enumerate(puzzles):  # baseline once per puzzle, then every direction
        inp, label = _flat(batch["inputs"]), _flat(batch["labels"])
        base_cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, base_cache, max_steps=args.max_steps)
        bm = score(base_cache, label, inp)
        for name, direction in all_dirs.items():
            _, abl_cache = ablator.run_with_directional_ablation(
                batch, direction, ablate_level=args.z_level, ablate_steps=None, max_steps=args.max_steps)
            am = score(abl_cache, label, inp)
            results_by_dir[name].append({"puzzle_idx": idx,
                **{f"delta_{k}": am[k] - bm[k] for k in PATH_METRICS}, "baseline_valid_sg_path": bm[PRIMARY]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(puzzles)} | top probe Δ{PRIMARY} so far = "
                  f"{np.mean([results_by_dir[next(iter(directions))][j]['delta_'+PRIMARY] for j in range(len(results_by_dir[next(iter(directions))]))]):+.4f}")
    for name in all_dirs:
        print(f"  [{name}] mean Δ{PRIMARY} = {np.mean([r['delta_'+PRIMARY] for r in results_by_dir[name]]):+.4f}")
    print(f"[maze_e9] ablations done in {time.time()-t0:.1f}s")

    # Stats: each probe direction vs per-puzzle mean of random controls
    analysis = {}
    n = len(puzzles)
    for name in directions:
        probe = [r["delta_" + PRIMARY] for r in results_by_dir[name]]
        rand_pp = [float(np.mean([results_by_dir[rn][pi]["delta_" + PRIMARY] for rn in randoms])) for pi in range(n)]
        try:
            t_stat, p_val = scipy_stats.ttest_rel(probe, rand_pp)
        except Exception:
            t_stat, p_val = float("nan"), float("nan")
        diffs = np.asarray(probe) - np.asarray(rand_pp)
        try:
            w_stat, w_p = scipy_stats.wilcoxon(probe, rand_pp) if np.any(diffs != 0) else (float("nan"), 1.0)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        analysis[name] = {
            "probe_delta_valid_sg_path": bootstrap_ci(probe),
            "random_control_delta_valid_sg_path": bootstrap_ci(rand_pp),
            "probe_delta_exact_solved": bootstrap_ci([r["delta_exact_solved"] for r in results_by_dir[name]]),
            "probe_delta_path_f1": bootstrap_ci([r["delta_path_f1"] for r in results_by_dir[name]]),
            "paired_t_stat": float(t_stat), "paired_p_value": float(p_val),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p_value": float(w_p),
            "cohens_d": cohens_d(probe, rand_pp),
            "significant_at_005": bool(p_val < 0.05) if p_val == p_val else False,
        }

    with open(os.path.join(args.output_dir, "per_direction_results.json"), "w") as f:
        json.dump(results_by_dir, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else
                  bool(o) if isinstance(o, np.bool_) else str(o))
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "maze_directed_ablation", {
            "num_puzzles": n, "z_level": args.z_level, "max_steps": args.max_steps,
            "probe_weights": args.probe_weights, "n_random_controls": args.n_random_controls,
            "seed": args.seed, "metric": PRIMARY}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[maze_e9] WARN _meta: {e}")

    print("\n" + "=" * 80)
    print(f"{'direction':24s} {'probe Δsg':>12s} {'random Δsg':>12s} {'Cohen d':>8s} {'p':>10s}")
    for name, a in analysis.items():
        print(f"{name:24s} {a['probe_delta_valid_sg_path']['mean']:>+12.4f} "
              f"{a['random_control_delta_valid_sg_path']['mean']:>+12.4f} "
              f"{a['cohens_d']:>8.3f} {a['paired_p_value']:>10.5f}")
    print(f"[maze_e9] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
