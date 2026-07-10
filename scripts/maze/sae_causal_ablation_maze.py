#!/usr/bin/env python3
"""Maze SAE causal ablation (E10) — ablate SAE features, score path-validity.

Maze analog of scripts/sae/sae_causal_ablation.py. Reuses the task-agnostic
SAE-feature ablation engine (SAEFeatureAblator: encode z_H → zero a feature →
decode → continue) and the activation-frequency ranking (select_top_features),
but scores the damage on **path-validity** (valid_sg_path) rather than Sudoku
cell accuracy.

Four conditions (same controls as Sudoku):
  sae_top_features      – ablate each of the top-k SAE features
  random_sae_features   – ablate random SAE features (matched control)
  probe_directions      – ablate each maze probe-feature direction
  random_directions     – ablate random directions (matched control)
Reports per-condition mean Δvalid_sg_path + bootstrap CI, plus t-tests
(top vs random features; probe vs random directions).

Requires a trained maze SAE + the maze z_H activation dump, and (optional) the
maze probe weights from probe_geometry_maze.py.

Usage
  python scripts/maze/sae_causal_ablation_maze.py \
      --sae_path results/maze/sae_study/sae_d2048_l10.01.pt \
      --activations_path results/maze/sae_study/activations_zH.pt \
      --probe_weights results/maze/hardened/probe_geometry/probe_weights.pt \
      --n_puzzles 300 --top_k 50 --device cuda \
      --output_dir results/maze/sae_study/causal_ablation
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
from scripts.sae.sae_causal_ablation import SAEFeatureAblator, select_top_features
from models.sae import SparseAutoencoder, TopKSparseAutoencoder
from scripts.maze.maze_common import MAZE_CHECKPOINT, maze_prediction_metrics
from scripts.maze.linear_probes_maze import _flat, _slice_preds
from scripts.maze.directed_ablation_maze import select_maze_directions

PRIMARY = "valid_sg_path"
PATH_METRICS = ["valid_sg_path", "exact_solved", "path_f1"]


def load_sae(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("activation", "relu") == "topk":
        sae = TopKSparseAutoencoder(input_dim=cfg["input_dim"], dict_size=cfg["dict_size"], k=cfg["k"])
    else:
        sae = SparseAutoencoder(input_dim=cfg["input_dim"], dict_size=cfg["dict_size"], l1_coeff=cfg["l1_coeff"])
    # strict=False: SAEs trained before the act_mean buffer was added lack that
    # key; it defaults to zeros (no mean-centering), which is correct for them.
    sae.load_state_dict(ckpt["model_state_dict"], strict=False)
    return sae.to(device).eval(), cfg


def score(cache: Dict[int, ActivationCache], label, inp):
    preds = _slice_preds(_flat(cache[max(cache.keys())].preds), label.size)
    return maze_prediction_metrics(preds, label, inp)


def cond_stats(deltas: List[float]):
    a = np.asarray(deltas, dtype=float)
    ci = bootstrap_ci(a) if a.size else {"mean": float("nan")}
    return {"mean_delta_valid_sg_path": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size),
            "bootstrap_ci": ci}


def main():
    p = argparse.ArgumentParser(description="Maze SAE causal ablation (E10, path-validity)")
    p.add_argument("--checkpoint", default=MAZE_CHECKPOINT)
    p.add_argument("--sae_path", required=True)
    p.add_argument("--activations_path", default="results/maze/sae_study/activations_zH.pt")
    p.add_argument("--probe_weights", default="results/maze/hardened/probe_geometry/probe_weights.pt")
    p.add_argument("--n_puzzles", type=int, default=300)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--n_random_features", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/maze/sae_study/causal_ablation")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sae, cfg = load_sae(args.sae_path, device)
    print(f"[maze_e10] SAE dict_size={cfg['dict_size']} l1={cfg.get('l1_coeff')}")
    top_features = select_top_features(sae, args.activations_path, top_k=args.top_k, device=device)
    pool = [f for f in range(sae.dict_size) if f not in set(top_features)]
    random_features = np.random.choice(pool, size=min(args.n_random_features, len(pool)), replace=False).tolist()

    directions = {}
    if os.path.exists(args.probe_weights):
        pw = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
        directions = select_maze_directions(pw, "z_H")
    rng = torch.Generator().manual_seed(args.seed)
    rand_dirs = {f"random_dir_{i}": (lambda v: v / v.norm())(torch.randn(sae.input_dim, generator=rng))
                 for i in range(min(3, max(1, len(directions))))}

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    ablator = SAEFeatureAblator(model, sae, device=device)
    puzzles = collect_puzzles(test_loader, device, args.n_puzzles, seed=args.seed)
    print(f"[maze_e10] {len(puzzles)} puzzles | top_k={len(top_features)} rand_feat={len(random_features)} "
          f"probe_dirs={len(directions)}")

    # R3 control: reconstruction_only = encode->decode with NO feature zeroed,
    # isolating the SAE's reconstruction error from any causal feature effect.
    cond = {"sae_top_features": [], "random_sae_features": [],
            "probe_directions": [], "random_directions": [],
            "reconstruction_only": []}
    per_puzzle = []
    t0 = time.time()
    for pi, (idx, batch) in enumerate(puzzles):
        inp, label = _flat(batch["inputs"]), _flat(batch["labels"])
        base_cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, base_cache, max_steps=args.max_steps)
        bm = score(base_cache, label, inp)
        rec = {"puzzle_idx": idx, "baseline_valid_sg_path": bm[PRIMARY]}

        # Reconstruction-only control: pass an empty feature list (nothing zeroed).
        _, ac = ablator.run_with_sae_feature_ablation(batch, [], max_steps=args.max_steps)
        cond["reconstruction_only"].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])

        for feats, key in ((top_features, "sae_top_features"), (random_features, "random_sae_features")):
            for fi in feats:
                _, ac = ablator.run_with_sae_feature_ablation(batch, [fi], max_steps=args.max_steps)
                cond[key].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])
        for name, d in directions.items():
            _, ac = ablator.run_with_direction_ablation(batch, d, max_steps=args.max_steps)
            cond["probe_directions"].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])
        for name, d in rand_dirs.items():
            _, ac = ablator.run_with_direction_ablation(batch, d, max_steps=args.max_steps)
            cond["random_directions"].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])
        per_puzzle.append(rec)
        if (pi + 1) % 25 == 0:
            print(f"  {pi+1}/{len(puzzles)} | top Δsg so far = {np.mean(cond['sae_top_features']):+.4f}")
    print(f"[maze_e10] done in {time.time()-t0:.1f}s")

    def tt(a, b):
        a, b = np.asarray(cond[a]), np.asarray(cond[b])
        if a.size < 2 or b.size < 2:
            return {"t_statistic": float("nan"), "p_value": float("nan"), "significant_0.05": False}
        t, pv = scipy_stats.ttest_ind(a, b, equal_var=False)
        return {"t_statistic": float(t), "p_value": float(pv), "significant_0.05": bool(pv < 0.05)}

    agg = {
        "n_puzzles": len(puzzles), "sae_path": args.sae_path, "metric": PRIMARY,
        "conditions": {k: cond_stats(v) for k, v in cond.items()},
        "statistical_tests": {
            "sae_top_vs_random_features": tt("sae_top_features", "random_sae_features"),
            "probe_vs_random_directions": tt("probe_directions", "random_directions"),
            "sae_top_vs_reconstruction": tt("sae_top_features", "reconstruction_only"),
        },
    }
    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(args.output_dir, "per_puzzle.jsonl"), "w") as f:
        for r in per_puzzle:
            f.write(json.dumps(r) + "\n")
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "maze_sae_causal_ablation", {
            "sae_path": args.sae_path, "n_puzzles": len(puzzles), "top_k": args.top_k,
            "n_random_features": args.n_random_features, "max_steps": args.max_steps,
            "seed": args.seed, "metric": PRIMARY}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[maze_e10] WARN _meta: {e}")

    print("\n" + "=" * 70)
    for k, v in agg["conditions"].items():
        print(f"  {k:24s} mean Δsg = {v['mean_delta_valid_sg_path']:+.4f} (n={v['n']})")
    print("  tests:", json.dumps(agg["statistical_tests"]))
    print(f"[maze_e10] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
