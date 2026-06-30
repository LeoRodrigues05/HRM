#!/usr/bin/env python3
"""ARC SAE causal ablation — ablate SAE features, score colour-cell accuracy.

ARC analog of ``scripts/maze/sae_causal_ablation_maze.py`` /
``scripts/sae/sae_causal_ablation.py``. Reuses the task-agnostic SAE-feature ablation
engine (``SAEFeatureAblator``: encode z_H -> zero a feature -> decode -> continue) and
the activation-frequency ranking (``select_top_features``), but takes its probe
directions from the **ARC** probe weights and scores the damage on
``colour_cell_acc`` (the ARC task metric; exact-grid solving is ~0 for the frozen-core
checkpoint).

Four conditions (same controls as Sudoku/Maze):
  sae_top_features    – ablate each of the top-k SAE features (by activation freq)
  random_sae_features – ablate random SAE features (matched control)
  probe_directions    – ablate each ARC probe-feature direction
  random_directions   – ablate random unit directions (matched control)
Finding 3 prediction: top-k SAE ≈ random-k SAE, and both ≪ full-z_H ablation
(computation is distributed; no mono-semantic feature carries it).

Requires: a trained ARC SAE + z_H activation dump (sae_collect_activations_arc.py +
scripts/sae/sae_train.py) and the ARC probe weights
(results/arc/hardened/linear_probes/probe_weights.pt).

Usage (GPU, via srun on ws-l3-019):
  python scripts/arc/sae_causal_ablation_arc.py \
      --sae_path results/arc/sae_study/sae_d2048_l10.01.pt \
      --activations_path results/arc/sae_study/activations_zH.pt \
      --probe_weights results/arc/hardened/linear_probes/probe_weights.pt \
      --n_puzzles 300 --top_k 50 --output_dir results/arc/sae_study/causal_ablation
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
from scripts.arc.arc_common import arc_prediction_metrics
from scripts.arc.directed_ablation_arc import select_arc_directions
from scripts.maze.linear_probes_maze import _flat, _slice_preds

ARC_ADAPTED_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "arc2-adapted-evalonly", "step_7391")
PRIMARY = "colour_cell_acc"


def load_sae(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("activation", "relu") == "topk":
        sae = TopKSparseAutoencoder(input_dim=cfg["input_dim"], dict_size=cfg["dict_size"], k=cfg["k"])
    else:
        sae = SparseAutoencoder(input_dim=cfg["input_dim"], dict_size=cfg["dict_size"], l1_coeff=cfg["l1_coeff"])
    sae.load_state_dict(ckpt["model_state_dict"])
    return sae.to(device).eval(), cfg


def score(cache: Dict[int, ActivationCache], label, inp):
    preds = _slice_preds(_flat(cache[max(cache.keys())].preds), label.size)
    return arc_prediction_metrics(preds, label, inp)


def cond_stats(deltas: List[float]):
    a = np.asarray(deltas, dtype=float)
    return {f"mean_delta_{PRIMARY}": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size),
            "bootstrap_ci": bootstrap_ci(a) if a.size else {"mean": float("nan")}}


def main():
    p = argparse.ArgumentParser(description="ARC SAE causal ablation (colour-cell acc)")
    p.add_argument("--checkpoint", default=ARC_ADAPTED_CKPT)
    p.add_argument("--sae_path", required=True)
    p.add_argument("--activations_path", default="results/arc/sae_study/activations_zH.pt")
    p.add_argument("--probe_weights", default="results/arc/hardened/linear_probes/probe_weights.pt")
    p.add_argument("--n_puzzles", type=int, default=300)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--n_random_features", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/arc/sae_study/causal_ablation")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sae, cfg = load_sae(args.sae_path, device)
    print(f"[arc_sae] SAE dict_size={cfg['dict_size']} l1={cfg.get('l1_coeff')} input_dim={cfg['input_dim']}")
    top_features = select_top_features(sae, args.activations_path, top_k=args.top_k, device=device)
    pool = [f for f in range(sae.dict_size) if f not in set(top_features)]
    random_features = np.random.choice(pool, size=min(args.n_random_features, len(pool)), replace=False).tolist()

    directions = {}
    if os.path.exists(args.probe_weights):
        pw = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
        directions = select_arc_directions(pw, "z_H")
    rng = torch.Generator().manual_seed(args.seed)
    rand_dirs = {f"random_dir_{i}": (lambda v: v / v.norm())(torch.randn(sae.input_dim, generator=rng))
                 for i in range(min(3, max(1, len(directions))))}

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    ablator = SAEFeatureAblator(model, sae, device=device)
    puzzles = collect_puzzles(test_loader, device, args.n_puzzles, seed=args.seed)
    print(f"[arc_sae] {len(puzzles)} puzzles | top_k={len(top_features)} rand_feat={len(random_features)} "
          f"probe_dirs={len(directions)}")

    cond = {"sae_top_features": [], "random_sae_features": [],
            "probe_directions": [], "random_directions": []}
    base_primary: List[float] = []
    t0 = time.time()
    for pi, (idx, batch) in enumerate(puzzles):
        inp, label = _flat(batch["inputs"]), _flat(batch["labels"])
        base_cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, base_cache, max_steps=args.max_steps)
        bm = score(base_cache, label, inp)
        base_primary.append(bm[PRIMARY])
        for feats, key in ((top_features, "sae_top_features"), (random_features, "random_sae_features")):
            for fi in feats:
                _, ac = ablator.run_with_sae_feature_ablation(batch, [int(fi)], max_steps=args.max_steps)
                cond[key].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])
        for _name, d in directions.items():
            _, ac = ablator.run_with_direction_ablation(batch, d, max_steps=args.max_steps)
            cond["probe_directions"].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])
        for _name, d in rand_dirs.items():
            _, ac = ablator.run_with_direction_ablation(batch, d, max_steps=args.max_steps)
            cond["random_directions"].append(score(ac, label, inp)[PRIMARY] - bm[PRIMARY])
        if (pi + 1) % 25 == 0:
            print(f"  {pi+1}/{len(puzzles)} | top Δ{PRIMARY} so far = {np.mean(cond['sae_top_features']):+.4f}")
    print(f"[arc_sae] done in {time.time()-t0:.1f}s")

    def tt(a, b):
        a, b = np.asarray(cond[a]), np.asarray(cond[b])
        if a.size < 2 or b.size < 2:
            return {"t_statistic": float("nan"), "p_value": float("nan"), "significant_0.05": False}
        t, pv = scipy_stats.ttest_ind(a, b, equal_var=False)
        return {"t_statistic": float(t), "p_value": float(pv), "significant_0.05": bool(pv < 0.05)}

    agg = {
        "task": "arc", "n_puzzles": len(puzzles), "sae_path": args.sae_path,
        "checkpoint": args.checkpoint, "metric": PRIMARY,
        "baseline_accuracy": bootstrap_ci(base_primary),
        "conditions": {k: cond_stats(v) for k, v in cond.items()},
        "statistical_tests": {
            "sae_top_vs_random_features": tt("sae_top_features", "random_sae_features"),
            "probe_vs_random_directions": tt("probe_directions", "random_directions"),
        },
    }
    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "arc_sae_causal_ablation", {
            "sae_path": args.sae_path, "checkpoint": args.checkpoint, "n_puzzles": len(puzzles),
            "top_k": args.top_k, "n_random_features": args.n_random_features,
            "max_steps": args.max_steps, "seed": args.seed, "metric": PRIMARY}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[arc_sae] WARN _meta: {e}")

    print("\n" + "=" * 70)
    print(f"ARC SAE causal ablation  Δ{PRIMARY}  (base={agg['baseline_accuracy']['mean']:.3f}, n={len(puzzles)})")
    for k, v in agg["conditions"].items():
        print(f"  {k:24s} mean Δ = {v['mean_delta_'+PRIMARY]:+.4f} (n={v['n']})")
    print("  tests:", json.dumps(agg["statistical_tests"]))
    print(f"[arc_sae] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
