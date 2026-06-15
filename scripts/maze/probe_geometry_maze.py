#!/usr/bin/env python3
"""Maze probe-weight geometry — replication of the Sudoku E8 geometric analysis.

Fits a single-direction (BCE) linear probe per (stream, ACT step, binary maze
feature) under the same puzzle-disjoint split + seed ensemble as the hardened
maze probes, captures each probe's weight *direction*, then analyses the
geometry of those directions:

  - pairwise cosine similarity between feature directions at the same (stream,
    step), with a seed-ensemble mean ± 95% t-CI,
  - PCA of the stacked direction matrix: PC1 explained variance (how low-
    dimensional the feature subspace is) with a seed CI,
  - per-direction weight norms.

This is the maze analog of ``results/probes/e8_constraint_probes/geometric_analysis.json``.
Only binary structural features have a single direction, so regression targets
(distances, degree) are skipped.

Usage
  python scripts/maze/probe_geometry_maze.py --num_puzzles 1000 \
      --steps 0,1,2,4,8,15 --streams z_H,z_L --seeds 0,1,2,3,4 --device cuda \
      --output_dir results/maze/hardened/probe_geometry
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import itertools
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch import nn

from scripts.controlled.controlled_common import load_model_and_dataloader
from scripts.maze.maze_common import MAZE_CHECKPOINT, get_puzzle_emb_len
from scripts.maze.linear_probes_maze import (
    LOCAL_TARGETS, collect_probe_data, _build_local_dataset,
    puzzle_disjoint_split, _standardize, _stable_seed, _mean_ci,
)

# Binary structural features that define a single direction.
GEOMETRY_FEATURES = [t for t, k in LOCAL_TARGETS.items() if k == "binary"]


def _fit_direction(x, y, tr_idx, te_idx, *, epochs, lr, fit_seed, device) -> Tuple[torch.Tensor, float]:
    """Fit a 1-output BCE linear probe; return (unit weight direction [D], val_acc)."""
    tr = torch.from_numpy(np.asarray(tr_idx, np.int64)); te = torch.from_numpy(np.asarray(te_idx, np.int64))
    xt, xv = x[tr].float(), x[te].float()
    yt, yv = y[tr].float().view(-1, 1), y[te].float().view(-1, 1)
    if torch.unique(y[tr]).numel() < 2:
        return None, float("nan")
    xt, xv = _standardize(xt, xv)
    xt, xv, yt, yv = xt.to(device), xv.to(device), yt.to(device), yv.to(device)
    torch.manual_seed(fit_seed)
    model = nn.Linear(xt.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        crit(model(xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(xv)) > 0.5).float() == yv).float().mean().item()
        w = model.weight.detach().cpu()[0]
    return w, float(acc)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=MAZE_CHECKPOINT)
    ap.add_argument("--num_puzzles", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--steps", default="0,1,2,4,8,15")
    ap.add_argument("--streams", default="z_H,z_L")
    ap.add_argument("--positions_per_sample", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_device", default="auto")
    ap.add_argument("--output_dir", default="results/maze/hardened/probe_geometry")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    tdev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.train_device == "auto" \
        else torch.device(args.train_device)
    streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    _, local_blocks, _ = collect_probe_data(
        model=model, test_loader=test_loader, device=device, num_puzzles=args.num_puzzles,
        steps=steps, max_steps=args.max_steps, positions_per_sample=args.positions_per_sample,
        puzzle_emb_len=pel, seed=args.seed)
    blocks_by_step = {s: [b for b in local_blocks if b["step"] == s] for s in steps}

    # Fit a direction per (stream, step, feature, seed) ----------------------
    probe_weights: Dict[str, Dict] = {}   # key -> {W_mean[D], W_per_seed:[D], val_score, ...}
    for stream in streams:
        for step in steps:
            blocks = blocks_by_step.get(step, [])
            if not blocks:
                continue
            for feat in GEOMETRY_FEATURES:
                x, y, groups = _build_local_dataset(blocks, stream, feat)
                if torch.unique(y).numel() < 2:
                    continue
                per_seed_w, per_seed_acc = [], []
                for s in seeds:
                    split_seed = _stable_seed(args.seed + s, "geom", stream, str(step), feat)
                    tr, te = puzzle_disjoint_split(groups, args.val_frac, split_seed)
                    w, acc = _fit_direction(x, y, tr, te, epochs=args.epochs, lr=args.lr,
                                            fit_seed=split_seed, device=tdev)
                    if w is not None:
                        per_seed_w.append(w); per_seed_acc.append(acc)
                if not per_seed_w:
                    continue
                W = torch.stack(per_seed_w, 0).mean(0)
                m, lo, hi, sd = _mean_ci(per_seed_acc)
                probe_weights[f"{stream}_step{step}_{feat}"] = {
                    "W_mean": W, "W_per_seed": per_seed_w, "stream": stream, "step": step,
                    "target": feat, "val_acc_mean": m, "val_acc_ci": [lo, hi], "n_seeds": len(per_seed_w),
                    "W_norm": float(W.norm().item()),
                }
            print(f"[geom] fitted {stream} step {step}: "
                  f"{sum(1 for k in probe_weights if k.startswith(f'{stream}_step{step}_'))} directions")

    # Geometry: cosines + PCA per (stream, step) -----------------------------
    groups: Dict[str, Dict[str, Dict]] = {}
    for key, v in probe_weights.items():
        groups.setdefault(f"{v['stream']}_step{v['step']}", {})[v["target"]] = v

    cosines, cosines_ensemble, pca_pc1, weight_norms = [], [], [], []
    for gk, feats in groups.items():
        names = sorted(feats)
        for a, b in itertools.combinations(names, 2):
            cosines.append({"group": gk, "probe_a": a, "probe_b": b,
                            "cosine_mean": round(_cos(feats[a]["W_mean"], feats[b]["W_mean"]), 5)})
            ns = min(len(feats[a]["W_per_seed"]), len(feats[b]["W_per_seed"]))
            per = [_cos(feats[a]["W_per_seed"][i], feats[b]["W_per_seed"][i]) for i in range(ns)]
            cm, clo, chi, csd = _mean_ci(per)
            cosines_ensemble.append({"group": gk, "probe_a": a, "probe_b": b,
                                     "cosine_mean": round(cm, 5), "cosine_ci_lower": round(clo, 5),
                                     "cosine_ci_upper": round(chi, 5), "n_seeds": ns,
                                     "cosine_per_seed": [round(x, 5) for x in per]})
        for n in names:
            weight_norms.append({"group": gk, "target": n, "W_norm": round(feats[n]["W_norm"], 5),
                                 "val_acc_mean": round(feats[n]["val_acc_mean"], 5)})
        if len(names) >= 2:
            ns = min(len(feats[n]["W_per_seed"]) for n in names)
            pc1_per = []
            for i in range(ns):
                mat = torch.stack([feats[n]["W_per_seed"][i] for n in names], 0)
                mat = mat - mat.mean(0, keepdim=True)
                try:
                    S = torch.linalg.svdvals(mat)
                    pc1_per.append(float(((S ** 2) / (S ** 2).sum())[0].item()))
                except Exception:
                    pass
            if pc1_per:
                m, lo, hi, sd = _mean_ci(pc1_per)
                pca_pc1.append({"group": gk, "n_directions": len(names),
                                "pc1_explained_mean": round(m, 5), "pc1_explained_ci_lower": round(lo, 5),
                                "pc1_explained_ci_upper": round(hi, 5), "n_seeds": ns,
                                "pc1_per_seed": [round(x, 5) for x in pc1_per]})

    out = {"features": GEOMETRY_FEATURES, "streams": streams, "steps": steps,
           "constraint_cosines": cosines, "constraint_cosines_ensemble": cosines_ensemble,
           "pca_pc1_ensemble": pca_pc1, "weight_norms": weight_norms}
    with open(os.path.join(args.output_dir, "geometric_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save({k: {kk: vv for kk, vv in v.items() if kk != "W_per_seed"} | {"W_per_seed": v["W_per_seed"]}
                for k, v in probe_weights.items()}, os.path.join(args.output_dir, "probe_weights.pt"))
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "maze_probe_geometry", {
            "num_puzzles": args.num_puzzles, "steps": steps, "streams": streams, "seeds": seeds,
            "features": GEOMETRY_FEATURES, "split": "puzzle_disjoint"}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[geom] WARN _meta: {e}")
    print(f"[geom] wrote {args.output_dir}/geometric_analysis.json ({len(cosines)} cosine pairs, "
          f"{len(pca_pc1)} PCA groups)")


if __name__ == "__main__":
    main()
