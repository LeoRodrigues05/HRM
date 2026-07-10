#!/usr/bin/env python3
"""ARC probe-weight geometry — closes the "partial" probe-geometry cell for ARC.

ARC analog of the Sudoku ``geometric_analysis.json`` (E8) and
``scripts/maze/probe_geometry_maze.py``, computed directly from the saved hardened
ARC probe weights (``results/arc/hardened/linear_probes/probe_weights.pt``; entries
are seed-ensemble means, so no re-fitting or GPU is needed).

Per (stream, step): pairwise cosine similarity between binary-target probe
directions, PCA explained-variance spectrum of the stacked unit directions (how
low-dimensional the readable subspace is), per-direction weight norms, and the
probes' validation accuracies for reference.

Usage (CPU, seconds):
  python scripts/arc/probe_geometry_arc.py \
      --probe_weights results/arc/hardened/linear_probes/probe_weights.pt \
      --output_dir results/arc/hardened/probe_geometry
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="ARC probe-weight geometry analysis")
    p.add_argument("--probe_weights", default="results/arc/hardened/linear_probes/probe_weights.pt")
    p.add_argument("--output_dir", default="results/arc/hardened/probe_geometry")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pw = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
    # group[(stream, step)] -> {target: W_mean}
    group = defaultdict(dict)
    accs = defaultdict(dict)
    for entry in pw.values():
        w = entry["W_mean"].float()
        if w.ndim != 1:
            continue  # only binary targets define a single direction
        group[(entry["stream"], int(entry["step"]))][entry["target"]] = w
        accs[(entry["stream"], int(entry["step"]))][entry["target"]] = float(entry["val_acc_mean"])

    out = {"probe_weights": args.probe_weights, "analysis": {}}
    for (stream, step), targets in sorted(group.items()):
        names = sorted(targets)
        W = torch.stack([targets[t] / targets[t].norm().clamp(min=1e-8) for t in names])  # [T, D]
        cos = (W @ W.T).numpy()
        # PCA of the stacked unit directions
        Wc = W.numpy() - W.numpy().mean(0, keepdims=True)
        _, s, _ = np.linalg.svd(Wc, full_matrices=False)
        var = s ** 2
        evr = (var / var.sum()).tolist() if var.sum() > 0 else []
        node = {
            "targets": names,
            "val_acc": {t: accs[(stream, step)][t] for t in names},
            "weight_norms": {t: float(group[(stream, step)][t].norm()) for t in names},
            "cosine_matrix": [[round(float(c), 4) for c in row] for row in cos],
            "pca_explained_variance_ratio": [round(float(v), 4) for v in evr],
            "pc1_explained_variance": round(float(evr[0]), 4) if evr else None,
            "n_dims_for_90pct_var": int(np.searchsorted(np.cumsum(evr), 0.90) + 1) if evr else None,
            "subspace_dim": len(names),
            "hidden_dim": int(W.shape[1]),
        }
        out["analysis"][f"{stream}_step{step}"] = node

    with open(os.path.join(args.output_dir, "geometry_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "arc_probe_geometry",
                   {"probe_weights": args.probe_weights}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[arc_geom] WARN _meta: {e}")

    # human-readable summary for the final step of each stream
    for stream in sorted({s for s, _ in group}):
        steps = sorted(st for s2, st in group if s2 == stream)
        key = f"{stream}_step{steps[-1]}"
        node = out["analysis"][key]
        print(f"\n== {key} ==  ({len(node['targets'])} directions in {node['hidden_dim']}-d)")
        print("  PC1 var:", node["pc1_explained_variance"],
              "| dims for 90% var:", node["n_dims_for_90pct_var"])
        names = node["targets"]
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                c = node["cosine_matrix"][i][j]
                if abs(c) >= 0.4:
                    print(f"  cos({a}, {names[j]}) = {c:+.3f}")
    print(f"\n[arc_geom] wrote {args.output_dir}/geometry_analysis.json")


if __name__ == "__main__":
    main()
