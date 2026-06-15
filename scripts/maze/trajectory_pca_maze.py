#!/usr/bin/env python3
"""Maze z_H (or z_L) latent-trajectory PCA — replication of Sudoku exp6.

For every puzzle we take the per-step post-step state ``z_*_out``, mean-pool over
the 900 answer cells to a [max_steps, D] trajectory, then fit a PCA across all
(puzzle, step) points and report:

  - explained-variance ratio of the top components (the "PC1 ≈ X% / 2-D captures
    Y%" numbers, with a seed/bootstrap CI),
  - 2-D projected trajectories (saved for the figure),
  - per-trajectory geometry: path length & net displacement in PCA space,
    drift (cosine of step k vs final) and smoothness (cosine of consecutive
    steps) in the full D-dim space, with bootstrap CIs across puzzles,
  - per-step **path-validity** (exact_solved, valid_sg_path, connects_start_goal,
    path_f1) so the latent trajectory is annotated with solution progress — not
    token accuracy.

Input: reuses the SAE activation dump (``results/maze/sae_study/activations_zH.pt``)
when present (CPU-only, fast). Otherwise collects inline from the checkpoint (GPU).

Outputs (under --output_dir)
  trajectory_pca.json   explained variance + per-step metrics + trajectory stats
  trajectory_pca.npz    trajectories_2d [N,steps,2], components, per-step arrays
  _meta.json            provenance

Usage
  python scripts/maze/trajectory_pca_maze.py --stream z_H \
      --activations results/maze/sae_study/activations_zH.pt --num_puzzles 300
  python scripts/maze/trajectory_pca_maze.py --stream z_H --num_puzzles 300 --device cuda
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import bootstrap_ci
from scripts.maze.maze_common import maze_prediction_metrics
from utils.maze_targets import SEQ_LEN

PATH_VALIDITY_KEYS = ["exact_solved", "valid_sg_path", "connects_start_goal", "path_f1"]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── trajectory collection ─────────────────────────────────────────────────

def load_from_dump(path: str, stream: str) -> Dict[str, np.ndarray]:
    """Build [N, steps, D] pooled trajectories + per-step path metrics from a dump."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    key = "z_H" if stream == "z_H" else "z_L"
    if key not in data:
        raise KeyError(f"{path} has no '{key}' (collect with --also_collect_zL for z_L)")
    z = data[key].float()                       # [N, steps, cells, D]
    pooled = z.mean(dim=2).numpy()              # [N, steps, D]
    inputs = data["inputs"].numpy()             # [N, cells]
    labels = data["labels"].numpy()
    preds = data["per_step_preds"].numpy()      # [N, steps, cells]
    n, steps = pooled.shape[0], pooled.shape[1]
    metrics = np.zeros((steps, len(PATH_VALIDITY_KEYS)), dtype=np.float64)
    for s in range(steps):
        rows = [maze_prediction_metrics(preds[i, s], labels[i], inputs[i]) for i in range(n)]
        for j, k in enumerate(PATH_VALIDITY_KEYS):
            metrics[s, j] = float(np.mean([r[k] for r in rows]))
    return {"pooled": pooled, "per_step_metrics": metrics}


def collect_inline(checkpoint: str, num_puzzles: int, max_steps: int, stream: str,
                   device: torch.device) -> Dict[str, np.ndarray]:
    from scripts.controlled.controlled_common import load_model_and_dataloader, collect_puzzles
    from scripts.core.activation_ablation import ActivationAblator, ActivationCache
    from scripts.maze.linear_probes_maze import _answer_slice, _flat, _slice_preds
    from scripts.maze.maze_common import get_puzzle_emb_len

    model, test_loader, _ = load_model_and_dataloader(checkpoint, device)
    pel = get_puzzle_emb_len(model)
    ablator = ActivationAblator(model, device=device)
    puzzles = collect_puzzles(test_loader, device, num_puzzles)
    field = "z_H_out" if stream == "z_H" else "z_L_out"

    pooled_all, metric_rows = [], {}
    for n, (pidx, batch) in enumerate(puzzles):
        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
        inp, label = _flat(batch["inputs"]), _flat(batch["labels"])
        steps = sorted(cache.keys())
        traj = []
        for s in steps:
            z = _answer_slice(getattr(cache[s], field), pel)[0].float().mean(dim=0).cpu().numpy()
            traj.append(z)
            preds = _slice_preds(_flat(cache[s].preds), label.size)
            m = maze_prediction_metrics(preds, label, inp)
            metric_rows.setdefault(s, []).append([m[k] for k in PATH_VALIDITY_KEYS])
        pooled_all.append(np.stack(traj, 0))
        if (n + 1) % 50 == 0:
            print(f"[traj] {n+1}/{len(puzzles)}")
    pooled = np.stack(pooled_all, 0)            # [N, steps, D]
    steps_sorted = sorted(metric_rows.keys())
    metrics = np.stack([np.mean(metric_rows[s], axis=0) for s in steps_sorted], 0)
    return {"pooled": pooled, "per_step_metrics": metrics}


# ── analysis ──────────────────────────────────────────────────────────────

def analyze(pooled: np.ndarray, metrics: np.ndarray, seeds: List[int]) -> Tuple[Dict, Dict]:
    from sklearn.decomposition import PCA
    N, steps, D = pooled.shape
    flat = pooled.reshape(N * steps, D)

    ncomp = min(10, D, N * steps)
    pca = PCA(n_components=ncomp)
    proj_flat = pca.fit_transform(flat)
    evr = pca.explained_variance_ratio_
    proj2d = proj_flat[:, :2].reshape(N, steps, 2)

    # Bootstrap CI on explained variance by resampling puzzles and refitting PCA
    # with the same component count so the ratios are computed identically.
    rng = np.random.default_rng(0)
    pc1s, cum2s = [], []
    for _ in range(max(50, 10 * len(seeds))):
        idx = rng.integers(0, N, size=N)
        f = pooled[idx].reshape(N * steps, D)
        p = PCA(n_components=ncomp).fit(f)
        pc1s.append(float(p.explained_variance_ratio_[0]))
        cum2s.append(float(p.explained_variance_ratio_[:2].sum()))
    def ci(x):
        a = np.asarray(x); return [float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    # Per-trajectory geometry (full-D drift/smoothness, 2-D path length)
    path_len, net_disp, drift_final, smooth = [], [], [], []
    for i in range(N):
        t2 = proj2d[i]; tD = pooled[i]
        path_len.append(float(np.sum(np.linalg.norm(np.diff(t2, axis=0), axis=1))))
        net_disp.append(float(np.linalg.norm(t2[-1] - t2[0])))
        drift_final.append(_cos(tD[0], tD[-1]))
        smooth.append(float(np.mean([_cos(tD[s], tD[s + 1]) for s in range(steps - 1)])))

    def boot(x):
        r = bootstrap_ci(np.asarray(x, dtype=float))
        return {"mean": float(r["mean"]), "ci_lower": float(r["ci_lower"]), "ci_upper": float(r["ci_upper"])}

    result = {
        "n_puzzles": int(N), "max_steps": int(steps), "hidden_dim": int(D),
        "pca_explained_variance_ratio": [round(float(v), 5) for v in evr],
        "pc1_explained": round(float(evr[0]), 5),
        "pc2_explained": round(float(evr[1]), 5) if len(evr) > 1 else None,
        "cumulative_2pc_explained": round(float(evr[:2].sum()), 5),
        "pc1_explained_bootstrap_ci": [round(v, 5) for v in ci(pc1s)],
        "cumulative_2pc_bootstrap_ci": [round(v, 5) for v in ci(cum2s)],
        "trajectory_geometry": {
            "path_length_2d": boot(path_len),
            "net_displacement_2d": boot(net_disp),
            "cos_step0_to_final": boot(drift_final),
            "mean_consecutive_cos": boot(smooth),
        },
        "per_step_path_validity": {
            PATH_VALIDITY_KEYS[j]: [round(float(metrics[s, j]), 5) for s in range(steps)]
            for j in range(len(PATH_VALIDITY_KEYS))
        },
    }
    npz = {"trajectories_2d": proj2d, "components": pca.components_,
           "explained_variance_ratio": evr, "per_step_metrics": metrics}
    return result, npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="z_H", choices=["z_H", "z_L"])
    ap.add_argument("--activations", default="results/maze/sae_study/activations_zH.pt",
                    help="SAE dump to reuse (CPU). If missing, collect inline (GPU).")
    ap.add_argument("--checkpoint", default="checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint")
    ap.add_argument("--num_puzzles", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output_dir", default="results/maze/hardened/trajectory_pca")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if os.path.exists(args.activations):
        print(f"[traj] loading pooled trajectories from {args.activations}")
        data = load_from_dump(args.activations, args.stream)
    else:
        dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                           else ("cpu" if args.device == "auto" else args.device))
        print(f"[traj] no dump at {args.activations}; collecting inline on {dev}")
        data = collect_inline(args.checkpoint, args.num_puzzles, args.max_steps, args.stream, dev)

    result, npz = analyze(data["pooled"], data["per_step_metrics"], seeds)
    result["stream"] = args.stream
    with open(os.path.join(args.output_dir, "trajectory_pca.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(args.output_dir, "trajectory_pca.npz"), **npz)
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, f"maze_trajectory_pca_{args.stream}", {
            "stream": args.stream, "activations": args.activations,
            "num_puzzles": result["n_puzzles"], "max_steps": result["max_steps"],
            "seeds": seeds}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[traj] WARN _meta: {e}")

    print(f"[traj] PC1={result['pc1_explained']:.3f} cum2={result['cumulative_2pc_explained']:.3f} "
          f"(PC1 CI {result['pc1_explained_bootstrap_ci']})")
    print(f"[traj] wrote {args.output_dir}/trajectory_pca.json + .npz")


if __name__ == "__main__":
    main()
