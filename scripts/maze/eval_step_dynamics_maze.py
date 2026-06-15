"""Baseline eval + z_H trajectory + per-step accuracy dynamics on Maze 30x30.

Single forward pass per puzzle; for each ACT step record:
  - token-level accuracy on labels
  - path-cell accuracy (on PATH/START/GOAL only)
  - structural path metrics (precision/recall, wall hits, S-G connectivity,
    branch-free path validity, optimal-length validity)
  - z_H, z_L norms (mean over batch / positions)
  - cosine sim of z_H_out[step] vs z_H_out[step-1]   (trajectory smoothness)
  - cosine sim of z_H_out[step] vs z_H_out[-1]       (drift from final)

Outputs JSON aggregate to results/maze/step_dynamics/aggregate.json
and per-puzzle JSONL.

Usage:
    python scripts/maze/eval_step_dynamics_maze.py --num_puzzles 200
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
    load_model_and_dataloader, collect_puzzles, extract_batch, bootstrap_ci,
)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, maze_prediction_metrics,
    MAZE_METRIC_KEYS,
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def run_one(ablator: ActivationAblator, batch: Dict[str, torch.Tensor],
            max_steps: int, puzzle_emb_len: int):
    cache: Dict[int, ActivationCache] = {}
    ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
    steps = sorted(cache.keys())
    if not steps:
        return None
    label = batch["labels"].detach().to("cpu").to(torch.int64).numpy()
    if label.ndim == 2:
        label = label[0]
    inp = batch["inputs"].detach().to("cpu").to(torch.int64).numpy()
    if inp.ndim == 2:
        inp = inp[0]

    final_zH_out = cache[steps[-1]].z_H_out

    per_step = []
    prev_zH = None
    for s in steps:
        c = cache[s]
        # Slice off puzzle-embedding prefix to align with labels
        preds = c.preds.detach().to("cpu").to(torch.int64).numpy()
        if preds.ndim == 2:
            preds = preds[0]
        if preds.size > label.size:  # safety
            preds = preds[-label.size:]
        accs = maze_prediction_metrics(preds, label, inp)
        zH = c.z_H_out
        zL = c.z_L_out
        row = {
            "step": s,
            "zH_norm": float(zH.float().norm().item() / (zH.numel() ** 0.5)),
            "zL_norm": float(zL.float().norm().item() / (zL.numel() ** 0.5)),
            "cos_zH_prev": _cos(zH, prev_zH) if prev_zH is not None else 1.0,
            "cos_zH_final": _cos(zH, final_zH_out),
        }
        row.update(accs)
        per_step.append(row)
        prev_zH = zH
    return per_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--num_puzzles", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="results/maze/step_dynamics")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[step_dynamics] device={device}")

    model, test_loader, _cfg = load_model_and_dataloader(args.checkpoint, device)
    puzzle_emb_len = get_puzzle_emb_len(model)
    print(f"[step_dynamics] puzzle_emb_len={puzzle_emb_len}")
    ablator = ActivationAblator(model, device=device)

    puzzles = collect_puzzles(test_loader, device, args.num_puzzles)
    print(f"[step_dynamics] collected {len(puzzles)} puzzles")

    per_puzzle_path = os.path.join(args.output_dir, "per_puzzle.jsonl")
    out_f = open(per_puzzle_path, "w")
    all_traj: List[List[Dict]] = []
    t0 = time.time()
    for i, (idx, batch) in enumerate(puzzles):
        traj = run_one(ablator, batch, args.max_steps, puzzle_emb_len)
        if traj is None:
            continue
        rec = {"puzzle_idx": idx, "per_step": traj}
        out_f.write(json.dumps(rec) + "\n")
        all_traj.append(traj)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(puzzles)}] elapsed={time.time()-t0:.1f}s")
    out_f.close()

    # Aggregate
    if not all_traj:
        print("No trajectories collected")
        return
    n_steps = max(len(t) for t in all_traj)
    agg = {"num_puzzles": len(all_traj), "max_steps": n_steps, "per_step": []}
    for s in range(n_steps):
        bucket = {k: [] for k in MAZE_METRIC_KEYS + [
            "zH_norm", "zL_norm", "cos_zH_prev", "cos_zH_final",
        ]}
        for traj in all_traj:
            if s < len(traj):
                for k in bucket:
                    bucket[k].append(traj[s][k])
        row = {"step": s}
        for k, vals in bucket.items():
            if vals:
                row[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        agg["per_step"].append(row)

    # Final-step (best halt) baseline summary
    agg["final"] = {
        metric: bootstrap_ci([t[-1][metric] for t in all_traj])
        for metric in MAZE_METRIC_KEYS
    }

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[step_dynamics] wrote {agg_path}")
    print(f"  final token_acc={agg['final']['token_acc']['mean']:.4f}  "
          f"path_cell_acc={agg['final']['path_cell_acc']['mean']:.4f}  "
          f"exact_solved={agg['final']['exact_solved']['mean']:.4f}")


if __name__ == "__main__":
    main()
