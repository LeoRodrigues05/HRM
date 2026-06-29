#!/usr/bin/env python3
"""Test-time-augmentation VOTING eval for ARC-2 — the metric the HRM authors report.

This is a self-contained, committable port of the voting logic in ``arc_eval.ipynb``.
It does NOT run the model; it consumes the per-example prediction dump that
``evaluate.py`` writes (``<checkpoint>_all_preds.<rank>``) and aggregates across all
augmentations of each puzzle.

Why this differs from scripts/arc/measure_arc_accuracy.py:
  measure_arc_accuracy.py reports SINGLE-SHOT per-augmentation exact match (a strict,
  much lower number). The published ~40.3% is pass@2 AFTER voting across ~1000
  augmentations per puzzle, inverse-transforming each prediction back to canonical
  orientation, then ranking candidate answers by (vote count, mean q_halt confidence).

Pipeline:
  1) Dump predictions over the full eval set (GPU, ~hours on a small GPU):
       PYTHONPATH=$PWD python evaluate.py \
           checkpoint=checkpoints/arc2-adapted-evalonly/step_7391
     (evaluate.py defaults save_outputs to inputs/labels/puzzle_identifiers/logits/
      q_halt_logits — exactly what voting needs. It writes step_7391_all_preds.0
      next to the checkpoint.)
  2) Vote + score (CPU, fast):
       PYTHONPATH=$PWD python scripts/arc/voting_eval_arc.py \
           --checkpoint checkpoints/arc2-adapted-evalonly/step_7391 \
           --dataset_path data/arc-2-evalonly

A puzzle counts as solved at K if EVERY test input's true answer appears among the
top-K voted candidate grids (ARC's official rule allows K=2 guesses).
"""
from __future__ import annotations
import os, sys, json, argparse
from glob import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
try:
    from numba import njit                      # optional: ~10× faster crop over 165k grids
except ModuleNotFoundError:                     # pure-Python fallback (numba not installed)
    def njit(f=None, **kw):
        return (lambda g: g) if f is None else f

from dataset.common import inverse_dihedral_transform

PAD_PUZZLE_IDENTIFIER = 0


def load_identifiers_and_preds(dataset_path: str, checkpoint_path: str):
    with open(os.path.join(dataset_path, "identifiers.json")) as f:
        identifier_map = json.load(f)

    all_preds: dict = {}
    files = sorted(glob(f"{checkpoint_path}_all_preds.*"))
    if not files:
        raise FileNotFoundError(
            f"No prediction dump found at {checkpoint_path}_all_preds.* — run evaluate.py first "
            f"(see this file's docstring)."
        )
    # evaluate() always records an "intermediate_preds" tensor of shape
    # [steps, N, seq] (a leading ACT-step axis) alongside the per-example tensors,
    # because "intermediate_preds_step" is always requested. Voting never uses it
    # and its leading dim breaks the per-example mask, so keep only what we need.
    NEEDED = ("puzzle_identifiers", "inputs", "labels", "logits", "q_halt_logits")
    for filename in files:
        preds = torch.load(filename, map_location="cpu")
        for k, v in preds.items():
            if k in NEEDED:
                all_preds.setdefault(k, []).append(v)
        del preds
    all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

    mask = all_preds["puzzle_identifiers"] != PAD_PUZZLE_IDENTIFIER
    all_preds = {k: v[mask] for k, v in all_preds.items()}
    return identifier_map, all_preds, files


def inverse_aug(name: str, grid: np.ndarray):
    """Undo the (dihedral transform, colour permutation) encoded in the identifier."""
    if "_" not in name:
        return grid
    trans_id, perm = name.split("_")[-2:]
    trans_id = int(trans_id[1:])               # strip leading "t"
    inv_perm = np.argsort(list(perm))
    return inv_perm[inverse_dihedral_transform(grid, trans_id)]


def grid_hash(grid: np.ndarray):
    return hash((grid.tobytes(), grid.shape))


@njit
def crop(grid: np.ndarray):
    """Largest top-left rectangle containing no EOS/pad token; map tokens back to colours."""
    grid = grid.reshape(30, 30)
    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return grid[:max_size[0], :max_size[1]] - 2


def vote_and_score(identifier_map, all_preds, Ks):
    # Ground-truth answers, keyed by puzzle name then input-grid hash.
    global_hmap: dict = {}
    puzzle_labels: dict = {}
    for identifier, inp, label in zip(
        all_preds["puzzle_identifiers"], all_preds["inputs"], all_preds["labels"]
    ):
        name = identifier_map[identifier]
        if "_" in name:        # only the canonical (non-augmented) rows carry the true test pair
            continue
        puzzle_labels.setdefault(name, {})
        ci = crop(inp.numpy())
        cl = crop(label.numpy())
        ih, lh = grid_hash(ci), grid_hash(cl)
        global_hmap[ih] = ci
        global_hmap[lh] = cl
        puzzle_labels[name][ih] = lh

    preds = all_preds["logits"].argmax(-1)

    # Collate candidate answers across augmentations, with q_halt confidence.
    pred_answers: dict = {}
    for identifier, inp, pred, q in zip(
        all_preds["puzzle_identifiers"], all_preds["inputs"], preds,
        all_preds["q_halt_logits"].sigmoid(),
    ):
        name = identifier_map[identifier]
        orig = name.split("_")[0]
        ih = grid_hash(inverse_aug(name, crop(inp.numpy())))
        if orig not in puzzle_labels or ih not in puzzle_labels[orig]:
            continue
        pg = inverse_aug(name, crop(pred.numpy()))
        ph = grid_hash(pg)
        global_hmap[ph] = pg
        pred_answers.setdefault(orig, {}).setdefault(ih, []).append((ph, q.item()))

    correct = [0 for _ in Ks]
    for name, tests in puzzle_labels.items():
        num_ok = [0 for _ in Ks]
        for ih, lh in tests.items():
            cand = pred_answers.get(name, {}).get(ih, [])
            stats: dict = {}
            for h, q in cand:
                s = stats.setdefault(h, [0, 0.0])
                s[0] += 1
                s[1] += q
            for h in stats:
                stats[h][1] /= stats[h][0]
            ranked = sorted(stats.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
            for i, k in enumerate(Ks):
                if any(h == lh for h, _ in ranked[:k]):
                    num_ok[i] += 1
        for i in range(len(Ks)):
            correct[i] += (num_ok[i] == len(tests))
    return puzzle_labels, correct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to the checkpoint whose <ckpt>_all_preds.* dump to score")
    p.add_argument("--dataset_path", required=True,
                   help="Dataset dir holding identifiers.json (must match the build the "
                        "checkpoint's puzzle_emb was trained on)")
    p.add_argument("--Ks", default="1,2,10,100,1000")
    p.add_argument("--output_dir", default="results/arc/diagnostics")
    args = p.parse_args()

    Ks = [int(x) for x in args.Ks.split(",")]
    identifier_map, all_preds, files = load_identifiers_and_preds(args.dataset_path, args.checkpoint)
    puzzle_labels, correct = vote_and_score(identifier_map, all_preds, Ks)
    n = len(puzzle_labels)

    summary = {f"pass@{k}": (correct[i] / n if n else 0.0) for i, k in enumerate(Ks)}
    summary["n_puzzles"] = n
    summary["dump_files"] = files
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "arc_voting_accuracy.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"ARC-2 VOTING accuracy over {n} puzzles ({os.path.basename(args.checkpoint)}):")
    for i, k in enumerate(Ks):
        print(f"  pass@{k:<5d} {correct[i] / n * 100:6.2f}%")
    print("=" * 60)
    print("pass@2 is the authors' headline metric (ARC allows 2 guesses).")
    print(f"[voting_eval] wrote {out}")


if __name__ == "__main__":
    main()
