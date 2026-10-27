#!/usr/bin/env python3
"""E7: clean accuracy and perturbation sensitivity of the BPTT-study models.

The paper compares SAE ablations in a stock (one-step gradient) HRM and a BPTT
HRM (-1.59 vs -29.64 pp) without reporting how accurate either model is. This
script evaluates each model on the same Sudoku puzzles:

  clean     cell, blank-cell and puzzle accuracy at every ACT step
  noise     Gaussian noise added to the carry at every step, with norm
            eps * |z| per position, for each eps in --noise_levels; change in
            final accuracy against the clean run
  recovery  noise added to z_H once, at step k (eps = --recovery_eps); cosine
            similarity of z_H to its clean trajectory at every later step, and
            the change in final accuracy

If the one-step gradient makes each update a robust refinement step, the stock
model should tolerate larger perturbations and return to its clean trajectory
faster than the BPTT model.

Usage
-----
  python scripts/analysis/bptt_arm_eval.py \\
      --arms released=checkpoints/sapientinc-sudoku-extreme,stock=checkpoints/bptt_study/hrm_stock,bptt=checkpoints/bptt_study/hrm_bptt \\
      --n_puzzles 200 --output_dir results/Sudoku/sae_study/bptt_study/arm_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.sae.sae_error_controls import summarize, paired_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_CELLS = 81
DEFAULT_INDICES = "results/Sudoku/sae_study/bptt_study/hrm_bptt/causal_ablation/puzzle_indices.json"


def _extract_batch(data):
    if isinstance(data, (tuple, list)):
        return data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
    return data


def sudoku_scores(preds: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    p = preds[:, -SUDOKU_CELLS:].cpu()
    t = batch["labels"][:, -SUDOKU_CELLS:].cpu()
    inp = batch["inputs"][:, -SUDOKU_CELLS:].cpu()
    correct = p == t
    blank = (inp - 1).clamp(0, 9) == 0          # same rule as evaluate_baselines.py
    return {
        "cell_acc": float(correct.float().mean().item()),
        "blank_cell_acc": float(correct[blank].float().mean().item()) if bool(blank.any()) else float("nan"),
        "puzzle_acc": float(correct.all().item()),
    }


def noise_transform(eps: float, on: str, generator: torch.Generator,
                    only_step: Optional[int] = None) -> Callable:
    """Add noise with norm eps * |z| (per position) to the incoming carry."""
    def add(z: torch.Tensor) -> torch.Tensor:
        zf = z.float()
        u = torch.randn(zf.shape, generator=generator, device=zf.device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (zf + eps * zf.norm(dim=-1, keepdim=True) * u).to(z.dtype)

    def fn(z_H: torch.Tensor, z_L: torch.Tensor, step: int):
        if only_step is not None and step != only_step:
            return z_H, z_L
        if on in ("H", "both"):
            z_H = add(z_H)
        if on in ("L", "both"):
            z_L = add(z_L)
        return z_H, z_L
    return fn


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean over positions of the cosine between two [1, T, D] states."""
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


def evaluate_arm(name: str, ckpt_dir: str, args, device: torch.device) -> Dict[str, Any]:
    from scripts.sae.sae_causal_ablation import load_model_and_data
    from scripts.core.sudoku_sample import collect_indexed_batches, load_puzzle_indices
    from scripts.core.intervention_runner import InterventionRunner

    model, loader, _ = load_model_and_data(device, ckpt_dir=ckpt_dir)
    indices = (load_puzzle_indices(args.puzzle_indices, limit=args.n_puzzles)
               if args.puzzle_indices else None)
    puzzles = collect_indexed_batches(loader, device, num_puzzles=args.n_puzzles,
                                      puzzle_indices=indices, extract_batch=_extract_batch)
    runner = InterventionRunner(model, device=device)
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    recovery_steps = [int(x) for x in args.recovery_steps.split(",") if x.strip()]
    logger.info(f"[{name}] {ckpt_dir}: {len(puzzles)} puzzles")

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for pi, (pidx, batch) in enumerate(puzzles):
        clean = runner.rollout(batch, None, args.max_steps, record=True)
        per_step = [sudoku_scores(sp, batch) for sp in clean["step_preds"]]
        final = per_step[-1]
        rec: Dict[str, Any] = {"puzzle_idx": int(pidx), "clean_per_step": per_step,
                               "noise": {}, "recovery": {}}
        for eps in noise_levels:
            deltas = []
            for r in range(args.noise_repeats):
                gen = torch.Generator(device=device).manual_seed(args.seed + 7919 * r + 104_729 * pi)
                out = runner.rollout(batch, noise_transform(eps, args.noise_on, gen), args.max_steps)
                s = sudoku_scores(out["preds"], batch)
                deltas.append({k: s[k] - final[k] for k in ("cell_acc", "puzzle_acc")})
            rec["noise"][str(eps)] = {k: float(np.mean([d[k] for d in deltas]))
                                      for k in ("cell_acc", "puzzle_acc")}
        for k in recovery_steps:
            gen = torch.Generator(device=device).manual_seed(args.seed + 31 * k + 104_729 * pi)
            out = runner.rollout(batch, noise_transform(args.recovery_eps, "H", gen, only_step=k),
                                 args.max_steps, record=True)
            rec["recovery"][str(k)] = {
                "cos_to_clean": [mean_cosine(out["z_H_out"][t], clean["z_H_out"][t])
                                 for t in range(k, args.max_steps)],
                "final_cell_acc_delta": sudoku_scores(out["preds"], batch)["cell_acc"] - final["cell_acc"],
            }
        records.append(rec)
        if (pi + 1) % 20 == 0:
            el = time.time() - t0
            logger.info(f"[{name}] {pi + 1}/{len(puzzles)}  {el:.0f}s, "
                        f"~{el / (pi + 1) * (len(puzzles) - pi - 1):.0f}s left")

    signature = [int(b["inputs"].sum().item()) for _, b in puzzles[:10]]
    return {"name": name, "checkpoint_dir": ckpt_dir, "records": records, "signature": signature}


def summarize_arm(arm: Dict[str, Any], args) -> Dict[str, Any]:
    recs = arm["records"]
    n_steps = len(recs[0]["clean_per_step"])
    out: Dict[str, Any] = {"n_puzzles": len(recs), "checkpoint_dir": arm["checkpoint_dir"]}
    out["clean_final"] = {k: summarize([r["clean_per_step"][-1][k] for r in recs
                                        if not np.isnan(r["clean_per_step"][-1][k])])
                          for k in ("cell_acc", "blank_cell_acc", "puzzle_acc")}
    out["clean_per_step_mean"] = {
        k: [float(np.nanmean([r["clean_per_step"][s][k] for r in recs])) for s in range(n_steps)]
        for k in ("cell_acc", "blank_cell_acc", "puzzle_acc")}
    out["noise"] = {eps: {k: summarize([r["noise"][eps][k] for r in recs])
                          for k in ("cell_acc", "puzzle_acc")}
                    for eps in recs[0]["noise"]}
    out["recovery"] = {}
    for k in recs[0]["recovery"]:
        curves = np.array([r["recovery"][k]["cos_to_clean"] for r in recs])
        out["recovery"][k] = {
            "mean_cos_to_clean_from_step_k": curves.mean(axis=0).tolist(),
            "final_cell_acc_delta": summarize([r["recovery"][k]["final_cell_acc_delta"] for r in recs]),
        }
    return out


def main():
    p = argparse.ArgumentParser(description="E7: BPTT-study model accuracy and perturbation sensitivity")
    p.add_argument("--arms", default=("released=checkpoints/sapientinc-sudoku-extreme,"
                                      "stock=checkpoints/bptt_study/hrm_stock,"
                                      "bptt=checkpoints/bptt_study/hrm_bptt"),
                   help="comma-separated name=checkpoint_dir pairs")
    p.add_argument("--n_puzzles", type=int, default=200)
    p.add_argument("--puzzle_indices", default=DEFAULT_INDICES,
                   help="JSON puzzle-index manifest (default: the BPTT SAE-ablation puzzles); "
                        "pass '' to use the first n_puzzles")
    p.add_argument("--noise_levels", default="0.003,0.01,0.03,0.1",
                   help="noise norm as a fraction of the state norm, per position")
    p.add_argument("--noise_on", choices=["H", "L", "both"], default="H")
    p.add_argument("--noise_repeats", type=int, default=3)
    p.add_argument("--recovery_steps", default="1,4,8")
    p.add_argument("--recovery_eps", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="results/Sudoku/sae_study/bptt_study/arm_eval")
    p.add_argument("--tost_bound_pp", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if args.puzzle_indices and not os.path.exists(args.puzzle_indices):
        logger.warning(f"{args.puzzle_indices} not found; using the first {args.n_puzzles} puzzles")
        args.puzzle_indices = None
    os.makedirs(args.output_dir, exist_ok=True)

    arms = [pair.split("=", 1) for pair in args.arms.split(",") if pair.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    for name, ckpt_dir in arms:
        if not os.path.isdir(ckpt_dir):
            logger.warning(f"[{name}] {ckpt_dir} not found, skipping")
            continue
        results[name] = evaluate_arm(name, ckpt_dir, args, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    signatures = {n: r["signature"] for n, r in results.items()}
    same_puzzles = len({tuple(s) for s in signatures.values()}) <= 1
    if not same_puzzles:
        logger.warning("Arms saw different puzzles (input signatures differ); "
                       "cross-arm paired comparisons are skipped.")

    summary: Dict[str, Any] = {"settings": vars(args), "same_puzzles_across_arms": same_puzzles,
                               "arms": {n: summarize_arm(r, args) for n, r in results.items()}}

    if same_puzzles and "stock" in results and "bptt" in results:
        bound = args.tost_bound_pp / 100.0
        s_recs, b_recs = results["stock"]["records"], results["bptt"]["records"]
        summary["bptt_minus_stock"] = {
            "clean_cell_acc": paired_comparison(
                [b["clean_per_step"][-1]["cell_acc"] - s["clean_per_step"][-1]["cell_acc"]
                 for s, b in zip(s_recs, b_recs)], bound),
            "noise_cell_acc_delta": {
                eps: paired_comparison([b["noise"][eps]["cell_acc"] - s["noise"][eps]["cell_acc"]
                                        for s, b in zip(s_recs, b_recs)], bound)
                for eps in s_recs[0]["noise"]},
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for name, r in results.items():
        with open(os.path.join(args.output_dir, f"per_puzzle_{name}.jsonl"), "w") as f:
            for rec in r["records"]:
                f.write(json.dumps(rec) + "\n")
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "bptt_arm_eval", vars(args), repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"Could not write _meta.json: {e}")

    print("\n" + "=" * 72)
    print("E7 BPTT ARM EVALUATION (Sudoku)")
    print("=" * 72)
    for name, s in summary["arms"].items():
        c = s["clean_final"]
        noise = "  ".join(f"eps={eps}: {100 * v['cell_acc']['mean']:+.2f}" for eps, v in s["noise"].items())
        print(f"  {name:9s} cell {100 * c['cell_acc']['mean']:.2f}%  blank {100 * c['blank_cell_acc']['mean']:.2f}%  "
              f"puzzle {100 * c['puzzle_acc']['mean']:.2f}%  | noise on z_{args.noise_on} (pp): {noise}")
        for k, rv in s["recovery"].items():
            cos = rv["mean_cos_to_clean_from_step_k"]
            print(f"            recovery from step {k}: cos {cos[0]:.3f} -> {cos[-1]:.3f}, "
                  f"final cell {100 * rv['final_cell_acc_delta']['mean']:+.2f} pp")
    if "bptt_minus_stock" in summary:
        d = summary["bptt_minus_stock"]["clean_cell_acc"]
        print(f"  BPTT minus stock, clean cell accuracy: {100 * d['mean']:+.2f} pp "
              f"[{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]")
    print(f"  wrote {args.output_dir}")


if __name__ == "__main__":
    main()
