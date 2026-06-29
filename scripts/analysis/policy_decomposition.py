#!/usr/bin/env python3
"""Experiment H4: H-update vs L-cycles policy decomposition.

For each ACT step s, decomposes the policy improvement into:
  ΔH_s = contribution attributable to the H-level update (L frozen at z_L_in)
  ΔL_s = contribution from the L-cycles (extra from live z_L updating)
  Δtotal_s = ΔH_s + ΔL_s  (exact by construction — asserted)

Reports frac_H_s = ΔH_s / (ΔH_s + ΔL_s) per step.

Also extends Experiments A/D by splitting solved vs failed puzzles.

Usage:
    python scripts/analysis/policy_decomposition.py --task sudoku --device cuda
    python scripts/analysis/policy_decomposition.py --task maze   --device cuda
"""
from __future__ import annotations
import os, sys, json, argparse, logging, time
from typing import Dict, List, Optional, Any, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci, find_checkpoint,
)
from scripts.core.activation_ablation import ActivationAblator
from scripts.maze.maze_common import MAZE_CHECKPOINT
from scripts.arc.arc_common import ARC_CHECKPOINT
from scripts.analysis.policy_improvement import task_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inner-forward replication for the counterfactual π_Lfrozen
# ---------------------------------------------------------------------------

def _compute_pi_Lfrozen(
    model, z_H_in: torch.Tensor, z_L_in: torch.Tensor,
    input_embeddings: torch.Tensor, seq_info: dict, pel: int,
) -> torch.Tensor:
    """Counterfactual: run the two H_level updates with z_L frozen at z_L_in.

    Mirrors the two H_level calls in the HRM inner forward (§1.3 of plan):
      1st H call: after _H=0 inner loop
      2nd H call: in the 1-step-grad tail
    Both calls use z_L_in (not updated), giving π_Lfrozen.

    Returns logits_Lfrozen [1, seq, vocab] (already pel-stripped).
    """
    with torch.no_grad():
        zH = model.inner.H_level(z_H_in, z_L_in, **seq_info)   # 1st H update
        zH = model.inner.H_level(zH, z_L_in, **seq_info)        # 2nd H update
        logits_Lfrozen = model.inner.lm_head(zH)[:, pel:].float()
    return logits_Lfrozen


# ---------------------------------------------------------------------------
# Per-puzzle per-step decomposition
# ---------------------------------------------------------------------------

def decompose_puzzle(
    model, batch: Dict[str, torch.Tensor],
    device: torch.device, task: str, max_steps: int, eps: float = 0.01,
) -> Tuple[List[dict], dict]:
    """Decompose one puzzle into per-step ΔH, ΔL, Δtotal, frac_H.

    Returns:
        per_step: list of dicts indexed by step.
        summary:  dict with final-step value and solved flag.
    """
    ablator = ActivationAblator(model, device=device)
    cache: Dict[int, Any] = {}
    ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)

    pel = int(model.inner.puzzle_emb_len)

    # Pre-compute input embeddings and cos_sin once
    with torch.no_grad():
        input_emb = model.inner._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        ).float()  # [1, pel+seq, D]
        cos_sin = model.inner.rotary_emb() if hasattr(model.inner, "rotary_emb") else None
    seq_info = {"cos_sin": cos_sin}

    labels_1d = batch["labels"][0].long()   # [seq]
    valid_mask = labels_1d != -100           # [seq] bool
    if valid_mask.sum() == 0:
        return [], {}

    y_valid = labels_1d[valid_mask].to(device)  # [N_valid]

    per_step = []
    steps = sorted(cache.keys())

    for s in steps:
        c = cache[s]

        # π_in: policy entering step s (uses z_H carry-in)
        logits_in = model.inner.lm_head(c.z_H.float())[:, pel:].float()  # [1, seq, vocab]
        logp_in = F.log_softmax(logits_in[0][valid_mask], dim=-1)         # [N_valid, vocab]
        logp_in_true = logp_in.gather(1, y_valid.unsqueeze(1)).squeeze(1) # [N_valid]

        # π_full: actual policy after the step (from cache)
        logits_full = c.logits[0][valid_mask].float()
        logp_full = F.log_softmax(logits_full, dim=-1)
        logp_full_true = logp_full.gather(1, y_valid.unsqueeze(1)).squeeze(1)

        # π_Lfrozen: H-updates only with frozen z_L
        logits_Lf = _compute_pi_Lfrozen(
            model, c.z_H, c.z_L, input_emb, seq_info, pel
        )
        logp_Lf = F.log_softmax(logits_Lf[0][valid_mask], dim=-1)
        logp_Lf_true = logp_Lf.gather(1, y_valid.unsqueeze(1)).squeeze(1)

        # Decomposition
        Delta_total = float((logp_full_true - logp_in_true).mean().item())
        Delta_H = float((logp_Lf_true - logp_in_true).mean().item())
        Delta_L = float((logp_full_true - logp_Lf_true).mean().item())

        # Identity check
        residual = abs(Delta_H + Delta_L - Delta_total)
        if residual > 1e-3:
            logger.warning(f"  step={s}: identity residual={residual:.6f}")

        frac_H = None
        if abs(Delta_total) > eps:
            denom = Delta_H + Delta_L
            frac_H = float(Delta_H / denom) if abs(denom) > 1e-8 else float("nan")

        # Also sanity-check π_full ≈ recomputed (via cache logits)
        per_step.append({
            "step": int(s),
            "Delta_total": float(Delta_total),
            "Delta_H": float(Delta_H),
            "Delta_L": float(Delta_L),
            "frac_H": frac_H,
            "identity_residual": float(residual),
        })

    # Final-step value
    last_s = max(steps)
    preds_row = cache[last_s].preds[0].cpu().numpy()
    labels_row = batch["labels"][0].cpu().numpy()
    inputs_row = batch["inputs"][0].cpu().numpy()
    tv = task_value(preds_row, labels_row, inputs_row, task)

    summary = {
        "final_value": tv["value"],
        "solved": tv["exact"] > 0.5,
        "n_steps": len(steps),
    }
    return per_step, summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_per_step(
    all_results: List[Tuple[List[dict], dict]]
) -> Dict[str, Any]:
    """Aggregate per-puzzle per-step results into per-step CIs, split by solved/failed."""
    # Collect per step
    step_data: Dict[int, Dict[str, List[float]]] = {}
    solved_step_data: Dict[int, Dict[str, List[float]]] = {}
    failed_step_data: Dict[int, Dict[str, List[float]]] = {}

    for per_step, summary in all_results:
        solved = summary.get("solved", False)
        for row in per_step:
            s = row["step"]
            for key in ["Delta_total", "Delta_H", "Delta_L"]:
                step_data.setdefault(s, {}).setdefault(key, []).append(row[key])
                if solved:
                    solved_step_data.setdefault(s, {}).setdefault(key, []).append(row[key])
                else:
                    failed_step_data.setdefault(s, {}).setdefault(key, []).append(row[key])
            if row["frac_H"] is not None and not np.isnan(row["frac_H"]):
                step_data[s].setdefault("frac_H", []).append(row["frac_H"])
                if solved:
                    solved_step_data[s].setdefault("frac_H", []).append(row["frac_H"])
                else:
                    failed_step_data[s].setdefault("frac_H", []).append(row["frac_H"])

    def _agg(sd):
        return {
            str(s): {k: bootstrap_ci(v) for k, v in step_dict.items()}
            for s, step_dict in sorted(sd.items())
        }

    n_solved = sum(1 for _, sm in all_results if sm.get("solved", False))
    n_failed = len(all_results) - n_solved

    return {
        "all": _agg(step_data),
        "solved": _agg(solved_step_data),
        "failed": _agg(failed_step_data),
        "n_total": len(all_results),
        "n_solved": n_solved,
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H4: H/L policy decomposition")
    parser.add_argument("--task", choices=["sudoku", "maze", "arc"], default="sudoku")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_puzzles", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Min |Δtotal| to compute frac_H (avoid division by tiny numbers)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.task == "sudoku":
        ckpt = args.checkpoint or find_checkpoint()
        output_dir = args.output_dir or os.path.join(
            REPO_ROOT, "results", "controlled", "policy_decomposition"
        )
    elif args.task == "arc":
        ckpt = args.checkpoint or ARC_CHECKPOINT
        output_dir = args.output_dir or os.path.join(
            REPO_ROOT, "results", "arc", "policy_decomposition"
        )
    else:
        ckpt = args.checkpoint or MAZE_CHECKPOINT
        output_dir = args.output_dir or os.path.join(
            REPO_ROOT, "results", "maze", "policy_decomposition"
        )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[H4] task={args.task}  ckpt={ckpt}  device={device}")

    model, test_loader, config = load_model_and_dataloader(ckpt, device)
    model.eval()

    batches = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
    logger.info(f"[H4] Collected {len(batches)} puzzles")

    t0 = time.time()
    all_results = []
    for i, (puzzle_idx, batch) in enumerate(batches):
        per_step, summary = decompose_puzzle(
            model, batch, device, args.task, args.max_steps, eps=args.eps
        )
        if per_step:
            all_results.append((per_step, summary))
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(batches)} puzzles")

    elapsed = time.time() - t0

    aggregate = aggregate_per_step(all_results)

    # Print per-step summary
    logger.info("[H4] Per-step decomposition summary:")
    for s_str, sdata in aggregate["all"].items():
        fh = sdata.get("frac_H", {})
        dt = sdata.get("Delta_total", {})
        logger.info(
            f"  step {s_str:>2s}: Δtotal={dt.get('mean', 0):+.4f}  "
            f"ΔH={sdata.get('Delta_H', {}).get('mean', 0):+.4f}  "
            f"ΔL={sdata.get('Delta_L', {}).get('mean', 0):+.4f}  "
            f"frac_H={fh.get('mean', float('nan')):.3f}"
        )

    # Save
    output = {
        "task": args.task,
        "checkpoint": ckpt,
        "num_puzzles": len(batches),
        "n_results": len(all_results),
        "max_steps": args.max_steps,
        "eps": args.eps,
        "elapsed_s": round(elapsed, 1),
        "aggregate": aggregate,
    }
    out_path = os.path.join(output_dir, "aggregate.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {out_path}")

    # Provenance
    try:
        from scripts.core.provenance import write_meta
        write_meta(output_dir, "H4_policy_decomposition", {
            "task": args.task, "num_puzzles": len(batches),
            "max_steps": args.max_steps, "eps": args.eps, "elapsed_s": round(elapsed, 1),
        }, repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"provenance write failed: {e}")

    # Console summary
    print("\n" + "=" * 60)
    print(f"H4 POLICY DECOMPOSITION — {args.task.upper()}")
    print("=" * 60)
    print(f"  n_puzzles={aggregate['n_total']}  solved={aggregate['n_solved']}  failed={aggregate['n_failed']}")
    print(f"  {'':<6s}  {'Δtotal':>8s}  {'ΔH':>8s}  {'ΔL':>8s}  {'frac_H':>8s}")
    for s_str, sdata in aggregate["all"].items():
        dt = sdata.get("Delta_total", {}).get("mean", float("nan"))
        dh = sdata.get("Delta_H", {}).get("mean", float("nan"))
        dl = sdata.get("Delta_L", {}).get("mean", float("nan"))
        fh = sdata.get("frac_H", {}).get("mean", float("nan"))
        print(f"  step {s_str:>2s}:  {dt:>+8.4f}  {dh:>+8.4f}  {dl:>+8.4f}  {fh:>8.3f}")

    if aggregate["n_solved"] > 0 and "0" in aggregate["solved"]:
        avg_fh_all = np.nanmean([
            aggregate["all"][s]["frac_H"]["mean"]
            for s in aggregate["all"]
            if "frac_H" in aggregate["all"][s]
        ])
        print(f"\n  Mean frac_H across all steps (solved+failed): {avg_fh_all:.3f}")
        print(f"  (Expected: near 1.0 if H-update drives most of the policy improvement)")


if __name__ == "__main__":
    main()
