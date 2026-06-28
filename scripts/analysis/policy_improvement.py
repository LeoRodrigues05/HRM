#!/usr/bin/env python3
"""Experiments A & D: HRM recursion as a policy-improvement operator.

Experiment A — per-step policy-improvement curves:
  logp_true, p_true, adv_true, frac_eq18, kl_prev per step.

Experiment D — dead compute & early halting:
  D1: n_alive steps (d_logp > eps or kl_prev > eps_kl)
  D2: value_at_halt_s curve, s_star, speedup (per-puzzle + from mean curve)
  D3: s_act (ACT halting head) vs s_star

One forward pass per puzzle covers all experiments.

Usage:
    python scripts/analysis/policy_improvement.py --task maze   --num_puzzles 500
    python scripts/analysis/policy_improvement.py --task sudoku --num_puzzles 500
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Any, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from scripts.controlled.controlled_common import (
    load_model_and_dataloader,
    collect_puzzles,
    bootstrap_ci,
    find_checkpoint,
)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.core.activation_patching import compute_metrics
from scripts.maze.maze_common import MAZE_CHECKPOINT, maze_prediction_metrics
from scripts.arc.arc_common import ARC_CHECKPOINT, arc_prediction_metrics


# ---------------------------------------------------------------------------
# Task-value helpers
# ---------------------------------------------------------------------------

def task_value(preds_row: np.ndarray, label_row: np.ndarray,
               input_row: Optional[np.ndarray], task: str) -> dict:
    """Compute task-specific value metrics for one puzzle (all 1-D arrays)."""
    if task == "maze":
        m = maze_prediction_metrics(preds_row, label_row, input_row)
        return {
            "value": float(m["valid_sg_path"]),
            "exact": float(m["exact_solved"]),
            "token_acc": float(m["token_acc"]),
        }
    if task == "arc":
        m = arc_prediction_metrics(preds_row, label_row, input_row)
        return {
            "value": float(m["colour_cell_acc"]),
            "exact": float(m["exact_solved"]),
            "token_acc": float(m["token_acc"]),
        }
    # sudoku
    p = torch.as_tensor(preds_row).reshape(1, -1)
    y = torch.as_tensor(label_row).reshape(1, -1)
    cm = compute_metrics(p, y)
    exact = float(cm["correct"] == cm["total_positions"])
    return {
        "value": float(cm["accuracy"]),
        "exact": exact,
        "token_acc": float(cm["accuracy"]),
    }


# ---------------------------------------------------------------------------
# Per-step policy statistics (Experiment A core)
# ---------------------------------------------------------------------------

def per_step_policy_stats(
    cache: Dict[int, "ActivationCache"],
    labels_row: torch.Tensor,       # [seq_len] on device
    inputs_row: Optional[torch.Tensor],  # [seq_len] on device or None for sudoku
    task: str,
    compute_step0_adv: bool,
    model: Any,
) -> List[dict]:
    """Return list of per-step dicts for one puzzle."""
    steps = sorted(cache.keys())
    y = labels_row.long()
    valid = y != -100
    yv = y.clamp(min=0)            # safe gather index; only used where valid=True

    # Pre-compute logp / p for all steps (float32 cast here, once)
    logps: Dict[int, torch.Tensor] = {}
    ps: Dict[int, torch.Tensor] = {}
    preds_t: Dict[int, torch.Tensor] = {}
    for s in steps:
        L = cache[s].logits[0].float()      # [seq_len, vocab]
        logps[s] = F.log_softmax(L, dim=-1)
        ps[s] = logps[s].exp()
        preds_t[s] = cache[s].preds[0].long()

    y_np = labels_row.cpu().numpy()
    inp_np = inputs_row.cpu().numpy() if inputs_row is not None else None

    rows: List[dict] = []
    for s in steps:
        logp_true_per_cell = logps[s].gather(-1, yv[:, None]).squeeze(-1)   # [seq_len]
        p_true_per_cell = ps[s].gather(-1, yv[:, None]).squeeze(-1)         # [seq_len]

        p_np = preds_t[s].cpu().numpy()
        tv = task_value(p_np, y_np, inp_np, task)

        row: dict = {
            "step": s,
            "logp_true": float(logp_true_per_cell[valid].mean()),
            "p_true":    float(p_true_per_cell[valid].mean()),
            "value":     tv["value"],
            "exact":     tv["exact"],
            "token_acc": tv["token_acc"],
            "q_halt":    float(cache[s].q_halt_logits.item()),
            "q_continue": float(cache[s].q_continue_logits.item()),
        }

        # Determine the reference policy (π̂) for this step
        ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if s >= 1:
            ref = (logps[s - 1], ps[s - 1])
        elif compute_step0_adv:
            # Step-0: reference = lm_head(z_H_init), the learned initial state
            try:
                pel = model.inner.puzzle_emb_len
                L0pre = model.inner.lm_head(cache[0].z_H)[:, pel:].float()[0]  # [seq_len, vocab]
                ref = (F.log_softmax(L0pre, -1), F.softmax(L0pre, -1))
            except Exception as exc:
                print(f"  [warn] step-0 advantage failed: {exc}")

        if ref is not None:
            logp_ref, p_ref = ref
            # A_s(j,a) = log π⁺(a|j) - log π̂(a|j)
            A = logps[s] - logp_ref                                  # [seq_len, vocab]
            # KL(π̂ || π⁺)[j] = Σ_a π̂(a|j) * (log π̂(a|j) - log π⁺(a|j)) >= 0
            kl = (p_ref * (logp_ref - logps[s])).sum(-1)             # [seq_len], >=0
            A_true = A.gather(-1, yv[:, None]).squeeze(-1)           # advantage on y*
            E_adv = -kl                                               # E_{a~π̂}[A], per cell

            row.update({
                "adv_true":  float(A_true[valid].mean()),
                "kl_prev":   float(kl[valid].mean()),
                # Eq. 18: fraction of valid cells where A(y*) > E_{a~π̂}[A]
                "frac_eq18": float((A_true[valid] > E_adv[valid]).float().mean()),
            })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# D1 – dead/alive step identification
# ---------------------------------------------------------------------------

def compute_d1_alive(per_step: List[dict], eps: float, eps_kl: float) -> dict:
    """Count alive steps by d_logp and kl_prev thresholds."""
    alive_logp: List[bool] = []
    alive_kl: List[bool] = []
    d_logp_list: List[float] = []
    for i in range(1, len(per_step)):
        d = per_step[i]["logp_true"] - per_step[i - 1]["logp_true"]
        d_logp_list.append(d)
        alive_logp.append(d > eps)
        if "kl_prev" in per_step[i]:
            alive_kl.append(per_step[i]["kl_prev"] > eps_kl)
    return {
        "n_alive_logp": int(sum(alive_logp)),
        "n_alive_kl": int(sum(alive_kl)) if alive_kl else None,
        "d_logp_per_step": [round(v, 6) for v in d_logp_list],
    }


# ---------------------------------------------------------------------------
# D2 – early-halt accuracy-vs-compute curve
# ---------------------------------------------------------------------------

def compute_d2_earlyhalt(per_step: List[dict], tau: float) -> dict:
    """Value_at_halt curve and s_star for one puzzle."""
    value_at_halt = [row["value"] for row in per_step]
    T = len(per_step)
    final_value = value_at_halt[-1]
    threshold = final_value - tau
    s_star = T - 1
    for s, v in enumerate(value_at_halt):
        if v >= threshold:
            s_star = s
            break
    return {
        "value_at_halt": value_at_halt,
        "s_star": s_star,
        "speedup": T / (s_star + 1),
        "final_value": final_value,
    }


# ---------------------------------------------------------------------------
# D3 – ACT halting step
# ---------------------------------------------------------------------------

def compute_d3_sact(per_step: List[dict]) -> int:
    """First step where q_halt > q_continue (HRM's own halting criterion)."""
    for row in per_step:
        if row["q_halt"] > row["q_continue"]:
            return int(row["step"])
    return int(per_step[-1]["step"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_steps(records: List[dict], max_steps: int) -> List[dict]:
    """Per-step bootstrap_ci over all records."""
    agg_keys = [
        "logp_true", "p_true", "value", "exact", "token_acc",
        "adv_true", "kl_prev", "frac_eq18", "q_halt", "q_continue",
    ]
    result = []
    for s in range(max_steps):
        row: dict = {"step": s}
        for k in agg_keys:
            vals = [
                r["per_step"][s][k]
                for r in records
                if s < len(r["per_step"]) and k in r["per_step"][s]
            ]
            if vals:
                row[k] = bootstrap_ci(vals)
        result.append(row)
    return result


def _agg_group(records: List[dict], max_steps: int, tau_values: List[float]) -> dict:
    """Aggregate per_step + D1/D2/D3 for a group of puzzle records."""
    if not records:
        return {"n": 0}

    per_step_agg = _agg_steps(records, max_steps)

    # D1
    n_alive_logp = [r["d1"]["n_alive_logp"] for r in records]
    n_alive_kl = [r["d1"]["n_alive_kl"] for r in records if r["d1"]["n_alive_kl"] is not None]
    d1_agg = {
        "n_alive_logp": bootstrap_ci(n_alive_logp),
        "n_alive_kl": bootstrap_ci(n_alive_kl) if n_alive_kl else None,
    }

    # D2 — one dict per tau
    d2_agg: dict = {}
    for tau in tau_values:
        key = f"tau_{tau}"
        per_s_star = [r["d2"][key]["s_star"] for r in records]
        per_speedup = [r["d2"][key]["speedup"] for r in records]
        # value_at_halt curve: bootstrap_ci per step
        vah_per_step = []
        for s in range(max_steps):
            vals = [
                r["d2"][key]["value_at_halt"][s]
                for r in records
                if s < len(r["d2"][key]["value_at_halt"])
            ]
            vah_per_step.append(bootstrap_ci(vals) if vals else {})
        # s_star from the population mean curve
        mean_vah = [v.get("mean", 0.0) for v in vah_per_step]
        final_mean = mean_vah[-1] if mean_vah else 0.0
        thresh = final_mean - tau
        s_star_mean_curve = max_steps - 1
        for s, v in enumerate(mean_vah):
            if v >= thresh:
                s_star_mean_curve = s
                break
        d2_agg[key] = {
            "s_star_per_puzzle": bootstrap_ci(per_s_star),
            "speedup_per_puzzle": bootstrap_ci(per_speedup),
            "s_star_mean_curve": s_star_mean_curve,
            "speedup_mean_curve": max_steps / (s_star_mean_curve + 1),
            "value_at_halt_per_step": vah_per_step,
        }

    # D3
    s_act_vals = [r["d3"]["s_act"] for r in records]
    d3_agg = {
        "s_act": bootstrap_ci(s_act_vals),
        "s_act_hist": {str(s): s_act_vals.count(s) for s in range(max_steps)},
    }

    return {
        "n": len(records),
        "per_step": per_step_agg,
        "d1": d1_agg,
        "d2": d2_agg,
        "d3": d3_agg,
    }


def build_aggregate(
    all_records: List[dict],
    max_steps: int,
    tau_values: List[float],
    task: str,
    args_dict: dict,
) -> dict:
    """Build the full aggregate.json content."""
    # Split by whether the puzzle was solved at the final step
    solved = [r for r in all_records if r["per_step"][-1].get("exact", 0.0) >= 0.5]
    failed = [r for r in all_records if r["per_step"][-1].get("exact", 0.0) < 0.5]

    return {
        "task": task,
        "n_total": len(all_records),
        "n_solved": len(solved),
        "n_failed": len(failed),
        "max_steps": max_steps,
        "tau_values": tau_values,
        "run_args": args_dict,
        "all":    _agg_group(all_records, max_steps, tau_values),
        "solved": _agg_group(solved,      max_steps, tau_values),
        "failed": _agg_group(failed,      max_steps, tau_values),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiments A & D: policy-improvement operator in HRM"
    )
    ap.add_argument("--task", required=True, choices=["maze", "sudoku", "arc"])
    ap.add_argument("--checkpoint", default=None,
                    help="Path to checkpoint (default: task-specific)")
    ap.add_argument("--num_puzzles", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: results/<task>/policy_improvement)")
    ap.add_argument("--compute_step0_adv", dest="compute_step0_adv",
                    action="store_true", default=True,
                    help="Compute step-0 advantage via lm_head(z_H_init) (default: on)")
    ap.add_argument("--no_step0_adv", dest="compute_step0_adv", action="store_false",
                    help="Disable step-0 advantage computation")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="Alive threshold for d_logp (D1, default 0.01)")
    ap.add_argument("--eps_kl", type=float, default=1e-3,
                    help="Alive threshold for kl_prev (D1, default 1e-3)")
    ap.add_argument("--tau", type=float, default=0.01,
                    help="Primary tolerance for s_star (D2, default 0.01)")
    args = ap.parse_args()

    # Resolve defaults
    if args.output_dir is None:
        if args.task == "maze":
            args.output_dir = os.path.join(REPO_ROOT, "results/maze/policy_improvement")
        elif args.task == "arc":
            args.output_dir = os.path.join(REPO_ROOT, "results/arc/policy_improvement")
        else:
            args.output_dir = os.path.join(REPO_ROOT, "results/controlled/policy_improvement")
    else:
        args.output_dir = os.path.join(REPO_ROOT, args.output_dir)

    if args.checkpoint is None:
        args.checkpoint = {
            "maze": MAZE_CHECKPOINT, "arc": ARC_CHECKPOINT,
        }.get(args.task, None) or find_checkpoint()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[policy_improvement] task={args.task}  device={device}  "
          f"n={args.num_puzzles}  max_steps={args.max_steps}")
    print(f"[policy_improvement] checkpoint={args.checkpoint}")
    print(f"[policy_improvement] output_dir={args.output_dir}")

    model, test_loader, _cfg = load_model_and_dataloader(args.checkpoint, device)
    ablator = ActivationAblator(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.num_puzzles)
    print(f"[policy_improvement] collected {len(puzzles)} puzzles")

    tau_values = [0.0, args.tau, args.tau * 2] if args.tau > 0 else [0.0, 0.01, 0.02]
    tau_values = sorted(set(tau_values))  # dedup

    per_puzzle_path = os.path.join(args.output_dir, "per_puzzle.jsonl")
    all_records: List[dict] = []
    t0 = time.time()

    with open(per_puzzle_path, "w") as out_f:
        for i, (idx, batch) in enumerate(puzzles):
            cache: Dict[int, ActivationCache] = {}
            ablator.run_and_cache_activations(batch, cache, max_steps=args.max_steps)
            steps = sorted(cache.keys())
            if not steps:
                continue

            labels_row = batch["labels"][0]    # [seq_len]
            inputs_row = batch["inputs"][0] if args.task == "maze" else None

            per_step = per_step_policy_stats(
                cache, labels_row, inputs_row, args.task,
                args.compute_step0_adv, model,
            )

            d1 = compute_d1_alive(per_step, args.eps, args.eps_kl)

            d2 = {}
            for tau in tau_values:
                d2[f"tau_{tau}"] = compute_d2_earlyhalt(per_step, tau)

            d3 = {"s_act": compute_d3_sact(per_step)}

            rec = {
                "puzzle_idx": int(idx),
                "per_step": per_step,
                "d1": d1,
                "d2": d2,
                "d3": d3,
            }
            out_f.write(json.dumps(rec) + "\n")
            all_records.append(rec)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                last_val = per_step[-1]["value"]
                print(f"  [{i+1}/{len(puzzles)}]  elapsed={elapsed:.1f}s  "
                      f"value={last_val:.3f}  s_act={d3['s_act']}")

    print(f"[policy_improvement] {len(all_records)} puzzles done; aggregating ...")

    if not all_records:
        print("[policy_improvement] no results — exiting")
        return

    max_steps_actual = max(len(r["per_step"]) for r in all_records)
    agg = build_aggregate(all_records, max_steps_actual, tau_values, args.task, vars(args))

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[policy_improvement] wrote {agg_path}")

    # Provenance
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "policy_improvement_A_D", vars(args))
    except Exception:
        pass

    # Human-readable summary
    ag = agg["all"]
    last_ps = ag["per_step"][-1]
    print("\n=== Summary ===")
    print(f"  task={args.task}  n_total={agg['n_total']}  "
          f"n_solved={agg['n_solved']}  n_failed={agg['n_failed']}")
    val_m = last_ps.get("value", {}).get("mean", float("nan"))
    logp_m = last_ps.get("logp_true", {}).get("mean", float("nan"))
    print(f"  final step: value={val_m:.4f}  logp_true={logp_m:.4f}")

    d1_agg = ag["d1"]
    na_logp = d1_agg["n_alive_logp"]["mean"]
    na_kl = d1_agg["n_alive_kl"]["mean"] if d1_agg["n_alive_kl"] else float("nan")
    print(f"  n_alive_logp(eps={args.eps})={na_logp:.2f}  "
          f"n_alive_kl(eps_kl={args.eps_kl})={na_kl:.2f}")

    primary_key = f"tau_{args.tau}"
    d2p = ag["d2"][primary_key]
    s_star_pp = d2p["s_star_per_puzzle"]["mean"]
    speedup_pp = d2p["speedup_per_puzzle"]["mean"]
    s_star_mc = d2p["s_star_mean_curve"]
    speedup_mc = d2p["speedup_mean_curve"]
    print(f"  s_star(tau={args.tau}): per-puzzle={s_star_pp:.2f}  "
          f"mean-curve={s_star_mc}  "
          f"speedup: per-puzzle={speedup_pp:.2f}x  mean-curve={speedup_mc:.2f}x")

    s_act_m = ag["d3"]["s_act"]["mean"]
    s_act_std = ag["d3"]["s_act"]["std"]
    print(f"  ACT s_act: mean={s_act_m:.2f} ± {s_act_std:.2f}")

    # frac_eq18 at step 1 (first step with advantage)
    if len(ag["per_step"]) > 1:
        eq18_s1 = ag["per_step"][1].get("frac_eq18", {}).get("mean", float("nan"))
        print(f"  frac_eq18 at step 1: {eq18_s1:.4f}  (>0.5 during alive phase is the Eq.18 signature)")

    print("[policy_improvement] done")


if __name__ == "__main__":
    main()
