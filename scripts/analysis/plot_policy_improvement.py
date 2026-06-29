#!/usr/bin/env python3
"""Figures for Experiments A & D: HRM as a policy-improvement operator.

Reads results/maze/policy_improvement/aggregate.json
     results/controlled/policy_improvement/aggregate.json

Produces 5 figures (PDF + PNG) in results/reports/policy_improvement_figures/:
  figA1_value_logp    — per-step value & logp_true (Sudoku vs Maze)
  figA2_eq18          — per-step frac_eq18 (Eq. 18 satisfaction, both tasks)
  figA3_kl            — per-step kl_prev (compute signal, both tasks)
  figD2_earlyhalt     — value_at_halt curve vs ACT-step budget, both tasks
  figD3_halting       — s_act distribution vs s_star, per task

Every plotted value is echoed to stdout for traceability.

Usage:
    python scripts/analysis/plot_policy_improvement.py
    python scripts/analysis/plot_policy_improvement.py --outdir results/my_figures
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAZE_AGG    = os.path.join(REPO_ROOT, "results/maze/policy_improvement/aggregate.json")
SUDOKU_AGG  = os.path.join(REPO_ROOT, "results/controlled/policy_improvement/aggregate.json")

MAZE_COLOR   = "#e67e22"
SUDOKU_COLOR = "#2980b9"
MAZE_LABEL   = "Maze (valid_sg_path)"
SUDOKU_LABEL = "Sudoku (cell accuracy)"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save(fig: Any, outdir: str, name: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _steps_and_ci(per_step: List[dict], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (steps, mean, lo, hi) arrays for a key from per_step list."""
    steps, m, lo, hi = [], [], [], []
    for row in per_step:
        if key not in row:
            continue
        ci = row[key]
        steps.append(row["step"])
        m.append(ci["mean"])
        lo.append(ci["ci_lower"])
        hi.append(ci["ci_upper"])
    return np.array(steps), np.array(m), np.array(lo), np.array(hi)


# ---------------------------------------------------------------------------
# figA1 — value and logp_true per step
# ---------------------------------------------------------------------------

def fig_value_logp(maze: dict, sudoku: dict, outdir: str) -> None:
    print("\n[figA1_value_logp]")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for ax, key, ylabel, title_suffix in [
        (axes[0], "logp_true", "mean log P(y*) per cell", "log-likelihood of solution"),
        (axes[1], "value",     "task value",              "task-value (valid path / cell acc)"),
    ]:
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        for agg, color, label in [
            (sudoku["all"]["per_step"], SUDOKU_COLOR, SUDOKU_LABEL),
            (maze["all"]["per_step"],   MAZE_COLOR,   MAZE_LABEL),
        ]:
            s, m, lo, hi = _steps_and_ci(agg, key)
            print(f"  {label}  {key}: s0={m[0]:.4f}  s1={m[1]:.4f}  s15={m[-1]:.4f}")
            ax.plot(s, m, lw=1.8, marker="o", ms=3, color=color, label=label)
            ax.fill_between(s, lo, hi, alpha=0.18, color=color)
        ax.set_xlabel("ACT step")
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)
        ax.legend(fontsize=8)
        ax.set_xticks(range(0, 16, 2))

    fig.suptitle("HRM per-step policy improvement: solution log-likelihood & task value",
                 fontsize=10, y=1.01)
    save(fig, outdir, "figA1_value_logp")


# ---------------------------------------------------------------------------
# figA2 — frac_eq18
# ---------------------------------------------------------------------------

def fig_eq18(maze: dict, sudoku: dict, outdir: str) -> None:
    print("\n[figA2_eq18]")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0.5, color="0.6", lw=0.9, ls="--", label="chance (0.5)")
    ax.axhline(1.0, color="0.85", lw=0.6)

    for agg, color, label in [
        (sudoku["all"]["per_step"], SUDOKU_COLOR, SUDOKU_LABEL),
        (maze["all"]["per_step"],   MAZE_COLOR,   MAZE_LABEL),
    ]:
        s, m, lo, hi = _steps_and_ci(agg, "frac_eq18")
        if len(s) == 0:
            print(f"  [warn] no frac_eq18 data for {label}")
            continue
        print(f"  {label}  frac_eq18: s1={m[0]:.4f}  s8={m[min(7,len(m)-1)]:.4f}  s15={m[-1]:.4f}")
        ax.plot(s, m, lw=1.8, marker="o", ms=3.5, color=color, label=label)
        ax.fill_between(s, lo, hi, alpha=0.18, color=color)

    ax.set_xlabel("ACT step s  (advantage uses step s−1 as reference)")
    ax.set_ylabel("fraction of cells satisfying Eq. 18")
    ax.set_title(
        "Fraction of cells satisfying the policy-improvement inequality\n"
        "A(s, y*) > E_{a~π̂}[A(s, a)]  (Asadulaev et al. Eq. 18)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks(range(1, 16))
    ax.set_ylim(0.0, 1.05)
    save(fig, outdir, "figA2_eq18")


# ---------------------------------------------------------------------------
# figA3 — kl_prev (compute signal)
# ---------------------------------------------------------------------------

def fig_kl(maze: dict, sudoku: dict, outdir: str) -> None:
    print("\n[figA3_kl]")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0, color="0.7", lw=0.8)

    for agg, color, label in [
        (sudoku["all"]["per_step"], SUDOKU_COLOR, SUDOKU_LABEL),
        (maze["all"]["per_step"],   MAZE_COLOR,   MAZE_LABEL),
    ]:
        s, m, lo, hi = _steps_and_ci(agg, "kl_prev")
        if len(s) == 0:
            print(f"  [warn] no kl_prev data for {label}")
            continue
        print(f"  {label}  kl_prev: s1={m[0]:.6f}  s8={m[min(7,len(m)-1)]:.6f}  s15={m[-1]:.6f}")
        ax.plot(s, m, lw=1.8, marker="o", ms=3.5, color=color, label=label)
        ax.fill_between(s, lo, hi, alpha=0.18, color=color)

    ax.set_xlabel("ACT step s")
    ax.set_ylabel("mean KL(π_{s-1} ‖ π_s) per cell")
    ax.set_title(
        "Per-step compute signal: KL divergence from previous policy\n"
        "Large KL = active refinement; near-0 = dead compute"
    )
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 16))
    save(fig, outdir, "figA3_kl")


# ---------------------------------------------------------------------------
# figD2 — early-halt value curve
# ---------------------------------------------------------------------------

def fig_earlyhalt(maze: dict, sudoku: dict, outdir: str, tau: float = 0.01) -> None:
    print(f"\n[figD2_earlyhalt]  tau={tau}")
    tau_key = f"tau_{tau}"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for ax, agg, color, label, task_name in [
        (axes[0], sudoku, SUDOKU_COLOR, SUDOKU_LABEL, "Sudoku"),
        (axes[1], maze,   MAZE_COLOR,   MAZE_LABEL,   "Maze"),
    ]:
        d2 = agg["all"]["d2"].get(tau_key)
        if d2 is None:
            ax.set_title(f"{task_name}: data not found")
            continue

        vah_raw = d2["value_at_halt_per_step"]
        steps = list(range(len(vah_raw)))
        m = np.array([v.get("mean", float("nan")) for v in vah_raw])
        lo = np.array([v.get("ci_lower", float("nan")) for v in vah_raw])
        hi = np.array([v.get("ci_upper", float("nan")) for v in vah_raw])

        s_star_pp = d2["s_star_per_puzzle"]["mean"]
        speedup_pp = d2["speedup_per_puzzle"]["mean"]
        s_star_mc = d2["s_star_mean_curve"]
        speedup_mc = d2["speedup_mean_curve"]

        print(f"  {task_name}  s_star(per-puzzle)={s_star_pp:.2f}  "
              f"speedup={speedup_pp:.2f}x  "
              f"s_star(mean-curve)={s_star_mc}  speedup={speedup_mc:.2f}x")
        print(f"  {task_name}  value_at_halt: s0={m[0]:.4f}  s1={m[1]:.4f}  s15={m[-1]:.4f}")

        ax.plot(steps, m, lw=1.8, marker="o", ms=3.5, color=color)
        ax.fill_between(steps, lo, hi, alpha=0.18, color=color)
        ax.axvline(s_star_mc, color="0.35", lw=1.2, ls="--")
        ax.annotate(
            f"s*={s_star_mc}\n{speedup_mc:.1f}×",
            (s_star_mc, m[s_star_mc]),
            xytext=(s_star_mc + 0.5, m[s_star_mc] - 0.06),
            fontsize=8, color="0.25",
        )
        ax.set_xlabel("ACT steps used (halt after step s)")
        ax.set_ylabel("task value")
        ax.set_title(f"{task_name}: value vs compute (τ={tau})")
        ax.set_xticks(range(0, 16, 2))

    fig.suptitle("D2: Early-halt accuracy-vs-compute curve  (no retraining)",
                 fontsize=10, y=1.01)
    save(fig, outdir, "figD2_earlyhalt")


# ---------------------------------------------------------------------------
# figD3 — ACT halting distribution vs s_star
# ---------------------------------------------------------------------------

def fig_halting(maze: dict, sudoku: dict, outdir: str, tau: float = 0.01) -> None:
    print(f"\n[figD3_halting]  tau={tau}")
    tau_key = f"tau_{tau}"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for ax, agg, color, task_name in [
        (axes[0], sudoku, SUDOKU_COLOR, "Sudoku"),
        (axes[1], maze,   MAZE_COLOR,   "Maze"),
    ]:
        d3 = agg["all"]["d3"]
        d2 = agg["all"]["d2"].get(tau_key)
        if d2 is None:
            ax.set_title(f"{task_name}: data not found")
            continue

        hist = d3["s_act_hist"]
        steps = sorted(int(k) for k in hist)
        counts = np.array([hist[str(s)] for s in steps])
        total = counts.sum()
        fracs = counts / total if total > 0 else counts

        s_act_mean = d3["s_act"]["mean"]
        s_star_mc = d2["s_star_mean_curve"]
        s_star_pp = d2["s_star_per_puzzle"]["mean"]

        print(f"  {task_name}  s_act_mean={s_act_mean:.2f}  "
              f"s_star(mean-curve)={s_star_mc}  s_star(per-puzzle)={s_star_pp:.2f}")
        print(f"  {task_name}  s_act_hist={dict(zip(steps, counts.tolist()))}")

        ax.bar(steps, fracs, color=color, alpha=0.75, label="ACT halt dist.")
        ax.axvline(s_act_mean, color=color, lw=1.8, ls="--",
                   label=f"E[s_act]={s_act_mean:.1f}")
        ax.axvline(s_star_mc, color="0.3", lw=1.8, ls="-",
                   label=f"s* (mean-curve)={s_star_mc}")
        ax.axvline(s_star_pp, color="0.55", lw=1.2, ls=":",
                   label=f"s* (per-puzzle)={s_star_pp:.1f}")

        gap = s_act_mean - s_star_mc
        ax.set_xlabel("ACT step")
        ax.set_ylabel("fraction of puzzles")
        ax.set_title(
            f"{task_name}: where ACT halts vs where it could\n"
            f"gap = E[s_act] − s* = {gap:.1f} steps of dead compute"
        )
        ax.legend(fontsize=7)
        ax.set_xticks(range(0, 16, 2))

    fig.suptitle("D3: Does HRM's ACT halting track policy-improvement saturation?",
                 fontsize=10, y=1.01)
    save(fig, outdir, "figD3_halting")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze_agg", default=MAZE_AGG)
    ap.add_argument("--sudoku_agg", default=SUDOKU_AGG)
    ap.add_argument("--outdir", default=os.path.join(
        REPO_ROOT, "results/reports/policy_improvement_figures"))
    ap.add_argument("--tau", type=float, default=0.01,
                    help="tau for D2/D3 figures (must match a tau used during compute)")
    args = ap.parse_args()

    # Verify inputs exist
    missing = [p for p in (args.maze_agg, args.sudoku_agg) if not os.path.exists(p)]
    if missing:
        print(f"[plot_policy_improvement] missing aggregate files: {missing}")
        print("  Run policy_improvement.py for both tasks first.")
        sys.exit(1)

    maze   = load(args.maze_agg)
    sudoku = load(args.sudoku_agg)

    print(f"[plot_policy_improvement] maze:   n={maze.get('n_total')}  "
          f"solved={maze.get('n_solved')}")
    print(f"[plot_policy_improvement] sudoku: n={sudoku.get('n_total')}  "
          f"solved={sudoku.get('n_solved')}")
    print(f"[plot_policy_improvement] outdir: {args.outdir}")

    os.makedirs(args.outdir, exist_ok=True)

    fig_value_logp(maze, sudoku, args.outdir)
    fig_eq18(maze, sudoku, args.outdir)
    fig_kl(maze, sudoku, args.outdir)
    fig_earlyhalt(maze, sudoku, args.outdir, tau=args.tau)
    fig_halting(maze, sudoku, args.outdir, tau=args.tau)

    print(f"\n[plot_policy_improvement] done — 5 figures in {args.outdir}")


if __name__ == "__main__":
    main()
