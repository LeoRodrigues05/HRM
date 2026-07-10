#!/usr/bin/env python3
"""Combined cross-task (Sudoku / Maze / ARC) figures for the AAAI reframe.

Renders the multi-task versions of the paper's mechanistic figures, reading ONLY
from on-disk result JSONs (no hand-transcribed numbers) and echoing every plotted
value to stdout for traceability. Each task/series is skipped gracefully if its
aggregate is missing, so this is safe to run before the ARC SAE/patching/freeze runs
finish (it just renders fewer series, then re-render once they land).

Figures -> results/reports/paper_aaai_figures/ (new folder; nothing overwritten):
  fig_crosstask_freeze      Finding 1: freeze-z_H-after-k damage vs k, 3 tasks
                            (Sudoku decays progressively; Maze/ARC ~flat = static plan)
  fig_crosstask_patching    Finding 1: cross-puzzle z_H patch damage vs donor step
                            (decisive only at the readout step)
  fig_crosstask_sae_ladder  Finding 3: SAE top-k ≈ random-k ≈ probe ≈ random ≈ 0,
                            grouped by task (computation is distributed)

Usage:
  python scripts/plotting/plot_crosstask_figures.py \
      --output_dir results/reports/paper_aaai_figures
"""
from __future__ import annotations
import os, sys, json, argparse
from typing import Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_COLOR = {"Sudoku": "#2980b9", "Maze": "#e67e22", "ARC": "#6a3d9a"}


def _load(path: str) -> Optional[dict]:
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.exists(p):
        print(f"  skip (missing): {path}")
        return None
    with open(p) as f:
        return json.load(f)


def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _ci(node: dict) -> Tuple[float, float, float]:
    """(mean, lo, hi) in percent from a bootstrap_ci dict."""
    m = node.get("mean", 0.0)
    return m * 100, node.get("ci_lower", m) * 100, node.get("ci_upper", m) * 100


# --------------------------------------------------------------- Finding 1: freeze
def _freeze_series(agg, metric_path) -> Optional[Tuple[List[int], List[float]]]:
    """Extract (ks, delta% ) for freeze_H. metric_path picks the metric node."""
    if not agg or "freeze_H" not in agg:
        return None
    ks, ys = [], []
    for k_str, row in sorted(agg["freeze_H"].items(), key=lambda kv: int(kv[0])):
        node = metric_path(row)
        if node is None:
            continue
        ks.append(int(k_str)); ys.append(node.get("mean", 0.0) * 100)
    return (ks, ys) if ks else None


def fig_freeze(output_dir):
    print("[fig_crosstask_freeze]")
    series = {
        "Sudoku": (_load("results/controlled/freeze/aggregate.json"),
                   lambda r: r.get("delta_accuracy")),
        "Maze": (_load("results/maze/hardened/freeze_controlled/aggregate.json"),
                 lambda r: (r.get("maze_metric_deltas") or {}).get("valid_sg_path")),
        "ARC": (_load("results/arc/freeze/aggregate.json"),
                lambda r: r.get("delta_accuracy")),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="0.7", lw=0.8)
    any_plotted = False
    for task, (agg, mp) in series.items():
        s = _freeze_series(agg, mp)
        if not s:
            continue
        ks, ys = s
        ax.plot(ks, ys, "-o", ms=4, color=TASK_COLOR[task], label=task)
        any_plotted = True
        print(f"   {task:7s} freeze_H Δ: k0={ys[0]:+.2f}%  k{ks[-1]}={ys[-1]:+.2f}%")
    if not any_plotted:
        plt.close(fig); print("   nothing to plot"); return
    ax.set_xlabel("freeze z_H after step k"); ax.set_ylabel("Δ task metric (%)")
    ax.set_title("Finding 1: task-dependent refinement depth\n"
                 "Sudoku decays progressively; Maze/ARC z_H is a static plan")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, output_dir, "fig_crosstask_freeze")


# --------------------------------------------------------------- Finding 1: patching
def fig_patching(output_dir):
    print("[fig_crosstask_patching]")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="0.7", lw=0.8)
    any_plotted = False

    sd = _load("results/patching/patching_full_steps/aggregate.json")
    if sd and "patch_H" in sd:
        node = sd["patch_H"]
        steps = sorted((int(k) for k in node), key=int)
        ys = [node[str(s)]["delta_accuracy"]["mean"] * 100 for s in steps]
        ax.plot(steps, ys, "-^", ms=4, color=TASK_COLOR["Sudoku"], label="Sudoku (cell acc)")
        any_plotted = True
        print(f"   Sudoku patch Δ: s{steps[0]}={ys[0]:+.2f}%  s{steps[-1]}={ys[-1]:+.2f}%")

    mz = _load("results/maze/hardened/patching_full_steps/aggregate.json")
    if mz:
        node = (mz.get("maze_metric_deltas_by_group_level_step") or {}).get(
            "valid_sg_path", {}).get("all", {}).get("H", {})
        if node:
            steps = sorted((int(k) for k in node), key=int)
            ys = [node[str(s)]["mean"] * 100 for s in steps]
            ax.plot(steps, ys, "-s", ms=4, color=TASK_COLOR["Maze"], label="Maze (valid path)")
            any_plotted = True
            print(f"   Maze patch Δ: s{steps[0]}={ys[0]:+.2f}%  s{steps[-1]}={ys[-1]:+.2f}%")

    arc = _load("results/arc/patching_full_steps/aggregate.json")
    if arc and "patch_H" in arc:
        node = arc["patch_H"]
        steps = sorted((int(k) for k in node), key=int)
        ys = [node[str(s)]["delta_accuracy"]["mean"] * 100 for s in steps]
        ax.plot(steps, ys, "-o", ms=4, color=TASK_COLOR["ARC"], label="ARC (colour-cell)")
        any_plotted = True
        print(f"   ARC  patch Δ: s{steps[0]}={ys[0]:+.2f}%  s{steps[-1]}={ys[-1]:+.2f}%")

    if not any_plotted:
        plt.close(fig); print("   nothing to plot"); return
    ax.set_xlabel("donor z_H patched at step k"); ax.set_ylabel("Δ task metric (%)")
    ax.set_title("Finding 1: cross-puzzle z_H patch\n"
                 "Sudoku: destructive at every step; Maze/ARC: decisive only at the readout")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, output_dir, "fig_crosstask_patching")


# --------------------------------------------------------------- Finding 3: SAE ladder
SAE_CONDS = ["sae_top_features", "random_sae_features", "probe_directions", "random_directions"]
COND_LABEL = {"sae_top_features": "SAE top-50", "random_sae_features": "SAE rand-50",
              "probe_directions": "probe dirs", "random_directions": "random dirs"}


def _sae_vals(agg) -> Optional[Dict[str, float]]:
    if not agg or "conditions" not in agg:
        return None
    out = {}
    for c in SAE_CONDS:
        node = agg["conditions"].get(c, {})
        mk = next((k for k in node if k.startswith("mean_delta_")), None)
        out[c] = node.get(mk, float("nan")) * 100 if mk else float("nan")
    return out


def fig_sae_ladder(output_dir):
    print("[fig_crosstask_sae_ladder]")
    tasks = {
        "Sudoku": _load("results/sae_study/causal_ablation/aggregate.json"),
        "Maze": _load("results/maze/sae_study/causal_ablation/aggregate.json"),
        "ARC": _load("results/arc/sae_study/causal_ablation/aggregate.json"),
    }
    data = {t: _sae_vals(a) for t, a in tasks.items()}
    data = {t: v for t, v in data.items() if v is not None}
    if not data:
        print("   no SAE aggregates yet — skipped (re-run after ARC SAE completes)"); return

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhline(0, color="0.7", lw=0.8)
    n_t = len(data); w = 0.8 / len(SAE_CONDS)
    x = np.arange(n_t)
    cond_colors = ["#c0392b", "#e89", "#7a7", "#bbb"]
    for ci, cond in enumerate(SAE_CONDS):
        vals = [data[t][cond] for t in data]
        ax.bar(x + (ci - 1.5) * w, vals, w, label=COND_LABEL[cond], color=cond_colors[ci])
        print(f"   {cond:20s}: " + "  ".join(f"{t}={data[t][cond]:+.2f}%" for t in data))
    ax.set_xticks(x); ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel("Δ task metric (%)")
    ax.set_title("Finding 3: computation is distributed\n"
                 "SAE top-50 ≈ random-50 ≈ probe ≈ random (≪ full-state ablation)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, "fig_crosstask_sae_ladder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/reports/paper_aaai_figures")
    args = ap.parse_args()
    outdir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(REPO, args.output_dir)
    fig_freeze(outdir)
    fig_patching(outdir)
    fig_sae_ladder(outdir)
    print(f"[plot_crosstask] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
