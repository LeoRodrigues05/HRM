#!/usr/bin/env python3
"""Paper-style mechanistic figures for ARC-AGI-2, mirroring the Sudoku/Maze set.

This recreates the *paper's* signature interpretability figures (the ones built for
Sudoku/Maze in docs/ICMLWorkshop2026_HRM_Interpretability_Final.pdf) using the ARC
result JSONs this project produced from the Path-A adapted checkpoint
(checkpoints/arc2-adapted-evalonly/step_7391, frozen reasoning core, re-fit
puzzle_emb). Unlike scripts/arc/plot_arc_figures.py (which renders each metric as a
standalone panel), this script reproduces the paper's *combined* figure layouts so
ARC slots in beside Sudoku/Maze:

  fig_arc_readout_vs_causal      Paper Fig. 5  — (a) linear-probe readout accuracy vs
                                 majority baseline, (b) directed ablation of those
                                 same probe directions vs a random-control band.
                                 The dissociation: features are highly decodable but
                                 ablating them barely moves task accuracy.
  fig_arc_distributed_computation Paper Fig. 10 / maze causal-ladder — ablating the
                                 low-rank causal z_H subspace (top-r PCA) ≫ single
                                 probe directions ≈ random directions. Computation is
                                 distributed, with the causal mass in a small subspace.
  fig_arc_decodability_by_step   Paper Fig. 5a / Table 4 — per-target linear-probe
                                 decodability across ACT steps (the refinement trend).

Every number is read from disk (no hand-transcribed values) and echoed to stdout so
each bar is traceable. Missing inputs are skipped with a notice. Outputs go to a NEW
folder by default so the earlier ARC diagram sets are never overwritten.

Usage:
  python scripts/arc/plot_arc_paper_figures.py \
      --results_dir results/arc \
      --output_dir results/reports/arc_paper_figures
"""
from __future__ import annotations
import os, sys, json, argparse
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Targets that are BOTH probed and directly ablated, so Fig. 5's two panels share an
# x-axis (the paper shows Row/Col/Box/Cell-correct; ARC's analogues are these).
ABLATION_TARGETS = [
    "per_cell_correct", "colour_changed", "same_as_input",
    "input_inside_grid", "output_inside_grid", "is_eos", "is_object_boundary",
]
# Short, readable axis labels.
PRETTY = {
    "per_cell_correct": "cell\ncorrect",
    "colour_changed": "colour\nchanged",
    "same_as_input": "same as\ninput",
    "input_inside_grid": "input in\ngrid",
    "output_inside_grid": "output in\ngrid",
    "is_eos": "is\nEOS",
    "is_object_boundary": "object\nboundary",
    "input_colour": "input\ncolour",
    "output_colour": "output\ncolour",
}


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[plot_arc_paper] skip (missing): {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _probe_row(summary: dict, stream: str, step: int, target: str) -> Optional[dict]:
    """Find a reportable probe row by (stream, step, target)."""
    for v in summary.values():
        if (v.get("stream") == stream and int(v.get("step", -1)) == step
                and v.get("target") == target and v.get("status") == "ok"
                and v.get("score_mean") is not None):
            return v
    return None


# ---------------------------------------------------------------- Fig. 5 analogue
def fig_readout_vs_causal(probe_summary: dict, ablation: dict, output_dir: str,
                          stream: str = "z_H", step: int = 15):
    """Paper Fig. 5: readout (decodable) vs causal (ablation) for the same directions."""
    print(f"[fig_arc_readout_vs_causal] {stream} @ step {step}")
    targets = [t for t in ABLATION_TARGETS
               if _probe_row(probe_summary, stream, step, t) is not None
               and t in ablation and "probe_delta_colour_cell_acc" in ablation[t]]
    if not targets:
        print("  no shared probe+ablation targets — skipped")
        return

    acc = [float(_probe_row(probe_summary, stream, step, t)["score_mean"]) for t in targets]
    base = [float(_probe_row(probe_summary, stream, step, t).get("baseline_mean") or np.nan)
            for t in targets]
    probe_d = [ablation[t]["probe_delta_colour_cell_acc"]["mean"] * 100 for t in targets]
    sig = [ablation[t].get("significant_at_005", False) for t in targets]
    # Random control band (target-independent in the analysis) -> grey span in panel (b).
    rc = next(a["random_control_delta_colour_cell_acc"] for a in ablation.values()
              if isinstance(a, dict) and "random_control_delta_colour_cell_acc" in a)
    rc_lo, rc_hi = rc["ci_lower"] * 100, rc["ci_upper"] * 100

    labels = [PRETTY.get(t, t) for t in targets]
    x = np.arange(len(targets))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # (a) Readout: probe accuracy vs majority baseline
    axL.bar(x, acc, color="#3b6ea5", label="linear probe acc")
    axL.scatter(x, base, marker="_", s=420, color="#d1495b", zorder=3, label="majority baseline")
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylim(0, 1.02); axL.set_ylabel("held-out probe accuracy")
    axL.set_title(f"(a) Linear probe readout ({stream}, step {step})")
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, acc):
        axL.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=7)

    # (b) Causal: directed ablation Δ colour-cell acc, with random-control band
    axR.axhspan(rc_lo, rc_hi, color="0.82", label="random control range")
    axR.axhline(0, color="0.5", lw=0.8)
    bars = axR.bar(x, probe_d, color="#e76f51", label="ablate probe direction")
    for xi, v, s in zip(x, probe_d, sig):
        if s:
            axR.annotate("*", (xi, v), ha="center",
                         va="top" if v < 0 else "bottom", fontsize=13, color="#7a2f1d")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
    axR.set_ylabel("Δ colour-cell acc (ablated − base, %)")
    axR.set_title(f"(b) Directed ablation (causal effect, {stream})")
    axR.legend(fontsize=8, loc="lower left")
    axR.grid(axis="y", alpha=0.3)

    fig.suptitle("ARC-AGI-2: Readout ≠ Causally Relevant — features decode at "
                 f"{min(acc):.2f}–{max(acc):.2f} but ablating them moves task acc "
                 f"by ≤ {max(abs(min(probe_d)), abs(max(probe_d))):.2f}%",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    print("   targets:", targets)
    print("   probe acc:", [round(a, 3) for a in acc])
    print("   baseline :", [round(b, 3) for b in base])
    print(f"   ablation Δcolour-cell (%): {[round(d,3) for d in probe_d]}  "
          f"random band [{rc_lo:.3f}, {rc_hi:.3f}]%")
    _save(fig, output_dir, "fig_arc_readout_vs_causal")


# ---------------------------------------------------------------- Fig. 10 analogue
def fig_distributed_computation(subspace: dict, ablation: dict, output_dir: str):
    """Paper Fig. 10 / maze causal-ladder: low-rank z_H subspace ≫ probe ≈ random."""
    print("[fig_arc_distributed_computation]")
    if not subspace:
        print("  no causal_subspace data — skipped")
        return
    r = int(subspace.get("min_causal_rank") or subspace["meta"]["ranks"][0])
    top = subspace["curves"]["pca_top"]
    # Most damaging top-r PCA subspace ablation (the causal subspace).
    if str(r) not in top:
        r = min((int(k) for k in top), key=lambda k: top[str(k)]["mean"])
    sub_node = top[str(r)]
    sub_delta = sub_node["mean"] * 100
    sub_err = [[(sub_node["mean"] - sub_node["ci_lower"]) * 100],
               [(sub_node["ci_upper"] - sub_node["mean"]) * 100]]

    # Single probe directions: mean over ablated targets (each ≈ random); plus the
    # strongest single one for context.
    probe_deltas = [a["probe_delta_colour_cell_acc"]["mean"] * 100
                    for a in ablation.values()
                    if isinstance(a, dict) and "probe_delta_colour_cell_acc" in a]
    probe_mean = float(np.mean(probe_deltas)) if probe_deltas else np.nan
    probe_best = min(probe_deltas) if probe_deltas else np.nan  # most negative

    # Random directions (target-independent control).
    rc = next(a["random_control_delta_colour_cell_acc"]["mean"] * 100
              for a in ablation.values()
              if isinstance(a, dict) and "random_control_delta_colour_cell_acc" in a)

    labels = [f"top-{r} z_H\nsubspace", "best single\nprobe dir.",
              "mean probe\ndirections", "random\ndirections"]
    vals = [sub_delta, probe_best, probe_mean, rc]
    colors = ["#6a3d9a", "#e76f51", "#f4a582", "#bbbbbb"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(0, color="0.5", lw=0.8)
    bars = ax.bar(np.arange(len(labels)), vals, color=colors)
    ax.errorbar([0], [sub_delta], yerr=sub_err, fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ colour-cell acc (ablated − base, %)")
    ax.set_title("ARC-AGI-2: computation is deeply distributed\n"
                 f"causal mass sits in a rank-{r} z_H subspace, not in readable directions")
    for xi, v in zip(np.arange(len(labels)), vals):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center",
                    va="top" if v < 0 else "bottom", fontsize=8)
    print(f"   top-{r} subspace Δ={sub_delta:.2f}%  best probe Δ={probe_best:.2f}%  "
          f"mean probe Δ={probe_mean:.2f}%  random Δ={rc:.2f}%")
    print(f"   subspace_linearity={subspace.get('subspace_linearity')}  "
          f"baseline={subspace.get('baseline_mean')}")
    _save(fig, output_dir, "fig_arc_distributed_computation")


# ---------------------------------------------------------------- Fig. 5a / Table 4
def fig_decodability_by_step(probe_summary: dict, output_dir: str, stream: str = "z_H"):
    """Paper Fig. 5a / Table 4: per-target probe decodability across ACT steps."""
    print(f"[fig_arc_decodability_by_step] {stream}")
    targets = ["per_cell_correct", "output_colour", "colour_changed",
               "is_object_boundary", "same_as_input"]
    steps = sorted({int(v["step"]) for v in probe_summary.values()
                    if v.get("stream") == stream and v.get("status") == "ok"
                    and v.get("score_mean") is not None})
    if not steps:
        print("  no probed steps — skipped")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("viridis")
    plotted = 0
    for i, t in enumerate(targets):
        ys = []
        for s in steps:
            row = _probe_row(probe_summary, stream, s, t)
            ys.append(float(row["score_mean"]) if row else np.nan)
        if all(np.isnan(ys)):
            continue
        ax.plot(steps, ys, "-o", ms=4, color=cmap(i / max(1, len(targets) - 1)),
                label=t.replace("_", " "))
        plotted += 1
        print(f"   {t:<20} step0={ys[0]:.3f} -> step{steps[-1]}={ys[-1]:.3f}")
    if not plotted:
        plt.close(fig); print("  nothing plotted — skipped"); return
    ax.set_xlabel("ACT step"); ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title(f"ARC-AGI-2 linear-probe decodability across steps ({stream})")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    _save(fig, output_dir, f"fig_arc_decodability_by_step_{stream}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/arc")
    ap.add_argument("--output_dir", default="results/reports/arc_paper_figures")
    ap.add_argument("--stream", default="z_H")
    ap.add_argument("--step", type=int, default=15)
    args = ap.parse_args()

    rd = args.results_dir
    probe = _load(os.path.join(rd, "hardened/linear_probes/probe_summary.json"))
    abl = _load(os.path.join(rd, "hardened/directed_ablation/analysis.json"))
    sub = _load(os.path.join(rd, "causal_subspace/subspace_curve.json"))

    if probe and abl:
        fig_readout_vs_causal(probe, abl, args.output_dir, stream=args.stream, step=args.step)
    if sub and abl:
        fig_distributed_computation(sub, abl, args.output_dir)
    if probe:
        fig_decodability_by_step(probe, args.output_dir, stream=args.stream)
    print(f"[plot_arc_paper] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
