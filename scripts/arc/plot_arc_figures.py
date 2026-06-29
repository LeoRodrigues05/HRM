#!/usr/bin/env python3
"""Consolidated figures for the ARC-AGI MI suite.

Reads the JSON artefacts produced by the ARC experiment drivers and renders
publication-style figures (mirrors scripts/maze/plot_consolidated_figures.py for
the maze suite). Each panel is skipped gracefully if its inputs are missing, so
this can be run after a partial suite.

Inputs (defaults under results/arc/):
  hardened/linear_probes/probe_summary.json       (linear probe decodability)
  hardened/linear_probes_mlp/probe_summary.json   (MLP probe + linear baseline)
  hardened/directed_ablation/analysis.json         (E9 causal validation)
  policy_improvement/aggregate.json                (per-step value, optional)

Usage
  python scripts/arc/plot_arc_figures.py --results_dir results/arc \
      --output_dir results/reports/arc_figures
"""
from __future__ import annotations
import os, sys, json, argparse
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[plot_arc] skip (missing): {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot_arc] wrote {output_dir}/{name}.png|pdf")


def plot_decodability(summary: dict, output_dir: str, stream: str = "z_H"):
    """Per-target decodability at the last probed step (binary/multiclass only)."""
    rows = [v for v in summary.values()
            if v.get("stream") == stream and v.get("reportable")]
    if not rows:
        print(f"[plot_arc] no reportable {stream} rows for decodability")
        return
    last_step = max(int(v["step"]) for v in rows)
    rows = [v for v in rows if int(v["step"]) == last_step and v.get("score_mean") is not None]
    rows.sort(key=lambda v: float(v["score_mean"]), reverse=True)
    names = [v["target"] for v in rows]
    scores = [float(v["score_mean"]) for v in rows]
    los = [float(v["score_mean"]) - float(v.get("score_ci_lower", v["score_mean"])) for v in rows]
    his = [float(v.get("score_ci_upper", v["score_mean"])) - float(v["score_mean"]) for v in rows]
    base = [float(v["baseline_mean"]) if v.get("baseline_mean") is not None else np.nan for v in rows]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.55), 4.2))
    xpos = np.arange(len(names))
    ax.bar(xpos, scores, yerr=[los, his], capsize=3, color="#3b6ea5", label="probe acc")
    ax.scatter(xpos, base, marker="_", s=320, color="#d1495b", label="majority baseline", zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ARC linear-probe decodability ({stream}, step {last_step})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, f"fig_arc_decodability_{stream}")


def plot_mlp_vs_linear(summary_mlp: dict, output_dir: str, stream: str = "z_H"):
    """MLP-minus-linear delta per target (non-linear structure not captured by a line)."""
    rows = [v for v in summary_mlp.values()
            if v.get("stream") == stream and v.get("delta_mlp_minus_linear") is not None]
    if not rows:
        print("[plot_arc] no MLP delta rows")
        return
    last_step = max(int(v["step"]) for v in rows)
    rows = [v for v in rows if int(v["step"]) == last_step]
    rows.sort(key=lambda v: float(v["delta_mlp_minus_linear"]), reverse=True)
    names = [v["target"] for v in rows]
    deltas = [float(v["delta_mlp_minus_linear"]) for v in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.55), 4.0))
    colors = ["#2a9d8f" if d >= 0 else "#999" for d in deltas]
    ax.bar(np.arange(len(names)), deltas, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MLP − linear (acc)")
    ax.set_title(f"ARC non-linear probe gain ({stream}, step {last_step})")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, f"fig_arc_mlp_minus_linear_{stream}")


def plot_directed_ablation(analysis: dict, output_dir: str, metric: str = "colour_cell_acc"):
    """Probe-direction vs random-direction ablation damage (E9 causal validation)."""
    key = f"probe_delta_{metric}"
    rkey = f"random_control_delta_{metric}"
    items = [(name, a) for name, a in analysis.items() if key in a]
    if not items:
        print("[plot_arc] no directed-ablation rows")
        return
    items.sort(key=lambda kv: kv[1][key]["mean"])
    names = [n for n, _ in items]
    probe = [a[key]["mean"] for _, a in items]
    rand = [a[rkey]["mean"] for _, a in items]
    sig = [a.get("significant_at_005", False) for _, a in items]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.5), 4.4))
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, probe, 0.4, label="probe direction", color="#e76f51")
    ax.bar(xpos + 0.2, rand, 0.4, label="random control", color="#bbb")
    for i, s in enumerate(sig):
        if s:
            ax.text(xpos[i] - 0.2, probe[i], "*", ha="center",
                    va="top" if probe[i] < 0 else "bottom", fontsize=12)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Δ {metric} (ablated − base)")
    ax.set_title("ARC directed ablation: probe vs random directions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, "fig_arc_directed_ablation")


def _mean(x):
    """Accept either a scalar or a bootstrap dict {'mean':..,'ci_lower':..}."""
    return x.get("mean") if isinstance(x, dict) else x


def plot_value_by_step(agg: dict, output_dir: str):
    """Per-step task value (policy-improvement). Reads policy_improvement aggregate:
    agg['all']['per_step'] is a list of {step, value:{mean,ci_*}, ...}. Falls back to
    older flat {per_step: {step: {value_mean}}} layouts."""
    per_step = (agg.get("all", {}) or {}).get("per_step") or agg.get("per_step") or agg.get("by_step")
    if not per_step:
        print("[plot_arc] no per-step value in aggregate")
        return

    if isinstance(per_step, list):                 # current format: list of per-step dicts
        rows = sorted(per_step, key=lambda r: int(r["step"]))
        steps = [int(r["step"]) for r in rows]
        vals = [_mean(r.get("value", r.get("value_mean"))) for r in rows]
        los = [r.get("value", {}).get("ci_lower") if isinstance(r.get("value"), dict) else None for r in rows]
        his = [r.get("value", {}).get("ci_upper") if isinstance(r.get("value"), dict) else None for r in rows]
    else:                                          # legacy dict format
        steps = sorted(int(k) for k in per_step.keys())
        vals = [_mean(per_step[str(s)].get("value", per_step[str(s)].get("value_mean"))) for s in steps]
        los = his = [None] * len(steps)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, vals, "-o", color="#3b6ea5", label="task value (colour-cell acc)")
    if all(l is not None for l in los):
        ax.fill_between(steps, los, his, color="#3b6ea5", alpha=0.2)
    ax.set_xlabel("ACT step")
    ax.set_ylabel("task value (colour-cell acc)")
    ax.set_title("ARC value by reasoning step (policy improvement)")
    ax.grid(alpha=0.3)
    _save(fig, output_dir, "fig_arc_value_by_step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/arc")
    ap.add_argument("--output_dir", default="results/reports/arc_figures")
    ap.add_argument("--streams", default="z_H,z_L")
    args = ap.parse_args()

    rd = args.results_dir
    streams = [s.strip() for s in args.streams.split(",") if s.strip()]

    lin = _load(os.path.join(rd, "hardened/linear_probes/probe_summary.json"))
    if lin:
        for s in streams:
            plot_decodability(lin, args.output_dir, stream=s)
    mlp = _load(os.path.join(rd, "hardened/linear_probes_mlp/probe_summary.json"))
    if mlp:
        for s in streams:
            plot_mlp_vs_linear(mlp, args.output_dir, stream=s)
    da = _load(os.path.join(rd, "hardened/directed_ablation/analysis.json"))
    if da:
        plot_directed_ablation(da, args.output_dir)
    agg = _load(os.path.join(rd, "policy_improvement/aggregate.json"))
    if agg:
        plot_value_by_step(agg, args.output_dir)
    print(f"[plot_arc] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
