#!/usr/bin/env python3
"""H2 figures: damage-vs-rank curves and alignment bar plot.

Reads results/controlled/causal_subspace/subspace_curve.json (and maze variant)
and writes figures to results/reports/causal_subspace_figures/.

Usage:
    python scripts/analysis/plot_causal_subspace.py
    python scripts/analysis/plot_causal_subspace.py --task maze
"""
from __future__ import annotations
import os, sys, json, argparse
from typing import Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


ORDERING_STYLES = {
    "pca_top":  {"color": "#E63946", "lw": 2.0, "label": "PCA-top (causal candidate)", "zorder": 5},
    "pca_bottom": {"color": "#457B9D", "lw": 1.5, "ls": "--", "label": "PCA-bottom"},
    "random":   {"color": "#888888", "lw": 1.5, "ls": ":", "label": "Random (mean ± CI)"},
    "probe_sae": {"color": "#2A9D8F", "lw": 1.5, "ls": "-.", "label": "Probe+SAE directions"},
}


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig_damage_curves(results: dict, output_path: str, task: str):
    """Damage-vs-rank curve: one line per ordering."""
    curves = results.get("curves", {})
    delta_full = results.get("delta_full", None)

    fig, ax = plt.subplots(figsize=(6, 4))

    for ord_name, crv in curves.items():
        if ord_name not in ORDERING_STYLES:
            continue
        style = ORDERING_STYLES[ord_name]
        rs = sorted([int(k) for k in crv.keys()])
        means = [crv[str(r)]["mean"] for r in rs]
        los = [crv[str(r)].get("ci_lower", crv[str(r)]["mean"]) for r in rs]
        his = [crv[str(r)].get("ci_upper", crv[str(r)]["mean"]) for r in rs]

        ax.plot(rs, means, color=style["color"], lw=style.get("lw", 1.5),
                ls=style.get("ls", "-"), label=style["label"], zorder=style.get("zorder", 3))
        ax.fill_between(rs, los, his, alpha=0.15, color=style["color"])

    if delta_full is not None:
        ax.axhline(delta_full, color="black", ls="--", lw=1.0, label=f"Full ablation ({delta_full:+.3f})")

    r_star = results.get("r_star", {}).get("ninety", {}).get("pca_top", None)
    if r_star is not None:
        ax.axvline(r_star, color="#E63946", ls=":", lw=1.0, alpha=0.6, label=f"r*(0.9) = {r_star}")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Subspace rank $r$", fontsize=11)
    ax.set_ylabel("Mean Δvalue (ablated − baseline)", fontsize=11)
    ax.set_title(f"H2 Damage-vs-Rank Curve — {task.upper()}", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def fig_alignment_bar(results: dict, output_path: str, task: str):
    """Alignment bar plot: probe+SAE energy vs random control in causal subspace."""
    aln = results.get("alignment", {})
    r_star = aln.get("r_star", "?")

    labels, values, errs = [], [], []

    if "probe_sae_energy" in aln:
        labels.append("Probe+SAE")
        values.append(aln["probe_sae_energy"])
        errs.append(0.0)

    labels.append("Random (ctrl)")
    values.append(aln.get("random_control_energy_mean", 0.0))
    errs.append(aln.get("random_control_energy_std", 0.0))

    if not values:
        print(f"  No alignment data — skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(4, 3))
    colors = ["#2A9D8F", "#888888"][:len(labels)]
    bars = ax.bar(labels, values, color=colors, yerr=errs, capsize=4, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Projection energy", fontsize=11)
    ax.set_title(f"Readable basis alignment in causal subspace\n"
                 f"(r*={r_star}, {task.upper()})", fontsize=10)
    ax.axhline(r_star / 512 if isinstance(r_star, int) else 0, color="gray",
               ls="--", lw=0.8, alpha=0.5, label="Chance (r/D)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="H2 figures")
    parser.add_argument("--task", choices=["sudoku", "maze", "arc"], default="sudoku")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.input_dir is None:
        _default_in = {
            "sudoku": os.path.join(REPO_ROOT, "results", "controlled", "causal_subspace"),
            "maze": os.path.join(REPO_ROOT, "results", "maze", "causal_subspace"),
            "arc": os.path.join(REPO_ROOT, "results", "arc", "causal_subspace"),
        }
        args.input_dir = _default_in[args.task]

    outdir = args.outdir or os.path.join(REPO_ROOT, "results", "reports", "causal_subspace_figures")
    os.makedirs(outdir, exist_ok=True)

    results = _load_json(os.path.join(args.input_dir, "subspace_curve.json"))
    if results is None:
        print(f"No results found at {args.input_dir}/subspace_curve.json — run causal_subspace.py first.")
        return

    tag = args.task
    fig_damage_curves(results, os.path.join(outdir, f"fig_damage_curves_{tag}.pdf"), args.task)
    fig_damage_curves(results, os.path.join(outdir, f"fig_damage_curves_{tag}.png"), args.task)
    fig_alignment_bar(results, os.path.join(outdir, f"fig_alignment_{tag}.pdf"), args.task)
    fig_alignment_bar(results, os.path.join(outdir, f"fig_alignment_{tag}.png"), args.task)

    print(f"\nAll H2 figures written to {outdir}")


if __name__ == "__main__":
    main()
