#!/usr/bin/env python3
"""H4 figures: per-step H/L decomposition and two-timescale comparison.

Reads:
  results/controlled/policy_decomposition/aggregate.json  (sudoku)
  results/maze/policy_decomposition/aggregate.json        (maze, optional)
  results/baseline_comparison/two_timescale/two_timescale.json

Writes figures to results/reports/policy_decomposition_figures/.

Usage:
    python scripts/analysis/plot_policy_decomposition.py
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


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Per-step decomposition (ΔH, ΔL, frac_H)
# ---------------------------------------------------------------------------

def fig_decomposition(data: dict, output_path: str, task: str, split: str = "all"):
    """Three-panel: Δtotal, ΔH + ΔL stacked, frac_H over steps."""
    agg = data.get("aggregate", {}).get(split, {})
    if not agg:
        return

    steps = sorted(agg.keys(), key=int)
    x = [int(s) for s in steps]

    dt_mean = [agg[s].get("Delta_total", {}).get("mean", float("nan")) for s in steps]
    dh_mean = [agg[s].get("Delta_H", {}).get("mean", float("nan")) for s in steps]
    dl_mean = [agg[s].get("Delta_L", {}).get("mean", float("nan")) for s in steps]
    fh_mean = [agg[s].get("frac_H", {}).get("mean", float("nan")) for s in steps]
    fh_lo = [agg[s].get("frac_H", {}).get("ci_lower", float("nan")) for s in steps]
    fh_hi = [agg[s].get("frac_H", {}).get("ci_upper", float("nan")) for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: Δtotal
    ax = axes[0]
    ax.bar(x, dt_mean, color="#457B9D", alpha=0.8, width=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("ACT step")
    ax.set_ylabel("Mean log-prob gain")
    ax.set_title(f"Δtotal per step ({task}, {split})", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: ΔH + ΔL stacked bars
    ax = axes[1]
    x_np = np.array(x)
    dh_np = np.array(dh_mean)
    dl_np = np.array(dl_mean)
    # Positive and negative parts for proper stacking
    ax.bar(x_np, dh_np, color="#E63946", alpha=0.8, width=0.7, label="ΔH (H-update)")
    ax.bar(x_np, dl_np, bottom=dh_np, color="#2A9D8F", alpha=0.8, width=0.7, label="ΔL (L-cycles)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("ACT step")
    ax.set_ylabel("Mean log-prob gain")
    ax.set_title(f"ΔH + ΔL breakdown ({task})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: frac_H
    ax = axes[2]
    fh_plot = [v if not np.isnan(v) else np.nan for v in fh_mean]
    ax.plot(x, fh_plot, color="#E63946", lw=2, marker="o", ms=5, label="frac_H")
    ax.fill_between(x,
                    [v if not np.isnan(v) else np.nan for v in fh_lo],
                    [v if not np.isnan(v) else np.nan for v in fh_hi],
                    alpha=0.2, color="#E63946")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="frac_H = 0.5")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_ylim(-0.2, 1.5)
    ax.set_xlabel("ACT step")
    ax.set_ylabel("frac_H = ΔH / (ΔH + ΔL)")
    ax.set_title(f"H-update share ({task})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    mean_fh = np.nanmean(fh_plot)
    ax.text(0.05, 0.95, f"mean frac_H = {mean_fh:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    n_total = data.get("n_results", data.get("aggregate", {}).get("n_total", "?"))
    n_solved = data.get("aggregate", {}).get("n_solved", "?")
    fig.suptitle(
        f"H4 Policy Decomposition — {task.upper()} ({split})  "
        f"[n={n_total}, solved={n_solved}]",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Two-timescale comparison (HRM vs UT)
# ---------------------------------------------------------------------------

def fig_two_timescale(ts_data: dict, output_path: str):
    """Side-by-side: HRM z_H/z_L change rates vs UT per-iteration rates."""
    hrm = ts_data.get("hrm")
    ut = ts_data.get("ut")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: HRM per-step change rates
    ax = axes[0]
    if hrm and "per_step" in hrm:
        per_step = hrm["per_step"]
        steps = sorted(per_step.keys(), key=int)
        x = [int(s) for s in steps]
        rates_H = [per_step[s]["rate_H"]["mean"] for s in steps]
        rates_L = [per_step[s]["rate_L"]["mean"] for s in steps]
        ax.plot(x, rates_H, color="#E63946", lw=2, marker="o", ms=4, label="z_H (slow)")
        ax.plot(x, rates_L, color="#2A9D8F", lw=2, marker="s", ms=4, label="z_L (fast)")
        ratio = hrm.get("mean_ratio_L_over_H", 0)
        ax.text(0.05, 0.95, f"L/H ratio = {ratio:.2f}×",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    else:
        ax.text(0.5, 0.5, "HRM data not available", transform=ax.transAxes, ha="center")
    ax.set_xlabel("ACT step")
    ax.set_ylabel("‖Δz‖ / ‖z‖")
    ax.set_title("HRM: z_H vs z_L change rate", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: UT per-iteration change rates
    ax = axes[1]
    if ut and "per_iteration_change_rate" in ut:
        per_iter = ut["per_iteration_change_rate"]
        iters = sorted(per_iter.keys(), key=int)
        x = [int(i) for i in iters]
        rates = [per_iter[i]["mean"] for i in iters]
        los = [per_iter[i]["ci_lower"] for i in iters]
        his = [per_iter[i]["ci_upper"] for i in iters]
        ax.plot(x, rates, color="#457B9D", lw=2, marker="o", ms=4, label="UT z (flat)")
        ax.fill_between(x, los, his, alpha=0.2, color="#457B9D")
        cv = ut.get("coeff_of_variation", 0)
        ax.text(0.05, 0.95, f"CV = {cv:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    else:
        tag = ts_data.get("tag", "ut")
        ax.text(0.5, 0.5, f"UT ({tag}) data not available\n(no checkpoint or failed)",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Inner iteration")
    ax.set_ylabel("‖Δz‖ / ‖z‖")
    ax.set_title("UT: per-iteration change rate (one ACT step)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Verdict annotation
    verdict = ts_data.get("verdict", {})
    if verdict:
        fig.text(0.5, -0.02,
                 verdict.get("conclusion", "")[:120],
                 ha="center", fontsize=8, style="italic", color="#555555")

    fig.suptitle("H4 Two-Timescale Analysis: HRM vs UT", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="H4 policy decomposition figures")
    parser.add_argument("--results_root", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        args.results_root, "reports", "policy_decomposition_figures"
    )
    os.makedirs(outdir, exist_ok=True)

    # Load results
    sudoku_data = _load(os.path.join(
        args.results_root, "controlled", "policy_decomposition", "aggregate.json"
    ))
    maze_data = _load(os.path.join(
        args.results_root, "maze", "policy_decomposition", "aggregate.json"
    ))
    arc_data = _load(os.path.join(
        args.results_root, "arc", "policy_decomposition", "aggregate.json"
    ))
    ts_data = _load(os.path.join(
        args.results_root, "baseline_comparison", "two_timescale", "two_timescale.json"
    ))

    if sudoku_data:
        for split in ["all", "solved", "failed"]:
            fig_decomposition(
                sudoku_data,
                os.path.join(outdir, f"fig_h4_decomp_sudoku_{split}.pdf"), "sudoku", split
            )
            fig_decomposition(
                sudoku_data,
                os.path.join(outdir, f"fig_h4_decomp_sudoku_{split}.png"), "sudoku", split
            )
    else:
        print("No Sudoku policy decomposition results found — run policy_decomposition.py first.")

    if maze_data:
        fig_decomposition(
            maze_data, os.path.join(outdir, "fig_h4_decomp_maze_all.pdf"), "maze", "all"
        )
        fig_decomposition(
            maze_data, os.path.join(outdir, "fig_h4_decomp_maze_all.png"), "maze", "all"
        )

    if arc_data:
        for split in ["all", "solved", "failed"]:
            fig_decomposition(
                arc_data, os.path.join(outdir, f"fig_h4_decomp_arc_{split}.pdf"), "arc", split
            )
            fig_decomposition(
                arc_data, os.path.join(outdir, f"fig_h4_decomp_arc_{split}.png"), "arc", split
            )

    if ts_data:
        fig_two_timescale(ts_data, os.path.join(outdir, "fig_h4_two_timescale.pdf"))
        fig_two_timescale(ts_data, os.path.join(outdir, "fig_h4_two_timescale.png"))
    else:
        print("No two-timescale results found — run two_timescale_baseline.py first.")

    print(f"\nAll H4 figures written to {outdir}")


if __name__ == "__main__":
    main()
