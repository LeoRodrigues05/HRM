"""scripts/plot_controlled_experiments.py

Publication-quality figures for controlled experiments.

Reads aggregate JSON files from each experiment and produces:
  - Fig 1: Ablation step sensitivity (z_H vs z_L)
  - Fig 2: Freeze crossover curves (z_H vs z_L)
  - Fig 3: Time-shift transfer heatmap or bar chart
  - Fig 4: Directed ablation probe vs random controls
  - Fig 5: Combined 2x2 summary panel

Usage:
    python scripts/plot_controlled_experiments.py \\
        --results_root results/controlled \\
        --output_dir results/controlled/figures
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_json(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def setup_style():
    """Publication style defaults."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_ablation(agg_H: Dict, agg_L: Dict, output_dir: str):
    """Fig 1: Per-step ablation delta for z_H and z_L."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, agg, color, marker in [
        ("$z_H$ ablation", agg_H, "#d62728", "o"),
        ("$z_L$ ablation", agg_L, "#1f77b4", "s"),
    ]:
        steps_data = agg.get("per_step_ablation", {})
        steps = sorted(int(s) for s in steps_data.keys())
        means = [steps_data[str(s) if str(s) in steps_data else s]["delta_accuracy"]["mean"]
                 for s in steps]
        ci_lo = [steps_data[str(s) if str(s) in steps_data else s]["delta_accuracy"]["ci_lower"]
                 for s in steps]
        ci_hi = [steps_data[str(s) if str(s) in steps_data else s]["delta_accuracy"]["ci_upper"]
                 for s in steps]

        ax.plot(steps, means, f"-{marker}", color=color, label=label, markersize=5, linewidth=1.5)
        ax.fill_between(steps, ci_lo, ci_hi, alpha=0.15, color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ACT Step Ablated")
    ax.set_ylabel("$\\Delta$ Accuracy (ablated $-$ baseline)")
    ax.set_title("Single-Step Zero Ablation: $z_H$ vs $z_L$")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = os.path.join(output_dir, "fig1_ablation_step_sensitivity.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def plot_freeze(agg: Dict, output_dir: str):
    """Fig 2: Freeze crossover curves."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for level, color, marker, label in [
        ("freeze_H", "#d62728", "o", "Freeze $z_H$ after step $k$"),
        ("freeze_L", "#1f77b4", "s", "Freeze $z_L$ after step $k$"),
    ]:
        data = agg.get(level, {})
        steps = sorted(int(s) for s in data.keys())
        means = [data[str(s) if str(s) in data else s]["delta_accuracy"]["mean"] for s in steps]
        ci_lo = [data[str(s) if str(s) in data else s]["delta_accuracy"]["ci_lower"] for s in steps]
        ci_hi = [data[str(s) if str(s) in data else s]["delta_accuracy"]["ci_upper"] for s in steps]

        ax.plot(steps, means, f"-{marker}", color=color, label=label, markersize=5, linewidth=1.5)
        ax.fill_between(steps, ci_lo, ci_hi, alpha=0.15, color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-0.05, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Freeze After Step $k$")
    ax.set_ylabel("$\\Delta$ Accuracy (frozen $-$ baseline)")
    ax.set_title("Freeze Crossover: $z_H$ vs $z_L$")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = os.path.join(output_dir, "fig2_freeze_crossover.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def plot_time_shift(agg: Dict, output_dir: str):
    """Fig 3: Time-shift transfer results (bar chart for fixed mode)."""
    per_pair = agg.get("per_pair", {})
    if not per_pair:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Separate into two sweeps based on pattern
    pairs = list(per_pair.items())
    # Sort by (donor, recipient)
    pairs.sort(key=lambda x: (x[1].get("donor_step", 0), x[1].get("recipient_step", 0)))

    labels = [k for k, _ in pairs]
    means = [v["delta_accuracy"]["mean"] for _, v in pairs]
    ci_lo = [v["delta_accuracy"]["ci_lower"] for _, v in pairs]
    ci_hi = [v["delta_accuracy"]["ci_upper"] for _, v in pairs]
    errors_lo = [m - lo for m, lo in zip(means, ci_lo)]
    errors_hi = [hi - m for m, hi in zip(means, ci_hi)]

    colors = ["#2ca02c" if m > 0 else "#d62728" for m in means]
    x = range(len(labels))
    ax.bar(x, means, yerr=[errors_lo, errors_hi], color=colors, alpha=0.7,
           capsize=2, edgecolor="gray", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_xlabel("Donor → Recipient Step")
    ax.set_ylabel("$\\Delta$ Accuracy")
    ax.set_title("Time-Shift Transfer: $z_H$ from Donor to Recipient Step")

    path = os.path.join(output_dir, "fig3_time_shift_transfer.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def plot_time_shift_matrix(results_root: str, output_dir: str):
    """Fig 3b: Full transfer matrix heatmap (if matrix mode was run)."""
    matrix_path = os.path.join(results_root, "time_shift", "transfer_matrix.npy")
    if not os.path.exists(matrix_path):
        return

    matrix = np.load(matrix_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xlabel("Recipient Step")
    ax.set_ylabel("Donor Step")
    ax.set_title("Time-Shift Transfer Matrix ($\\Delta$ Accuracy)")
    plt.colorbar(im, ax=ax, label="$\\Delta$ Accuracy")

    path = os.path.join(output_dir, "fig3b_transfer_matrix.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def plot_directed_ablation(analysis: Dict, output_dir: str):
    """Fig 4: Probe vs random controls."""
    # Filter to only entries with proper structure
    probe_entries = {k: v for k, v in analysis.items()
                     if "probe_delta_accuracy" in v and "cohens_d" in v}
    if not probe_entries:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: delta accuracy comparison
    names = list(probe_entries.keys())
    probe_means = [probe_entries[n]["probe_delta_accuracy"]["mean"] for n in names]
    probe_ci_lo = [probe_entries[n]["probe_delta_accuracy"]["ci_lower"] for n in names]
    probe_ci_hi = [probe_entries[n]["probe_delta_accuracy"]["ci_upper"] for n in names]
    rand_means = [probe_entries[n]["random_control_delta_accuracy"]["mean"] for n in names]
    rand_ci_lo = [probe_entries[n]["random_control_delta_accuracy"]["ci_lower"] for n in names]
    rand_ci_hi = [probe_entries[n]["random_control_delta_accuracy"]["ci_upper"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width/2, probe_means,
            yerr=[[m - lo for m, lo in zip(probe_means, probe_ci_lo)],
                  [hi - m for m, hi in zip(probe_means, probe_ci_hi)]],
            width=width, label="Probe direction", color="#d62728", alpha=0.7, capsize=3)
    ax1.bar(x + width/2, rand_means,
            yerr=[[m - lo for m, lo in zip(rand_means, rand_ci_lo)],
                  [hi - m for m, hi in zip(rand_means, rand_ci_hi)]],
            width=width, label="Random controls (avg)", color="#7f7f7f", alpha=0.7, capsize=3)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("$\\Delta$ Accuracy")
    ax1.set_title("Probe vs Random Direction Ablation")
    ax1.legend()

    # Add significance stars
    for i, name in enumerate(names):
        p = probe_entries[name]["paired_p_value"]
        if p < 0.001:
            ax1.text(i, min(probe_means[i], rand_means[i]) - 0.01, "***", ha="center", fontsize=10)
        elif p < 0.01:
            ax1.text(i, min(probe_means[i], rand_means[i]) - 0.01, "**", ha="center", fontsize=10)
        elif p < 0.05:
            ax1.text(i, min(probe_means[i], rand_means[i]) - 0.01, "*", ha="center", fontsize=10)

    # Right: Cohen's d
    d_vals = [probe_entries[n]["cohens_d"] for n in names]
    colors = ["#d62728" if abs(d) > 0.8 else "#ff7f0e" if abs(d) > 0.5
              else "#2ca02c" if abs(d) > 0.2 else "#7f7f7f" for d in d_vals]
    ax2.barh(x, d_vals, color=colors, alpha=0.7)
    ax2.axvline(0, color="gray", linewidth=0.8)
    ax2.axvline(-0.8, color="red", linewidth=0.5, linestyle=":", alpha=0.5, label="Large effect")
    ax2.axvline(0.8, color="red", linewidth=0.5, linestyle=":", alpha=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Cohen's $d$ (probe $-$ random)")
    ax2.set_title("Effect Size")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_directed_ablation.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot controlled experiment results")
    parser.add_argument("--results_root", type=str, default="results/controlled",
                        help="Root directory containing experiment subdirs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures (default: results_root/figures)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_root, "figures")
    os.makedirs(args.output_dir, exist_ok=True)

    setup_style()

    print("=" * 70)
    print("PLOTTING CONTROLLED EXPERIMENT RESULTS")
    print("=" * 70)

    # Fig 1: Ablation
    abl_H = load_json(os.path.join(args.results_root, "ablation", "zH", "aggregate.json"))
    abl_L = load_json(os.path.join(args.results_root, "ablation", "zL", "aggregate.json"))
    if abl_H and abl_L:
        plot_ablation(abl_H, abl_L, args.output_dir)
    else:
        print("  SKIP Fig 1 (ablation): aggregate files not found")

    # Fig 2: Freeze
    freeze_agg = load_json(os.path.join(args.results_root, "freeze", "aggregate.json"))
    if freeze_agg:
        plot_freeze(freeze_agg, args.output_dir)
    else:
        print("  SKIP Fig 2 (freeze): aggregate file not found")

    # Fig 3: Time-shift
    ts_agg = load_json(os.path.join(args.results_root, "time_shift", "aggregate.json"))
    if ts_agg:
        plot_time_shift(ts_agg, args.output_dir)
    else:
        print("  SKIP Fig 3 (time-shift): aggregate file not found")

    # Fig 3b: Transfer matrix (optional)
    plot_time_shift_matrix(args.results_root, args.output_dir)

    # Fig 4: Directed ablation
    directed_analysis = load_json(
        os.path.join(args.results_root, "directed_ablation", "analysis.json")
    )
    if directed_analysis:
        plot_directed_ablation(directed_analysis, args.output_dir)
    else:
        print("  SKIP Fig 4 (directed ablation): analysis file not found")

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
