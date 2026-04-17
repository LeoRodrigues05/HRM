#!/usr/bin/env python3
"""
Generate a comprehensive set of plots from z_H ablation experiment results.

Reads:
  - results/batch_ablation_zH/results_per_puzzle.jsonl
  - results/batch_ablation_zH/aggregate_stats.json
  - results/batch_ablation_zH/step_accuracy_matrix.json

Outputs PDF figures to results/ablation_plots/
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "batch_ablation_zH"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "ablation_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
PALETTE = sns.color_palette("colorblind")
BASELINE_COLOR = PALETTE[0]       # blue
ALL_STEPS_COLOR = PALETTE[3]      # red
SINGLE_STEP_COLORS = {
    "4":  PALETTE[1],   # orange
    "6":  PALETTE[2],   # green
    "8":  PALETTE[4],   # purple
    "10": PALETTE[5],   # brown
}
FIGSIZE = (10, 6)
DPI = 150


def load_data():
    """Load all three result files and return (per_puzzle_records, aggregate_stats, matrices)."""
    records = []
    with open(RESULTS_DIR / "results_per_puzzle.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    with open(RESULTS_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    with open(RESULTS_DIR / "step_accuracy_matrix.json") as f:
        mat = json.load(f)

    return records, agg, mat


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Accuracy Distribution: Baseline vs All-Steps Ablated
# ═══════════════════════════════════════════════════════════════════════════════
def plot_accuracy_distribution(records):
    baseline_acc = [r["baseline"]["final_accuracy"] for r in records]
    ablated_acc = [r["ablation_all_steps"]["final_accuracy"] for r in records]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, 51)

    ax.hist(baseline_acc, bins=bins, alpha=0.55, color=BASELINE_COLOR,
            label=f"Baseline (mean={np.mean(baseline_acc):.3f})", edgecolor="white", linewidth=0.5)
    ax.hist(ablated_acc, bins=bins, alpha=0.55, color=ALL_STEPS_COLOR,
            label=f"All-Steps Ablated (mean={np.mean(ablated_acc):.3f})", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Final Accuracy")
    ax.set_ylabel("Number of Puzzles")
    ax.set_title("Accuracy Distribution: Baseline vs z$_H$ All-Steps Ablation")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_accuracy_distribution.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "01_accuracy_distribution.png", dpi=DPI)
    plt.close(fig)
    print("  [1/9] Accuracy distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Mean Step-by-Step Accuracy Trajectories
# ═══════════════════════════════════════════════════════════════════════════════
def plot_step_trajectories(mat):
    steps = np.array(mat["steps"])
    baseline = np.array(mat["baseline_matrix"])       # (N, 16)
    allstep  = np.array(mat["allstep_ablated_matrix"]) # (N, 16)

    mean_bl = baseline.mean(axis=0)
    std_bl  = baseline.std(axis=0)
    mean_al = allstep.mean(axis=0)
    std_al  = allstep.std(axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Baseline
    ax.plot(steps, mean_bl, "-o", color=BASELINE_COLOR, markersize=5,
            label="Baseline", linewidth=2)
    ax.fill_between(steps, mean_bl - std_bl, mean_bl + std_bl,
                    color=BASELINE_COLOR, alpha=0.15)

    # All-steps ablated
    ax.plot(steps, mean_al, "-s", color=ALL_STEPS_COLOR, markersize=5,
            label="z$_H$ Ablated (all steps)", linewidth=2)
    ax.fill_between(steps, mean_al - std_al, mean_al + std_al,
                    color=ALL_STEPS_COLOR, alpha=0.15)

    # Single-step ablated
    for step_key in ["4", "6", "8", "10"]:
        sm = np.array(mat["single_step_ablated_matrices"][step_key])
        mean_sm = sm.mean(axis=0)
        ax.plot(steps, mean_sm, "--^", color=SINGLE_STEP_COLORS[step_key],
                markersize=4, label=f"Ablate step {step_key} only", linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Step-by-Step Accuracy Trajectory Under z$_H$ Ablation")
    ax.set_xticks(steps)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.45, 1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_step_trajectories.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "02_step_trajectories.png", dpi=DPI)
    plt.close(fig)
    print("  [2/9] Step trajectories")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Single-Step Impact Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_single_step_impact(agg):
    ranking = agg["single_step_impact_ranking"]
    steps  = [str(r["step"]) for r in ranking]
    deltas = [r["mean_accuracy_delta"] for r in ranking]
    broken = [r["mean_cells_broken"] for r in ranking]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [SINGLE_STEP_COLORS[s] for s in steps]

    # Accuracy delta bars
    bars1 = ax1.bar(steps, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_xlabel("Ablated Step")
    ax1.set_ylabel("Mean Accuracy Δ")
    ax1.set_title("Accuracy Impact by Ablated Step")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars1, deltas):
        ax1.text(bar.get_x() + bar.get_width()/2, v - 0.005,
                 f"{v:.3f}", ha="center", va="top", fontsize=10, fontweight="bold")

    # Cells broken bars
    bars2 = ax2.bar(steps, broken, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_xlabel("Ablated Step")
    ax2.set_ylabel("Mean Cells Broken")
    ax2.set_title("Cells Broken by Ablated Step")
    for bar, v in zip(bars2, broken):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Single-Step z$_H$ Ablation: Impact by Step (n=2,803 puzzles)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "03_single_step_impact.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "03_single_step_impact.png", dpi=DPI)
    plt.close(fig)
    print("  [3/9] Single-step impact bars")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Accuracy Delta Violin/Box Plot
# ═══════════════════════════════════════════════════════════════════════════════
def plot_accuracy_delta_violins(records):
    data, labels = [], []

    # All-steps
    for r in records:
        data.append(r["ablation_all_steps"]["accuracy_delta"])
        labels.append("All Steps")

    # Single-step
    for step_key in ["4", "6", "8", "10"]:
        for r in records:
            data.append(r["ablation_single_step"][step_key]["accuracy_delta"])
            labels.append(f"Step {step_key}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    order = ["All Steps", "Step 4", "Step 6", "Step 8", "Step 10"]
    palette = {
        "All Steps": ALL_STEPS_COLOR,
        "Step 4": SINGLE_STEP_COLORS["4"],
        "Step 6": SINGLE_STEP_COLORS["6"],
        "Step 8": SINGLE_STEP_COLORS["8"],
        "Step 10": SINGLE_STEP_COLORS["10"],
    }
    sns.violinplot(x=labels, y=data, order=order, hue=labels, hue_order=order,
                   palette=palette, ax=ax, inner="box", cut=0, linewidth=1.2,
                   saturation=0.8, legend=False)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Ablation Condition")
    ax.set_ylabel("Accuracy Δ (ablated – baseline)")
    ax.set_title("Distribution of Accuracy Changes Under z$_H$ Ablation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_accuracy_delta_violins.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "04_accuracy_delta_violins.png", dpi=DPI)
    plt.close(fig)
    print("  [4/9] Accuracy delta violins")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Cell Transition Stacked Bars
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cell_transitions(records):
    conditions = ["All Steps", "Step 4", "Step 6", "Step 8", "Step 10"]
    categories = ["stayed_correct", "stayed_wrong", "fixed", "broken"]
    cat_labels = ["Stayed Correct", "Stayed Wrong", "Fixed", "Broken"]
    cat_colors = [PALETTE[0], PALETTE[7], PALETTE[2], PALETTE[3]]

    means = {cat: [] for cat in categories}
    for cond in conditions:
        vals = {cat: [] for cat in categories}
        for r in records:
            if cond == "All Steps":
                ct = r["ablation_all_steps"]["cell_transitions"]
            else:
                step_key = cond.split()[-1]
                ct = r["ablation_single_step"][step_key]["cell_transitions"]
            for cat in categories:
                vals[cat].append(ct[cat])
        for cat in categories:
            means[cat].append(np.mean(vals[cat]))

    # Normalize to fractions of 81
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(conditions))
    width = 0.6
    bottoms = np.zeros(len(conditions))
    for cat, lbl, col in zip(categories, cat_labels, cat_colors):
        vals = np.array(means[cat])
        ax.bar(x, vals, width, bottom=bottoms, label=lbl, color=col,
               edgecolor="white", linewidth=0.5)
        # Label in the middle of the segment
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 2:  # only label visible segments
                ax.text(i, b + v/2, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean Number of Cells (out of 81)")
    ax.set_title("Cell Fate Under z$_H$ Ablation")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_cell_transitions.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "05_cell_transitions.png", dpi=DPI)
    plt.close(fig)
    print("  [5/9] Cell transitions")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Entropy & Confidence Change
# ═══════════════════════════════════════════════════════════════════════════════
def plot_entropy_confidence(records):
    conditions = ["All Steps", "Step 4", "Step 6", "Step 8", "Step 10"]
    ent_means, conf_means = [], []
    ent_stds, conf_stds = [], []

    for cond in conditions:
        ent_vals, conf_vals = [], []
        for r in records:
            if cond == "All Steps":
                ent_vals.append(r["ablation_all_steps"]["entropy"])
                conf_vals.append(r["ablation_all_steps"]["confidence"])
            else:
                step_key = cond.split()[-1]
                ent_vals.append(r["ablation_single_step"][step_key]["entropy"])
                conf_vals.append(r["ablation_single_step"][step_key]["confidence"])
        ent_means.append(np.mean(ent_vals))
        ent_stds.append(np.std(ent_vals))
        conf_means.append(np.mean(conf_vals))
        conf_stds.append(np.std(conf_vals))

    # Also get baseline values
    bl_ent = np.mean([r["baseline"]["entropy"] for r in records])
    bl_conf = np.mean([r["baseline"]["confidence"] for r in records])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(conditions))
    colors = [ALL_STEPS_COLOR] + [SINGLE_STEP_COLORS[s] for s in ["4", "6", "8", "10"]]

    # Entropy
    ax1.bar(x, ent_means, color=colors, edgecolor="white", linewidth=0.8,
            yerr=ent_stds, capsize=3, error_kw={"linewidth": 1})
    ax1.axhline(bl_ent, color=BASELINE_COLOR, linewidth=1.5, linestyle="--",
                label=f"Baseline ({bl_ent:.4f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=15)
    ax1.set_ylabel("Mean Entropy")
    ax1.set_title("Prediction Entropy")
    ax1.legend(fontsize=9)

    # Confidence
    ax2.bar(x, conf_means, color=colors, edgecolor="white", linewidth=0.8,
            yerr=conf_stds, capsize=3, error_kw={"linewidth": 1})
    ax2.axhline(bl_conf, color=BASELINE_COLOR, linewidth=1.5, linestyle="--",
                label=f"Baseline ({bl_conf:.4f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=15)
    ax2.set_ylabel("Mean Confidence")
    ax2.set_title("Prediction Confidence")
    ax2.legend(fontsize=9)

    fig.suptitle("Entropy & Confidence Under z$_H$ Ablation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "06_entropy_confidence.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "06_entropy_confidence.png", dpi=DPI)
    plt.close(fig)
    print("  [6/9] Entropy & confidence")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Heatmap: Mean Accuracy at Each Step Under Each Ablation Condition
# ═══════════════════════════════════════════════════════════════════════════════
def plot_step_accuracy_heatmap(mat):
    steps = mat["steps"]
    baseline_mean = np.array(mat["baseline_matrix"]).mean(axis=0)
    allstep_mean  = np.array(mat["allstep_ablated_matrix"]).mean(axis=0)

    rows = ["Baseline", "Ablate All Steps"]
    data = [baseline_mean, allstep_mean]

    for step_key in ["4", "6", "8", "10"]:
        sm = np.array(mat["single_step_ablated_matrices"][step_key])
        rows.append(f"Ablate Step {step_key}")
        data.append(sm.mean(axis=0))

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Reasoning Step")
    ax.set_title("Mean Accuracy at Each Step Under Different z$_H$ Ablation Conditions")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = "white" if v < 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_step_accuracy_heatmap.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "07_step_accuracy_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  [7/9] Step accuracy heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 8 — Baseline Accuracy vs Ablation Damage (scatter + trend)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_baseline_vs_damage(records):
    bl_acc = np.array([r["baseline"]["final_accuracy"] for r in records])
    all_delta = np.array([r["ablation_all_steps"]["accuracy_delta"] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Left panel: all-steps ablation --
    ax = axes[0]
    # Hex-bin for density (many overlapping points)
    hb = ax.hexbin(bl_acc, all_delta, gridsize=40, cmap="YlOrRd", mincnt=1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Baseline Accuracy")
    ax.set_ylabel("Accuracy Δ (all-steps ablation)")
    ax.set_title("All-Steps z$_H$ Ablation")
    fig.colorbar(hb, ax=ax, label="Count", shrink=0.8)

    # Trend line
    z = np.polyfit(bl_acc, all_delta, 1)
    p = np.poly1d(z)
    xs = np.linspace(bl_acc.min(), bl_acc.max(), 100)
    ax.plot(xs, p(xs), "--", color=BASELINE_COLOR, linewidth=2,
            label=f"Trend: slope={z[0]:.2f}")
    ax.legend(fontsize=9)

    # -- Right panel: single-step ablations overlay --
    ax2 = axes[1]
    for step_key in ["4", "6", "8", "10"]:
        deltas = np.array([r["ablation_single_step"][step_key]["accuracy_delta"] for r in records])
        ax2.scatter(bl_acc, deltas, s=3, alpha=0.15,
                    color=SINGLE_STEP_COLORS[step_key], label=f"Step {step_key}")
        z = np.polyfit(bl_acc, deltas, 1)
        p = np.poly1d(z)
        ax2.plot(xs, p(xs), "-", color=SINGLE_STEP_COLORS[step_key], linewidth=2)

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Baseline Accuracy")
    ax2.set_ylabel("Accuracy Δ (single-step ablation)")
    ax2.set_title("Single-Step z$_H$ Ablation")
    ax2.legend(fontsize=9, markerscale=5)

    fig.suptitle("Does Ablation Hurt High-Accuracy Puzzles More?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "08_baseline_vs_damage.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "08_baseline_vs_damage.png", dpi=DPI)
    plt.close(fig)
    print("  [8/9] Baseline vs damage scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 9 — Activation Norm Profiles Across Steps
# ═══════════════════════════════════════════════════════════════════════════════
def plot_activation_norms(records):
    """Compare mean z_H / z_L norms per step between baseline and single-step ablation."""
    n_steps = 16
    steps_arr = np.arange(n_steps)

    # Baseline norms
    zh_norms_bl = np.zeros((len(records), n_steps))
    zl_norms_bl = np.zeros((len(records), n_steps))
    for i, r in enumerate(records):
        for s in range(n_steps):
            norms = r["baseline"]["activation_norms"].get(str(s), {})
            zh_norms_bl[i, s] = norms.get("z_H_norm", 0)
            zl_norms_bl[i, s] = norms.get("z_L_norm", 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # z_H norms
    mean_zh = zh_norms_bl.mean(axis=0)
    std_zh  = zh_norms_bl.std(axis=0)
    ax1.plot(steps_arr, mean_zh, "-o", color=BASELINE_COLOR, linewidth=2,
             markersize=5, label="Baseline z$_H$")
    ax1.fill_between(steps_arr, mean_zh - std_zh, mean_zh + std_zh,
                     color=BASELINE_COLOR, alpha=0.15)

    # Single-step ablation norms (show how norms change after ablation point)
    for step_key in ["4", "6", "8", "10"]:
        zh_ab = np.zeros((len(records), n_steps))
        for i, r in enumerate(records):
            norms_dict = r["ablation_single_step"][step_key].get("activation_norms", {})
            for s in range(n_steps):
                norms = norms_dict.get(str(s), {})
                zh_ab[i, s] = norms.get("z_H_norm", 0)
        mean_ab = zh_ab.mean(axis=0)
        ax1.plot(steps_arr, mean_ab, "--", color=SINGLE_STEP_COLORS[step_key],
                 linewidth=1.5, label=f"Ablate step {step_key}")

    ax1.set_xlabel("Reasoning Step")
    ax1.set_ylabel("Mean ‖z$_H$‖")
    ax1.set_title("z$_H$ Norm Profile")
    ax1.legend(fontsize=8)
    ax1.set_xticks(steps_arr)

    # z_L norms  (baseline only — z_L isn't ablated in these experiments)
    mean_zl = zl_norms_bl.mean(axis=0)
    std_zl  = zl_norms_bl.std(axis=0)
    ax2.plot(steps_arr, mean_zl, "-o", color=BASELINE_COLOR, linewidth=2,
             markersize=5, label="Baseline z$_L$")
    ax2.fill_between(steps_arr, mean_zl - std_zl, mean_zl + std_zl,
                     color=BASELINE_COLOR, alpha=0.15)

    # z_L norms under single-step H-ablation (to see if ablating z_H affects z_L)
    for step_key in ["4", "6", "8", "10"]:
        zl_ab = np.zeros((len(records), n_steps))
        for i, r in enumerate(records):
            norms_dict = r["ablation_single_step"][step_key].get("activation_norms", {})
            for s in range(n_steps):
                norms = norms_dict.get(str(s), {})
                zl_ab[i, s] = norms.get("z_L_norm", 0)
        mean_ab = zl_ab.mean(axis=0)
        ax2.plot(steps_arr, mean_ab, "--", color=SINGLE_STEP_COLORS[step_key],
                 linewidth=1.5, label=f"Ablate z$_H$ step {step_key}")

    ax2.set_xlabel("Reasoning Step")
    ax2.set_ylabel("Mean ‖z$_L$‖")
    ax2.set_title("z$_L$ Norm Profile (z$_H$ ablation effect on z$_L$)")
    ax2.legend(fontsize=8)
    ax2.set_xticks(steps_arr)

    fig.suptitle("Activation Norm Profiles Across Reasoning Steps", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "09_activation_norms.pdf", dpi=DPI)
    fig.savefig(OUT_DIR / "09_activation_norms.png", dpi=DPI)
    plt.close(fig)
    print("  [9/9] Activation norms")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Loading data from {RESULTS_DIR} ...")
    records, agg, mat = load_data()
    print(f"  → {len(records)} puzzles loaded")
    print(f"Generating plots → {OUT_DIR}/\n")

    plot_accuracy_distribution(records)
    plot_step_trajectories(mat)
    plot_single_step_impact(agg)
    plot_accuracy_delta_violins(records)
    plot_cell_transitions(records)
    plot_entropy_confidence(records)
    plot_step_accuracy_heatmap(mat)
    plot_baseline_vs_damage(records)
    plot_activation_norms(records)

    print(f"\nDone — 9 figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
