#!/usr/bin/env python3
"""
Generate presentation-quality plots for all three interpretability experiments:
  1. z_H Ablation (existing data from batch_ablation_zH)
  2. Freeze z_H (new data from freeze_h)
  3. Time-Shift Patching (new data from time_shift)

Outputs to results/presentation_plots/

Usage:
    python scripts/plot_presentation.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = ROOT / "results" / "batch_ablation_zH"
FREEZE_DIR = ROOT / "results" / "freeze_h"
TIMESHIFT_DIR = ROOT / "results" / "time_shift"
OUT_DIR = ROOT / "results" / "presentation_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
PALETTE = sns.color_palette("colorblind")
FIGSIZE = (11, 6)
FIGSIZE_WIDE = (14, 6)
DPI = 200

# Consistent colors
C_BASELINE = PALETTE[0]   # blue
C_ABLATE = PALETTE[3]     # red
C_FREEZE = PALETTE[2]     # green
C_FORWARD = PALETTE[1]    # orange
C_BACKWARD = PALETTE[4]   # purple


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 1: Ablation — Step-accuracy trajectory (clean version)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_trajectories():
    """Step-by-step accuracy under baseline vs ablation conditions."""
    with open(ABLATION_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Extract trajectory from all-steps ablation (which has per_step_trajectory)
    all_steps_traj = agg["ablation_all_steps"]["per_step_trajectory"]
    steps_sorted = sorted(all_steps_traj.keys(), key=int)
    x = [int(s) for s in steps_sorted]

    # Baseline trajectory
    y_base = [all_steps_traj[s]["baseline_acc"]["mean"] for s in steps_sorted]
    ax.plot(x, y_base, "o-", color=C_BASELINE, lw=2.5, ms=6, label="Baseline", zorder=5)

    # All-steps ablation trajectory
    y_ablated = [all_steps_traj[s]["ablated_acc"]["mean"] for s in steps_sorted]
    ax.plot(x, y_ablated, "s--", color=C_ABLATE, lw=2, ms=5, label="Ablate z$_H$ (all steps)", zorder=4)

    # Single-step ablation trajectories
    single_colors = [PALETTE[1], PALETTE[2], PALETTE[4], PALETTE[5]]
    single_steps = ["4", "6", "8", "10"]
    for i, ss in enumerate(single_steps):
        if ss in agg["ablation_single_step"]:
            traj = agg["ablation_single_step"][ss]["per_step_trajectory"]
            s_sorted = sorted(traj.keys(), key=int)
            y_s = [traj[s]["ablated_acc"]["mean"] for s in s_sorted]
            ax.plot([int(s) for s in s_sorted], y_s, "^--", color=single_colors[i],
                    lw=1.5, ms=4, alpha=0.8, label=f"Ablate step {ss} only")

    ax.set_xlabel("ACT Reasoning Step")
    ax.set_ylabel("Mean Cell Accuracy")
    n = agg["num_puzzles"]
    ax.set_title(f"z$_H$ Ablation: Accuracy Across Reasoning Steps\n(n = {n:,} puzzles)")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0.55, 0.90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    save(fig, "01_ablation_step_trajectories")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 2: Ablation — Single-step impact bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_impact_bars():
    """Bar chart: accuracy drop from ablating z_H at each single step."""
    with open(ABLATION_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    ranking = agg["single_step_impact_ranking"]
    # Sort by step ascending  
    ranking = sorted(ranking, key=lambda r: r["step"])

    steps = [r["step"] for r in ranking]
    deltas = [abs(r["mean_accuracy_delta"]) * 100 for r in ranking]
    broken = [r["mean_cells_broken"] for r in ranking]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    colors = [PALETTE[1], PALETTE[2], PALETTE[4], PALETTE[5]][:len(steps)]

    ax1.bar(range(len(steps)), deltas, color=colors, edgecolor="white", lw=1.5)
    ax1.set_xticks(range(len(steps)))
    ax1.set_xticklabels([f"Step {s}" for s in steps])
    ax1.set_ylabel("Accuracy Drop (%)")
    ax1.set_title("Damage by Ablated Step")
    for i, (d, s) in enumerate(zip(deltas, steps)):
        ax1.text(i, d + 0.2, f"{d:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.bar(range(len(steps)), broken, color=colors, edgecolor="white", lw=1.5)
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels([f"Step {s}" for s in steps])
    ax2.set_ylabel("Mean Cells Broken")
    ax2.set_title("Cells Broken by Ablated Step")
    for i, (b, s) in enumerate(zip(broken, steps)):
        ax2.text(i, b + 0.2, f"{b:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    n = agg["num_puzzles"]
    fig.suptitle(f"Single-Step z$_H$ Ablation: Later Steps Carry More Information\n(n = {n:,})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "02_ablation_single_step_impact")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 3: Freeze — Accuracy decay curve (THE key freeze plot)
# ═══════════════════════════════════════════════════════════════════════════

def plot_freeze_decay():
    """Bar chart showing accuracy loss vs freeze-at step."""
    with open(FREEZE_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    baseline_acc = agg["baseline"]["mean_accuracy"]

    freeze_steps = []
    deltas = []
    stds = []
    for key, val in sorted(agg["freeze_conditions"].items(),
                           key=lambda x: int(x[0].split("_")[-1])):
        s = int(key.split("_")[-1])
        freeze_steps.append(s)
        deltas.append(val["mean_delta_accuracy"])
        stds.append(val["std_delta_accuracy"])

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.arange(len(freeze_steps))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(freeze_steps)))

    bars = ax.bar(x, [-d * 100 for d in deltas], color=colors, edgecolor="white", lw=1.5,
                  yerr=[s * 100 for s in stds], capsize=4, error_kw={"lw": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in freeze_steps])
    ax.set_xlabel("Freeze z$_H$ After Step k")
    ax.set_ylabel("Accuracy Loss (%)")
    ax.set_title(
        "E2-FREEZE: How Much Does Freezing z$_H$ Hurt?\n"
        f"(Baseline = {baseline_acc:.1%}, n = {agg['n_puzzles']} puzzles)",
        fontsize=13
    )

    # Annotate bars
    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{-d:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    save(fig, "03_freeze_accuracy_decay")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 4: Freeze — Step-accuracy curves under each freeze condition
# ═══════════════════════════════════════════════════════════════════════════

def plot_freeze_trajectories():
    """Line plot: accuracy at each reasoning step for baseline vs each freeze condition."""
    with open(FREEZE_DIR / "freeze_accuracy_matrix.json") as f:
        mat = json.load(f)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Baseline
    steps = sorted(mat["baseline"].keys(), key=int)
    x = [int(s) for s in steps]
    y_base = [mat["baseline"][s] for s in steps]
    ax.plot(x, y_base, "o-", color=C_BASELINE, lw=3, ms=7, label="Baseline (no freeze)", zorder=10)

    # Freeze conditions — use a gradient from red to green
    freeze_keys = [k for k in mat.keys() if k.startswith("freeze_at_")]
    freeze_keys.sort(key=lambda k: int(k.split("_")[-1]))

    cmap = plt.cm.coolwarm_r
    n_conditions = len(freeze_keys)

    for i, fk in enumerate(freeze_keys):
        fs = int(fk.split("_")[-1])
        y = [mat[fk][s] for s in steps]
        color = cmap(i / max(n_conditions - 1, 1))

        ax.plot(x, y, "s--", color=color, lw=1.8, ms=4, alpha=0.85,
                label=f"Freeze after step {fs}")

        # Draw vertical line at freeze point
        ax.axvline(fs, color=color, ls=":", alpha=0.3, lw=1)

    ax.set_xlabel("ACT Reasoning Step")
    ax.set_ylabel("Mean Cell Accuracy")
    ax.set_title("Freeze z$_H$: Step-Level Accuracy Trajectories\n"
                 "(curves plateau immediately after freeze point)")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_ylim(0.64, 0.84)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    save(fig, "04_freeze_step_trajectories")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 5: Freeze — Phase portrait (% of total refinement by step)
# ═══════════════════════════════════════════════════════════════════════════

def plot_freeze_phase_portrait():
    """Shows what fraction of z_H refinement is complete by each step."""
    with open(FREEZE_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    # freeze@0 damage = total z_H contribution = "100% of refinement still needed"
    # freeze@k damage = refinement still needed after step k
    # So: % refinement done by step k = 1 - (damage_at_k / damage_at_0)
    damage_at_0 = abs(agg["freeze_conditions"]["freeze_at_0"]["mean_delta_accuracy"])

    freeze_steps = []
    pct_complete = []
    for key, val in sorted(agg["freeze_conditions"].items(),
                           key=lambda x: int(x[0].split("_")[-1])):
        s = int(key.split("_")[-1])
        damage = abs(val["mean_delta_accuracy"])
        pct = 1.0 - (damage / damage_at_0) if damage_at_0 > 0 else 1.0
        freeze_steps.append(s)
        pct_complete.append(pct * 100)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.fill_between(freeze_steps, pct_complete, alpha=0.3, color=C_FREEZE)
    ax.plot(freeze_steps, pct_complete, "o-", color=C_FREEZE, lw=3, ms=8, zorder=5)

    # Annotate key points
    for s, p in zip(freeze_steps, pct_complete):
        ax.annotate(f"{p:.0f}%", (s, p), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11, fontweight="bold")

    # Phase regions
    ax.axvspan(-0.5, 4, alpha=0.08, color="red", label="Rapid planning (0–4)")
    ax.axvspan(4, 8, alpha=0.08, color="orange", label="Fine-tuning (4–8)")
    ax.axvspan(8, 13, alpha=0.08, color="green", label="Converged (8+)")

    ax.set_xlabel("ACT Reasoning Step")
    ax.set_ylabel("% of z$_H$ Refinement Complete")
    ax.set_title("Phase Portrait: When Does z$_H$ Planning Crystallize?", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-0.5, max(freeze_steps) + 0.5)
    ax.set_ylim(0, 110)
    save(fig, "05_freeze_phase_portrait")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 6: Time-Shift — Asymmetry bar chart (THE key time-shift plot)
# ═══════════════════════════════════════════════════════════════════════════

def plot_timeshift_asymmetry():
    """Bar chart comparing forward vs backward transfer effects."""
    with open(TIMESHIFT_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    pairs = agg["transfer_pairs"]
    labels = []
    recip_deltas = []
    final_deltas = []
    directions = []

    for pair_key in pairs:
        p = pairs[pair_key]
        labels.append(pair_key)
        recip_deltas.append(p["mean_recipient_delta"] * 100)
        final_deltas.append(p["mean_delta_accuracy"] * 100)
        directions.append("forward" if "forward" in p["direction"] else "backward")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=False)

    # --- Panel A: Delta at recipient step ---
    x = np.arange(len(labels))
    colors = [C_FORWARD if d == "forward" else C_BACKWARD for d in directions]

    bars1 = ax1.bar(x, recip_deltas, color=colors, edgecolor="white", lw=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Accuracy Change at Recipient Step (%)")
    ax1.set_title("Immediate Effect\n(at the patched step)")
    ax1.axhline(0, color="gray", ls="--", lw=1)

    for bar, val in zip(bars1, recip_deltas):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.15 if val >= 0 else -0.5),
                 f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=10, fontweight="bold")

    # --- Panel B: Delta at final step ---
    bars2 = ax2.bar(x, final_deltas, color=colors, edgecolor="white", lw=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Accuracy Change at Final Step (%)")
    ax2.set_title("Downstream Effect\n(at step 15)")
    ax2.axhline(0, color="gray", ls="--", lw=1)

    for bar, val in zip(bars2, final_deltas):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.1 if val >= 0 else -0.3),
                 f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=10, fontweight="bold")

    # Legend
    fwd_patch = mpatches.Patch(color=C_FORWARD, label="Forward (future → past)")
    bwd_patch = mpatches.Patch(color=C_BACKWARD, label="Backward (past → future)")
    fig.legend(handles=[fwd_patch, bwd_patch], loc="upper center", ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("E5: Time-Shift Patching — Later z$_H$ Encodes More\n"
                 f"(n = {agg['n_puzzles']} puzzles, transfer level = {agg['transfer_level']})",
                 fontsize=14, fontweight="bold", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "06_timeshift_asymmetry")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 7: Time-Shift — Boost vs Hurt puzzle counts
# ═══════════════════════════════════════════════════════════════════════════

def plot_timeshift_puzzle_counts():
    """Stacked bar showing how many puzzles are boosted, hurt, or unchanged per pair."""
    with open(TIMESHIFT_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    pairs = agg["transfer_pairs"]
    labels = list(pairs.keys())
    boosted = [pairs[k]["n_puzzles_boosted"] for k in labels]
    hurt = [pairs[k]["n_puzzles_hurt"] for k in labels]
    unchanged = [pairs[k]["n_puzzles_unchanged"] for k in labels]
    directions = ["forward" if "forward" in pairs[k]["direction"] else "backward" for k in labels]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(labels))
    w = 0.6

    ax.bar(x, boosted, w, label="Boosted (Δ > 0)", color=PALETTE[2], edgecolor="white")
    ax.bar(x, unchanged, w, bottom=boosted, label="Unchanged", color=PALETTE[7], edgecolor="white")
    ax.bar(x, hurt, w, bottom=[b + u for b, u in zip(boosted, unchanged)],
           label="Hurt (Δ < 0)", color=PALETTE[3], edgecolor="white")

    ax.set_xticks(x)
    xlabels = []
    for l, d in zip(labels, directions):
        arrow = "→" if d == "forward" else "←"
        xlabels.append(f"{l}\n({arrow})")
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Number of Puzzles")
    ax.set_title("Time-Shift: Per-Puzzle Outcome Distribution", fontsize=13)
    ax.legend(fontsize=10)
    save(fig, "07_timeshift_puzzle_outcomes")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 8: Combined summary — The "money plot" for the presentation
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined_summary():
    """Three-panel figure summarizing all three experiments."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: Ablation — single-step damage grows with step ---
    with open(ABLATION_DIR / "aggregate_stats.json") as f:
        abl_agg = json.load(f)

    ranking = sorted(abl_agg["single_step_impact_ranking"], key=lambda r: r["step"])
    abl_steps = [r["step"] for r in ranking]
    abl_deltas = [abs(r["mean_accuracy_delta"]) * 100 for r in ranking]

    bars_a = ax1.bar(range(len(abl_steps)), abl_deltas, color=C_ABLATE, edgecolor="white", lw=1.5)

    # Also add all-steps bar
    all_delta = abs(abl_agg["ablation_all_steps"]["accuracy_delta"]["mean"]) * 100
    ax1.bar(len(abl_steps), all_delta, color=C_ABLATE, edgecolor="black", lw=2, alpha=0.7)
    ax1.set_xticks(list(range(len(abl_steps))) + [len(abl_steps)])
    labels_a = [f"Step {s}" for s in abl_steps] + ["All"]
    ax1.set_xticklabels(labels_a, fontsize=9)

    ax1.set_ylabel("Accuracy Drop (%)")
    ax1.set_title("A. Ablation:\nLater Steps Matter More", fontsize=12, fontweight="bold")

    # --- Panel B: Freeze — exponential decay of damage ---
    with open(FREEZE_DIR / "aggregate_stats.json") as f:
        frz_agg = json.load(f)

    frz_steps = []
    frz_deltas = []
    for key, val in sorted(frz_agg["freeze_conditions"].items(),
                           key=lambda x: int(x[0].split("_")[-1])):
        s = int(key.split("_")[-1])
        frz_steps.append(s)
        frz_deltas.append(abs(val["mean_delta_accuracy"]) * 100)

    colors_frz = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(frz_steps)))
    ax2.bar(range(len(frz_steps)), frz_deltas, color=colors_frz, edgecolor="white", lw=1.5)
    ax2.set_xticks(range(len(frz_steps)))
    ax2.set_xticklabels([f"Step {s}" for s in frz_steps])
    ax2.set_ylabel("Accuracy Loss (%)")
    ax2.set_title("B. Freeze z$_H$:\nPlan Crystallizes by Step 4", fontsize=12, fontweight="bold")

    # --- Panel C: Time-shift — forward helps, backward hurts ---
    with open(TIMESHIFT_DIR / "aggregate_stats.json") as f:
        ts_agg = json.load(f)

    pairs = ts_agg["transfer_pairs"]
    ts_labels = list(pairs.keys())
    ts_recip = [pairs[k]["mean_recipient_delta"] * 100 for k in ts_labels]
    ts_dirs = ["forward" if "forward" in pairs[k]["direction"] else "backward" for k in ts_labels]
    ts_colors = [C_FORWARD if d == "forward" else C_BACKWARD for d in ts_dirs]

    ax3.bar(range(len(ts_labels)), ts_recip, color=ts_colors, edgecolor="white", lw=1.5)
    ax3.set_xticks(range(len(ts_labels)))
    ax3.set_xticklabels(ts_labels, rotation=35, ha="right", fontsize=9)
    ax3.set_ylabel("Δ Accuracy at Recipient Step (%)")
    ax3.axhline(0, color="gray", ls="--", lw=1)
    ax3.set_title("C. Time-Shift:\nFuture z$_H$ > Past z$_H$", fontsize=12, fontweight="bold")

    # Legend for panel C
    fwd_patch = mpatches.Patch(color=C_FORWARD, label="Future→Past")
    bwd_patch = mpatches.Patch(color=C_BACKWARD, label="Past→Future")
    ax3.legend(handles=[fwd_patch, bwd_patch], fontsize=9, loc="lower left")

    fig.suptitle("Three Experiments, One Story: z$_H$ is Progressively Refined",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "08_combined_three_experiments")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 9: Freeze — Puzzles hurt/helped/unchanged stacked
# ═══════════════════════════════════════════════════════════════════════════

def plot_freeze_puzzle_counts():
    """How many puzzles are hurt vs helped at each freeze point."""
    with open(FREEZE_DIR / "aggregate_stats.json") as f:
        agg = json.load(f)

    freeze_steps = []
    hurt = []
    helped = []
    unchanged = []
    for key, val in sorted(agg["freeze_conditions"].items(),
                           key=lambda x: int(x[0].split("_")[-1])):
        s = int(key.split("_")[-1])
        freeze_steps.append(s)
        hurt.append(val["n_puzzles_hurt"])
        helped.append(val["n_puzzles_helped"])
        unchanged.append(val["n_puzzles_unchanged"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(freeze_steps))
    w = 0.6

    ax.bar(x, helped, w, label="Helped", color=PALETTE[2], edgecolor="white")
    ax.bar(x, unchanged, w, bottom=helped, label="Unchanged", color=PALETTE[7], edgecolor="white")
    ax.bar(x, hurt, w, bottom=[h + u for h, u in zip(helped, unchanged)],
           label="Hurt", color=PALETTE[3], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in freeze_steps])
    ax.set_xlabel("Freeze z$_H$ After Step k")
    ax.set_ylabel("Number of Puzzles")
    ax.set_title(f"Freeze z$_H$: Per-Puzzle Outcomes (n = {agg['n_puzzles']})")
    ax.legend(fontsize=10)
    save(fig, "09_freeze_puzzle_outcomes")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Output directory: {OUT_DIR}\n")

    # Check which data is available
    has_ablation = (ABLATION_DIR / "aggregate_stats.json").exists()
    has_freeze = (FREEZE_DIR / "aggregate_stats.json").exists()
    has_timeshift = (TIMESHIFT_DIR / "aggregate_stats.json").exists()

    if has_ablation:
        print("Ablation plots:")
        plot_ablation_trajectories()
        plot_ablation_impact_bars()
    else:
        print("  [SKIP] No ablation data found")

    if has_freeze:
        print("Freeze z_H plots:")
        plot_freeze_decay()
        plot_freeze_trajectories()
        plot_freeze_phase_portrait()
        plot_freeze_puzzle_counts()
    else:
        print("  [SKIP] No freeze data found")

    if has_timeshift:
        print("Time-shift plots:")
        plot_timeshift_asymmetry()
        plot_timeshift_puzzle_counts()
    else:
        print("  [SKIP] No time-shift data found")

    if has_ablation and has_freeze and has_timeshift:
        print("Combined summary:")
        plot_combined_summary()
    else:
        print("  [SKIP] Combined plot requires all three experiments")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
