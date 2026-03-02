#!/usr/bin/env python3
"""Generate publication-quality plots for E8 (Constraint Probes) and E9 (Directed Ablation).

Output: results/e8_e9_plots/  (PNG + PDF for each figure)

Figures:
  1. Probe accuracy vs ACT step (line plot, all targets)
  2. Cosine similarity heatmap (constraint directions, step 15)
  3. Cosine evolution across steps (line plot)
  4. PCA explained variance (stacked bar per step)
  5. E9 specificity matrix (grouped bar: Δrow/Δcol/Δbox per direction)
  6. E9 accuracy impact (bar chart with error bars)
  7. E9 cells broken vs fixed (paired bar)
  8. Readout vs Computation comparison (combined E8+E9 panel)
"""

import os
import sys
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "results", "e8_e9_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Pretty labels
PRETTY = {
    "violated_in_row": "Row violations",
    "violated_in_col": "Col violations",
    "violated_in_box": "Box violations",
    "per_cell_correct": "Cell correct",
    "is_given": "Is given (clue)",
    "cell_digit": "Cell digit (10-class)",
    "random_control_0": "Random #1",
    "random_control_1": "Random #2",
    "random_control_2": "Random #3",
}

COLORS = {
    "violated_in_row": "#e63946",
    "violated_in_col": "#457b9d",
    "violated_in_box": "#2a9d8f",
    "per_cell_correct": "#f4a261",
    "is_given": "#9b9b9b",
    "cell_digit": "#6a4c93",
    "random_control_0": "#bbb",
    "random_control_1": "#ccc",
    "random_control_2": "#ddd",
}

MARKERS = {
    "violated_in_row": "o",
    "violated_in_col": "s",
    "violated_in_box": "D",
    "per_cell_correct": "^",
    "is_given": "x",
    "cell_digit": "P",
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}")


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════

# E8 sweep results
e8_csv = os.path.join(REPO, "results", "e8_constraint_probes", "sweep_results.csv")
e8_rows = []
with open(e8_csv) as f:
    for row in csv.DictReader(f):
        row["step"] = int(row["step"])
        row["val_score"] = float(row["val_score"])
        row["train_score"] = float(row["train_score"])
        e8_rows.append(row)

# E8 geometry
e8_geo = os.path.join(REPO, "results", "e8_constraint_probes", "geometric_analysis.json")
with open(e8_geo) as f:
    geo = json.load(f)

# E9 aggregate
e9_agg_path = os.path.join(REPO, "results", "e9_directed_ablation", "aggregate_results.json")
with open(e9_agg_path) as f:
    e9_agg = json.load(f)

# E9 specificity matrix
e9_spec_path = os.path.join(REPO, "results", "e9_directed_ablation", "specificity_matrix.json")
with open(e9_spec_path) as f:
    e9_spec = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Probe Accuracy vs ACT Step
# ═══════════════════════════════════════════════════════════════════════════

def plot_probe_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))

    targets = ["violated_in_row", "violated_in_col", "violated_in_box",
               "per_cell_correct", "is_given", "cell_digit"]

    for tgt in targets:
        rows = sorted([r for r in e8_rows if r["target"] == tgt], key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        accs = [r["val_score"] * 100 for r in rows]
        ax.plot(steps, accs, marker=MARKERS.get(tgt, "o"), color=COLORS[tgt],
                label=PRETTY[tgt], linewidth=2, markersize=7)

    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("E8: Linear Probe Accuracy by ACT Step\n(500 puzzles, z_H representations)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xticks([0, 4, 8, 12, 15])
    ax.set_ylim(65, 101)
    ax.axhline(50, color="#ccc", linestyle="--", linewidth=0.8, label=None)
    ax.text(0.2, 51, "chance (binary)", fontsize=8, color="#aaa")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig1_probe_accuracy_vs_step")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Cosine Similarity Heatmap (Step 15)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cosine_heatmap():
    labels = ["Row viols", "Col viols", "Box viols", "Cell correct"]
    keys = ["violated_in_row", "violated_in_col", "violated_in_box", "per_cell_correct"]
    n = len(keys)
    mat = np.eye(n)

    step15_cosines = [c for c in geo["constraint_cosines"] if c["group"] == "z_H_step15"]
    for entry in step15_cosines:
        a_idx = keys.index(entry["probe_a"]) if entry["probe_a"] in keys else None
        b_idx = keys.index(entry["probe_b"]) if entry["probe_b"] in keys else None
        if a_idx is not None and b_idx is not None:
            mat[a_idx, b_idx] = entry["cosine"]
            mat[b_idx, a_idx] = entry["cosine"]

    fig, ax = plt.subplots(figsize=(6, 5))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    ax.set_title("E8: Probe Direction Cosine Similarity\n(z_H, ACT step 15)")
    save(fig, "fig2_cosine_heatmap_step15")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Cosine Similarity Evolution Across Steps
# ═══════════════════════════════════════════════════════════════════════════

def plot_cosine_evolution():
    pairs = [
        ("violated_in_row", "violated_in_col", "Row ↔ Col"),
        ("violated_in_row", "violated_in_box", "Row ↔ Box"),
        ("violated_in_col", "violated_in_box", "Col ↔ Box"),
        ("per_cell_correct", "violated_in_row", "Correct ↔ Row"),
        ("per_cell_correct", "violated_in_col", "Correct ↔ Col"),
        ("per_cell_correct", "violated_in_box", "Correct ↔ Box"),
    ]
    pair_colors = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261", "#e9c46a", "#264653"]

    steps_order = [0, 4, 8, 12, 15]
    step_groups = {s: f"z_H_step{s}" for s in steps_order}

    fig, ax = plt.subplots(figsize=(8, 5))

    for (pa, pb, label), color in zip(pairs, pair_colors):
        vals = []
        for s in steps_order:
            group = step_groups[s]
            for c in geo["constraint_cosines"]:
                if c["group"] == group:
                    if (c["probe_a"] == pa and c["probe_b"] == pb) or \
                       (c["probe_a"] == pb and c["probe_b"] == pa):
                        vals.append(c["cosine"])
                        break

        ls = "--" if "Correct" in label else "-"
        ax.plot(steps_order, vals, marker="o", label=label, color=color,
                linewidth=2, linestyle=ls, markersize=6)

    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("E8: How Probe Directions Relate Across ACT Steps")
    ax.legend(loc="center right", framealpha=0.9, fontsize=9)
    ax.set_xticks(steps_order)
    ax.axhline(0, color="#ccc", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.85, 0.85)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig3_cosine_evolution")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: PCA Explained Variance
# ═══════════════════════════════════════════════════════════════════════════

def plot_pca_variance():
    steps_order = [0, 4, 8, 12, 15]
    step_groups = {s: f"z_H_step{s}" for s in steps_order}

    pc1, pc2, pc3 = [], [], []
    for s in steps_order:
        for p in geo["pca_of_weight_vectors"]:
            if p["group"] == step_groups[s]:
                evr = p["explained_variance_ratio"]
                pc1.append(evr[0] * 100)
                pc2.append(evr[1] * 100)
                pc3.append(evr[2] * 100)
                break

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(steps_order))
    w = 0.55

    ax.bar(x, pc1, w, label="PC1 (shared constraint axis)", color="#264653")
    ax.bar(x, pc2, w, bottom=pc1, label="PC2", color="#2a9d8f")
    ax.bar(x, pc3, w, bottom=[a + b for a, b in zip(pc1, pc2)],
           label="PC3", color="#e9c46a")

    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("E8: PCA of Constraint Probe Weight Vectors\n(4 probes → 3 effective components)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps_order])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)

    for i in range(len(steps_order)):
        ax.text(i, pc1[i] / 2, f"{pc1[i]:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig4_pca_explained_variance")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: E9 Specificity Matrix (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_specificity_matrix():
    directions = ["violated_in_row", "violated_in_col", "violated_in_box",
                  "per_cell_correct", "is_given",
                  "random_control_0", "random_control_1", "random_control_2"]
    display_names = [PRETTY.get(d, d) for d in directions]

    row_deltas = [e9_spec[d]["delta_row_viols"] for d in directions]
    col_deltas = [e9_spec[d]["delta_col_viols"] for d in directions]
    box_deltas = [e9_spec[d]["delta_box_viols"] for d in directions]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(directions))
    w = 0.25

    bars_row = ax.bar(x - w, row_deltas, w, label="Δ Row violations", color="#e63946", alpha=0.85)
    bars_col = ax.bar(x, col_deltas, w, label="Δ Col violations", color="#457b9d", alpha=0.85)
    bars_box = ax.bar(x + w, box_deltas, w, label="Δ Box violations", color="#2a9d8f", alpha=0.85)

    # Mark "expected" bars with stars
    expected = {
        "violated_in_row": 0,   # row bar index
        "violated_in_col": 1,   # col bar index
        "violated_in_box": 2,   # box bar index
    }
    for d_idx, d in enumerate(directions):
        if d in expected:
            bar_group = [bars_row, bars_col, bars_box][expected[d]]
            bar = bar_group[d_idx]
            ax.annotate("★", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha="center", fontsize=14, color="gold",
                       xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("Direction Ablated")
    ax.set_ylabel("Δ Violations (ablated − baseline)")
    ax.set_title("E9: Violation Changes After Directional Ablation\n(200 puzzles, ★ = expected to be most affected)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Add separator between constraint and control directions
    ax.axvline(4.5, color="#aaa", linewidth=1, linestyle="--")
    ax.text(2.0, ax.get_ylim()[1] * 0.9, "Probe directions", ha="center",
            fontsize=9, color="#666", style="italic")
    ax.text(6.0, ax.get_ylim()[1] * 0.9, "Controls", ha="center",
            fontsize=9, color="#666", style="italic")

    save(fig, "fig5_e9_specificity_matrix")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: E9 Accuracy Impact
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_impact():
    directions = ["violated_in_row", "violated_in_col", "violated_in_box",
                  "per_cell_correct", "is_given",
                  "random_control_0", "random_control_1", "random_control_2"]
    display_names = [PRETTY.get(d, d) for d in directions]

    delta_accs = [e9_agg[d]["mean_delta_accuracy"] * 100 for d in directions]
    std_accs = [e9_agg[d]["std_delta_accuracy"] * 100 for d in directions]
    # SE = std / sqrt(n)
    n = 200
    se_accs = [s / np.sqrt(n) for s in std_accs]

    bar_colors = [COLORS.get(d, "#aaa") for d in directions]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(directions))
    bars = ax.bar(x, delta_accs, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.errorbar(x, delta_accs, yerr=se_accs, fmt="none", ecolor="black",
                capsize=4, capthick=1.5, linewidth=1.5)

    ax.set_xlabel("Direction Ablated")
    ax.set_ylabel("Δ Cell Accuracy (%)")
    ax.set_title("E9: Accuracy Change from Single-Direction Ablation\n(200 puzzles, error bars = ±1 SE)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Shade the "noise floor" region
    max_random_se = max(abs(delta_accs[5]), abs(delta_accs[6]), abs(delta_accs[7]))
    noise_bound = max(max_random_se, 1.0)
    ax.axhspan(-noise_bound, noise_bound, alpha=0.08, color="gray")
    ax.text(7.5, noise_bound * 0.6, "noise floor\n(random controls)",
            ha="right", fontsize=8, color="#888", style="italic")

    # Separator
    ax.axvline(4.5, color="#aaa", linewidth=1, linestyle="--")

    save(fig, "fig6_e9_accuracy_impact")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Cells Broken vs Fixed
# ═══════════════════════════════════════════════════════════════════════════

def plot_broken_vs_fixed():
    directions = ["violated_in_row", "violated_in_col", "violated_in_box",
                  "per_cell_correct", "is_given",
                  "random_control_0", "random_control_1", "random_control_2"]
    display_names = [PRETTY.get(d, d) for d in directions]

    broken = [e9_agg[d]["mean_cells_broken"] for d in directions]
    fixed = [e9_agg[d]["mean_cells_fixed"] for d in directions]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(directions))
    w = 0.35

    ax.bar(x - w/2, broken, w, label="Cells broken", color="#e63946", alpha=0.8)
    ax.bar(x + w/2, fixed, w, label="Cells fixed", color="#2a9d8f", alpha=0.8)

    ax.set_xlabel("Direction Ablated")
    ax.set_ylabel("Mean cells per puzzle")
    ax.set_title("E9: Cells Broken vs Fixed by Directional Ablation\n(near-symmetry indicates non-causal perturbation)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Separator
    ax.axvline(4.5, color="#aaa", linewidth=1, linestyle="--")

    save(fig, "fig7_e9_broken_vs_fixed")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Readout vs Computation Panel (combined E8 + E9)
# ═══════════════════════════════════════════════════════════════════════════

def plot_readout_vs_computation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.35})

    # ── Left: E8 probe accuracies at step 15 ──
    targets = ["violated_in_row", "violated_in_col", "violated_in_box", "per_cell_correct"]
    vals_e8 = []
    for tgt in targets:
        for r in e8_rows:
            if r["target"] == tgt and r["step"] == 15:
                vals_e8.append(r["val_score"] * 100)
                break

    bar_colors = [COLORS[t] for t in targets]
    display = [PRETTY[t].replace(" violations", "\nviols") for t in targets]

    ax1.bar(range(len(targets)), vals_e8, color=bar_colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(len(targets)))
    ax1.set_xticklabels(display, fontsize=9)
    ax1.set_ylabel("Probe Accuracy (%)")
    ax1.set_title("E8: Readout Accuracy\n(linear probe on z_H, step 15)", fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.axhline(50, color="#ccc", linestyle="--", linewidth=0.8)
    ax1.text(0, 52, "chance", fontsize=8, color="#aaa")

    for i, v in enumerate(vals_e8):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax1.grid(axis="y", alpha=0.3)

    # ── Right: E9 Δaccuracy for same directions ──
    e9_deltas = [e9_agg[t]["mean_delta_accuracy"] * 100 for t in targets]
    e9_stds = [e9_agg[t]["std_delta_accuracy"] * 100 for t in targets]
    e9_ses = [s / np.sqrt(200) for s in e9_stds]

    # Also get random control stats
    rand_deltas = [e9_agg[f"random_control_{i}"]["mean_delta_accuracy"] * 100 for i in range(3)]
    rand_mean = np.mean(rand_deltas)
    rand_max = max(abs(d) for d in rand_deltas)

    ax2.bar(range(len(targets)), e9_deltas, color=bar_colors, alpha=0.85, edgecolor="white")
    ax2.errorbar(range(len(targets)), e9_deltas, yerr=e9_ses, fmt="none",
                 ecolor="black", capsize=5, capthick=1.5, linewidth=1.5)
    ax2.set_xticks(range(len(targets)))
    ax2.set_xticklabels(display, fontsize=9)
    ax2.set_ylabel("Δ Cell Accuracy (%)")
    ax2.set_title("E9: Causal Impact\n(ablate direction from z_H, 200 puzzles)", fontsize=12)

    # Noise floor band
    ax2.axhspan(-rand_max, rand_max, alpha=0.12, color="gray")
    ax2.text(3.4, rand_max * 0.55, "random\nnoise floor", ha="center",
             fontsize=8, color="#888", style="italic")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)

    # Big arrows annotation
    fig.text(0.26, 0.02,
             "← High readout accuracy (89–90%)", fontsize=10, color="#264653",
             ha="center", fontweight="bold")
    fig.text(0.72, 0.02,
             "Near-zero causal effect (<1%) →", fontsize=10, color="#c1121f",
             ha="center", fontweight="bold")

    fig.suptitle("The Readout ≠ Computation Dissociation", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig8_readout_vs_computation")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Weight norm comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_weight_norms():
    fig, ax = plt.subplots(figsize=(8, 5))

    targets = ["violated_in_row", "violated_in_col", "violated_in_box",
               "per_cell_correct", "is_given", "cell_digit"]
    steps = [0, 4, 8, 12, 15]

    for tgt in targets:
        norms = []
        for s in steps:
            for w in geo["weight_norms"]:
                if w["target"] == tgt and w["step"] == s:
                    norms.append(w["W_norm"])
                    break
        ax.plot(steps, norms, marker=MARKERS.get(tgt, "o"), color=COLORS[tgt],
                label=PRETTY[tgt], linewidth=2, markersize=7)

    ax.set_xlabel("ACT Step")
    ax.set_ylabel("‖W‖ (probe weight norm)")
    ax.set_title("E8: Probe Weight Norms Across ACT Steps\n(larger norm = stronger signal in z_H)")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xticks(steps)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig9_weight_norms")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating E8/E9 plots...")
    plot_probe_accuracy()
    plot_cosine_heatmap()
    plot_cosine_evolution()
    plot_pca_variance()
    plot_specificity_matrix()
    plot_accuracy_impact()
    plot_broken_vs_fixed()
    plot_readout_vs_computation()
    plot_weight_norms()
    print(f"\nAll 9 figures saved to {OUT_DIR}")
    print(f"  {len(os.listdir(OUT_DIR))} files total (PNG + PDF)")
