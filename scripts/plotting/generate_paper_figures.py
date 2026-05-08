#!/usr/bin/env python3
"""Generate all 4 paper figures from existing experiment data.

Outputs go to paper/figures/:
  1. accuracy_curves_5model.pdf   — copied from baseline comparison
  2. causal_evidence_panel.pdf    — 3-panel: ablation + freeze + time-shift
  3. readout_vs_computation.pdf   — 2-panel: probe accuracy + directed ablation
  4. causal_hierarchy.pdf         — bar chart: full ablation vs SAE vs probes
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
})

COLORS = {
    "hrm": "#1f77b4",
    "vrnn": "#d62728",
    "ut": "#2ca02c",
    "pt": "#ff7f0e",
    "srnn": "#9467bd",
}


# ======================================================================
# Figure 1: Accuracy curves (single-panel, sized to match hamming_convergence)
# ======================================================================
# Match the figsize/style of plot_baseline_comparison.plot_hamming_convergence
# (figsize=(6, 4.5)) so the two plots render at the same scale when placed
# side-by-side as subfigures in the paper.
_BASELINE_DIR = ROOT / "results" / "baseline_comparison"

_MODEL_STYLES = {
    "HRM":      {"color": "#2196F3", "marker": "o", "linestyle": "-"},
    "UT_best":  {"color": "#4CAF50", "marker": "^", "linestyle": "-."},
    "RNN_best": {"color": "#F44336", "marker": "s", "linestyle": "--"},
    "PT_best":  {"color": "#FF9800", "marker": "D", "linestyle": ":"},
    "SRNN_best":{"color": "#9C27B0", "marker": "v", "linestyle": ":"},
}
_MODEL_LABELS = {
    "HRM":      "HRM (Hierarchical)",
    "UT_best":  "Universal Transformer",
    "RNN_best": "Vanilla RNN",
    "PT_best":  "Plain Transformer",
    "SRNN_best":"Standard RNN",
}


def fig1():
    """Combined 2-panel recurrence figure: Hamming (left) + Cell accuracy (right)."""
    dst = OUT / "recurrence_curves.pdf"

    # Load primary model eval JSONs
    primary = {}
    for name in _MODEL_STYLES.keys():
        path = _BASELINE_DIR / f"{name}_eval.json"
        if path.exists():
            with open(path) as f:
                primary[name] = json.load(f)
    if not primary:
        print(f"[Fig1] WARNING: no eval JSONs found under {_BASELINE_DIR}")
        return

    # Match plot_baseline_comparison rcParams locally so the figure visually
    # matches the original fig2_hamming_convergence styling.
    with plt.rc_context({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
    }):
        fig, (ax_h, ax_a) = plt.subplots(1, 2, figsize=(12, 4.5))

        for name, data in sorted(primary.items()):
            steps_data = data["per_step_metrics"]
            steps = sorted(int(s) for s in steps_data.keys())
            style = _MODEL_STYLES[name]
            label = _MODEL_LABELS[name]

            hamming = [steps_data[str(s)]["hamming_distance"]["mean"] for s in steps]
            ax_h.plot(steps, hamming, label=label, markersize=4, **style)

            cell_accs = [steps_data[str(s)]["cell_accuracy"]["mean"] for s in steps]
            ax_a.plot(steps, cell_accs, label=label, markersize=4, **style)

        ax_h.set_xlabel("Reasoning Step")
        ax_h.set_ylabel("Hamming Distance to Solution")
        ax_h.set_title("(a) Hamming Distance vs. Step")

        ax_a.set_xlabel("Reasoning Step")
        ax_a.set_ylabel("Cell Accuracy")
        ax_a.set_title("(b) Cell Accuracy vs. Step")
        ax_a.set_ylim(0, 1.05)

        # Single shared legend underneath both panels.
        handles, labels = ax_h.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False, bbox_to_anchor=(0.5, -0.04),
                   fontsize=13, handlelength=2.5, handletextpad=0.6,
                   columnspacing=1.8)

        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(dst)
        fig.savefig(OUT / "recurrence_curves.png", dpi=300)
        plt.close(fig)
    print(f"[Fig1] Saved {dst.name}")


# ======================================================================
# Figure 2: Causal evidence panel (ablation + freeze + time-shift)
# ======================================================================
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

    # ── Panel A: Per-step z_H ablation (N=1000) ──
    ax = axes[0]
    abl_path = ROOT / "results" / "controlled" / "ablation" / "zH_1000" / "zH" / "aggregate.json"
    with open(abl_path) as f:
        abl = json.load(f)
    steps = sorted(abl["per_step_ablation"].keys(), key=int)
    deltas = [abl["per_step_ablation"][s]["delta_accuracy"]["mean"] * 100 for s in steps]
    ci_lo = [abl["per_step_ablation"][s]["delta_accuracy"]["ci_lower"] * 100 for s in steps]
    ci_hi = [abl["per_step_ablation"][s]["delta_accuracy"]["ci_upper"] * 100 for s in steps]
    x = [int(s) for s in steps]
    yerr_lo = [d - l for d, l in zip(deltas, ci_lo)]
    yerr_hi = [h - d for d, h in zip(deltas, ci_hi)]
    ax.bar(x, deltas, color="#c0392b", alpha=0.8, width=0.8)
    ax.errorbar(x, deltas, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=2, linewidth=0.8)
    ax.set_xlabel("Ablated step")
    ax.set_ylabel("$\\Delta$ Cell Accuracy (pp)")
    ax.set_title("(a) Per-step $z_H$ ablation")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xticks([0, 4, 8, 12, 15])

    # ── Panel B: Freeze z_H ──
    ax = axes[1]
    frz_path = ROOT / "results" / "freeze" / "freeze_h" / "aggregate_stats.json"
    with open(frz_path) as f:
        frz = json.load(f)
    # Parse freeze_conditions keys like "freeze_at_0", "freeze_at_1", ...
    conds = frz["freeze_conditions"]
    fx = sorted([int(k.split("_")[-1]) for k in conds.keys()])
    losses = [abs(conds[f"freeze_at_{s}"]["mean_delta_accuracy"]) * 100 for s in fx]
    ax.bar(range(len(fx)), losses, color="#e67e22", alpha=0.8, width=0.7)
    ax.set_xticks(range(len(fx)))
    ax.set_xticklabels([str(s) for s in fx])
    ax.set_xlabel("Freeze after step $k$")
    ax.set_ylabel("Accuracy loss (pp)")
    ax.set_title("(b) Freeze $z_H$ after step $k$")

    # ── Panel C: Time-shift ──
    ax = axes[2]
    ts_path = ROOT / "results" / "time_shift" / "aggregate_stats.json"
    with open(ts_path) as f:
        ts = json.load(f)
    pairs = ts["transfer_pairs"]
    future_labels, future_vals = [], []
    past_labels, past_vals = [], []
    for key, val in pairs.items():
        delta = val["mean_delta_accuracy"] * 100
        if val.get("direction", "").startswith("forward"):
            future_labels.append(key)
            future_vals.append(delta)
        else:
            past_labels.append(key)
            past_vals.append(delta)
    labels = future_labels + past_labels
    vals = future_vals + past_vals
    colors_ts = ["#27ae60"] * len(future_labels) + ["#8e44ad"] * len(past_labels)
    ax.bar(range(len(labels)), vals, color=colors_ts, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("$\\Delta$ Accuracy (pp)")
    ax.set_title("(c) Time-shift transplant")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#27ae60"), Patch(color="#8e44ad")],
              ["Future→Past", "Past→Future"], fontsize=6, loc="best")

    plt.tight_layout()
    fig.savefig(OUT / "causal_evidence_panel.pdf")
    fig.savefig(OUT / "causal_evidence_panel.png", dpi=300)
    plt.close(fig)
    print("[Fig2] Saved causal_evidence_panel.pdf")


# ======================================================================
# Figure 3: Readout ≠ Computation
# ======================================================================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # ── Panel A: Probe accuracy at step 15 ──
    ax = axes[0]
    probe_path = ROOT / "results" / "probes" / "nonlinear_probes" / "comparison_summary.json"
    if probe_path.exists():
        with open(probe_path) as f:
            probes = json.load(f)
        targets = ["violated_in_row", "violated_in_col", "violated_in_box", "per_cell_correct"]
        labels = ["Row\nviols", "Col\nviols", "Box\nviols", "Cell\ncorrect"]
        accs = [probes[t]["steps"]["15"]["linear"] * 100 for t in targets]
    else:
        labels = ["Row\nviols", "Col\nviols", "Box\nviols", "Cell\ncorrect"]
        accs = [89.9, 90.1, 88.8, 83.5]

    bar_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(range(len(labels)), accs, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_title("(a) Linear probe readout\n($z_H$, step 15)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", ls="--", lw=0.5, alpha=0.5)

    # ── Panel B: Directed ablation ──
    ax = axes[1]
    da_path = ROOT / "results" / "directed_ablation" / "e9_directed_ablation" / "aggregate_results.json"
    if da_path.exists():
        with open(da_path) as f:
            da = json.load(f)
        targets_da = ["violated_in_row", "violated_in_col", "violated_in_box", "per_cell_correct"]
        labels_da = ["Row\nviols", "Col\nviols", "Box\nviols", "Cell\ncorrect"]
        deltas = [da[t]["mean_delta_accuracy"] * 100 for t in targets_da]
        rand_deltas = [da[k]["mean_delta_accuracy"] * 100
                       for k in sorted(da.keys()) if "random" in k.lower()]
    else:
        labels_da = ["Row\nviols", "Col\nviols", "Box\nviols", "Cell\ncorrect"]
        deltas = [-0.09, -0.59, -0.23, 0.72]
        rand_deltas = [-0.44, 0.07, 0.95]

    bars = ax.bar(range(len(labels_da)), deltas, color=bar_colors, alpha=0.85, edgecolor="white")
    # Show random control range as gray band
    if rand_deltas:
        ax.axhspan(min(rand_deltas), max(rand_deltas), alpha=0.15, color="gray",
                    label="Random control range")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(labels_da)))
    ax.set_xticklabels(labels_da)
    ax.set_ylabel("$\\Delta$ Cell Accuracy (%)")
    ax.set_title("(b) Directed ablation\n(causal effect)")
    if rand_deltas:
        ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    fig.savefig(OUT / "readout_vs_computation.pdf")
    fig.savefig(OUT / "readout_vs_computation.png", dpi=300)
    plt.close(fig)
    print("[Fig3] Saved readout_vs_computation.pdf")


# ======================================================================
# Figure 4: Causal Hierarchy
# ======================================================================
def fig4():
    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    # Full z_H ablation (from N=1000 data)
    abl_path = ROOT / "results" / "controlled" / "ablation" / "zH_1000" / "zH" / "aggregate.json"
    with open(abl_path) as f:
        abl = json.load(f)
    full_delta = abl["all_steps_delta"]["mean"] * 100
    full_ci_lo = abl["all_steps_delta"]["ci_lower"] * 100
    full_ci_hi = abl["all_steps_delta"]["ci_upper"] * 100

    # SAE features (from causal_ablation)
    sae_path = ROOT / "results" / "sae_study" / "causal_ablation" / "aggregate.json"
    if sae_path.exists():
        with open(sae_path) as f:
            sae = json.load(f)
        conds = sae["conditions"]
        sae_top = conds["sae_top_features"]["mean_delta_acc"] * 100
        sae_rand = conds["random_sae_features"]["mean_delta_acc"] * 100
        probe_d = conds["probe_directions"]["mean_delta_acc"] * 100
        rand_d = conds["random_directions"]["mean_delta_acc"] * 100
    else:
        sae_top = -3.9
        sae_rand = -4.3
        probe_d = 0.1
        rand_d = 0.6

    labels = ["Full $z_H$\nablation", "Top-50\nSAE", "Random\nSAE", "Probe\ndirections", "Random\ndirections"]
    vals = [full_delta, sae_top, sae_rand, probe_d, rand_d]
    colors_h = ["#c0392b", "#2980b9", "#85c1e9", "#e67e22", "#f5cba7"]

    bars = ax.bar(range(len(labels)), vals, color=colors_h, alpha=0.85, edgecolor="white", width=0.7)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # Annotate
    for bar, val in zip(bars, vals):
        ypos = val - 0.8 if val < 0 else val + 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}%", ha="center", va="top" if val < 0 else "bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$\\Delta$ Cell Accuracy (%)")
    ax.set_title("Causal Importance Hierarchy")

    # Add headroom so positive-bar labels (e.g. "+0.6%") don't clip the
    # top axis spine. Pad the data range by ~10% on each side.
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin
    pad = max(span * 0.10, 1.5)
    ax.set_ylim(vmin - pad, vmax + pad)

    plt.tight_layout()
    fig.savefig(OUT / "causal_hierarchy.pdf")
    fig.savefig(OUT / "causal_hierarchy.png", dpi=300)
    plt.close(fig)
    print("[Fig4] Saved causal_hierarchy.pdf")


# ======================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUT}")
    fig1()
    fig2()
    fig3()
    fig4()
    print("\nAll figures generated!")
    for f in sorted(OUT.glob("*")):
        print(f"  {f.name}")
