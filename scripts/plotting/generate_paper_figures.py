#!/usr/bin/env python3
"""Generate the four core paper figures from existing experiment data.

Outputs go to results/Final/figures/:
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

from paper_style import apply_paper_style, scaled_rc, COL_W, TEXT_W

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "Final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style (single source of truth for all paper figures) ──────
apply_paper_style()

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
_BASELINE_DIR = ROOT / "results" / "Sudoku" / "baseline_comparison"

_MODEL_STYLES = {
    "HRM":      {"color": "#2196F3", "marker": "o", "linestyle": "-"},
    "UT_best":  {"color": "#4CAF50", "marker": "^", "linestyle": "-."},
    "RNN_best": {"color": "#F44336", "marker": "s", "linestyle": "--"},
    "PT_best":  {"color": "#FF9800", "marker": "D", "linestyle": ":"},
    "SRNN_best":{"color": "#9C27B0", "marker": "v", "linestyle": ":"},
}
# Naming scheme carried over from the earlier version of this work. Note these
# are not just shorter: `RNN_best` is VanillaRNNModel, which applies a
# Transformer block recurrently (hence "Recurrent Trans", not "Vanilla RNN"),
# and `SRNN_best` is StandardRNNModel, which is an nn.GRU (hence "GRU").
_MODEL_LABELS = {
    "HRM":      "HRM",
    "UT_best":  "Univ. Trans",
    "RNN_best": "Recurrent Trans",
    "PT_best":  "Plain Trans",
    "SRNN_best":"GRU",
}

# Legend/plot order. Matplotlib fills a multi-column legend COLUMN-major, so with
# ncol=3 this order lays the five models out as
#     row 1:  HRM             GRU            Plain Trans
#     row 2:  Recurrent Trans Univ. Trans
# which is the arrangement used across all three baseline figures.
_MODEL_ORDER = ["HRM", "RNN_best", "SRNN_best", "UT_best", "PT_best"]
_LEGEND_NCOL = 3


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

    # Included at .8\columnwidth in 04finding1.tex, so LaTeX shrinks this by
    # 0.8x; scaled_rc compensates so the on-page text is PAPER_RC-sized.
    with plt.rc_context(scaled_rc(COL_W, 0.8 * COL_W,
                                  **{"axes.grid": True, "grid.alpha": 0.3})):
        fig, (ax_h, ax_a) = plt.subplots(1, 2, figsize=(COL_W, 2.15))

        for name in [m for m in _MODEL_ORDER if m in primary]:
            data = primary[name]
            steps_data = data["per_step_metrics"]
            steps = sorted(int(s) for s in steps_data.keys())
            style = _MODEL_STYLES[name]
            label = _MODEL_LABELS[name]

            hamming = [steps_data[str(s)]["hamming_distance"]["mean"] for s in steps]
            ax_h.plot(steps, hamming, label=label, markersize=3, markevery=3, **style)

            cell_accs = [steps_data[str(s)]["cell_accuracy"]["mean"] for s in steps]
            ax_a.plot(steps, cell_accs, label=label, markersize=3, markevery=3, **style)

        ax_h.set_xlabel("Reasoning step")
        ax_h.set_ylabel("Hamming distance")
        ax_h.set_title("(a) Hamming vs. step")

        ax_a.set_xlabel("Reasoning step")
        ax_a.set_ylabel("Cell accuracy")
        ax_a.set_title("(b) Cell acc. vs. step")
        ax_a.set_ylim(0, 1.05)

        # Single shared legend underneath both panels; see _MODEL_ORDER for the
        # row layout ncol=3 produces. Handles/spacing are kept tight because the
        # widest row (Recurrent Trans + Univ. Trans) has to fit in one column.
        handles, labels = ax_h.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=_LEGEND_NCOL,
                   frameon=False, bbox_to_anchor=(0.5, -0.02),
                   handlelength=1.2, handletextpad=0.3, columnspacing=0.7)

        fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.4)
        fig.savefig(dst)
        fig.savefig(OUT / "recurrence_curves.png", dpi=300)
        plt.close(fig)
    print(f"[Fig1] Saved {dst.name}")


# ======================================================================
# Figure 2: Causal evidence panel (ablation + freeze + time-shift)
# ======================================================================
def fig2():
    # Included at 0.5\textwidth in 04finding1.tex while authored at
    # 0.75*TEXT_W, so LaTeX shrinks it to 0.667x. scaled_rc pre-multiplies the
    # point sizes so the on-page text is PAPER_RC-sized; the figure's own
    # dimensions are unchanged, so the paper's layout does not move.
    with plt.rc_context(scaled_rc(0.75 * TEXT_W, 0.5 * TEXT_W)):
        _fig2_body()


def _fig2_body():
    fig, axes = plt.subplots(1, 3, figsize=(0.75 * TEXT_W, 2.35))

    # ── Panel A: Per-step z_H ablation (N=1000) ──
    ax = axes[0]
    abl_path = ROOT / "results" / "Sudoku" / "controlled" / "ablation" / "zH_1000" / "zH" / "aggregate.json"
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
    ax.set_xlabel("Step ablated")
    ax.set_ylabel("$\\Delta$ acc. (pp)")
    ax.set_title("(a) Ablation")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xticks([0, 4, 8, 12, 15])

    # ── Panel B: Freeze z_H ──
    ax = axes[1]
    # Read the controlled N=1000 run, the same family panel (a) draws from, so
    # all three panels share one checkpoint, one puzzle prefix and one baseline
    # (81.77%). The superseded results/Sudoku/freeze/freeze_h run was N=200 with
    # its own 81.98% baseline.
    frz_path = ROOT / "results" / "Sudoku" / "controlled" / "freeze" / "aggregate.json"
    with open(frz_path) as f:
        frz = json.load(f)
    conds = frz["freeze_H"]
    # The controlled run stores every step 0..15; keep the sparse tick set the
    # figure has always shown so only the data source changes, not the layout.
    fx = [s for s in (0, 1, 2, 4, 8, 12) if str(s) in conds]
    losses = [abs(conds[str(s)]["delta_accuracy"]["mean"]) * 100 for s in fx]
    ax.bar(range(len(fx)), losses, color="#e67e22", alpha=0.8, width=0.7)
    ax.set_xticks(range(len(fx)))
    ax.set_xticklabels([str(s) for s in fx])
    ax.set_xlabel("Freeze after $k$")
    ax.set_ylabel("Loss (pp)")
    ax.set_title("(b) Freeze")

    # ── Panel C: Time-shift ──
    ax = axes[2]
    # Controlled N=1000 run (mode=fixed, fixed_donor=10). The superseded
    # results/Sudoku/time_shift run was N=100 over a different, non-overlapping
    # set of donor/recipient pairs.
    ts_path = ROOT / "results" / "Sudoku" / "controlled" / "time_shift" / "aggregate.json"
    with open(ts_path) as f:
        ts = json.load(f)
    pairs = ts["per_pair"]
    # Hold the donor fixed at step 10 and sweep the recipient, so a single
    # experiment supplies both directions: recipient < 10 transplants a late
    # state into an early one (future→past), recipient > 10 the reverse.
    # Six recipients, matching this panel's long-standing bar density; they span
    # the full sweep, so the plotted extremes are the reported extremes.
    _RECIPIENTS = (0, 4, 8, 12, 14, 15)
    future_labels, future_vals = [], []
    past_labels, past_vals = [], []
    for recipient in _RECIPIENTS:
        key = f"10->{recipient}"
        val = pairs.get(key)
        if val is None:
            continue
        delta = val["delta_accuracy"]["mean"] * 100
        if val["donor_step"] > val["recipient_step"]:
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
    # The donor is fixed at step 10, so the recipient index alone identifies each
    # pair; printing the full "10->k" key here would overlap at this panel width.
    # Two-digit recipients sit close together at this panel width, so ease the
    # tick size a little; the donor step belongs in the caption, not the axis,
    # where a longer label would overrun the panel.
    ax.set_xticklabels([lab.split("->")[1] for lab in labels],
                       fontsize=plt.rcParams["xtick.labelsize"] * 0.85)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Recipient step")
    ax.set_ylabel("$\\Delta$ acc. (pp)")
    ax.set_title("(c) Transplant")
    # The direction legend belongs to this panel alone, so it sits inside the
    # axes rather than floating outside the grid. Upper-right is free (both bars
    # under it are negative); the extra top headroom keeps it off the tallest
    # positive bar, and it is set slightly below the panel's text size so the two
    # labels fit within this narrow third axis.
    vmin_ts, vmax_ts = min(vals), max(vals)
    span_ts = vmax_ts - vmin_ts
    ax.set_ylim(vmin_ts - 0.10 * span_ts, vmax_ts + 0.62 * span_ts)
    from matplotlib.patches import Patch
    direction_handles = [Patch(color="#27ae60"), Patch(color="#8e44ad")]
    ax.legend(direction_handles, ["Future→Past", "Past→Future"],
              loc="upper right", frameon=True, framealpha=0.8,
              facecolor="white", edgecolor="none",
              fontsize=plt.rcParams["legend.fontsize"] * 0.85,
              handlelength=1.1, handletextpad=0.35, labelspacing=0.25,
              borderpad=0.3, borderaxespad=0.25)

    # No bottom legend band to reserve any more — the panels use the full height,
    # which in turn needs a real gutter so (b)'s y-label clears (a).
    fig.tight_layout(pad=0.22, w_pad=1.0)
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
    probe_path = ROOT / "results" / "Sudoku" / "probes" / "nonlinear_probes" / "comparison_summary.json"
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
    da_path = ROOT / "results" / "Sudoku" / "directed_ablation" / "e9_directed_ablation" / "aggregate_results.json"
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
    abl_path = ROOT / "results" / "Sudoku" / "controlled" / "ablation" / "zH_1000" / "zH" / "aggregate.json"
    with open(abl_path) as f:
        abl = json.load(f)
    full_delta = abl["all_steps_delta"]["mean"] * 100
    full_ci_lo = abl["all_steps_delta"]["ci_lower"] * 100
    full_ci_hi = abl["all_steps_delta"]["ci_upper"] * 100

    # SAE features (from causal_ablation)
    sae_path = ROOT / "results" / "Sudoku" / "sae_study" / "causal_ablation" / "aggregate.json"
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
    fig1()  # recurrence_curves.pdf   (used in paper)
    fig2()  # causal_evidence_panel.pdf (used in paper)
    # fig3/fig4 produce readout_vs_computation / causal_hierarchy, which are not
    # included in the current paper; keep them best-effort so a data-shape change
    # in those unused figures never blocks the two figures the paper needs.
    for _f in (fig3, fig4):
        try:
            _f()
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {_f.__name__}: {e}")
    print("\nAll figures generated!")
    for f in sorted(OUT.glob("*")):
        print(f"  {f.name}")
