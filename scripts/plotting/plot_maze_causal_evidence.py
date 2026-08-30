#!/usr/bin/env python3
r"""Generate the Maze causal-evidence panel used in \S4 (Figure~\ref{fig}).

Three panels, all in percentage points:
  (a) All-step z_H / z_L ablation across four Maze metrics
  (b) Per-step ablation, valid start-goal path
  (c) Cross-puzzle donor patching, valid start-goal path

Every value is read from the hardened Maze aggregates and echoed to stdout so
each bar is traceable to a file.

The figure is authored at 0.85*TEXT_W, exactly the \includegraphics width used
in 04finding1.tex, so LaTeX applies a 1.0x scale and the PAPER_RC point sizes
are the sizes the reader sees.

Outputs → results/Final/figures/maze_causal_evidence_panel.{pdf,png}

Usage:  python scripts/plotting/plot_maze_causal_evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import apply_paper_style, scaled_rc, TEXT_W

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "Final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

apply_paper_style()

ABLATION = {lvl: ROOT / "results" / "Maze" / "hardened" / "ablation_controlled" /
            lvl / "aggregate.json" for lvl in ("zH", "zL")}
PATCHING = ROOT / "results" / "Maze" / "hardened" / "patching_full_steps" / "aggregate.json"

# Geometry is deliberately identical to the Sudoku causal panel
# (generate_paper_figures.fig2): same authored size, same display width, same
# scale compensation. Drop the two in at the same \includegraphics width and
# their on-page text comes out the same size.
FIG_W = 0.75 * TEXT_W
FIG_H = 2.35
DISPLAY_W = 0.5 * TEXT_W

C_H, C_L = "#1f77b4", "#ff7f0e"

# (aggregate key, tick label). "Valid path" and "Connectivity" are the short
# forms; the caption spells both out. Rotated rather than stacked, because at
# half-text-width panel (a) is too narrow for four horizontal category labels —
# the same treatment the Sudoku panel gives its transplant-pair labels.
METRICS = [
    ("token_acc", "Token acc."),
    ("path_cell_acc", "Path-cell acc."),
    ("valid_sg_path", "Valid path"),
    ("connects_start_goal", "Connectivity"),
]

# The patching run only covers these steps, so panel (b) is restricted to the
# same ones — the two panels are meant to be read side by side.
STEPS = [0, 4, 8, 12, 14, 15]

# The legends sit on top of the bars (there is no free corner wide enough at
# this text size), so they get a semi-transparent white backing instead. Labels
# are bare state names: the panel title already says ablation vs. patching, and
# "Ablate $z_H$" does not fit a third of a half-width figure.
LEGEND_KW = dict(loc="lower left", frameon=True, framealpha=0.75,
                 facecolor="white", edgecolor="none", handlelength=1.0,
                 handletextpad=0.35, labelspacing=0.25, borderpad=0.3)


def _pct(node: dict) -> float:
    return node["mean"] * 100


def main() -> None:
    with plt.rc_context(scaled_rc(FIG_W, DISPLAY_W)):
        _build()


def _build() -> None:
    abl = {lvl: json.loads(p.read_text()) for lvl, p in ABLATION.items()}
    patch = json.loads(PATCHING.read_text())

    all_step = {lvl: [_pct(abl[lvl]["all_steps_maze_deltas"][k]) for k, _lab in METRICS]
                for lvl in ("zH", "zL")}
    per_step = {lvl: [_pct(abl[lvl]["per_step_ablation"][str(s)]
                           ["maze_metric_deltas"]["valid_sg_path"]) for s in STEPS]
                for lvl in ("zH", "zL")}
    patch_all = patch["maze_metric_deltas_by_group_level_step"]["valid_sg_path"]["all"]
    patched = {lvl: [_pct(patch_all[key][str(s)]) for s in STEPS]
               for lvl, key in (("zH", "H"), ("zL", "L"))}

    for name, d in (("all-step", all_step), ("per-step", per_step), ("patch", patched)):
        for lvl in ("zH", "zL"):
            print(f"  {name:9s} {lvl}: " + "  ".join(f"{v:+.1f}" for v in d[lvl]))

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    # Six step ticks in a panel this narrow need to come down a notch, or
    # "12 14 15" runs together.
    STEP_TICK_FS = plt.rcParams["xtick.labelsize"] * 0.8

    # Full-wording titles need a slightly smaller size than the panel default:
    # a third of a half-width figure is about 1.7in, and "(a) All-step ablation"
    # overruns that at the inherited title size.
    TITLE_FS = plt.rcParams["axes.titlesize"] * 0.86

    # ── (a) All-step ablation ──
    ax = axes[0]
    x = np.arange(len(METRICS))
    ax.bar(x - 0.2, all_step["zH"], 0.4, color=C_H, label="$z_H$")
    ax.bar(x + 0.2, all_step["zL"], 0.4, color=C_L, label="$z_L$")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _k, lab in METRICS], rotation=55, ha="right",
                       rotation_mode="anchor",
                       fontsize=plt.rcParams["xtick.labelsize"] * 0.85)
    ax.set_ylabel("$\\Delta$ metric (pp)")
    ax.set_title("(a) All-step ablation", fontsize=TITLE_FS)

    # ── (b) Per-step ablation ──
    ax = axes[1]
    x = np.arange(len(STEPS))
    ax.bar(x - 0.2, per_step["zH"], 0.4, color=C_H, label="$z_H$")
    ax.bar(x + 0.2, per_step["zL"], 0.4, color=C_L, label="$z_L$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in STEPS], fontsize=STEP_TICK_FS)
    ax.set_xlabel("Ablated step")
    ax.set_ylabel("$\\Delta$ path (pp)")
    ax.set_title("(b) Per-step ablation", fontsize=TITLE_FS)

    # ── (c) Cross-puzzle donor patching ──
    ax = axes[2]
    ax.bar(x - 0.2, patched["zH"], 0.4, color=C_H, label="$z_H$")
    ax.bar(x + 0.2, patched["zL"], 0.4, color=C_L, label="$z_L$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in STEPS], fontsize=STEP_TICK_FS)
    ax.set_xlabel("Patched step")
    ax.set_ylabel("$\\Delta$ path (pp)")
    ax.set_title("(c) Donor patching", fontsize=TITLE_FS)

    # A little slack below the deepest bar in each panel so the lower-left legend
    # sits mostly in empty space and only grazes the bars behind it.
    for ax, vals in zip(axes, (all_step, per_step, patched)):
        flat = vals["zH"] + vals["zL"]
        lo, hi = min(flat), max(flat)
        span = hi - lo
        ax.set_ylim(lo - 0.18 * span, hi + 0.08 * span)
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(**LEGEND_KW)

    fig.tight_layout(pad=0.25, w_pad=0.5)
    fig.savefig(OUT / "maze_causal_evidence_panel.pdf")
    fig.savefig(OUT / "maze_causal_evidence_panel.png", dpi=300)
    plt.close(fig)
    print(f"[Maze causal] Saved {OUT / 'maze_causal_evidence_panel.pdf'}")


if __name__ == "__main__":
    main()
