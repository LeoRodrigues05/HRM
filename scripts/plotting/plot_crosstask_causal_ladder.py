#!/usr/bin/env python3
r"""Generate the cross-task intervention-hierarchy figure (Figure~\\ref{fig:causal_ladder}).

Grouped bars over Sudoku / Maze / ARC-AGI-2 comparing four SAE / probe ablation
conditions. Every value is read from the on-disk causal-ablation aggregates and
echoed to stdout so each bar is traceable to a file.

The figure is authored at TEXT_W but included at \linewidth (= \columnwidth)
in 06finding3.tex, so LaTeX shrinks it to 0.47x. scaled_rc pre-multiplies the
point sizes so the on-page text comes out PAPER_RC-sized, without changing the
figure's dimensions or the paper's layout.

Outputs → results/Final/figures/crosstask_causal_ladder.{pdf,png}

Usage:  python scripts/plotting/plot_crosstask_causal_ladder.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import apply_paper_style, scaled_rc, COL_W, TEXT_W

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "Final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

apply_paper_style()

# (display name, metric sub-label, aggregate path, the mean_* key it stores)
#
# These read from results/Final/, which is the tree Table~\ref{tab:sae_causal} was
# built from. It matters only for Sudoku: the run under results/Sudoku/ was
# invoked without the probe/random-direction conditions (see its _meta.json argv),
# so it stores 0.0 for both with n_samples=0 — absence of data, not absence of
# effect. Maze and ARC are byte-identical between the two trees.
TASKS = [
    ("Sudoku", "cell acc.",
     "results/Final/sudoku/sae_study/causal_ablation/aggregate.json", "mean_delta_acc"),
    ("Maze", "valid S–G path",
     "results/Final/maze/sae_study/causal_ablation/aggregate.json", "mean_delta_valid_sg_path"),
    ("ARC-AGI-2", "colour cell acc.",
     "results/Final/arc/sae_study/causal_ablation/aggregate.json", "mean_delta_colour_cell_acc"),
]

# Condition key → (legend label, colour)
CONDS = [
    ("sae_top_features",    "Top-50 SAE",       "#1f77b4"),
    ("random_sae_features", "Random-50 SAE",    "#ff7f0e"),
    ("probe_directions",    "Probe dirs.",      "#2ca02c"),
    ("random_directions",   "Random dirs.",     "#d62728"),
]


def main() -> None:
    with plt.rc_context(scaled_rc(TEXT_W, COL_W)):
        _build()


def _build() -> None:
    values = {}
    for task, _sub, rel, key in TASKS:
        with open(ROOT / rel) as f:
            conds = json.load(f)["conditions"]
        # A condition that was never run still writes mean=0.0, which would plot
        # as a real "no effect" bar. Refuse to draw one.
        for c, label, _col in CONDS:
            if conds[c].get("n_samples") == 0:
                raise SystemExit(
                    f"{task}: condition '{c}' ({label}) has n_samples=0 in {rel} "
                    "— it was not run, so its 0.0 is not a measurement.")
        values[task] = [conds[c][key] * 100 for c, _l, _c in CONDS]
        print(f"  {task:10s} ({key}): " +
              "  ".join(f"{l}={v:+.2f}pp" for (_c, l, _col), v in zip(CONDS, values[task])))

    fig, ax = plt.subplots(figsize=(TEXT_W, 3.2))

    x = np.arange(len(TASKS))
    # Bars fill 0.88 of the slot rather than 0.8: the extra pitch is what lets
    # every value label sit at one uniform distance from its bar without
    # neighbours touching.
    w = 0.90 / len(CONDS)
    bars = []
    for ci, (_key, label, colour) in enumerate(CONDS):
        vals = [values[t][ci] for t, _s, _r, _k in TASKS]
        pos = x + (ci - (len(CONDS) - 1) / 2) * w
        ax.bar(pos, vals, w, label=label, color=colour)
        bars.extend((p, v, ci) for p, v in zip(pos, vals))

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t, _s, _r, _k in TASKS])
    ax.set_ylabel("$\\Delta$ task metric (pp)")
    ax.set_title("Intervention hierarchy across tasks", pad=6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Room below the bars for both the value labels and the legend, which sits
    # bottom-right where the (small, positive) ARC bars leave the axes empty.
    lo = min(min(v) for v in values.values())
    hi = max(max(v) for v in values.values())
    span = hi - lo
    ax.set_ylim(lo - 0.68 * span, hi + 0.30 * span)
    # Ticks (and therefore gridlines) only across the band the bars occupy; the
    # margins above and below exist for the labels and legend, not for data.
    ax.set_yticks(np.arange(np.floor(lo), np.ceil(hi) + 0.5, 1.0))
    # Flush into the bottom-right corner: that keeps it clear of the deepest
    # Maze label, which sits well to its left.
    ax.legend(loc="lower right", ncol=2, frameon=False,
              fontsize=plt.rcParams["legend.fontsize"],
              handlelength=1.2, handletextpad=0.35, columnspacing=1.1,
              labelspacing=0.35, borderaxespad=0.0, borderpad=0.0)

    # Per-bar values, outside the bar (below if negative, above if positive).
    # All labels sit the same distance from their bar. That only works because
    # the widened bar pitch above leaves each label just enough room; the size is
    # set to the largest that keeps neighbours apart.
    pad = 0.022 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for pos, val, _ci in bars:
        ax.text(pos, val - pad if val < 0 else val + pad, f"{val:+.2f}",
                ha="center", va="top" if val < 0 else "bottom",
                fontsize=plt.rcParams["xtick.labelsize"] * 0.71)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "crosstask_causal_ladder.pdf")
    fig.savefig(OUT / "crosstask_causal_ladder.png", dpi=300)
    plt.close(fig)
    print(f"[Ladder] Saved {OUT / 'crosstask_causal_ladder.pdf'}")


if __name__ == "__main__":
    main()
