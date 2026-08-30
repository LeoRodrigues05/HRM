#!/usr/bin/env python3
r"""Generate the cross-task readout-vs-causality figure (Figure~\ref{fig:readout_vs_causality}).

Three panels:
  (a) Linear probe readout, z_H at step 15, across Sudoku / Maze / ARC-AGI-2
  (b) Directed ablation of probe directions vs. rank-matched random directions
  (c) Maze only: near-identical readout from both streams, very different causal locus

Every value is read from the on-disk aggregates and echoed to stdout so each bar
is traceable to a file.

The figure is authored at 0.95*TEXT_W, the \includegraphics width used in
05finding2.tex, so LaTeX applies a 1.0x scale and the PAPER_RC point sizes are
the sizes the reader sees.

Outputs → results/Final/figures/readout_vs_causality_crosstask.{pdf,png}

Usage:  python scripts/plotting/plot_readout_vs_causality.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paper_style import apply_paper_style, TEXT_W

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "Final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

apply_paper_style()

FIG_W = 0.95 * TEXT_W
FIG_H = 1.95

SUDOKU_PROBES = ROOT / "results/Sudoku/probes/e8_constraint_probes/probe_summary.json"
MAZE_PROBES = ROOT / "results/Maze/hardened/linear_probes/probe_summary.json"
ARC_PROBES = ROOT / "results/ARC/arc/hardened/linear_probes/probe_summary.json"
MAZE_PATCHING = ROOT / "results/Maze/hardened/patching_full_steps/aggregate.json"

# Panel (b) reads results/Final/, the tree Table~\ref{tab:sae_causal} was built
# from; the run under results/Sudoku/ omitted the direction conditions entirely.
ABLATION = {
    "Sudoku": (ROOT / "results/Final/sudoku/sae_study/causal_ablation/aggregate.json",
               "mean_delta_acc", "Sudoku\n($\\Delta$ cell acc.)"),
    "Maze": (ROOT / "results/Final/maze/sae_study/causal_ablation/aggregate.json",
             "mean_delta_valid_sg_path", "Maze\n($\\Delta$ valid path)"),
    "ARC-AGI-2": (ROOT / "results/Final/arc/sae_study/causal_ablation/aggregate.json",
                  "mean_delta_colour_cell_acc", "ARC-AGI\n($\\Delta$ colour cell)"),
}

# Donor patching at step 0 is a no-op for both streams (exactly 0.0), so panel
# (c) averages the steps where an intervention actually happens — the same
# "from step 4 onward" window quoted in Section 4.3.
PATCH_STEPS = [4, 8, 12, 14, 15]

TASK_COLOURS = {                      # (darker bar, lighter bar)
    "Sudoku": ("#c0392b", "#e08a80"),
    "Maze": ("#2e9e6b", "#8fd4b0"),
    "ARC-AGI": ("#8e6bbf", "#c3aede"),
}
C_PROBE, C_RANDOM = "#1f77b4", "#b0b0b0"
C_ZH, C_ZL = "#1f77b4", "#ff7f0e"

VAL_FS = 0.8          # value annotations, relative to the tick size
TICK_FS = 0.85        # rotated category labels, relative to the tick size


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _score(row: dict) -> float:
    """Held-out accuracy, whichever key this task's summary uses."""
    return float(row["score_mean"] if "score_mean" in row else row["val_mean"]) * 100


def collect() -> dict:
    sud, maz, arc = _load(SUDOKU_PROBES), _load(MAZE_PROBES), _load(ARC_PROBES)

    viol = np.mean([_score(sud[f"z_H_step15_violated_in_{t}"]) for t in ("row", "col", "box")])
    readout = [
        ("Sudoku", "Violation (avg)", viol),
        ("Sudoku", "Cell correct", _score(sud["z_H_step15_per_cell_correct"])),
        ("Maze", "Valid path", _score(maz["z_H_step15_valid_sg_path"])),
        ("Maze", "Connectivity", _score(maz["z_H_step15_connects_start_goal"])),
        ("ARC-AGI", "Colour changed", _score(arc["z_H_step15_colour_changed"])),
        ("ARC-AGI", "Cell correct", _score(arc["z_H_step15_per_cell_correct"])),
    ]

    ablation = {}
    for task, (path, key, _lab) in ABLATION.items():
        conds = _load(path)["conditions"]
        for cond in ("probe_directions", "random_directions"):
            if conds[cond].get("n_samples") == 0:
                raise SystemExit(
                    f"{task}: '{cond}' has n_samples=0 in {path} — not a measurement.")
        ablation[task] = (conds["probe_directions"][key] * 100,
                          conds["random_directions"][key] * 100)

    patch = _load(MAZE_PATCHING)["maze_metric_deltas_by_group_level_step"]
    patch = patch["valid_sg_path"]["all"]
    maze_cmp = {
        "readout": (_score(maz["z_H_step15_valid_sg_path"]),
                    _score(maz["z_L_step15_valid_sg_path"])),
        "patch": tuple(float(np.mean([patch[lvl][str(s)]["mean"] * 100
                                      for s in PATCH_STEPS])) for lvl in ("H", "L")),
    }
    return {"readout": readout, "ablation": ablation, "maze": maze_cmp}


def main() -> None:
    d = collect()

    print("  (a) probe readout, z_H step 15")
    for task, lab, v in d["readout"]:
        print(f"        {task:8s} {lab:16s} {v:5.2f}%")
    print("  (b) directed ablation (probe / random directions)")
    for task, (p, r) in d["ablation"].items():
        print(f"        {task:10s} {p:+.2f}pp / {r:+.2f}pp")
    print(f"  (c) maze readout z_H/z_L      : {d['maze']['readout'][0]:.2f} / "
          f"{d['maze']['readout'][1]:.2f}%")
    print(f"      maze donor patching (4-15): {d['maze']['patch'][0]:+.2f} / "
          f"{d['maze']['patch'][1]:+.2f}pp")

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(FIG_W, FIG_H), gridspec_kw={"width_ratios": [1.15, 1, 0.85]})
    tick_fs = plt.rcParams["xtick.labelsize"] * TICK_FS
    val_fs = plt.rcParams["xtick.labelsize"] * VAL_FS

    # ── (a) Linear probe readout ──
    x = np.arange(len(d["readout"]))
    colours = [TASK_COLOURS[t][i % 2] for i, (t, _l, _v) in enumerate(d["readout"])]
    vals = [v for _t, _l, v in d["readout"]]
    axa.bar(x, vals, 0.72, color=colours)
    for xi, v in zip(x, vals):
        axa.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=val_fs, fontweight="bold")
    axa.axhline(50, color="gray", ls="--", lw=0.7, alpha=0.7)
    axa.set_xticks(x)
    axa.set_xticklabels([l for _t, l, _v in d["readout"]], rotation=30, ha="right",
                        rotation_mode="anchor", fontsize=tick_fs)
    axa.set_ylabel("Probe accuracy (%)")
    axa.set_ylim(0, 112)
    axa.set_yticks([0, 25, 50, 75, 100])
    axa.set_title("(a) Linear probe readout")
    axa.legend(handles=[Patch(color=TASK_COLOURS[t][0], label=t) for t in TASK_COLOURS],
               loc="lower right", fontsize=val_fs, frameon=True, framealpha=0.8,
               facecolor="white", edgecolor="none", handlelength=1.0,
               handletextpad=0.35, labelspacing=0.2, borderpad=0.25)

    # ── (b) Directed ablation of probe directions ──
    tasks = list(ABLATION)
    x = np.arange(len(tasks))
    probe = [d["ablation"][t][0] for t in tasks]
    rand = [d["ablation"][t][1] for t in tasks]
    axb.bar(x - 0.2, probe, 0.4, color=C_PROBE, label="Probe directions")
    axb.bar(x + 0.2, rand, 0.4, color=C_RANDOM, label="Random directions")
    for xi, v in list(zip(x - 0.2, probe)) + list(zip(x + 0.2, rand)):
        axb.text(xi, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=val_fs)
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xticks(x)
    axb.set_xticklabels([ABLATION[t][2] for t in tasks], rotation=30, ha="right",
                        rotation_mode="anchor", fontsize=tick_fs)
    axb.set_ylabel("$\\Delta$ task metric (pp)")
    axb.set_ylim(-1.0, 1.0)
    axb.set_title("(b) Directed ablation of probe directions")
    axb.legend(loc="lower left", fontsize=val_fs, frameon=True, framealpha=0.8,
               facecolor="white", edgecolor="none", handlelength=1.0,
               handletextpad=0.35, labelspacing=0.2, borderpad=0.25)

    # ── (c) Maze: readout vs. causal locus ──
    x = np.arange(2)
    zh = [d["maze"]["readout"][0], d["maze"]["patch"][0]]
    zl = [d["maze"]["readout"][1], d["maze"]["patch"][1]]
    axc.bar(x - 0.2, zh, 0.4, color=C_ZH, label="$z_H$")
    axc.bar(x + 0.2, zl, 0.4, color=C_ZL, label="$z_L$")
    for xi, v in list(zip(x - 0.2, zh)) + list(zip(x + 0.2, zl)):
        axc.text(xi, v + (3 if v >= 0 else -3), f"{v:.1f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=val_fs)
    axc.axhline(0, color="black", lw=0.8)
    axc.set_xticks(x)
    axc.set_xticklabels(["Probe readout\nvalid path (%)",
                         "Donor patching\n$\\Delta$ valid path (pp)"],
                        rotation=30, ha="right", rotation_mode="anchor",
                        fontsize=tick_fs)
    axc.set_ylabel("% / pp")
    axc.set_ylim(-70, 120)
    axc.set_title("(c) Maze: readout vs. causal locus")
    axc.legend(loc="lower left", fontsize=val_fs, frameon=True, framealpha=0.8,
               facecolor="white", edgecolor="none", handlelength=1.0,
               handletextpad=0.35, labelspacing=0.2, borderpad=0.25)

    fig.tight_layout(pad=0.25, w_pad=0.9)
    fig.savefig(OUT / "readout_vs_causality_crosstask.pdf")
    fig.savefig(OUT / "readout_vs_causality_crosstask.png", dpi=300)
    plt.close(fig)
    print(f"[Readout vs causality] Saved {OUT / 'readout_vs_causality_crosstask.pdf'}")


if __name__ == "__main__":
    main()
