"""Regenerate supporting paper figures from the on-disk result JSON/CSVs:

  probe_sweeps_sudoku_maze.pdf   2x3 probe-accuracy-vs-step sweeps
                                  (Sudoku/Maze/ARC-AGI-2)
  sudoku_probe_pca_variance.pdf  constraint-probe weight-vector PCA (stacked area)
  maze_local_probe_readout.pdf   maze local-feature probes vs majority baseline
  maze_mlp_gain_step15.pdf       MLP-minus-linear probe gains at step 15
  sudoku_mlp_gain_step15.pdf     same, for the six Sudoku constraint targets
  arc_probe_readout_summary.pdf  ARC z_H probe readout at step 15 vs baseline

Layouts/values mirror the figures shipped in the AAAI submission zip; fonts are
Times-matched and embedded as TrueType (Type 42) for AAAI compliance.

The causal-ladder figure has its own traceable generator in
``plot_crosstask_causal_ladder.py``.

Usage:  python scripts/plotting/generate_supporting_paper_figures.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "results", "Final", "figures")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "scripts", "plotting"))
from paper_style import apply_paper_style, COL_W, TEXT_W

# Single source of truth for all paper figures.
apply_paper_style()

C = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # tab10


def jload(p):
    with open(os.path.join(R, p)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------- probe_sweeps_sudoku_maze
def probe_sweeps_sudoku_maze():
    print("[probe_sweeps_sudoku_maze]")
    # Sudoku: e8 constraint-probe sweep CSV
    rows = list(csv.DictReader(open(os.path.join(
        R, "Sudoku/probes/e8_constraint_probes/sweep_results.csv"))))
    sud = {}  # (level, step, target) -> val acc %
    for r in rows:
        sud[(r["z_level"], int(r["step"]), r["target"])] = float(r["val_score"]) * 100
    sud_steps = sorted({k[1] for k in sud})
    sud_series = [
        ("Cell digit", lambda lv, s: sud[(lv, s, "cell_digit")]),
        ("Is given", lambda lv, s: sud[(lv, s, "is_given")]),
        ("Correct", lambda lv, s: sud[(lv, s, "per_cell_correct")]),
        ("Violation avg.", lambda lv, s: np.mean([sud[(lv, s, f"violated_in_{a}")]
                                                  for a in ("row", "col", "box")])),
    ]

    # Maze: hardened linear-probe summary (global path-level targets)
    mz = jload("Maze/hardened/linear_probes/probe_summary_clean.json")["classification"]
    mz_steps = sorted({v["step"] for v in mz.values()})
    mz_series = [
        ("Exact solved", "exact_solved"),
        ("Connects S-G", "connects_start_goal"),
        ("Valid S-G path", "valid_sg_path"),
        ("Optimal path", "valid_optimal_path"),
    ]

    # ARC-AGI-2: one static geometry target and two
    # solution/transformation-dependent targets.
    arc = jload("ARC/arc/hardened/linear_probes/probe_summary.json")
    arc_steps = sorted({v["step"] for v in arc.values()})
    arc_series = [
        ("Output inside", "output_inside_grid", C[1], "-", "o", 3),
        ("Per-cell correct", "per_cell_correct", C[2], "-", "o", 3),
        ("Colour changed", "colour_changed", C[3], "-", "o", 3),
    ]

    # Use only checkpoints observed for every task so that columns are
    # directly comparable and no values need to be interpolated.
    common_steps = sorted(set(sud_steps) & set(mz_steps) & set(arc_steps))
    print("   shared ACT steps:", common_steps)

    # Columns are tasks and rows are recurrent streams. This keeps the
    # cross-task sweep compact enough for a full-width two-column figure.
    fig, axes = plt.subplots(2, 3, figsize=(0.98 * TEXT_W, 4.5))

    for row, lv in enumerate(("H", "L")):
        ax = axes[row, 0]
        for si, (lab, fn) in enumerate(sud_series):
            ys = [fn(lv, s) for s in common_steps]
            ax.plot(common_steps, ys, "-o", ms=3, lw=1.2, color=C[si], label=lab)
            print(f"   Sudoku z_{lv} {lab:14s} " + " ".join(f"{y:.1f}" for y in ys))
        ax.set_title(f"Sudoku $z_{{{lv}}}$")
        ax.set_xticks(common_steps)
        ax.set_ylim(60, 101)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Probe validation accuracy (%)")
        if row == 1:
            ax.legend(fontsize=6, loc="lower right")
        ax.set_xlabel("ACT step")

    for row, lv in enumerate(("z_H", "z_L")):
        ax = axes[row, 1]
        for si, (lab, tgt) in enumerate(mz_series):
            ys = [mz[f"{lv}_step{s}_{tgt}"]["score_mean"] * 100
                  for s in common_steps]
            ax.plot(common_steps, ys, "-o", ms=3, lw=1.2, color=C[si], label=lab)
            print(f"   Maze {lv} {lab:14s} " + " ".join(f"{y:.1f}" for y in ys))
        ax.set_title(f"Maze ${lv}$")
        ax.set_xticks(common_steps)
        ax.set_ylim(60, 101)
        ax.grid(alpha=0.3)
        if row == 1:
            ax.legend(fontsize=6, loc="lower right")
        ax.set_xlabel("ACT step")

    for row, lv in enumerate(("z_H", "z_L")):
        ax = axes[row, 2]
        for lab, tgt, color, linestyle, marker, zorder in arc_series:
            ys = [arc[f"{lv}_step{s}_{tgt}"]["score_mean"] * 100
                  for s in common_steps]
            ax.plot(
                common_steps,
                ys,
                linestyle=linestyle,
                marker=marker,
                ms=3.5,
                lw=1.2,
                color=color,
                markerfacecolor=color,
                markeredgewidth=0.9,
                zorder=zorder,
                label=lab,
            )
            print(f"   ARC {lv} {lab:16s} " + " ".join(f"{y:.1f}" for y in ys))
        ax.set_title(f"ARC-AGI-2 ${lv}$")
        ax.set_xticks(common_steps)
        ax.set_ylim(60, 101)
        ax.grid(alpha=0.3)
        if row == 1:
            ax.legend(fontsize=5.5, loc="lower right")
        ax.set_xlabel("ACT step")

    fig.tight_layout()
    save(fig, "probe_sweeps_sudoku_maze")


# -------------------------------------------------- sudoku_probe_pca_variance
def sudoku_probe_pca_variance():
    print("[sudoku_probe_pca_variance]")
    g = jload("Sudoku/probes/e8_constraint_probes/geometric_analysis.json")
    rows = {r["group"]: np.array(r["singular_values"], dtype=float)
            for r in g["pca_of_weight_vectors"] if r["group"].startswith("z_H")}
    steps = sorted(int(k.split("step")[-1]) for k in rows)
    # Explained variance is s^2 / sum(s^2); normalising the raw singular values
    # instead (s / sum(s)) understates PC1 badly here -- 74-77% vs the true
    # 94-96%, which is what this file's own pca_pc1_ensemble rows report.
    fracs = np.array([rows[f"z_H_step{s}"] ** 2 / (rows[f"z_H_step{s}"] ** 2).sum() * 100
                      for s in steps])  # [steps, PCs]
    for s, f in zip(steps, fracs):
        print(f"   step {s:2d}: PC1={f[0]:.1f}%  PC2={f[1]:.1f}%  PC3={f[2]:.1f}%")

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.stackplot(steps, fracs[:, 0], fracs[:, 1], fracs[:, 2],
                 labels=["PC1", "PC2", "PC3"], colors=C[:3], alpha=0.9)
    ax.set_xlabel("ACT step")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Sudoku constraint-probe weight-vector PCA")
    ax.set_xticks(steps)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", ncol=3, frameon=True)
    ax.grid(alpha=0.3)
    save(fig, "sudoku_probe_pca_variance")


# -------------------------------------------------- maze_local_probe_readout
def maze_local_probe_readout():
    print("[maze_local_probe_readout]")
    mz = jload("Maze/hardened/linear_probes/probe_summary_clean.json")["classification"]
    feats = [
        ("Optimal path", "on_optimal_path"),
        ("Wall", "is_wall"),
        ("Free", "is_free"),
        ("Dead end", "is_dead_end"),
        ("Junction", "is_junction"),
        ("Path corner", "is_corner_on_path"),
        ("Off-path\npassable", "off_path_passable"),
        ("Near start", "near_start_5"),
        ("Near goal", "near_goal_5"),
    ]
    probe = [mz[f"z_H_step0_{t}"]["score_mean"] * 100 for _, t in feats]
    base = [mz[f"z_H_step0_{t}"]["baseline_mean"] * 100 for _, t in feats]
    for (lab, _), p, b in zip(feats, probe, base):
        print(f"   {lab.replace(chr(10), ' '):18s} probe={p:.1f}  baseline={b:.1f}")

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    x = np.arange(len(feats))
    w = 0.38
    ax.bar(x - w / 2, probe, w, color=C[0], label="Probe")
    ax.bar(x + w / 2, base, w, color=C[1], label="Majority baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in feats], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Maze local-feature readout from $z_H$ at step 0")
    ax.set_ylim(45, 102)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save(fig, "maze_local_probe_readout")


# ----------------------------------------------------- maze_mlp_gain_step15
def maze_mlp_gain_step15():
    print("[maze_mlp_gain_step15]")
    mm = jload("Maze/hardened/mlp_probes/probe_summary.json")
    targets = [
        ("Exact\nsolved", "exact_solved"),
        ("Connects\nS-G", "connects_start_goal"),
        ("Valid\nS-G path", "valid_sg_path"),
        ("Optimal\npath", "valid_optimal_path"),
    ]
    gains = {lv: [mm[f"{lv}_step15_{t}"]["delta_mlp_minus_linear"] * 100
                  for _, t in targets] for lv in ("z_H", "z_L")}
    for lv in gains:
        print(f"   {lv}: " + "  ".join(f"{v:+.1f}" for v in gains[lv]))

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(targets))
    w = 0.38
    ax.bar(x - w / 2, gains["z_H"], w, color=C[0], label="$z_H$ MLP $-$ linear")
    ax.bar(x + w / 2, gains["z_L"], w, color=C[1], label="$z_L$ MLP $-$ linear")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in targets], fontsize=8)
    ax.set_ylabel("Validation accuracy gain (pp)")
    ax.set_title("Maze step-15 MLP probe gains")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save(fig, "maze_mlp_gain_step15")


# --------------------------------------------------- sudoku_mlp_gain_step15
def sudoku_mlp_gain_step15():
    print("[sudoku_mlp_gain_step15]")
    path = os.path.join(R, "Final/sudoku/probes/nonlinear_probes/sweep_results.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["step"] == "15"]
    by_key = {(r["z_level"], r["target"]): r for r in rows}
    targets = [
        ("Cell\ndigit", "cell_digit"),
        ("Is\ngiven", "is_given"),
        ("Per-cell\ncorrect", "per_cell_correct"),
        ("Viol.\nrow", "violated_in_row"),
        ("Viol.\ncol", "violated_in_col"),
        ("Viol.\nbox", "violated_in_box"),
    ]
    gains = {lv: [float(by_key[(lv, t)]["delta_val"]) * 100 for _, t in targets]
             for lv in ("H", "L")}
    for lv in gains:
        print(f"   z_{lv}: " + "  ".join(f"{v:+.2f}" for v in gains[lv])
              + f"   mean={np.mean(gains[lv]):+.2f}")

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(targets))
    w = 0.38
    ax.bar(x - w / 2, gains["H"], w, color=C[0], label="$z_H$ MLP $-$ linear")
    ax.bar(x + w / 2, gains["L"], w, color=C[1], label="$z_L$ MLP $-$ linear")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in targets], fontsize=8)
    ax.set_ylabel("Validation accuracy gain (pp)")
    ax.set_title("Sudoku step-15 MLP probe gains")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save(fig, "sudoku_mlp_gain_step15")


# ------------------------------------------------- arc_probe_readout_summary
def arc_probe_readout_summary():
    print("[arc_probe_readout_summary]")
    a = jload("ARC/arc/hardened/linear_probes/probe_summary.json")
    feats = [
        ("Per-cell\ncorrect", "per_cell_correct"),
        ("Input\ninside", "input_inside_grid"),
        ("Output\ninside", "output_inside_grid"),
        ("Colour\nchanged", "colour_changed"),
        ("Same as\ninput", "same_as_input"),
        ("Object\nboundary", "is_object_boundary"),
        ("Input\ncolour", "input_colour"),
        ("Output\ncolour", "output_colour"),
    ]
    probe = [a[f"z_H_step15_{t}"]["score_mean"] * 100 for _, t in feats]
    base = [a[f"z_H_step15_{t}"]["baseline_mean"] * 100 for _, t in feats]
    for (lab, _), p, b in zip(feats, probe, base):
        print(f"   {lab.replace(chr(10), ' '):16s} probe={p:.1f}  baseline={b:.1f}")

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    x = np.arange(len(feats))
    w = 0.38
    ax.bar(x - w / 2, base, w, color=C[0], label="Majority / baseline")
    ax.bar(x + w / 2, probe, w, color=C[1], label="$z_H$ linear probe")
    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in feats], fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ARC-AGI-2 selected $z_H$ probe readout at step 15")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=6.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save(fig, "arc_probe_readout_summary")


if __name__ == "__main__":
    probe_sweeps_sudoku_maze()
    sudoku_probe_pca_variance()
    maze_local_probe_readout()
    maze_mlp_gain_step15()
    sudoku_mlp_gain_step15()
    arc_probe_readout_summary()
    print("done ->", OUT)
