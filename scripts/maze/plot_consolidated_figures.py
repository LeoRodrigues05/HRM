#!/usr/bin/env python3
"""Consolidated cross-task figures for the HRM maze<->Sudoku study.

Every value is read from the on-disk result JSONs (no hand-transcribed numbers)
and echoed to stdout so each figure is traceable. Produces 4 figures (PDF+PNG):

  fig_patch_ablation_asymmetry  z_H per-step ablation vs cross-puzzle patch (maze)
  fig_f3_causal_ladder          full-z_H >> SAE >> probe, Sudoku vs maze
  fig_crosstask_ablation_shape  z_H per-step ablation shape, Sudoku vs maze (norm.)
  fig_dual_metric_ablation      maze z_H/z_L: token-acc vs path-validity

Usage:  python scripts/maze/plot_consolidated_figures.py
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load(p):
    with open(os.path.join(REPO, p)) as f:
        return json.load(f)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- data loaders
def maze_ablation_perstep(level):
    """maze per-step ablation Δvalid_sg_path with CIs, steps 0..15."""
    d = load(f"results/maze/hardened/ablation_controlled/{level}/aggregate.json")
    ps = d["per_step_ablation"]
    steps = sorted((int(k) for k in ps), key=int)
    m, lo, hi = [], [], []
    for s in steps:
        v = ps[str(s)]["maze_metric_deltas"]["valid_sg_path"]
        m.append(v["mean"] * 100); lo.append(v["ci_lower"] * 100); hi.append(v["ci_upper"] * 100)
    return np.array(steps), np.array(m), np.array(lo), np.array(hi)


def maze_patch_allgroup(level):
    """full-grid cross-puzzle patch Δvalid_sg_path (all group), with CIs."""
    d = load("results/maze/hardened/patching_full_steps/aggregate.json")
    node = d["maze_metric_deltas_by_group_level_step"]["valid_sg_path"]["all"][level]
    steps = sorted((int(k) for k in node), key=int)
    m, lo, hi = [], [], []
    for s in steps:
        v = node[str(s)]
        m.append(v["mean"] * 100); lo.append(v["ci_lower"] * 100); hi.append(v["ci_upper"] * 100)
    return np.array(steps), np.array(m), np.array(lo), np.array(hi)


def sudoku_ablation_perstep(level):
    d = load(f"results/controlled/ablation/{level}/aggregate.json")
    ps = d["per_step_ablation"]
    steps = sorted((int(k) for k in ps), key=int)
    m = np.array([ps[str(s)]["delta_accuracy"]["mean"] * 100 for s in steps])
    return np.array(steps), m


# ---------------------------------------------------------------- figures
def fig_asymmetry(outdir):
    print("[fig_patch_ablation_asymmetry]")
    s_ab, m_ab, lo_ab, hi_ab = maze_ablation_perstep("zH")
    s_pa, m_pa, lo_pa, hi_pa = maze_patch_allgroup("H")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.errorbar(s_ab, m_ab, yerr=[m_ab - lo_ab, hi_ab - m_ab], marker="o", ms=4, lw=1.6,
                capsize=2, color="#c0392b", label="z_H ablation (zero at step k)")
    ax.errorbar(s_pa, m_pa, yerr=[m_pa - lo_pa, hi_pa - m_pa], marker="s", ms=5, lw=1.6,
                capsize=2, color="#2c3e50", label="z_H cross-puzzle patch (foreign z_H at step k)")
    ax.set_xlabel("ACT step of intervention"); ax.set_ylabel("Δ valid start→goal path (%)")
    ax.set_title("Maze z_H is replaceable mid-trajectory,\nload-bearing only at the readout (step 15)")
    ax.legend(fontsize=8, loc="lower left"); ax.set_xticks(range(0, 16, 2))
    for s, m in zip(s_ab, m_ab):
        if s == 15: ax.annotate(f"{m:.0f}%", (s, m), textcoords="offset points", xytext=(-2, 8), fontsize=8, color="#c0392b")
    for s, m in zip(s_pa, m_pa):
        if s == 15: ax.annotate(f"{m:.0f}%", (s, m), textcoords="offset points", xytext=(-18, -4), fontsize=8, color="#2c3e50")
    print(f"   ablation s15={m_ab[-1]:.1f}%  patch s15={m_pa[-1]:.1f}%  patch s8={m_pa[s_pa.tolist().index(8)]:.1f}%")
    save(fig, outdir, "fig_patch_ablation_asymmetry")


def fig_f3_ladder(outdir):
    print("[fig_f3_causal_ladder]")
    # maze
    mz = load("results/maze/sae_study/causal_ablation/aggregate.json")["conditions"]
    mz_full = load("results/maze/hardened/ablation_controlled/zH/aggregate.json")["all_steps_maze_deltas"]["valid_sg_path"]["mean"] * 100
    # sudoku
    sd = load("results/sae_study/causal_ablation/aggregate.json")["conditions"]
    sd_full = load("results/controlled/ablation/zH/aggregate.json")["all_steps_delta"]["mean"] * 100

    labels = ["full z_H\nablation", "SAE top-50", "SAE random-50", "probe\ndirections", "random\ndirections"]
    maze_vals = [mz_full,
                 mz["sae_top_features"]["mean_delta_valid_sg_path"] * 100,
                 mz["random_sae_features"]["mean_delta_valid_sg_path"] * 100,
                 mz["probe_directions"]["mean_delta_valid_sg_path"] * 100,
                 mz["random_directions"]["mean_delta_valid_sg_path"] * 100]
    sud_vals = [sd_full,
                sd["sae_top_features"]["mean_delta_acc"] * 100,
                sd["random_sae_features"]["mean_delta_acc"] * 100,
                sd["probe_directions"]["mean_delta_acc"] * 100,
                sd["random_directions"]["mean_delta_acc"] * 100]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.bar(x - w/2, sud_vals, w, color="#2980b9", label="Sudoku (Δ cell acc)")
    ax.bar(x + w/2, maze_vals, w, color="#e67e22", label="Maze (Δ valid path)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ task accuracy (%)")
    ax.set_title("Computation is deeply distributed (both tasks):\nfull z_H ≫ SAE features ≫ probe directions ≈ 0")
    ax.legend(fontsize=8)
    for xi, v in zip(x - w/2, sud_vals): ax.annotate(f"{v:.1f}", (xi, v), ha="center",
        va="top" if v < 0 else "bottom", fontsize=7, color="#1b5e8a")
    for xi, v in zip(x + w/2, maze_vals): ax.annotate(f"{v:.1f}", (xi, v), ha="center",
        va="top" if v < 0 else "bottom", fontsize=7, color="#a85710")
    print(f"   sudoku: {[round(v,2) for v in sud_vals]}")
    print(f"   maze:   {[round(v,2) for v in maze_vals]}")
    save(fig, outdir, "fig_f3_causal_ladder")


def fig_crosstask_shape(outdir):
    print("[fig_crosstask_ablation_shape]")
    ssu, msu = sudoku_ablation_perstep("zH")
    sma, mma, _, _ = maze_ablation_perstep("zH")
    # normalize each by its own step-15 magnitude to compare SHAPE
    nsu = msu / abs(msu[list(ssu).index(15)])
    nma = mma / abs(mma[list(sma).index(15)])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(ssu, nsu, marker="o", ms=4, lw=1.8, color="#2980b9", label="Sudoku (rising / distributed)")
    ax.plot(sma, nma, marker="s", ms=4, lw=1.8, color="#e67e22", label="Maze (flat then step-15 spike)")
    ax.set_xlabel("ACT step of z_H ablation")
    ax.set_ylabel("per-step damage ÷ |step-15 damage|")
    ax.set_title("Task-dependent depth: where z_H damage accumulates\n(normalized per-step z_H ablation)")
    ax.legend(fontsize=8, loc="upper left"); ax.set_xticks(range(0, 16, 2))
    print(f"   sudoku norm: s0={nsu[0]:.2f} s8={nsu[list(ssu).index(8)]:.2f} s15={nsu[-1]:.2f}")
    print(f"   maze   norm: s0={nma[0]:.2f} s8={nma[list(sma).index(8)]:.2f} s15={nma[-1]:.2f}")
    save(fig, outdir, "fig_crosstask_ablation_shape")


def fig_dual_metric(outdir):
    print("[fig_dual_metric_ablation]")
    rows = []
    for lvl in ("zH", "zL"):
        d = load(f"results/maze/hardened/ablation_controlled/{lvl}/aggregate.json")
        tok = d["all_steps_delta"]["mean"] * 100
        tok_lo = d["all_steps_delta"]["ci_lower"] * 100; tok_hi = d["all_steps_delta"]["ci_upper"] * 100
        pv = d["all_steps_maze_deltas"]["valid_sg_path"]
        rows.append((lvl, tok, tok_lo, tok_hi, pv["mean"] * 100, pv["ci_lower"] * 100, pv["ci_upper"] * 100))
    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(0, color="0.7", lw=0.8)
    tok = [r[1] for r in rows]; tok_err = [[r[1]-r[2] for r in rows], [r[3]-r[1] for r in rows]]
    pv = [r[4] for r in rows]; pv_err = [[r[4]-r[5] for r in rows], [r[6]-r[4] for r in rows]]
    ax.bar(x - w/2, tok, w, yerr=tok_err, capsize=3, color="#95a5a6", label="token accuracy (misleading)")
    ax.bar(x + w/2, pv, w, yerr=pv_err, capsize=3, color="#c0392b", label="valid start→goal path (true objective)")
    ax.set_xticks(x); ax.set_xticklabels(["z_H all-steps ablation", "z_L all-steps ablation"])
    ax.set_ylabel("Δ (%)")
    ax.set_title("Metric artifact: token accuracy hides the damage\nthat path-validity reveals")
    ax.legend(fontsize=8, loc="lower left")
    for xi, v in zip(x - w/2, tok): ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w/2, pv): ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="top", fontsize=8)
    print(f"   z_H token={rows[0][1]:.1f}% path={rows[0][4]:.1f}% | z_L token={rows[1][1]:.1f}% path={rows[1][4]:.1f}%")
    save(fig, outdir, "fig_dual_metric_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/maze/plots")
    args = ap.parse_args()
    outdir = os.path.join(REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    fig_asymmetry(outdir)
    fig_f3_ladder(outdir)
    fig_crosstask_shape(outdir)
    fig_dual_metric(outdir)
    print(f"[done] 4 figures in {args.outdir}")


if __name__ == "__main__":
    main()
