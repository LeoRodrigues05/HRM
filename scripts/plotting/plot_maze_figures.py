#!/usr/bin/env python3
"""Generate Maze figures mirroring the Sudoku figure set.

Reproduces, for the Maze task, the kinds of plots that already exist for Sudoku
(results/controlled/figures/fig{1,2,3}, results/sae_study/plots/*,
plot_zh_trajectories), but scored on **path-validity**:

  fig1_ablation_step_sensitivity   per-step z_H/z_L ablation Δ valid_sg_path (+CI)
  fig2_freeze_crossover            freeze-from-k Δ valid_sg_path, z_H/z_L (+CI)
  fig3_time_shift_transfer         cross-step transfer Δ valid_sg_path (+CI)
  fig4_step_dynamics               path metrics vs ACT step
  fig5_probe_decodability          probe accuracy heatmaps (local + global, z_H/z_L)
  fig6_probe_mlp_vs_linear         linear vs MLP probe accuracy per feature
  fig7_sae_frontier                SAE reconstruction–sparsity frontier (d×λ)
  fig8_trajectory_pca              z_H trajectory in PCA space + per-step path-validity
  fig9_probe_geometry              feature-direction cosine matrix + PC1 per step

Each figure is written as PNG + PDF to results/maze/plots/. Pure CPU.

Usage:  python scripts/plotting/plot_maze_figures.py
"""
from __future__ import annotations
import os, sys, json, csv
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/maze/plots"
os.makedirs(OUT, exist_ok=True)
H_COLOR, L_COLOR = "#1f77b4", "#ff7f0e"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load(p) -> Optional[dict]:
    try:
        return json.load(open(p))
    except Exception:
        return None


def mlh(node):
    """(mean, lo, hi) from a {mean,ci_lower,ci_upper} dict or a bare number."""
    if isinstance(node, dict) and "mean" in node:
        return node.get("mean"), node.get("ci_lower"), node.get("ci_upper")
    try:
        return float(node), None, None
    except Exception:
        return None, None, None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png")


def per_step_series(agg, metric="valid_sg_path"):
    """Return (steps, mean, lo, hi) arrays from a per_step_ablation dict/list."""
    ps = agg.get("per_step_ablation", {})
    items = ps.items() if isinstance(ps, dict) else enumerate(ps)
    rows = []
    for k, v in items:
        m, lo, hi = mlh(v.get("maze_metric_deltas", {}).get(metric))
        if m is not None:
            rows.append((int(k), m * 100, (lo or m) * 100, (hi or m) * 100))
    rows.sort()
    a = np.array(rows)
    return (a[:, 0], a[:, 1], a[:, 2], a[:, 3]) if len(a) else (None,) * 4


# ───────────────────────── figures ─────────────────────────

def fig1_ablation():
    zh = load("results/maze/hardened/ablation_controlled/zH/aggregate.json")
    zl = load("results/maze/hardened/ablation_controlled/zL/aggregate.json")
    if not zh:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for agg, color, lab in [(zh, H_COLOR, "z_H"), (zl, L_COLOR, "z_L")]:
        if not agg:
            continue
        s, m, lo, hi = per_step_series(agg)
        if s is None:
            continue
        ax.plot(s, m, "-o", color=color, label=lab, ms=4)
        ax.fill_between(s, lo, hi, color=color, alpha=0.18)
        allm, allo, allhi = mlh(agg.get("all_steps_maze_deltas", {}).get("valid_sg_path"))
        if allm is not None:
            ax.axhline(allm * 100, color=color, ls="--", alpha=0.6,
                       label=f"{lab} all-steps ({allm*100:.0f}%)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("ACT step ablated"); ax.set_ylabel("Δ valid start→goal path (%)")
    ax.set_title("Maze — single-step ablation sensitivity (path-validity)")
    ax.legend(fontsize=8)
    save(fig, "fig1_ablation_step_sensitivity")


def fig2_freeze():
    agg = load("results/maze/hardened/freeze_controlled/aggregate.json")
    if not agg:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, color, lab in [("freeze_H", H_COLOR, "z_H"), ("freeze_L", L_COLOR, "z_L")]:
        node = agg.get(key, {})
        items = node.items() if isinstance(node, dict) else enumerate(node)
        rows = []
        for k, v in items:
            m, lo, hi = mlh(v.get("maze_metric_deltas", {}).get("valid_sg_path"))
            if m is not None:
                rows.append((int(k), m * 100, (lo or m) * 100, (hi or m) * 100))
        rows.sort()
        if not rows:
            continue
        a = np.array(rows)
        ax.plot(a[:, 0], a[:, 1], "-o", color=color, label=lab, ms=4)
        ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=color, alpha=0.18)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("freeze stream from step k onward"); ax.set_ylabel("Δ valid start→goal path (%)")
    ax.set_title("Maze — freeze crossover (path-validity)")
    ax.legend(fontsize=8)
    save(fig, "fig2_freeze_crossover")


def fig3_timeshift():
    agg = load("results/maze/hardened/time_shift_controlled/aggregate.json")
    if not agg:
        return
    pp = agg.get("per_pair", {})
    pairs = list(pp.values()) if isinstance(pp, dict) else pp
    rows = []
    for p in pairs:
        m, lo, hi = mlh(p.get("maze_metric_deltas", {}).get("valid_sg_path"))
        if m is not None:
            rows.append((p.get("donor_step"), p.get("recipient_step"), m * 100,
                         (lo or m) * 100, (hi or m) * 100))
    if not rows:
        return
    # two sweeps: vary recipient (donor fixed at mode), vary donor (recipient fixed)
    donors = [r[0] for r in rows]; recips = [r[1] for r in rows]
    fd = max(set(donors), key=donors.count); frec = max(set(recips), key=recips.count)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, fixed, idx, xlab, title in [
        (axes[0], fd, 1, "recipient step", f"donor fixed = {fd}"),
        (axes[1], frec, 0, "donor step", f"recipient fixed = {frec}")]:
        # idx=1 → vary recipient with donor fixed; idx=0 → vary donor with recipient fixed
        sel = sorted([r for r in rows if (r[0] if idx == 1 else r[1]) == fixed], key=lambda r: r[idx])
        if not sel:
            continue
        a = np.array([(r[idx], r[2], r[3], r[4]) for r in sel])
        ax.plot(a[:, 0], a[:, 1], "-o", color="#2ca02c", ms=4)
        ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color="#2ca02c", alpha=0.18)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(xlab); ax.set_ylabel("Δ valid start→goal path (%)"); ax.set_title(title)
    fig.suptitle("Maze — time-shift transfer (path-validity)")
    save(fig, "fig3_time_shift_transfer")


def fig4_stepdyn():
    agg = load("results/maze/hardened/step_dynamics/aggregate.json")
    if not agg:
        return
    ps = agg.get("per_step", [])
    if not ps:
        return
    steps = [p["step"] for p in ps]

    def val(node):  # metrics are {mean,std,n} dicts; return (mean, 95%-half-width)
        if isinstance(node, dict):
            m, s, n = node.get("mean"), node.get("std"), node.get("n")
            return m, (1.96 * s / (n ** 0.5) if (s is not None and n) else 0.0)
        return float(node), 0.0
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, c in [("exact_solved", "#d62728"), ("valid_sg_path", "#1f77b4"),
                 ("connects_start_goal", "#2ca02c"), ("path_f1", "#9467bd"),
                 ("token_acc", "#7f7f7f")]:
        if k in ps[0]:
            mh = [val(p[k]) for p in ps]
            m = [x[0] for x in mh]; h = [x[1] for x in mh]
            ax.plot(steps, m, "-o", ms=3, label=k, color=c)
            ax.fill_between(steps, [a - b for a, b in zip(m, h)], [a + b for a, b in zip(m, h)],
                            color=c, alpha=0.12)
    ax.set_xlabel("ACT step"); ax.set_ylabel("metric"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Maze — solution dynamics across ACT steps (N={agg.get('num_puzzles')})")
    ax.legend(fontsize=8)
    save(fig, "fig4_step_dynamics")


def _probe_matrix(probes, stream, targets):
    idx = {(r["stream"], r["target"], r["step"]): r.get("score_mean")
           for r in probes if r.get("status") == "ok"}
    steps = sorted({r["step"] for r in probes})
    M = np.full((len(targets), len(steps)), np.nan)
    for i, t in enumerate(targets):
        for j, s in enumerate(steps):
            v = idx.get((stream, t, s))
            if v is not None:
                M[i, j] = v
    return M, steps


def fig5_probes():
    d = load("results/maze/hardened/linear_probes/probe_results.json")
    if not d:
        return
    local = d.get("local_probes", [])
    glob = d.get("global_probes", [])
    local_bin = [t for t in dict.fromkeys(r["target"] for r in local if r.get("task") == "binary")]
    glob_bin = [t for t in dict.fromkeys(r["target"] for r in glob if r.get("task") == "binary")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, stream in enumerate(["z_H", "z_L"]):
        for row, (probes, targets, label) in enumerate(
                [(local, local_bin, "local (per-cell)"), (glob, glob_bin, "global (per-puzzle)")]):
            ax = axes[row][col]
            M, steps = _probe_matrix(probes, stream, targets)
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
            ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps)
            ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets, fontsize=8)
            ax.set_title(f"{stream} — {label} probe acc"); ax.set_xlabel("ACT step")
            ax.grid(False)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                color="w" if M[i, j] < 0.8 else "k", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Maze — linear probe decodability (test accuracy, 5-seed mean)")
    save(fig, "fig5_probe_decodability")


def fig6_mlp():
    d = load("results/maze/hardened/mlp_probes/probe_results.json")
    if not d:
        return
    rows = [r for r in d.get("local_probes", []) if r.get("task") == "binary"
            and r.get("stream") == "z_H" and r.get("linear_score_mean") is not None]
    if not rows:
        return
    last = max(r["step"] for r in rows)
    rows = [r for r in rows if r["step"] == last]
    rows.sort(key=lambda r: -(r.get("score_mean") or 0))
    feats = [r["target"] for r in rows]
    lin = [r.get("linear_score_mean") for r in rows]
    mlp = [r.get("score_mean") for r in rows]
    x = np.arange(len(feats)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(feats) * 1.1), 4.4))
    ax.bar(x - w/2, lin, w, label="linear", color=H_COLOR)
    ax.bar(x + w/2, mlp, w, label="MLP", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Maze z_H — linear vs MLP probe (step {last})"); ax.legend()
    save(fig, "fig6_probe_mlp_vs_linear")


def fig7_sae():
    p = "results/maze/sae_study/sweep_results.csv"
    if not os.path.exists(p):
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    dicts = sorted({int(r["dict_size"]) for r in rows})
    l1s = sorted({float(r["l1_coeff"]) for r in rows})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(dicts)))
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for r in rows:
        di = dicts.index(int(r["dict_size"])); li = l1s.index(float(r["l1_coeff"]))
        ax.scatter(float(r["L0"]), float(r["final_recon_loss"]), color=cmap[di],
                   marker=markers[li % len(markers)], s=70, edgecolor="k", lw=0.4)
    ax.set_xlabel("L0 (avg active features ↓ sparser)")
    ax.set_ylabel("reconstruction loss ↓ better")
    ax.set_title("Maze z_H SAE — reconstruction vs sparsity frontier")
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[i], markersize=9,
                  markeredgecolor="k", label=f"d={d}") for i, d in enumerate(dicts)]
    leg += [Line2D([0], [0], marker=markers[i % len(markers)], color="w", markerfacecolor="gray",
                   markersize=9, markeredgecolor="k", label=f"λ={l}") for i, l in enumerate(l1s)]
    ax.legend(handles=leg, fontsize=8, ncol=2)
    save(fig, "fig7_sae_frontier")


def fig8_trajectory():
    npz = "results/maze/hardened/trajectory_pca/trajectory_pca.npz"
    js = load("results/maze/hardened/trajectory_pca/trajectory_pca.json")
    if not os.path.exists(npz):
        return
    z = np.load(npz)
    traj = z["trajectories_2d"]            # [N, steps, 2]
    evr = z["explained_variance_ratio"]
    psm = z["per_step_metrics"]            # [steps, 4]: exact, valid_sg_path, connects, path_f1
    N, steps, _ = traj.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    pts = traj.reshape(-1, 2)
    cols = np.tile(np.arange(steps), N)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=cols, cmap="plasma", s=5, alpha=0.35)
    mean_traj = traj.mean(axis=0)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], "-o", color="k", ms=4, lw=1.5, label="mean trajectory")
    for s in range(steps - 1):
        ax.annotate("", xy=mean_traj[s + 1], xytext=mean_traj[s],
                    arrowprops=dict(arrowstyle="->", color="k", alpha=0.6))
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")
    ax.set_title("Maze z_H trajectory in PCA space (colored by ACT step)")
    fig.colorbar(sc, ax=ax, label="ACT step"); ax.legend(fontsize=8)
    ax2 = axes[1]
    labels = ["exact_solved", "valid_sg_path", "connects_start_goal", "path_f1"]
    for i, lab in enumerate(labels):
        ax2.plot(range(steps), psm[:, i], "-o", ms=3, label=lab)
    ax2.set_xlabel("ACT step"); ax2.set_ylabel("metric"); ax2.set_ylim(0, 1.02)
    ax2.set_title("Path-validity along the trajectory"); ax2.legend(fontsize=8)
    pc1 = js.get("pc1_explained") if js else evr[0]
    fig.suptitle(f"Maze z_H trajectory — PC1={pc1*100:.0f}%, 2-PC={ (js.get('cumulative_2pc_explained') if js else evr[:2].sum())*100:.0f}%")
    save(fig, "fig8_trajectory_pca")


def fig9_geometry():
    d = load("results/maze/hardened/probe_geometry/geometric_analysis.json")
    if not d:
        return
    cos = d.get("constraint_cosines_ensemble", [])
    pca = d.get("pca_pc1_ensemble", [])
    if not cos:
        return
    # pick the z_H group with the most steps' final entry; build a feature×feature matrix
    groups = sorted({c["group"] for c in cos})
    zh_groups = [g for g in groups if g.startswith("z_H")]
    grp = (zh_groups[-1] if zh_groups else groups[-1])
    feats = sorted({c["probe_a"] for c in cos if c["group"] == grp} |
                   {c["probe_b"] for c in cos if c["group"] == grp})
    M = np.eye(len(feats))
    for c in cos:
        if c["group"] != grp:
            continue
        i, j = feats.index(c["probe_a"]), feats.index(c["probe_b"])
        M[i, j] = M[j, i] = c["cosine_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    ax.set_title(f"Feature-direction cosine matrix ({grp})"); ax.grid(False)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="k" if abs(M[i, j]) < 0.6 else "w")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax2 = axes[1]
    rows = sorted([(p["group"], p.get("pc1_explained_mean"), p.get("pc1_explained_ci_lower"),
                    p.get("pc1_explained_ci_upper")) for p in pca])
    if rows:
        y = np.arange(len(rows))
        means = [r[1] * 100 for r in rows]
        err = [[(r[1] - (r[2] or r[1])) * 100 for r in rows], [((r[3] or r[1]) - r[1]) * 100 for r in rows]]
        ax2.barh(y, means, xerr=err, color="#9467bd", alpha=0.8)
        ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax2.set_xlabel("PC1 explained variance (%)")
        ax2.set_title("Direction subspace PC1 by (stream, step)")
    fig.suptitle("Maze probe-weight geometry")
    save(fig, "fig9_probe_geometry")


def main():
    for fn in [fig1_ablation, fig2_freeze, fig3_timeshift, fig4_stepdyn, fig5_probes,
               fig6_mlp, fig7_sae, fig8_trajectory, fig9_geometry]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  [skip] {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\nAll maze figures in {OUT}/")


if __name__ == "__main__":
    main()
