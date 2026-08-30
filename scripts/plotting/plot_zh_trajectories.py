#!/usr/bin/env python3
"""Generate z_H trajectory plots from pre-collected data.

Reads zh_trajectories.pt and zh_trajectory_metrics.json (already on disk)
and produces publication-quality figures for the paper.

Outputs → results/Final/figures/:
  - zh_pca_trajectories.pdf    (3-panel: solved, failed, overlay)
  - zh_directional_metrics.pdf (2×2: cosine, norms, alignment, delta)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from paper_style import apply_paper_style, scaled_rc, COL_W, TEXT_W

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "Sudoku" / "plots" / "zh_trajectories"
OUT = ROOT / "results" / "Final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style (single source of truth for all paper figures) ─────────────
apply_paper_style()


def load_data():
    data = torch.load(DATA_DIR / "zh_trajectories.pt", map_location="cpu",
                      weights_only=False)
    with open(DATA_DIR / "zh_trajectory_metrics.json") as f:
        metrics = json.load(f)
    return data, metrics


# ======================================================================
# Figure A: PCA Trajectory Plots (3-panel)
# ======================================================================
def plot_pca_trajectories(data):
    # Included at \columnwidth in 09appendix.tex while authored at TEXT_W, so
    # LaTeX shrinks it to 0.47x — the reason its text was unreadable. scaled_rc
    # compensates; the figure keeps its dimensions so the paper does not move.
    with plt.rc_context(scaled_rc(TEXT_W, COL_W)):
        _plot_pca_trajectories(data)


def _plot_pca_trajectories(data):
    zh = data["zh_mean"].numpy()               # [N, 16, 512]
    final_puzzle = data["final_puzzle_acc"].numpy()  # [N]
    N, S, D = zh.shape

    solved_mask = final_puzzle == 1.0
    failed_mask = ~solved_mask
    n_solved = solved_mask.sum()
    n_failed = failed_mask.sum()

    # Fit PCA on all trajectories flattened
    zh_flat = zh.reshape(-1, D)                # [N*16, 512]
    pca = PCA(n_components=2)
    coords_flat = pca.fit_transform(zh_flat)   # [N*16, 2]
    coords = coords_flat.reshape(N, S, 2)      # [N, 16, 2]

    var_explained = pca.explained_variance_ratio_

    # Full-width (figure*) layout: 3 panels side by side, shared legend below
    # and an ACT-step colorbar on the right — matches the paper figure.
    # All three panels show the SAME 2-D PCA projection, so they share x/y
    # limits: the panels become directly comparable, and dropping the inner
    # y-tick labels lets the three sit close together.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 4.05),
                             sharex=True, sharey=True)
    cmap = plt.cm.viridis

    # ── Panel A: Solved ──
    ax = axes[0]
    idxs = np.where(solved_mask)[0]
    show = idxs[:25]  # up to 25 trajectories
    for i in show:
        traj = coords[i]
        for t in range(S - 1):
            ax.plot(traj[t:t+2, 0], traj[t:t+2, 1],
                    color=cmap(t / (S - 1)), alpha=0.5, lw=0.6)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="#2ecc71", ms=2.5, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color="#e74c3c", ms=3, zorder=5,
                mew=0.8)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("(a) Solved")

    # ── Panel B: Failed ──
    ax = axes[1]
    idxs = np.where(failed_mask)[0]
    show = idxs[:25]
    for i in show:
        traj = coords[i]
        for t in range(S - 1):
            ax.plot(traj[t:t+2, 0], traj[t:t+2, 1],
                    color=cmap(t / (S - 1)), alpha=0.5, lw=0.6)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="#2ecc71", ms=2.5, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color="#e74c3c", ms=3, zorder=5,
                mew=0.8)
    ax.set_title("(b) Failed")

    # ── Panel C: Overlay ──
    ax = axes[2]
    s_idxs = np.where(solved_mask)[0][:12]
    f_idxs = np.where(failed_mask)[0][:12]
    for i in s_idxs:
        traj = coords[i]
        ax.plot(traj[:, 0], traj[:, 1], color="#2980b9", alpha=0.4, lw=0.7)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color="#2980b9", ms=3, mew=0.8,
                alpha=0.6)
    for i in f_idxs:
        traj = coords[i]
        ax.plot(traj[:, 0], traj[:, 1], color="#e67e22", alpha=0.4, lw=0.7)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color="#e67e22", ms=3, mew=0.8,
                alpha=0.6)
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color="#2980b9", lw=1.5),
                      Line2D([0], [0], color="#e67e22", lw=1.5)]
    ax.set_title("(c) Overlay")

    # Reserve room: bottom for the shared legend, right for the colorbar. The
    # three panels share the same PC1/PC2 scale and only panel (a) carries a
    # y-label, so they can sit close together (small wspace).
    bottom, top = 0.34, 0.92
    fig.subplots_adjust(left=0.085, right=0.885, bottom=bottom, top=top,
                        wspace=0.10)
    cax = fig.add_axes([0.905, bottom, 0.016, top - bottom])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, S - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("ACT step")
    cbar.set_ticks([0, 5, 10, 15])

    # The common PC1 label and the population legend sit on their own rows below
    # the panels. Their heights are measured rather than guessed: at this
    # figure's scale compensation the tick labels are ~18pt, so a hardcoded y
    # lands on top of them.
    def _fig_bbox(artist):
        return artist.get_window_extent(
            fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())

    fig.canvas.draw()
    gap = 0.025
    ticks_y0 = min(_fig_bbox(t).y0 for ax in axes
                   for t in ax.get_xticklabels() if t.get_text())
    xlab = fig.supxlabel(f"PC1 ({var_explained[0]*100:.1f}%)", x=0.47,
                         y=ticks_y0 - gap, va="top",
                         fontsize=plt.rcParams["axes.labelsize"])
    fig.canvas.draw()
    fig.legend(legend_handles, ["Solved", "Failed"],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.47, _fig_bbox(xlab).y0 - gap),
               handlelength=1.8, handletextpad=0.5, columnspacing=1.6)

    fig.savefig(OUT / "zh_pca_trajectories.pdf")
    fig.savefig(OUT / "zh_pca_trajectories.png", dpi=300)
    plt.close(fig)
    print("[Trajectories] Saved zh_pca_trajectories.pdf")


# ======================================================================
# Figure B: Directional Metrics (2×2)
# ======================================================================
def plot_directional_metrics(metrics):
    steps = np.arange(16)

    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.0))

    def _plot_metric(ax, key, ylabel, title, x_offset=0):
        m = metrics[key]
        x = steps[x_offset:x_offset + len(m["all_mean"])]
        all_m = np.array(m["all_mean"])
        all_s = np.array(m["all_std"])
        suc_m = np.array(m["success_mean"])
        fail_m = np.array(m["failure_mean"])

        ax.plot(x, suc_m, "o-", color="#2980b9", ms=2.5, markevery=3, lw=1.2, label="Solved")
        ax.plot(x, fail_m, "s-", color="#e67e22", ms=2.5, markevery=3, lw=1.2, label="Failed")
        ax.fill_between(x, all_m - all_s, all_m + all_s,
                        alpha=0.1, color="gray")
        ax.set_xlabel("ACT Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([0, 4, 8, 12, 15])

    # (a) Consecutive cosine similarity
    _plot_metric(axes[0, 0], "cos_consecutive",
                 "Cosine sim.", "(a) Consecutive step similarity",
                 x_offset=1)

    # (b) Norm evolution
    _plot_metric(axes[0, 1], "norms",
                 "$\\|z_H\\|$", "(b) $z_H$ norm evolution")

    # (c) Alignment with final state
    _plot_metric(axes[1, 0], "cos_final_alignment",
                 "Cosine sim. to final", "(c) Alignment with final $z_H$")

    # (d) Update magnitude
    _plot_metric(axes[1, 1], "delta_norms",
                 "$\\|\\Delta z_H\\|$", "(d) Update magnitude",
                 x_offset=1)

    plt.tight_layout()
    fig.savefig(OUT / "zh_directional_metrics.pdf")
    fig.savefig(OUT / "zh_directional_metrics.png", dpi=300)
    plt.close(fig)
    print("[Metrics] Saved zh_directional_metrics.pdf")


# ======================================================================
if __name__ == "__main__":
    print(f"Loading data from: {DATA_DIR}")
    print(f"Output directory:  {OUT}")
    data, metrics = load_data()
    print(f"  {data['n_puzzles']} puzzles, {data['max_steps']} steps")
    n_solved = int(data["final_puzzle_acc"].sum())
    print(f"  {n_solved} solved, {data['n_puzzles'] - n_solved} failed")

    plot_pca_trajectories(data)
    plot_directional_metrics(metrics)
    print("\nDone!")
