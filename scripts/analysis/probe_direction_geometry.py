"""Probe-weight direction geometry for the paper's app:geometry appendix.

Recomputes every number quoted in "Interpretable Geometry of Probe Weight
Vectors" (manuscript appendix A.9) from the cached probe weights:

  * pairwise cosine similarity between unit-normalized probe directions,
  * PCA of the centered direction set (PC1, PC1--2, dims for 90% variance),
  * a seed-reliability floor per target: the mean |cos| between the weight
    vectors the same probe recovers under different training seeds,
  * random-direction nulls for both cosine and PC1 at d = 512.

Directions are unit-normalized before PCA. The cached geometric_analysis.json
files do not normalize, which inflates Maze PC1 from ~25% to ~50% because the
wall/free/path probes carry 3-5x the weight norm of the rest.

Usage:  python scripts/analysis/probe_direction_geometry.py
"""

import itertools
import numpy as np
import torch
from scipy import stats

SUDOKU_W = "results/Sudoku/probes/e8_constraint_probes/probe_weights.pt"
MAZE_W = "results/Maze/hardened/probe_geometry/probe_weights.pt"
ARC_W = "results/ARC/arc/hardened/linear_probes/probe_weights.pt"

SUDOKU_TARGETS = ["per_cell_correct", "violated_in_row", "violated_in_col", "violated_in_box"]
SUDOKU_VIOL = ["violated_in_row", "violated_in_col", "violated_in_box"]
SUDOKU_STEPS = [0, 4, 8, 12, 15]

MAZE_TARGETS = ["is_wall", "is_free", "on_optimal_path", "off_path_passable",
                "is_corner_on_path", "is_junction", "near_goal_5", "near_start_5", "is_dead_end"]
MAZE_IDENTIFIED = ["is_wall", "is_free", "on_optimal_path", "off_path_passable"]

# output_inside_grid is an exact duplicate of input_inside_grid on this dataset.
ARC_TARGETS = ["per_cell_correct", "input_inside_grid", "is_eos",
               "colour_changed", "same_as_input", "is_object_boundary"]
ARC_PAIRS = [("colour_changed", "same_as_input"),
             ("per_cell_correct", "same_as_input"),
             ("per_cell_correct", "colour_changed"),
             ("colour_changed", "is_object_boundary"),
             ("per_cell_correct", "input_inside_grid"),
             ("input_inside_grid", "is_eos")]


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    m, sd, n = xs.mean(), xs.std(ddof=1), len(xs)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, h


def explained_variance(dirs):
    """Explained-variance ratios of the centered, unit-normalized direction set."""
    X = np.stack([unit(d) for d in dirs])
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return s / s.sum()


def reliability(per_seed_weights):
    """Mean |cos| between the same probe's directions across seeds."""
    W = [unit(w) for w in per_seed_weights]
    return float(np.mean([abs(W[a] @ W[b]) for a, b in itertools.combinations(range(len(W)), 2)]))


def random_nulls(d=512, n_cos=200_000, n_pca=2000, ks=(4, 6, 9), seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_cos, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.standard_normal((n_cos, d)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.abs((a * b).sum(1))
    print(f"random |cos| (d={d}): mean {c.mean():.4f}  95th pct {np.percentile(c, 95):.4f}")
    for k in ks:
        pc1 = [explained_variance(rng.standard_normal((k, d)))[0] for _ in range(n_pca)]
        print(f"random PC1 (k={k}): {np.mean(pc1) * 100:.1f}%")


def sudoku(path=SUDOKU_W):
    P = torch.load(path, map_location="cpu", weights_only=False)
    print("\n=== Sudoku (Table: geometry_sudoku) ===")
    print(f"{'state':>5} {'step':>4} {'V-V':>16} {'C-V':>16} {'PC1 %':>14} {'Rel.':>6}")
    for stream in ["z_H", "z_L"]:
        for step in SUDOKU_STEPS:
            vv, cv, pc1 = [], [], []
            n_seeds = len(P[f"{stream}_step{step}_{SUDOKU_TARGETS[0]}"]["W_per_seed"])
            for s in range(n_seeds):
                D = {t: unit(P[f"{stream}_step{step}_{t}"]["W_per_seed"][s]) for t in SUDOKU_TARGETS}
                vv.append(np.mean([D[a] @ D[b] for a, b in itertools.combinations(SUDOKU_VIOL, 2)]))
                cv.append(np.mean([D["per_cell_correct"] @ D[b] for b in SUDOKU_VIOL]))
                pc1.append(explained_variance([D[t] for t in SUDOKU_TARGETS])[0])
            rel = np.mean([reliability(P[f"{stream}_step{step}_{t}"]["W_per_seed"]) for t in SUDOKU_TARGETS])
            (m1, h1), (m2, h2), (m3, h3) = mean_ci(vv), mean_ci(cv), mean_ci(pc1)
            print(f"{stream:>5} {step:>4} {m1:>+9.3f}+-{h1:.3f} {m2:>+9.3f}+-{h2:.3f} "
                  f"{m3 * 100:>8.1f}+-{h3 * 100:.1f} {rel:>6.2f}")

    print("\nis_given specificity (z_H step15):")
    D = {t: unit(P[f"z_H_step15_{t}"]["W"]) for t in SUDOKU_TARGETS + ["is_given"]}
    for t in SUDOKU_TARGETS:
        print(f"  is_given vs {t:>18}: {D['is_given'] @ D[t]:+.3f}")


def maze(path=MAZE_W, step=15):
    P = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== Maze (Table: geometry_maze), step {step} ===")
    print("Panel A: accuracy and direction reliability")
    print(f"{'target':>20} {'zH acc':>7} {'zH rel':>7} {'zL acc':>7} {'zL rel':>7} {'zH |W|':>7}")
    for t in MAZE_TARGETS:
        h, l = P[f"z_H_step{step}_{t}"], P[f"z_L_step{step}_{t}"]
        print(f"{t:>20} {h['val_acc_mean'] * 100:>7.1f} {reliability(h['W_per_seed']):>7.2f} "
              f"{l['val_acc_mean'] * 100:>7.1f} {reliability(l['W_per_seed']):>7.2f} {h['W_norm']:>7.2f}")

    print("\nPanel B: geometry of the identified directions")
    for stream in ["z_H", "z_L"]:
        n_seeds = len(P[f"{stream}_step{step}_{MAZE_TARGETS[0]}"]["W_per_seed"])
        row = []
        for a, b in itertools.combinations(MAZE_IDENTIFIED, 2):
            per = [unit(P[f"{stream}_step{step}_{a}"]["W_per_seed"][s]) @
                   unit(P[f"{stream}_step{step}_{b}"]["W_per_seed"][s]) for s in range(n_seeds)]
            m, h = mean_ci(per)
            row.append(f"{a[:9]}-{b[:9]} {m:+.3f}+-{h:.3f}")
        for subset, tag in [(MAZE_IDENTIFIED, "4 dir."), (MAZE_TARGETS, "9 dir.")]:
            per = [explained_variance([P[f"{stream}_step{step}_{t}"]["W_per_seed"][s] for t in subset])[0]
                   for s in range(n_seeds)]
            m, h = mean_ci(per)
            row.append(f"PC1 {tag} {m * 100:.1f}+-{h * 100:.1f}")
        print(f"  {stream}: " + " | ".join(row))


def arc(path=ARC_W):
    P = torch.load(path, map_location="cpu", weights_only=False)
    print("\n=== ARC-AGI-2 (Table: geometry_arc) ===")
    print("(seed-averaged weights only; no reliability floor available)")
    dup = [unit(P[f"z_H_step{s}_input_inside_grid"]["W_mean"]) @
           unit(P[f"z_H_step{s}_output_inside_grid"]["W_mean"]) for s in (0, 15)]
    print(f"input_inside_grid vs output_inside_grid: {dup[0]:.6f}, {dup[1]:.6f} (duplicate; excluded)")
    header = f"{'pair':>22}" + "".join(f"{g + ' s' + str(s):>10}" for g in ("z_H", "z_L") for s in (0, 15))
    print(header)
    cols = [(g, s) for g in ("z_H", "z_L") for s in (0, 15)]
    D = {(g, s): {t: unit(P[f"{g}_step{s}_{t}"]["W_mean"]) for t in ARC_TARGETS} for g, s in cols}
    for a, b in ARC_PAIRS:
        print(f"{a[:10] + '-' + b[:10]:>22}" + "".join(f"{D[c][a] @ D[c][b]:>+10.3f}" for c in cols))
    for label, idx in [("PC1 (%)", 0), ("PC1-2 (%)", 1)]:
        vals = []
        for c in cols:
            ev = np.cumsum(explained_variance([D[c][t] for t in ARC_TARGETS]))
            vals.append(ev[idx] * 100)
        print(f"{label:>22}" + "".join(f"{v:>10.1f}" for v in vals))
    for c in cols:
        ev = np.cumsum(explained_variance([D[c][t] for t in ARC_TARGETS]))
        print(f"  {c[0]} step{c[1]}: dims for 90% variance = {int(np.searchsorted(ev, 0.90) + 1)}")


if __name__ == "__main__":
    random_nulls()
    sudoku()
    maze()
    arc()
