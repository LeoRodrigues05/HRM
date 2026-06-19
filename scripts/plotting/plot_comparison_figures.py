#!/usr/bin/env python3
"""Side-by-side Sudoku vs Maze comparison figures + the constraint-localization result.

Produces (in results/reports/comparison_figures/):
  cmp1_ablation_step      per-step z_H/z_L ablation: Sudoku (token) vs Maze (valid_sg_path)
  cmp2_freeze             freeze-from-k: Sudoku vs Maze
  cmp3_sae_frontier       SAE reconstruction vs L0: Sudoku vs Maze (sparsity contrast)
  cmp4_mlp_linearity      mean Δ(MLP-linear) binary probes: Sudoku vs Maze
  cmp5_constraint_localization   unit vs random flip (the localized causal signal)

Pure CPU/JSON/CSV. Re-run anytime.
"""
from __future__ import annotations
import os, sys, json, csv
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/reports/comparison_figures"
os.makedirs(OUT, exist_ok=True)
H, L = "#1f77b4", "#ff7f0e"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def load_csv(p):
    try:
        return list(csv.DictReader(open(p)))
    except Exception:
        return None


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def mean(node):
    if isinstance(node, dict):
        return node.get("mean")
    return num(node)


def lohi(node):
    if isinstance(node, dict):
        return node.get("ci_lower"), node.get("ci_upper")
    return None, None


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png")


def ablation_series(path, metric):
    """Return {stream: (steps, mean%, lo%, hi%)} from an ablation aggregate."""
    out = {}
    for stream, sub in [("z_H", path.replace("STREAM", "zH")), ("z_L", path.replace("STREAM", "zL"))]:
        d = load(sub)
        if not d:
            continue
        ps = d.get("per_step_ablation")
        items = ps.items() if isinstance(ps, dict) else list(enumerate(ps or []))
        rows = []
        for k, v in items:
            node = v.get("delta_accuracy") if metric == "token" else (v.get("maze_metric_deltas") or {}).get(metric)
            m = mean(node); lo, hi = lohi(node)
            if m is not None:
                rows.append((int(k), m * 100, (lo if lo is not None else m) * 100,
                             (hi if hi is not None else m) * 100))
        rows.sort()
        if rows:
            a = np.array(rows)
            out[stream] = (a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    return out


def cmp1_ablation():
    sud = ablation_series("results/controlled/ablation/STREAM/aggregate.json", "token")
    maze = ablation_series("results/maze/hardened/ablation_controlled/STREAM/aggregate.json", "valid_sg_path")
    if not sud and not maze:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    for ax, data, title, ylab in [
        (axes[0], sud, "Sudoku — single-step ablation (token/exact Δ)", "Δ accuracy (%)"),
        (axes[1], maze, "Maze — single-step ablation (valid path Δ)", "Δ valid start→goal path (%)")]:
        for stream, c in [("z_H", H), ("z_L", L)]:
            if stream in data:
                s, m, lo, hi = data[stream]
                ax.plot(s, m, "-o", color=c, ms=4, label=stream)
                if not np.allclose(lo, m):
                    ax.fill_between(s, lo, hi, color=c, alpha=0.18)
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("ACT step ablated")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Where does z_H matter? Sudoku = distributed/accumulating · Maze = concentrated at readout")
    save(fig, "cmp1_ablation_step")


def freeze_series(path, metric):
    d = load(path)
    if not d:
        return {}
    out = {}
    for key, stream in [("freeze_H", "z_H"), ("freeze_L", "z_L")]:
        node = d.get(key, {})
        items = node.items() if isinstance(node, dict) else enumerate(node)
        rows = []
        for k, v in items:
            if metric == "token":
                nd = v.get("delta_accuracy")
            else:
                nd = (v.get("maze_metric_deltas") or {}).get(metric)
            m = mean(nd); lo, hi = lohi(nd)
            if m is not None:
                rows.append((int(k), m * 100, (lo if lo is not None else m) * 100,
                             (hi if hi is not None else m) * 100))
        rows.sort()
        if rows:
            out[stream] = np.array(rows)
    return out


def cmp2_freeze():
    sud = freeze_series("results/controlled/freeze/aggregate.json", "token")
    maze = freeze_series("results/maze/hardened/freeze_controlled/aggregate.json", "valid_sg_path")
    if not sud and not maze:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, data, title, ylab in [
        (axes[0], sud, "Sudoku — freeze z_* from step k (token Δ)", "Δ accuracy (%)"),
        (axes[1], maze, "Maze — freeze z_* from step k (valid path Δ)", "Δ valid path (%)")]:
        for stream, c in [("z_H", H), ("z_L", L)]:
            if stream in data:
                a = data[stream]
                ax.plot(a[:, 0], a[:, 1], "-o", color=c, ms=4, label=stream)
                ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=c, alpha=0.18)
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("freeze from step k onward")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Freeze cost: Sudoku expensive early (blocks accumulation) · Maze ~free (route already found)")
    save(fig, "cmp2_freeze")


def cmp3_sae():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, path, title in [
        (axes[0], "results/sae_study/sweep_results.csv", "Sudoku z_H SAE"),
        (axes[1], "results/maze/sae_study/sweep_results.csv", "Maze z_H SAE")]:
        rows = load_csv(path)
        if not rows:
            continue
        dicts = sorted({int(r["dict_size"]) for r in rows})
        l1s = sorted({float(r["l1_coeff"]) for r in rows})
        cmap = plt.cm.viridis(np.linspace(0, 1, len(dicts)))
        markers = ["o", "s", "^", "D", "v"]
        for r in rows:
            di = dicts.index(int(r["dict_size"])); li = l1s.index(float(r["l1_coeff"]))
            ax.scatter(num(r["L0"]), num(r["final_recon_loss"]), color=cmap[di],
                       marker=markers[li % len(markers)], s=70, edgecolor="k", lw=0.4)
        ax.set_xlabel("L0 (avg active features)"); ax.set_ylabel("reconstruction loss")
        ax.set_title(title)
    fig.suptitle("SAE frontier: Maze z_H is far sparser (lower L0) at matched config — fewer active features")
    save(fig, "cmp3_sae_frontier")


def cmp4_mlp():
    def sud_delta():
        rows = load_csv("results/probes/nonlinear_probes/sweep_results.csv")
        if not rows:
            return None
        ds = [num(r["delta_val"]) for r in rows if r.get("task") == "binary" and num(r.get("delta_val")) is not None]
        return ds
    def maze_delta():
        d = load("results/maze/hardened/mlp_probes/probe_results.json")
        if not d:
            return None
        return [r.get("delta_mlp_minus_linear") for r in d.get("local_probes", []) + d.get("global_probes", [])
                if r.get("task") == "binary" and isinstance(r.get("delta_mlp_minus_linear"), (int, float))]
    sd, mz = sud_delta(), maze_delta()
    if not sd and not mz:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data, labels, cols = [], [], []
    for arr, lab, c in [(sd, "Sudoku", H), (mz, "Maze", "#d62728")]:
        if arr:
            data.append(arr); labels.append(f"{lab}\n(mean={np.mean(arr):+.3f})"); cols.append(c)
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ accuracy (MLP − linear), binary probes")
    ax.set_title("Are representations linear? Both ≈0 → linear readout suffices")
    save(fig, "cmp4_mlp_linearity")


def cmp5_ccl():
    rows = load_csv("results/constraint_localization/sweep/sweep_results.csv")
    if not rows:
        print("  [skip] cmp5: no constraint-localization sweep")
        return
    labels = [f"{r['patch_level']}/{r['patch_steps']}" for r in rows]
    unit = [num(r["unit_flip"]) for r in rows]
    rand = [num(r["random_flip"]) for r in rows]
    umr = [num(r["unit_minus_random"]) for r in rows]
    lo = [num(r["unit_minus_random"]) - num(r["umr_ci_lo"]) for r in rows]
    hi = [num(r["umr_ci_hi"]) - num(r["unit_minus_random"]) for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    ax.bar(x - w/2, unit, w, label="patch C's UNIT (20 peers)", color="#2ca02c")
    ax.bar(x + w/2, rand, w, label="patch RANDOM 20 cells", color="#999999")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P(target cell C flips)"); ax.set_title("Unit vs random interchange — flip rate of C")
    ax.legend(fontsize=8)
    ax2 = axes[1]
    colors = ["#2ca02c" if (num(r["umr_ci_lo"]) > 0) else "#cccccc" for r in rows]
    ax2.bar(x, umr, yerr=[lo, hi], color=colors, capsize=3)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("unit − random flip rate [95% CI]")
    ax2.set_title("Localized causal excess (green = CI excludes 0)")
    fig.suptitle("Constraint localization (Sudoku): z_H at C's unit causally controls C more than random cells")
    save(fig, "cmp5_constraint_localization")


def main():
    for fn in [cmp1_ablation, cmp2_freeze, cmp3_sae, cmp4_mlp, cmp5_ccl]:
        try:
            fn()
        except Exception as e:
            import traceback; print(f"  [skip] {fn.__name__}: {e}"); traceback.print_exc()
    print(f"\nComparison figures in {OUT}/")


if __name__ == "__main__":
    main()
