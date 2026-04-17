#!/usr/bin/env python3
# hamming_multi_puzzle_plots.py
# Compute and plot Hamming (step-to-step fraction of changed cells) across MANY puzzles.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Token mapping identical to your report ---
def id2num(i: int) -> str:
    if 2 <= i <= 10:
        return str(i - 1)
    return "."

def to_chars(arr_ids: np.ndarray) -> np.ndarray:
    vfunc = np.vectorize(id2num)
    return vfunc(arr_ids.astype(int))

# --- Load intermediates in either form ---
def load_steps(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)

    steps_logits = None
    if "intermediate_preds" in z:
        steps_ids = z["intermediate_preds"]              # [S,N,81]
    elif "intermediate_logits" in z:
        steps_logits = z["intermediate_logits"]          # [S,N,81,V]
        steps_ids = steps_logits.argmax(-1)              # [S,N,81]
    else:
        raise ValueError("NPZ must contain 'intermediate_preds' or 'intermediate_logits'.")

    givens_ids = z["inputs"] if "inputs" in z else None
    return steps_ids, steps_logits, givens_ids

# --- Hamming computation ---
def build_mask_from_givens(givens_chars: np.ndarray, ignore_givens: bool) -> np.ndarray:
    mask = np.ones(81, dtype=bool)
    if ignore_givens and givens_chars is not None:
        mask &= (givens_chars == ".")
    return mask

def hamming_curve_for_puzzle(steps_chars: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    S = steps_chars.shape[0]
    out = []
    for s in range(1, S):
        a, b = steps_chars[s-1], steps_chars[s]
        if mask is None:
            denom = 81
            out.append(float(np.sum(a != b)) / denom)
        else:
            denom = int(mask.sum())
            out.append(0.0 if denom == 0 else float(np.sum((a != b) & mask)) / denom)
    return np.asarray(out, dtype=float)

def compute_all_hamming(npz_path: str, ignore_givens: bool):
    steps_ids, _, givens_ids = load_steps(npz_path)
    S, N, _ = steps_ids.shape

    steps_chars  = to_chars(steps_ids)                            # [S,N,81]
    givens_chars = to_chars(givens_ids) if givens_ids is not None else None

    curves = []
    for i in range(N):
        mask = None
        if ignore_givens and givens_chars is not None:
            mask = build_mask_from_givens(givens_chars[i], ignore_givens=True)
        curves.append(hamming_curve_for_puzzle(steps_chars[:, i, :], mask))
    curves = np.stack(curves, axis=0)                             # [N, S-1]
    return curves  # per-puzzle curves

# --- Plotting ---
def plot_mean_std(curves: np.ndarray, out_png: Path, title: str):
    mean_curve = curves.mean(0)
    std_curve  = curves.std(0)
    x = np.arange(1, len(mean_curve) + 1)

    plt.figure(figsize=(7.5,4.2))
    plt.plot(x, mean_curve, label="mean Δ", linewidth=2)
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.25, label="±1σ")
    plt.xlabel("Step (s → s+1)")
    plt.ylabel("Fraction of cells changed")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_overlay(curves: np.ndarray, out_png: Path, max_curves: int = 100, title: str = "Per-puzzle Hamming (overlay)"):
    N = curves.shape[0]
    idx = np.arange(N) if N <= max_curves else np.random.choice(N, size=max_curves, replace=False)
    x = np.arange(1, curves.shape[1] + 1)

    plt.figure(figsize=(7.5,4.2))
    for i in idx:
        plt.plot(x, curves[i], alpha=0.2, linewidth=1)
    plt.xlabel("Step (s → s+1)")
    plt.ylabel("Fraction of cells changed")
    plt.title(f"{title}  (n={len(idx)})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_small_multiples(curves: np.ndarray, out_png: Path, rows: int = 4, cols: int = 6, seed: int = 0, title: str = "Per-puzzle Hamming (samples)"):
    random.seed(seed)
    N = curves.shape[0]
    k = min(N, rows*cols)
    idx = random.sample(range(N), k)
    x = np.arange(1, curves.shape[1] + 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.0), sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    for ax, i in zip(axes.flat, idx):
        ax.plot(x, curves[i], linewidth=1.8)
        ax.grid(alpha=0.25)
        ax.set_title(f"Puzzle {i}", fontsize=9)
    # hide any unused subplots
    for ax in axes.flat[k:]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    for ax in axes[:,0]:
        ax.set_ylabel("Δ fraction")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to NPZ with intermediates")
    ap.add_argument("--ignore_givens", action="store_true", help="Exclude givens from Hamming")
    ap.add_argument("--outdir", default="results/hamming_multi", help="Output directory")
    ap.add_argument("--grid_rows", type=int, default=4)
    ap.add_argument("--grid_cols", type=int, default=6)
    ap.add_argument("--overlay_max", type=int, default=100, help="Max curves to overlay")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    curves = compute_all_hamming(args.npz, ignore_givens=args.ignore_givens)  # [N,S-1]
    N, L = curves.shape
    print(f"Loaded {N} puzzles, {L} step transitions per puzzle.")

    # Save raw CSV (per-puzzle)
    csv_path = outdir / "hamming_per_puzzle.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = "puzzle_index," + ",".join([f"step_{s}" for s in range(1, L+1)]) + "\n"
        f.write(header)
        for i in range(N):
            f.write(str(i) + "," + ",".join(str(float(v)) for v in curves[i]) + "\n")
    print(f"Wrote {csv_path}")

    # Aggregate mean ± std
    plot_mean_std(curves, outdir / "hamming_mean_std.png",
                  title="Hamming Δ per step (mean ± std across puzzles)")

    # Overlay of many curves (alpha)
    plot_overlay(curves, outdir / "hamming_overlay.png",
                 max_curves=args.overlay_max,
                 title="Per-puzzle Hamming overlay")

    # Small multiples grid (sample)
    plot_small_multiples(curves, outdir / "hamming_small_multiples.png",
                         rows=args.grid_rows, cols=args.grid_cols,
                         title="Per-puzzle Hamming (samples)")

    print(f"Plots saved in {outdir.resolve()}")

if __name__ == "__main__":
    main()
