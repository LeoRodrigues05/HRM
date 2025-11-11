#!/usr/bin/env python3
"""
Compute & plot Sudoku solver dynamics:
  1) Step-to-step Hamming distance (mean ± std across puzzles)
  2) Stepwise logit KL (mean across puzzles & cells) if logits exist

Matches your sudoku_report.py conventions:
  - tokens 2..10 -> '1'..'9', else '.'
  - intermediates in either:
      * intermediate_preds: [S, N, 81]
      * intermediate_logits: [S, N, 81, V]
  - 'inputs' are givens (optional mask for Hamming)

Usage:
  python metrics_sudoku_dynamics_with_plots.py \
      --npz Checkpoint_HRM_Sudoku/.../step_0_all_preds.npz \
      --ignore_givens \
      --outdir results/metrics

This will emit:
  results/metrics/hamming.csv
  results/metrics/hamming_mean_std.png
  results/metrics/kl.csv             (if logits exist)
  results/metrics/kl_mean.png        (if logits exist)
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- token mapping identical to your report ----------
def id2num(i: int) -> str:
    if 2 <= i <= 10:
        return str(i - 1)
    return "."

def to_chars(arr_ids: np.ndarray) -> np.ndarray:
    vfunc = np.vectorize(id2num)
    return vfunc(arr_ids.astype(int))

# ---------- loading & normalization ----------
def load_steps(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)

    steps_logits = None
    if "intermediate_preds" in z:
        steps_ids = z["intermediate_preds"]           # [S,N,81]
    elif "intermediate_logits" in z:
        steps_logits = z["intermediate_logits"]       # [S,N,81,V]
        steps_ids = steps_logits.argmax(-1)           # [S,N,81]
    else:
        raise ValueError("NPZ must have 'intermediate_preds' or 'intermediate_logits'.")

    givens_ids = z["inputs"] if "inputs" in z else None
    return steps_ids, steps_logits, givens_ids

# ---------- Hamming ----------
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

def compute_hamming(npz_path: str, ignore_givens: bool):
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

    mean_curve = curves.mean(0)
    std_curve  = curves.std(0)
    return mean_curve, std_curve

# ---------- KL ----------
def safe_log_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    logsum = np.log(np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-12)
    return x - logsum

def stepwise_kl(npz_path: str) -> np.ndarray:
    _, steps_logits, _ = load_steps(npz_path)
    if steps_logits is None:
        raise ValueError("Stepwise KL needs 'intermediate_logits' in the NPZ.")

    S = steps_logits.shape[0]
    kls = []
    for s in range(1, S):
        l_prev = steps_logits[s-1]                  # [N,81,V]
        l_curr = steps_logits[s]                    # [N,81,V]
        lp = safe_log_softmax(l_prev, axis=-1)
        lq = safe_log_softmax(l_curr, axis=-1)
        p  = np.exp(lp)
        kl = (p * (lp - lq)).sum(axis=-1)          # [N,81]
        kls.append(kl.mean())
    return np.asarray(kls, dtype=float)

# ---------- plotting helpers ----------
def save_hamming_plot(mean_curve, std_curve, out_png: Path, title="Hamming Δ per step"):
    x = np.arange(1, len(mean_curve) + 1)
    plt.figure(figsize=(7.5, 4.2))
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

def save_kl_plot(kl_curve, out_png: Path, title="Stepwise KL (mean over puzzles & cells)"):
    x = np.arange(1, len(kl_curve) + 1)
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(x, kl_curve, linewidth=2)
    plt.xlabel("Step (s → s+1)")
    plt.ylabel("KL(p^s || p^{s+1})")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to your step_0_all_preds.npz")
    ap.add_argument("--ignore_givens", action="store_true", help="Exclude givens from Hamming distance")
    ap.add_argument("--outdir", default="results/metrics", help="Directory to write plots & CSVs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Hamming
    mean_ham, std_ham = compute_hamming(args.npz, ignore_givens=args.ignore_givens)
    print("Hamming Δ (fraction of cells changed per step):")
    for s, (m, sd) in enumerate(zip(mean_ham, std_ham), start=1):
        print(f" step {s:>2}: {m:.4f} ± {sd:.4f}")

    # Save CSV + plot
    (outdir / "hamming.csv").write_text(
        "step_index,mean_hamming,std_hamming\n" +
        "\n".join(f"{i+1},{float(m)},{float(sd)}" for i, (m, sd) in enumerate(zip(mean_ham, std_ham))),
        encoding="utf-8"
    )
    save_hamming_plot(mean_ham, std_ham, outdir / "hamming_mean_std.png",
                      title="Hamming Δ per step (mean ± std)")

    # KL (if logits exist)
    try:
        kls = stepwise_kl(args.npz)
        print("\nStepwise KL (mean over all puzzles & cells):")
        for s, v in enumerate(kls, start=1):
            print(f" step {s:>2}: {v:.6f}")

        (outdir / "kl.csv").write_text(
            "step_index,mean_kl\n" +
            "\n".join(f"{i+1},{float(v)}" for i, v in enumerate(kls)),
            encoding="utf-8"
        )
        save_kl_plot(kls, outdir / "kl_mean.png")
    except ValueError as e:
        print("\n[Info] KL skipped:", e)

    print(f"\nWrote outputs to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
