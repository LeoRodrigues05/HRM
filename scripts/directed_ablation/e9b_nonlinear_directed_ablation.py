#!/usr/bin/env python3
"""E9b: Directed Ablation Using Non-Linear (MLP) Probe Subspaces.

Extends E9 by extracting causal directions from *non-linear* MLP probes
(trained in Gap 3) and ablating those subspaces from z_H.

Three direction-extraction strategies:
  (A) Gradient subspace — average ∂probe/∂z_H over many samples, PCA → top-K
  (B) Weight subspace   — SVD of MLP first-layer W1 → top-K singular vectors
  (C) Random subspace   — K random orthogonal directions (control)
  Also includes:
  (D) Linear probe direction — single 1-d direction from E8 (for comparison)

If MLP-derived subspaces show larger causal effects than linear probes AND
larger than random controls, it means non-linear probes capture
computationally relevant features that linear probes miss.  If they also
show ≈0% effect, it further cements "readout ≠ computation."

Output
------
  results/directed_ablation/e9b_nonlinear/
    ablation_results.json        – per-direction, per-puzzle metrics
    aggregate_results.json       – mean effects across puzzles
    specificity_matrix.json      – [direction × violation_type] Δ matrix
    subspace_geometry.json       – cosines between extracted subspaces

Usage
-----
    # Quick test
    python scripts/directed_ablation/e9b_nonlinear_directed_ablation.py --n_puzzles 5 --quick

    # Full run (500 puzzles, GPU)
    python scripts/directed_ablation/e9b_nonlinear_directed_ablation.py --n_puzzles 500 --device cuda

    # Custom subspace dimension
    python scripts/directed_ablation/e9b_nonlinear_directed_ablation.py --n_puzzles 500 --subspace_k 5
"""

import os
import sys
import json
import argparse
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.directed_ablation.e9_directed_ablation import (
    DirectionalAblator,
    load_model_and_data,
    count_violations,
    cell_accuracy,
    count_per_unit_broken,
    SUDOKU_CELLS,
)
from scripts.core.activation_ablation import ActivationCache
from scripts.probes.nonlinear_probes import MLPProbe
from scripts.probes.e8_constraint_probes import (
    collect_activations,
    BINARY_TARGETS,
    MULTICLASS_TARGETS,
    ALL_TARGETS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

CONSTRAINT_TARGETS = ["violated_in_row", "violated_in_col", "violated_in_box"]
BEST_STEP = "15"  # Use step-15 probes (highest accuracy from Gap 3 results)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy A: Gradient-based subspace extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_gradient_subspace(
    mlp_model: MLPProbe,
    X_samples: torch.Tensor,       # [N, D] — z_H activations
    K: int = 3,
    max_samples: int = 5000,
) -> torch.Tensor:
    """Extract top-K gradient directions from the MLP probe.

    For each sample x_i, compute g_i = ∂f(x_i)/∂x_i where f is the probe
    output logit. Stack gradients into G [N, D], then PCA → top-K directions.

    Returns: [K, D] orthonormal basis for the gradient subspace.
    """
    mlp_model.eval()
    device = X_samples.device
    N = min(X_samples.shape[0], max_samples)
    X = X_samples[:N].clone().detach().requires_grad_(True)

    # Compute gradients in batches for memory efficiency
    batch_size = 512
    grads = []
    for i in range(0, N, batch_size):
        x_batch = X[i:i+batch_size]
        x_batch = x_batch.clone().detach().requires_grad_(True)
        out = mlp_model(x_batch)

        # For binary: out is [B, 1]; for multiclass: [B, C] — use sum of logits
        if out.shape[-1] == 1:
            scalar = out.sum()
        else:
            scalar = out.sum()

        scalar.backward()
        grads.append(x_batch.grad.detach().clone())

    G = torch.cat(grads, dim=0)  # [N, D]

    # Center gradients
    G = G - G.mean(dim=0, keepdim=True)

    # PCA via SVD
    U, S, Vt = torch.linalg.svd(G.float(), full_matrices=False)
    # Vt[i] is the i-th principal direction
    top_K = Vt[:K]  # [K, D]

    # Report variance explained
    total_var = (S ** 2).sum()
    explained = (S[:K] ** 2).sum() / total_var
    logger.info(f"    Gradient subspace: top-{K} PCs explain {explained:.1%} of gradient variance "
                f"(singular values: {S[:K].tolist()[:5]})")

    return top_K


# ═══════════════════════════════════════════════════════════════════════════
# Strategy B: Weight-based subspace extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_weight_subspace(
    mlp_model: MLPProbe,
    K: int = 3,
) -> torch.Tensor:
    """Extract top-K input directions from the MLP's first layer weights.

    W1 is [hidden_dim, D]. SVD(W1) → top-K right singular vectors are
    the directions in input space that W1 "reads from" most strongly.

    Returns: [K, D] orthonormal basis for the weight subspace.
    """
    # Get first linear layer weights
    W1 = None
    for name, param in mlp_model.named_parameters():
        if "0.weight" in name or (name.endswith(".weight") and W1 is None):
            W1 = param.detach().float()
            break

    if W1 is None:
        raise ValueError("Could not find first-layer weights in MLP probe")

    # SVD: W1 = U @ diag(S) @ Vt
    # Vt rows are the input-space directions
    U, S, Vt = torch.linalg.svd(W1, full_matrices=False)
    top_K = Vt[:K]  # [K, D]

    logger.info(f"    Weight subspace: top-{K} SVs of W1 [{W1.shape[0]}×{W1.shape[1]}], "
                f"singular values: {S[:K].tolist()[:5]}")

    return top_K


# ═══════════════════════════════════════════════════════════════════════════
# Strategy C: Random subspace (control)
# ═══════════════════════════════════════════════════════════════════════════

def make_random_subspace(D: int, K: int, seed: int = 0) -> torch.Tensor:
    """Generate K orthonormal random directions in D-dim space.

    Returns: [K, D] orthonormal basis.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    R = torch.randn(K, D, generator=rng)
    Q, _ = torch.linalg.qr(R.T)  # [D, K]
    return Q.T[:K]  # [K, D]


# ═══════════════════════════════════════════════════════════════════════════
# Strategy D: Linear probe direction (1-d baseline from E8)
# ═══════════════════════════════════════════════════════════════════════════

def extract_linear_direction(
    linear_probe_weights: dict,
    target: str,
    z_level: str = "H",
    step: int = 15,
) -> torch.Tensor:
    """Get the 1-d weight vector from the linear probe for comparison.

    Returns: [1, D] direction matrix.
    """
    key = f"z_{z_level}_step{step}_{target}"
    if key not in linear_probe_weights:
        # Try to find any matching key
        for k, v in linear_probe_weights.items():
            if v.get("target") == target and v.get("z_level") == z_level:
                key = k
                break
        else:
            raise KeyError(f"No linear probe found for {target}")

    val = linear_probe_weights[key]
    W = val["W"]  # [out, D]
    if W.shape[0] == 1:
        w_vec = W[0]
    else:
        U, S, _ = torch.svd(W.float())
        w_vec = (W.float().T @ U[:, 0])
        w_vec = w_vec / w_vec.norm().clamp(min=1e-8)

    return w_vec.unsqueeze(0)  # [1, D]


# ═══════════════════════════════════════════════════════════════════════════
# Load MLP probe from saved weights
# ═══════════════════════════════════════════════════════════════════════════

def load_mlp_probe(mlp_weights: dict, target: str, step: str = "15",
                   device: torch.device = torch.device("cpu")) -> Tuple[MLPProbe, dict]:
    """Reconstruct a trained MLP probe from saved state dict."""
    key = f"z_H_step{step}_{target}"
    if key not in mlp_weights:
        raise KeyError(f"MLP weights not found for key={key}. Available: {list(mlp_weights.keys())[:10]}")

    info = mlp_weights[key]
    sd = info["state_dict"]

    # Infer dimensions from state dict
    # net.0.weight → [hidden_dim, in_dim]
    # net.2.weight → [out_dim, hidden_dim]
    w0 = sd["net.0.weight"]
    w2 = sd["net.2.weight"]
    in_dim = w0.shape[1]
    hidden_dim = w0.shape[0]
    out_dim = w2.shape[0]

    model = MLPProbe(in_dim, out_dim, hidden_dim).to(device)
    model.load_state_dict(sd)
    model.eval()

    return model, info


# ═══════════════════════════════════════════════════════════════════════════
# Subspace geometry analysis
# ═══════════════════════════════════════════════════════════════════════════

def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Compute the principal angle overlap between two subspaces.

    A: [K1, D], B: [K2, D] — orthonormal bases.
    Returns mean of squared cosines of principal angles (1.0 = identical, 0.0 = orthogonal).
    """
    # Orthonormalize both via QR (in case they aren't perfectly orthonormal)
    Qa, _ = torch.linalg.qr(A.T.float())  # [D, K1]
    Qb, _ = torch.linalg.qr(B.T.float())  # [D, K2]

    # Singular values of Qa^T @ Qb give cosines of principal angles
    M = Qa.T @ Qb  # [K1, K2]
    svs = torch.linalg.svdvals(M)
    return float((svs ** 2).mean().item())


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E9b: Non-Linear Directed Ablation")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--subspace_k", type=int, default=3,
                        help="Dimensionality of extracted subspaces")
    parser.add_argument("--n_activation_samples", type=int, default=200,
                        help="Number of puzzles to use for gradient/activation collection")
    parser.add_argument("--n_random_controls", type=int, default=3,
                        help="Number of random K-d subspace controls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with small N")
    parser.add_argument("--mlp_weights", type=str,
                        default="results/probes/nonlinear_probes/probe_weights.pt",
                        help="Path to MLP probe weights from Gap 3")
    parser.add_argument("--linear_weights", type=str,
                        default="results/probes/e8_constraint_probes/probe_weights.pt",
                        help="Path to linear probe weights from E8")
    parser.add_argument("--output_dir", type=str,
                        default="results/directed_ablation/e9b_nonlinear")
    args = parser.parse_args()

    if args.quick:
        args.n_puzzles = 10
        args.n_activation_samples = 10
        args.subspace_k = 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    K = args.subspace_k

    logger.info(f"Device: {device}, K={K}, n_puzzles={args.n_puzzles}")

    # ── Load probe weights ────────────────────────────────────────────
    if not os.path.exists(args.mlp_weights):
        logger.error(f"MLP weights not found at {args.mlp_weights}. Run nonlinear_probes.py first.")
        sys.exit(1)

    mlp_weights = torch.load(args.mlp_weights, map_location="cpu", weights_only=False)
    logger.info(f"Loaded MLP weights: {len(mlp_weights)} entries")

    linear_probe_weights = None
    if os.path.exists(args.linear_weights):
        linear_probe_weights = torch.load(args.linear_weights, map_location="cpu", weights_only=False)
        logger.info(f"Loaded linear probe weights: {len(linear_probe_weights)} entries")

    # ── Load model + data ─────────────────────────────────────────────
    model, test_loader, test_meta = load_model_and_data(device)
    ablator = DirectionalAblator(model, device=device)

    # ── Collect activations for gradient computation ──────────────────
    logger.info(f"Collecting activations from {args.n_activation_samples} puzzles for gradient subspace...")
    features_H, features_L, label_bank, hidden_dim = collect_activations(
        model, test_loader, device,
        n_puzzles=args.n_activation_samples,
        steps_to_record=[int(BEST_STEP)],
        max_steps=args.max_steps,
    )

    # Stack all step-15 activations
    X_all = torch.cat(features_H[BEST_STEP], dim=0).to(device)
    logger.info(f"Activation tensor: {X_all.shape}")

    # ── Extract subspaces for each constraint target ──────────────────
    all_subspaces: Dict[str, Dict[str, torch.Tensor]] = {}
    # Structure: all_subspaces[target][strategy_name] = [K, D] tensor

    for target in CONSTRAINT_TARGETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting subspaces for: {target}")
        logger.info(f"{'='*60}")

        subspaces: Dict[str, torch.Tensor] = {}

        # (A) Gradient subspace
        try:
            mlp_model, info = load_mlp_probe(mlp_weights, target, step=BEST_STEP, device=device)
            grad_sub = extract_gradient_subspace(mlp_model, X_all, K=K)
            subspaces[f"{target}_mlp_gradient_{K}d"] = grad_sub.to(device)
        except Exception as e:
            logger.warning(f"  Gradient subspace failed for {target}: {e}")

        # (B) Weight subspace
        try:
            mlp_model, info = load_mlp_probe(mlp_weights, target, step=BEST_STEP, device=device)
            weight_sub = extract_weight_subspace(mlp_model, K=K)
            subspaces[f"{target}_mlp_weight_{K}d"] = weight_sub.to(device)
        except Exception as e:
            logger.warning(f"  Weight subspace failed for {target}: {e}")

        # (D) Linear probe direction (1-d)
        if linear_probe_weights is not None:
            try:
                lin_dir = extract_linear_direction(
                    linear_probe_weights, target, z_level="H", step=int(BEST_STEP))
                subspaces[f"{target}_linear_1d"] = lin_dir.to(device)
                logger.info(f"    Linear direction: 1-d")
            except Exception as e:
                logger.warning(f"  Linear direction failed for {target}: {e}")

        # (A+D) Combined: Linear direction + MLP gradient directions (K+1 dim)
        if f"{target}_mlp_gradient_{K}d" in subspaces and f"{target}_linear_1d" in subspaces:
            combined = torch.cat([
                subspaces[f"{target}_linear_1d"],
                subspaces[f"{target}_mlp_gradient_{K}d"],
            ], dim=0)  # [K+1, D]
            # Orthonormalize
            Q, _ = torch.linalg.qr(combined.T.float())
            combined_orth = Q.T[:K+1]
            subspaces[f"{target}_combined_{K+1}d"] = combined_orth.to(device)
            logger.info(f"    Combined (linear + gradient): {K+1}-d")

        all_subspaces[target] = subspaces

    # (C) Random subspace controls — shared across targets
    random_subspaces = {}
    for i in range(args.n_random_controls):
        name = f"random_control_{i}_{K}d"
        random_subspaces[name] = make_random_subspace(hidden_dim, K, seed=args.seed + i).to(device)
        logger.info(f"  Random subspace control {i}: {K}-d")

    # ── Geometry analysis: overlap between subspaces ──────────────────
    logger.info("\n── Subspace Geometry ──")
    geometry = {}
    for target in CONSTRAINT_TARGETS:
        subs = all_subspaces.get(target, {})
        target_geometry = {}
        names = sorted(subs.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                overlap = subspace_overlap(subs[n1], subs[n2])
                key = f"{n1} vs {n2}"
                target_geometry[key] = round(overlap, 4)
                logger.info(f"  {key}: overlap = {overlap:.4f}")

        # Also compare with random
        for rname, rsub in random_subspaces.items():
            for sname, ssub in subs.items():
                overlap = subspace_overlap(ssub, rsub)
                target_geometry[f"{sname} vs {rname}"] = round(overlap, 4)

        geometry[target] = target_geometry

    with open(os.path.join(args.output_dir, "subspace_geometry.json"), "w") as f:
        json.dump(geometry, f, indent=2)

    # ── Collect puzzle batches for ablation ────────────────────────────
    # Re-initialize the dataloader to start from the beginning
    model2, test_loader2, _ = load_model_and_data(device)
    # Use the same model reference
    ablator2 = DirectionalAblator(model, device=device)

    batches = []
    for i, data in enumerate(test_loader2):
        if i >= args.n_puzzles:
            break
        if isinstance(data, (tuple, list)):
            batch = data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        else:
            batch = data
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batches.append(batch)
    logger.info(f"\nLoaded {len(batches)} puzzles for ablation")

    # ── Run baseline (no ablation) ────────────────────────────────────
    logger.info("Running baseline (no ablation)...")
    baselines = []
    for pi, batch in enumerate(batches):
        cache: Dict[int, ActivationCache] = {}
        ablator2.run_and_cache_activations(batch, cache, max_steps=args.max_steps)
        final_step = max(cache.keys())
        final_preds = cache[final_step].preds[:, -SUDOKU_CELLS:]
        targets_tok = batch["labels"][:, -SUDOKU_CELLS:]

        viols = count_violations(final_preds)
        acc = cell_accuracy(final_preds, targets_tok)
        baselines.append({
            "puzzle_idx": pi,
            "accuracy": acc,
            "final_preds": final_preds.cpu(),
            "targets_tok": targets_tok.cpu(),
            **viols,
        })
        if (pi + 1) % 50 == 0:
            logger.info(f"  Baseline: {pi+1}/{len(batches)}")

    mean_baseline_acc = np.mean([b["accuracy"] for b in baselines])
    logger.info(f"Baseline accuracy: {mean_baseline_acc:.4f}")

    # ── Build the complete set of ablation directions ──────────────────
    # Each entry: (name, direction_matrix [K, D])
    ablation_dirs: List[Tuple[str, torch.Tensor]] = []

    for target in CONSTRAINT_TARGETS:
        for name, sub in all_subspaces.get(target, {}).items():
            ablation_dirs.append((name, sub))

    for name, sub in random_subspaces.items():
        ablation_dirs.append((name, sub))

    logger.info(f"\nTotal ablation conditions: {len(ablation_dirs)}")
    for name, sub in ablation_dirs:
        logger.info(f"  {name}: {sub.shape[0]}-d subspace")

    # ── Run directional ablation for each subspace ────────────────────
    all_results: Dict[str, List[dict]] = {}
    aggregate: Dict[str, dict] = {}

    t0 = time.time()
    for dir_name, direction_matrix in ablation_dirs:
        logger.info(f"\nAblating subspace: {dir_name} ({direction_matrix.shape[0]}-d)")
        puzzle_results = []

        for pi, batch in enumerate(batches):
            # Use direction_matrix for multi-dim subspace ablation
            if direction_matrix.shape[0] == 1:
                # 1-d: pass as vector
                final_outputs, abl_cache = ablator2.run_with_directional_ablation(
                    batch, direction_matrix[0],
                    ablate_level="H",
                    ablate_steps=None,
                    max_steps=args.max_steps,
                )
            else:
                # K-d: pass as direction_matrix
                final_outputs, abl_cache = ablator2.run_with_directional_ablation(
                    batch, direction_matrix[0],  # dummy, overridden by direction_matrix
                    ablate_level="H",
                    ablate_steps=None,
                    max_steps=args.max_steps,
                    direction_matrix=direction_matrix,
                )

            final_step = max(abl_cache.keys())
            abl_preds = abl_cache[final_step].preds[:, -SUDOKU_CELLS:]
            targets_tok = batch["labels"][:, -SUDOKU_CELLS:]

            abl_viols = count_violations(abl_preds)
            abl_acc = cell_accuracy(abl_preds, targets_tok)
            broken_info = count_per_unit_broken(
                baselines[pi]["final_preds"].to(device),
                abl_preds,
                targets_tok,
            )

            puzzle_results.append({
                "puzzle_idx": pi,
                "baseline_accuracy": baselines[pi]["accuracy"],
                "ablated_accuracy": abl_acc,
                "delta_accuracy": abl_acc - baselines[pi]["accuracy"],
                "baseline_row_viols": baselines[pi]["violated_rows"],
                "baseline_col_viols": baselines[pi]["violated_cols"],
                "baseline_box_viols": baselines[pi]["violated_boxes"],
                "ablated_row_viols": abl_viols["violated_rows"],
                "ablated_col_viols": abl_viols["violated_cols"],
                "ablated_box_viols": abl_viols["violated_boxes"],
                "delta_row_viols": abl_viols["violated_rows"] - baselines[pi]["violated_rows"],
                "delta_col_viols": abl_viols["violated_cols"] - baselines[pi]["violated_cols"],
                "delta_box_viols": abl_viols["violated_boxes"] - baselines[pi]["violated_boxes"],
                **broken_info,
            })

            if (pi + 1) % 100 == 0:
                logger.info(f"  {dir_name}: {pi+1}/{len(batches)}")

        all_results[dir_name] = puzzle_results

        # Aggregate
        n = len(puzzle_results)
        deltas_acc = [r["delta_accuracy"] for r in puzzle_results]
        agg = {
            "direction": dir_name,
            "subspace_dim": int(direction_matrix.shape[0]),
            "n_puzzles": n,
            "mean_delta_accuracy": float(np.mean(deltas_acc)),
            "std_delta_accuracy": float(np.std(deltas_acc)),
            "sem_delta_accuracy": float(np.std(deltas_acc) / np.sqrt(n)),
            "mean_delta_row_viols": float(np.mean([r["delta_row_viols"] for r in puzzle_results])),
            "mean_delta_col_viols": float(np.mean([r["delta_col_viols"] for r in puzzle_results])),
            "mean_delta_box_viols": float(np.mean([r["delta_box_viols"] for r in puzzle_results])),
            "mean_cells_broken": float(np.mean([r["cells_broken"] for r in puzzle_results])),
            "mean_cells_fixed": float(np.mean([r["cells_fixed"] for r in puzzle_results])),
            "n_puzzles_hurt": sum(1 for r in puzzle_results if r["delta_accuracy"] < -0.001),
            "n_puzzles_helped": sum(1 for r in puzzle_results if r["delta_accuracy"] > 0.001),
            "n_puzzles_unchanged": sum(1 for r in puzzle_results if abs(r["delta_accuracy"]) <= 0.001),
        }
        aggregate[dir_name] = agg
        logger.info(
            f"  → Δacc={agg['mean_delta_accuracy']:+.4f} ± {agg['sem_delta_accuracy']:.4f}, "
            f"Δrow={agg['mean_delta_row_viols']:+.2f}, "
            f"Δcol={agg['mean_delta_col_viols']:+.2f}, "
            f"Δbox={agg['mean_delta_box_viols']:+.2f}, "
            f"broken={agg['mean_cells_broken']:.1f}"
        )

    elapsed = time.time() - t0
    logger.info(f"\nAll ablations done in {elapsed:.1f}s")

    # ── Statistical tests: probe subspaces vs random ──────────────────
    from scipy import stats as sp_stats

    logger.info("\n── Statistical Comparisons ──")
    stat_tests = {}

    # Collect random control deltas for pooled comparison
    random_deltas_all = []
    for name in random_subspaces:
        random_deltas_all.extend([r["delta_accuracy"] for r in all_results[name]])
    random_mean = np.mean(random_deltas_all) if random_deltas_all else 0.0

    for target in CONSTRAINT_TARGETS:
        target_stats = {}
        for strategy_label in ["mlp_gradient", "mlp_weight", "linear", "combined"]:
            # Find matching key
            matching = [k for k in all_results if target in k and strategy_label in k]
            if not matching:
                continue
            name = matching[0]
            probe_deltas = [r["delta_accuracy"] for r in all_results[name]]

            # Compare against each random control
            for rname in random_subspaces:
                rand_deltas = [r["delta_accuracy"] for r in all_results[rname]]
                t_stat, p_val = sp_stats.ttest_ind(probe_deltas, rand_deltas, equal_var=False)
                # Cohen's d
                pooled_std = np.sqrt((np.var(probe_deltas) + np.var(rand_deltas)) / 2)
                cohens_d = (np.mean(probe_deltas) - np.mean(rand_deltas)) / pooled_std if pooled_std > 0 else 0

                key = f"{name} vs {rname}"
                target_stats[key] = {
                    "t_stat": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "cohens_d": round(float(cohens_d), 4),
                    "probe_mean": round(float(np.mean(probe_deltas)), 5),
                    "random_mean": round(float(np.mean(rand_deltas)), 5),
                }
                logger.info(f"  {key}: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

            # Compare MLP gradient vs linear probe (key comparison)
            linear_match = [k for k in all_results if target in k and "linear" in k]
            gradient_match = [k for k in all_results if target in k and "mlp_gradient" in k]
            if linear_match and gradient_match:
                lin_deltas = [r["delta_accuracy"] for r in all_results[linear_match[0]]]
                grad_deltas = [r["delta_accuracy"] for r in all_results[gradient_match[0]]]
                t_stat, p_val = sp_stats.ttest_ind(grad_deltas, lin_deltas, equal_var=False)
                pooled_std = np.sqrt((np.var(grad_deltas) + np.var(lin_deltas)) / 2)
                cohens_d = (np.mean(grad_deltas) - np.mean(lin_deltas)) / pooled_std if pooled_std > 0 else 0
                key = f"{gradient_match[0]} vs {linear_match[0]}"
                target_stats[key] = {
                    "t_stat": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "cohens_d": round(float(cohens_d), 4),
                    "mlp_mean": round(float(np.mean(grad_deltas)), 5),
                    "linear_mean": round(float(np.mean(lin_deltas)), 5),
                }
                logger.info(f"  {key}: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

        stat_tests[target] = target_stats

    # ── Build specificity matrix ──────────────────────────────────────
    spec_matrix = {}
    for dir_name, agg in aggregate.items():
        spec_matrix[dir_name] = {
            "subspace_dim": agg["subspace_dim"],
            "delta_accuracy": round(agg["mean_delta_accuracy"], 5),
            "sem_accuracy": round(agg["sem_delta_accuracy"], 5),
            "delta_row_viols": round(agg["mean_delta_row_viols"], 4),
            "delta_col_viols": round(agg["mean_delta_col_viols"], 4),
            "delta_box_viols": round(agg["mean_delta_box_viols"], 4),
            "cells_broken": round(agg["mean_cells_broken"], 2),
        }

    # ── Save all results ──────────────────────────────────────────────
    # Serialization helper
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=_default)

    with open(os.path.join(args.output_dir, "aggregate_results.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=_default)

    with open(os.path.join(args.output_dir, "specificity_matrix.json"), "w") as f:
        json.dump(spec_matrix, f, indent=2, default=_default)

    with open(os.path.join(args.output_dir, "statistical_tests.json"), "w") as f:
        json.dump(stat_tests, f, indent=2, default=_default)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 110)
    print("E9b NON-LINEAR DIRECTED ABLATION — RESULTS SUMMARY")
    print("=" * 110)
    print(f"{'Subspace ablated':45s}  {'dim':>4s}  {'Δacc':>9s}  {'±SEM':>7s}  "
          f"{'Δrow':>7s}  {'Δcol':>7s}  {'Δbox':>7s}  {'broken':>7s}")
    print("-" * 110)

    # Group by target
    for target in CONSTRAINT_TARGETS:
        print(f"\n  ── {target} ──")
        for dir_name in sorted(aggregate.keys()):
            if target not in dir_name and "random" not in dir_name:
                continue
            if "random" in dir_name and target != CONSTRAINT_TARGETS[0]:
                continue  # Print random only once
            s = spec_matrix[dir_name]
            print(
                f"  {dir_name:43s}  {s['subspace_dim']:>4d}  "
                f"{s['delta_accuracy']:>+9.4f}  {s['sem_accuracy']:>7.4f}  "
                f"{s['delta_row_viols']:>+7.3f}  "
                f"{s['delta_col_viols']:>+7.3f}  "
                f"{s['delta_box_viols']:>+7.3f}  "
                f"{s['cells_broken']:>7.1f}"
            )

    # Print random controls
    print(f"\n  ── Random Controls ──")
    for dir_name in sorted(aggregate.keys()):
        if "random" not in dir_name:
            continue
        s = spec_matrix[dir_name]
        print(
            f"  {dir_name:43s}  {s['subspace_dim']:>4d}  "
            f"{s['delta_accuracy']:>+9.4f}  {s['sem_accuracy']:>7.4f}  "
            f"{s['delta_row_viols']:>+7.3f}  "
            f"{s['delta_col_viols']:>+7.3f}  "
            f"{s['delta_box_viols']:>+7.3f}  "
            f"{s['cells_broken']:>7.1f}"
        )

    # ── Interpretation ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("INTERPRETATION")
    print("=" * 110)

    # Compare MLP gradient vs linear vs random
    for target in CONSTRAINT_TARGETS:
        grad_key = f"{target}_mlp_gradient_{K}d"
        lin_key = f"{target}_linear_1d"
        if grad_key in aggregate and lin_key in aggregate:
            grad_eff = aggregate[grad_key]["mean_delta_accuracy"]
            lin_eff = aggregate[lin_key]["mean_delta_accuracy"]
            rand_effs = [aggregate[k]["mean_delta_accuracy"]
                         for k in random_subspaces if k in aggregate]
            rand_mean = np.mean(rand_effs) if rand_effs else 0.0

            print(f"\n  {target}:")
            print(f"    Linear probe (1-d):       Δacc = {lin_eff:+.4f}")
            print(f"    MLP gradient ({K}-d):      Δacc = {grad_eff:+.4f}")
            print(f"    Random control ({K}-d):    Δacc = {rand_mean:+.4f}")

            if abs(grad_eff) > abs(rand_mean) * 2 and abs(grad_eff) > 0.01:
                print(f"    → MLP subspace has STRONGER causal effect than random")
            elif abs(grad_eff) > abs(lin_eff) * 2 and abs(grad_eff) > 0.005:
                print(f"    → MLP subspace has stronger effect than linear probe")
            else:
                print(f"    → MLP subspace shows SIMILAR effect to random — readout ≠ computation holds")

    print(f"\nResults saved to {args.output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
