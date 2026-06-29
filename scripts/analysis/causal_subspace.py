#!/usr/bin/env python3
"""Experiment H2: Minimal causal subspace.

Builds candidate direction pools (PCA-top, PCA-bottom, random, probe+SAE),
runs damage-vs-rank curves via subspace ablation, and measures alignment
between the causal subspace and the readable (probe/SAE) basis.

Importable entry point (consumed by localizability_scorecard.py):
    from scripts.analysis.causal_subspace import find_causal_subspace

CLI:
    python scripts/analysis/causal_subspace.py --task sudoku --device cuda
    python scripts/analysis/causal_subspace.py --task maze --device cuda
"""
from __future__ import annotations
import os, sys, json, argparse, logging, time
from typing import Dict, List, Optional, Tuple, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci, find_checkpoint,
)
from scripts.core.activation_ablation import ActivationAblator
from scripts.directed_ablation.e9_directed_ablation import (
    DirectionalAblator, select_best_directions,
)
from scripts.core.activation_patching import compute_metrics
from scripts.maze.maze_common import MAZE_CHECKPOINT, maze_prediction_metrics
from scripts.arc.arc_common import ARC_CHECKPOINT
from scripts.analysis.policy_improvement import task_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PCA pool construction
# ---------------------------------------------------------------------------

def _build_pca_pool(
    model, batches: list, device: torch.device,
    max_steps: int, pel: int, n_limit: int = 200_000, seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect z_H_out across puzzles/steps, subsample, center, SVD.

    Returns:
        V   [D, D] float32 — rows are PC directions sorted descending by variance.
        mu  [1, D] float32 — column mean (for centering).
    """
    ablator = ActivationAblator(model, device=device)
    rows: List[torch.Tensor] = []
    total = 0
    for puzzle_idx, batch in batches:
        cache: Dict[int, Any] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
        for s in sorted(cache.keys()):
            z_out = cache[s].z_H_out[0, pel:].float().cpu()  # [seq, D]
            rows.append(z_out)
            total += z_out.shape[0]
        if total >= n_limit:
            break

    M = torch.cat(rows, dim=0)  # [M, D]
    if M.shape[0] > n_limit:
        rng = torch.Generator()
        rng.manual_seed(seed)
        idx = torch.randperm(M.shape[0], generator=rng)[:n_limit]
        M = M[idx]

    mu = M.mean(0, keepdim=True)
    M_c = M - mu
    # SVD: Vh rows = right singular vectors = PC directions, sorted descending
    _, _, Vh = torch.linalg.svd(M_c, full_matrices=False)  # Vh [D, D]
    logger.info(f"  PCA: {M.shape[0]} rows, D={Vh.shape[1]}, got {Vh.shape[0]} PCs")
    return Vh.float(), mu.float()


# ---------------------------------------------------------------------------
# Random direction pools
# ---------------------------------------------------------------------------

def _build_random_pools(D: int, R: int, seed: int = 42) -> List[torch.Tensor]:
    """Return R random orthonormal bases, each [D, D] with orthonormal rows."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    pools = []
    for _ in range(R):
        G = torch.randn(D, D, generator=rng)
        Q, _ = torch.linalg.qr(G)  # Q [D, D] orthogonal
        pools.append(Q)  # rows are directions
    return pools


# ---------------------------------------------------------------------------
# Probe+SAE pool
# ---------------------------------------------------------------------------

def _build_probe_sae_pool(
    probe_weights_path: Optional[str],
    sae_path: Optional[str],
    D: int,
) -> Optional[torch.Tensor]:
    """Stack probe + SAE direction vectors, orthonormalize. Returns [K, D] or None."""
    directions: List[torch.Tensor] = []

    if probe_weights_path and os.path.exists(probe_weights_path):
        try:
            probe_weights = torch.load(probe_weights_path, map_location="cpu", weights_only=False)
            dir_dict = select_best_directions(probe_weights, z_level="H", step=None)
            for tgt, w in dir_dict.items():
                w_f = w.float().view(-1)
                w_f = w_f / w_f.norm().clamp(min=1e-8)
                directions.append(w_f)
            logger.info(f"  Probe pool: {len(dir_dict)} directions")
        except Exception as e:
            logger.warning(f"  Probe loading failed: {e}")

    if sae_path and os.path.exists(sae_path):
        try:
            from scripts.sae.sae_causal_ablation import select_top_features
            sae = torch.load(sae_path, map_location="cpu", weights_only=False)
            # select_top_features needs activations_path; use the Sudoku bank if it exists
            acts_path = os.path.join(REPO_ROOT, "results", "sae_study", "activations_zH.pt")
            if os.path.exists(acts_path):
                top_feats = select_top_features(sae, acts_path, top_k=20, device="cpu")
                # decoder.weight: [D, dict_size]; columns are feature directions
                feat_dirs = sae.decoder.weight[:, top_feats].T.float()  # [20, D]
                feat_dirs = F.normalize(feat_dirs, dim=-1)
                directions.extend([feat_dirs[i] for i in range(min(20, feat_dirs.shape[0]))])
                logger.info(f"  SAE pool: {feat_dirs.shape[0]} directions added")
        except Exception as e:
            logger.warning(f"  SAE loading failed: {e}")

    if not directions:
        return None

    B = torch.stack(directions, dim=0)  # [K, D]
    # Orthonormalize via QR
    Q, _ = torch.linalg.qr(B.T)  # Q [D, K]
    result = Q.T.float()  # [K, D] orthonormal rows
    logger.info(f"  Probe+SAE pool: {result.shape[0]} orthonormalized directions")
    return result


# ---------------------------------------------------------------------------
# Damage measurement helpers
# ---------------------------------------------------------------------------

def _get_baselines(
    model, batches: list, device: torch.device,
    task: str, max_steps: int,
) -> Dict[int, float]:
    ablator = ActivationAblator(model, device=device)
    baseline_values: Dict[int, float] = {}
    for puzzle_idx, batch in batches:
        cache: Dict[int, Any] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
        last_s = max(cache.keys())
        preds_row = cache[last_s].preds[0].cpu().numpy()
        labels_row = batch["labels"][0].cpu().numpy()
        inputs_row = batch["inputs"][0].cpu().numpy()
        tv = task_value(preds_row, labels_row, inputs_row, task)
        baseline_values[puzzle_idx] = tv["value"]
    return baseline_values


def _compute_delta_full(
    model, batches: list, device: torch.device,
    task: str, max_steps: int,
    baseline_values: Dict[int, float],
) -> Tuple[float, List[float]]:
    """Full z_H zeroing ablation reference."""
    ablator = ActivationAblator(model, device=device)
    deltas: List[float] = []
    for puzzle_idx, batch in batches:
        final_out, abl_cache, _ = ablator.run_with_ablation(
            batch, ablate_level="H", ablate_steps=None,
            max_steps=max_steps, ablation_value=0.0,
        )
        last_s = max(abl_cache.keys())
        preds_row = abl_cache[last_s].preds[0].cpu().numpy()
        labels_row = batch["labels"][0].cpu().numpy()
        inputs_row = batch["inputs"][0].cpu().numpy()
        tv = task_value(preds_row, labels_row, inputs_row, task)
        deltas.append(tv["value"] - baseline_values[puzzle_idx])
    return float(np.mean(deltas)), deltas


def _damage_curve(
    model, batches: list, device: torch.device,
    task: str, max_steps: int,
    baseline_values: Dict[int, float],
    pool: torch.Tensor,    # [P, D] rows are direction candidates
    ranks: List[int],
    desc: str = "",
) -> Dict[int, dict]:
    """For each rank r, project out pool[:r] and measure mean Δvalue."""
    ablator = DirectionalAblator(model, device=device)
    D = pool.shape[-1]
    curve: Dict[int, dict] = {}
    for r in ranks:
        r_capped = min(r, pool.shape[0])
        Q = pool[:r_capped].to(device)  # [r, D]
        deltas: List[float] = []
        for puzzle_idx, batch in batches:
            final_out, cache = ablator.run_with_directional_ablation(
                batch,
                direction=torch.zeros(D, device=device),  # ignored when direction_matrix given
                ablate_level="H",
                ablate_steps=None,
                max_steps=max_steps,
                direction_matrix=Q,
            )
            last_s = max(cache.keys())
            preds_row = cache[last_s].preds[0].cpu().numpy()
            labels_row = batch["labels"][0].cpu().numpy()
            inputs_row = batch["inputs"][0].cpu().numpy()
            tv = task_value(preds_row, labels_row, inputs_row, task)
            deltas.append(tv["value"] - baseline_values[puzzle_idx])
        ci = bootstrap_ci(deltas)
        curve[r_capped] = ci
        logger.info(
            f"  {desc} r={r_capped}: Δ={ci['mean']:+.4f} "
            f"[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]"
        )
    return curve


# ---------------------------------------------------------------------------
# r* and alignment analysis
# ---------------------------------------------------------------------------

def _find_r_star(curve: Dict[int, dict], delta_full: float, frac: float) -> int:
    """Smallest r where mean Δ ≤ frac × delta_full (both negative → more damage)."""
    threshold = frac * delta_full
    for r in sorted(curve.keys()):
        if curve[r]["mean"] <= threshold:
            return r
    return max(curve.keys())


def _alignment_analysis(
    pca_top: torch.Tensor,   # [D, D] rows = PCs
    r_star: int,
    probe_sae_pool: Optional[torch.Tensor],  # [K, D] or None
    D: int,
    n_random: int = 5,
    seed: int = 42,
) -> dict:
    """Project readable basis into causal subspace, compute energy + principal angles."""
    Qc = pca_top[:r_star].float()  # [r*, D]

    def _energy(B: torch.Tensor) -> float:
        # B [K, D]; fraction of B's variance in span(Qc)
        B_f = B.float()
        proj = Qc @ B_f.T  # [r*, K]
        return float(proj.pow(2).sum() / B_f.pow(2).sum().clamp(min=1e-10))

    results: dict = {"r_star": int(r_star)}

    if probe_sae_pool is not None and probe_sae_pool.shape[0] > 0:
        results["probe_sae_energy"] = _energy(probe_sae_pool)
        B_norm = F.normalize(probe_sae_pool.float(), dim=-1)
        sv = torch.linalg.svdvals(Qc @ B_norm.T)  # [min(r*, K)]
        results["principal_angles_cos"] = sv.clamp(-1.0, 1.0).tolist()

    # Random control
    rng = torch.Generator()
    rng.manual_seed(seed)
    rand_energies: List[float] = []
    n_dirs = max(r_star, 5)
    for _ in range(n_random):
        B_rand = F.normalize(torch.randn(n_dirs, D, generator=rng), dim=-1)
        rand_energies.append(_energy(B_rand))
    results["random_control_energy_mean"] = float(np.mean(rand_energies))
    results["random_control_energy_std"] = float(np.std(rand_energies))

    return results


def _subspace_linearity(curve: Dict[int, dict]) -> float:
    """Measure how linear the damage curve is. Returns R^2 of linear fit."""
    rs = sorted(curve.keys())
    if len(rs) < 3:
        return float("nan")
    x = np.array(rs, dtype=float)
    y = np.array([curve[r]["mean"] for r in rs], dtype=float)
    # Linear fit
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot.clip(1e-10))


# ---------------------------------------------------------------------------
# Main compute function (importable)
# ---------------------------------------------------------------------------

def find_causal_subspace(
    model,
    batches: list,
    device: torch.device,
    task: str,
    ranks: List[int],
    n_random_bases: int,
    probe_weights_path: Optional[str],
    sae_path: Optional[str],
    n_pca_puzzles: int,
    max_steps: int,
    z_level: str = "H",
    seed: int = 42,
) -> dict:
    """H2 main computation. Returns dict with all results."""
    torch.manual_seed(seed)
    pel = int(model.inner.puzzle_emb_len)
    D = int(model.inner.config.hidden_size)
    logger.info(f"[H2] D={D}, pel={pel}, n_puzzles={len(batches)}, ranks={ranks}")

    # Baselines
    logger.info("[H2] Computing baselines...")
    baseline_values = _get_baselines(model, batches, device, task, max_steps)
    baseline_mean = float(np.mean(list(baseline_values.values())))
    logger.info(f"[H2] baseline mean value: {baseline_mean:.4f}")

    # Full-ablation reference
    logger.info("[H2] Full z_H ablation reference...")
    delta_full_mean, delta_full_list = _compute_delta_full(
        model, batches, device, task, max_steps, baseline_values
    )
    logger.info(f"[H2] Δfull = {delta_full_mean:.4f}  (expected ≈ -0.19 for Sudoku)")

    # PCA pool
    logger.info("[H2] Building PCA pool...")
    pca_batches = batches[:max(n_pca_puzzles, len(batches))]
    pca_V, pca_mu = _build_pca_pool(model, pca_batches, device, max_steps, pel, seed=seed)

    # Random pools
    random_pools = _build_random_pools(D, n_random_bases, seed=seed)

    # Probe+SAE pool
    logger.info("[H2] Building probe+SAE pool...")
    probe_sae_pool = _build_probe_sae_pool(probe_weights_path, sae_path, D)

    # --- Damage curves ---
    curves: Dict[str, dict] = {}

    logger.info("[H2] PCA-top damage curve...")
    curves["pca_top"] = _damage_curve(
        model, batches, device, task, max_steps, baseline_values,
        pca_V, ranks, desc="PCA-top",
    )

    logger.info("[H2] PCA-bottom damage curve...")
    pca_bottom = torch.flip(pca_V, [0])  # reversed
    curves["pca_bottom"] = _damage_curve(
        model, batches, device, task, max_steps, baseline_values,
        pca_bottom, ranks, desc="PCA-bottom",
    )

    logger.info("[H2] Random damage curves...")
    random_all: List[Dict[int, dict]] = []
    for i, pool in enumerate(random_pools):
        rc = _damage_curve(
            model, batches, device, task, max_steps, baseline_values,
            pool, ranks, desc=f"rand[{i}]",
        )
        random_all.append(rc)
    # Average random curves across R draws
    curves["random"] = {}
    for r in ranks:
        r_c = min(r, D)
        vals_m = [rc.get(r_c, rc.get(r, {})).get("mean", float("nan")) for rc in random_all]
        vals_lo = [rc.get(r_c, rc.get(r, {})).get("ci_lower", float("nan")) for rc in random_all]
        vals_hi = [rc.get(r_c, rc.get(r, {})).get("ci_upper", float("nan")) for rc in random_all]
        good = [v for v in vals_m if not np.isnan(v)]
        curves["random"][r_c] = {
            "mean": float(np.mean(good)) if good else float("nan"),
            "ci_lower": float(np.mean([v for v in vals_lo if not np.isnan(v)])),
            "ci_upper": float(np.mean([v for v in vals_hi if not np.isnan(v)])),
            "std": float(np.std(good)) if good else float("nan"),
            "n": len(good),
        }

    if probe_sae_pool is not None:
        logger.info("[H2] Probe+SAE damage curve...")
        curves["probe_sae"] = _damage_curve(
            model, batches, device, task, max_steps, baseline_values,
            probe_sae_pool, ranks, desc="probe+SAE",
        )

    # r* for fracs 0.5 and 0.9
    r_star: Dict[str, Dict[str, int]] = {}
    for frac_label, frac in [("half", 0.5), ("ninety", 0.9)]:
        r_star[frac_label] = {}
        for ord_name in curves:
            r_star[frac_label][ord_name] = int(_find_r_star(curves[ord_name], delta_full_mean, frac))

    # Alignment analysis
    r_star_ninety_pca = r_star["ninety"]["pca_top"]
    logger.info(f"[H2] r*(0.9) PCA-top = {r_star_ninety_pca}. Running alignment...")
    alignment = _alignment_analysis(pca_V, r_star_ninety_pca, probe_sae_pool, D, seed=seed)

    # Subspace linearity of PCA-top curve
    linearity = _subspace_linearity(curves["pca_top"])

    return {
        "delta_full": float(delta_full_mean),
        "delta_full_ci": bootstrap_ci(delta_full_list),
        "baseline_mean": float(baseline_mean),
        "curves": {k: {str(r): v for r, v in crv.items()} for k, crv in curves.items()},
        "r_star": r_star,
        "alignment": alignment,
        "min_causal_rank": r_star_ninety_pca,
        "subspace_linearity": float(linearity),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H2: Minimal causal subspace")
    parser.add_argument("--task", choices=["sudoku", "maze", "arc"], default="sudoku")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_puzzles", type=int, default=300)
    parser.add_argument("--n_pca", type=int, default=200,
                        help="Puzzles to use for PCA pool (at most num_puzzles)")
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--n_random_bases", type=int, default=5)
    parser.add_argument("--probe_weights",
                        default="results/probes/e8_constraint_probes/probe_weights.pt")
    parser.add_argument("--sae_path",
                        default="results/sae_study/sae_d2048_l10.01.pt")
    parser.add_argument("--z_level", default="H", choices=["H", "L"])
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Set defaults
    if args.task == "sudoku":
        ckpt = args.checkpoint or find_checkpoint()
        output_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "controlled", "causal_subspace")
    elif args.task == "arc":
        ckpt = args.checkpoint or ARC_CHECKPOINT
        output_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "arc", "causal_subspace")
    else:
        ckpt = args.checkpoint or MAZE_CHECKPOINT
        output_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "maze", "causal_subspace")

    os.makedirs(output_dir, exist_ok=True)

    ranks = [int(r) for r in args.ranks.split(",")]
    probe_path = os.path.join(REPO_ROOT, args.probe_weights) if not os.path.isabs(args.probe_weights) else args.probe_weights
    sae_path = os.path.join(REPO_ROOT, args.sae_path) if not os.path.isabs(args.sae_path) else args.sae_path

    logger.info(f"[H2] task={args.task}  checkpoint={ckpt}  device={device}")
    logger.info(f"[H2] output_dir={output_dir}")

    # Load model
    model, test_loader, config = load_model_and_dataloader(ckpt, device)
    model.eval()

    # Collect puzzles
    batches = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
    logger.info(f"[H2] Collected {len(batches)} puzzles")

    t0 = time.time()
    results = find_causal_subspace(
        model=model,
        batches=batches,
        device=device,
        task=args.task,
        ranks=ranks,
        n_random_bases=args.n_random_bases,
        probe_weights_path=probe_path,
        sae_path=sae_path,
        n_pca_puzzles=min(args.n_pca, len(batches)),
        max_steps=args.max_steps,
        z_level=args.z_level,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info(f"[H2] done in {elapsed:.0f}s")

    # Add metadata
    results["meta"] = {
        "task": args.task, "checkpoint": ckpt,
        "num_puzzles": len(batches), "n_pca": args.n_pca,
        "ranks": ranks, "n_random_bases": args.n_random_bases,
        "max_steps": args.max_steps, "seed": args.seed,
        "elapsed_s": round(elapsed, 1),
    }

    # Save outputs
    curve_path = os.path.join(output_dir, "subspace_curve.json")
    with open(curve_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {curve_path}")

    align_path = os.path.join(output_dir, "alignment.json")
    with open(align_path, "w") as f:
        json.dump({"alignment": results["alignment"], "r_star": results["r_star"]}, f, indent=2)
    logger.info(f"Saved {align_path}")

    # Provenance
    try:
        from scripts.core.provenance import write_meta
        write_meta(output_dir, "H2_causal_subspace", results["meta"], repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"provenance write failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("H2 CAUSAL SUBSPACE — SUMMARY")
    print("=" * 60)
    print(f"  Δfull (z_H zeroing): {results['delta_full']:+.4f}  (expected ≈ -0.19 Sudoku)")
    print(f"  baseline mean value: {results['baseline_mean']:.4f}")
    for ord_name, crv in results["curves"].items():
        rs = sorted(crv.keys(), key=lambda x: int(x))
        r10 = rs[min(4, len(rs)-1)]
        r100 = rs[min(7, len(rs)-1)]
        print(f"  {ord_name:12s}  r={r10}: {crv[r10]['mean']:+.4f}  r={r100}: {crv[r100]['mean']:+.4f}")
    print(f"  r*(0.9) PCA-top: {results['r_star']['ninety']['pca_top']}")
    if "probe_sae_energy" in results["alignment"]:
        print(f"  probe+SAE energy in causal subspace: {results['alignment']['probe_sae_energy']:.4f}")
    print(f"  random control energy: {results['alignment']['random_control_energy_mean']:.4f} "
          f"± {results['alignment']['random_control_energy_std']:.4f}")


if __name__ == "__main__":
    main()
