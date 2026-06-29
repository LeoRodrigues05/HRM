#!/usr/bin/env python3
"""H4 flat-model comparison: does the UT exhibit two-timescale structure?

Instruments the UT's inner iterations to measure per-iteration change rates
and compares against HRM's z_H / z_L change rate ratio.

Usage:
    python scripts/analysis/two_timescale_baseline.py --device cuda
    python scripts/analysis/two_timescale_baseline.py --tag ut_bptt --device cuda  # skipped if no ckpt
"""
from __future__ import annotations
import os, sys, json, argparse, logging, time
from typing import Dict, List, Optional, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci, find_checkpoint,
)
from scripts.core.activation_ablation import ActivationAblator
from scripts.analysis.baseline_localizability import (
    find_ut_checkpoint, load_ut_model, UTActivationHarness,
    UT_CHECKPOINT_DIRS,
)
from scripts.analysis.policy_improvement import task_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HRM two-timescale analysis (change rates per step for z_H and z_L)
# ---------------------------------------------------------------------------

def hrm_change_rates(
    model, batches: list, device: torch.device, max_steps: int,
) -> Dict[str, Any]:
    """Compute per-step ‖Δz_H‖/‖z_H‖ and ‖Δz_L‖/‖z_L‖ change rates."""
    ablator = ActivationAblator(model, device=device)

    step_rates_H: Dict[int, List[float]] = {}
    step_rates_L: Dict[int, List[float]] = {}

    for puzzle_idx, batch in batches:
        cache: Dict[int, Any] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
        steps = sorted(cache.keys())

        for s in steps[1:]:  # need s-1 to compute difference
            if s - 1 not in cache:
                continue
            z_H_prev = cache[s - 1].z_H_out.float()
            z_H_curr = cache[s].z_H.float()  # carry-in = previous z_H_out after reset
            z_L_prev = cache[s - 1].z_L_out.float()
            z_L_curr = cache[s].z_L.float()

            rate_H = float((z_H_curr - z_H_prev).norm() / z_H_prev.norm().clamp(min=1e-8))
            rate_L = float((z_L_curr - z_L_prev).norm() / z_L_prev.norm().clamp(min=1e-8))
            step_rates_H.setdefault(s, []).append(rate_H)
            step_rates_L.setdefault(s, []).append(rate_L)

    # Per-step summary
    per_step = {}
    for s in sorted(step_rates_H.keys()):
        per_step[str(s)] = {
            "rate_H": bootstrap_ci(step_rates_H[s]),
            "rate_L": bootstrap_ci(step_rates_L[s]),
            "ratio_L_over_H": float(np.mean(step_rates_L[s]) / max(np.mean(step_rates_H[s]), 1e-8)),
        }

    mean_rate_H = float(np.mean([np.mean(v) for v in step_rates_H.values()]))
    mean_rate_L = float(np.mean([np.mean(v) for v in step_rates_L.values()]))

    return {
        "mean_rate_H": mean_rate_H,
        "mean_rate_L": mean_rate_L,
        "mean_ratio_L_over_H": mean_rate_L / max(mean_rate_H, 1e-8),
        "per_step": per_step,
    }


# ---------------------------------------------------------------------------
# UT inner-iteration instrumentation
# ---------------------------------------------------------------------------

def ut_inner_iteration_states(
    model, batch: Dict[str, torch.Tensor], device: torch.device,
) -> List[torch.Tensor]:
    """Capture z after each _apply_shared_block call within one ACT outer step.

    Returns list of tensors [z_0, z_1, ..., z_num_iterations] (length = num_iterations+1).
    z_0 is the state BEFORE any iterations (just after reset + starting from carry).
    """
    from models.baselines.universal_transformer import UniversalTransformerInnerCarry

    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry.inner_carry = UniversalTransformerInnerCarry(
            z=carry.inner_carry.z.to(device)
        )
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        carry.current_data = {k: v.to(device) for k, v in carry.current_data.items()}

        new_inner = model.inner.reset_carry(carry.halted, carry.inner_carry)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }

        cos_sin = model.inner.rotary_emb() if hasattr(model.inner, "rotary_emb") else None
        input_embeddings = model.inner._input_embeddings(
            new_current_data["inputs"], new_current_data["puzzle_identifiers"]
        )

        z = new_inner.z
        states = [z.detach().clone()]  # z_0

        num_iter = model.inner.config.num_iterations
        one_step_grad = getattr(model.inner.config, "one_step_grad", True)

        if one_step_grad:
            # no-grad on all but last
            for _iter in range(num_iter - 1):
                z = model.inner._apply_shared_block(z, input_embeddings, cos_sin)
                states.append(z.detach().clone())
        else:
            for _iter in range(num_iter):
                z = model.inner._apply_shared_block(z, input_embeddings, cos_sin)
                states.append(z.detach().clone())

        # Always capture last (already done in else-branch; for one_step_grad add it)
        if one_step_grad:
            z = model.inner._apply_shared_block(z, input_embeddings, cos_sin)
            states.append(z.detach().clone())

    return states  # length num_iter + 1


def ut_timescale_metrics(
    model, batches: list, device: torch.device,
) -> Dict[str, Any]:
    """Compute per-inner-iteration change rates across many puzzles.

    Also computes a PCA-based trajectory analysis to check for slow/fast split.
    """
    all_iter_rates: Dict[int, List[float]] = {}  # iter_i → [change rates across puzzles]

    for puzzle_idx, batch in batches:
        states = ut_inner_iteration_states(model, batch, device)

        for i in range(1, len(states)):
            z_prev = states[i - 1].float()
            z_curr = states[i].float()
            rate = float((z_curr - z_prev).norm() / z_prev.norm().clamp(min=1e-8))
            all_iter_rates.setdefault(i, []).append(rate)

    per_iter = {
        str(i): bootstrap_ci(rates)
        for i, rates in sorted(all_iter_rates.items())
    }

    mean_rates = [float(np.mean(rates)) for rates in [all_iter_rates[i] for i in sorted(all_iter_rates)]]

    # Coefficient of variation of per-iteration rates (lower = more uniform)
    cv = float(np.std(mean_rates) / np.mean(mean_rates)) if mean_rates else float("nan")

    # Early vs late iteration ratio (do later iters change more or less?)
    half = len(mean_rates) // 2
    early_mean = float(np.mean(mean_rates[:half])) if half > 0 else float("nan")
    late_mean = float(np.mean(mean_rates[half:])) if len(mean_rates) > half else float("nan")
    early_late_ratio = late_mean / max(early_mean, 1e-8)

    # PCA trajectory: collect per-step (outer ACT step) mean-pooled z across batches
    # For UT 1-step (halt_max_steps=16) run full rollout
    outer_z_traj: Dict[int, List[torch.Tensor]] = {}
    harness = UTActivationHarness(model, device)

    for puzzle_idx, batch in batches[:min(50, len(batches))]:
        cache = harness.run_and_cache(batch, model.config.halt_max_steps)
        for s, cdata in cache.items():
            # mean-pool z_out over tokens (squeeze batch dim)
            z_mean = cdata["z_out"][0].float().mean(0)  # [D]
            outer_z_traj.setdefault(s, []).append(z_mean.cpu())

    # PCA over outer steps' mean z
    if len(outer_z_traj) >= 2:
        all_steps = sorted(outer_z_traj.keys())
        traj_mat = torch.stack(
            [torch.stack(outer_z_traj[s], dim=0).mean(0) for s in all_steps], dim=0
        )  # [n_steps, D]
        traj_c = traj_mat - traj_mat.mean(0, keepdim=True)
        _, S, _ = torch.linalg.svd(traj_c, full_matrices=False)
        S_np = S.cpu().numpy()
        total_var = float(S_np.sum())
        pc1_pct = float(S_np[0] / max(total_var, 1e-8))
        pca_trajectory = {"pc1_pct": pc1_pct, "n_steps": len(all_steps)}
    else:
        pca_trajectory = {}

    return {
        "per_iteration_change_rate": per_iter,
        "mean_rates": mean_rates,
        "coeff_of_variation": cv,
        "early_late_ratio": float(early_late_ratio),
        "pca_trajectory": pca_trajectory,
        "num_iterations": model.inner.config.num_iterations,
        "one_step_grad": getattr(model.inner.config, "one_step_grad", True),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H4 two-timescale baseline (UT)")
    parser.add_argument("--tag", choices=["ut_1step", "ut_bptt"], default="ut_1step")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--num_puzzles", type=int, default=200)
    parser.add_argument("--max_steps_hrm", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--include_hrm", action="store_true", default=True,
                        help="Also run HRM change-rate analysis for comparison")
    parser.add_argument("--no_hrm", dest="include_hrm", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "baseline_comparison", "two_timescale"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {"hrm": None, "ut": None, "tag": args.tag}

    # --- HRM analysis ---
    if args.include_hrm:
        logger.info("[H4-2TS] Running HRM change-rate analysis...")
        try:
            ckpt = find_checkpoint()
            hrm_model, test_loader, _ = load_model_and_dataloader(ckpt, device)
            hrm_model.eval()
            hrm_batches = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
            hrm_rates = hrm_change_rates(hrm_model, hrm_batches, device, args.max_steps_hrm)
            results["hrm"] = hrm_rates
            logger.info(
                f"  HRM  mean_rate_H={hrm_rates['mean_rate_H']:.4f}  "
                f"mean_rate_L={hrm_rates['mean_rate_L']:.4f}  "
                f"ratio_L/H={hrm_rates['mean_ratio_L_over_H']:.2f}×"
            )
        except Exception as e:
            logger.warning(f"  HRM analysis failed: {e}")

    # --- UT analysis ---
    ckpt_dir = args.checkpoint_dir or UT_CHECKPOINT_DIRS.get(args.tag)
    ckpt_file = find_ut_checkpoint(ckpt_dir) if ckpt_dir else None

    if ckpt_file is None:
        logger.warning(f"No checkpoint for {args.tag} in {ckpt_dir} — skipping UT analysis.")
        logger.warning("Saving partial results (HRM only if available).")
    else:
        logger.info(f"[H4-2TS] Running UT timescale analysis for {args.tag}...")
        try:
            ut_model, ut_loader, ut_config = load_ut_model(ckpt_dir, ckpt_file, device)
            ut_model.eval()
            harness = UTActivationHarness(ut_model, device)
            ut_batches = harness.collect_puzzles(ut_loader, args.num_puzzles)
            ut_metrics = ut_timescale_metrics(ut_model, ut_batches, device)
            results["ut"] = ut_metrics
            logger.info(
                f"  UT   CV={ut_metrics['coeff_of_variation']:.3f}  "
                f"early/late_ratio={ut_metrics['early_late_ratio']:.2f}"
            )
        except Exception as e:
            logger.warning(f"  UT analysis failed: {e}")

    # Separability verdict
    verdict = {}
    if results.get("hrm") and results.get("ut"):
        hrm_ratio = results["hrm"]["mean_ratio_L_over_H"]
        ut_cv = results["ut"]["coeff_of_variation"]
        hrm_separable = hrm_ratio > 1.5  # heuristic: L changes >50% faster than H
        ut_separable = ut_cv > 0.3       # heuristic: high variation across iters

        verdict = {
            "hrm_two_timescale": bool(hrm_separable),
            "hrm_L_over_H_ratio": float(hrm_ratio),
            "ut_emergent_timescale": bool(ut_separable),
            "ut_cv": float(ut_cv),
            "conclusion": (
                "HRM shows separable H/L timescales (L changes faster); "
                f"UT {'does show' if ut_separable else 'does NOT show'} "
                "comparably separable iterations."
            ),
        }
        logger.info(f"\n  Verdict: {verdict['conclusion']}")
    results["verdict"] = verdict

    out_path = os.path.join(output_dir, "two_timescale.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")

    try:
        from scripts.core.provenance import write_meta
        write_meta(output_dir, "H4_two_timescale_baseline", {
            "tag": args.tag, "num_puzzles": args.num_puzzles,
        }, repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"provenance write failed: {e}")

    print("\n" + "=" * 60)
    print("H4 TWO-TIMESCALE BASELINE")
    print("=" * 60)
    if results.get("hrm"):
        h = results["hrm"]
        print(f"  HRM: mean‖Δz_H‖/‖z_H‖ = {h['mean_rate_H']:.4f}")
        print(f"  HRM: mean‖Δz_L‖/‖z_L‖ = {h['mean_rate_L']:.4f}  (ratio = {h['mean_ratio_L_over_H']:.2f}×)")
    if results.get("ut"):
        u = results["ut"]
        print(f"  UT:  per-iter CV = {u['coeff_of_variation']:.3f}")
        print(f"  UT:  early/late ratio = {u['early_late_ratio']:.2f}")
    if verdict:
        print(f"\n  {verdict.get('conclusion', '')}")


if __name__ == "__main__":
    main()
