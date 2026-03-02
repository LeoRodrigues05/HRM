#!/usr/bin/env python3
"""E9: Causal Validation via Directed Ablation.

Uses the probe weight vectors from E8 to ablate *specific directions*
in z_H space, then measures whether the ablation selectively impairs
the corresponding constraint type.

Methodology
-----------
For each binary probe direction w_target (e.g. w_row from the
violated_in_row probe):

    z_H' = z_H  −  (z_H · ŵ) ŵ          (project out the direction)

We apply this projection at every ACT step, then measure:
  - Row violation count change        → should increase for w_row
  - Col violation count change        → should be ~unchanged for w_row
  - Box violation count change        → should be ~unchanged for w_row
  - Overall cell accuracy change      → some decrease expected
  - Number of cells broken            → should cluster in affected constraint

This establishes *causal specificity*: if ablating the row direction
selectively breaks row consistency without hurting col/box, the direction
is genuinely used by the model for row constraint tracking.

Output
------
  results/e9_directed_ablation/
    ablation_results.json        – per-direction, per-puzzle metrics
    aggregate_results.json       – mean effects across puzzles
    specificity_matrix.json      – [direction × violation_type] Δ matrix

Usage
-----
    # Quick test
    python scripts/e9_directed_ablation.py --n_puzzles 5

    # Full run (after E8 has produced probe weights)
    python scripts/e9_directed_ablation.py --n_puzzles 200

    # Custom probe weights path
    python scripts/e9_directed_ablation.py --probe_weights results/e8_constraint_probes/probe_weights.pt
"""

import os
import sys
import json
import argparse
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)
from scripts.activation_ablation import (
    ActivationAblator, ActivationCache, _patch_attention_for_cpu,
    _make_inner_carry, _make_carry, ACTCarry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_SIZE = 9
SUDOKU_CELLS = 81
DIGIT_OFFSET = 1


# ═══════════════════════════════════════════════════════════════════════════
# Constraint violation counting
# ═══════════════════════════════════════════════════════════════════════════

def count_violations(preds_tok: torch.Tensor) -> Dict[str, int]:
    """Count row/col/box violations and cell accuracy info from predicted tokens.

    Args:
        preds_tok: [1, 81] or [81] predicted token IDs

    Returns:
        dict with violation counts
    """
    if preds_tok.ndim == 1:
        preds_tok = preds_tok.unsqueeze(0)
    B = preds_tok.shape[0]
    digits = (preds_tok.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    grid = digits.view(B, SUDOKU_SIZE, SUDOKU_SIZE)

    def _unit_violations(unit: torch.Tensor) -> int:
        """Count units (rows/cols/boxes) that contain duplicated nonzero digits."""
        # unit: [B, 9]  — we only support B=1 here
        u = unit[0]
        nonzero = u[u > 0]
        return (len(nonzero) - len(nonzero.unique())) if len(nonzero) > 0 else 0

    row_viol = sum(_unit_violations(grid[:, r, :]) for r in range(SUDOKU_SIZE))
    col_viol = sum(_unit_violations(grid[:, :, c]) for c in range(SUDOKU_SIZE))
    box_viol = 0
    for br in range(3):
        for bc in range(3):
            box = grid[:, br*3:(br+1)*3, bc*3:(bc+1)*3].reshape(B, 9)
            box_viol += _unit_violations(box)

    return {
        "violated_rows": row_viol,
        "violated_cols": col_viol,
        "violated_boxes": box_viol,
        "violated_total": row_viol + col_viol + box_viol,
    }


def cell_accuracy(preds_tok: torch.Tensor, targets_tok: torch.Tensor) -> float:
    """Fraction of cells correctly predicted."""
    return float((preds_tok.view(-1) == targets_tok.view(-1)).float().mean().item())


def count_per_unit_broken(
    baseline_preds: torch.Tensor,
    ablated_preds: torch.Tensor,
    targets_tok: torch.Tensor,
) -> Dict[str, int]:
    """Count cells that were correct in baseline but wrong after ablation,
    broken down by which unit they belong to (row/col/box)."""
    b_correct = (baseline_preds.view(-1) == targets_tok.view(-1))
    a_correct = (ablated_preds.view(-1) == targets_tok.view(-1))
    broken = b_correct & ~a_correct  # was right, now wrong

    fixed = ~b_correct & a_correct    # was wrong, now right

    return {
        "cells_broken": int(broken.sum().item()),
        "cells_fixed": int(fixed.sum().item()),
        "cells_changed": int((baseline_preds.view(-1) != ablated_preds.view(-1)).sum().item()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Directional ablation runner
# ═══════════════════════════════════════════════════════════════════════════

class DirectionalAblator(ActivationAblator):
    """Extends ActivationAblator to project out specific directions from z_H."""

    def run_with_directional_ablation(
        self,
        batch: Dict[str, torch.Tensor],
        direction: torch.Tensor,          # [D] unit vector
        ablate_level: str = "H",          # "H" or "L"
        ablate_steps: Optional[List[int]] = None,
        max_steps: Optional[int] = None,
        n_directions: int = 1,            # number of directions to project out (for subspace)
        direction_matrix: Optional[torch.Tensor] = None,  # [K, D] for multi-direction ablation
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache]]:
        """Forward pass with a single direction projected out of z_H (or z_L) at each step.

        z' = z - (z · d_hat) * d_hat   for each position in the sequence

        If direction_matrix [K, D] is supplied, the entire K-dim subspace is projected out.
        """
        cache: Dict[int, ActivationCache] = {}
        carry = self._init_carry(batch)

        original_max = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps

        # Normalise direction(s)
        if direction_matrix is not None:
            # Orthonormalise via QR
            Q, _ = torch.linalg.qr(direction_matrix.T.float())  # [D, K]
            proj_basis = Q.to(self.device)
        else:
            d_hat = direction.float().to(self.device)
            d_hat = d_hat / d_hat.norm().clamp(min=1e-8)
            proj_basis = d_hat.unsqueeze(1)  # [D, 1]

        all_outputs: List[Dict[str, torch.Tensor]] = []
        step = 0

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, steps_in, current_data = self._prepare_step_inputs(carry, batch)

                    should_ablate = (ablate_steps is None) or (step in ablate_steps)

                    if should_ablate:
                        if ablate_level in ("H", "both"):
                            z = inner_in.z_H.float()  # [B, T, D]
                            # Project out: z' = z - proj_basis @ proj_basis^T @ z
                            # proj_basis is [D, K]
                            coeffs = z @ proj_basis            # [B, T, K]
                            projection = coeffs @ proj_basis.T # [B, T, D]
                            z_new = (z - projection).to(inner_in.z_H.dtype)
                            inner_in = _make_inner_carry(self.model, z_H=z_new, z_L=inner_in.z_L)

                        if ablate_level in ("L", "both"):
                            z = inner_in.z_L.float()
                            coeffs = z @ proj_basis
                            projection = coeffs @ proj_basis.T
                            z_new = (z - projection).to(inner_in.z_L.dtype)
                            inner_in = _make_inner_carry(self.model, z_H=inner_in.z_H, z_L=z_new)

                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    cache[step] = ActivationCache(
                        z_H=inner_used.z_H.detach().clone(),
                        z_L=inner_used.z_L.detach().clone(),
                        step=step,
                        z_H_out=new_carry.inner_carry.z_H.detach().clone(),
                        z_L_out=new_carry.inner_carry.z_L.detach().clone(),
                        logits=outputs["logits"].detach().clone(),
                        preds=outputs["intermediate_preds_step"].detach().clone(),
                        q_halt_logits=outputs["q_halt_logits"].detach().clone(),
                        q_continue_logits=outputs["q_continue_logits"].detach().clone(),
                    )

                    all_outputs.append(outputs)
                    carry = new_carry
                    step += 1
                    if step >= (max_steps or original_max):
                        break
        finally:
            if max_steps is not None:
                self.model.config.halt_max_steps = original_max

        return all_outputs[-1] if all_outputs else {}, cache


# ═══════════════════════════════════════════════════════════════════════════
# Model loading (reuse from E8)
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data(device: torch.device):
    ckpt_dir = os.path.join(REPO_ROOT,
        "Checkpoint_HRM_Sudoku", "Checkpoint_HRM_Sudoku", "Checkpoint_HRM_Sudoku")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(os.path.join(ckpt_dir, "all_config.yaml")):
        config_path = os.path.join(ckpt_dir, "all_config.yaml")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_meta = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__, batch_size=1,
        vocab_size=test_meta.vocab_size, seq_len=test_meta.seq_len,
        num_puzzle_identifiers=test_meta.num_puzzle_identifiers, causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"),
                       map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(k.startswith("_orig_mod.") for k in ck):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(k.startswith("_orig_mod.") for k in mk):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, "model"):
        m = m.model

    return m, test_loader, test_meta


# ═══════════════════════════════════════════════════════════════════════════
# Select best probe direction per constraint target
# ═══════════════════════════════════════════════════════════════════════════

CONSTRAINT_DIRECTIONS = [
    "violated_in_row",
    "violated_in_col",
    "violated_in_box",
    "is_naked_single",
    "per_cell_correct",
]

CONTROL_DIRECTION = "is_given"  # should be orthogonal to dynamic constraints


def select_best_directions(
    probe_weights: dict,
    z_level: str = "H",
    step: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pick the best-scoring probe weight vector for each constraint target.

    If step is None, choose whichever step had the highest val_score for that target.

    Returns: dict  target_name → weight vector [D]
    """
    best: Dict[str, Tuple[float, torch.Tensor, int]] = {}  # target → (score, W, step)

    for key, val in probe_weights.items():
        if val["z_level"] != z_level:
            continue
        tgt = val["target"]
        if tgt not in CONSTRAINT_DIRECTIONS and tgt != CONTROL_DIRECTION:
            continue
        if step is not None and val["step"] != step:
            continue
        score = val["val_score"]
        W = val["W"]  # [out, in]
        # For binary probes W is [1, D]; for multi-class [C, D] — take the first row (or the norm direction)
        if W.shape[0] == 1:
            w_vec = W[0]
        else:
            # Use the first principal component of W
            U, S, _ = torch.svd(W.float())
            w_vec = (W.float().T @ U[:, 0])
            w_vec = w_vec / w_vec.norm().clamp(min=1e-8)

        s = val["step"]
        if tgt not in best or score > best[tgt][0]:
            best[tgt] = (score, w_vec, s)

    directions = {}
    for tgt, (score, w_vec, s) in best.items():
        directions[tgt] = w_vec
        logger.info(f"  Direction '{tgt}': step={s}, val_score={score:.4f}, ‖w‖={w_vec.norm():.4f}")

    return directions


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E9: Causal Validation via Directed Ablation")
    parser.add_argument("--probe_weights", type=str,
                        default="results/e8_constraint_probes/probe_weights.pt",
                        help="Path to probe weights from E8")
    parser.add_argument("--n_puzzles", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--z_level", type=str, default="H", choices=["H", "L"])
    parser.add_argument("--step", type=int, default=None,
                        help="Which step's probe to use. None = best across steps.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/e9_directed_ablation")
    # Random direction control
    parser.add_argument("--n_random_controls", type=int, default=3,
                        help="Number of random-direction ablations as baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ── Load probe weights ────────────────────────────────────────────
    if not os.path.exists(args.probe_weights):
        logger.error(f"Probe weights not found at {args.probe_weights}. Run E8 first.")
        sys.exit(1)

    probe_weights = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
    logger.info(f"Loaded {len(probe_weights)} probe weights from {args.probe_weights}")

    directions = select_best_directions(probe_weights, z_level=args.z_level, step=args.step)
    if not directions:
        logger.error("No usable directions found. Check E8 output.")
        sys.exit(1)

    # Infer hidden dimension from first direction
    hidden_dim = list(directions.values())[0].shape[0]

    # Add random control directions
    for i in range(args.n_random_controls):
        rand_dir = torch.randn(hidden_dim)
        rand_dir = rand_dir / rand_dir.norm()
        directions[f"random_control_{i}"] = rand_dir
        logger.info(f"  Direction 'random_control_{i}': random unit vector")

    # ── Load model + data ─────────────────────────────────────────────
    model, test_loader, test_meta = load_model_and_data(device)
    ablator = DirectionalAblator(model, device=device)

    # ── Collect puzzle batches ────────────────────────────────────────
    batches = []
    for i, data in enumerate(test_loader):
        if i >= args.n_puzzles:
            break
        if isinstance(data, (tuple, list)):
            batch = data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        else:
            batch = data
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batches.append(batch)
    logger.info(f"Loaded {len(batches)} puzzles")

    # ── Run baseline ──────────────────────────────────────────────────
    logger.info("Running baseline (no ablation)...")
    baselines = []
    for pi, batch in enumerate(batches):
        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=args.max_steps)
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

    # ── Run directional ablation for each direction ───────────────────
    all_results: Dict[str, List[dict]] = {}
    aggregate: Dict[str, dict] = {}

    t0 = time.time()
    for dir_name, direction in directions.items():
        logger.info(f"\nAblating direction: {dir_name}")
        puzzle_results = []

        for pi, batch in enumerate(batches):
            final_outputs, abl_cache = ablator.run_with_directional_ablation(
                batch, direction,
                ablate_level=args.z_level,
                ablate_steps=None,  # all steps
                max_steps=args.max_steps,
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

            if (pi + 1) % 50 == 0:
                logger.info(f"  {dir_name}: {pi+1}/{len(batches)}")

        all_results[dir_name] = puzzle_results

        # Aggregate
        n = len(puzzle_results)
        agg = {
            "direction": dir_name,
            "n_puzzles": n,
            "mean_delta_accuracy": np.mean([r["delta_accuracy"] for r in puzzle_results]),
            "std_delta_accuracy": np.std([r["delta_accuracy"] for r in puzzle_results]),
            "mean_delta_row_viols": np.mean([r["delta_row_viols"] for r in puzzle_results]),
            "mean_delta_col_viols": np.mean([r["delta_col_viols"] for r in puzzle_results]),
            "mean_delta_box_viols": np.mean([r["delta_box_viols"] for r in puzzle_results]),
            "mean_cells_broken": np.mean([r["cells_broken"] for r in puzzle_results]),
            "mean_cells_fixed": np.mean([r["cells_fixed"] for r in puzzle_results]),
            "n_puzzles_hurt": sum(1 for r in puzzle_results if r["delta_accuracy"] < -0.001),
            "n_puzzles_helped": sum(1 for r in puzzle_results if r["delta_accuracy"] > 0.001),
            "n_puzzles_unchanged": sum(1 for r in puzzle_results if abs(r["delta_accuracy"]) <= 0.001),
        }
        aggregate[dir_name] = agg
        logger.info(
            f"  {dir_name}: Δacc={agg['mean_delta_accuracy']:+.4f}, "
            f"Δrow={agg['mean_delta_row_viols']:+.2f}, "
            f"Δcol={agg['mean_delta_col_viols']:+.2f}, "
            f"Δbox={agg['mean_delta_box_viols']:+.2f}, "
            f"broken={agg['mean_cells_broken']:.1f}"
        )

    elapsed = time.time() - t0
    logger.info(f"\nAll ablations done in {elapsed:.1f}s")

    # ── Build specificity matrix ──────────────────────────────────────
    # Rows = direction ablated, Cols = violation type changed
    spec_matrix = {}
    for dir_name, agg in aggregate.items():
        spec_matrix[dir_name] = {
            "delta_accuracy": round(agg["mean_delta_accuracy"], 5),
            "delta_row_viols": round(agg["mean_delta_row_viols"], 4),
            "delta_col_viols": round(agg["mean_delta_col_viols"], 4),
            "delta_box_viols": round(agg["mean_delta_box_viols"], 4),
            "cells_broken": round(agg["mean_cells_broken"], 2),
        }

    # ── Save results ──────────────────────────────────────────────────
    # Per-puzzle results
    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Aggregate
    with open(os.path.join(args.output_dir, "aggregate_results.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Specificity matrix
    with open(os.path.join(args.output_dir, "specificity_matrix.json"), "w") as f:
        json.dump(spec_matrix, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("E9 DIRECTED ABLATION — SPECIFICITY MATRIX")
    print("=" * 90)
    print(f"{'Direction ablated':28s}  {'Δacc':>8s}  {'Δrow':>8s}  {'Δcol':>8s}  {'Δbox':>8s}  {'broken':>8s}")
    print("-" * 90)
    for name in CONSTRAINT_DIRECTIONS + [CONTROL_DIRECTION] + [f"random_control_{i}" for i in range(args.n_random_controls)]:
        if name not in spec_matrix:
            continue
        s = spec_matrix[name]
        # Highlight the "expected" column
        is_row = name == "violated_in_row"
        is_col = name == "violated_in_col"
        is_box = name == "violated_in_box"

        print(
            f"  {name:26s}  {s['delta_accuracy']:>+8.4f}  "
            f"{s['delta_row_viols']:>+8.3f}{'*' if is_row else ' '} "
            f"{s['delta_col_viols']:>+8.3f}{'*' if is_col else ' '} "
            f"{s['delta_box_viols']:>+8.3f}{'*' if is_box else ' '} "
            f"{s['cells_broken']:>8.1f}"
        )

    print("\n* = expected to be most affected by this direction")
    print(f"\nResults saved to {args.output_dir}")

    # ── Compute specificity score ─────────────────────────────────────
    # For row/col/box directions: specificity = |Δtarget| / (|Δrow| + |Δcol| + |Δbox|)
    print("\n── Specificity Scores ──")
    for tgt_name, viol_key in [
        ("violated_in_row", "delta_row_viols"),
        ("violated_in_col", "delta_col_viols"),
        ("violated_in_box", "delta_box_viols"),
    ]:
        if tgt_name not in spec_matrix:
            continue
        s = spec_matrix[tgt_name]
        total = abs(s["delta_row_viols"]) + abs(s["delta_col_viols"]) + abs(s["delta_box_viols"])
        if total > 0.001:
            specificity = abs(s[viol_key]) / total
            print(f"  {tgt_name}: specificity = {specificity:.3f}  "
                  f"(1.0 = perfectly selective, 0.33 = no selectivity)")
        else:
            print(f"  {tgt_name}: no measurable violation change")


if __name__ == "__main__":
    main()
