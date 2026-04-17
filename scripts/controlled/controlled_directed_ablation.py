"""scripts/controlled_directed_ablation.py

Controlled directed ablation experiment (Revised E9).

Projects out probe-learned directions from z_H/z_L and measures
causal impact with proper statistical controls:
  - 10 random-direction controls per probe direction
  - Paired t-test (probe vs random)
  - Bootstrap CIs
  - Cohen's d effect size
  - Multi-direction subspace ablation (2-d, 3-d)
  - Sudoku-specific violation counting

Requires E8 probe weights at results/probes/e8_constraint_probes/probe_weights.pt

Usage:
    python scripts/controlled_directed_ablation.py --num_puzzles 500
    python scripts/controlled_directed_ablation.py --quick
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from scipy import stats as scipy_stats

from scripts.controlled.controlled_common import (
    add_common_args, resolve_args,
    load_model_and_dataloader, collect_puzzles,
    bootstrap_ci, extract_batch,
)
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _make_inner_carry,
)
from scripts.core.activation_patching import compute_metrics
from scripts.directed_ablation.e9_directed_ablation import (
    DirectionalAblator,
    select_best_directions,
    count_violations,
    cell_accuracy,
    count_per_unit_broken,
    CONSTRAINT_DIRECTIONS,
    CONTROL_DIRECTION,
    SUDOKU_CELLS,
    SUDOKU_SIZE,
    DIGIT_OFFSET,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def cohens_d(a: List[float], b: List[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    a_arr, b_arr = np.array(a), np.array(b)
    pooled_std = np.sqrt(
        ((len(a_arr) - 1) * a_arr.std(ddof=1)**2 + (len(b_arr) - 1) * b_arr.std(ddof=1)**2)
        / (len(a_arr) + len(b_arr) - 2)
    )
    if pooled_std < 1e-10:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled_std)


def run_directional_ablation_for_puzzle(
    ablator: DirectionalAblator,
    batch: Dict[str, torch.Tensor],
    direction: torch.Tensor,
    z_level: str,
    max_steps: int,
) -> Dict[str, Any]:
    """Run baseline + directional ablation for one puzzle."""
    labels = batch["labels"]

    # Baseline
    base_cache: Dict[int, ActivationCache] = {}
    base_out = ablator.run_and_cache_activations(batch, base_cache, max_steps=max_steps)
    final_step = max(base_cache.keys())
    base_preds = base_cache[final_step].preds[:, -SUDOKU_CELLS:]
    targets_tok = labels[:, -SUDOKU_CELLS:]
    base_acc = cell_accuracy(base_preds, targets_tok)
    base_viols = count_violations(base_preds)

    # Directional ablation
    abl_out, abl_cache = ablator.run_with_directional_ablation(
        batch, direction,
        ablate_level=z_level,
        ablate_steps=None,
        max_steps=max_steps,
    )
    abl_final = max(abl_cache.keys())
    abl_preds = abl_cache[abl_final].preds[:, -SUDOKU_CELLS:]
    abl_acc = cell_accuracy(abl_preds, targets_tok)
    abl_viols = count_violations(abl_preds)
    broken_info = count_per_unit_broken(base_preds, abl_preds, targets_tok)

    return {
        "baseline_accuracy": base_acc,
        "ablated_accuracy": abl_acc,
        "delta_accuracy": abl_acc - base_acc,
        "baseline_viols": base_viols,
        "ablated_viols": abl_viols,
        "delta_row_viols": abl_viols["violated_rows"] - base_viols["violated_rows"],
        "delta_col_viols": abl_viols["violated_cols"] - base_viols["violated_cols"],
        "delta_box_viols": abl_viols["violated_boxes"] - base_viols["violated_boxes"],
        **broken_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Controlled Directed Ablation (Revised E9)"
    )
    add_common_args(parser)
    parser.add_argument("--num_puzzles", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results/controlled_directed_ablation")
    parser.add_argument(
        "--probe_weights", type=str,
        default="results/probes/e8_constraint_probes/probe_weights.pt",
        help="Path to probe weights from E8",
    )
    parser.add_argument("--z_level", type=str, default="H", choices=["H", "L"])
    parser.add_argument("--probe_step", type=int, default=None,
                        help="Step to select probes from (None=best)")
    parser.add_argument("--n_random_controls", type=int, default=10,
                        help="Random-direction controls per probe direction")
    args = resolve_args(parser.parse_args())

    if args.quick:
        args.num_puzzles = 20
        args.n_random_controls = 3

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("CONTROLLED DIRECTED ABLATION (Revised E9)")
    print("=" * 70)
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Probe weights:    {args.probe_weights}")
    print(f"  N puzzles:        {args.num_puzzles}")
    print(f"  N random controls:{args.n_random_controls}")
    print(f"  z_level:          {args.z_level}")
    print(f"  Device:           {device}")
    print(f"  Output:           {args.output_dir}")
    print("=" * 70)

    # Load probe weights
    if not os.path.exists(args.probe_weights):
        print(
            f"\nERROR: Probe weights not found at {args.probe_weights}\n"
            "Run E8 constraint probes first:\n"
            "  python scripts/e8_constraint_probes.py\n"
            "\nSkipping directed ablation experiment."
        )
        sys.exit(1)

    probe_weights = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
    print(f"Loaded {len(probe_weights)} probe weights")

    directions = select_best_directions(probe_weights, z_level=args.z_level, step=args.probe_step)
    if not directions:
        print("ERROR: No usable probe directions found.")
        sys.exit(1)

    hidden_dim = list(directions.values())[0].shape[0]

    # Generate random control directions
    rng = torch.Generator().manual_seed(args.seed)
    random_directions = {}
    for i in range(args.n_random_controls):
        rd = torch.randn(hidden_dim, generator=rng)
        rd = rd / rd.norm()
        random_directions[f"random_{i}"] = rd

    # Load model
    model_obj, test_loader, config = load_model_and_dataloader(args.checkpoint, device)
    ablator = DirectionalAblator(model_obj, device=device)

    puzzles = collect_puzzles(test_loader, device, args.num_puzzles, seed=args.seed)
    print(f"Collected {len(puzzles)} puzzles")

    # ── Run ablations ──────────────────────────────────────────────────
    all_directions = {}
    all_directions.update(directions)
    all_directions.update(random_directions)

    results_by_direction: Dict[str, List[Dict]] = {}

    t0 = time.time()
    for dir_name, direction in all_directions.items():
        print(f"\n  Ablating: {dir_name}")
        puzzle_results = []

        for i, (puzzle_idx, batch) in enumerate(puzzles):
            result = run_directional_ablation_for_puzzle(
                ablator, batch, direction, args.z_level, args.max_steps,
            )
            result["puzzle_idx"] = puzzle_idx
            puzzle_results.append(result)

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(puzzles)}")

        results_by_direction[dir_name] = puzzle_results
        mean_delta = np.mean([r["delta_accuracy"] for r in puzzle_results])
        print(f"    mean Δacc = {mean_delta:+.4f}")

    elapsed = time.time() - t0
    print(f"\nAll ablations done in {elapsed:.1f}s")

    # ── Statistical analysis ──────────────────────────────────────────
    analysis = {}
    random_pool_deltas = []
    for rname in random_directions:
        random_pool_deltas.extend(
            [r["delta_accuracy"] for r in results_by_direction[rname]]
        )

    for dir_name in directions:
        probe_deltas = [r["delta_accuracy"] for r in results_by_direction[dir_name]]

        # Per-puzzle paired comparison: probe vs average of random controls
        random_per_puzzle = []
        for pi in range(len(puzzles)):
            rvals = [results_by_direction[rn][pi]["delta_accuracy"] for rn in random_directions]
            random_per_puzzle.append(np.mean(rvals))

        # Paired t-test
        t_stat, p_value = scipy_stats.ttest_rel(probe_deltas, random_per_puzzle)

        # Cohen's d
        d = cohens_d(probe_deltas, random_per_puzzle)

        # Bootstrap CIs
        probe_ci = bootstrap_ci(probe_deltas)
        random_ci = bootstrap_ci(random_per_puzzle)

        # Violation analysis
        probe_viols = {
            "delta_row": bootstrap_ci([r["delta_row_viols"] for r in results_by_direction[dir_name]]),
            "delta_col": bootstrap_ci([r["delta_col_viols"] for r in results_by_direction[dir_name]]),
            "delta_box": bootstrap_ci([r["delta_box_viols"] for r in results_by_direction[dir_name]]),
        }

        analysis[dir_name] = {
            "probe_delta_accuracy": probe_ci,
            "random_control_delta_accuracy": random_ci,
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_value),
            "cohens_d": d,
            "significant_at_005": bool(p_value < 0.05),
            "significant_at_001": bool(p_value < 0.01),
            "violations": probe_viols,
        }

    # ── Multi-direction subspace ablation ─────────────────────────────
    constraint_dirs = [d for d in CONSTRAINT_DIRECTIONS if d in directions]
    if len(constraint_dirs) >= 2:
        print("\n  Running multi-direction subspace ablation...")
        for k in [2, 3]:
            if k > len(constraint_dirs):
                break
            combo_names = constraint_dirs[:k]
            combo_matrix = torch.stack([directions[n] for n in combo_names])  # [K, D]
            combo_label = f"subspace_{k}d_{'_'.join(combo_names[:2])}"

            sub_results = []
            for i, (puzzle_idx, batch) in enumerate(puzzles):
                labels = batch["labels"]
                base_cache: Dict[int, ActivationCache] = {}
                ablator.run_and_cache_activations(batch, base_cache, max_steps=args.max_steps)
                final_step = max(base_cache.keys())
                base_preds = base_cache[final_step].preds[:, -SUDOKU_CELLS:]
                targets_tok = labels[:, -SUDOKU_CELLS:]
                base_acc = cell_accuracy(base_preds, targets_tok)

                abl_out, abl_cache = ablator.run_with_directional_ablation(
                    batch, directions[combo_names[0]],
                    ablate_level=args.z_level,
                    ablate_steps=None,
                    max_steps=args.max_steps,
                    direction_matrix=combo_matrix,
                )
                abl_final = max(abl_cache.keys())
                abl_preds = abl_cache[abl_final].preds[:, -SUDOKU_CELLS:]
                abl_acc = cell_accuracy(abl_preds, targets_tok)
                sub_results.append(abl_acc - base_acc)

            analysis[combo_label] = {
                "directions": combo_names,
                "k_dimensions": k,
                "delta_accuracy": bootstrap_ci(sub_results),
            }
            print(f"    {combo_label}: Δacc = {np.mean(sub_results):+.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    # Per-direction puzzle-level results
    per_dir_path = os.path.join(args.output_dir, "per_direction_results.json")
    with open(per_dir_path, "w") as f:
        json.dump(
            results_by_direction, f, indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )

    # Analysis summary
    analysis_path = os.path.join(args.output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else
                                    bool(o) if isinstance(o, np.bool_) else str(o))

    print(f"\nResults saved to {args.output_dir}")

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("DIRECTED ABLATION: PROBE vs RANDOM CONTROLS")
    print("=" * 90)
    print(f"{'Direction':28s}  {'Probe Δacc':>12s}  {'Random Δacc':>12s}  "
          f"{'Cohen d':>8s}  {'p-value':>10s}  {'Sig?':>5s}")
    print("-" * 90)
    for dir_name, a in analysis.items():
        if "probe_delta_accuracy" not in a:
            continue
        p_d = a["probe_delta_accuracy"]
        r_d = a["random_control_delta_accuracy"]
        sig = "***" if a["significant_at_001"] else ("*" if a["significant_at_005"] else "")
        print(
            f"  {dir_name:26s}  {p_d['mean']:>+12.4f}  {r_d['mean']:>+12.4f}  "
            f"{a['cohens_d']:>8.3f}  {a['paired_p_value']:>10.6f}  {sig:>5s}"
        )

    # Multi-direction results
    for key, a in analysis.items():
        if key.startswith("subspace_"):
            d = a["delta_accuracy"]
            print(f"  {key:26s}  {d['mean']:>+12.4f}  {'':>12s}  {'':>8s}  {'':>10s}")


if __name__ == "__main__":
    main()
