#!/usr/bin/env python3
"""Experiment H1 (Track A): Localizability scorecard for HRM checkpoints.

Computes four per-checkpoint metrics:
  1. probe_decodability   — mean val accuracy of constraint probes on z_H
  2. probe_causal_gap     — probe-direction Δacc minus random-direction Δacc
  3. min_causal_rank      — r*(0.9) from the H2 causal subspace tool
  4. subspace_linearity   — R² of linear fit to the PCA-top damage curve

Run on the current 1-step HRM:
    python scripts/analysis/localizability_scorecard.py --tag hrm_1step --device cuda

Run on an HRM-BPTT checkpoint when it arrives:
    python scripts/analysis/localizability_scorecard.py \\
        --tag hrm_bptt --checkpoint <path> --device cuda

Outputs: results/localizability/<tag>/scorecard.json
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
from scripts.directed_ablation.e9_directed_ablation import DirectionalAblator
from scripts.probes.e8_constraint_probes import (
    derive_per_cell_labels, train_binary, LinearProbe, puzzle_disjoint_split, BINARY_TARGETS,
)
from scripts.analysis.causal_subspace import (
    find_causal_subspace, _build_pca_pool, _get_baselines,
)
from scripts.analysis.policy_improvement import task_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

PROBE_TARGETS = ["violated_in_row", "violated_in_col", "violated_in_box", "per_cell_correct"]
PROBE_STEPS = [0, 4, 8, 12, 15]


# ---------------------------------------------------------------------------
# Activation collection (for probe training)
# ---------------------------------------------------------------------------

def collect_activations_for_probes(
    model, batches: list, device: torch.device,
    steps_to_record: List[int], max_steps: int,
    task: str,
) -> Dict[int, Dict[str, Any]]:
    """Returns {step: {"X": [N, D] float32, "labels": {target: [N]}}}.
    Only works for Sudoku (81-cell grid logic)."""
    ablator = ActivationAblator(model, device=device)
    pel = int(model.inner.puzzle_emb_len)

    # per-step accumulators
    step_features: Dict[int, List[torch.Tensor]] = {s: [] for s in steps_to_record}
    step_labels: Dict[int, Dict[str, List[torch.Tensor]]] = {s: {} for s in steps_to_record}

    for puzzle_idx, batch in batches:
        cache: Dict[int, Any] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)

        preds_last = cache[max(cache.keys())].preds[0]  # [seq]
        labels_1d = batch["labels"][0]                  # [seq]
        inputs_1d = batch["inputs"][0]                  # [seq]

        # Derive per-cell labels
        cell_labels = derive_per_cell_labels(
            preds_last.unsqueeze(0).cpu(),
            labels_1d.unsqueeze(0).cpu(),
            inputs_1d.unsqueeze(0).cpu(),
        )  # {target: [1, 81]}

        for s in steps_to_record:
            if s >= max_steps or s not in cache:
                continue
            z_out = cache[s].z_H_out[0, pel:].float().cpu()  # [seq, D] (81 cells for Sudoku)
            step_features[s].append(z_out)

            for tgt in PROBE_TARGETS:
                if tgt in cell_labels:
                    lbl = cell_labels[tgt][0].float().cpu()  # [81]
                    if tgt not in step_labels[s]:
                        step_labels[s][tgt] = []
                    step_labels[s][tgt].append(lbl)

    result = {}
    for s in steps_to_record:
        if not step_features[s]:
            continue
        X = torch.cat(step_features[s], dim=0)  # [N, D]
        lbls = {}
        for tgt in PROBE_TARGETS:
            if step_labels[s].get(tgt):
                lbls[tgt] = torch.cat(step_labels[s][tgt], dim=0)  # [N]
        result[s] = {"X": X, "labels": lbls}
    return result


# ---------------------------------------------------------------------------
# Metric 1: probe_decodability
# ---------------------------------------------------------------------------

def compute_probe_decodability(
    act_data: Dict[int, Dict[str, Any]],
    seeds: List[int],
    device: torch.device,
) -> Dict[str, Any]:
    """Train binary probes per step/target; return mean val accuracy."""
    all_accs: List[float] = []
    per_target: Dict[str, List[float]] = {t: [] for t in PROBE_TARGETS}

    for s, data in act_data.items():
        X = data["X"].to(device)
        n_rows = X.shape[0]
        for tgt in PROBE_TARGETS:
            if tgt not in data["labels"]:
                continue
            y = data["labels"][tgt].to(device)
            seed_accs = []
            for seed in seeds:
                tr_idx, va_idx = puzzle_disjoint_split(n_rows, val_frac=0.2, seed=seed, device=device)
                if len(tr_idx) < 5 or len(va_idx) < 5:
                    continue
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                _, tr_acc, va_acc = train_binary(X_tr, y_tr, X_va, y_va, epochs=50, lr=1e-2)
                seed_accs.append(va_acc)
            if seed_accs:
                mean_acc = float(np.mean(seed_accs))
                all_accs.append(mean_acc)
                per_target[tgt].append(mean_acc)

    return {
        "mean_val_acc": float(np.mean(all_accs)) if all_accs else float("nan"),
        "ci": bootstrap_ci(all_accs) if all_accs else {},
        "per_target": {t: float(np.mean(v)) if v else float("nan") for t, v in per_target.items()},
    }


# ---------------------------------------------------------------------------
# Metric 2: probe_causal_gap
# ---------------------------------------------------------------------------

def compute_probe_causal_gap(
    model, batches: list, device: torch.device,
    task: str, max_steps: int,
    act_data: Dict[int, Dict[str, Any]],
    seeds: List[int],
    n_random: int = 5,
    seed_random: int = 42,
) -> Dict[str, Any]:
    """
    Ablate each probe direction; compare mean Δacc to random unit directions.
    probe_causal_gap = mean(probe Δacc) - mean(random Δacc)
    ≈ 0 for current HRM (readable ≠ causal).
    """
    ablator = DirectionalAblator(model, device=device)
    # Baselines
    baseline_values = _get_baselines(model, batches, device, task, max_steps)

    def _ablation_delta(w: torch.Tensor) -> List[float]:
        """Single-direction ablation delta per puzzle."""
        deltas = []
        for puzzle_idx, batch in batches:
            final_out, cache = ablator.run_with_directional_ablation(
                batch,
                direction=w.to(device),
                ablate_level="H",
                ablate_steps=None,
                max_steps=max_steps,
                direction_matrix=None,
            )
            last_s = max(cache.keys())
            preds_row = cache[last_s].preds[0].cpu().numpy()
            labels_row = batch["labels"][0].cpu().numpy()
            inputs_row = batch["inputs"][0].cpu().numpy()
            tv = task_value(preds_row, labels_row, inputs_row, task)
            deltas.append(tv["value"] - baseline_values[puzzle_idx])
        return deltas

    # Probe direction ablations (use best step for each target)
    probe_deltas: List[float] = []
    # Pick best step (highest n, or last available)
    best_step = max(act_data.keys())
    X_best = act_data[best_step]["X"].to(device)
    D = X_best.shape[-1]

    for tgt in PROBE_TARGETS:
        if tgt not in act_data[best_step]["labels"]:
            continue
        # Train one probe to get its weight vector
        y = act_data[best_step]["labels"][tgt].to(device)
        n_rows = X_best.shape[0]
        tr_idx, va_idx = puzzle_disjoint_split(n_rows, val_frac=0.2, seed=seeds[0], device=device)
        probe, _, _ = train_binary(X_best[tr_idx], y[tr_idx], X_best[va_idx], y[va_idx],
                                   epochs=50, lr=1e-2)
        w_probe = probe.linear.weight[0].detach().float()  # [D]
        w_probe = w_probe / w_probe.norm().clamp(min=1e-8)
        deltas = _ablation_delta(w_probe)
        probe_deltas.extend(deltas)
        logger.info(f"  probe '{tgt}' Δacc = {np.mean(deltas):+.4f}")

    # Random direction ablations
    rng = torch.Generator()
    rng.manual_seed(seed_random)
    random_deltas: List[float] = []
    for _ in range(n_random):
        w_rand = F.normalize(torch.randn(D, generator=rng), dim=0).to(device)
        deltas = _ablation_delta(w_rand)
        random_deltas.extend(deltas)

    probe_mean = float(np.mean(probe_deltas)) if probe_deltas else float("nan")
    random_mean = float(np.mean(random_deltas)) if random_deltas else float("nan")
    gap = probe_mean - random_mean if not (np.isnan(probe_mean) or np.isnan(random_mean)) else float("nan")

    return {
        "probe_mean_delta": probe_mean,
        "probe_ci": bootstrap_ci(probe_deltas),
        "random_mean_delta": random_mean,
        "random_ci": bootstrap_ci(random_deltas),
        "probe_causal_gap": float(gap),
    }


# ---------------------------------------------------------------------------
# Main scorecard
# ---------------------------------------------------------------------------

def run_scorecard(
    model, batches: list, device: torch.device, task: str,
    probe_steps: List[int], seeds: List[int],
    ranks: List[int], max_steps: int,
    probe_weights_path: Optional[str] = None,
    sae_path: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Compute all four scorecard metrics for one checkpoint."""
    logger.info("[H1] Collecting activations for probe training...")
    act_data = collect_activations_for_probes(
        model, batches, device, probe_steps, max_steps, task
    )

    logger.info("[H1] Metric 1: probe_decodability...")
    probe_dec = compute_probe_decodability(act_data, seeds, device)
    logger.info(f"  probe_decodability = {probe_dec['mean_val_acc']:.4f}")

    logger.info("[H1] Metric 2: probe_causal_gap...")
    causal_gap = compute_probe_causal_gap(
        model, batches, device, task, max_steps, act_data, seeds, seed_random=seed
    )
    logger.info(f"  probe_causal_gap = {causal_gap['probe_causal_gap']:+.4f}")

    logger.info("[H1] Metrics 3+4: min_causal_rank + subspace_linearity (calls H2)...")
    h2_results = find_causal_subspace(
        model=model,
        batches=batches,
        device=device,
        task=task,
        ranks=ranks,
        n_random_bases=3,  # fewer for speed inside scorecard
        probe_weights_path=probe_weights_path,
        sae_path=sae_path,
        n_pca_puzzles=min(100, len(batches)),
        max_steps=max_steps,
        z_level="H",
        seed=seed,
    )
    min_causal_rank = h2_results["min_causal_rank"]
    subspace_linearity = h2_results["subspace_linearity"]
    logger.info(f"  min_causal_rank (r* 0.9) = {min_causal_rank}")
    logger.info(f"  subspace_linearity (R²)  = {subspace_linearity:.4f}")

    return {
        "probe_decodability": probe_dec,
        "probe_causal_gap": causal_gap,
        "min_causal_rank": int(min_causal_rank),
        "subspace_linearity": float(subspace_linearity),
        "delta_full": float(h2_results["delta_full"]),
        "baseline_mean": float(h2_results["baseline_mean"]),
        "h2_r_star": h2_results["r_star"],
        "h2_alignment": h2_results["alignment"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H1 Localizability Scorecard (HRM)")
    parser.add_argument("--tag", required=True,
                        help="Identifier e.g. hrm_1step or hrm_bptt")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint file (default: find_checkpoint())")
    parser.add_argument("--task", choices=["sudoku", "maze"], default="sudoku")
    parser.add_argument("--num_puzzles", type=int, default=300)
    parser.add_argument("--probe_steps", default="0,4,8,12,15")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = args.checkpoint or find_checkpoint()
    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "localizability", args.tag
    )
    os.makedirs(output_dir, exist_ok=True)

    probe_steps = [int(s) for s in args.probe_steps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]

    probe_path = os.path.join(REPO_ROOT, "results", "probes", "e8_constraint_probes", "probe_weights.pt")
    sae_path = os.path.join(REPO_ROOT, "results", "sae_study", "sae_d2048_l10.01.pt")

    logger.info(f"[H1] tag={args.tag}  ckpt={ckpt}  task={args.task}  device={device}")

    model, test_loader, config = load_model_and_dataloader(ckpt, device)
    model.eval()

    batches = collect_puzzles(test_loader, device, args.num_puzzles, seed=42)
    logger.info(f"[H1] Collected {len(batches)} puzzles")

    t0 = time.time()
    scorecard = run_scorecard(
        model=model,
        batches=batches,
        device=device,
        task=args.task,
        probe_steps=probe_steps,
        seeds=seeds,
        ranks=ranks,
        max_steps=args.max_steps,
        probe_weights_path=probe_path if os.path.exists(probe_path) else None,
        sae_path=sae_path if os.path.exists(sae_path) else None,
    )
    elapsed = time.time() - t0

    # Meta
    meta = {
        "tag": args.tag,
        "checkpoint": ckpt,
        "task": args.task,
        "num_puzzles": len(batches),
        "probe_steps": probe_steps,
        "seeds": seeds,
        "ranks": ranks,
        "max_steps": args.max_steps,
        "one_step_grad": getattr(getattr(model, "config", None), "one_step_grad", "unknown"),
        "hidden_size": getattr(getattr(model.inner, "config", None), "hidden_size", "unknown"),
        "elapsed_s": round(elapsed, 1),
    }

    output = {"meta": meta, "scorecard": scorecard}

    out_path = os.path.join(output_dir, "scorecard.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {out_path}")

    try:
        from scripts.core.provenance import write_meta
        write_meta(output_dir, f"H1_scorecard_{args.tag}", meta, repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"provenance write failed: {e}")

    print("\n" + "=" * 60)
    print(f"H1 LOCALIZABILITY SCORECARD — {args.tag}")
    print("=" * 60)
    sc = scorecard
    print(f"  probe_decodability  : {sc['probe_decodability']['mean_val_acc']:.4f}")
    print(f"  probe_causal_gap    : {sc['probe_causal_gap']['probe_causal_gap']:+.4f}  "
          f"(probe={sc['probe_causal_gap']['probe_mean_delta']:+.4f}, "
          f"random={sc['probe_causal_gap']['random_mean_delta']:+.4f})")
    print(f"  min_causal_rank r*  : {sc['min_causal_rank']}")
    print(f"  subspace_linearity  : {sc['subspace_linearity']:.4f}  (R², 1=linear, 0=knee)")
    print(f"  Δfull (z_H zeroing) : {sc['delta_full']:+.4f}")


if __name__ == "__main__":
    main()
