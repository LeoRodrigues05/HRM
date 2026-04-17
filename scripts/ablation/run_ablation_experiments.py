"""scripts/run_ablation_experiments.py

Efficient multi-puzzle, multi-step ablation experiment runner.

Loads the model ONCE, pre-fetches all required puzzles, then runs ablation
for every (puzzle, step) combination.  Generates one HTML report per run.

Usage example (z_H ablation at single steps 4,6,8,10 for 5 puzzles):
    python scripts/run_ablation_experiments.py \
        --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt \
        --puzzle_idxs 572,243,118,777,333 \
        --ablate_level H \
        --ablate_steps 4,6,8,10 \
        --max_steps 16 \
        --output_dir results/ablation_zH_single_step
"""

import os
import sys
import argparse
import json
import time
import yaml
from typing import Any, Dict, List, Optional, Tuple, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
import torch.nn.functional as F

from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
)
from models.hrm_v2.hrm_v2 import (
    HierarchicalReasoningModel_V2,
)
from scripts.core.activation_ablation import (
    ActivationAblator,
    ACTModel,
    _patch_attention_for_cpu,
)
from scripts.core.activation_patching import (
    ActivationCache,
    compute_metrics,
    compute_diff_metrics,
    make_colored_html_report,
    _normalize_patch_level,
    _to_chars,
    _as_int_list,
)


def _extract_batch(item):
    """Pull the batch dict out of whatever the dataloader yields."""
    if isinstance(item, (tuple, list)):
        if len(item) == 3 and isinstance(item[1], dict):
            return item[1]
        if len(item) == 2 and isinstance(item[1], dict):
            return item[1]
        if len(item) == 1 and isinstance(item[0], dict):
            return item[0]
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported dataloader item: {type(item)}")


def load_model_and_data(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[Any, Any, Any]:
    """Load checkpoint, build model, create test dataloader.
    
    Returns (ablator, test_loader, test_metadata).
    """
    config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1,
        rank=0, world_size=1,
    )

    # Build model on target device
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=test_metadata.vocab_size,
        seq_len=test_metadata.seq_len,
        num_puzzle_identifiers=test_metadata.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw: nn.Module = model_cls(model_cfg)
        model_full = loss_head_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle _orig_mod prefix mismatch
    model_keys = set(model_full.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
    checkpoint_has_prefix = any(k.startswith("_orig_mod.") for k in checkpoint_keys)
    if model_has_prefix and not checkpoint_has_prefix:
        checkpoint = {f"_orig_mod.{k}": v for k, v in checkpoint.items()}
    elif checkpoint_has_prefix and not model_has_prefix:
        checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}

    model_full.load_state_dict(checkpoint, assign=True)
    model_full.to(device)
    model_full.eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    print("Model loaded successfully.")

    # Unwrap to ACT model
    model_obj: Any = model_full
    if hasattr(model_obj, "_orig_mod"):
        model_obj = model_obj._orig_mod
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)) \
            and hasattr(model_obj, "model"):
        model_obj = getattr(model_obj, "model")
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        raise TypeError(f"Expected ACTV1 or V2 model, got {type(model_obj)}")
    model_obj = cast(ACTModel, model_obj)

    ablator = ActivationAblator(model_obj, device=device)
    return ablator, test_loader, test_metadata


def prefetch_puzzles(
    test_loader,
    puzzle_idxs: List[int],
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Iterate through the dataloader once to grab all requested puzzles."""
    max_idx = max(puzzle_idxs)
    needed = set(puzzle_idxs)
    found: Dict[int, Dict[str, torch.Tensor]] = {}

    print(f"\nPre-fetching puzzles {sorted(needed)} (iterating up to index {max_idx})...")
    t0 = time.time()
    for i, data in enumerate(test_loader):
        if i > max_idx:
            break
        if i in needed:
            batch = _extract_batch(data)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            found[i] = batch
            print(f"  Loaded puzzle {i}  ({len(found)}/{len(needed)})")
    elapsed = time.time() - t0
    print(f"Pre-fetch done in {elapsed:.1f}s — loaded {len(found)}/{len(needed)} puzzles.\n")

    missing = needed - set(found.keys())
    if missing:
        print(f"WARNING: Could not find puzzles: {sorted(missing)}")
    return found


def run_single_experiment(
    ablator: ActivationAblator,
    puzzle_batch: Dict[str, torch.Tensor],
    puzzle_idx: int,
    ablate_level: str,
    ablate_step: Optional[int],      # None → all steps
    max_steps: int,
    output_dir: str,
    num_runs: int = 1,
) -> Dict[str, Any]:
    """Run one ablation experiment (single puzzle, single ablation config).
    
    Returns summary dict with key metrics.
    """
    ablate_steps_list = [ablate_step] if ablate_step is not None else None
    step_label = f"step{ablate_step}" if ablate_step is not None else "all_steps"
    run_label = f"p{puzzle_idx}_{ablate_level}_{step_label}"

    run_dir = os.path.join(output_dir, run_label)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"Experiment: puzzle={puzzle_idx}  ablate={ablate_level}  steps={step_label}  max_steps={max_steps}")
    print(f"{'─'*60}")

    # ── Baseline ──────────────────────────────────────────────
    baseline_run_metrics: List[Dict[str, float]] = []
    baseline_caches: List[Dict[int, ActivationCache]] = []
    baseline_outputs_list = []
    for run in range(num_runs):
        cache: Dict[int, ActivationCache] = {}
        outputs = ablator.run_and_cache_activations(puzzle_batch, cache, max_steps=max_steps)
        preds = outputs["logits"].argmax(-1)
        metrics = compute_metrics(preds, puzzle_batch["labels"])
        baseline_run_metrics.append(metrics)
        baseline_caches.append(cache)
        baseline_outputs_list.append(outputs)

    baseline_outputs = baseline_outputs_list[0]
    baseline_preds = baseline_outputs["logits"].argmax(-1)
    baseline_metrics = baseline_run_metrics[0]
    baseline_cache = baseline_caches[0]
    print(f"  Baseline: acc={baseline_metrics['accuracy']:.4f} "
          f"({baseline_metrics['correct']}/{baseline_metrics['total_positions']})")

    # ── Ablated ───────────────────────────────────────────────
    ablated_run_metrics: List[Dict[str, float]] = []
    ablated_caches: List[Dict[int, ActivationCache]] = []
    ablated_outputs_list = []
    ablation_info_first: Dict = {}
    for run in range(num_runs):
        abl_out, abl_cache, abl_info = ablator.run_with_ablation(
            puzzle_batch,
            ablate_level=ablate_level,
            ablate_steps=ablate_steps_list,
            max_steps=max_steps,
        )
        preds = abl_out["logits"].argmax(-1)
        metrics = compute_metrics(preds, puzzle_batch["labels"])
        ablated_run_metrics.append(metrics)
        ablated_caches.append(abl_cache)
        ablated_outputs_list.append(abl_out)
        if run == 0:
            ablation_info_first = {str(k): v for k, v in abl_info.items()}

    ablated_outputs = ablated_outputs_list[0]
    ablated_preds = ablated_outputs["logits"].argmax(-1)
    ablated_metrics = ablated_run_metrics[0]
    ablated_cache = ablated_caches[0]

    accuracy_change = ablated_metrics["accuracy"] - baseline_metrics["accuracy"]
    print(f"  Ablated:  acc={ablated_metrics['accuracy']:.4f} "
          f"({ablated_metrics['correct']}/{ablated_metrics['total_positions']})  "
          f"Δ={accuracy_change:+.4f}")

    # ── Stepwise metrics ──────────────────────────────────────
    labels = puzzle_batch["labels"]
    common_steps = sorted(set(baseline_cache.keys()) & set(ablated_cache.keys()))
    ablated_steps_effective = common_steps if ablate_steps_list is None else \
        [s for s in ablate_steps_list if s in common_steps]

    stepwise_metrics: Dict[str, Dict[str, object]] = {}
    step_outputs: Dict[str, Dict[str, object]] = {}
    for s in common_steps:
        base_preds_s = baseline_cache[s].preds
        abl_preds_s = ablated_cache[s].preds
        base_m = compute_metrics(base_preds_s, labels)
        abl_m = compute_metrics(abl_preds_s, labels)
        diff_m = compute_diff_metrics(base_preds_s, abl_preds_s, labels)
        stepwise_metrics[str(s)] = {
            "baseline_accuracy": base_m["accuracy"],
            "ablated_accuracy": abl_m["accuracy"],
            "delta_accuracy": float(abl_m["accuracy"] - base_m["accuracy"]),
            "changed_count": diff_m["changed_count"],
        }
        # Include step outputs for the ablated step and the step after
        if (s in ablated_steps_effective) or (ablate_step is not None and s == ablate_step + 1) or \
                ablate_steps_list is None or s == common_steps[-1]:
            step_outputs[str(s)] = {
                "baseline_preds": base_preds_s[0].detach().cpu().tolist(),
                "patched_preds": abl_preds_s[0].detach().cpu().tolist(),
                "diff": diff_m,
            }

    # ── Save YAML results ────────────────────────────────────
    results = {
        "experiment": "activation_ablation",
        "config": {
            "puzzle_idx": puzzle_idx,
            "ablate_level": ablate_level,
            "ablate_steps": ablate_steps_list,
            "ablation_mode": "single_step" if ablate_step is not None else "all_steps",
            "max_steps": max_steps,
            "num_runs": num_runs,
        },
        "ablation_info": ablation_info_first,
        "metrics": {
            "baseline": baseline_metrics,
            "ablated": ablated_metrics,
            "accuracy_change": accuracy_change,
        },
        "stepwise_metrics": stepwise_metrics,
        "step_outputs": step_outputs,
        "predictions": {
            "baseline": baseline_preds.cpu().numpy().tolist(),
            "ablated": ablated_preds.cpu().numpy().tolist(),
        },
        "run_metrics": {
            "target_baseline_runs": baseline_run_metrics,
            "target_patched_runs": ablated_run_metrics,
        },
    }
    yaml_path = os.path.join(run_dir, "results.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    # ── HTML report ───────────────────────────────────────────
    context = {
        "experiment": "Activation Ablation",
        "puzzle_idx": puzzle_idx,
        "ablation_target": f"z_{ablate_level}",
        "ablation_mode": f"single step {ablate_step}" if ablate_step is not None else "all steps",
        "max_steps": max_steps,
        "num_runs": num_runs,
        "baseline_accuracy": f"{baseline_metrics['accuracy']:.4f}",
        "ablated_accuracy": f"{ablated_metrics['accuracy']:.4f}",
        "accuracy_change": f"{accuracy_change:+.4f}",
    }
    labels_flat = puzzle_batch["labels"][0].detach().cpu().tolist()
    input_flat = puzzle_batch["inputs"][0].detach().cpu().tolist()
    baseline_final_flat = baseline_preds[0].detach().cpu().tolist()
    ablated_final_flat = ablated_preds[0].detach().cpu().tolist()

    report_path = os.path.join(run_dir, "report.html")
    make_colored_html_report(
        report_path,
        context,
        labels_flat,
        input_flat,
        baseline_final_flat,
        ablated_final_flat,
        step_outputs,
        results["run_metrics"],
        ablation_info_first,
        ablated_steps_effective,
        source_input=None,
        source_labels=None,
    )
    print(f"  Report → {report_path}")

    return {
        "puzzle_idx": puzzle_idx,
        "ablate_level": ablate_level,
        "ablate_step": ablate_step,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "ablated_accuracy": ablated_metrics["accuracy"],
        "accuracy_change": accuracy_change,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch ablation experiments — load model once, run many (puzzle × step) combos")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--puzzle_idxs", type=str, required=True,
                        help="Comma-separated puzzle indices (e.g. 572,243,118,777,333)")
    parser.add_argument("--ablate_level", type=str, default="H",
                        help="Which stream to ablate: H (z_H) or L (z_L)")
    parser.add_argument("--ablate_steps", type=str, default=None,
                        help="Comma-separated step indices for single-step ablation (e.g. 4,6,8,10). "
                             "Each step runs as an independent experiment. "
                             "Omit to run all-steps ablation only.")
    parser.add_argument("--max_steps", type=int, default=16,
                        help="Maximum reasoning steps (default: 16 = model native)")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Repeated forward passes per experiment (default: 1)")
    parser.add_argument("--output_dir", type=str, default="results/ablation_experiments",
                        help="Root output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")
    args = parser.parse_args()

    args.ablate_level = _normalize_patch_level(args.ablate_level)
    if args.ablate_level == "both":
        raise ValueError("Ablating 'both' z_H and z_L simultaneously is not supported. "
                         "Use --ablate_level H or --ablate_level L.")

    puzzle_idxs = [int(x.strip()) for x in args.puzzle_idxs.split(",")]
    ablate_steps: Optional[List[int]] = None
    if args.ablate_steps is not None:
        ablate_steps = [int(x.strip()) for x in args.ablate_steps.split(",")]

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
    ablator, test_loader, _ = load_model_and_data(args.checkpoint, device)

    # Pre-fetch all puzzles
    puzzles = prefetch_puzzles(test_loader, puzzle_idxs, device)

    # Build experiment grid
    experiments = []
    for pidx in puzzle_idxs:
        if pidx not in puzzles:
            print(f"SKIP puzzle {pidx} — not found in dataset")
            continue
        if ablate_steps is not None:
            # One experiment per (puzzle, step)
            for step in ablate_steps:
                experiments.append((pidx, step))
        else:
            # Single all-steps experiment per puzzle
            experiments.append((pidx, None))

    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"Running {total} ablation experiments")
    print(f"  Puzzles: {puzzle_idxs}")
    print(f"  Ablation target: z_{args.ablate_level}")
    print(f"  Steps: {ablate_steps if ablate_steps else 'all'}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}")

    results_summary: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx, (pidx, step) in enumerate(experiments):
        print(f"\n[{idx+1}/{total}]", end="")
        result = run_single_experiment(
            ablator,
            puzzles[pidx],
            pidx,
            ablate_level=args.ablate_level,
            ablate_step=step,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            num_runs=args.num_runs,
        )
        results_summary.append(result)

    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ALL {total} EXPERIMENTS COMPLETE  ({elapsed:.1f}s total)")
    print(f"{'='*60}")
    print(f"{'Puzzle':>8} {'Step':>6} {'Baseline':>10} {'Ablated':>10} {'Δ':>10}")
    print(f"{'─'*8} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for r in results_summary:
        step_str = str(r["ablate_step"]) if r["ablate_step"] is not None else "all"
        print(f"{r['puzzle_idx']:>8} {step_str:>6} "
              f"{r['baseline_accuracy']:>10.4f} {r['ablated_accuracy']:>10.4f} "
              f"{r['accuracy_change']:>+10.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "checkpoint": args.checkpoint,
                "ablate_level": args.ablate_level,
                "ablate_steps": ablate_steps,
                "max_steps": args.max_steps,
                "puzzle_idxs": puzzle_idxs,
            },
            "results": results_summary,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print(f"Reports saved → {args.output_dir}/p<idx>_<level>_step<N>/report.html")


if __name__ == "__main__":
    main()
