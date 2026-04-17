"""scripts/batch_freeze_h.py

E2-FREEZE experiment: Freeze z_H after a specified ACT step.

For each puzzle, runs:
  1. Baseline forward (no intervention)
  2. Freeze-H at each specified freeze_at_step: cache z_H_out from step k,
     then force z_H = cached_z_H for ALL subsequent steps > k.

This tests whether z_H serves as a *static plan* set early and merely read by
later steps, or whether continuous z_H updates are essential for accuracy.

If freezing at step k loses little accuracy → z_H is a static sketch / plan.
If freezing at step k crushes accuracy → z_H is dynamically refined each step.

Output files:
  - results_per_puzzle.jsonl  (one JSON object per puzzle)
  - aggregate_stats.json      (summary across all puzzles)
  - freeze_accuracy_matrix.json  (per-step accuracy under each freeze condition)

Usage:
    python scripts/batch_freeze_h.py \\
        --checkpoint Checkpoint_HRM_Sudoku/.../checkpoint.pt \\
        --freeze_at_steps 0,1,2,4,8 \\
        --max_steps 16 \\
        --max_puzzles 100 \\
        --output_dir results/freeze_h \\
        --device cpu
"""

import os
import sys
import json
import time
import argparse
import yaml
from typing import Any, Dict, Optional, List, Tuple, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)
from models.hrm_v2.hrm_v2 import HierarchicalReasoningModel_V2
from scripts.core.activation_ablation import (
    ActivationAblator,
    ACTModel,
    _patch_attention_for_cpu,
    _make_inner_carry,
    _make_carry,
)
from scripts.core.activation_patching import (
    ActivationCache,
    compute_metrics,
    _normalize_patch_level,
    _parse_int_list_arg,
)


# ─────────────────────────────────────────────────────────────
# Freeze-H runner (extends ActivationAblator)
# ─────────────────────────────────────────────────────────────

class FreezeHRunner(ActivationAblator):
    """Forward pass that freezes z_H after a specified step.

    At step ``freeze_at``, we store z_H_out. For all steps > freeze_at,
    we replace the input z_H with that frozen snapshot before running the
    inner model. z_L continues to evolve normally.
    """

    def run_with_freeze(
        self,
        batch: Dict[str, torch.Tensor],
        freeze_at_step: int,
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache], Dict[str, Any]]:
        """Forward pass freezing z_H after ``freeze_at_step``.

        Returns: (final_outputs, cache, freeze_info)
        """
        frozen_cache: Dict[int, ActivationCache] = {}
        freeze_info: Dict[str, Any] = {
            "freeze_at_step": freeze_at_step,
            "frozen_z_H_norm": None,
            "steps_frozen": [],
        }

        carry = self._init_carry(batch)
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps

        all_outputs: List[Dict[str, torch.Tensor]] = []
        step = 0
        frozen_z_H: Optional[torch.Tensor] = None

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, _steps_in, _current_data = self._prepare_step_inputs(carry, batch)

                    # If we're past the freeze step and have a frozen z_H, inject it
                    if frozen_z_H is not None and step > freeze_at_step:
                        inner_in = _make_inner_carry(
                            self.model,
                            z_H=frozen_z_H.clone(),
                            z_L=inner_in.z_L,
                        )
                        freeze_info["steps_frozen"].append(step)

                    # Forward step
                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    # Capture z_H_out at the freeze step
                    if step == freeze_at_step:
                        frozen_z_H = new_carry.inner_carry.z_H.detach().clone()
                        freeze_info["frozen_z_H_norm"] = float(frozen_z_H.norm().item())

                    frozen_cache[step] = ActivationCache(
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
                    if step >= (max_steps or original_max_steps):
                        break
        finally:
            if max_steps is not None:
                self.model.config.halt_max_steps = original_max_steps

        return all_outputs[-1] if all_outputs else {}, frozen_cache, freeze_info


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def _logit_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    ent = -(probs * log_probs).sum(-1)
    return float(ent.mean().item())


def _cell_transitions(
    baseline_preds: torch.Tensor,
    frozen_preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_label_id: int = -100,
) -> Dict[str, int]:
    valid = labels != ignore_label_id
    base_ok = (baseline_preds == labels) & valid
    frz_ok = (frozen_preds == labels) & valid
    return {
        "stayed_correct": int((base_ok & frz_ok).sum().item()),
        "stayed_wrong": int((~base_ok & ~frz_ok & valid).sum().item()),
        "fixed": int((~base_ok & frz_ok).sum().item()),
        "broken": int((base_ok & ~frz_ok).sum().item()),
        "total_changed": int(((baseline_preds != frozen_preds) & valid).sum().item()),
    }


# ─────────────────────────────────────────────────────────────
# Model loading (reused from batch_ablation_1k.py pattern)
# ─────────────────────────────────────────────────────────────

def _load_model_and_dataloader(checkpoint_path: str, device: torch.device):
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

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_keys = set(model_full.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    m_pfx = any(k.startswith("_orig_mod.") for k in model_keys)
    c_pfx = any(k.startswith("_orig_mod.") for k in ckpt_keys)
    if m_pfx and not c_pfx:
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif c_pfx and not m_pfx:
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}

    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device)
    model_full.eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    # Unwrap to ACT model
    model_obj: Any = model_full
    if hasattr(model_obj, "_orig_mod"):
        model_obj = model_obj._orig_mod
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        if hasattr(model_obj, "model"):
            model_obj = model_obj.model
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        raise TypeError(f"Expected ACT model, got {type(model_obj)}")

    return model_obj, test_loader, config


def _extract_batch(item):
    if isinstance(item, (tuple, list)):
        if len(item) >= 2 and isinstance(item[1], dict):
            return item[1]
        if isinstance(item[0], dict):
            return item[0]
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported dataloader item: {type(item)}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Freeze-H Experiment (E2-FREEZE)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--freeze_at_steps", type=str, default="0,1,2,4,8",
                        help="Comma-separated ACT steps at which to freeze z_H")
    parser.add_argument("--max_steps", type=int, default=16,
                        help="Maximum ACT steps per forward pass")
    parser.add_argument("--max_puzzles", type=int, default=100,
                        help="Maximum number of puzzles to process (0=all)")
    parser.add_argument("--output_dir", type=str, default="results/freeze_h")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results_per_puzzle.jsonl")
    args = parser.parse_args()

    freeze_at_steps = [int(x) for x in args.freeze_at_steps.split(",")]
    freeze_at_steps = sorted(set(freeze_at_steps))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    print("=" * 70)
    print("E2-FREEZE: Freeze z_H After Step k")
    print("=" * 70)
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Freeze at steps: {freeze_at_steps}")
    print(f"Max ACT steps:   {args.max_steps}")
    print(f"Max puzzles:     {args.max_puzzles if args.max_puzzles > 0 else 'all'}")
    print(f"Device:          {device}")
    print(f"Output dir:      {args.output_dir}")
    print("=" * 70)

    model_obj, test_loader, config = _load_model_and_dataloader(args.checkpoint, device)
    runner = FreezeHRunner(model_obj, device=device)

    # Resume support
    done_puzzles: set = set()
    jsonl_path = os.path.join(args.output_dir, "results_per_puzzle.jsonl")
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                done_puzzles.add(rec["puzzle_idx"])
        print(f"Resuming: {len(done_puzzles)} puzzles already done")

    jsonl_file = open(jsonl_path, "a" if args.resume else "w")

    # Aggregation accumulators
    all_records: List[Dict] = []
    # step_acc_matrix[freeze_label][step] = list of accuracies across puzzles
    step_acc_matrix: Dict[str, Dict[int, List[float]]] = {"baseline": {}}
    for fs in freeze_at_steps:
        step_acc_matrix[f"freeze_at_{fs}"] = {}

    t0 = time.time()
    n_processed = 0

    for puzzle_idx, data in enumerate(test_loader):
        if args.max_puzzles > 0 and n_processed >= args.max_puzzles:
            break
        if puzzle_idx in done_puzzles:
            continue

        batch = _extract_batch(data)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        labels = batch["labels"]

        # ---- Baseline ----
        baseline_cache: Dict[int, ActivationCache] = {}
        baseline_out = runner.run_and_cache_activations(batch, baseline_cache, max_steps=args.max_steps)
        if not baseline_out:
            continue

        baseline_preds = baseline_out["logits"].argmax(-1)
        baseline_metrics = compute_metrics(baseline_preds, labels)
        baseline_final_acc = baseline_metrics["accuracy"]

        # Per-step baseline accuracy
        baseline_step_acc = {}
        for s in sorted(baseline_cache.keys()):
            acc = compute_metrics(baseline_cache[s].preds, labels)["accuracy"]
            baseline_step_acc[s] = acc
            step_acc_matrix["baseline"].setdefault(s, []).append(acc)

        # ---- Freeze conditions ----
        freeze_results: Dict[str, Dict] = {}

        for fs in freeze_at_steps:
            if fs >= args.max_steps:
                continue

            frozen_out, frozen_cache, freeze_info = runner.run_with_freeze(
                batch, freeze_at_step=fs, max_steps=args.max_steps,
            )
            if not frozen_out:
                continue

            frozen_preds = frozen_out["logits"].argmax(-1)
            frozen_metrics = compute_metrics(frozen_preds, labels)
            frozen_final_acc = frozen_metrics["accuracy"]

            # Per-step accuracy under freeze
            frozen_step_acc = {}
            for s in sorted(frozen_cache.keys()):
                acc = compute_metrics(frozen_cache[s].preds, labels)["accuracy"]
                frozen_step_acc[s] = acc
                label = f"freeze_at_{fs}"
                step_acc_matrix[label].setdefault(s, []).append(acc)

            # Cell transitions at final step
            transitions = _cell_transitions(baseline_preds, frozen_preds, labels)

            # Entropy comparison at final step
            base_ent = _logit_entropy(baseline_out["logits"])
            frz_ent = _logit_entropy(frozen_out["logits"])

            freeze_results[str(fs)] = {
                "freeze_at_step": fs,
                "final_accuracy": frozen_final_acc,
                "delta_accuracy": frozen_final_acc - baseline_final_acc,
                "transitions": transitions,
                "entropy_baseline": base_ent,
                "entropy_frozen": frz_ent,
                "frozen_z_H_norm": freeze_info["frozen_z_H_norm"],
                "num_steps_frozen": len(freeze_info["steps_frozen"]),
                "step_accuracies": frozen_step_acc,
            }

        record = {
            "puzzle_idx": puzzle_idx,
            "baseline_accuracy": baseline_final_acc,
            "baseline_step_accuracies": baseline_step_acc,
            "freeze_results": freeze_results,
        }
        all_records.append(record)
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()

        n_processed += 1
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        if n_processed % 10 == 0 or n_processed <= 3:
            print(
                f"[{n_processed:4d}] puzzle={puzzle_idx:4d} | "
                f"baseline={baseline_final_acc:.3f} | "
                + " | ".join(
                    f"frz@{fs}={freeze_results.get(str(fs), {}).get('delta_accuracy', float('nan')):+.3f}"
                    for fs in freeze_at_steps
                )
                + f" | {rate:.1f} puz/s"
            )

    jsonl_file.close()
    elapsed = time.time() - t0
    print(f"\nProcessed {n_processed} puzzles in {elapsed:.1f}s ({n_processed/elapsed:.1f} puz/s)")

    # ─── Aggregate statistics ───
    if not all_records:
        print("No puzzles processed. Exiting.")
        return

    baseline_accs = [r["baseline_accuracy"] for r in all_records]
    agg: Dict[str, Any] = {
        "n_puzzles": n_processed,
        "baseline": {
            "mean_accuracy": float(np.mean(baseline_accs)),
            "std_accuracy": float(np.std(baseline_accs)),
            "median_accuracy": float(np.median(baseline_accs)),
        },
        "freeze_conditions": {},
    }

    for fs in freeze_at_steps:
        deltas = [
            r["freeze_results"].get(str(fs), {}).get("delta_accuracy", float("nan"))
            for r in all_records
            if str(fs) in r.get("freeze_results", {})
        ]
        final_accs = [
            r["freeze_results"][str(fs)]["final_accuracy"]
            for r in all_records
            if str(fs) in r.get("freeze_results", {})
        ]
        broken_counts = [
            r["freeze_results"][str(fs)]["transitions"]["broken"]
            for r in all_records
            if str(fs) in r.get("freeze_results", {})
        ]

        if deltas:
            agg["freeze_conditions"][f"freeze_at_{fs}"] = {
                "mean_delta_accuracy": float(np.mean(deltas)),
                "std_delta_accuracy": float(np.std(deltas)),
                "median_delta_accuracy": float(np.median(deltas)),
                "mean_final_accuracy": float(np.mean(final_accs)),
                "mean_broken_cells": float(np.mean(broken_counts)),
                "n_puzzles_hurt": int(sum(1 for d in deltas if d < -0.01)),
                "n_puzzles_helped": int(sum(1 for d in deltas if d > 0.01)),
                "n_puzzles_unchanged": int(sum(1 for d in deltas if abs(d) <= 0.01)),
                "n_puzzles": len(deltas),
            }

    agg_path = os.path.join(args.output_dir, "aggregate_stats.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate stats saved to {agg_path}")

    # ─── Step accuracy matrix ───
    matrix: Dict[str, Dict[str, float]] = {}
    for label, step_data in step_acc_matrix.items():
        matrix[label] = {}
        for s, accs in sorted(step_data.items()):
            matrix[label][str(s)] = float(np.mean(accs))

    matrix_path = os.path.join(args.output_dir, "freeze_accuracy_matrix.json")
    with open(matrix_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Step accuracy matrix saved to {matrix_path}")

    # ─── Summary table ───
    print("\n" + "=" * 70)
    print("SUMMARY: Mean accuracy by freeze condition")
    print("=" * 70)
    print(f"{'Condition':<20} {'Mean Acc':>10} {'Δ Acc':>10} {'Broken':>10} {'Hurt/Helped':>14}")
    print("-" * 70)
    print(f"{'Baseline':<20} {agg['baseline']['mean_accuracy']:>10.4f} {'--':>10} {'--':>10} {'--':>14}")
    for fs in freeze_at_steps:
        key = f"freeze_at_{fs}"
        if key in agg["freeze_conditions"]:
            c = agg["freeze_conditions"][key]
            print(
                f"{'Freeze@step ' + str(fs):<20} "
                f"{c['mean_final_accuracy']:>10.4f} "
                f"{c['mean_delta_accuracy']:>+10.4f} "
                f"{c['mean_broken_cells']:>10.1f} "
                f"{c['n_puzzles_hurt']}/{c['n_puzzles_helped']:>12}"
            )
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
