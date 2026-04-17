"""scripts/batch_time_shift.py

E5-RUN experiment: Batch time-shift patching across multiple puzzles.

For each puzzle, runs the full forward pass (donor), then for each
(donor_step, recipient_step) pair, transplants z_H from the later step into
the earlier step and measures accuracy impact.

This tests the central claim: does z_H at step k encode richer solution
information than z_H at step j < k? If injecting step-10 z_H into step-2
boosts accuracy at step 2, the model is progressively refining z_H.

Output files:
  - results_per_puzzle.jsonl       (one JSON per puzzle per transfer pair)
  - aggregate_stats.json           (summary across all puzzles)
  - transfer_accuracy_matrix.json  (step-level accuracy grids)

Usage:
    python scripts/batch_time_shift.py \\
        --checkpoint Checkpoint_HRM_Sudoku/.../checkpoint.pt \\
        --transfer_pairs "10->2,12->4,8->2,14->6,4->10,2->8" \\
        --transfer_level H \\
        --max_steps 16 \\
        --max_puzzles 50 \\
        --output_dir results/time_shift \\
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
)
from scripts.core.activation_cross_step_transfer import CrossStepTransferer
from scripts.core.activation_patching import (
    ActivationCache,
    ActivationPatcher,
    compute_metrics,
    compute_diff_metrics,
    _normalize_patch_level,
    _parse_int_list_arg,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def parse_transfer_pairs(s: str) -> List[Tuple[int, int]]:
    """Parse "10->2,12->4" into [(10,2), (12,4)]."""
    pairs = []
    for part in s.split(","):
        part = part.strip()
        if "->" in part:
            donor, recip = part.split("->")
        elif ":" in part:
            donor, recip = part.split(":")
        else:
            raise ValueError(f"Invalid transfer pair: {part}. Use format donor->recipient")
        pairs.append((int(donor.strip()), int(recip.strip())))
    return pairs


def _logit_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    ent = -(probs * log_probs).sum(-1)
    return float(ent.mean().item())


def _cell_transitions(
    baseline_preds: torch.Tensor,
    transferred_preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_label_id: int = -100,
) -> Dict[str, int]:
    valid = labels != ignore_label_id
    base_ok = (baseline_preds == labels) & valid
    xfer_ok = (transferred_preds == labels) & valid
    return {
        "stayed_correct": int((base_ok & xfer_ok).sum().item()),
        "stayed_wrong": int((~base_ok & ~xfer_ok & valid).sum().item()),
        "fixed": int((~base_ok & xfer_ok).sum().item()),
        "broken": int((base_ok & ~xfer_ok).sum().item()),
        "total_changed": int(((baseline_preds != transferred_preds) & valid).sum().item()),
    }


# ─────────────────────────────────────────────────────────────
# Model loading
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
    parser = argparse.ArgumentParser(description="Batch Time-Shift Patching (E5-RUN)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--transfer_pairs", type=str,
                        default="10->2,12->4,8->2,14->6,4->10,2->8",
                        help="Comma-separated donor->recipient pairs")
    parser.add_argument("--transfer_level", type=str, default="H",
                        help="Which stream to transfer: H, L, or both")
    parser.add_argument("--max_steps", type=int, default=16,
                        help="Maximum ACT steps per forward pass")
    parser.add_argument("--max_puzzles", type=int, default=50,
                        help="Maximum number of puzzles (0=all)")
    parser.add_argument("--output_dir", type=str, default="results/time_shift")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    transfer_pairs = parse_transfer_pairs(args.transfer_pairs)
    transfer_level = _normalize_patch_level(args.transfer_level)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    # Validate: all donor steps must be < max_steps
    for d, r in transfer_pairs:
        if d >= args.max_steps:
            raise ValueError(f"donor_step {d} >= max_steps {args.max_steps}")
        if r >= args.max_steps:
            raise ValueError(f"recipient_step {r} >= max_steps {args.max_steps}")

    print("=" * 70)
    print("E5-RUN: Batch Time-Shift Patching")
    print("=" * 70)
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Transfer level:  {transfer_level}")
    print(f"Transfer pairs:  {transfer_pairs}")
    print(f"Max ACT steps:   {args.max_steps}")
    print(f"Max puzzles:     {args.max_puzzles if args.max_puzzles > 0 else 'all'}")
    print(f"Device:          {device}")
    print(f"Output dir:      {args.output_dir}")
    print("=" * 70)

    model_obj, test_loader, config = _load_model_and_dataloader(args.checkpoint, device)

    # CrossStepTransferer inherits from ActivationPatcher which has run_and_cache_activations
    transferer = CrossStepTransferer(model_obj, device=device)

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

    # Aggregation
    all_records: List[Dict] = []
    # pair_deltas["10->2"] = list of accuracy deltas
    pair_deltas: Dict[str, List[float]] = {f"{d}->{r}": [] for d, r in transfer_pairs}
    pair_fixed: Dict[str, List[int]] = {f"{d}->{r}": [] for d, r in transfer_pairs}
    pair_broken: Dict[str, List[int]] = {f"{d}->{r}": [] for d, r in transfer_pairs}
    # recipient step accuracy: how accurate is step r normally vs after injection?
    pair_recip_baseline: Dict[str, List[float]] = {f"{d}->{r}": [] for d, r in transfer_pairs}
    pair_recip_patched: Dict[str, List[float]] = {f"{d}->{r}": [] for d, r in transfer_pairs}

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

        # ---- Donor pass (baseline / full forward) ----
        donor_cache: Dict[int, ActivationCache] = {}
        donor_out = transferer.run_and_cache_activations(batch, donor_cache, max_steps=args.max_steps)
        if not donor_out:
            continue

        baseline_preds = donor_out["logits"].argmax(-1)
        baseline_metrics = compute_metrics(baseline_preds, labels)
        baseline_acc = baseline_metrics["accuracy"]

        # Per-step baseline accuracy
        baseline_step_acc = {}
        for s in sorted(donor_cache.keys()):
            baseline_step_acc[s] = compute_metrics(donor_cache[s].preds, labels)["accuracy"]

        # ---- Transfer for each pair ----
        pair_results: Dict[str, Dict] = {}

        for donor_step, recipient_step in transfer_pairs:
            pair_key = f"{donor_step}->{recipient_step}"

            # Ensure donor step exists in cache
            if donor_step not in donor_cache:
                pair_results[pair_key] = {"error": f"donor_step {donor_step} not in cache"}
                continue

            try:
                xfer_out, xfer_cache, xfer_info = transferer.run_with_cross_step_transfer(
                    batch,
                    donor_cache,
                    transfer_level=transfer_level,
                    donor_step=donor_step,
                    recipient_step=recipient_step,
                    transfer_positions=None,
                    max_steps=args.max_steps,
                )
            except Exception as e:
                pair_results[pair_key] = {"error": str(e)}
                continue

            if not xfer_out:
                continue

            xfer_preds = xfer_out["logits"].argmax(-1)
            xfer_metrics = compute_metrics(xfer_preds, labels)
            xfer_acc = xfer_metrics["accuracy"]

            transitions = _cell_transitions(baseline_preds, xfer_preds, labels)

            # Accuracy at the recipient step specifically
            recip_baseline_acc = baseline_step_acc.get(recipient_step, float("nan"))
            recip_patched_acc = float("nan")
            if recipient_step in xfer_cache:
                recip_patched_acc = compute_metrics(xfer_cache[recipient_step].preds, labels)["accuracy"]

            # Accuracy at the donor step in baseline (for comparison)
            donor_baseline_acc = baseline_step_acc.get(donor_step, float("nan"))

            pair_results[pair_key] = {
                "donor_step": donor_step,
                "recipient_step": recipient_step,
                "final_accuracy": xfer_acc,
                "delta_accuracy": xfer_acc - baseline_acc,
                "transitions": transitions,
                "recipient_baseline_acc": recip_baseline_acc,
                "recipient_patched_acc": recip_patched_acc,
                "recipient_delta": recip_patched_acc - recip_baseline_acc if not (
                    recip_patched_acc != recip_patched_acc
                ) else float("nan"),
                "donor_step_accuracy": donor_baseline_acc,
            }

            pair_deltas[pair_key].append(xfer_acc - baseline_acc)
            pair_fixed[pair_key].append(transitions["fixed"])
            pair_broken[pair_key].append(transitions["broken"])
            pair_recip_baseline[pair_key].append(recip_baseline_acc)
            pair_recip_patched[pair_key].append(recip_patched_acc)

        record = {
            "puzzle_idx": puzzle_idx,
            "baseline_accuracy": baseline_acc,
            "baseline_step_accuracies": baseline_step_acc,
            "transfer_results": pair_results,
        }
        all_records.append(record)
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()

        n_processed += 1
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0

        if n_processed % 5 == 0 or n_processed <= 3:
            summary_parts = [f"base={baseline_acc:.3f}"]
            for d, r in transfer_pairs[:3]:  # show first 3
                pk = f"{d}->{r}"
                delta = pair_results.get(pk, {}).get("delta_accuracy", float("nan"))
                summary_parts.append(f"{pk}:{delta:+.3f}")
            print(f"[{n_processed:4d}] puz={puzzle_idx:4d} | {' | '.join(summary_parts)} | {rate:.1f} puz/s")

    jsonl_file.close()
    elapsed = time.time() - t0
    print(f"\nProcessed {n_processed} puzzles in {elapsed:.1f}s ({n_processed/elapsed:.1f} puz/s)")

    if not all_records:
        print("No puzzles processed. Exiting.")
        return

    # ─── Aggregate statistics ───
    baseline_accs = [r["baseline_accuracy"] for r in all_records]
    agg: Dict[str, Any] = {
        "n_puzzles": n_processed,
        "transfer_level": transfer_level,
        "baseline": {
            "mean_accuracy": float(np.mean(baseline_accs)),
            "std_accuracy": float(np.std(baseline_accs)),
        },
        "transfer_pairs": {},
    }

    for d, r in transfer_pairs:
        pk = f"{d}->{r}"
        deltas = pair_deltas[pk]
        if not deltas:
            continue

        forward = d > r  # donor is later → injecting future info
        agg["transfer_pairs"][pk] = {
            "donor_step": d,
            "recipient_step": r,
            "direction": "forward (future→past)" if forward else "backward (past→future)",
            "n_puzzles": len(deltas),
            "mean_delta_accuracy": float(np.mean(deltas)),
            "std_delta_accuracy": float(np.std(deltas)),
            "median_delta_accuracy": float(np.median(deltas)),
            "mean_fixed_cells": float(np.mean(pair_fixed[pk])),
            "mean_broken_cells": float(np.mean(pair_broken[pk])),
            "n_puzzles_boosted": int(sum(1 for d_ in deltas if d_ > 0.01)),
            "n_puzzles_hurt": int(sum(1 for d_ in deltas if d_ < -0.01)),
            "n_puzzles_unchanged": int(sum(1 for d_ in deltas if abs(d_) <= 0.01)),
            "mean_recipient_baseline_acc": float(np.mean(pair_recip_baseline[pk])),
            "mean_recipient_patched_acc": float(np.mean(pair_recip_patched[pk])),
            "mean_recipient_delta": float(np.mean(pair_recip_patched[pk])) - float(np.mean(pair_recip_baseline[pk])),
        }

    agg_path = os.path.join(args.output_dir, "aggregate_stats.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate stats saved to {agg_path}")

    # ─── Summary table ───
    print("\n" + "=" * 90)
    print("SUMMARY: Time-Shift Transfer Results")
    print("=" * 90)
    print(f"{'Pair':<12} {'Direction':<22} {'Δ Final':>10} {'Δ@Recip':>10} "
          f"{'Fixed':>7} {'Broken':>7} {'Boost/Hurt':>12}")
    print("-" * 90)
    for d, r in transfer_pairs:
        pk = f"{d}->{r}"
        if pk not in agg["transfer_pairs"]:
            continue
        c = agg["transfer_pairs"][pk]
        print(
            f"{pk:<12} {c['direction']:<22} "
            f"{c['mean_delta_accuracy']:>+10.4f} "
            f"{c['mean_recipient_delta']:>+10.4f} "
            f"{c['mean_fixed_cells']:>7.1f} "
            f"{c['mean_broken_cells']:>7.1f} "
            f"{c['n_puzzles_boosted']}/{c['n_puzzles_hurt']:>10}"
        )
    print("=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
