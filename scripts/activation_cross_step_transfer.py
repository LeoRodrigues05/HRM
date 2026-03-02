"""scripts/activation_cross_step_transfer.py

Cross-step activation transfer experiment for the HRM ACT model.

Runs TWO independent forward passes of the **same puzzle**.
- Pass A: runs for a *donor_steps* number of ACT steps (the "advanced" pass).
- Pass B: runs for *recipient_steps* number of ACT steps (the "early" pass).

At step ``recipient_steps`` in Pass B, the activations of z_H or z_L (not both
at once by default, but configurable) are swapped with the corresponding
activations from Pass A at step ``donor_step`` (which is further along).

This tests whether information from a later reasoning step can accelerate or
improve an earlier reasoning step.

Example:
    Pass A has completed 12 steps; Pass B has completed 4 steps.
    Replace z_H at step 4 in Pass B with z_H from step 10 of Pass A.

Produces:
  - YAML results comparing baseline accuracy vs transferred accuracy
  - Colored HTML report (reuses the activation patching report infrastructure)

Usage:
    python scripts/activation_cross_step_transfer.py \\
        --checkpoint Checkpoint_HRM_v2_Sudoku/best.pt \\
        --puzzle_idx 0 \\
        --transfer_level H \\
        --donor_step 10 \\
        --recipient_step 4 \\
        --max_steps 12 \\
        --output_dir results/cross_step_transfer
"""

import os
import sys
import argparse
import yaml
from typing import Any, Dict, Optional, Tuple, List, cast
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)
from scripts.activation_patching import (
    ActivationPatcher,
    ActivationCache,
    compute_metrics,
    compute_diff_metrics,
    make_colored_html_report,
    _parse_int_list_arg,
    _normalize_patch_level,
)


# ---------------------------------------------------------------------------
# Cross-step transfer runner
# ---------------------------------------------------------------------------

class CrossStepTransferer(ActivationPatcher):
    """Run the same puzzle twice, transplanting activations from a later step
    into an earlier step of a second pass."""

    def run_with_cross_step_transfer(
        self,
        batch: Dict[str, torch.Tensor],
        donor_cache: Dict[int, ActivationCache],
        transfer_level: str = "H",            # "H", "L", or "both"
        donor_step: int = 10,                  # step from the donor (advanced) pass
        recipient_step: int = 4,               # step in the recipient (early) pass to inject
        transfer_positions: Optional[List[int]] = None,  # None = all positions
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache], Dict[int, Dict[str, float]]]:
        """Forward the same puzzle but at ``recipient_step`` swap activations
        from the donor pass at ``donor_step``.

        Only the specified ``transfer_level`` (H/L/both) is replaced.

        Returns: (final_outputs, cache, transfer_info)
        """
        if donor_step not in donor_cache:
            raise KeyError(
                f"donor_step={donor_step} not found in donor cache "
                f"(available steps: {sorted(donor_cache.keys())}). "
                "Increase --max_steps so the donor pass reaches that step."
            )

        transferred_cache: Dict[int, ActivationCache] = {}
        transfer_info: Dict[int, Dict[str, float]] = {}

        carry = self._init_carry(batch)
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps

        all_outputs: List[Dict[str, torch.Tensor]] = []
        step = 0

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, _steps_in, _current_data = self._prepare_step_inputs(carry, batch)

                    if step == recipient_step:
                        donor_act = donor_cache[donor_step]

                        # Pre-transfer diagnostics
                        info: Dict[str, float] = {
                            "donor_step": float(donor_step),
                            "recipient_step": float(recipient_step),
                        }

                        if transfer_level in ("H", "both"):
                            pre_diff_H = float((inner_in.z_H - donor_act.z_H).abs().max().item())
                            info["pre_diff_H"] = pre_diff_H

                            if transfer_positions is None:
                                z_H_new = donor_act.z_H.detach().clone()
                            else:
                                z_H_new = inner_in.z_H.clone()
                                for pos in transfer_positions:
                                    z_H_new[:, pos, :] = donor_act.z_H[:, pos, :].detach()
                            inner_in = HierarchicalReasoningModel_ACTV1InnerCarry(
                                z_H=z_H_new, z_L=inner_in.z_L,
                            )

                        if transfer_level in ("L", "both"):
                            pre_diff_L = float((inner_in.z_L - donor_act.z_L).abs().max().item())
                            info["pre_diff_L"] = pre_diff_L

                            if transfer_positions is None:
                                z_L_new = donor_act.z_L.detach().clone()
                            else:
                                z_L_new = inner_in.z_L.clone()
                                for pos in transfer_positions:
                                    z_L_new[:, pos, :] = donor_act.z_L[:, pos, :].detach()
                            inner_in = HierarchicalReasoningModel_ACTV1InnerCarry(
                                z_H=inner_in.z_H, z_L=z_L_new,
                            )

                        transfer_info[step] = info

                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    transferred_cache[step] = ActivationCache(
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

        return all_outputs[-1] if all_outputs else {}, transferred_cache, transfer_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-step activation transfer for HRM"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--puzzle_idx", type=int, default=0,
                        help="Puzzle index in the test set")
    parser.add_argument("--transfer_level", type=str, default="H",
                        help="Which stream to transfer: H, L, or both (default: H)")
    parser.add_argument("--donor_step", type=int, required=True,
                        help="ACT step from the *donor* (advanced) forward pass "
                             "whose activations will be transplanted")
    parser.add_argument("--recipient_step", type=int, required=True,
                        help="ACT step in the *recipient* (early) forward pass "
                             "where donor activations will be injected")
    parser.add_argument("--transfer_positions", type=str, default=None,
                        help="Comma-separated positions to transfer (default: all)")
    parser.add_argument("--max_steps", type=int, default=12,
                        help="Max ACT steps for every forward pass")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Repeated runs for stability")
    parser.add_argument("--output_dir", type=str,
                        default="results/cross_step_transfer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report_html", type=str,
                        default="cross_step_transfer_report.html")
    args = parser.parse_args()

    if args.num_runs <= 0:
        raise ValueError("--num_runs must be >= 1")
    if args.donor_step <= args.recipient_step:
        print(
            f"WARNING: donor_step ({args.donor_step}) <= recipient_step ({args.recipient_step}). "
            "Typically the donor should be further along than the recipient."
        )
    if args.donor_step >= args.max_steps:
        raise ValueError(
            f"donor_step ({args.donor_step}) >= max_steps ({args.max_steps}). "
            "Increase --max_steps."
        )

    args.transfer_level = _normalize_patch_level(args.transfer_level)
    os.makedirs(args.output_dir, exist_ok=True)
    transfer_positions = _parse_int_list_arg(args.transfer_positions)

    print("=" * 60)
    print("Cross-Step Activation Transfer Experiment")
    print("=" * 60)
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Puzzle index:     {args.puzzle_idx}")
    print(f"Transfer level:   {args.transfer_level}")
    print(f"Donor step:       {args.donor_step}")
    print(f"Recipient step:   {args.recipient_step}")
    print(f"Transfer positions: {transfer_positions if transfer_positions else 'all'}")
    print(f"Max steps:        {args.max_steps}")
    print(f"Num runs:         {args.num_runs}")
    print("=" * 60)

    # ---- Device / config / model loading (same as activation_patching.py) ----
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1,
        rank=0, world_size=1,
    )

    train_state = init_train_state(config, test_metadata, world_size=1)
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_keys = set(train_state.model.state_dict().keys())
    ckpt_keys = set(checkpoint.keys())
    m_pfx = any(k.startswith("_orig_mod.") for k in model_keys)
    c_pfx = any(k.startswith("_orig_mod.") for k in ckpt_keys)
    if m_pfx and not c_pfx:
        checkpoint = {f"_orig_mod.{k}": v for k, v in checkpoint.items()}
    elif c_pfx and not m_pfx:
        checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}

    train_state.model.load_state_dict(checkpoint, assign=True)
    train_state.model.to(device)
    train_state.model.eval()
    print("Model loaded successfully")

    model_obj: Any = train_state.model
    if hasattr(model_obj, "_orig_mod"):
        model_obj = model_obj._orig_mod
    if not isinstance(model_obj, HierarchicalReasoningModel_ACTV1) and hasattr(model_obj, "model"):
        model_obj = getattr(model_obj, "model")
    if not isinstance(model_obj, HierarchicalReasoningModel_ACTV1):
        raise TypeError(f"Expected HierarchicalReasoningModel_ACTV1, got {type(model_obj)}")
    model_obj = cast(HierarchicalReasoningModel_ACTV1, model_obj)

    transferer = CrossStepTransferer(model_obj, device=device)

    # ---- Helpers -------------------------------------------------------------
    def _extract_batch(item):
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

    # ---- Load puzzle ---------------------------------------------------------
    print(f"\nGathering puzzle {args.puzzle_idx} from test set...")
    puzzles = []
    for i, data in enumerate(test_loader):
        if i > args.puzzle_idx:
            break
        batch = _extract_batch(data)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        puzzles.append(batch)

    if len(puzzles) <= args.puzzle_idx:
        print(f"Error: Not enough puzzles. Found {len(puzzles)}, need idx {args.puzzle_idx}")
        return

    puzzle_batch = puzzles[args.puzzle_idx]
    print(f"Puzzle shape: {puzzle_batch['inputs'].shape}")

    # ---- Donor pass (full) ---------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Pass A (donor): running full {args.max_steps}-step forward pass...")
    print("=" * 60)
    donor_cache: Dict[int, ActivationCache] = {}
    donor_outputs = transferer.run_and_cache_activations(
        puzzle_batch, donor_cache, max_steps=args.max_steps,
    )
    donor_preds = donor_outputs["logits"].argmax(-1)
    donor_metrics = compute_metrics(donor_preds, puzzle_batch["labels"])
    print(f"Donor accuracy (step {max(donor_cache.keys())}): {donor_metrics['accuracy']:.4f}")

    # Show per-step accuracy of the donor for context
    for s in sorted(donor_cache.keys()):
        s_acc = compute_metrics(donor_cache[s].preds, puzzle_batch["labels"])["accuracy"]
        print(f"  donor step {s}: acc={s_acc:.4f}")

    # ---- Baseline pass (identical puzzle, same max_steps) --------------------
    print(f"\n{'=' * 60}")
    print("Pass B baseline (no transfer)...")
    print("=" * 60)
    baseline_run_metrics: List[Dict[str, float]] = []
    baseline_caches: List[Dict[int, ActivationCache]] = []
    baseline_outputs_list: List[Dict[str, torch.Tensor]] = []
    for run in range(args.num_runs):
        cache: Dict[int, ActivationCache] = {}
        outputs = transferer.run_and_cache_activations(
            puzzle_batch, cache, max_steps=args.max_steps,
        )
        preds = outputs["logits"].argmax(-1)
        m = compute_metrics(preds, puzzle_batch["labels"])
        baseline_run_metrics.append(m)
        baseline_caches.append(cache)
        baseline_outputs_list.append(outputs)
        print(f"  Baseline run {run}: acc={m['accuracy']:.4f}")

    baseline_outputs = baseline_outputs_list[0]
    baseline_preds = baseline_outputs["logits"].argmax(-1)
    baseline_metrics = baseline_run_metrics[0]
    baseline_cache = baseline_caches[0]

    # ---- Transferred pass ----------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Pass B with transfer: donor step {args.donor_step} → recipient step {args.recipient_step} ({args.transfer_level})...")
    print("=" * 60)
    transferred_run_metrics: List[Dict[str, float]] = []
    transferred_caches: List[Dict[int, ActivationCache]] = []
    transferred_outputs_list: List[Dict[str, torch.Tensor]] = []
    transfer_info_first: Dict[str, Dict[str, float]] = {}
    for run in range(args.num_runs):
        t_outputs, t_cache, t_info = transferer.run_with_cross_step_transfer(
            puzzle_batch,
            donor_cache,
            transfer_level=args.transfer_level,
            donor_step=args.donor_step,
            recipient_step=args.recipient_step,
            transfer_positions=transfer_positions,
            max_steps=args.max_steps,
        )
        preds = t_outputs["logits"].argmax(-1)
        m = compute_metrics(preds, puzzle_batch["labels"])
        transferred_run_metrics.append(m)
        transferred_caches.append(t_cache)
        transferred_outputs_list.append(t_outputs)
        if run == 0:
            transfer_info_first = {str(k): v for k, v in t_info.items()}
        print(f"  Transferred run {run}: acc={m['accuracy']:.4f}")

    transferred_outputs = transferred_outputs_list[0]
    transferred_preds = transferred_outputs["logits"].argmax(-1)
    transferred_metrics = transferred_run_metrics[0]
    transferred_cache = transferred_caches[0]

    accuracy_change = transferred_metrics["accuracy"] - baseline_metrics["accuracy"]
    print(f"\n{'=' * 60}")
    print("Impact Analysis")
    print("=" * 60)
    print(f"Baseline accuracy:     {baseline_metrics['accuracy']:.4f}")
    print(f"Transferred accuracy:  {transferred_metrics['accuracy']:.4f}")
    print(f"Accuracy change:       {accuracy_change:+.4f}")
    print(f"Donor step {args.donor_step} accuracy:  {compute_metrics(donor_cache[args.donor_step].preds, puzzle_batch['labels'])['accuracy']:.4f}")
    print(f"Recipient step {args.recipient_step} baseline accuracy: {compute_metrics(baseline_cache[args.recipient_step].preds, puzzle_batch['labels'])['accuracy']:.4f}")

    # ---- Stepwise comparison -------------------------------------------------
    labels = puzzle_batch["labels"]
    common_steps = sorted(set(baseline_cache.keys()) & set(transferred_cache.keys()))

    # The only "patched" step is recipient_step
    patched_steps_effective = [args.recipient_step]

    stepwise_metrics: Dict[str, Dict[str, object]] = {}
    step_outputs: Dict[str, Dict[str, object]] = {}
    for s in common_steps:
        base_preds_s = baseline_cache[s].preds
        trans_preds_s = transferred_cache[s].preds
        base_m = compute_metrics(base_preds_s, labels)
        trans_m = compute_metrics(trans_preds_s, labels)
        diff_m = compute_diff_metrics(base_preds_s, trans_preds_s, labels)
        stepwise_metrics[str(s)] = {
            "baseline_accuracy": base_m["accuracy"],
            "transferred_accuracy": trans_m["accuracy"],
            "delta_accuracy": float(trans_m["accuracy"] - base_m["accuracy"]),
            "changed_count": diff_m["changed_count"],
            "changed_indices_sample": diff_m["changed_indices_sample"],
        }
        # Store grid outputs for the recipient step and its neighbors
        if abs(s - args.recipient_step) <= 1:
            step_outputs[str(s)] = {
                "baseline_preds": base_preds_s[0].detach().cpu().tolist(),
                "patched_preds": trans_preds_s[0].detach().cpu().tolist(),
                "diff": diff_m,
            }

    for s in common_steps:
        sm = stepwise_metrics.get(str(s), {})
        marker = " <-- transfer" if s == args.recipient_step else ""
        print(
            f"Step {s}: Δacc={sm.get('delta_accuracy', 0):+.4f} | "
            f"changed={sm.get('changed_count', 0)}{marker}"
        )

    # ---- Save YAML -----------------------------------------------------------
    run_metrics = {
        "target_baseline_runs": baseline_run_metrics,
        "target_patched_runs": transferred_run_metrics,
    }
    results = {
        "experiment": "cross_step_activation_transfer",
        "config": {
            "checkpoint": args.checkpoint,
            "puzzle_idx": args.puzzle_idx,
            "transfer_level": args.transfer_level,
            "donor_step": args.donor_step,
            "recipient_step": args.recipient_step,
            "transfer_positions": transfer_positions,
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
        },
        "transfer_info": transfer_info_first,
        "metrics": {
            "baseline": baseline_metrics,
            "transferred": transferred_metrics,
            "accuracy_change": accuracy_change,
            "donor_final": donor_metrics,
        },
        "stepwise_metrics": stepwise_metrics,
        "step_outputs": step_outputs,
        "predictions": {
            "baseline": baseline_preds.cpu().numpy().tolist(),
            "transferred": transferred_preds.cpu().numpy().tolist(),
            "donor_final": donor_preds.cpu().numpy().tolist(),
        },
        "run_metrics": run_metrics,
    }

    yaml_path = os.path.join(
        args.output_dir,
        f"transfer_p{args.puzzle_idx}_{args.transfer_level}_d{args.donor_step}_r{args.recipient_step}.yaml",
    )
    with open(yaml_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {yaml_path}")

    # ---- HTML report ---------------------------------------------------------
    if args.report_html:
        report_path = os.path.join(args.output_dir, args.report_html)
        context = {
            "experiment": "Cross-Step Activation Transfer",
            "checkpoint": args.checkpoint,
            "puzzle_idx": args.puzzle_idx,
            "transfer_level": args.transfer_level,
            "donor_step": args.donor_step,
            "recipient_step": args.recipient_step,
            "transfer_positions": transfer_positions if transfer_positions else "all",
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
            "baseline_accuracy_run0": baseline_metrics["accuracy"],
            "transferred_accuracy_run0": transferred_metrics["accuracy"],
        }
        labels_flat = puzzle_batch["labels"][0].detach().cpu().tolist()
        input_flat = puzzle_batch["inputs"][0].detach().cpu().tolist()
        baseline_final_flat = baseline_preds[0].detach().cpu().tolist()
        transferred_final_flat = transferred_preds[0].detach().cpu().tolist()

        make_colored_html_report(
            report_path,
            context,
            labels_flat,
            input_flat,
            baseline_final_flat,
            transferred_final_flat,
            step_outputs,
            run_metrics,
            transfer_info_first,
            patched_steps_effective,
            source_input=None,
            source_labels=None,
        )
        print(f"HTML report saved to: {report_path}")

    # ---- Save caches ---------------------------------------------------------
    cache_path = os.path.join(
        args.output_dir,
        f"activations_transfer_p{args.puzzle_idx}.pt",
    )
    def _cache_to_dict(c):
        return {k: {
            "z_H": v.z_H.cpu(), "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(), "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(), "step": v.step,
        } for k, v in c.items()}
    torch.save({
        "donor": _cache_to_dict(donor_cache),
        "baseline": _cache_to_dict(baseline_cache),
        "transferred": _cache_to_dict(transferred_cache),
    }, cache_path)
    print(f"Activation caches saved to: {cache_path}")

    print(f"\n{'=' * 60}")
    print("Cross-step transfer experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
