"""scripts/activation_ablation.py

Activation ablation experiment for the HRM ACT model.

Instead of replacing activations with those from another puzzle (patching),
this experiment *zeroes out* (ablates) the z_H or z_L activations at
specified ACT steps and observes the impact on model predictions.

This answers: "What happens if the H/L module contributes nothing at step N?"

Ablation targets (--ablate_level):
  - H : zero z_H only (high-level reasoning ablated)
  - L : zero z_L only (low-level reasoning ablated)

Ablation step modes (--ablate_steps):
  - Single-step:  supply ONE step index (e.g. "4").  The model is ablated at
                  that step only, then continues the forward pass normally.
  - All-steps:    omit the flag (default).  Ablation is applied at every step.

Produces:
  - YAML results comparing baseline vs ablated accuracy per step
  - Colored HTML report (reuses the activation patching report format)

Usage (single-step ablation of z_H at step 4):
    python scripts/activation_ablation.py \\
        --checkpoint Checkpoint_HRM_Sudoku/.../checkpoint.pt \\
        --puzzle_idx 572 \\
        --ablate_level H \\
        --ablate_steps 4 \\
        --max_steps 16 \\
        --output_dir results/activation_ablation

Usage (all-step ablation of z_L):
    python scripts/activation_ablation.py \\
        --checkpoint Checkpoint_HRM_Sudoku/.../checkpoint.pt \\
        --puzzle_idx 572 \\
        --ablate_level L \\
        --output_dir results/activation_ablation
"""

import os
import sys
import argparse
import yaml
from typing import Any, Dict, Optional, Tuple, List, cast, Union
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.layers import Attention, apply_rotary_pos_emb
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)
from models.hrm_v2.hrm_v2 import (
    HierarchicalReasoningModel_V2,
    HierarchicalReasoningModel_V2Carry,
    HierarchicalReasoningModel_V2InnerCarry,
)
from scripts.core.activation_patching import (
    ActivationCache,
    compute_metrics,
    compute_diff_metrics,
    make_colored_html_report,
    _parse_int_list_arg,
    _normalize_patch_level,
    _to_chars,
    _as_int_list,
)


# ---------------------------------------------------------------------------
# CPU-compatible attention (flash_attn is CUDA-only)
# ---------------------------------------------------------------------------

def _cpu_attention_forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for Attention.forward using PyTorch SDPA (CPU-safe)."""
    batch_size, seq_len, _ = hidden_states.shape
    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    query = qkv[:, :, :self.num_heads]
    key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
    value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
    if cos_sin is not None:
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
    # SDPA expects (B, num_heads, seq_len, head_dim)
    query = query.transpose(1, 2).float()
    key = key.transpose(1, 2).float()
    value = value.transpose(1, 2).float()
    attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=self.causal)
    attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype)
    attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
    return self.o_proj(attn_output)


def _patch_attention_for_cpu(model: nn.Module):
    """Replace flash_attn calls with SDPA for all Attention modules in the model."""
    import types
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(_cpu_attention_forward, module)


# Type aliases for v1/v2 compatibility
ACTModel = Union[HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2]
ACTCarry = Union[HierarchicalReasoningModel_ACTV1Carry, HierarchicalReasoningModel_V2Carry]
ACTInnerCarry = Union[HierarchicalReasoningModel_ACTV1InnerCarry, HierarchicalReasoningModel_V2InnerCarry]


def _make_inner_carry(model: ACTModel, z_H: torch.Tensor, z_L: torch.Tensor) -> ACTInnerCarry:
    """Create the correct InnerCarry type for the model version."""
    if isinstance(model, HierarchicalReasoningModel_V2):
        return HierarchicalReasoningModel_V2InnerCarry(z_H=z_H, z_L=z_L)
    return HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)


def _make_carry(model: ACTModel, inner_carry: ACTInnerCarry, steps: torch.Tensor,
                halted: torch.Tensor, current_data: Dict[str, torch.Tensor]) -> ACTCarry:
    """Create the correct Carry type for the model version."""
    if isinstance(model, HierarchicalReasoningModel_V2):
        return HierarchicalReasoningModel_V2Carry(
            inner_carry=inner_carry, steps=steps, halted=halted, current_data=current_data)
    return HierarchicalReasoningModel_ACTV1Carry(
        inner_carry=inner_carry, steps=steps, halted=halted, current_data=current_data)


# ---------------------------------------------------------------------------
# Ablation runner — standalone (works with both v1 and v2 models)
# ---------------------------------------------------------------------------

class ActivationAblator:
    """Run a forward pass with H/L activations ablated (zeroed) at specified steps.
    
    Works with both HRM v1 (ACTV1) and v2 models.
    """

    def __init__(self, model: ACTModel, device: torch.device = torch.device("cpu")):
        self.model = model
        self.device = device
        self._is_v2 = isinstance(model, HierarchicalReasoningModel_V2)

    @staticmethod
    def _bool_all(x: torch.Tensor) -> bool:
        return bool(torch.all(x).item())

    def _move_carry_to_device(self, carry: ACTCarry) -> ACTCarry:
        carry.inner_carry = _make_inner_carry(
            self.model,
            z_H=carry.inner_carry.z_H.to(self.device),
            z_L=carry.inner_carry.z_L.to(self.device),
        )
        carry.steps = carry.steps.to(self.device)
        carry.halted = carry.halted.to(self.device)
        carry.current_data = {k: v.to(self.device) for k, v in carry.current_data.items()}
        return carry

    def _init_carry(self, batch: Dict[str, torch.Tensor]) -> ACTCarry:
        carry = self.model.initial_carry(batch)
        return self._move_carry_to_device(carry)

    def _prepare_step_inputs(self, carry: ACTCarry, batch: Dict[str, torch.Tensor]):
        halted = carry.halted
        inner_carry = self.model.inner.reset_carry(halted, carry.inner_carry)
        steps = torch.where(halted, 0, carry.steps)
        current_data = {
            k: torch.where(
                halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }
        return inner_carry, steps, current_data

    def _forward_one_step(self, carry: ACTCarry, batch: Dict[str, torch.Tensor],
                          *, patched_inner_carry=None):
        inner_carry_in, steps_in, current_data = self._prepare_step_inputs(carry, batch)
        if patched_inner_carry is not None:
            inner_carry_in = patched_inner_carry

        step_index = int(steps_in.max().item())
        result = self.model.inner(
            inner_carry_in, current_data, probe_recorder=None, step_index=step_index,
        )

        if self._is_v2:
            new_inner_carry, logits, (q_halt_logits, q_continue_logits), _aux = result
        else:
            new_inner_carry, logits, (q_halt_logits, q_continue_logits) = result

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "intermediate_preds_step": logits.argmax(-1),
        }

        new_steps = steps_in + 1
        is_last_step = new_steps >= self.model.config.halt_max_steps
        halted = is_last_step
        new_carry = _make_carry(self.model, new_inner_carry, new_steps, halted, current_data)
        return new_carry, outputs, inner_carry_in

    def run_and_cache_activations(self, batch: Dict[str, torch.Tensor],
                                   cache_dict: Dict[int, ActivationCache],
                                   max_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        cache_dict.clear()
        carry = self._init_carry(batch)
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps

        all_outputs: List[Dict[str, torch.Tensor]] = []
        step = 0
        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    new_carry, outputs, inner_in = self._forward_one_step(carry, batch)
                    cache_dict[step] = ActivationCache(
                        z_H=inner_in.z_H.detach().clone(),
                        z_L=inner_in.z_L.detach().clone(),
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
        return all_outputs[-1] if all_outputs else {}

    def run_with_ablation(
        self,
        batch: Dict[str, torch.Tensor],
        ablate_level: str = "both",       # "H", "L", or "both"
        ablate_steps: Optional[List[int]] = None,  # None = all
        ablate_positions: Optional[List[int]] = None,  # None = all
        max_steps: Optional[int] = None,
        *,
        ablation_value: float = 0.0,      # value to fill (0 = zero ablation)
        ablation_mode: str = "zero",      # "zero" | "mean" | "resample"
        replacements: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache], Dict[int, Dict[str, float]]]:
        """Forward pass with activations replaced at the chosen steps.

        ``ablation_mode`` selects the replacement distribution — the point of
        review comment R2 is that plain zeroing is off-distribution, so we also
        support within-distribution controls:
          - "zero"     : fill with the constant ``ablation_value`` (default;
                         backwards-compatible with every existing caller).
          - "mean"     : fill with a mean activation vector supplied per
                         step/level via ``replacements[step][level]`` with a
                         shape broadcastable to the activation (e.g. [1, 1, D]).
          - "resample" : fill with a donor activation tensor (broadcastable to
                         the activation) via ``replacements[step][level]``.
        For "mean"/"resample" the caller precomputes ``replacements`` (see
        ``controlled_ablation.py``). Mirror of ``run_with_patching`` but the
        replacement source is a control tensor rather than another puzzle's
        cached activations.
        """
        if ablation_mode not in ("zero", "mean", "resample"):
            raise ValueError(f"ablation_mode must be zero|mean|resample, got {ablation_mode!r}")
        if ablation_mode != "zero" and replacements is None:
            raise ValueError(f"ablation_mode={ablation_mode!r} requires `replacements`")
        ablated_cache: Dict[int, ActivationCache] = {}
        ablation_info: Dict[int, Dict[str, float]] = {}

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

                    should_ablate = (ablate_steps is None) or (step in ablate_steps)

                    pre_norm_H: Optional[float] = None
                    pre_norm_L: Optional[float] = None

                    if should_ablate:
                        # Record pre-ablation activation norms for diagnostics
                        if ablate_level in ("H", "both"):
                            pre_norm_H = float(inner_in.z_H.norm().item())
                        if ablate_level in ("L", "both"):
                            pre_norm_L = float(inner_in.z_L.norm().item())

                        # Replacement tensor for one level under the active mode.
                        def _replacement(ref: torch.Tensor, level: str) -> torch.Tensor:
                            if ablation_mode == "zero":
                                return torch.full_like(ref, ablation_value)
                            step_rep = (replacements or {}).get(step, {})
                            if level not in step_rep:
                                raise ValueError(
                                    f"ablation_mode={ablation_mode!r} missing "
                                    f"replacements[{step}]['{level}']")
                            rep = step_rep[level].to(device=ref.device, dtype=ref.dtype)
                            if rep.shape != ref.shape:
                                rep = rep.expand_as(ref)
                            return rep.clone()

                        # Perform the ablation
                        if ablate_level in ("H", "both"):
                            rep_H = _replacement(inner_in.z_H, "H")
                            if ablate_positions is None:
                                z_H_new = rep_H
                            else:
                                z_H_new = inner_in.z_H.clone()
                                for pos in ablate_positions:
                                    z_H_new[:, pos, :] = rep_H[:, pos, :]
                            inner_in = _make_inner_carry(
                                self.model, z_H=z_H_new, z_L=inner_in.z_L,
                            )

                        if ablate_level in ("L", "both"):
                            rep_L = _replacement(inner_in.z_L, "L")
                            if ablate_positions is None:
                                z_L_new = rep_L
                            else:
                                z_L_new = inner_in.z_L.clone()
                                for pos in ablate_positions:
                                    z_L_new[:, pos, :] = rep_L[:, pos, :]
                            inner_in = _make_inner_carry(
                                self.model, z_H=inner_in.z_H, z_L=z_L_new,
                            )

                        ablation_info[step] = {
                            "pre_norm_H": float("nan") if pre_norm_H is None else pre_norm_H,
                            "pre_norm_L": float("nan") if pre_norm_L is None else pre_norm_L,
                            "ablation_value": ablation_value,
                            "ablation_mode": ablation_mode,
                        }

                    # Forward with (possibly ablated) inner_in
                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    ablated_cache[step] = ActivationCache(
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

        return all_outputs[-1] if all_outputs else {}, ablated_cache, ablation_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Activation Ablation for HRM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--puzzle_idx", type=int, default=0,
                        help="Index of the puzzle in the test set")
    parser.add_argument("--ablate_level", type=str, default="H",
                        help="Which stream to ablate: H (z_H) or L (z_L)")
    parser.add_argument("--ablate_steps", type=str, default=None,
                        help="Comma-separated ACT steps to ablate (default: all steps)")
    parser.add_argument("--ablate_positions", type=str, default=None,
                        help="Comma-separated sequence positions to ablate (default: all)")
    parser.add_argument("--ablation_value", type=float, default=0.0,
                        help="Value to fill ablated activations with (default: 0.0 = zero ablation)")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Maximum number of ACT reasoning steps")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Repeated forward passes for stability")
    parser.add_argument("--output_dir", type=str, default="results/activation_ablation",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--report_html", type=str, default="ablation_report.html",
                        help="HTML report filename (inside output_dir)")
    args = parser.parse_args()

    if args.num_runs <= 0:
        raise ValueError("--num_runs must be >= 1")

    args.ablate_level = _normalize_patch_level(args.ablate_level)
    os.makedirs(args.output_dir, exist_ok=True)

    ablate_steps = _parse_int_list_arg(args.ablate_steps)
    ablate_positions = _parse_int_list_arg(args.ablate_positions)

    if ablate_steps is not None:
        ablate_steps = sorted(set(ablate_steps))
    if ablate_positions is not None:
        ablate_positions = sorted(set(ablate_positions))

    print("=" * 60)
    print("Activation Ablation Experiment")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Puzzle index: {args.puzzle_idx}")
    print(f"Ablate level: {args.ablate_level}")
    print(f"Ablate steps: {ablate_steps if ablate_steps else 'all'}")
    print(f"Ablate positions: {ablate_positions if ablate_positions else 'all'}")
    print(f"Ablation value: {args.ablation_value}")
    print(f"Max steps: {args.max_steps}")
    print(f"Num runs: {args.num_runs}")
    print("=" * 60)

    # Device
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Load config
    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    # Dataloader
    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1,
        rank=0, world_size=1,
    )

    # Build model directly on target device (avoids hardcoded CUDA in create_model)
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
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
        model_full = loss_head_cls(model_raw, **config.arch.loss.__pydantic_extra__)  # type: ignore

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle _orig_mod prefix mismatch (torch.compile adds it)
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

    # Patch flash_attn → SDPA when running on CPU
    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    print("Model loaded successfully")

    # Unwrap to ACT model (supports both v1 and v2)
    model_obj: Any = model_full
    if hasattr(model_obj, "_orig_mod"):
        model_obj = model_obj._orig_mod
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)) and hasattr(model_obj, "model"):
        model_obj = getattr(model_obj, "model")
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        raise TypeError(f"Expected HierarchicalReasoningModel_ACTV1 or V2, got {type(model_obj)}")
    model_obj = cast(ACTModel, model_obj)

    ablator = ActivationAblator(model_obj, device=device)

    # -- helper ----------------------------------------------------------------
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

    # Load puzzle
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

    # ---- Baseline runs -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Running baseline (no ablation)...")
    print("=" * 60)
    baseline_run_metrics: List[Dict[str, float]] = []
    baseline_caches: List[Dict[int, ActivationCache]] = []
    baseline_outputs_list: List[Dict[str, torch.Tensor]] = []
    for run in range(args.num_runs):
        cache: Dict[int, ActivationCache] = {}
        outputs = ablator.run_and_cache_activations(puzzle_batch, cache, max_steps=args.max_steps)
        preds = outputs["logits"].argmax(-1)
        metrics = compute_metrics(preds, puzzle_batch["labels"])
        baseline_run_metrics.append(metrics)
        baseline_caches.append(cache)
        baseline_outputs_list.append(outputs)
        print(f"  Baseline run {run}: acc={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_positions']})")

    baseline_outputs = baseline_outputs_list[0]
    baseline_preds = baseline_outputs["logits"].argmax(-1)
    baseline_metrics = baseline_run_metrics[0]
    baseline_cache = baseline_caches[0]

    # ---- Ablated runs --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Running with ablation...")
    print("=" * 60)
    ablated_run_metrics: List[Dict[str, float]] = []
    ablated_caches: List[Dict[int, ActivationCache]] = []
    ablated_outputs_list: List[Dict[str, torch.Tensor]] = []
    ablation_info_first: Dict[int, Dict[str, float]] = {}
    for run in range(args.num_runs):
        abl_outputs, abl_cache, abl_info = ablator.run_with_ablation(
            puzzle_batch,
            ablate_level=args.ablate_level,
            ablate_steps=ablate_steps,
            ablate_positions=ablate_positions,
            max_steps=args.max_steps,
            ablation_value=args.ablation_value,
        )
        preds = abl_outputs["logits"].argmax(-1)
        metrics = compute_metrics(preds, puzzle_batch["labels"])
        ablated_run_metrics.append(metrics)
        ablated_caches.append(abl_cache)
        ablated_outputs_list.append(abl_outputs)
        if run == 0:
            ablation_info_first = {str(k): v for k, v in abl_info.items()}
        print(f"  Ablated run {run}: acc={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_positions']})")

    ablated_outputs = ablated_outputs_list[0]
    ablated_preds = ablated_outputs["logits"].argmax(-1)
    ablated_metrics = ablated_run_metrics[0]
    ablated_cache = ablated_caches[0]

    accuracy_change = ablated_metrics["accuracy"] - baseline_metrics["accuracy"]
    print(f"\n{'=' * 60}")
    print("Impact Analysis")
    print("=" * 60)
    print(f"Baseline accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"Ablated accuracy:   {ablated_metrics['accuracy']:.4f}")
    print(f"Accuracy change:    {accuracy_change:+.4f}")

    # ---- Stepwise comparison -------------------------------------------------
    labels = puzzle_batch["labels"]
    common_steps = sorted(set(baseline_cache.keys()) & set(ablated_cache.keys()))
    ablated_steps_effective = common_steps if ablate_steps is None else [s for s in ablate_steps if s in common_steps]

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
            "changed_indices_sample": diff_m["changed_indices_sample"],
        }
        if (s in ablated_steps_effective) or (s - 1 in ablated_steps_effective):
            step_outputs[str(s)] = {
                "baseline_preds": base_preds_s[0].detach().cpu().tolist(),
                "patched_preds": abl_preds_s[0].detach().cpu().tolist(),  # key kept as "patched_preds" for report compatibility
                "diff": diff_m,
            }

    # ---- Console stepwise report ---------------------------------------------
    for s in ablated_steps_effective:
        sm = stepwise_metrics.get(str(s), {})
        print(f"Step {s}: Δacc={sm.get('delta_accuracy', 0):+.4f} | changed={sm.get('changed_count', 0)}")

    # ---- Save results --------------------------------------------------------
    run_metrics = {
        "target_baseline_runs": baseline_run_metrics,
        "target_patched_runs": ablated_run_metrics,  # named for report compat
    }
    results = {
        "experiment": "activation_ablation",
        "config": {
            "checkpoint": args.checkpoint,
            "puzzle_idx": args.puzzle_idx,
            "ablate_level": args.ablate_level,
            "ablate_steps": ablate_steps,
            "ablate_positions": ablate_positions,
            "ablation_value": args.ablation_value,
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
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
        "run_metrics": run_metrics,
    }

    yaml_path = os.path.join(args.output_dir, f"ablation_p{args.puzzle_idx}_{args.ablate_level}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {yaml_path}")

    # ---- HTML report ---------------------------------------------------------
    if args.report_html:
        report_path = os.path.join(args.output_dir, args.report_html)
        context = {
            "experiment": "Activation Ablation",
            "checkpoint": args.checkpoint,
            "puzzle_idx": args.puzzle_idx,
            "ablate_level": args.ablate_level,
            "ablate_steps": ablate_steps if ablate_steps is not None else "all",
            "ablate_positions": ablate_positions if ablate_positions is not None else "all",
            "ablation_value": args.ablation_value,
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
            "baseline_accuracy_run0": baseline_metrics["accuracy"],
            "ablated_accuracy_run0": ablated_metrics["accuracy"],
        }
        labels_flat = puzzle_batch["labels"][0].detach().cpu().tolist()
        input_flat = puzzle_batch["inputs"][0].detach().cpu().tolist()
        baseline_final_flat = baseline_preds[0].detach().cpu().tolist()
        ablated_final_flat = ablated_preds[0].detach().cpu().tolist()

        # Reuse the patching report — "baseline" is unmodified, "patched" is ablated.
        # patch_validation maps step -> diffs; we pass ablation_info instead.
        make_colored_html_report(
            report_path,
            context,
            labels_flat,
            input_flat,
            baseline_final_flat,
            ablated_final_flat,
            step_outputs,
            run_metrics,
            ablation_info_first,
            ablated_steps_effective,
            source_input=None,
            source_labels=None,
        )
        print(f"HTML report saved to: {report_path}")

    # ---- Save activation caches ----------------------------------------------
    cache_path = os.path.join(args.output_dir, f"activations_ablation_p{args.puzzle_idx}.pt")
    torch.save({
        "baseline": {k: {
            "z_H": v.z_H.cpu(), "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(), "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(), "step": v.step,
        } for k, v in baseline_cache.items()},
        "ablated": {k: {
            "z_H": v.z_H.cpu(), "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(), "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(), "step": v.step,
        } for k, v in ablated_cache.items()},
    }, cache_path)
    print(f"Activation caches saved to: {cache_path}")

    print(f"\n{'=' * 60}")
    print("Ablation experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
