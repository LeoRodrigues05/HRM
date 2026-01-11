"""scripts/activation_patching.py

Activation patching for the HRM ACT model.

This runs 3 forwards:
1) Source puzzle baseline (captures activations per ACT step)
2) Target puzzle baseline
3) Target puzzle with activations patched from the source

Important implementation detail (accuracy-critical):
`HierarchicalReasoningModel_ACTV1.forward()` *resets* z_H/z_L for halted sequences at the
start of each step (via `inner.reset_carry(carry.halted, carry.inner_carry)`). Therefore
patching must be applied to the *post-reset* carry that is actually fed into the inner
model; patching `carry.inner_carry` directly at step 0 would otherwise be overwritten.

Usage:
    python scripts/activation_patching.py \
        --checkpoint <path_to_checkpoint> \
        --source_puzzle_idx 0 \
        --target_puzzle_idx 1 \
        --patch_level both \
        --patch_steps 0,1,2 \
        --output_dir results/activation_patching
"""

import os
import sys
import argparse
import yaml
from typing import Any, Dict, Optional, Tuple, List, cast
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1, 
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry
)


def _parse_int_list_arg(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _normalize_patch_level(s: str) -> str:
    """Normalize user-provided patch level names to one of: 'H', 'L', 'both'.

    Accepts common aliases like: z_h, z_l, h, l, both (case-insensitive).
    """
    raw = (s or "").strip()
    if raw == "":
        return "both"
    t = raw.lower().replace("-", "_")
    if t in {"h", "z_h", "zh", "z_h_only", "only_h"}:
        return "H"
    if t in {"l", "z_l", "zl", "z_l_only", "only_l"}:
        return "L"
    if t in {"both", "hl", "h_l", "h+l", "z_h_z_l", "z_h+z_l"}:
        return "both"
    # Allow already-canonical spelling
    if raw in {"H", "L", "both"}:
        return raw
    raise ValueError(f"Unknown --patch_level '{s}'. Use H, L, both (also accepts z_h / z_l).")


def _make_row_masked_sudoku_batch_from_labels(
    batch: Dict[str, torch.Tensor],
    *,
    missing_rows_1idx: List[int],
    blank_token_id: int = 1,
) -> Dict[str, torch.Tensor]:
    """Create a synthetic Sudoku batch whose inputs equal labels except blanked rows.

    - Uses batch['labels'] as the solution grid.
    - Sets rows in missing_rows_1idx (1..9) to blank_token_id in inputs.
    - Leaves batch['labels'] unchanged.
    """
    if "labels" not in batch or "inputs" not in batch:
        raise KeyError("Expected batch to contain 'inputs' and 'labels'")

    labels = batch["labels"]
    if labels.ndim != 2 or labels.shape[1] != 81:
        raise ValueError(f"Expected labels shape [B,81], got {tuple(labels.shape)}")

    # Copy tensors to avoid mutating the original dataloader batch.
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            # Non-tensor metadata (rare in these scripts) is passed through.
            out[k] = v

    inputs = labels.clone()
    inputs = torch.where(inputs == -100, torch.tensor(blank_token_id, device=inputs.device, dtype=inputs.dtype), inputs)

    rows0 = []
    for r in missing_rows_1idx:
        if r < 1 or r > 9:
            raise ValueError("missing_rows must be in 1..9")
        rows0.append(r - 1)

    for r0 in rows0:
        start = r0 * 9
        end = start + 9
        inputs[:, start:end] = blank_token_id

    out["inputs"] = inputs
    return out


@dataclass
class ActivationCache:
    # Activations *fed into* the inner model for this ACT step (i.e. post-reset, pre-forward)
    z_H: torch.Tensor
    z_L: torch.Tensor
    step: int
    # Activations after the inner forward (new carry)
    z_H_out: torch.Tensor
    z_L_out: torch.Tensor
    logits: torch.Tensor
    preds: torch.Tensor
    q_halt_logits: torch.Tensor
    q_continue_logits: torch.Tensor


class ActivationPatcher:
    """
    Handles activation patching between two puzzles.
    
    Captures activations from a source puzzle and injects them into a target puzzle
    at specified layers and timesteps.
    """
    
    def __init__(
        self, 
        model: HierarchicalReasoningModel_ACTV1,
        device: torch.device = torch.device("cuda")
    ):
        self.model = model
        self.device = device
        self.source_cache: Dict[int, ActivationCache] = {}
        self.target_cache: Dict[int, ActivationCache] = {}

    @staticmethod
    def _bool_all(x: torch.Tensor) -> bool:
        """Safe conversion of a bool tensor to a Python bool."""
        return bool(torch.all(x).item())

    def _move_carry_to_device(self, carry: HierarchicalReasoningModel_ACTV1Carry) -> HierarchicalReasoningModel_ACTV1Carry:
        carry.inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=carry.inner_carry.z_H.to(self.device),
            z_L=carry.inner_carry.z_L.to(self.device),
        )
        carry.steps = carry.steps.to(self.device)
        carry.halted = carry.halted.to(self.device)
        carry.current_data = {k: v.to(self.device) for k, v in carry.current_data.items()}
        return carry

    def _init_carry(self, batch: Dict[str, torch.Tensor]) -> HierarchicalReasoningModel_ACTV1Carry:
        # Create carry (model code creates CPU tensors by default) then move explicitly.
        # This avoids relying on any global/default-device context.
        carry = self.model.initial_carry(batch)
        return self._move_carry_to_device(carry)

    def _prepare_step_inputs(self,carry: HierarchicalReasoningModel_ACTV1Carry,batch: Dict[str, torch.Tensor],) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Dict[str, torch.Tensor]]:
        """Mirror ACTV1.forward() prelude (reset + current_data update).

        Returns the *post-reset* inner carry, the step counter tensor (pre-increment),
        and the current_data dict that will be fed to the inner model.
        """
        halted = carry.halted
        inner_carry = self.model.inner.reset_carry(halted, carry.inner_carry)
        steps = torch.where(halted, 0, carry.steps)

        current_data = {
            k: torch.where(
                halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }
        return inner_carry, steps, current_data

    def _forward_one_step(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
        *,
        patched_inner_carry: Optional[HierarchicalReasoningModel_ACTV1InnerCarry] = None,
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor], HierarchicalReasoningModel_ACTV1InnerCarry]:
        """Run a single ACT step, optionally overriding the post-reset inner carry.

        Returns (new_carry, outputs, inner_carry_in) where inner_carry_in is the
        activation state actually fed into the inner model.
        """
        if self.model.training:
            raise RuntimeError(
                "Activation patching expects model.eval(). "
                "Training-mode halting introduces stochasticity and changes semantics."
            )

        inner_carry_in, steps_in, current_data = self._prepare_step_inputs(carry, batch)
        if patched_inner_carry is not None:
            inner_carry_in = patched_inner_carry

        step_index = int(steps_in.max().item())
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.model.inner(
            inner_carry_in,
            current_data,
            probe_recorder=None,
            step_index=step_index,
        )

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "intermediate_preds_step": logits.argmax(-1),
        }

        new_steps = steps_in + 1
        is_last_step = new_steps >= self.model.config.halt_max_steps
        halted = is_last_step
        new_carry = HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=current_data,
        )
        return new_carry, outputs, inner_carry_in
        
    def run_and_cache_activations(
        self, 
        batch: Dict[str, torch.Tensor],
        cache_dict: Dict[int, ActivationCache],
        max_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run model forward pass and cache all intermediate activations.
        
        Args:
            batch: Input batch dictionary
            cache_dict: Dictionary to store activations at each step
            max_steps: Maximum number of steps to run (None = use model default)
            
        Returns:
            Final outputs from the model
        """
        cache_dict.clear()
        
        # Initialize carry on correct device
        carry = self._init_carry(batch)

        # Store/override halt_max_steps
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps
        
        all_outputs = []
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
            # Restore original setting
            if max_steps is not None:
                self.model.config.halt_max_steps = original_max_steps
        
        # Return final outputs
        return all_outputs[-1] if all_outputs else {}
    
    def run_with_patching(
        self,
        target_batch: Dict[str, torch.Tensor],
        source_activations: Dict[int, ActivationCache],
        patch_level: str = "both",  # "H", "L", or "both"
        patch_steps: Optional[List[int]] = None,  # Which steps to patch (None = all)
        patch_positions: Optional[List[int]] = None,  # Which positions to patch (None = all)
        max_steps: Optional[int] = None,
        *,
        verify: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache], Dict[int, Dict[str, float]]]:
        """
        Run model with activation patching from source to target.
        
        Args:
            target_batch: Target puzzle batch
            source_activations: Cached activations from source puzzle
            patch_level: Which level to patch - "H", "L", or "both"
            patch_steps: List of steps at which to apply patching (None = all steps)
            patch_positions: List of sequence positions to patch (None = all positions)
            max_steps: Maximum steps to run
            
        Returns:
            Tuple of (final outputs, cache of patched activations)
        """
        patched_cache: Dict[int, ActivationCache] = {}
        patch_validation: Dict[int, Dict[str, float]] = {}
        
        # Initialize carry on correct device
        carry = self._init_carry(target_batch)
        
        # Store original halt_max_steps
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps
        
        all_outputs = []
        step = 0
        warned_step0_noop = False
        
        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    # Prepare post-reset inner carry; apply patching to this state (so step-0 patching works).
                    inner_in, _steps_in, _current_data = self._prepare_step_inputs(carry, target_batch)
                    should_patch = (patch_steps is None) or (step in patch_steps)

                    pre_diff_H: Optional[float] = None
                    pre_diff_L: Optional[float] = None
                    post_diff_H: Optional[float] = None
                    post_diff_L: Optional[float] = None

                    if should_patch:
                        if step not in source_activations:
                            raise KeyError(
                                f"Requested patch at step={step}, but source cache has steps {sorted(source_activations.keys())}. "
                                "Ensure source/target are run with identical max_steps."
                            )
                        source_act = source_activations[step]

                        # Pre-patch diffs (max abs): if ~0, patching will have little/no effect.
                        if patch_level in ["H", "both"]:
                            if patch_positions is None:
                                pre_diff_H = float((inner_in.z_H - source_act.z_H).abs().max().item())
                            else:
                                pre_diff_H = float((inner_in.z_H[:, patch_positions, :] - source_act.z_H[:, patch_positions, :]).abs().max().item())
                        if patch_level in ["L", "both"]:
                            if patch_positions is None:
                                pre_diff_L = float((inner_in.z_L - source_act.z_L).abs().max().item())
                            else:
                                pre_diff_L = float((inner_in.z_L[:, patch_positions, :] - source_act.z_L[:, patch_positions, :]).abs().max().item())

                        # Validate shape compatibility (accuracy-critical; avoid silent truncation).
                        if patch_positions is None:
                            if patch_level in ["H", "both"] and inner_in.z_H.shape != source_act.z_H.shape:
                                raise ValueError(f"z_H shape mismatch: target {tuple(inner_in.z_H.shape)} vs source {tuple(source_act.z_H.shape)}")
                            if patch_level in ["L", "both"] and inner_in.z_L.shape != source_act.z_L.shape:
                                raise ValueError(f"z_L shape mismatch: target {tuple(inner_in.z_L.shape)} vs source {tuple(source_act.z_L.shape)}")
                        else:
                            if any(p < 0 for p in patch_positions):
                                raise ValueError("patch_positions must be >= 0")
                            max_pos = max(patch_positions) if patch_positions else -1
                            if max_pos >= inner_in.z_H.shape[1] or max_pos >= source_act.z_H.shape[1]:
                                raise ValueError(
                                    f"patch_positions contains {max_pos}, but sequence length is target={inner_in.z_H.shape[1]} source={source_act.z_H.shape[1]}"
                                )

                        if patch_level in ["H", "both"]:
                            if patch_positions is None:
                                inner_in = HierarchicalReasoningModel_ACTV1InnerCarry(
                                    z_H=source_act.z_H.detach().clone(),
                                    z_L=inner_in.z_L,
                                )
                            else:
                                z_H = inner_in.z_H.clone()
                                for pos in patch_positions:
                                    z_H[:, pos, :] = source_act.z_H[:, pos, :].detach()
                                inner_in = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=inner_in.z_L)

                        if patch_level in ["L", "both"]:
                            if patch_positions is None:
                                inner_in = HierarchicalReasoningModel_ACTV1InnerCarry(
                                    z_H=inner_in.z_H,
                                    z_L=source_act.z_L.detach().clone(),
                                )
                            else:
                                z_L = inner_in.z_L.clone()
                                for pos in patch_positions:
                                    z_L[:, pos, :] = source_act.z_L[:, pos, :].detach()
                                inner_in = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=inner_in.z_H, z_L=z_L)

                        # Post-patch diffs (max abs): should be ~0 on patched positions.
                        if patch_level in ["H", "both"]:
                            if patch_positions is None:
                                post_diff_H = float((inner_in.z_H - source_act.z_H).abs().max().item())
                            else:
                                post_diff_H = float((inner_in.z_H[:, patch_positions, :] - source_act.z_H[:, patch_positions, :]).abs().max().item())
                        if patch_level in ["L", "both"]:
                            if patch_positions is None:
                                post_diff_L = float((inner_in.z_L - source_act.z_L).abs().max().item())
                            else:
                                post_diff_L = float((inner_in.z_L[:, patch_positions, :] - source_act.z_L[:, patch_positions, :]).abs().max().item())

                        patch_validation[step] = {
                            "pre_diff_H": float("nan") if pre_diff_H is None else float(pre_diff_H),
                            "pre_diff_L": float("nan") if pre_diff_L is None else float(pre_diff_L),
                            "post_diff_H": float("nan") if post_diff_H is None else float(post_diff_H),
                            "post_diff_L": float("nan") if post_diff_L is None else float(post_diff_L),
                        }

                        # ACTV1 semantic warning:
                        # initial_carry() sets halted=True, so step 0 reset_carry() returns fixed H_init/L_init.
                        # That makes step-0 pre-forward activations essentially identical across puzzles.
                        if (not warned_step0_noop) and step == 0 and self._bool_all(carry.halted):
                            if ((pre_diff_H is None or pre_diff_H == 0.0) and (pre_diff_L is None or pre_diff_L == 0.0)):
                                print(
                                    "WARNING: step 0 patching is typically a no-op in ACTV1 because the post-reset carry is H_init/L_init for all puzzles. "
                                    "Use --patch_steps 1,2,... to patch puzzle-dependent states."
                                )
                                warned_step0_noop = True

                        if verify:
                            tol = 1e-6
                            if patch_level in ["H", "both"] and post_diff_H is not None and post_diff_H > tol:
                                raise AssertionError(f"Patch verification failed for z_H at step {step}: post_diff_H={post_diff_H}")
                            if patch_level in ["L", "both"] and post_diff_L is not None and post_diff_L > tol:
                                raise AssertionError(f"Patch verification failed for z_L at step {step}: post_diff_L={post_diff_L}")

                    new_carry, outputs, inner_used = self._forward_one_step(carry, target_batch, patched_inner_carry=inner_in)

                    patched_cache[step] = ActivationCache(
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
        
        return all_outputs[-1] if all_outputs else {}, patched_cache, patch_validation


def compute_metrics(
    predictions: torch.Tensor, 
    labels: torch.Tensor,
    ignore_label_id: int = -100
) -> Dict[str, float]:
    """Compute accuracy metrics for predictions"""
    # Mask out ignore labels
    valid_mask = (labels != ignore_label_id)
    
    if valid_mask.sum() == 0:
        return {"accuracy": 0.0, "total_positions": 0}
    
    correct = (predictions == labels) & valid_mask
    accuracy = correct.sum().item() / valid_mask.sum().item()
    
    return {
        "accuracy": accuracy,
        "correct": correct.sum().item(),
        "total_positions": valid_mask.sum().item()
    }


def compute_diff_metrics(
    baseline_preds: torch.Tensor,
    patched_preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_label_id: int = -100,
) -> Dict[str, object]:
    """Compare baseline vs patched predictions at a single step.

    Returns counts on valid (non-ignore) positions.
    """
    if baseline_preds.shape != patched_preds.shape:
        raise ValueError(f"Pred shape mismatch: {tuple(baseline_preds.shape)} vs {tuple(patched_preds.shape)}")
    if labels.shape != baseline_preds.shape:
        # In this repo labels are B x SeqLen; keep it strict for accuracy.
        raise ValueError(f"Label shape mismatch: labels {tuple(labels.shape)} vs preds {tuple(baseline_preds.shape)}")

    valid = labels != ignore_label_id
    changed = (baseline_preds != patched_preds) & valid
    changed_count = int(changed.sum().item())

    # For readability: return indices for batch 0 only (batch is always 1 in these scripts).
    idx0 = (changed[0].nonzero(as_tuple=False).view(-1).tolist() if changed.shape[0] > 0 else [])
    idx0_sample = idx0[:25]

    return {
        "changed_count": changed_count,
        "changed_indices_sample": idx0_sample,
    }


def sudoku_tokens_to_grid_str(tokens: torch.Tensor) -> str:
    """Render a 9x9 Sudoku grid from token IDs.

    Dataset encoding for Sudoku in this repo (see dataset/build_sudoku_dataset.py):
    - vocab: PAD(0) + "0".."9" (tokens 1..10)
    - We display digit 0 as '.'
    - Therefore token_id -> digit = token_id - 1
    """
    if torch.is_tensor(tokens):
        t = tokens.detach().to("cpu")
        if t.ndim == 2:
            t = t[0]
        t = t.to(torch.int64)
    else:
        raise TypeError("tokens must be a torch.Tensor")

    if t.numel() != 81:
        return f"<expected 81 tokens, got {t.numel()}>"

    def cell_str(token_id: int) -> str:
        if token_id <= 0:
            return "."
        digit = token_id - 1
        if digit == 0:
            return "."
        if 1 <= digit <= 9:
            return str(digit)
        return "?"

    rows = []
    for r in range(9):
        if r in (3, 6):
            rows.append("------+-------+------")
        parts = []
        for c in range(9):
            if c in (3, 6):
                parts.append("|")
            parts.append(cell_str(int(t[r * 9 + c].item())))
        # Add spaces between digits, keep separators tight.
        line = " ".join(parts).replace("| ", "| ")
        rows.append(line)
    return "\n".join(rows)


def visualize_puzzle(
    inputs: np.ndarray, 
    predictions: np.ndarray, 
    labels: np.ndarray,
    title: str = "Puzzle"
) -> str:
    """Create a simple text visualization of a puzzle"""
    lines = [f"\n{'='*60}", title, '='*60]
    
    if inputs.ndim == 2:
        inputs = inputs[0]
        predictions = predictions[0] if predictions.ndim == 2 else predictions
        labels = labels[0] if labels.ndim == 2 else labels
    
    lines.append(f"Input shape: {inputs.shape}")
    lines.append(f"Predictions: {predictions[:20]}...")  # First 20 tokens
    lines.append(f"Labels:      {labels[:20]}...")
    lines.append(f"Match: {(predictions == labels)[:20]}...")
    
    return "\n".join(lines)


def _digit_str(token_id: int) -> str:
    """Convert dataset token id to human-readable digit."""
    if token_id <= 0:
        return "."
    digit = token_id - 1
    if digit == 0:
        return "."
    if 1 <= digit <= 9:
        return str(digit)
    return "?"


def _id2num(i: int) -> str:
    """Match the repo's Sudoku report token mapping.

    - tokens 2..10 -> '1'..'9'
    - everything else -> '.'
    """
    if 2 <= i <= 10:
        return str(i - 1)
    return "."


def _to_chars(vec: List[int]) -> List[str]:
    return [_id2num(int(x)) for x in vec]


def _as_int_list(x: Any) -> List[int]:
    """Best-effort conversion to a list[int] for report rendering."""
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().to("cpu").view(-1).to(torch.int64).tolist()]
    raise TypeError(f"Unsupported value type for int list conversion: {type(x)}")


def _to_rows(vec81: List[str]) -> List[List[str]]:
    return [vec81[r * 9 : (r + 1) * 9] for r in range(9)]


def _compute_violations(chars81: List[str]) -> List[bool]:
    """True at i if that filled cell violates row/col/box uniqueness."""
    viol = [False] * 81

    # rows
    for r in range(9):
        vals = [chars81[r * 9 + c] for c in range(9) if chars81[r * 9 + c] != "."]
        for c in range(9):
            ch = chars81[r * 9 + c]
            if ch != "." and vals.count(ch) > 1:
                viol[r * 9 + c] = True

    # cols
    for c in range(9):
        col_vals = [chars81[r * 9 + c] for r in range(9) if chars81[r * 9 + c] != "."]
        for r in range(9):
            ch = chars81[r * 9 + c]
            if ch != "." and col_vals.count(ch) > 1:
                viol[r * 9 + c] = True

    # boxes
    for br in range(3):
        for bc in range(3):
            idxs = [(br * 3 + rr) * 9 + (bc * 3 + cc) for rr in range(3) for cc in range(3)]
            box_vals = [chars81[i] for i in idxs if chars81[i] != "."]
            for i in idxs:
                ch = chars81[i]
                if ch != "." and box_vals.count(ch) > 1:
                    viol[i] = True

    return viol


def _compute_classes(curr_chars: List[str], given_chars: List[str], prev_chars: Optional[List[str]] = None) -> List[str]:
    """Match sudoku_report_colored_ai.py / activation_patching_sudoku_report.py coloring.

    Priority:
      changed_ok / changed_bad (half yellow) > given (blue) > ok/bad > blank

    "changed" is defined relative to prev_chars (when provided).
    """
    viol = _compute_violations(curr_chars)
    classes: List[str] = []
    for i in range(81):
        ch = curr_chars[i]
        given = (given_chars[i] != ".")
        changed = (prev_chars is not None and ch != prev_chars[i] and ch != ".")

        if changed:
            cls = "changed_ok" if not viol[i] else "changed_bad"
        elif given:
            cls = "given"
        elif ch == ".":
            cls = "blank"
        else:
            cls = "ok" if not viol[i] else "bad"
        classes.append(cls)

    return classes


def _table_html(title: str, arr81_chars: List[str], classes: Optional[List[str]] = None) -> str:
    rows = _to_rows(arr81_chars)
    h = ["<div class='gridBlock'>", f"<div class='gridTitle'>{title}</div>", "<table class='sgrid'>"]
    for r in range(9):
        h.append("<tr>")
        for c in range(9):
            ch = rows[r][c]
            cls = classes[r * 9 + c] if classes else ""
            borders = []
            if r in (2, 5):
                borders.append("bb")
            if c in (2, 5):
                borders.append("br")
            cls = (cls + " " + " ".join(borders)).strip()
            h.append(f"<td class='{cls}'>{ch}</td>")
        h.append("</tr>")
    h.append("</table>")
    h.append("</div>")
    return "\n".join(h)


def make_colored_html_report(
    path: str,
    context: Dict[str, object],
    labels: List[int],
    target_input: List[int],
    baseline_final: List[int],
    patched_final: List[int],
    step_outputs: Dict[str, Dict[str, object]],
    run_metrics: Dict[str, List[Dict[str, float]]],
    patch_validation: Dict[str, Dict[str, float]],
    patched_steps: List[int],
    source_input: Optional[List[int]] = None,
    source_labels: Optional[List[int]] = None,
):
    """Generate a colored HTML report comparing baseline vs patched runs.

    Styling and class naming match:
    - sudoku_report_colored_ai.py
    - scripts/activation_patching_sudoku_report.py
    
    Args:
        source_input: Optional source puzzle input (will be displayed if provided).
        source_labels: Optional source puzzle labels (will be displayed if provided).
    """

    givens = _to_chars(target_input)
    label_chars = _to_chars(labels)
    base_final = _to_chars(baseline_final)
    patched_final_chars = _to_chars(patched_final)

    # CSS copied from sudoku_report_colored_ai.py / activation_patching_sudoku_report.py
    css = """
    <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; color:#222; }
    h1 { margin: 8px 0 0 0; }
    h2 { margin: 18px 0 6px 0; }
    .meta { color:#555; margin-bottom: 16px; }
    .row3 { display:grid; grid-template-columns: repeat(3, max-content); gap:20px; align-items:start; }
    .row2 { display:grid; grid-template-columns: repeat(2, max-content); gap:20px; align-items:start; }
    .gridBlock { display:block; }
    .sgrid { border-collapse:collapse; margin:6px 0 12px 0; }
    .sgrid td { width:22px; height:22px; text-align:center; border:1px solid #777; padding:2px 4px; position:relative; }
    .sgrid td.br { border-right:2px solid #111; }
    .sgrid td.bb { border-bottom:2px solid #111; }
    .gridTitle { font-weight:700; margin:4px 0; }

    /* Darker, more distinct fills */
    .given   { background:#9AB0FF; font-weight:700; }
    .ok      { background:#57B97B; color:#101; }
    .bad     { background:#E86B6B; color:#101; }
    .blank   { color:#667; }

    /* Changed cells: diagonal split yellow/green or yellow/red */
    .changed_ok {
      background-image: linear-gradient(135deg, #FFE15A 50%, #57B97B 50%);
      background-color: #57B97B;
    }
    .changed_bad {
      background-image: linear-gradient(135deg, #FFE15A 50%, #E86B6B 50%);
      background-color: #E86B6B;
    }

    .legend { margin: 10px 0 16px 0; color:#444; }
    .legend span { display:inline-block; padding:2px 8px; margin-right:8px; border:1px solid #999; }
    table.metrics { border-collapse:collapse; margin:8px 0 12px 0; }
    table.metrics th, table.metrics td { border:1px solid #999; padding:4px 8px; }
    table.metrics th { background:#eee; }
    </style>
    """

    legend = """
    <div class='legend'>
      <span class='given'>given</span>
      <span class='ok'>ok</span>
      <span class='bad'>bad</span>
      <span class='changed_ok'>changed_ok</span>
      <span class='changed_bad'>changed_bad</span>
      <span class='blank'>blank</span>
    </div>
    """

    # Meta lines
    meta_lines = [f"{k}: {v}" for k, v in context.items()]

    def _run_table(title: str, rows: List[Dict[str, float]]) -> str:
        parts = [f"<h3>{title}</h3>", "<table class='metrics'>"]
        parts.append("<tr><th>Run</th><th>Accuracy</th><th>Correct</th><th>Total</th></tr>")
        for i, r in enumerate(rows):
            parts.append(
                f"<tr><td>{i}</td><td>{r.get('accuracy', 0):.4f}</td>"
                f"<td>{int(r.get('correct', 0))}</td><td>{int(r.get('total_positions', 0))}</td></tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    sections: List[str] = []

    # Source puzzle context (if provided)
    if source_input is not None and source_labels is not None:
        source_givens = _to_chars(source_input)
        source_label_chars = _to_chars(source_labels)
        src_ctx = ["<h2>Source Puzzle</h2>", "<div class='row3'>"]
        src_ctx.append(_table_html("Input", source_givens))
        src_ctx.append(_table_html("Label", source_label_chars))
        src_ctx.append("<div class='gridBlock'></div>")
        src_ctx.append("</div>")
        sections.append("\n".join(src_ctx))

    # Target puzzle context
    ctx = ["<h2>Target Puzzle</h2>", "<div class='row3'>"]
    ctx.append(_table_html("Input", givens))
    ctx.append(_table_html("Label", label_chars))
    ctx.append("<div class='gridBlock'></div>")
    ctx.append("</div>")
    sections.append("\n".join(ctx))

    # Run accuracy
    sections.append("<h2>Run Accuracy</h2>")
    sections.append(_run_table("Baseline (unpatched)", run_metrics.get("target_baseline_runs", [])))
    sections.append(_run_table("Patched", run_metrics.get("target_patched_runs", [])))

    # Final baseline vs patched
    sections.append("<h2>Final Prediction (baseline vs patched)</h2>")
    base_classes_final = _compute_classes(base_final, givens, prev_chars=None)
    patched_classes_final = _compute_classes(patched_final_chars, givens, prev_chars=base_final)
    sec_final = ["<div class='row2'>"]
    sec_final.append(_table_html("Baseline prediction", base_final, base_classes_final))
    sec_final.append(_table_html("Patched prediction (changes vs baseline highlighted)", patched_final_chars, patched_classes_final))
    sec_final.append("</div>")
    sections.append("\n".join(sec_final))

    # Steps
    sections.append("<h2>Baseline vs Patched (per step)</h2>")
    for step_key in sorted(step_outputs.keys(), key=lambda x: int(x)):
        so = step_outputs[step_key]
        baseline_preds = _to_chars(_as_int_list(so.get("baseline_preds")))
        patched_preds_step = _to_chars(_as_int_list(so.get("patched_preds")))

        base_classes = _compute_classes(baseline_preds, givens, prev_chars=None)
        patched_classes = _compute_classes(patched_preds_step, givens, prev_chars=baseline_preds)

        sec = [f"<h3>Step {step_key}</h3>", "<div class='row2'>"]
        sec.append(_table_html("Baseline target preds", baseline_preds, base_classes))
        sec.append(_table_html("Patched target preds (changes vs baseline highlighted)", patched_preds_step, patched_classes))
        sec.append("</div>")
        sections.append("\n".join(sec))

    # Patch validation (keep this as numbers; styling now matches overall)
    if patch_validation:
        sections.append("<h2>Patch Validation (run 0)</h2>")
        pv = ["<table class='metrics'>", "<tr><th>Step</th><th>pre_diff_H</th><th>pre_diff_L</th><th>post_diff_H</th><th>post_diff_L</th></tr>"]
        for step, vals in sorted(patch_validation.items(), key=lambda x: int(x[0])):
            pv.append(
                f"<tr><td>{step}</td><td>{vals.get('pre_diff_H', float('nan')):.4e}</td>"
                f"<td>{vals.get('pre_diff_L', float('nan')):.4e}</td>"
                f"<td>{vals.get('post_diff_H', float('nan')):.4e}</td>"
                f"<td>{vals.get('post_diff_L', float('nan')):.4e}</td></tr>"
            )
        pv.append("</table>")
        sections.append("\n".join(pv))

    sections.append("<h2>Patched Steps Requested</h2>")
    sections.append(f"<div class='meta'>{patched_steps}</div>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Activation Patching Report</title>{css}</head>
<body>
<h1>Activation Patching Report</h1>
<div class='meta'>{'<br/>'.join(meta_lines)}</div>
{legend}
{''.join(sections)}
</body></html>
"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Activation Patching for HRM")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--source_puzzle_idx", type=int, default=0,
                        help="Index of source puzzle in test set")
    parser.add_argument("--target_puzzle_idx", type=int, default=1,
                        help="Index of target puzzle in test set")
    parser.add_argument(
        "--patch_level",
        type=str,
        default="both",
        help="Which stream to patch: H (z_H), L (z_L), or both. Also accepts z_h/z_l and lowercase.",
    )
    parser.add_argument("--patch_steps", type=str, default=None,
                        help="Comma-separated list of steps to patch (e.g., '0,1,2'). Default: all steps")
    # Backwards-compatible alias (older docs used --patch_step)
    parser.add_argument("--patch_step", type=int, default=None,
                        help="(DEPRECATED) Single step to patch; use --patch_steps")
    parser.add_argument("--patch_positions", type=str, default=None,
                        help="Comma-separated list of positions to patch (e.g., '0,9,18'). None = all positions")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Maximum number of reasoning steps")
    parser.add_argument("--output_dir", type=str, default="results/activation_patching",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument(
        "--verify_patching",
        action="store_true",
        help="Assert patched activations match source on patched steps (post_diff ~= 0).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of repeated forwards for both baseline and patched runs (>=1).",
    )
    parser.add_argument(
        "--report_html",
        type=str,
        default="activation_patching_report.html",
        help="File name (inside output_dir) for the colored HTML report; leave empty to skip.",
    )

    # Synthetic Sudoku inputs from labels
    parser.add_argument(
        "--source_missing_rows",
        type=str,
        default=None,
        help="Comma-separated 1-indexed Sudoku rows (1..9) to blank in the SOURCE input. Input is built from SOURCE labels.",
    )
    parser.add_argument(
        "--target_missing_rows",
        type=str,
        default=None,
        help="Comma-separated 1-indexed Sudoku rows (1..9) to blank in the TARGET input. Input is built from TARGET labels.",
    )
    parser.add_argument(
        "--blank_token_id",
        type=int,
        default=1,
        help="Token id to use for blank cells when synthesizing inputs (default 1, which renders as '.').",
    )
    
    args = parser.parse_args()
    
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be >= 1")

    # Canonicalize patch level so downstream logic/filenames are consistent.
    args.patch_level = _normalize_patch_level(args.patch_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse patch steps and positions
    if args.patch_steps is not None and args.patch_steps.strip() == "":
        args.patch_steps = None
    if args.patch_positions is not None and args.patch_positions.strip() == "":
        args.patch_positions = None

    patch_steps = _parse_int_list_arg(args.patch_steps)
    if args.patch_step is not None:
        if patch_steps is not None:
            raise ValueError("Use only one of --patch_step or --patch_steps")
        patch_steps = [int(args.patch_step)]

    patch_positions = _parse_int_list_arg(args.patch_positions)

    source_missing_rows = _parse_int_list_arg(args.source_missing_rows)
    target_missing_rows = _parse_int_list_arg(args.target_missing_rows)

    if patch_steps is not None:
        if any(s < 0 for s in patch_steps):
            raise ValueError("patch_steps must be >= 0")
        patch_steps = sorted(set(patch_steps))
    if patch_positions is not None:
        if any(p < 0 for p in patch_positions):
            raise ValueError("patch_positions must be >= 0")
        patch_positions = sorted(set(patch_positions))
    
    print("="*60)
    print("Activation Patching Experiment")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source puzzle: {args.source_puzzle_idx}")
    print(f"Target puzzle: {args.target_puzzle_idx}")
    print(f"Patch level: {args.patch_level}")
    print(f"Patch steps: {patch_steps if patch_steps else 'all'}")
    print(f"Patch positions: {patch_positions if patch_positions else 'all'}")
    print(f"Max steps: {args.max_steps}")
    print(f"Num runs per setting: {args.num_runs}")
    if source_missing_rows is not None:
        print(f"Synthetic SOURCE input: missing_rows={source_missing_rows} (from labels)")
    if target_missing_rows is not None:
        print(f"Synthetic TARGET input: missing_rows={target_missing_rows} (from labels)")
    if (source_missing_rows is not None) or (target_missing_rows is not None):
        print(f"Synthetic blank_token_id: {args.blank_token_id}")
    if args.report_html:
        print(f"HTML report file: {os.path.join(args.output_dir, args.report_html)}")
    print("="*60)
    
    # Setup device
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    # Create dataloader (using small batch size for individual puzzles)
    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True, 
        epochs_per_iter=1, global_batch_size=1,
        rank=0, world_size=1
    )
    
    # Initialize model
    train_state = init_train_state(config, test_metadata, world_size=1)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle compiled models - check if model expects _orig_mod prefix
    model_keys = set(train_state.model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
    checkpoint_has_prefix = any(k.startswith("_orig_mod.") for k in checkpoint_keys)
    
    # Add prefix if model has it but checkpoint doesn't
    if model_has_prefix and not checkpoint_has_prefix:
        checkpoint = {f"_orig_mod.{k}": v for k, v in checkpoint.items()}
    # Remove prefix if checkpoint has it but model doesn't
    elif checkpoint_has_prefix and not model_has_prefix:
        checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
    
    train_state.model.load_state_dict(checkpoint, assign=True)
    train_state.model.to(device)
    train_state.model.eval()
    
    print("Model loaded successfully")
    
    # Unwrap compiled modules and loss-head wrappers to get the ACT base model.
    model_obj: Any = train_state.model
    if hasattr(model_obj, "_orig_mod"):
        model_obj = model_obj._orig_mod
    if not isinstance(model_obj, HierarchicalReasoningModel_ACTV1) and hasattr(model_obj, "model"):
        model_obj = getattr(model_obj, "model")
    if not isinstance(model_obj, HierarchicalReasoningModel_ACTV1):
        raise TypeError(f"Expected HierarchicalReasoningModel_ACTV1, got {type(model_obj)}")

    model_obj = cast(HierarchicalReasoningModel_ACTV1, model_obj)

    patcher = ActivationPatcher(model_obj, device=device)

    def _extract_batch(item):
        """Extract the batch dict from dataloader yields.

        Observed formats in this repo:
        - (set_name, batch_dict, global_batch_size)
        - [set_name, batch_dict, global_batch_size]
        - batch_dict
        """
        if isinstance(item, (tuple, list)):
            if len(item) == 3 and isinstance(item[1], dict):
                return item[1]
            if len(item) == 2 and isinstance(item[1], dict):
                return item[1]
            if len(item) == 1 and isinstance(item[0], dict):
                return item[0]
        if isinstance(item, dict):
            return item
        raise TypeError(f"Unsupported dataloader item type/shape: {type(item)} repr={repr(item)[:200]}")
    
    # Get source and target puzzles
    print(f"\nGathering puzzles from test set...")
    puzzles = []
    for i, data in enumerate(test_loader):
        if i >= max(args.source_puzzle_idx, args.target_puzzle_idx) + 1:
            break
        batch = _extract_batch(data)
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        puzzles.append(batch)
    
    if len(puzzles) <= max(args.source_puzzle_idx, args.target_puzzle_idx):
        print(f"Error: Not enough puzzles in test set. Found {len(puzzles)}, need {max(args.source_puzzle_idx, args.target_puzzle_idx) + 1}")
        return
    
    source_batch = puzzles[args.source_puzzle_idx]
    target_batch = puzzles[args.target_puzzle_idx]

    # Optionally synthesize inputs from labels with missing rows.
    if source_missing_rows is not None:
        source_batch = _make_row_masked_sudoku_batch_from_labels(
            source_batch, missing_rows_1idx=source_missing_rows, blank_token_id=args.blank_token_id
        )
    if target_missing_rows is not None:
        target_batch = _make_row_masked_sudoku_batch_from_labels(
            target_batch, missing_rows_1idx=target_missing_rows, blank_token_id=args.blank_token_id
        )
    
    print(f"Source puzzle: shape {source_batch['inputs'].shape}")
    print(f"Target puzzle: shape {target_batch['inputs'].shape}")
    
    # Run baseline: source puzzle (no patching)
    print(f"\n{'='*60}")
    print("Running source puzzle (baseline)...")
    print('='*60)
    source_outputs = patcher.run_and_cache_activations(
        source_batch, patcher.source_cache, max_steps=args.max_steps
    )
    source_preds = source_outputs["logits"].argmax(-1)
    source_metrics = compute_metrics(source_preds, source_batch["labels"])
    print(f"Source accuracy: {source_metrics['accuracy']:.4f} "
          f"({source_metrics['correct']}/{source_metrics['total_positions']})")
    
    # Run baseline: target puzzle (no patching), repeated for stability
    print(f"\n{'='*60}")
    print("Running target puzzle (baseline, repeated)...")
    print('='*60)
    target_run_metrics: List[Dict[str, float]] = []
    target_run_caches: List[Dict[int, ActivationCache]] = []
    target_run_outputs: List[Dict[str, torch.Tensor]] = []
    for run in range(args.num_runs):
        cache: Dict[int, ActivationCache] = {}
        outputs = patcher.run_and_cache_activations(
            target_batch, cache, max_steps=args.max_steps
        )
        preds = outputs["logits"].argmax(-1)
        metrics = compute_metrics(preds, target_batch["labels"])
        target_run_metrics.append(metrics)
        target_run_caches.append(cache)
        target_run_outputs.append(outputs)
        print(
            f"Baseline run {run}: acc={metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['total_positions']})"
        )
    # First run is the canonical baseline for stepwise diffs / saved outputs.
    target_outputs = target_run_outputs[0]
    target_preds = target_outputs["logits"].argmax(-1)
    target_metrics = target_run_metrics[0]
    patcher.target_cache = target_run_caches[0]
    baseline_acc_values = [m.get("accuracy", 0.0) for m in target_run_metrics]
    print(
        f"Baseline accuracy range across {args.num_runs} runs: "
        f"min={min(baseline_acc_values):.4f}, max={max(baseline_acc_values):.4f}"
    )

    # Run with patching: target puzzle with source activations, repeated
    print(f"\n{'='*60}")
    print("Running target puzzle with patched activations (repeated)...")
    print('='*60)
    patched_run_metrics: List[Dict[str, float]] = []
    patched_run_caches: List[Dict[int, ActivationCache]] = []
    patched_run_outputs: List[Dict[str, torch.Tensor]] = []
    patch_validation_first: Dict[str, Dict[str, float]] = {}
    for run in range(args.num_runs):
        patched_outputs, patched_cache, patch_validation = patcher.run_with_patching(
            target_batch,
            patcher.source_cache,
            patch_level=args.patch_level,
            patch_steps=patch_steps,
            patch_positions=patch_positions,
            max_steps=args.max_steps,
            verify=args.verify_patching,
        )
        preds = patched_outputs["logits"].argmax(-1)
        metrics = compute_metrics(preds, target_batch["labels"])
        patched_run_metrics.append(metrics)
        patched_run_caches.append(patched_cache)
        patched_run_outputs.append(patched_outputs)
        print(
            f"Patched run {run}: acc={metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['total_positions']})"
        )
        if run == 0:
            patch_validation_first = {str(k): v for k, v in patch_validation.items()}

    # Canonical patched outputs are from run 0.
    patched_outputs0 = patched_run_outputs[0]
    patched_preds = patched_outputs0["logits"].argmax(-1)
    patched_metrics = patched_run_metrics[0]
    patched_cache_first = patched_run_caches[0]

    patched_acc_values = [m.get("accuracy", 0.0) for m in patched_run_metrics]
    print(
        f"Patched accuracy range across {args.num_runs} runs: "
        f"min={min(patched_acc_values):.4f}, max={max(patched_acc_values):.4f}"
    )

    # Compute impact of patching (first run for reference)
    accuracy_change = patched_metrics['accuracy'] - target_metrics['accuracy']
    print(f"\n{'='*60}")
    print("Impact Analysis")
    print('='*60)
    print(f"Accuracy change: {accuracy_change:+.4f}")
    print(f"Source → Target baseline: {target_metrics['accuracy']:.4f}")
    print(f"Source → Target patched:  {patched_metrics['accuracy']:.4f}")

    # Stepwise comparison: baseline target vs patched target
    labels = target_batch["labels"]
    common_steps = sorted(set(patcher.target_cache.keys()) & set(patched_cache_first.keys()))

    # Steps that are considered "patched" for reporting.
    if patch_steps is None:
        patched_steps_effective = common_steps
    else:
        patched_steps_effective = [s for s in patch_steps if s in common_steps]

    stepwise_metrics: Dict[str, Dict[str, object]] = {}
    step_outputs: Dict[str, Dict[str, object]] = {}

    for s in common_steps:
        base_preds_step = patcher.target_cache[s].preds
        patched_preds_step = patched_cache_first[s].preds

        base_m = compute_metrics(base_preds_step, labels)
        patched_m = compute_metrics(patched_preds_step, labels)
        diff_m = compute_diff_metrics(base_preds_step, patched_preds_step, labels)

        stepwise_metrics[str(s)] = {
            "baseline_accuracy": base_m["accuracy"],
            "patched_accuracy": patched_m["accuracy"],
            "delta_accuracy": float(patched_m["accuracy"] - base_m["accuracy"]),
            "changed_count": diff_m["changed_count"],
            "changed_indices_sample": diff_m["changed_indices_sample"],
        }

        # Only store full step outputs for steps of interest (patched steps and their immediate next step).
        if (s in patched_steps_effective) or (s - 1 in patched_steps_effective):
            step_outputs[str(s)] = {
                "baseline_preds": base_preds_step[0].detach().cpu().tolist(),
                "patched_preds": patched_preds_step[0].detach().cpu().tolist(),
                "diff": diff_m,
            }

    # Console report focusing on (patched step) and (next step) effects
    print(f"\n{'='*60}")
    print("Per-step intermediate output differences")
    print('='*60)
    if not patched_steps_effective:
        print("No patched steps intersected cached steps; nothing to compare.")
    else:
        # Context: show the target input puzzle ('.' = blank)
        print("\nTarget input puzzle (from batch['inputs']):")
        print(sudoku_tokens_to_grid_str(target_batch["inputs"]))
        for s in patched_steps_effective:
            immediate = stepwise_metrics.get(str(s), {})
            print(
                f"Patched step {s}: Δacc={immediate.get('delta_accuracy', 0):+.4f} | "
                f"changed={immediate.get('changed_count', 0)} | idx_sample={immediate.get('changed_indices_sample', [])}"
            )

            # Show baseline vs patched intermediate grids for this step when available
            so = step_outputs.get(str(s))
            if so is not None:
                b = torch.tensor(so["baseline_preds"], dtype=torch.int64)
                p = torch.tensor(so["patched_preds"], dtype=torch.int64)
                print("\nBaseline target preds grid (this step):")
                print(sudoku_tokens_to_grid_str(b))
                print("\nPatched  target preds grid (this step):")
                print(sudoku_tokens_to_grid_str(p))

            next_s = s + 1
            if str(next_s) in stepwise_metrics:
                nxt = stepwise_metrics[str(next_s)]
                print(
                    f"Next step  {next_s}: Δacc={nxt.get('delta_accuracy', 0):+.4f} | "
                    f"changed={nxt.get('changed_count', 0)} | idx_sample={nxt.get('changed_indices_sample', [])}"
                )

                so2 = step_outputs.get(str(next_s))
                if so2 is not None:
                    b2 = torch.tensor(so2["baseline_preds"], dtype=torch.int64)
                    p2 = torch.tensor(so2["patched_preds"], dtype=torch.int64)
                    print("\nBaseline target preds grid (next step):")
                    print(sudoku_tokens_to_grid_str(b2))
                    print("\nPatched  target preds grid (next step):")
                    print(sudoku_tokens_to_grid_str(p2))
    
    # Save results
    run_metrics = {
        "target_baseline_runs": target_run_metrics,
        "target_patched_runs": patched_run_metrics,
    }

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "source_puzzle_idx": args.source_puzzle_idx,
            "target_puzzle_idx": args.target_puzzle_idx,
            "patch_level": args.patch_level,
            "patch_steps": patch_steps,
            "patch_positions": patch_positions,
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
            "source_missing_rows": source_missing_rows,
            "target_missing_rows": target_missing_rows,
            "blank_token_id": args.blank_token_id,
        },
        "patch_validation": patch_validation_first,
        "metrics": {
            "source": source_metrics,
            "target_baseline": target_metrics,
            "target_patched": patched_metrics,
            "accuracy_change": accuracy_change,
        },
        "stepwise_metrics": stepwise_metrics,
        "step_outputs": step_outputs,
        "predictions": {
            "source": source_preds.cpu().numpy().tolist(),
            "target_baseline": target_preds.cpu().numpy().tolist(),
            "target_patched": patched_preds.cpu().numpy().tolist(),
        },
        "run_metrics": run_metrics,
    }
    
    # Save as YAML (add _forward suffix for clarity)
    results_path = os.path.join(
        args.output_dir, 
        f"patch_s{args.source_puzzle_idx}_t{args.target_puzzle_idx}_{args.patch_level}_forward.yaml"
    )
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")

    if args.report_html:
        report_path = os.path.join(args.output_dir, args.report_html)
        context = {
            "checkpoint": args.checkpoint,
            "source_puzzle_idx": args.source_puzzle_idx,
            "target_puzzle_idx": args.target_puzzle_idx,
            "patch_level": args.patch_level,
            "patch_steps": patch_steps if patch_steps is not None else "all",
            "patch_positions": patch_positions if patch_positions is not None else "all",
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
            "direction": "forward (source→target)",
            "baseline_accuracy_run0": target_metrics["accuracy"],
            "patched_accuracy_run0": patched_metrics["accuracy"],
        }

        labels_flat = target_batch["labels"][0].detach().cpu().tolist()
        target_input_flat = target_batch["inputs"][0].detach().cpu().tolist()
        baseline_final_flat = target_preds[0].detach().cpu().tolist()
        patched_final_flat = patched_preds[0].detach().cpu().tolist()
        source_input_flat = source_batch["inputs"][0].detach().cpu().tolist()
        source_labels_flat = source_batch["labels"][0].detach().cpu().tolist()

        make_colored_html_report(
            report_path,
            context,
            labels_flat,
            target_input_flat,
            baseline_final_flat,
            patched_final_flat,
            step_outputs,
            run_metrics,
            patch_validation_first,
            patched_steps_effective,
            source_input=source_input_flat,
            source_labels=source_labels_flat,
        )
        print(f"HTML report saved to: {report_path}")
    
    # INVERSE EXPERIMENT: Patch target activations into source puzzle
    print(f"\n{'='*60}")
    print("Running inverse experiment (target→source patching)...")
    print('='*60)
    
    # Run baseline forward on source (already have this from run_and_cache_activations)
    source_metrics = compute_metrics(source_preds, source_batch["labels"])
    
    # Run inverse patching for each run
    inverse_patched_cache_first = None
    inverse_patched_preds = None
    inverse_patched_metrics = None
    inverse_run_metrics: Dict[str, Dict[str, float]] = {}
    inverse_patch_validation_first: Dict[str, Dict[str, float]] = {}
    inverse_patched_steps_effective = None
    
    for run_idx in range(args.num_runs):
        print(f"\n  Inverse run {run_idx + 1}/{args.num_runs}...")
        
        # Patch target activations into source
        inverse_patched_outputs, inverse_patched_cache, inverse_patch_validation = patcher.run_with_patching(
            source_batch,
            patcher.target_cache,  # Patch target activations
            patch_level=args.patch_level,
            patch_steps=patch_steps,
            patch_positions=patch_positions,
            max_steps=args.max_steps,
            verify=args.verify_patching,
        )
        
        inverse_preds_run = inverse_patched_outputs["logits"].argmax(-1)
        inverse_patched_preds = inverse_preds_run
        if run_idx == 0:
            inverse_patched_cache_first = inverse_patched_cache
            inverse_patch_validation_first = {str(k): v for k, v in inverse_patch_validation.items()}
        
        inverse_metrics_run = compute_metrics(inverse_preds_run, source_batch["labels"])
        inverse_run_metrics[f"run_{run_idx}"] = {
            "accuracy": inverse_metrics_run["accuracy"],
        }
        inverse_patched_metrics = inverse_metrics_run
        
        print(
            f"Inverse run {run_idx}: acc={inverse_metrics_run['accuracy']:.4f} "
            f"({inverse_metrics_run['correct']}/{inverse_metrics_run['total_positions']})"
        )
    
    # Compute stepwise metrics for inverse (compare source baseline vs source with target patched)
    labels = source_batch["labels"]
    common_steps_inv = sorted(set(patcher.source_cache.keys()) & set(inverse_patched_cache_first.keys())) if inverse_patched_cache_first else []
    
    if patch_steps is None:
        inverse_patched_steps_effective = common_steps_inv
    else:
        inverse_patched_steps_effective = [s for s in patch_steps if s in common_steps_inv]
    
    inverse_stepwise_metrics: Dict[str, Dict[str, object]] = {}
    inverse_step_outputs: Dict[str, Dict[str, object]] = {}
    
    for s in common_steps_inv:
        base_preds_step = patcher.source_cache[s].preds
        inverse_patched_preds_step = inverse_patched_cache_first[s].preds if inverse_patched_cache_first else None
        
        if inverse_patched_preds_step is not None:
            base_m = compute_metrics(base_preds_step, labels)
            inv_patched_m = compute_metrics(inverse_patched_preds_step, labels)
            diff_m = compute_diff_metrics(base_preds_step, inverse_patched_preds_step, labels)

            inverse_stepwise_metrics[str(s)] = {
                "baseline_accuracy": base_m["accuracy"],
                "patched_accuracy": inv_patched_m["accuracy"],
                "delta_accuracy": float(inv_patched_m["accuracy"] - base_m["accuracy"]),
                "changed_count": diff_m["changed_count"],
                "changed_indices_sample": diff_m["changed_indices_sample"],
            }

            # Only store full step outputs for steps of interest (patched steps and their immediate next step).
            if (s in inverse_patched_steps_effective) or (s - 1 in inverse_patched_steps_effective):
                inverse_step_outputs[str(s)] = {
                    "baseline_preds": base_preds_step[0].detach().cpu().tolist(),
                    "patched_preds": inverse_patched_preds_step[0].detach().cpu().tolist(),
                    "diff": diff_m,
                }
    
    # Save inverse results
    inverse_results = {
        "config": {
            "checkpoint": args.checkpoint,
            "source_puzzle_idx": args.source_puzzle_idx,
            "target_puzzle_idx": args.target_puzzle_idx,
            "direction": "inverse (target→source)",
            "patch_level": args.patch_level,
            "patch_steps": patch_steps if patch_steps is not None else "all",
            "patch_positions": patch_positions if patch_positions is not None else "all",
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
        },
        "metrics": {
            "source_baseline_accuracy": source_metrics["accuracy"],
            "inverse_patched_accuracy": inverse_patched_metrics["accuracy"] if inverse_patched_metrics else 0.0,
            "source_baseline_solution": (source_preds == source_batch["labels"]).all().item(),
            "inverse_patched_solution": (inverse_patched_preds == source_batch["labels"]).all().item() if inverse_patched_preds is not None else False,
        },
        "stepwise_metrics": inverse_stepwise_metrics,
        "step_outputs": inverse_step_outputs,
        "predictions": {
            "source_baseline": source_preds.cpu().numpy().tolist(),
            "source_with_target_patched": inverse_patched_preds.cpu().numpy().tolist() if inverse_patched_preds is not None else [],
        },
        "run_metrics": inverse_run_metrics,
    }
    
    # Save inverse YAML
    inverse_results_path = os.path.join(
        args.output_dir,
        f"patch_s{args.source_puzzle_idx}_t{args.target_puzzle_idx}_{args.patch_level}_inverse.yaml"
    )
    with open(inverse_results_path, "w") as f:
        yaml.dump(inverse_results, f, default_flow_style=False)
    print(f"\nInverse results saved to: {inverse_results_path}")

    if args.report_html:
        inverse_report_name = args.report_html
        root, ext = os.path.splitext(inverse_report_name)
        inverse_report_name = f"{root}_inverse{ext or '.html'}"
        inverse_report_path = os.path.join(args.output_dir, inverse_report_name)
        
        inverse_context = {
            "checkpoint": args.checkpoint,
            "source_puzzle_idx": args.source_puzzle_idx,
            "target_puzzle_idx": args.target_puzzle_idx,
            "patch_level": args.patch_level,
            "patch_steps": patch_steps if patch_steps is not None else "all",
            "patch_positions": patch_positions if patch_positions is not None else "all",
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
            "direction": "inverse (target→source)",
            "baseline_accuracy_run0": source_metrics["accuracy"],
            "patched_accuracy_run0": inverse_patched_metrics["accuracy"] if inverse_patched_metrics else 0.0,
        }

        source_labels_flat = source_batch["labels"][0].detach().cpu().tolist()
        source_input_flat = source_batch["inputs"][0].detach().cpu().tolist()
        source_baseline_flat = source_preds[0].detach().cpu().tolist()
        inverse_patched_flat = inverse_patched_preds[0].detach().cpu().tolist() if inverse_patched_preds is not None else []
        target_input_flat_for_inv = target_batch["inputs"][0].detach().cpu().tolist()
        target_labels_flat_for_inv = target_batch["labels"][0].detach().cpu().tolist()

        make_colored_html_report(
            inverse_report_path,
            inverse_context,
            source_labels_flat,
            source_input_flat,
            source_baseline_flat,
            inverse_patched_flat,
            inverse_step_outputs,
            inverse_run_metrics,
            inverse_patch_validation_first,
            inverse_patched_steps_effective or [],
            source_input=target_input_flat_for_inv,
            source_labels=target_labels_flat_for_inv,
        )
        print(f"Inverse HTML report saved to: {inverse_report_path}")
    

    cache_path = os.path.join(
        args.output_dir,
        f"activations_s{args.source_puzzle_idx}_t{args.target_puzzle_idx}.pt"
    )
    torch.save({
        "source": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(),
            "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(),
            "step": v.step,
        } for k, v in patcher.source_cache.items()},
        "target": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(),
            "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(),
            "step": v.step,
        } for k, v in patcher.target_cache.items()},
        "patched_forward": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(),
            "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(),
            "step": v.step,
        } for k, v in patched_cache_first.items()} if patched_cache_first else {},
        "patched_inverse": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "z_H_out": v.z_H_out.cpu(),
            "z_L_out": v.z_L_out.cpu(),
            "preds": v.preds.cpu(),
            "step": v.step,
        } for k, v in inverse_patched_cache_first.items()} if inverse_patched_cache_first else {},
    }, cache_path)
    print(f"Activation caches saved to: {cache_path}")
    
    print(f"\n{'='*60}")
    print("Experiment completed successfully!")
    print('='*60)


if __name__ == "__main__":
    main()
