"""Lightweight rollout with an arbitrary per-step state transform.

``ActivationAblator`` caches every step's activations, which is slow when a
script runs thousands of interventions and only needs the final prediction.
``InterventionRunner.rollout`` runs the same ACT loop (same carry reset, same
``_forward_one_step``) and applies ``transform(z_H, z_L, step) -> (z_H, z_L)`` to
the carry entering each step, exactly where the existing ablation, patching and
SAE scripts intervene.

Used by the rebuttal controls:
  scripts/sae/sae_error_controls.py   (E1)
  scripts/analysis/bptt_arm_eval.py   (E7)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from scripts.core.activation_ablation import ActivationAblator, _make_inner_carry

StateTransform = Callable[[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]


class InterventionRunner(ActivationAblator):
    """ACT rollout with an optional transform on the incoming carry."""

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        transform: Optional[StateTransform] = None,
        max_steps: int = 16,
        record: bool = False,
    ) -> Dict[str, object]:
        """Run ``max_steps`` ACT steps.

        Returns ``{"preds": final [B, T] token predictions}``. With
        ``record=True`` also returns per-step ``step_preds`` and the per-step
        output states ``z_H_out`` / ``z_L_out`` (float32, on CPU).
        """
        carry = self._init_carry(batch)
        original_max = self.model.config.halt_max_steps
        self.model.config.halt_max_steps = max_steps

        preds: Optional[torch.Tensor] = None
        step_preds: List[torch.Tensor] = []
        z_H_out: List[torch.Tensor] = []
        z_L_out: List[torch.Tensor] = []
        try:
            with torch.no_grad():
                for step in range(max_steps):
                    inner_in, _steps_in, _data = self._prepare_step_inputs(carry, batch)
                    if transform is not None:
                        z_H, z_L = transform(inner_in.z_H, inner_in.z_L, step)
                        inner_in = _make_inner_carry(self.model, z_H=z_H, z_L=z_L)
                    carry, outputs, _ = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )
                    preds = outputs["intermediate_preds_step"]
                    if record:
                        step_preds.append(preds.detach().cpu())
                        z_H_out.append(carry.inner_carry.z_H.detach().float().cpu())
                        z_L_out.append(carry.inner_carry.z_L.detach().float().cpu())
        finally:
            self.model.config.halt_max_steps = original_max

        out: Dict[str, object] = {"preds": preds}
        if record:
            out.update(step_preds=step_preds, z_H_out=z_H_out, z_L_out=z_L_out)
        return out
