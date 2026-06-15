"""Capture intermediate z_L / z_H states inside the HRM inner loop.

The default `ActivationCache` only stores z_H/z_L at the start and end of each
ACT step. The HRM inner forward actually runs `H_cycles * L_cycles` micro
iterations per ACT step, so we use forward hooks on `model.inner.L_level` and
`model.inner.H_level` to record every intermediate state.

Schedule for HRM ACTv1 with H_cycles=H, L_cycles=L (maze: H=L=2):
    no-grad outer loop:
        for h in 0..H-1:
            for l in 0..L-1:
                if not (h==H-1 and l==L-1): z_L = L_level(z_L, z_H + emb)
            if h != H-1:                    z_H = H_level(z_H, z_L)
    1-step grad:
        z_L = L_level(...)   # final L update
        z_H = H_level(...)   # final H update (this is what lm_head reads)

Per ACT step we therefore see:
    - L_level calls:  H*L - 1 in no-grad + 1 in grad = H*L total
    - H_level calls:  (H - 1) in no-grad + 1 in grad = H   total
For maze (H=2,L=2): 4 L-calls, 2 H-calls per ACT step.

z_L decoded via `lm_head` is an OFF-DISTRIBUTION projection (the LM head was
trained on z_H only). Treat decoded z_L grids as qualitative "what does the
working memory contain right now" — not a correctness measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SubStepCapture:
    """Records per-ACT-step intermediate z_L and z_H tensors (post-update)."""
    z_L_substeps: List[torch.Tensor] = field(default_factory=list)
    z_H_substeps: List[torch.Tensor] = field(default_factory=list)


class SubStepRecorder:
    """Attach forward hooks on `model.inner.L_level` and `H_level`.

    Maintains a flat list of outputs; split into per-ACT-step groups by
    calling `take_groups(num_act_steps, H_cycles, L_cycles)` after the run.
    """

    def __init__(self, model: torch.nn.Module):
        # model is a HierarchicalReasoningModel_ACTV1 wrapper; inner has L_level/H_level
        self.inner = model.inner
        self._l_outs: List[torch.Tensor] = []
        self._h_outs: List[torch.Tensor] = []
        self._handles: List = []

    def __enter__(self) -> "SubStepRecorder":
        def _l_hook(_mod, _inp, out):
            self._l_outs.append(out.detach().clone())

        def _h_hook(_mod, _inp, out):
            self._h_outs.append(out.detach().clone())

        self._handles.append(self.inner.L_level.register_forward_hook(_l_hook))
        self._handles.append(self.inner.H_level.register_forward_hook(_h_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self._l_outs.clear()
        self._h_outs.clear()

    def take_groups(self, num_act_steps: int, H_cycles: int, L_cycles: int
                    ) -> List[SubStepCapture]:
        """Partition the recorded flat lists into per-ACT-step groups.

        Returns a list of length `num_act_steps`. If the recorded counts do
        not match the expected schedule, slices best-effort and pads with the
        last available tensor.
        """
        L_per_step = H_cycles * L_cycles      # e.g. 4
        H_per_step = H_cycles                 # e.g. 2
        groups: List[SubStepCapture] = []
        for s in range(num_act_steps):
            l_chunk = self._l_outs[s * L_per_step:(s + 1) * L_per_step]
            h_chunk = self._h_outs[s * H_per_step:(s + 1) * H_per_step]
            # pad if short (shouldn't normally happen)
            while len(l_chunk) < L_per_step and self._l_outs:
                l_chunk.append(self._l_outs[-1])
            while len(h_chunk) < H_per_step and self._h_outs:
                h_chunk.append(self._h_outs[-1])
            groups.append(SubStepCapture(z_L_substeps=l_chunk, z_H_substeps=h_chunk))
        return groups


@torch.no_grad()
def decode_via_lm_head(z: torch.Tensor, model, puzzle_emb_len: int) -> torch.Tensor:
    """Apply the (z_H-trained) lm_head to an arbitrary hidden state.

    Returns argmax token ids over the answer slice (puzzle_emb prefix stripped),
    shape [B, seq_len].
    """
    logits = model.inner.lm_head(z)[:, puzzle_emb_len:]
    return logits.argmax(-1)


def sublabels(H_cycles: int, L_cycles: int) -> Tuple[List[str], List[str]]:
    """Human-readable labels for each L / H sub-step within one ACT step.

    Mirrors the schedule comment at the top of this file.
    """
    l_labels: List[str] = []
    h_labels: List[str] = []
    # no-grad outer loop
    for h in range(H_cycles):
        for l in range(L_cycles):
            if not (h == H_cycles - 1 and l == L_cycles - 1):
                l_labels.append(f"z_L H{h}.L{l}")
        if h != H_cycles - 1:
            h_labels.append(f"z_H H{h}")
    # 1-step grad
    l_labels.append("z_L grad")
    h_labels.append("z_H grad (final)")
    return l_labels, h_labels
