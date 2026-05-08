"""Universal Transformer Baseline for Sudoku.

A recurrent Transformer with weight-sharing across iterations. Same compute
budget and recurrence as HRM, but NO hierarchical separation — single set of
shared Transformer layers applied repeatedly.

Key difference from HRM: Recurrent but flat — same compute budget with
recurrence, but no hierarchical separation of H and L modules.

Key difference from Vanilla RNN: Weight-sharing — same layers reused each
iteration (classic Universal Transformer design), rather than separate
per-layer parameters.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin,
    CastedEmbedding, CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding


# ═══════════════════════════════════════════════════════════════════════════
# Carry dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UniversalTransformerInnerCarry:
    z: torch.Tensor  # [B, T, D]


@dataclass
class UniversalTransformerCarry:
    inner_carry: UniversalTransformerInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class UniversalTransformerConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    num_shared_layers: int  # Shared layers (weight-tied)
    num_iterations: int     # Times to repeat the shared block per step

    # If True (default, matches HRM): wrap all but the final iteration in torch.no_grad()
    # so only the last step receives gradient. If False, all iterations are differentiated
    # (a fully-unrolled BPTT) -- used for the standalone Universal-Transformer baseline.
    one_step_grad: bool = True

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


# ═══════════════════════════════════════════════════════════════════════════
# Shared Transformer Block (weight-tied across iterations)
# ═══════════════════════════════════════════════════════════════════════════

class UniversalTransformerBlock(nn.Module):
    def __init__(self, config: UniversalTransformerConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════════
# Inner model
# ═══════════════════════════════════════════════════════════════════════════

class UniversalTransformer_Inner(nn.Module):
    def __init__(self, config: UniversalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype,
            )

        # Positional encoding
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype,
            )

        # Shared transformer layers (weight-tied — applied num_iterations times)
        self.shared_layers = nn.ModuleList([
            UniversalTransformerBlock(self.config)
            for _ in range(self.config.num_shared_layers)
        ])

        # Initial state
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return UniversalTransformerInnerCarry(
            z=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: UniversalTransformerInnerCarry):
        return UniversalTransformerInnerCarry(
            z=torch.where(reset_flag.view(-1, 1, 1), self.z_init, carry.z),
        )

    def _apply_shared_block(self, z: torch.Tensor, input_embeddings: torch.Tensor, cos_sin):
        """Apply input injection + all shared layers once."""
        z = z + input_embeddings
        for layer in self.shared_layers:
            z = layer(cos_sin=cos_sin, hidden_states=z)
        return z

    def forward(
        self,
        carry: UniversalTransformerInnerCarry,
        batch: Dict[str, torch.Tensor],
        probe_recorder: Optional[object] = None,
        step_index: Optional[int] = None,
    ) -> Tuple[UniversalTransformerInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        if self.config.one_step_grad:
            # Repeat shared layers num_iterations times (no-grad for all but last)
            with torch.no_grad():
                z = carry.z
                for _iter in range(self.config.num_iterations - 1):
                    z = self._apply_shared_block(z, input_embeddings, cos_sin)

            assert not z.requires_grad

            # 1-step grad
            z = self._apply_shared_block(z, input_embeddings, cos_sin)
        else:
            # Fully-unrolled recurrence with gradient through every iteration.
            z = carry.z
            for _iter in range(self.config.num_iterations):
                z = self._apply_shared_block(z, input_embeddings, cos_sin)

        # Output
        new_carry = UniversalTransformerInnerCarry(z=z.detach())
        output = self.lm_head(z)[:, self.puzzle_emb_len:]

        q_logits = self.q_head(z[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


# ═══════════════════════════════════════════════════════════════════════════
# ACT Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class UniversalTransformerModel(nn.Module):
    """ACT wrapper for Universal Transformer baseline."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = UniversalTransformerConfig(**config_dict)
        self.inner = UniversalTransformer_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return UniversalTransformerCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: UniversalTransformerCarry,
        batch: Dict[str, torch.Tensor],
        probe_recorder: Optional[object] = None,
    ) -> Tuple[UniversalTransformerCarry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data,
            probe_recorder=probe_recorder,
            step_index=int(new_steps.max().item()) if torch.is_tensor(new_steps) else None,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "intermediate_preds_step": logits.argmax(-1),
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry, new_current_data,
                )[-1]
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return UniversalTransformerCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
