"""Plain Transformer Baseline for Sudoku.

A standard non-recurrent Transformer: single forward pass through N layers,
standard cross-entropy loss, NO recurrence, NO ACT, NO deep supervision.

This is the simplest possible baseline — it isolates whether recurrence
and iterative refinement contribute anything beyond a deep feedforward model.
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
# Carry dataclasses — minimal, for ACTLossHead compatibility
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlainTransformerInnerCarry:
    z: torch.Tensor  # [B, T, D] — unused placeholder


@dataclass
class PlainTransformerCarry:
    inner_carry: PlainTransformerInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class PlainTransformerConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    num_layers: int  # Total transformer layers (single pass)

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting config — kept for ACTLossHead compat, but always halts after 1 step
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    forward_dtype: str = "bfloat16"


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block (reused from HRM)
# ═══════════════════════════════════════════════════════════════════════════

class PlainTransformerBlock(nn.Module):
    def __init__(self, config: PlainTransformerConfig) -> None:
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
# Model
# ═══════════════════════════════════════════════════════════════════════════

class PlainTransformerModel(nn.Module):
    """Plain Transformer — single forward pass, no recurrence."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = PlainTransformerConfig(**config_dict)
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

        # Transformer layers — all unique weights, single pass
        self.layers = nn.ModuleList([
            PlainTransformerBlock(self.config)
            for _ in range(self.config.num_layers)
        ])

        # Q head special init (dummy — always halts)
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

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return PlainTransformerCarry(
            inner_carry=PlainTransformerInnerCarry(
                z=torch.zeros(
                    batch_size, self.config.seq_len + self.puzzle_emb_len,
                    self.config.hidden_size, dtype=self.forward_dtype,
                ),
            ),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Start halted so first call resets
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: PlainTransformerCarry,
        batch: Dict[str, torch.Tensor],
        probe_recorder: Optional[object] = None,
    ) -> Tuple[PlainTransformerCarry, Dict[str, torch.Tensor]]:
        # Reset current data from batch (same pattern as other models)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        input_embeddings = self._input_embeddings(
            new_current_data["inputs"], new_current_data["puzzle_identifiers"]
        )

        # Single forward pass through all layers (no recurrence)
        z = input_embeddings
        for layer in self.layers:
            z = layer(cos_sin=cos_sin, hidden_states=z)

        # Output
        logits = self.lm_head(z)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z[:, 0]).to(torch.float32)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_logits[..., 0],
            "q_continue_logits": q_logits[..., 1],
            "intermediate_preds_step": logits.argmax(-1),
        }

        # Always halt after 1 step — no recurrence
        new_carry = PlainTransformerCarry(
            inner_carry=PlainTransformerInnerCarry(z=z.detach()),
            steps=torch.ones_like(carry.steps),
            halted=torch.ones_like(carry.halted),  # Always done
            current_data=new_current_data,
        )

        return new_carry, outputs
