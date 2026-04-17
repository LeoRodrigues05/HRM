"""Standard RNN Baseline for Sudoku.

A true recurrent neural network using nn.GRU — NOT a Transformer block applied
recurrently. Single forward pass: embed input → GRU layers → linear head.

This isolates whether the Transformer architecture itself matters, compared to
a classical sequence model on the same data.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


# ═══════════════════════════════════════════════════════════════════════════
# Carry dataclasses — for ACTLossHead compatibility
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StandardRNNInnerCarry:
    h: torch.Tensor  # GRU hidden state [num_layers, B, hidden_size]


@dataclass
class StandardRNNCarry:
    inner_carry: StandardRNNInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class StandardRNNConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    num_layers: int       # GRU layers (stacked)
    hidden_size: int      # GRU hidden dimension
    dropout: float = 0.0  # Dropout between GRU layers

    # Halting — always halts after 1 step (no recurrence over time)
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    forward_dtype: str = "bfloat16"


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class StandardRNNModel(nn.Module):
    """Standard GRU-based RNN — single forward pass, no iterative refinement."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = StandardRNNConfig(**config_dict)
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

        # Standard GRU — processes the 81-cell sequence
        self.gru = nn.GRU(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=False,
        )

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

        return self.embed_scale * embedding

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return StandardRNNCarry(
            inner_carry=StandardRNNInnerCarry(
                h=torch.zeros(
                    self.config.num_layers, batch_size, self.config.hidden_size,
                    dtype=self.forward_dtype,
                ),
            ),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: StandardRNNCarry,
        batch: Dict[str, torch.Tensor],
        probe_recorder: Optional[object] = None,
    ) -> Tuple[StandardRNNCarry, Dict[str, torch.Tensor]]:
        # Reset inner carry for halted examples (new puzzles get fresh h0=0)
        h0 = torch.where(
            carry.halted.view(1, -1, 1),  # [1, B, 1] broadcast over [num_layers, B, hidden]
            torch.zeros_like(carry.inner_carry.h),
            carry.inner_carry.h,
        )

        # Reset current data from batch
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }

        input_embeddings = self._input_embeddings(
            new_current_data["inputs"], new_current_data["puzzle_identifiers"]
        )

        # GRU forward — processes sequence of 81 cells
        # input: [B, T, hidden_size], h0: [num_layers, B, hidden_size]
        # GRU requires float32 — cast input, run, cast back
        gru_out, h_n = self.gru(
            input_embeddings.to(torch.float32),
            h0.to(torch.float32),
        )
        gru_out = gru_out.to(self.forward_dtype)

        # Output — skip puzzle_emb prefix tokens
        logits = self.lm_head(gru_out)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(gru_out[:, 0]).to(torch.float32)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_logits[..., 0],
            "q_continue_logits": q_logits[..., 1],
            "intermediate_preds_step": logits.argmax(-1),
        }

        # Always halt — no iterative refinement
        new_carry = StandardRNNCarry(
            inner_carry=StandardRNNInnerCarry(h=h_n.detach().to(self.forward_dtype)),
            steps=torch.ones_like(carry.steps),
            halted=torch.ones_like(carry.halted),
            current_data=new_current_data,
        )

        return new_carry, outputs
