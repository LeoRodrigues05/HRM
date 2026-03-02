"""
HRM v2: Hierarchical Reasoning Model with Structural Inductive Biases

This is the main model file that integrates:
- Point 2: Constraint-aware sparse attention in L_level
- Point 3: Explicit constraint satisfaction head
- Point 5: GNN layers for structural message passing

The architecture follows the original HRM but with key modifications:
1. L_level uses ConstraintSparseAttention (restricted to row/col/box neighbors)
2. GNN layers are interleaved with Transformer blocks for hybrid reasoning
3. A ConstraintSatisfactionHead provides auxiliary supervision
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, 
    CosSin, CastedEmbedding, CastedLinear
)
from models.sparse_embedding import CastedSparseEmbedding

from .constraint_attention import ConstraintSparseAttention, build_sudoku_attention_mask
from .constraint_head import ConstraintSatisfactionHead, ConstraintViolationCounter
from .gnn_layers import ConstraintGNNLayer, ConstraintGATLayer


@dataclass
class HierarchicalReasoningModel_V2InnerCarry:
    """Internal state for the reasoning model."""
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_V2Carry:
    """Full carry state including ACT control."""
    inner_carry: HierarchicalReasoningModel_V2InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_V2Config(BaseModel):
    """Configuration for HRM v2."""
    
    # Basic dimensions
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Hierarchy structure
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # ACT config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    
    # === V2 Additions ===
    
    # Point 2: Constraint-aware sparse attention
    use_sparse_attention: bool = True  # Enable sparse attention in L_level
    sparse_attention_layers: str = "L"  # Which level(s) use sparse attention: "L", "H", "both"
    
    # Point 3: Constraint satisfaction head
    use_constraint_head: bool = True
    constraint_head_weight: float = 0.5  # Loss weight for constraint satisfaction
    use_violation_loss: bool = True  # Also use differentiable violation counter
    violation_loss_weight: float = 0.1
    
    # Point 5: GNN layers
    use_gnn_layers: bool = True
    gnn_type: str = "mpnn"  # "basic", "gat", or "mpnn"
    gnn_layers_per_level: int = 1  # Number of GNN layers per H/L level
    gnn_interleave: bool = True  # Interleave GNN with Transformer or use before
    gnn_edge_dim: int = 32  # Edge feature dimension for MPNN


class HierarchicalReasoningModel_V2Block(nn.Module):
    """
    Transformer block with optional constraint-aware attention.
    
    Differences from v1:
    - Can use ConstraintSparseAttention instead of full Attention
    - Post-norm architecture (same as v1)
    """
    
    def __init__(
        self,
        config: HierarchicalReasoningModel_V2Config,
        use_sparse_attention: bool = False
    ) -> None:
        super().__init__()
        
        self.use_sparse_attention = use_sparse_attention
        
        if use_sparse_attention:
            # Point 2: Constraint-aware sparse attention
            puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
            self.self_attn = ConstraintSparseAttention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                seq_len=config.seq_len,
                puzzle_emb_len=puzzle_emb_len if config.puzzle_emb_ndim > 0 else 0,
                use_sparse_mask=True,
                causal=False
            )
        else:
            # Standard full attention
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-Norm architecture (same as v1)
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )
        return hidden_states


class HierarchicalReasoningModel_V2ReasoningModule(nn.Module):
    """
    Reasoning module with optional GNN layers.
    
    Differences from v1:
    - Can interleave GNN layers with Transformer blocks
    - Supports different GNN architectures (basic, GAT, MPNN)
    """
    
    def __init__(
        self,
        transformer_layers: List[HierarchicalReasoningModel_V2Block],
        gnn_layers: Optional[List[nn.Module]] = None,
        gnn_interleave: bool = True,
        puzzle_emb_len: int = 0
    ):
        super().__init__()
        
        self.transformer_layers = torch.nn.ModuleList(transformer_layers)
        self.gnn_layers = torch.nn.ModuleList(gnn_layers) if gnn_layers else None
        self.gnn_interleave = gnn_interleave
        self.puzzle_emb_len = puzzle_emb_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # Input injection (same as v1)
        hidden_states = hidden_states + input_injection
        
        if self.gnn_layers is None:
            # No GNN layers, standard transformer forward
            for layer in self.transformer_layers:
                hidden_states = layer(hidden_states=hidden_states, **kwargs)
        elif self.gnn_interleave:
            # Interleave GNN and Transformer layers
            gnn_idx = 0
            for i, layer in enumerate(self.transformer_layers):
                hidden_states = layer(hidden_states=hidden_states, **kwargs)
                
                # Apply GNN after each transformer layer (if available)
                if gnn_idx < len(self.gnn_layers):
                    hidden_states = self.gnn_layers[gnn_idx](
                        hidden_states, puzzle_emb_len=self.puzzle_emb_len
                    )
                    gnn_idx += 1
        else:
            # Apply all GNN layers first, then transformer
            for gnn_layer in self.gnn_layers:
                hidden_states = gnn_layer(hidden_states, puzzle_emb_len=self.puzzle_emb_len)
            
            for layer in self.transformer_layers:
                hidden_states = layer(hidden_states=hidden_states, **kwargs)
        
        return hidden_states


class HierarchicalReasoningModel_V2_Inner(nn.Module):
    """
    Inner model with hierarchical reasoning and structural biases.
    
    Key changes from v1:
    1. L_level can use sparse attention (Point 2)
    2. Both levels can have GNN layers (Point 5)
    3. Constraint satisfaction head added (Point 3)
    """
    
    def __init__(self, config: HierarchicalReasoningModel_V2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O (same as v1)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype
            )

        # Position encodings (same as v1)
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype
            )
        else:
            raise NotImplementedError()

        # === Build reasoning modules with structural biases ===
        
        # Determine which levels use sparse attention (Point 2)
        h_sparse = config.use_sparse_attention and config.sparse_attention_layers in ["H", "both"]
        l_sparse = config.use_sparse_attention and config.sparse_attention_layers in ["L", "both"]
        
        # Build GNN layers (Point 5)
        def make_gnn_layers(num_layers: int) -> Optional[List[nn.Module]]:
            if not config.use_gnn_layers or num_layers == 0:
                return None
            
            layers = []
            for _ in range(num_layers):
                if config.gnn_type == "basic":
                    layers.append(ConstraintGNNLayer(
                        hidden_size=config.hidden_size,
                        num_edge_types=3,
                        aggregation='mean',
                        residual=True
                    ))
                elif config.gnn_type == "gat":
                    layers.append(ConstraintGATLayer(
                        hidden_size=config.hidden_size,
                        num_heads=4,
                        num_edge_types=3,
                        residual=True
                    ))
                elif config.gnn_type == "mpnn":
                    from .gnn_layers import ConstraintMPNNLayer
                    layers.append(ConstraintMPNNLayer(
                        hidden_size=config.hidden_size,
                        edge_dim=config.gnn_edge_dim,
                        num_edge_types=3,
                        residual=True
                    ))
                else:
                    raise ValueError(f"Unknown GNN type: {config.gnn_type}")
            return layers
        
        # Build H-level (high-level reasoning)
        h_transformer_layers = [
            HierarchicalReasoningModel_V2Block(self.config, use_sparse_attention=h_sparse)
            for _ in range(self.config.H_layers)
        ]
        h_gnn_layers = make_gnn_layers(config.gnn_layers_per_level if config.use_gnn_layers else 0)
        
        self.H_level = HierarchicalReasoningModel_V2ReasoningModule(
            transformer_layers=h_transformer_layers,
            gnn_layers=h_gnn_layers,
            gnn_interleave=config.gnn_interleave,
            puzzle_emb_len=self.puzzle_emb_len if config.puzzle_emb_ndim > 0 else 0
        )
        
        # Build L-level (low-level reasoning) with sparse attention
        l_transformer_layers = [
            HierarchicalReasoningModel_V2Block(self.config, use_sparse_attention=l_sparse)
            for _ in range(self.config.L_layers)
        ]
        l_gnn_layers = make_gnn_layers(config.gnn_layers_per_level if config.use_gnn_layers else 0)
        
        self.L_level = HierarchicalReasoningModel_V2ReasoningModule(
            transformer_layers=l_transformer_layers,
            gnn_layers=l_gnn_layers,
            gnn_interleave=config.gnn_interleave,
            puzzle_emb_len=self.puzzle_emb_len if config.puzzle_emb_ndim > 0 else 0
        )
        
        # Initial states (same as v1)
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Q head init (same as v1)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
        
        # === Point 3: Constraint satisfaction head ===
        if self.config.use_constraint_head:
            self.constraint_head = ConstraintSatisfactionHead(
                hidden_size=self.config.hidden_size,
                num_constraints=27,  # Sudoku: 9 rows + 9 cols + 9 boxes
                num_cells=81,
                use_global_pooling=True,
                use_local_features=True
            )
        
        if self.config.use_violation_loss:
            self.violation_counter = ConstraintViolationCounter(
                num_digits=9,
                temperature=0.1
            )

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Token and puzzle embedding (same as v1)."""
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_V2InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                           self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                           self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_V2InnerCarry):
        return HierarchicalReasoningModel_V2InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_V2InnerCarry,
        batch: Dict[str, torch.Tensor],
        probe_recorder: Optional[object] = None,
        step_index: Optional[int] = None
    ) -> Tuple[HierarchicalReasoningModel_V2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with hierarchical reasoning.
        
        Returns additional outputs for constraint losses.
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations (no grad for memory efficiency)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

                # Probe recording
                if probe_recorder is not None and step_index is not None:
                    try:
                        probe_recorder.record_hidden(
                            step_index=step_index, phase="nograd",
                            z_H=z_H, z_L=z_L, batch=batch
                        )
                    except Exception:
                        pass

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step with gradients
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Probe recording for grad step
        if probe_recorder is not None and step_index is not None:
            try:
                probe_recorder.record_hidden(
                    step_index=step_index, phase="grad",
                    z_H=z_H, z_L=z_L, batch=batch
                )
            except Exception:
                pass

        # LM outputs
        new_carry = HierarchicalReasoningModel_V2InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head for halting
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        # === Additional outputs for structural losses ===
        auxiliary_outputs = {}
        
        # Point 3: Constraint satisfaction predictions
        if self.config.use_constraint_head:
            constraint_logits = self.constraint_head(z_H, puzzle_emb_len=self.puzzle_emb_len)
            auxiliary_outputs["constraint_logits"] = constraint_logits
        
        # Differentiable violation score
        if self.config.use_violation_loss:
            total_violation, per_constraint = self.violation_counter(
                output.unsqueeze(0) if output.dim() == 2 else output,
                puzzle_emb_len=0,  # Already stripped puzzle embeddings
                digit_offset=2
            )
            auxiliary_outputs["violation_score"] = total_violation
            auxiliary_outputs["per_constraint_violations"] = per_constraint
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), auxiliary_outputs


class HierarchicalReasoningModel_V2(nn.Module):
    """
    ACT wrapper for HRM v2.
    
    Same structure as v1 but with additional auxiliary losses.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_V2Config(**config_dict)
        self.inner = HierarchicalReasoningModel_V2_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_V2Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(
        self,
        carry: HierarchicalReasoningModel_V2Carry,
        batch: Dict[str, torch.Tensor],
        probe_recorder: Optional[object] = None
    ) -> Tuple[HierarchicalReasoningModel_V2Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT control.
        
        Returns outputs dict with additional auxiliary outputs for losses.
        """
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)),
                batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        step_idx = int(new_steps.max().item()) if torch.is_tensor(new_steps) else None
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), auxiliary_outputs = self.inner(
            new_inner_carry, new_current_data,
            probe_recorder=probe_recorder, step_index=step_idx
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "intermediate_preds_step": logits.argmax(-1),
            **auxiliary_outputs  # Include constraint outputs
        }
        
        with torch.no_grad():
            # Step counting
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # ACT exploration during training
            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Target Q computation
                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry, new_current_data
                )[2]  # Index 2 is the Q-logits tuple
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(
                    is_last_step, next_q_halt_logits,
                    torch.maximum(next_q_halt_logits, next_q_continue_logits)
                ))

        return HierarchicalReasoningModel_V2Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
