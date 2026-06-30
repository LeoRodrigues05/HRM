"""
Point 2: Constraint-Aware Sparse Attention

This module implements sparse attention patterns that respect the constraint
structure of the reasoning task (e.g., Sudoku rows/columns/boxes).

Key insight: In Sudoku, a cell only needs to attend to cells in its:
- Same row (8 other cells)
- Same column (8 other cells)  
- Same 3x3 box (8 other cells, 4 overlap with row/col)

This gives 20 unique neighbors per cell instead of 80 (full attention).
The sparse pattern provides an inductive bias that the model would otherwise
need to learn from data.

Two implementations are provided:
1. ConstraintSparseAttention: Uses explicit attention masks with FlashAttention
2. ConstraintBlockSparseAttention: Block-sparse implementation for efficiency
"""

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        # Optional: HRM v2 is not used in the BPTT/SAE training path. Guard the
        # import so modules that transitively import this file still load without
        # flash-attn installed.
        flash_attn_func = None  # type: ignore[assignment]

from models.common import trunc_normal_init_
from models.layers import CastedLinear, apply_rotary_pos_emb, CosSin


def build_sudoku_attention_mask(
    seq_len: int = 81,
    puzzle_emb_len: int = 0,
    include_self: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Build a sparse attention mask for Sudoku constraint structure.
    
    For each cell (i, j) in the 9x9 grid, it can attend to:
    - All cells in the same row
    - All cells in the same column
    - All cells in the same 3x3 box
    
    Args:
        seq_len: Sequence length (81 for Sudoku grid)
        puzzle_emb_len: Number of puzzle embedding tokens prepended
        include_self: Whether a cell attends to itself
        device: Target device
        
    Returns:
        mask: [total_len, total_len] boolean tensor where True = can attend
    """
    total_len = puzzle_emb_len + seq_len
    
    # Start with puzzle embeddings attending to everything (if present)
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # Puzzle embedding tokens can attend to everything
    if puzzle_emb_len > 0:
        mask[:puzzle_emb_len, :] = True
        mask[:, :puzzle_emb_len] = True
    
    # Build Sudoku constraint mask for the 81-cell grid
    for idx in range(81):
        row, col = idx // 9, idx % 9
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        
        pos = puzzle_emb_len + idx
        
        for other_idx in range(81):
            other_row, other_col = other_idx // 9, other_idx % 9
            other_box_row, other_box_col = (other_row // 3) * 3, (other_col // 3) * 3
            
            other_pos = puzzle_emb_len + other_idx
            
            # Same cell
            if idx == other_idx:
                mask[pos, other_pos] = include_self
                continue
            
            # Same row
            if row == other_row:
                mask[pos, other_pos] = True
                continue
                
            # Same column
            if col == other_col:
                mask[pos, other_pos] = True
                continue
                
            # Same 3x3 box
            if box_row == other_box_row and box_col == other_box_col:
                mask[pos, other_pos] = True
                continue
    
    return mask


def build_sudoku_attention_bias(
    seq_len: int = 81,
    puzzle_emb_len: int = 0,
    mask_value: float = -1e9,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Build attention bias tensor for Sudoku (additive mask for softmax).
    
    Returns bias where:
    - 0 for allowed attention
    - large negative for disallowed attention
    
    Args:
        seq_len: Sequence length (81 for Sudoku)
        puzzle_emb_len: Number of puzzle embedding tokens
        mask_value: Large negative value for masked positions
        device: Target device
        dtype: Data type
        
    Returns:
        bias: [total_len, total_len] float tensor
    """
    bool_mask = build_sudoku_attention_mask(
        seq_len=seq_len,
        puzzle_emb_len=puzzle_emb_len,
        include_self=True,
        device=device
    )
    
    bias = torch.where(bool_mask, torch.tensor(0.0, dtype=dtype, device=device),
                       torch.tensor(mask_value, dtype=dtype, device=device))
    return bias


class ConstraintSparseAttention(nn.Module):
    """
    Attention module with constraint-aware sparse patterns.
    
    Unlike standard full attention, this module restricts attention to
    structurally relevant positions based on the constraint graph.
    
    For Sudoku: each cell only attends to its row, column, and box neighbors.
    This is an inductive bias that helps the model learn constraint propagation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        seq_len: int = 81,
        puzzle_emb_len: int = 0,
        use_sparse_mask: bool = True,
        causal: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.use_sparse_mask = use_sparse_mask
        
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        
        # QKV projections
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        
        # Pre-compute and register the attention mask as a buffer
        if use_sparse_mask:
            mask = build_sudoku_attention_mask(
                seq_len=seq_len,
                puzzle_emb_len=puzzle_emb_len,
                include_self=True
            )
            # Convert to attention bias format for scaled dot-product attention
            # FlashAttention doesn't support arbitrary masks, so we use PyTorch SDPA fallback
            self.register_buffer('attention_mask', mask, persistent=False)
        else:
            self.attention_mask = None
    
    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        attention_mask_override: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional sparse attention.
        
        Args:
            cos_sin: Tuple of (cos, sin) for RoPE
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask_override: Optional custom mask [seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # Apply RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Determine which mask to use
        mask = attention_mask_override if attention_mask_override is not None else self.attention_mask
        
        if mask is not None and self.use_sparse_mask:
            # Use PyTorch scaled_dot_product_attention with mask
            # Reshape for attention: [batch, heads, seq, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Expand key/value heads if using grouped query attention
            if self.num_key_value_heads < self.num_heads:
                repeat_factor = self.num_heads // self.num_key_value_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)
            
            # Convert bool mask to float mask for SDPA
            # True (can attend) -> 0, False (cannot attend) -> -inf
            attn_mask = torch.where(
                mask,
                torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.tensor(float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype)
            )
            
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                is_causal=False  # We handle masking explicitly
            )
            
            # Reshape back: [batch, seq, heads * head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        else:
            # Use FlashAttention for full attention (faster)
            attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)


class ConstraintTopKAttention(nn.Module):
    """
    Attention with learned top-k sparsity based on uncertainty.
    
    This module learns to attend to the k most "uncertain" or "relevant"
    positions, dynamically adapting the attention pattern based on content.
    
    This mimics how human solvers focus on cells with fewest candidates.
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        top_k: int = 20,  # ~20 neighbors in Sudoku constraint graph
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.top_k = top_k
        self.temperature = temperature
        
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        
        # Uncertainty scorer: predicts which positions need attention
        self.uncertainty_proj = CastedLinear(hidden_size, 1, bias=True)
    
    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute uncertainty scores for routing
        uncertainty = self.uncertainty_proj(hidden_states).squeeze(-1)  # [batch, seq]
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # Apply RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Reshape for attention
        query = query.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Expand KV heads if needed
        if self.num_key_value_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply top-k masking based on uncertainty-weighted scores
        # Positions with higher uncertainty get priority
        uncertainty_expanded = uncertainty.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
        routing_scores = attn_weights + uncertainty_expanded / self.temperature
        
        # Keep only top-k attention connections per query position
        k = min(self.top_k, seq_len)
        topk_values, topk_indices = torch.topk(routing_scores, k=k, dim=-1)
        
        # Create sparse mask
        mask = torch.full_like(attn_weights, float('-inf'))
        mask.scatter_(-1, topk_indices, 0.0)
        
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)
