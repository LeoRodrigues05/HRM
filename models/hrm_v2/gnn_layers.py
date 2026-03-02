"""
Point 5: Graph Neural Network Layers for Constraint Structure

This module implements GNN layers that explicitly encode the constraint graph
topology of the reasoning task (e.g., Sudoku).

Key insight: Standard Transformers must learn the constraint structure from data
via attention patterns. By providing GNN layers that explicitly encode which
cells are connected by constraints, we give strong structural inductive bias.

For Sudoku, the constraint graph has:
- 81 nodes (cells)
- Edges connecting cells that share a constraint (row, column, or box)
- Each cell has 20 unique neighbors

Three GNN variants are implemented:
1. ConstraintGNNLayer: Basic message-passing with constraint-type encoding
2. ConstraintGATLayer: Graph attention with learned edge importance
3. ConstraintMPNNLayer: Message-Passing Neural Network with edge features

These can be interleaved with Transformer blocks for hybrid reasoning.
"""

from typing import Tuple, Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F

from models.layers import CastedLinear, SwiGLU, rms_norm
from models.common import trunc_normal_init_


def rms_layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS layer norm that preserves input dtype."""
    return rms_norm(x, variance_epsilon=eps)


def build_sudoku_adjacency(
    include_self_loops: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build adjacency information for Sudoku constraint graph.
    
    Returns edge_index in COO format: [2, num_edges] where:
    - edge_index[0] = source nodes
    - edge_index[1] = target nodes
    
    Also returns edge_type indicating constraint type:
    - 0: row constraint
    - 1: column constraint  
    - 2: box constraint
    
    Args:
        include_self_loops: Whether to include self-edges
        device: Target device
        
    Returns:
        edge_index: [2, num_edges] source/target node indices
        edge_type: [num_edges] constraint type for each edge
    """
    edges = []
    edge_types = []
    
    for src in range(81):
        src_row, src_col = src // 9, src % 9
        src_box = (src_row // 3) * 3 + (src_col // 3)
        
        if include_self_loops:
            edges.append((src, src))
            edge_types.append(3)  # Self-loop type
        
        for tgt in range(81):
            if src == tgt:
                continue
                
            tgt_row, tgt_col = tgt // 9, tgt % 9
            tgt_box = (tgt_row // 3) * 3 + (tgt_col // 3)
            
            # Check constraint relationships
            # Note: we add edges for ALL shared constraints (a cell can share
            # multiple constraint types with another cell)
            
            if src_row == tgt_row:
                edges.append((src, tgt))
                edge_types.append(0)  # Row constraint
            
            if src_col == tgt_col:
                edges.append((src, tgt))
                edge_types.append(1)  # Column constraint
            
            # Only add box edge if not already connected by row/col
            if src_box == tgt_box and src_row != tgt_row and src_col != tgt_col:
                edges.append((src, tgt))
                edge_types.append(2)  # Box constraint
    
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
    edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
    
    return edge_index, edge_type


def build_sudoku_adjacency_simple(
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Build simple adjacency matrix for Sudoku (no edge types).
    
    Returns:
        adj: [81, 81] boolean adjacency matrix
    """
    adj = torch.zeros(81, 81, dtype=torch.bool, device=device)
    
    for src in range(81):
        src_row, src_col = src // 9, src % 9
        src_box_row, src_box_col = (src_row // 3) * 3, (src_col // 3) * 3
        
        for tgt in range(81):
            if src == tgt:
                continue
                
            tgt_row, tgt_col = tgt // 9, tgt % 9
            tgt_box_row, tgt_box_col = (tgt_row // 3) * 3, (tgt_col // 3) * 3
            
            # Connected if same row, column, or box
            if src_row == tgt_row or src_col == tgt_col:
                adj[src, tgt] = True
            elif src_box_row == tgt_box_row and src_box_col == tgt_box_col:
                adj[src, tgt] = True
    
    return adj


class ConstraintGNNLayer(nn.Module):
    """
    Graph Neural Network layer for constraint-structured message passing.
    
    This layer performs message passing along the constraint graph, where
    messages are typed by the constraint relationship (row/col/box).
    
    The update rule is:
        h_i' = MLP(h_i + Σ_{j∈N(i)} W_{type(i,j)} * h_j)
    
    where type(i,j) indicates the constraint type between nodes i and j.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_edge_types: int = 3,  # row, col, box
        aggregation: str = 'mean',
        dropout: float = 0.0,
        residual: bool = True
    ):
        """
        Args:
            hidden_size: Node feature dimension
            num_edge_types: Number of constraint types (3 for Sudoku)
            aggregation: How to aggregate messages ('mean', 'sum', 'max')
            dropout: Dropout probability
            residual: Whether to use residual connection
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.aggregation = aggregation
        self.residual = residual
        
        # Per-edge-type message transformation
        self.message_mlps = nn.ModuleList([
            CastedLinear(hidden_size, hidden_size, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # Update MLP after aggregation
        self.update_mlp = nn.Sequential(
            CastedLinear(hidden_size, hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            CastedLinear(hidden_size * 2, hidden_size, bias=True)
        )
        
        # Store norm eps for RMS norm
        self.norm_eps = 1e-5
        
        # Pre-compute adjacency (registered as buffer)
        edge_index, edge_type = build_sudoku_adjacency(include_self_loops=False)
        self.register_buffer('edge_index', edge_index, persistent=False)
        self.register_buffer('edge_type', edge_type, persistent=False)
        
        # Pre-compute degree for normalization
        self._precompute_degrees()
    
    def _precompute_degrees(self):
        """Compute node degrees for message normalization."""
        # Count incoming edges per node
        degrees = torch.zeros(81, dtype=torch.float32)
        for i in range(self.edge_index.shape[1]):
            tgt = self.edge_index[1, i].item()
            degrees[tgt] += 1
        
        # Avoid division by zero
        degrees = degrees.clamp(min=1)
        self.register_buffer('degrees', degrees, persistent=False)
    
    def forward(
        self,
        node_features: torch.Tensor,
        puzzle_emb_len: int = 0
    ) -> torch.Tensor:
        """
        Forward pass with message passing.
        
        Args:
            node_features: [batch_size, seq_len, hidden_size]
            puzzle_emb_len: Number of puzzle embedding tokens
            
        Returns:
            updated_features: [batch_size, seq_len, hidden_size]
        """
        batch_size = node_features.shape[0]
        
        # Extract cell features (skip puzzle embeddings)
        if puzzle_emb_len > 0:
            prefix = node_features[:, :puzzle_emb_len]
            cell_features = node_features[:, puzzle_emb_len:puzzle_emb_len + 81]
        else:
            prefix = None
            cell_features = node_features[:, :81]
        
        # Initialize aggregated messages
        aggregated = torch.zeros_like(cell_features)
        
        # Process each edge type separately
        for edge_type_idx in range(self.num_edge_types):
            # Find edges of this type
            type_mask = self.edge_type == edge_type_idx
            type_edges = self.edge_index[:, type_mask]  # [2, num_type_edges]
            
            if type_edges.shape[1] == 0:
                continue
            
            src_nodes = type_edges[0]  # [num_type_edges]
            tgt_nodes = type_edges[1]  # [num_type_edges]
            
            # Get source node features
            src_features = cell_features[:, src_nodes]  # [batch, num_edges, hidden]
            
            # Transform messages
            messages = self.message_mlps[edge_type_idx](src_features).to(dtype=cell_features.dtype)
            
            # Scatter-add messages to target nodes
            # Using index_add for efficiency
            for b in range(batch_size):
                aggregated[b].index_add_(0, tgt_nodes, messages[b])
        
        # Normalize by degree
        aggregated = aggregated / self.degrees.view(1, 81, 1)
        
        # Update with MLP (cast back to input dtype)
        updated = self.update_mlp(aggregated).to(dtype=cell_features.dtype)
        
        # Residual connection and RMS norm (preserves dtype)
        if self.residual:
            updated = rms_layer_norm(cell_features + updated, self.norm_eps)
        else:
            updated = rms_layer_norm(updated, self.norm_eps)
        
        # Reconstruct full sequence
        if prefix is not None:
            output = torch.cat([prefix, updated, node_features[:, puzzle_emb_len + 81:]], dim=1)
        else:
            output = torch.cat([updated, node_features[:, 81:]], dim=1)
        
        return output


class ConstraintGATLayer(nn.Module):
    """
    Graph Attention layer with constraint-aware edge attention.
    
    Unlike ConstraintGNNLayer, this layer learns attention weights for
    edges, allowing the model to dynamically weight different neighbors
    based on content.
    
    The attention mechanism is:
        α_{ij} = softmax_j(LeakyReLU(a^T [W h_i || W h_j || e_{type(i,j)}]))
        h_i' = σ(Σ_{j∈N(i)} α_{ij} W h_j)
    
    where e_{type} is a learned embedding for each constraint type.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
        include_self_loops: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # Add 1 for self-loop type if included
        self.num_edge_types = num_edge_types + 1 if include_self_loops else num_edge_types
        self.negative_slope = negative_slope
        self.residual = residual
        self.include_self_loops = include_self_loops
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Node feature projection
        self.W = CastedLinear(hidden_size, hidden_size, bias=False)
        
        # Attention parameters (per head) - include self-loop type
        self.attn_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.attn_tgt = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.attn_edge = nn.Parameter(torch.zeros(num_heads, self.num_edge_types))
        
        trunc_normal_init_(self.attn_src, std=0.02)
        trunc_normal_init_(self.attn_tgt, std=0.02)
        trunc_normal_init_(self.attn_edge, std=0.02)
        
        # Edge type embedding - include self-loop type
        self.edge_type_emb = nn.Embedding(self.num_edge_types, self.head_dim)
        
        # Output projection
        self.out_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_eps = 1e-5
        
        # Pre-compute adjacency
        edge_index, edge_type = build_sudoku_adjacency(include_self_loops=include_self_loops)
        self.register_buffer('edge_index', edge_index, persistent=False)
        self.register_buffer('edge_type', edge_type, persistent=False)
    
    def forward(
        self,
        node_features: torch.Tensor,
        puzzle_emb_len: int = 0
    ) -> torch.Tensor:
        """
        Forward pass with graph attention.
        
        Args:
            node_features: [batch_size, seq_len, hidden_size]
            puzzle_emb_len: Number of puzzle embedding tokens
            
        Returns:
            updated_features: [batch_size, seq_len, hidden_size]
        """
        batch_size = node_features.shape[0]
        
        # Extract cell features
        if puzzle_emb_len > 0:
            prefix = node_features[:, :puzzle_emb_len]
            cell_features = node_features[:, puzzle_emb_len:puzzle_emb_len + 81]
        else:
            prefix = None
            cell_features = node_features[:, :81]
        
        # Project features
        h = self.W(cell_features)  # [batch, 81, hidden]
        h = h.view(batch_size, 81, self.num_heads, self.head_dim)  # [batch, 81, heads, head_dim]
        
        # Get source and target node features for edges
        src_nodes = self.edge_index[0]  # [num_edges]
        tgt_nodes = self.edge_index[1]  # [num_edges]
        num_edges = src_nodes.shape[0]
        
        h_src = h[:, src_nodes]  # [batch, num_edges, heads, head_dim]
        h_tgt = h[:, tgt_nodes]  # [batch, num_edges, heads, head_dim]
        
        # Compute attention scores
        # a^T [h_i || h_j] decomposed as a_src^T h_i + a_tgt^T h_j
        attn_src = (h_src * self.attn_src).sum(dim=-1)  # [batch, num_edges, heads]
        attn_tgt = (h_tgt * self.attn_tgt).sum(dim=-1)  # [batch, num_edges, heads]
        
        # Edge type contribution
        edge_type_scores = self.attn_edge[:, self.edge_type].T  # [num_edges, heads]
        edge_type_scores = edge_type_scores.unsqueeze(0)  # [1, num_edges, heads]
        
        attn_scores = attn_src + attn_tgt + edge_type_scores
        attn_scores = F.leaky_relu(attn_scores, negative_slope=self.negative_slope)
        
        # Softmax over neighbors (need to do per-node)
        # Create attention coefficient matrix [batch, 81, 81, heads] sparse -> dense for softmax
        attn_matrix = torch.full(
            (batch_size, 81, 81, self.num_heads),
            float('-inf'),
            device=node_features.device,
            dtype=node_features.dtype
        )
        
        # Fill in attention scores
        for i in range(num_edges):
            src, tgt = src_nodes[i].item(), tgt_nodes[i].item()
            attn_matrix[:, tgt, src, :] = attn_scores[:, i, :]
        
        # Softmax over source dimension (neighbors)
        attn_weights = F.softmax(attn_matrix, dim=2)  # [batch, 81, 81, heads]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        # h: [batch, 81, heads, head_dim] -> [batch, 81, 81, heads, head_dim]
        h_expanded = h.unsqueeze(1).expand(-1, 81, -1, -1, -1)
        weighted = attn_weights.unsqueeze(-1) * h_expanded  # [batch, 81, 81, heads, head_dim]
        
        # Sum over neighbors
        output = weighted.sum(dim=2)  # [batch, 81, heads, head_dim]
        output = output.reshape(batch_size, 81, self.hidden_size)
        
        output = self.out_proj(output).to(dtype=cell_features.dtype)
        
        # Residual and RMS norm (preserves dtype)
        if self.residual:
            output = rms_layer_norm(cell_features + output, self.norm_eps)
        else:
            output = rms_layer_norm(output, self.norm_eps)
        
        # Reconstruct full sequence
        if prefix is not None:
            output = torch.cat([prefix, output, node_features[:, puzzle_emb_len + 81:]], dim=1)
        else:
            output = torch.cat([output, node_features[:, 81:]], dim=1)
        
        return output


class ConstraintMPNNLayer(nn.Module):
    """
    Message-Passing Neural Network layer with edge features.
    
    This layer uses explicit edge feature vectors (not just type embeddings)
    to compute messages. The edge features can encode:
    - Constraint type (row/col/box)
    - Relative position within constraint
    - Constraint difficulty/priority
    
    The update rule is:
        m_{ij} = MLP_msg([h_i || h_j || e_{ij}])
        h_i' = MLP_update(h_i + Σ_{j∈N(i)} m_{ij})
    """
    
    def __init__(
        self,
        hidden_size: int,
        edge_dim: int = 32,
        num_edge_types: int = 3,
        aggregation: str = 'mean',
        dropout: float = 0.0,
        residual: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types
        self.aggregation = aggregation
        self.residual = residual
        
        # Edge feature encoding
        self.edge_type_emb = nn.Embedding(num_edge_types + 1, edge_dim)  # +1 for self-loop
        
        # Message MLP: [h_src || h_tgt || e] -> message
        self.message_mlp = nn.Sequential(
            CastedLinear(hidden_size * 2 + edge_dim, hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size, hidden_size, bias=True)
        )
        
        # Update MLP
        self.update_mlp = nn.Sequential(
            CastedLinear(hidden_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            CastedLinear(hidden_size, hidden_size, bias=True)
        )
        
        self.norm_eps = 1e-5
        
        # Pre-compute adjacency with self-loops
        edge_index, edge_type = build_sudoku_adjacency(include_self_loops=True)
        self.register_buffer('edge_index', edge_index, persistent=False)
        self.register_buffer('edge_type', edge_type, persistent=False)
        
        # Degree for normalization
        degrees = torch.zeros(81, dtype=torch.float32)
        for i in range(edge_index.shape[1]):
            degrees[edge_index[1, i]] += 1
        self.register_buffer('degrees', degrees.clamp(min=1), persistent=False)
    
    def forward(
        self,
        node_features: torch.Tensor,
        puzzle_emb_len: int = 0
    ) -> torch.Tensor:
        """
        Forward pass with message passing.
        
        Args:
            node_features: [batch_size, seq_len, hidden_size]
            puzzle_emb_len: Number of puzzle embedding tokens
            
        Returns:
            updated_features: [batch_size, seq_len, hidden_size]
        """
        batch_size = node_features.shape[0]
        
        # Extract cell features
        if puzzle_emb_len > 0:
            prefix = node_features[:, :puzzle_emb_len]
            cell_features = node_features[:, puzzle_emb_len:puzzle_emb_len + 81]
        else:
            prefix = None
            cell_features = node_features[:, :81]
        
        src_nodes = self.edge_index[0]
        tgt_nodes = self.edge_index[1]
        num_edges = src_nodes.shape[0]
        
        # Get node features for edges
        h_src = cell_features[:, src_nodes]  # [batch, num_edges, hidden]
        h_tgt = cell_features[:, tgt_nodes]  # [batch, num_edges, hidden]
        
        # Get edge features
        edge_feat = self.edge_type_emb(self.edge_type)  # [num_edges, edge_dim]
        edge_feat = edge_feat.unsqueeze(0).expand(batch_size, -1, -1)
        edge_feat = edge_feat.to(dtype=node_features.dtype)
        
        # Compute messages
        msg_input = torch.cat([h_src, h_tgt, edge_feat], dim=-1)
        messages = self.message_mlp(msg_input).to(dtype=node_features.dtype)  # [batch, num_edges, hidden]
        
        # Aggregate messages per node
        aggregated = torch.zeros(batch_size, 81, self.hidden_size,
                                 device=node_features.device, dtype=node_features.dtype)
        
        for b in range(batch_size):
            aggregated[b].index_add_(0, tgt_nodes, messages[b])
        
        # Normalize
        if self.aggregation == 'mean':
            aggregated = aggregated / self.degrees.view(1, 81, 1)
        
        # Update (cast back to input dtype)
        update_input = torch.cat([cell_features, aggregated], dim=-1)
        updated = self.update_mlp(update_input).to(dtype=cell_features.dtype)
        
        # Residual and RMS norm (preserves dtype)
        if self.residual:
            output = rms_layer_norm(cell_features + updated, self.norm_eps)
        else:
            output = rms_layer_norm(updated, self.norm_eps)
        
        # Reconstruct full sequence with prefix and suffix
        if prefix is not None:
            output = torch.cat([prefix, output, node_features[:, puzzle_emb_len + 81:]], dim=1)
        else:
            output = torch.cat([output, node_features[:, 81:]], dim=1)
        
        return output
