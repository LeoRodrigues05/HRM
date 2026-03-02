"""
HRM v2: Hierarchical Reasoning Model with Structural Inductive Biases

This module extends the original HRM architecture with three key improvements:

1. Constraint-Aware Sparse Attention (Point 2):
   - Replaces full bidirectional attention in L_level with sparse patterns
   - For Sudoku: attention masks enforce row/column/box connectivity
   - Reduces compute while providing structural inductive bias

2. Explicit Constraint Satisfaction Head (Point 3):
   - Auxiliary output head predicting per-constraint validity
   - For Sudoku: 27 outputs (9 rows + 9 cols + 9 boxes)
   - Direct supervision for constraint satisfaction

3. Graph Neural Network Layers (Point 5):
   - Message-passing layers that respect constraint graph topology
   - Interleaved with Transformer blocks for hybrid reasoning
   - Cells share edges with row/col/box neighbors

These changes aim to move HRM from "soft iterative refinement" toward 
more structured algorithmic reasoning.
"""

from .hrm_v2 import (
    HierarchicalReasoningModel_V2,
    HierarchicalReasoningModel_V2Config,
    HierarchicalReasoningModel_V2Carry,
)
from .constraint_attention import ConstraintSparseAttention, build_sudoku_attention_mask
from .constraint_head import ConstraintSatisfactionHead
from .gnn_layers import ConstraintGNNLayer, build_sudoku_adjacency
from .act_loss_head import ACTLossHeadV2

__all__ = [
    "HierarchicalReasoningModel_V2",
    "HierarchicalReasoningModel_V2Config", 
    "HierarchicalReasoningModel_V2Carry",
    "ConstraintSparseAttention",
    "build_sudoku_attention_mask",
    "ConstraintSatisfactionHead",
    "ConstraintGNNLayer",
    "build_sudoku_adjacency",
    "ACTLossHeadV2",
]
