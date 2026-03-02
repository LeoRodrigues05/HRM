# HRM v2: Hierarchical Reasoning Model with Structural Inductive Biases

This module extends the original HRM architecture with three key improvements designed to move from "soft iterative refinement" toward more structured algorithmic reasoning.

## Motivation

The original HRM performs iterative refinement via nested recurrent cycles (H×L), but experiments show it doesn't encode a principled solver—it essentially "smears" updates across all cells simultaneously without understanding the constraint structure. While accuracy improves across steps (67.9% → 83.7%), the model lacks explicit algorithmic inductive biases.

## Architectural Changes

### Point 2: Constraint-Aware Sparse Attention

**File**: [constraint_attention.py](constraint_attention.py)

**Problem**: Standard full attention in the L_level allows each cell to attend to all 80 other cells, requiring the model to learn which cells are relevant through data.

**Solution**: Replace full attention with sparse attention patterns that respect the constraint graph:
- For Sudoku: each cell only attends to cells in its row (8), column (8), and box (8, with 4 overlapping) = 20 unique neighbors
- Reduces attention complexity from O(81²) to O(81 × 20)
- Provides strong inductive bias: the model "knows" which cells share constraints

**Implementation**:
```python
# In ConstraintSparseAttention
mask = build_sudoku_attention_mask(seq_len=81)
# mask[i, j] = True iff cell i and j share a constraint

# Forward pass uses masked attention
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=attn_mask  # Sparse pattern
)
```

**Variants**:
1. `ConstraintSparseAttention`: Uses explicit boolean mask (flexible, works with any backend)
2. `ConstraintTopKAttention`: Learns to attend to k most "uncertain" positions dynamically

### Point 3: Explicit Constraint Satisfaction Head

**File**: [constraint_head.py](constraint_head.py)

**Problem**: The standard HRM only receives token-level cross-entropy loss, which provides indirect signal for constraint satisfaction. The model must implicitly learn that rows/columns/boxes need unique digits.

**Solution**: Add an auxiliary output head that directly predicts whether each constraint is satisfied:
- For Sudoku: 27 binary outputs (9 rows + 9 cols + 9 boxes)
- Supervised with ground-truth constraint validity
- Provides direct gradient signal for learning constraint rules

**Implementation**:
```python
class ConstraintSatisfactionHead(nn.Module):
    def forward(self, z_H, puzzle_emb_len=0):
        # For each of 27 constraints, pool the 9 relevant cells
        # and predict validity
        constraint_logits = self.constraint_mlp(features)  # [batch, 27]
        return constraint_logits

# Training loss
loss = lm_loss + 0.5 * constraint_satisfaction_loss(constraint_logits, predictions)
```

**Additional Component**: `ConstraintViolationCounter` provides a differentiable approximation of constraint violations using soft probabilities, enabling smooth gradients even when predictions are discrete.

### Point 5: Graph Neural Network Layers

**File**: [gnn_layers.py](gnn_layers.py)

**Problem**: Standard Transformers must learn the constraint structure purely from attention patterns, which requires significant data and may not generalize to new puzzle instances.

**Solution**: Add GNN layers that explicitly encode the constraint graph topology:
- Nodes = 81 cells
- Edges = constraint relationships (typed: row, column, box)
- Message passing propagates information along constraint edges
- Can be interleaved with Transformer blocks for hybrid reasoning

**Implementation**:
```python
class ConstraintGNNLayer(nn.Module):
    def forward(self, node_features):
        # For each edge type (row, col, box):
        #   - Gather source node features
        #   - Apply type-specific transformation
        #   - Aggregate to target nodes
        aggregated = scatter_add(messages, target_nodes)
        return update_mlp(node_features + aggregated)
```

**Variants**:
1. `ConstraintGNNLayer`: Basic typed message passing
2. `ConstraintGATLayer`: Graph attention with learned edge importance
3. `ConstraintMPNNLayer`: Full message-passing NN with edge features

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HRM v2 Forward Pass                                 │
│                                                                             │
│  Input → Token Embed → Puzzle Embed (optional) → Position Embed             │
│                               ↓                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         ACT Loop (max_steps)                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      H_cycle (×2)                                │  │ │
│  │  │  ┌───────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │                   L_cycle (×2)                             │  │  │ │
│  │  │  │                                                            │  │  │ │
│  │  │  │  z_L = L_level(z_L, z_H + input_embeddings)               │  │  │ │
│  │  │  │         │                                                  │  │  │ │
│  │  │  │         ├─ [NEW] Sparse Attention (row/col/box mask)       │  │  │ │
│  │  │  │         ├─ SwiGLU MLP                                      │  │  │ │
│  │  │  │         └─ [NEW] GNN Layer (constraint message passing)    │  │  │ │
│  │  │  └───────────────────────────────────────────────────────────┘  │  │ │
│  │  │                                                                  │  │ │
│  │  │  z_H = H_level(z_H, z_L)                                        │  │ │
│  │  │         │                                                        │  │ │
│  │  │         ├─ Full Attention (global reasoning)                    │  │ │
│  │  │         ├─ SwiGLU MLP                                           │  │ │
│  │  │         └─ [NEW] GNN Layer (optional)                           │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  Outputs:                                                              │ │
│  │   ├─ lm_head(z_H) → Token Logits                                      │ │
│  │   ├─ q_head(z_H[:, 0]) → Halt/Continue Q-values                       │ │
│  │   └─ [NEW] constraint_head(z_H) → Constraint Validity (27)            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Loss = LM_loss + Q_losses + [NEW] Constraint_loss + [NEW] Violation_loss  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

See [config/arch/hrm_v2.yaml](../../config/arch/hrm_v2.yaml) for full configuration options:

```yaml
# Point 2: Sparse Attention
use_sparse_attention: true
sparse_attention_layers: "L"  # L_level uses sparse, H_level uses full

# Point 3: Constraint Head
use_constraint_head: true
constraint_head_weight: 0.5
use_violation_loss: true
violation_loss_weight: 0.1

# Point 5: GNN Layers
use_gnn_layers: true
gnn_type: "mpnn"  # basic, gat, or mpnn
gnn_layers_per_level: 1
gnn_interleave: true
```

## Expected Benefits

1. **Sparse Attention (Point 2)**:
   - Faster training and inference (~4x fewer attention computations in L_level)
   - Better sample efficiency (structure is provided, not learned)
   - More interpretable attention patterns

2. **Constraint Head (Point 3)**:
   - Direct supervision for constraint satisfaction
   - Earlier detection of invalid predictions
   - Smoother loss landscape for constraint-related errors

3. **GNN Layers (Point 5)**:
   - Explicit structural inductive bias
   - Better generalization to unseen puzzles
   - More systematic constraint propagation (similar to real Sudoku algorithms)

## Usage

```python
from models.hrm_v2 import HierarchicalReasoningModel_V2
from models.hrm_v2.losses import HRMv2Loss, compute_metrics

# Load config and create model
config = load_yaml("config/arch/hrm_v2.yaml")
config.update({"batch_size": 32, "seq_len": 81, "vocab_size": 11, ...})
model = HierarchicalReasoningModel_V2(config)

# Create loss function
loss_fn = HRMv2Loss(
    q_loss_weight=0.5,
    constraint_loss_weight=0.5,
    violation_loss_weight=0.1
)

# Training step
carry = model.initial_carry(batch)
carry, outputs = model(carry, batch)
loss, loss_components = loss_fn(outputs, batch)

# Evaluation
metrics = compute_metrics(outputs, batch)
```

## Files

| File | Description |
|------|-------------|
| [hrm_v2.py](hrm_v2.py) | Main model class integrating all components |
| [constraint_attention.py](constraint_attention.py) | Sparse attention implementations |
| [constraint_head.py](constraint_head.py) | Constraint satisfaction head and violation counter |
| [gnn_layers.py](gnn_layers.py) | GNN layer implementations |
| [losses.py](losses.py) | Combined loss function and metrics |

## Experimental Validation

To validate these architectural changes:

1. **Ablation study**: Train models with each component enabled/disabled
2. **Probe analysis**: Use existing probe infrastructure to verify z_H/z_L specialization
3. **Step dynamics**: Compare refinement patterns (should show more targeted updates)
4. **Activation patching**: Verify that structural components are causally important

## References

- Original HRM paper (attached PDF)
- Graph Attention Networks: Veličković et al., 2018
- Message Passing Neural Networks: Gilmer et al., 2017
- Constraint Satisfaction Problems and Neural Networks: various
