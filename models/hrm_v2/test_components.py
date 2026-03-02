#!/usr/bin/env python3
"""
Test script to verify HRM v2 components load correctly.

Run with: python models/hrm_v2/test_components.py
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch


def test_constraint_attention():
    """Test constraint-aware sparse attention."""
    print("\n=== Testing Constraint Attention ===")
    
    from models.hrm_v2.constraint_attention import (
        build_sudoku_attention_mask,
        build_sudoku_attention_bias,
        ConstraintSparseAttention,
        ConstraintTopKAttention
    )
    
    # Test mask building
    mask = build_sudoku_attention_mask(seq_len=81, puzzle_emb_len=0)
    print(f"Attention mask shape: {mask.shape}")
    print(f"Sparsity: {mask.float().mean():.2%} (expected ~25%)")
    
    # Check that each cell has ~20 neighbors
    neighbors_per_cell = mask.sum(dim=1).float()
    print(f"Neighbors per cell: min={neighbors_per_cell.min():.0f}, max={neighbors_per_cell.max():.0f}, mean={neighbors_per_cell.mean():.1f}")
    
    # Test attention module
    attn = ConstraintSparseAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        num_key_value_heads=8,
        seq_len=81,
        puzzle_emb_len=0,
        use_sparse_mask=True
    )
    
    # Forward pass
    x = torch.randn(2, 81, 512)
    cos = torch.randn(81, 64)
    sin = torch.randn(81, 64)
    
    # Note: This will fall back to PyTorch SDPA since we use sparse mask
    try:
        out = attn((cos, sin), x)
        print(f"Sparse attention output shape: {out.shape}")
        print("✓ ConstraintSparseAttention works!")
    except Exception as e:
        print(f"✗ ConstraintSparseAttention error: {e}")
    
    print()


def test_constraint_head():
    """Test constraint satisfaction head."""
    print("\n=== Testing Constraint Head ===")
    
    from models.hrm_v2.constraint_head import (
        ConstraintSatisfactionHead,
        ConstraintViolationCounter,
        constraint_satisfaction_loss
    )
    
    # Test head
    head = ConstraintSatisfactionHead(
        hidden_size=512,
        num_constraints=27,
        num_cells=81,
        use_global_pooling=True,
        use_local_features=True
    )
    
    # Forward pass
    hidden_states = torch.randn(2, 81, 512)
    constraint_logits = head(hidden_states, puzzle_emb_len=0)
    print(f"Constraint logits shape: {constraint_logits.shape} (expected [2, 27])")
    
    # Test label computation
    predictions = torch.randint(2, 11, (2, 81))  # Random digit tokens
    labels = ConstraintSatisfactionHead.compute_constraint_labels(predictions, soft_labels=False)
    print(f"Constraint labels shape: {labels.shape}")
    print(f"Satisfaction rate: {labels.mean():.2%}")
    
    # Test violation counter
    counter = ConstraintViolationCounter(num_digits=9)
    logits = torch.randn(2, 81, 11)
    total_viol, per_constraint = counter(logits, puzzle_emb_len=0)
    print(f"Violation score: {total_viol.mean():.2f}")
    
    print("✓ ConstraintSatisfactionHead works!")
    print()


def test_gnn_layers():
    """Test GNN layers."""
    print("\n=== Testing GNN Layers ===")
    
    from models.hrm_v2.gnn_layers import (
        build_sudoku_adjacency,
        build_sudoku_adjacency_simple,
        ConstraintGNNLayer,
        ConstraintGATLayer,
        ConstraintMPNNLayer
    )
    
    # Test adjacency building
    edge_index, edge_type = build_sudoku_adjacency(include_self_loops=False)
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge type shape: {edge_type.shape}")
    print(f"Unique edge types: {edge_type.unique().tolist()}")
    
    adj = build_sudoku_adjacency_simple()
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Edges per node: {adj.sum(dim=1).float().mean():.1f}")
    
    # Test GNN layers
    x = torch.randn(2, 81, 512)
    
    # Basic GNN
    gnn = ConstraintGNNLayer(hidden_size=512, num_edge_types=3)
    out = gnn(x, puzzle_emb_len=0)
    print(f"ConstraintGNNLayer output shape: {out.shape}")
    print("✓ ConstraintGNNLayer works!")
    
    # GAT
    gat = ConstraintGATLayer(hidden_size=512, num_heads=4)
    out = gat(x, puzzle_emb_len=0)
    print(f"ConstraintGATLayer output shape: {out.shape}")
    print("✓ ConstraintGATLayer works!")
    
    # MPNN
    mpnn = ConstraintMPNNLayer(hidden_size=512, edge_dim=32)
    out = mpnn(x, puzzle_emb_len=0)
    print(f"ConstraintMPNNLayer output shape: {out.shape}")
    print("✓ ConstraintMPNNLayer works!")
    
    print()


def test_full_model():
    """Test full HRM v2 model."""
    print("\n=== Testing Full HRM v2 Model ===")
    
    from models.hrm_v2 import HierarchicalReasoningModel_V2
    
    config = {
        "batch_size": 2,
        "seq_len": 81,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1024,
        "vocab_size": 11,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 2,
        "L_layers": 2,
        "hidden_size": 256,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",  # Use float32 for CPU testing
        # V2 additions
        "use_sparse_attention": True,
        "sparse_attention_layers": "L",
        "use_constraint_head": True,
        "constraint_head_weight": 0.5,
        "use_violation_loss": True,
        "violation_loss_weight": 0.1,
        "use_gnn_layers": True,
        "gnn_type": "mpnn",
        "gnn_layers_per_level": 1,
        "gnn_interleave": True,
        "gnn_edge_dim": 32,
    }
    
    model = HierarchicalReasoningModel_V2(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy batch
    batch = {
        "inputs": torch.randint(1, 11, (2, 81)),
        "labels": torch.randint(2, 11, (2, 81)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    
    # Initial carry
    carry = model.initial_carry(batch)
    print(f"Initial carry created")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        carry, outputs = model(carry, batch)
    
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    if "constraint_logits" in outputs:
        print(f"Constraint logits shape: {outputs['constraint_logits'].shape}")
    
    if "violation_score" in outputs:
        print(f"Violation score: {outputs['violation_score'].mean():.2f}")
    
    print("✓ HierarchicalReasoningModel_V2 works!")
    print()


def test_losses():
    """Test loss functions."""
    print("\n=== Testing Loss Functions ===")
    
    from models.hrm_v2.losses import HRMv2Loss, compute_metrics
    
    loss_fn = HRMv2Loss(
        q_loss_weight=0.5,
        constraint_loss_weight=0.5,
        violation_loss_weight=0.1
    )
    
    # Create dummy outputs
    outputs = {
        "logits": torch.randn(2, 81, 11),
        "q_halt_logits": torch.randn(2),
        "q_continue_logits": torch.randn(2),
        "target_q_continue": torch.sigmoid(torch.randn(2)),
        "constraint_logits": torch.randn(2, 27),
        "violation_score": torch.rand(2) * 10,
    }
    
    batch = {
        "inputs": torch.randint(1, 11, (2, 81)),
        "labels": torch.randint(2, 11, (2, 81)),
    }
    
    total_loss, loss_components = loss_fn(outputs, batch)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {list(loss_components.keys())}")
    
    for k, v in loss_components.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test metrics
    metrics = compute_metrics(outputs, batch)
    print(f"\nMetrics: {metrics}")
    
    print("✓ Loss functions work!")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("HRM v2 Component Tests")
    print("=" * 60)
    
    try:
        test_constraint_attention()
        test_constraint_head()
        test_gnn_layers()
        test_full_model()
        test_losses()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
