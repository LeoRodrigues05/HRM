"""
Point 3: Explicit Constraint Satisfaction Head

This module implements an auxiliary output head that predicts per-constraint
validity, providing direct supervision for constraint satisfaction.

Key insight: The standard HRM only receives token-level cross-entropy loss,
which provides indirect signal for constraint satisfaction. By adding explicit
constraint supervision, we give the model direct gradient signal for learning
to respect Sudoku rules.

For Sudoku, we predict validity of 27 constraints:
- 9 row constraints (each row must contain digits 1-9)
- 9 column constraints (each column must contain digits 1-9)
- 9 box constraints (each 3x3 box must contain digits 1-9)

This can be extended to other constraint satisfaction problems by defining
appropriate constraint extraction functions.
"""

from typing import Dict, List, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

from models.layers import CastedLinear
from models.common import trunc_normal_init_


class ConstraintSatisfactionHead(nn.Module):
    """
    Auxiliary head for predicting constraint satisfaction.
    
    For Sudoku: predicts validity of 27 constraints (9 rows + 9 cols + 9 boxes).
    Can operate on either raw logits or the hidden state z_H.
    
    The head can be supervised with:
    - Hard labels: binary valid/invalid per constraint
    - Soft labels: probability of satisfaction based on current predictions
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_constraints: int = 27,  # 9 rows + 9 cols + 9 boxes for Sudoku
        num_cells: int = 81,
        use_global_pooling: bool = True,
        use_local_features: bool = True
    ):
        """
        Args:
            hidden_size: Dimension of input hidden states
            num_constraints: Number of constraints to predict (27 for Sudoku)
            num_cells: Number of cells in the puzzle (81 for Sudoku)
            use_global_pooling: Whether to use global-pooled features
            use_local_features: Whether to use per-constraint local features
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_constraints = num_constraints
        self.num_cells = num_cells
        self.use_global_pooling = use_global_pooling
        self.use_local_features = use_local_features
        
        # Compute input size based on feature usage
        input_size = 0
        if use_global_pooling:
            input_size += hidden_size
        if use_local_features:
            # 9 cells per constraint, each with hidden_size
            input_size += hidden_size
        
        # Per-constraint prediction heads
        # Using a shared MLP is more parameter-efficient than separate heads
        self.constraint_mlp = nn.Sequential(
            CastedLinear(input_size, hidden_size // 2, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size // 2, 1, bias=True)
        )
        
        # Register constraint-to-cell mapping for Sudoku
        self._build_constraint_mappings()
        
    def _build_constraint_mappings(self):
        """Build mappings from constraints to their constituent cells."""
        # Constraint indices:
        # 0-8: rows, 9-17: columns, 18-26: boxes
        
        constraint_cells = []
        
        # Row constraints (0-8)
        for row in range(9):
            cells = [row * 9 + col for col in range(9)]
            constraint_cells.append(cells)
        
        # Column constraints (9-17)
        for col in range(9):
            cells = [row * 9 + col for row in range(9)]
            constraint_cells.append(cells)
        
        # Box constraints (18-26)
        for box_row in range(3):
            for box_col in range(3):
                cells = []
                for r in range(3):
                    for c in range(3):
                        row = box_row * 3 + r
                        col = box_col * 3 + c
                        cells.append(row * 9 + col)
                constraint_cells.append(cells)
        
        # Register as buffer: [27, 9] indices
        self.register_buffer(
            'constraint_cell_indices',
            torch.tensor(constraint_cells, dtype=torch.long),
            persistent=False
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        puzzle_emb_len: int = 0
    ) -> torch.Tensor:
        """
        Predict constraint satisfaction probabilities.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] from z_H
            puzzle_emb_len: Number of puzzle embedding tokens to skip
            
        Returns:
            constraint_logits: [batch_size, num_constraints] validity logits
        """
        batch_size = hidden_states.shape[0]
        
        # Extract cell hidden states (skip puzzle embeddings)
        cell_hidden = hidden_states[:, puzzle_emb_len:puzzle_emb_len + self.num_cells]
        
        features_list = []
        
        if self.use_global_pooling:
            # Global pooled feature
            global_feat = cell_hidden.mean(dim=1)  # [batch, hidden]
            features_list.append(
                global_feat.unsqueeze(1).expand(-1, self.num_constraints, -1)
            )  # [batch, 27, hidden]
        
        if self.use_local_features:
            # Gather cells for each constraint and pool
            # constraint_cell_indices: [27, 9]
            # cell_hidden: [batch, 81, hidden]
            
            # Expand indices for batched gathering
            indices = self.constraint_cell_indices.unsqueeze(0).expand(batch_size, -1, -1)
            # indices: [batch, 27, 9]
            
            # Gather: for each constraint, get its 9 cells
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)
            cell_hidden_expanded = cell_hidden.unsqueeze(1).expand(-1, self.num_constraints, -1, -1)
            
            # Manual gather along cell dimension
            constraint_cells = torch.gather(
                cell_hidden_expanded,
                dim=2,
                index=indices_expanded
            )  # [batch, 27, 9, hidden]
            
            # Pool cells within each constraint (mean pooling)
            local_feat = constraint_cells.mean(dim=2)  # [batch, 27, hidden]
            features_list.append(local_feat)
        
        # Concatenate features
        features = torch.cat(features_list, dim=-1)  # [batch, 27, input_size]
        
        # Predict per-constraint validity
        constraint_logits = self.constraint_mlp(features).squeeze(-1)  # [batch, 27]
        
        return constraint_logits
    
    @staticmethod
    def compute_constraint_labels(
        predictions: torch.Tensor,
        soft_labels: bool = False,
        vocab_offset: int = 2  # Token 2 = digit 1, etc.
    ) -> torch.Tensor:
        """
        Compute ground-truth constraint satisfaction labels from predictions.
        
        Args:
            predictions: [batch_size, 81] predicted token IDs
            soft_labels: If True, return probability; if False, return binary
            vocab_offset: Offset to convert token IDs to digits (1-9)
            
        Returns:
            labels: [batch_size, 27] constraint satisfaction labels
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Convert tokens to digits (1-9, 0 for invalid)
        digits = torch.zeros_like(predictions)
        valid_mask = (predictions >= vocab_offset) & (predictions < vocab_offset + 9)
        digits[valid_mask] = predictions[valid_mask] - vocab_offset + 1
        
        digits = digits.view(batch_size, 9, 9)  # [batch, row, col]
        
        labels = []
        
        # Row constraints (0-8)
        for row in range(9):
            row_digits = digits[:, row, :]  # [batch, 9]
            if soft_labels:
                # Count unique non-zero digits / 9
                valid = (row_digits > 0).sum(dim=1).float()
                unique = torch.tensor([
                    len(set(row_digits[b].tolist()) - {0})
                    for b in range(batch_size)
                ], device=device, dtype=torch.float32)
                label = (unique == valid).float()  # 1 if no duplicates
            else:
                # Binary: all 9 unique digits present?
                label = torch.tensor([
                    len(set(row_digits[b].tolist()) - {0}) == 9
                    for b in range(batch_size)
                ], device=device, dtype=torch.float32)
            labels.append(label)
        
        # Column constraints (9-17)
        for col in range(9):
            col_digits = digits[:, :, col]  # [batch, 9]
            if soft_labels:
                valid = (col_digits > 0).sum(dim=1).float()
                unique = torch.tensor([
                    len(set(col_digits[b].tolist()) - {0})
                    for b in range(batch_size)
                ], device=device, dtype=torch.float32)
                label = (unique == valid).float()
            else:
                label = torch.tensor([
                    len(set(col_digits[b].tolist()) - {0}) == 9
                    for b in range(batch_size)
                ], device=device, dtype=torch.float32)
            labels.append(label)
        
        # Box constraints (18-26)
        for box_row in range(3):
            for box_col in range(3):
                box_digits = digits[
                    :,
                    box_row * 3:(box_row + 1) * 3,
                    box_col * 3:(box_col + 1) * 3
                ].reshape(batch_size, 9)  # [batch, 9]
                
                if soft_labels:
                    valid = (box_digits > 0).sum(dim=1).float()
                    unique = torch.tensor([
                        len(set(box_digits[b].tolist()) - {0})
                        for b in range(batch_size)
                    ], device=device, dtype=torch.float32)
                    label = (unique == valid).float()
                else:
                    label = torch.tensor([
                        len(set(box_digits[b].tolist()) - {0}) == 9
                        for b in range(batch_size)
                    ], device=device, dtype=torch.float32)
                labels.append(label)
        
        return torch.stack(labels, dim=1)  # [batch, 27]


class ConstraintViolationCounter(nn.Module):
    """
    Differentiable constraint violation counter using soft predictions.
    
    Instead of hard counting, this module computes a differentiable
    approximation of constraint violations that can be used as an
    auxiliary loss term.
    """
    
    def __init__(self, num_digits: int = 9, temperature: float = 0.1):
        super().__init__()
        self.num_digits = num_digits
        self.temperature = temperature
        
        # Build constraint-to-cell mapping
        self._build_constraint_mappings()
    
    def _build_constraint_mappings(self):
        """Same mapping as ConstraintSatisfactionHead."""
        constraint_cells = []
        
        for row in range(9):
            cells = [row * 9 + col for col in range(9)]
            constraint_cells.append(cells)
        
        for col in range(9):
            cells = [row * 9 + col for row in range(9)]
            constraint_cells.append(cells)
        
        for box_row in range(3):
            for box_col in range(3):
                cells = []
                for r in range(3):
                    for c in range(3):
                        cells.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
                constraint_cells.append(cells)
        
        self.register_buffer(
            'constraint_cell_indices',
            torch.tensor(constraint_cells, dtype=torch.long),
            persistent=False
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        puzzle_emb_len: int = 0,
        digit_offset: int = 2  # Vocab offset for digit tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute differentiable constraint violation score.
        
        The idea: for each constraint, compute the sum of probabilities
        for each digit. If a digit appears twice, its probability sum > 1.
        The violation score is sum(max(0, prob_sum - 1)) over all digits.
        
        Args:
            logits: [batch_size, seq_len, vocab_size] prediction logits
            puzzle_emb_len: Number of puzzle embedding tokens
            digit_offset: Vocab index offset for digit 1
            
        Returns:
            total_violation: [batch_size] total violation score
            per_constraint: [batch_size, 27] per-constraint violations
        """
        batch_size = logits.shape[0]
        
        # Extract cell logits and convert to probabilities
        cell_logits = logits[:, puzzle_emb_len:puzzle_emb_len + 81]  # [batch, 81, vocab]
        
        # Extract digit probabilities (tokens 2-10 for digits 1-9)
        digit_logits = cell_logits[:, :, digit_offset:digit_offset + self.num_digits]
        digit_probs = F.softmax(digit_logits / self.temperature, dim=-1)  # [batch, 81, 9]
        
        per_constraint_violations = []
        
        for constraint_idx in range(27):
            cell_indices = self.constraint_cell_indices[constraint_idx]  # [9]
            
            # Get probabilities for cells in this constraint
            constraint_probs = digit_probs[:, cell_indices, :]  # [batch, 9 cells, 9 digits]
            
            # Sum probabilities for each digit across cells
            digit_sums = constraint_probs.sum(dim=1)  # [batch, 9 digits]
            
            # Violation: if any digit sum > 1, it means duplicates
            # Use ReLU to only penalize when sum > 1
            violations = F.relu(digit_sums - 1.0).sum(dim=-1)  # [batch]
            per_constraint_violations.append(violations)
        
        per_constraint = torch.stack(per_constraint_violations, dim=1)  # [batch, 27]
        total_violation = per_constraint.sum(dim=1)  # [batch]
        
        return total_violation, per_constraint


def constraint_satisfaction_loss(
    constraint_logits: torch.Tensor,
    predictions: torch.Tensor,
    vocab_offset: int = 2,
    soft_labels: bool = True,
    label_smoothing: float = 0.1
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for constraint satisfaction.
    
    Args:
        constraint_logits: [batch, 27] predicted constraint validity logits
        predictions: [batch, 81] predicted token IDs
        vocab_offset: Token offset for digit 1
        soft_labels: Use soft (probability) or hard (binary) labels
        label_smoothing: Label smoothing factor
        
    Returns:
        loss: Scalar loss value
    """
    labels = ConstraintSatisfactionHead.compute_constraint_labels(
        predictions, soft_labels=soft_labels, vocab_offset=vocab_offset
    )
    
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    
    loss = F.binary_cross_entropy_with_logits(constraint_logits, labels)
    return loss
