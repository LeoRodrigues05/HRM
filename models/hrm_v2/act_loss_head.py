"""
ACT Loss Head for HRM v2 with Constraint Losses.

Extends the original ACTLossHead to include:
- Constraint satisfaction head loss
- Differentiable violation loss
"""

from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy, softmax_cross_entropy
from models.hrm_v2.constraint_head import ConstraintSatisfactionHead


class ACTLossHeadV2(nn.Module):
    """
    ACT Loss head for HRM v2 with constraint-aware losses.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        constraint_loss_weight: float = 0.5,
        violation_loss_weight: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals().get(loss_type, stablemax_cross_entropy)
        self.constraint_loss_weight = constraint_loss_weight
        self.violation_loss_weight = violation_loss_weight
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model forward
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness metrics
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # === Standard Losses ===
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            seq_is_correct.to(outputs["q_halt_logits"].dtype), 
            reduction="sum"
        )

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue loss
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], 
                outputs["target_q_continue"], 
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # === V2 Constraint Losses ===
        constraint_loss = torch.tensor(0.0, device=labels.device)
        violation_loss = torch.tensor(0.0, device=labels.device)
        
        # Constraint satisfaction head loss
        if "constraint_logits" in outputs and self.constraint_loss_weight > 0:
            predictions = outputs["logits"].argmax(dim=-1)
            constraint_labels = self._compute_constraint_labels(predictions, vocab_offset=2)
            
            constraint_loss = F.binary_cross_entropy_with_logits(
                outputs["constraint_logits"],
                constraint_labels,
                reduction="sum"
            )
            metrics["constraint_loss"] = constraint_loss.detach()
            
            # Track constraint satisfaction rate
            with torch.no_grad():
                constraint_sat = (torch.sigmoid(outputs["constraint_logits"]) > 0.5).float().mean()
                metrics["constraint_satisfaction"] = constraint_sat * valid_metrics.sum()
        
        # Violation loss
        if "violation_score" in outputs and self.violation_loss_weight > 0:
            violation_loss = outputs["violation_score"].sum()
            metrics["violation_loss"] = violation_loss.detach()

        # Total loss
        total_loss = (
            lm_loss + 
            0.5 * (q_halt_loss + q_continue_loss) +
            self.constraint_loss_weight * constraint_loss +
            self.violation_loss_weight * violation_loss
        )

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
    
    @staticmethod
    def _compute_constraint_labels(
        predictions: torch.Tensor,
        vocab_offset: int = 2
    ) -> torch.Tensor:
        """
        Compute constraint satisfaction labels from predictions.
        
        For efficiency, uses vectorized operations.
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Convert tokens to digits (1-9, 0 for invalid)
        digits = torch.zeros_like(predictions, dtype=torch.int64)
        valid_mask = (predictions >= vocab_offset) & (predictions < vocab_offset + 9)
        digits[valid_mask] = predictions[valid_mask] - vocab_offset + 1
        
        digits = digits.view(batch_size, 9, 9)
        labels = []
        
        # Row constraints (0-8)
        for row in range(9):
            row_digits = digits[:, row, :]
            # Count unique non-zero elements
            label = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                unique_nonzero = torch.unique(row_digits[b][row_digits[b] > 0])
                filled = (row_digits[b] > 0).sum()
                label[b] = 1.0 if (len(unique_nonzero) == filled and filled == 9) else 0.0
            labels.append(label)
        
        # Column constraints (9-17)
        for col in range(9):
            col_digits = digits[:, :, col]
            label = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                unique_nonzero = torch.unique(col_digits[b][col_digits[b] > 0])
                filled = (col_digits[b] > 0).sum()
                label[b] = 1.0 if (len(unique_nonzero) == filled and filled == 9) else 0.0
            labels.append(label)
        
        # Box constraints (18-26)
        for box_row in range(3):
            for box_col in range(3):
                box_digits = digits[:, box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].reshape(batch_size, 9)
                label = torch.zeros(batch_size, device=device)
                for b in range(batch_size):
                    unique_nonzero = torch.unique(box_digits[b][box_digits[b] > 0])
                    filled = (box_digits[b] > 0).sum()
                    label[b] = 1.0 if (len(unique_nonzero) == filled and filled == 9) else 0.0
                labels.append(label)
        
        return torch.stack(labels, dim=1)
