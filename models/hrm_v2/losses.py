"""
Combined Loss Function for HRM v2

This module combines:
1. Standard LM loss (cross-entropy on token predictions)
2. Q-learning losses for ACT halting decisions
3. Constraint satisfaction loss (Point 3)
4. Differentiable violation loss (Point 3)

The total loss is:
    L = L_lm + α * (L_q_halt + L_q_continue) + β * L_constraint + γ * L_violation

Where α, β, γ are configurable weights.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn

from models.losses import log_stablemax
from .constraint_head import constraint_satisfaction_loss


class HRMv2Loss(nn.Module):
    """
    Combined loss function for HRM v2 training.
    """
    
    def __init__(
        self,
        q_loss_weight: float = 0.5,
        constraint_loss_weight: float = 0.5,
        violation_loss_weight: float = 0.1,
        label_smoothing: float = 0.1,
        use_stablemax: bool = True
    ):
        """
        Args:
            q_loss_weight: Weight for Q-learning losses
            constraint_loss_weight: Weight for constraint satisfaction head loss
            violation_loss_weight: Weight for differentiable violation loss
            label_smoothing: Label smoothing for constraint labels
            use_stablemax: Use StableMax instead of Softmax for LM loss
        """
        super().__init__()
        
        self.q_loss_weight = q_loss_weight
        self.constraint_loss_weight = constraint_loss_weight
        self.violation_loss_weight = violation_loss_weight
        self.label_smoothing = label_smoothing
        self.use_stablemax = use_stablemax
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        vocab_offset: int = 2
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict with keys:
                - logits: [batch, seq_len, vocab_size]
                - q_halt_logits: [batch]
                - q_continue_logits: [batch]
                - constraint_logits: [batch, 27] (optional)
                - violation_score: [batch] (optional)
                - target_q_continue: [batch] (optional, for training)
            batch: Input batch with:
                - inputs: [batch, seq_len]
                - labels: [batch, seq_len]
            vocab_offset: Token offset for digit 1
            
        Returns:
            total_loss: Scalar loss
            loss_components: Dict of individual loss terms for logging
        """
        logits = outputs["logits"]
        labels = batch["labels"]
        
        loss_components = {}
        
        # 1. LM Loss (token prediction)
        if self.use_stablemax:
            log_probs = log_stablemax(logits, dim=-1)
            lm_loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                labels.view(-1),
                reduction='mean'
            )
        else:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='mean'
            )
        
        loss_components["lm_loss"] = lm_loss
        total_loss = lm_loss
        
        # 2. Q-Learning Losses (if training)
        if "target_q_continue" in outputs:
            q_halt = outputs["q_halt_logits"]
            q_continue = outputs["q_continue_logits"]
            target_q_continue = outputs["target_q_continue"]
            
            # Get ground truth: is the puzzle solved?
            predictions = logits.argmax(dim=-1)
            is_solved = (predictions == labels).all(dim=-1).float()
            
            # Q-halt target: probability of being solved
            q_halt_loss = F.binary_cross_entropy_with_logits(
                q_halt, is_solved, reduction='mean'
            )
            
            # Q-continue target: from TD learning
            q_continue_loss = F.mse_loss(
                torch.sigmoid(q_continue), target_q_continue, reduction='mean'
            )
            
            q_loss = self.q_loss_weight * (q_halt_loss + q_continue_loss)
            loss_components["q_halt_loss"] = q_halt_loss
            loss_components["q_continue_loss"] = q_continue_loss
            total_loss = total_loss + q_loss
        
        # 3. Constraint Satisfaction Loss (Point 3)
        if "constraint_logits" in outputs and self.constraint_loss_weight > 0:
            constraint_logits = outputs["constraint_logits"]
            predictions = logits.argmax(dim=-1)
            
            constraint_loss = constraint_satisfaction_loss(
                constraint_logits=constraint_logits,
                predictions=predictions,
                vocab_offset=vocab_offset,
                soft_labels=True,
                label_smoothing=self.label_smoothing
            )
            
            loss_components["constraint_loss"] = constraint_loss
            total_loss = total_loss + self.constraint_loss_weight * constraint_loss
        
        # 4. Differentiable Violation Loss (Point 3)
        if "violation_score" in outputs and self.violation_loss_weight > 0:
            violation_score = outputs["violation_score"]
            violation_loss = violation_score.mean()
            
            loss_components["violation_loss"] = violation_loss
            total_loss = total_loss + self.violation_loss_weight * violation_loss
        
        loss_components["total_loss"] = total_loss
        
        return total_loss, loss_components


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Returns:
        metrics: Dict with accuracy, constraint satisfaction rate, etc.
    """
    logits = outputs["logits"]
    labels = batch["labels"]
    inputs = batch["inputs"]
    
    predictions = logits.argmax(dim=-1)
    
    # Overall accuracy
    accuracy = (predictions == labels).float().mean().item()
    
    # Per-puzzle accuracy (all cells correct)
    puzzle_accuracy = (predictions == labels).all(dim=-1).float().mean().item()
    
    # Unknown cell accuracy (cells that were blank in input)
    # Assuming token 1 = BLANK
    unknown_mask = (inputs == 1)
    if unknown_mask.any():
        unknown_accuracy = (predictions[unknown_mask] == labels[unknown_mask]).float().mean().item()
    else:
        unknown_accuracy = 1.0
    
    metrics = {
        "accuracy": accuracy,
        "puzzle_accuracy": puzzle_accuracy,
        "unknown_accuracy": unknown_accuracy,
    }
    
    # Constraint satisfaction (if available)
    if "constraint_logits" in outputs:
        constraint_probs = torch.sigmoid(outputs["constraint_logits"])
        constraint_satisfaction = (constraint_probs > 0.5).float().mean().item()
        metrics["constraint_satisfaction"] = constraint_satisfaction
    
    # Violation score (if available)
    if "violation_score" in outputs:
        metrics["violation_score"] = outputs["violation_score"].mean().item()
    
    return metrics
