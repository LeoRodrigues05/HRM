"""
Activation Patching Script for HRM Model

This script performs activation patching experiments where activations (z_H and z_L) 
from a source puzzle are transferred to a target puzzle during forward passes.

The key idea is that even though two puzzles require different actions (e.g., filling
row 1 in puzzle A vs row 3 in puzzle B), we can test the impact of patching activations
from one puzzle into another to understand what information is encoded in these activations.

Usage:
    python scripts/activation_patching.py \
        --checkpoint <path_to_checkpoint> \
        --source_puzzle_idx 0 \
        --target_puzzle_idx 1 \
        --patch_level H \
        --patch_step 2 \
        --output_dir results/activation_patching
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1, 
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry
)


@dataclass
class ActivationCache:
    """Stores activations from a forward pass"""
    z_H: torch.Tensor
    z_L: torch.Tensor
    step: int
    logits: torch.Tensor
    q_halt_logits: torch.Tensor
    q_continue_logits: torch.Tensor


class ActivationPatcher:
    """
    Handles activation patching between two puzzles.
    
    Captures activations from a source puzzle and injects them into a target puzzle
    at specified layers and timesteps.
    """
    
    def __init__(
        self, 
        model: HierarchicalReasoningModel_ACTV1,
        device: torch.device = torch.device("cuda")
    ):
        self.model = model
        self.device = device
        self.source_cache: Dict[int, ActivationCache] = {}
        self.target_cache: Dict[int, ActivationCache] = {}
        
    def run_and_cache_activations(
        self, 
        batch: Dict[str, torch.Tensor],
        cache_dict: Dict[int, ActivationCache],
        max_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run model forward pass and cache all intermediate activations.
        
        Args:
            batch: Input batch dictionary
            cache_dict: Dictionary to store activations at each step
            max_steps: Maximum number of steps to run (None = use model default)
            
        Returns:
            Final outputs from the model
        """
        cache_dict.clear()
        
        # Initialize carry
        carry = self.model.initial_carry(batch)
        
        # Store original halt_max_steps if we want to override
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps
        
        all_outputs = []
        step = 0
        
        with torch.no_grad():
            while not carry.halted.all() or step == 0:
                # Forward pass
                new_carry, outputs = self.model(carry, batch)
                
                # Cache activations
                cache_dict[step] = ActivationCache(
                    z_H=carry.inner_carry.z_H.clone(),
                    z_L=carry.inner_carry.z_L.clone(),
                    step=step,
                    logits=outputs["logits"].clone(),
                    q_halt_logits=outputs["q_halt_logits"].clone(),
                    q_continue_logits=outputs["q_continue_logits"].clone()
                )
                
                all_outputs.append(outputs)
                carry = new_carry
                step += 1
                
                if step >= (max_steps or original_max_steps):
                    break
        
        # Restore original setting
        if max_steps is not None:
            self.model.config.halt_max_steps = original_max_steps
        
        # Return final outputs
        return all_outputs[-1] if all_outputs else {}
    
    def run_with_patching(
        self,
        target_batch: Dict[str, torch.Tensor],
        source_activations: Dict[int, ActivationCache],
        patch_level: str = "both",  # "H", "L", or "both"
        patch_steps: Optional[List[int]] = None,  # Which steps to patch (None = all)
        patch_positions: Optional[List[int]] = None,  # Which positions to patch (None = all)
        max_steps: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache]]:
        """
        Run model with activation patching from source to target.
        
        Args:
            target_batch: Target puzzle batch
            source_activations: Cached activations from source puzzle
            patch_level: Which level to patch - "H", "L", or "both"
            patch_steps: List of steps at which to apply patching (None = all steps)
            patch_positions: List of sequence positions to patch (None = all positions)
            max_steps: Maximum steps to run
            
        Returns:
            Tuple of (final outputs, cache of patched activations)
        """
        patched_cache: Dict[int, ActivationCache] = {}
        
        # Initialize carry
        carry = self.model.initial_carry(target_batch)
        
        # Store original halt_max_steps
        original_max_steps = self.model.config.halt_max_steps
        if max_steps is not None:
            self.model.config.halt_max_steps = max_steps
        
        all_outputs = []
        step = 0
        
        with torch.no_grad():
            while not carry.halted.all() or step == 0:
                # Check if we should patch at this step
                should_patch = (patch_steps is None) or (step in patch_steps)
                
                if should_patch and step in source_activations:
                    # Apply patching
                    source_act = source_activations[step]
                    
                    if patch_level in ["H", "both"]:
                        if patch_positions is None:
                            # Patch all positions
                            carry.inner_carry.z_H = source_act.z_H.clone()
                        else:
                            # Patch specific positions
                            for pos in patch_positions:
                                if pos < carry.inner_carry.z_H.shape[1]:
                                    carry.inner_carry.z_H[:, pos, :] = source_act.z_H[:, pos, :].clone()
                    
                    if patch_level in ["L", "both"]:
                        if patch_positions is None:
                            # Patch all positions
                            carry.inner_carry.z_L = source_act.z_L.clone()
                        else:
                            # Patch specific positions
                            for pos in patch_positions:
                                if pos < carry.inner_carry.z_L.shape[1]:
                                    carry.inner_carry.z_L[:, pos, :] = source_act.z_L[:, pos, :].clone()
                
                # Forward pass
                new_carry, outputs = self.model(carry, target_batch)
                
                # Cache the patched activations
                patched_cache[step] = ActivationCache(
                    z_H=new_carry.inner_carry.z_H.clone(),
                    z_L=new_carry.inner_carry.z_L.clone(),
                    step=step,
                    logits=outputs["logits"].clone(),
                    q_halt_logits=outputs["q_halt_logits"].clone(),
                    q_continue_logits=outputs["q_continue_logits"].clone()
                )
                
                all_outputs.append(outputs)
                carry = new_carry
                step += 1
                
                if step >= (max_steps or original_max_steps):
                    break
        
        # Restore original setting
        if max_steps is not None:
            self.model.config.halt_max_steps = original_max_steps
        
        return all_outputs[-1] if all_outputs else {}, patched_cache


def compute_metrics(
    predictions: torch.Tensor, 
    labels: torch.Tensor,
    ignore_label_id: int = -100
) -> Dict[str, float]:
    """Compute accuracy metrics for predictions"""
    # Mask out ignore labels
    valid_mask = (labels != ignore_label_id)
    
    if valid_mask.sum() == 0:
        return {"accuracy": 0.0, "total_positions": 0}
    
    correct = (predictions == labels) & valid_mask
    accuracy = correct.sum().item() / valid_mask.sum().item()
    
    return {
        "accuracy": accuracy,
        "correct": correct.sum().item(),
        "total_positions": valid_mask.sum().item()
    }


def visualize_puzzle(
    inputs: np.ndarray, 
    predictions: np.ndarray, 
    labels: np.ndarray,
    title: str = "Puzzle"
) -> str:
    """Create a simple text visualization of a puzzle"""
    lines = [f"\n{'='*60}", title, '='*60]
    
    if inputs.ndim == 2:
        inputs = inputs[0]
        predictions = predictions[0] if predictions.ndim == 2 else predictions
        labels = labels[0] if labels.ndim == 2 else labels
    
    lines.append(f"Input shape: {inputs.shape}")
    lines.append(f"Predictions: {predictions[:20]}...")  # First 20 tokens
    lines.append(f"Labels:      {labels[:20]}...")
    lines.append(f"Match: {(predictions == labels)[:20]}...")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Activation Patching for HRM")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--source_puzzle_idx", type=int, default=0,
                        help="Index of source puzzle in test set")
    parser.add_argument("--target_puzzle_idx", type=int, default=1,
                        help="Index of target puzzle in test set")
    parser.add_argument("--patch_level", type=str, default="both", 
                        choices=["H", "L", "both"],
                        help="Which activation level to patch")
    parser.add_argument("--patch_steps", type=str, default=None,
                        help="Comma-separated list of steps to patch (e.g., '0,1,2'). None = all steps")
    parser.add_argument("--patch_positions", type=str, default=None,
                        help="Comma-separated list of positions to patch (e.g., '0,9,18'). None = all positions")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Maximum number of reasoning steps")
    parser.add_argument("--output_dir", type=str, default="results/activation_patching",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse patch steps and positions
    patch_steps = None if args.patch_steps is None else [int(s) for s in args.patch_steps.split(",")]
    patch_positions = None if args.patch_positions is None else [int(p) for p in args.patch_positions.split(",")]
    
    print("="*60)
    print("Activation Patching Experiment")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source puzzle: {args.source_puzzle_idx}")
    print(f"Target puzzle: {args.target_puzzle_idx}")
    print(f"Patch level: {args.patch_level}")
    print(f"Patch steps: {patch_steps if patch_steps else 'all'}")
    print(f"Patch positions: {patch_positions if patch_positions else 'all'}")
    print(f"Max steps: {args.max_steps}")
    print("="*60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    # Create dataloader (using small batch size for individual puzzles)
    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True, 
        epochs_per_iter=1, global_batch_size=1,
        rank=0, world_size=1
    )
    
    # Initialize model
    train_state = init_train_state(config, test_metadata, world_size=1)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle compiled models - check if model expects _orig_mod prefix
    model_keys = set(train_state.model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
    checkpoint_has_prefix = any(k.startswith("_orig_mod.") for k in checkpoint_keys)
    
    # Add prefix if model has it but checkpoint doesn't
    if model_has_prefix and not checkpoint_has_prefix:
        checkpoint = {f"_orig_mod.{k}": v for k, v in checkpoint.items()}
    # Remove prefix if checkpoint has it but model doesn't
    elif checkpoint_has_prefix and not model_has_prefix:
        checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
    
    train_state.model.load_state_dict(checkpoint, assign=True)
    train_state.model.to(device)
    train_state.model.eval()
    
    print("Model loaded successfully")
    
    # Initialize patcher
    patcher = ActivationPatcher(train_state.model, device=device)
    
    # Get source and target puzzles
    print(f"\nGathering puzzles from test set...")
    puzzles = []
    for i, data in enumerate(test_loader):
        if i >= max(args.source_puzzle_idx, args.target_puzzle_idx) + 1:
            break
        # Unpack dataloader output - returns (set_name, batch, effective_batch_size) in test mode
        if isinstance(data, tuple) and len(data) == 3:
            set_name, batch, effective_batch_size = data
        elif isinstance(data, list):
            batch = data[0]  # Take first element if it's a list
        else:
            batch = data
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        puzzles.append(batch)
    
    if len(puzzles) <= max(args.source_puzzle_idx, args.target_puzzle_idx):
        print(f"Error: Not enough puzzles in test set. Found {len(puzzles)}, need {max(args.source_puzzle_idx, args.target_puzzle_idx) + 1}")
        return
    
    source_batch = puzzles[args.source_puzzle_idx]
    target_batch = puzzles[args.target_puzzle_idx]
    
    print(f"Source puzzle: shape {source_batch['inputs'].shape}")
    print(f"Target puzzle: shape {target_batch['inputs'].shape}")
    
    # Run baseline: source puzzle (no patching)
    print(f"\n{'='*60}")
    print("Running source puzzle (baseline)...")
    print('='*60)
    source_outputs = patcher.run_and_cache_activations(
        source_batch, patcher.source_cache, max_steps=args.max_steps
    )
    source_preds = source_outputs["logits"].argmax(-1)
    source_metrics = compute_metrics(source_preds, source_batch["labels"])
    print(f"Source accuracy: {source_metrics['accuracy']:.4f} "
          f"({source_metrics['correct']}/{source_metrics['total_positions']})")
    
    # Run baseline: target puzzle (no patching)
    print(f"\n{'='*60}")
    print("Running target puzzle (baseline)...")
    print('='*60)
    target_outputs = patcher.run_and_cache_activations(
        target_batch, patcher.target_cache, max_steps=args.max_steps
    )
    target_preds = target_outputs["logits"].argmax(-1)
    target_metrics = compute_metrics(target_preds, target_batch["labels"])
    print(f"Target accuracy: {target_metrics['accuracy']:.4f} "
          f"({target_metrics['correct']}/{target_metrics['total_positions']})")
    
    # Run with patching: target puzzle with source activations
    print(f"\n{'='*60}")
    print("Running target puzzle with patched activations...")
    print('='*60)
    patched_outputs, patched_cache = patcher.run_with_patching(
        target_batch,
        patcher.source_cache,
        patch_level=args.patch_level,
        patch_steps=patch_steps,
        patch_positions=patch_positions,
        max_steps=args.max_steps
    )
    patched_preds = patched_outputs["logits"].argmax(-1)
    patched_metrics = compute_metrics(patched_preds, target_batch["labels"])
    print(f"Patched accuracy: {patched_metrics['accuracy']:.4f} "
          f"({patched_metrics['correct']}/{patched_metrics['total_positions']})")
    
    # Compute impact of patching
    accuracy_change = patched_metrics['accuracy'] - target_metrics['accuracy']
    print(f"\n{'='*60}")
    print("Impact Analysis")
    print('='*60)
    print(f"Accuracy change: {accuracy_change:+.4f}")
    print(f"Source → Target baseline: {target_metrics['accuracy']:.4f}")
    print(f"Source → Target patched:  {patched_metrics['accuracy']:.4f}")
    
    # Save results
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "source_puzzle_idx": args.source_puzzle_idx,
            "target_puzzle_idx": args.target_puzzle_idx,
            "patch_level": args.patch_level,
            "patch_steps": patch_steps,
            "patch_positions": patch_positions,
            "max_steps": args.max_steps,
        },
        "metrics": {
            "source": source_metrics,
            "target_baseline": target_metrics,
            "target_patched": patched_metrics,
            "accuracy_change": accuracy_change,
        },
        "predictions": {
            "source": source_preds.cpu().numpy().tolist(),
            "target_baseline": target_preds.cpu().numpy().tolist(),
            "target_patched": patched_preds.cpu().numpy().tolist(),
        }
    }
    
    # Save as YAML
    results_path = os.path.join(
        args.output_dir, 
        f"patch_s{args.source_puzzle_idx}_t{args.target_puzzle_idx}_{args.patch_level}.yaml"
    )
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save activation caches
    cache_path = os.path.join(
        args.output_dir,
        f"activations_s{args.source_puzzle_idx}_t{args.target_puzzle_idx}.pt"
    )
    torch.save({
        "source": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "step": v.step,
        } for k, v in patcher.source_cache.items()},
        "target": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "step": v.step,
        } for k, v in patcher.target_cache.items()},
        "patched": {k: {
            "z_H": v.z_H.cpu(),
            "z_L": v.z_L.cpu(),
            "step": v.step,
        } for k, v in patched_cache.items()},
    }, cache_path)
    print(f"Activation caches saved to: {cache_path}")
    
    print(f"\n{'='*60}")
    print("Experiment completed successfully!")
    print('='*60)


if __name__ == "__main__":
    main()
