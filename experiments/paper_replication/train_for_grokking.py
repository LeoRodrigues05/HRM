#!/usr/bin/env python3
"""Training script specifically for grokking analysis.

This script trains an HRM model from scratch on Sudoku puzzles, saving
checkpoints at regular intervals to capture grokking dynamics.

The script evaluates on both train and test sets at each checkpoint to
track the generalization gap over time.

Usage:
    python experiments/paper_replication/train_for_grokking.py \
        --data_path data/sudoku-extreme-1k-aug-1000 \
        --output_dir experiments/paper_replication/results/grokking_checkpoints \
        --num_checkpoints 10 \
        --total_epochs 50000

For faster experimentation (smaller model):
    python experiments/paper_replication/train_for_grokking.py \
        --data_path data/sudoku-extreme-1k-aug-1000 \
        --output_dir experiments/paper_replication/results/grokking_checkpoints \
        --num_checkpoints 10 \
        --total_epochs 20000 \
        --hidden_size 256 \
        --num_layers 4
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TrainingSnapshot:
    """Training metrics at a checkpoint."""
    checkpoint_id: int
    step: int
    epoch: float
    train_loss: float
    train_accuracy: float
    train_unknown_accuracy: float
    test_loss: float
    test_accuracy: float
    test_unknown_accuracy: float
    learning_rate: float
    timestamp: str


@dataclass  
class GrokingTrainConfig:
    """Configuration for grokking training."""
    # Data
    data_path: str
    output_dir: str
    
    # Training schedule - use STEPS not epochs for practical training times
    total_steps: int = 50000  # Total training steps (not epochs!)
    num_checkpoints: int = 10  # Will save this many checkpoints evenly spaced
    eval_batch_size: int = 64
    
    # Model architecture (can use smaller for faster experimentation)
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    H_cycles: int = 2
    L_cycles: int = 2
    halt_max_steps: int = 16
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    batch_size: int = 256  # Batch size
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    # Evaluation sample sizes (use subset for speed)
    train_eval_samples: int = 5000
    test_eval_samples: int = 10000
    
    # Training data subset (use smaller subset for faster iteration)
    train_subset: int = 100000  # Use 100k samples from training set


def load_data(data_path: str) -> Dict[str, np.ndarray]:
    """Load train and test data."""
    data_path = Path(data_path)
    
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    data = {
        "train_inputs": np.load(train_dir / "all__inputs.npy"),
        "train_labels": np.load(train_dir / "all__labels.npy"),
        "test_inputs": np.load(test_dir / "all__inputs.npy"),
        "test_labels": np.load(test_dir / "all__labels.npy"),
    }
    
    # Load metadata
    with open(test_dir / "dataset.json", "r") as f:
        data["metadata"] = json.load(f)
    
    return data


def create_model(config: GrokingTrainConfig, metadata: dict, device: str):
    """Create HRM model."""
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    
    # Full config matching hrm_v1.yaml structure
    model_cfg = {
        # Core architecture
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "expansion": 4,  # Standard expansion factor
        "pos_encodings": "rope",  # Rotary position encodings
        
        # Layer counts (split num_layers between H and L)
        "H_layers": config.num_layers // 2,
        "L_layers": config.num_layers // 2,
        
        # Cycles
        "H_cycles": config.H_cycles,
        "L_cycles": config.L_cycles,
        
        # ACT/Halting config
        "halt_max_steps": config.halt_max_steps,
        "halt_exploration_prob": 0.1,  # Standard exploration
        
        # Batch and sequence
        "batch_size": config.batch_size,
        "vocab_size": metadata["vocab_size"],
        "seq_len": metadata["seq_len"],
        "num_puzzle_identifiers": metadata["num_puzzle_identifiers"],
        
        # Puzzle embeddings (disabled for simplicity)
        "puzzle_emb_ndim": 0,
    }
    
    model = HierarchicalReasoningModel_ACTV1(model_cfg).to(device)
    return model, model_cfg


def cosine_schedule_with_warmup(step: int, total_steps: int, warmup_steps: int, 
                                 base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate_model(
    model: nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    config: GrokingTrainConfig,
    max_samples: int = 10000
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    device = config.device
    max_steps = config.halt_max_steps
    
    # Sample subset for speed
    N = min(len(inputs), max_samples)
    indices = np.random.choice(len(inputs), N, replace=False)
    inputs_subset = inputs[indices]
    labels_subset = labels[indices]
    
    total_loss = 0.0
    total_correct = 0
    total_unknown_correct = 0
    total_cells = 0
    total_unknown_cells = 0
    
    batch_size = config.eval_batch_size
    
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_inputs = torch.tensor(inputs_subset[start:end], dtype=torch.long, device=device)
            batch_labels = torch.tensor(labels_subset[start:end], dtype=torch.long, device=device)
            batch_size_actual = batch_inputs.shape[0]
            
            batch = {
                "inputs": batch_inputs,
                "labels": batch_labels,
                "puzzle_identifiers": torch.zeros(batch_size_actual, dtype=torch.long, device=device),
            }
            
            # Initialize carry
            carry = model.initial_carry(batch)
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
            carry.steps = carry.steps.to(device)
            carry.halted = carry.halted.to(device)
            
            # Run all steps
            for step in range(max_steps):
                carry, outputs = model(carry, batch)
            
            logits = outputs["logits"]  # [B, 81, V]
            preds = logits.argmax(dim=-1)
            
            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch_labels.reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            
            # Accuracy
            correct = (preds == batch_labels)
            total_correct += correct.sum().item()
            total_cells += batch_size_actual * 81
            
            # Unknown cell accuracy
            is_given = (batch_inputs >= 2) & (batch_inputs <= 10)
            is_unknown = ~is_given
            unknown_correct = (correct & is_unknown).sum().item()
            unknown_cells = is_unknown.sum().item()
            
            total_unknown_correct += unknown_correct
            total_unknown_cells += unknown_cells
    
    model.train()
    
    return {
        "loss": total_loss / total_cells,
        "accuracy": total_correct / total_cells,
        "unknown_accuracy": total_unknown_correct / max(total_unknown_cells, 1),
    }


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    config: GrokingTrainConfig,
    current_step: int,
    total_steps: int,
) -> tuple:
    """Train for one epoch, return metrics and updated step."""
    model.train()
    device = config.device
    max_steps = config.halt_max_steps
    batch_size = config.batch_size
    
    # Shuffle training data
    N = len(train_inputs)
    indices = np.random.permutation(N)
    
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_cells = 0
    batches = 0
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_indices = indices[start:end]
        
        batch_inputs = torch.tensor(train_inputs[batch_indices], dtype=torch.long, device=device)
        batch_labels = torch.tensor(train_labels[batch_indices], dtype=torch.long, device=device)
        batch_size_actual = batch_inputs.shape[0]
        
        batch = {
            "inputs": batch_inputs,
            "labels": batch_labels,
            "puzzle_identifiers": torch.zeros(batch_size_actual, dtype=torch.long, device=device),
        }
        
        # Update learning rate
        lr = cosine_schedule_with_warmup(
            current_step, total_steps, config.warmup_steps, config.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward
        optimizer.zero_grad()
        
        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        
        # Run all steps
        for step in range(max_steps):
            carry, outputs = model(carry, batch)
        
        logits = outputs["logits"]
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch_labels.reshape(-1)
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == batch_labels).sum().item()
            
            epoch_loss += loss.item() * batch_size_actual
            epoch_correct += correct
            epoch_cells += batch_size_actual * 81
            batches += 1
        
        current_step += 1
    
    avg_loss = epoch_loss / batches
    accuracy = epoch_correct / epoch_cells
    
    return avg_loss, accuracy, current_step, lr


def save_checkpoint(
    model: nn.Module,
    model_cfg: dict,
    optimizer: torch.optim.Optimizer,
    snapshot: TrainingSnapshot,
    output_dir: Path
):
    """Save a training checkpoint."""
    checkpoint_dir = output_dir / f"checkpoint_{snapshot.checkpoint_id:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), checkpoint_dir / "checkpoint.pt")
    
    # Save optimizer state (optional, for resuming)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    
    # Save config
    with open(checkpoint_dir / "all_config.yaml", "w") as f:
        yaml.dump({"arch": model_cfg}, f)
    
    # Save snapshot info
    with open(checkpoint_dir / "snapshot.json", "w") as f:
        json.dump(asdict(snapshot), f, indent=2)
    
    print(f"  Saved checkpoint {snapshot.checkpoint_id} at step {snapshot.step}")


def main():
    parser = argparse.ArgumentParser(description="Train HRM for Grokking Analysis")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000",
                       help="Path to Sudoku dataset")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/paper_replication/results/grokking_checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--total_steps", type=int, default=50000,
                       help="Total training steps (not epochs!)")
    parser.add_argument("--num_checkpoints", type=int, default=10,
                       help="Number of checkpoints to save")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=512,
                       help="Model hidden size")
    parser.add_argument("--num_layers", type=int, default=8,
                       help="Number of transformer layers")
    parser.add_argument("--halt_max_steps", type=int, default=16,
                       help="Maximum ACT steps")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--train_eval_samples", type=int, default=5000,
                       help="Number of train samples to evaluate")
    parser.add_argument("--test_eval_samples", type=int, default=10000,
                       help="Number of test samples to evaluate")
    parser.add_argument("--train_subset", type=int, default=100000,
                       help="Use subset of training data for faster training")
    
    args = parser.parse_args()
    
    # Create config
    config = GrokingTrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        num_checkpoints=args.num_checkpoints,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        halt_max_steps=args.halt_max_steps,
        device=args.device,
        seed=args.seed,
        train_eval_samples=args.train_eval_samples,
        test_eval_samples=args.test_eval_samples,
        train_subset=args.train_subset,
    )
    
    # Setup
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print("="*60)
    print("HRM GROKKING TRAINING")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Data path: {config.data_path}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Total steps: {config.total_steps}")
    print(f"  Num checkpoints: {config.num_checkpoints}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  ACT steps: {config.halt_max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Train subset: {config.train_subset}")
    
    # Load data
    print("\nLoading data...")
    data = load_data(config.data_path)
    
    # Use subset of training data
    if config.train_subset and config.train_subset < len(data['train_inputs']):
        indices = np.random.choice(len(data['train_inputs']), config.train_subset, replace=False)
        data['train_inputs'] = data['train_inputs'][indices]
        data['train_labels'] = data['train_labels'][indices]
    
    print(f"  Train samples: {len(data['train_inputs'])}")
    print(f"  Test samples: {len(data['test_inputs'])}")
    
    # Create model
    print("\nCreating model...")
    model, model_cfg = create_model(config, data["metadata"], config.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Calculate checkpoint intervals (by steps)
    steps_per_checkpoint = config.total_steps // config.num_checkpoints
    checkpoint_steps = [steps_per_checkpoint * (i + 1) for i in range(config.num_checkpoints)]
    checkpoint_steps_set = set(checkpoint_steps)
    
    print(f"\nCheckpoint at steps: {checkpoint_steps}")
    
    # Save training config
    config_dict = asdict(config)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    snapshots: List[TrainingSnapshot] = []
    checkpoint_id = 0
    
    # Initial evaluation (step 0)
    print("\nInitial evaluation...")
    train_metrics = evaluate_model(
        model, data['train_inputs'], data['train_labels'], 
        config, config.train_eval_samples
    )
    test_metrics = evaluate_model(
        model, data['test_inputs'], data['test_labels'],
        config, config.test_eval_samples
    )
    
    snapshot = TrainingSnapshot(
        checkpoint_id=checkpoint_id,
        step=0,
        epoch=0,
        train_loss=train_metrics['loss'],
        train_accuracy=train_metrics['accuracy'],
        train_unknown_accuracy=train_metrics['unknown_accuracy'],
        test_loss=test_metrics['loss'],
        test_accuracy=test_metrics['accuracy'],
        test_unknown_accuracy=test_metrics['unknown_accuracy'],
        learning_rate=config.learning_rate,
        timestamp=datetime.now().isoformat(),
    )
    snapshots.append(snapshot)
    save_checkpoint(model, model_cfg, optimizer, snapshot, output_dir)
    checkpoint_id += 1
    
    print(f"  Step 0: Train acc={train_metrics['accuracy']:.4f}, Test acc={test_metrics['accuracy']:.4f}")
    
    # Step-based training loop
    model.train()
    device = config.device
    max_act_steps = config.halt_max_steps
    N = len(data['train_inputs'])
    
    pbar = tqdm(range(1, config.total_steps + 1), desc="Training", file=sys.stdout)
    
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    for step in pbar:
        # Sample random batch
        batch_indices = np.random.choice(N, config.batch_size, replace=False)
        
        batch_inputs = torch.tensor(data['train_inputs'][batch_indices], dtype=torch.long, device=device)
        batch_labels = torch.tensor(data['train_labels'][batch_indices], dtype=torch.long, device=device)
        batch_size_actual = batch_inputs.shape[0]
        
        batch = {
            "inputs": batch_inputs,
            "labels": batch_labels,
            "puzzle_identifiers": torch.zeros(batch_size_actual, dtype=torch.long, device=device),
        }
        
        # Update learning rate
        lr = cosine_schedule_with_warmup(step, config.total_steps, config.warmup_steps, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward
        optimizer.zero_grad()
        
        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        
        # Run all ACT steps
        for act_step in range(max_act_steps):
            carry, outputs = model(carry, batch)
        
        logits = outputs["logits"]
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch_labels.reshape(-1)
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track running metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == batch_labels).sum().item()
            running_loss += loss.item()
            running_correct += correct
            running_total += batch_size_actual * 81
        
        # Update progress bar
        if step % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_correct / running_total
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc': f'{avg_acc:.3f}',
                'lr': f'{lr:.1e}'
            })
            running_loss = 0.0
            running_correct = 0
            running_total = 0
        
        # Checkpoint
        if step in checkpoint_steps_set:
            print(f"\n\nEvaluating at step {step}...")
            
            model.eval()
            train_metrics = evaluate_model(
                model, data['train_inputs'], data['train_labels'],
                config, config.train_eval_samples
            )
            test_metrics = evaluate_model(
                model, data['test_inputs'], data['test_labels'],
                config, config.test_eval_samples
            )
            model.train()
            
            epoch_approx = step * config.batch_size / N
            
            snapshot = TrainingSnapshot(
                checkpoint_id=checkpoint_id,
                step=step,
                epoch=epoch_approx,
                train_loss=train_metrics['loss'],
                train_accuracy=train_metrics['accuracy'],
                train_unknown_accuracy=train_metrics['unknown_accuracy'],
                test_loss=test_metrics['loss'],
                test_accuracy=test_metrics['accuracy'],
                test_unknown_accuracy=test_metrics['unknown_accuracy'],
                learning_rate=lr,
                timestamp=datetime.now().isoformat(),
            )
            snapshots.append(snapshot)
            save_checkpoint(model, model_cfg, optimizer, snapshot, output_dir)
            checkpoint_id += 1
            
            gen_gap = train_metrics['accuracy'] - test_metrics['accuracy']
            print(f"  Step {step}: Train acc={train_metrics['accuracy']:.4f}, "
                  f"Test acc={test_metrics['accuracy']:.4f}, "
                  f"Gen gap={gen_gap:.4f}")
            
            # Save intermediate snapshots
            with open(output_dir / "snapshots.json", "w") as f:
                json.dump([asdict(s) for s in snapshots], f, indent=2)
    
    # Save final snapshots
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    with open(output_dir / "snapshots.json", "w") as f:
        json.dump([asdict(s) for s in snapshots], f, indent=2)
    
    print(f"\nSaved {len(snapshots)} checkpoints to {output_dir}")
    
    # Print summary
    print("\nTraining Summary:")
    print("-"*80)
    print(f"{'Step':<10} {'Epoch':<10} {'Train Acc':<12} {'Test Acc':<12} {'Gen Gap':<10}")
    print("-"*80)
    for s in snapshots:
        gen_gap = s.train_accuracy - s.test_accuracy
        print(f"{s.step:<10} {s.epoch:<10.1f} {s.train_accuracy:<12.4f} {s.test_accuracy:<12.4f} {gen_gap:<10.4f}")
    
    print(f"\nTo analyze grokking, run:")
    print(f"  python experiments/paper_replication/analyze_grokking_checkpoints.py \\")
    print(f"      --checkpoint_dir {output_dir}")


if __name__ == "__main__":
    main()
