#!/usr/bin/env python3
"""
Training script for HRM v2 on Sudoku.

This script trains the improved HRM model with:
- Constraint-aware sparse attention
- Constraint satisfaction head
- GNN layers for structure

Usage:
    python pretrain_v2.py
    
    # With custom config
    python pretrain_v2.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=100
    
    # Ablation: disable sparse attention
    python pretrain_v2.py arch.use_sparse_attention=false
    
    # Ablation: disable GNN
    python pretrain_v2.py arch.use_gnn_layers=false
"""

import os
import sys
import math
import yaml
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any, Sequence, List, Dict

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf
from adam_atan2_pytorch import AdamAtan2

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class PretrainV2Config(pydantic.BaseModel):
    arch: ArchConfig
    data_path: str
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


def create_dataloader(config: PretrainV2Config, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainV2Config, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    # Instantiate model using the @ syntax loader
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    # Get loss head config
    loss_cfg = dict(config.arch.loss.__pydantic_extra__) if hasattr(config.arch.loss, '__pydantic_extra__') else {}

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_cfg)
        
        # Compile for performance (optional)
        if "DISABLE_COMPILE" not in os.environ:
            try:
                model = torch.compile(model, dynamic=False)
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        # Broadcast parameters for distributed
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),
            lr=config.puzzle_emb_lr,
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size
        ),
        AdamAtan2(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup(
    current_step: int, 
    base_lr: float, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    min_ratio: float = 0.0
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def init_train_state(config: PretrainV2Config, train_metadata: PuzzleDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)
    
    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_checkpoint(config: PretrainV2Config, train_state: TrainState, is_best: bool = False):
    if config.checkpoint_path is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Save model state
    save_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
    torch.save(train_state.model.state_dict(), save_path)
    
    # Save config
    config_path = os.path.join(config.checkpoint_path, "config.yaml")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(), f)
    
    if is_best:
        best_path = os.path.join(config.checkpoint_path, "best.pt")
        shutil.copy(save_path, best_path)
    
    print(f"Saved checkpoint to {save_path}")


def train_step(config: PretrainV2Config, train_state: TrainState, batch: Dict, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None
    
    # Move batch to GPU
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)
    
    # Forward pass
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )
    
    # Backward
    ((1 / global_batch_size) * loss).backward()
    
    # All-reduce gradients for distributed
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    # Update learning rate and step
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = cosine_schedule_with_warmup(
            train_state.step, base_lr, 
            config.lr_warmup_steps, train_state.total_steps, 
            config.lr_min_ratio
        )
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()
    
    # Process metrics
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k].float() for k in metric_keys])
        
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        
        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            count = max(reduced_metrics.get("count", 1), 1)
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) 
                for k, v in reduced_metrics.items()
            }
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics
    
    return None


def evaluate(config: PretrainV2Config, train_state: TrainState, eval_loader, eval_metadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        metric_keys = []
        metric_values = None
        metric_counts = [0 for _ in range(len(set_ids))]
        
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
            
            # Forward until halted
            while True:
                carry, _, metrics, _, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=[]
                )
                if all_finish:
                    break
            
            # Accumulate metrics
            set_idx = set_ids.get(set_name, 0)
            metric_counts[set_idx] += global_batch_size
            
            if metric_keys == []:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(len(set_ids), len(metric_keys), device="cuda")
            
            for i, k in enumerate(metric_keys):
                if k in metrics:
                    metric_values[set_idx, i] += metrics[k].float()
        
        # Reduce across processes
        if world_size > 1 and metric_values is not None:
            dist.reduce(metric_values, dst=0)
        
        if rank == 0 and metric_values is not None:
            results = {}
            metric_values = metric_values.cpu().numpy()
            
            for set_name, set_idx in set_ids.items():
                count = max(metric_counts[set_idx], 1)
                for i, k in enumerate(metric_keys):
                    val = metric_values[set_idx, i]
                    if k.endswith("loss"):
                        val /= count
                    elif k == "count":
                        pass
                    else:
                        val /= max(metric_values[set_idx, metric_keys.index("count")], 1)
                    results[f"eval/{set_name}/{k}"] = val
            
            return results
    
    return {}


@hydra.main(config_path="config", config_name="cfg_pretrain_v2", version_base=None)
def main(cfg: DictConfig):
    # Convert OmegaConf to dict for pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = PretrainV2Config(**cfg_dict)
    
    # Setup
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print("HRM v2 Training")
        print("=" * 60)
        print(f"Config: {config.model_dump()}")
    
    # Create dataloaders
    train_loader, train_metadata = create_dataloader(
        config, split="train", rank=rank, world_size=world_size,
        global_batch_size=config.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1
    )
    eval_loader, eval_metadata = create_dataloader(
        config, split="test", rank=rank, world_size=world_size,
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1
    )
    
    if rank == 0:
        print(f"Train metadata: {train_metadata}")
        print(f"Eval metadata: {eval_metadata}")
    
    # Initialize training state
    train_state = init_train_state(config, train_metadata, world_size)
    
    if rank == 0:
        total_params = sum(p.numel() for p in train_state.model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Total training steps: {train_state.total_steps}")
    
    # Training loop
    best_accuracy = 0.0
    pbar = tqdm.tqdm(total=train_state.total_steps, desc="Training", disable=(rank != 0))
    
    for set_name, batch, global_batch_size in train_loader:
        metrics = train_step(config, train_state, batch, global_batch_size, rank, world_size)
        
        if metrics and rank == 0:
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{metrics.get('train/lm_loss', 0):.4f}",
                "acc": f"{metrics.get('train/accuracy', 0):.4f}",
            })
        
        # Evaluate periodically
        if config.eval_interval and train_state.step % config.eval_interval == 0:
            eval_metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank, world_size)
            
            if rank == 0 and eval_metrics:
                # Find accuracy across test sets
                accuracies = [v for k, v in eval_metrics.items() if "accuracy" in k and "exact" not in k]
                avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
                
                print(f"\nStep {train_state.step}: Eval accuracy = {avg_acc:.4f}")
                for k, v in sorted(eval_metrics.items()):
                    print(f"  {k}: {v:.4f}")
                
                # Save checkpoint
                if config.checkpoint_every_eval:
                    is_best = avg_acc > best_accuracy
                    if is_best:
                        best_accuracy = avg_acc
                    save_checkpoint(config, train_state, is_best=is_best)
        
        if train_state.step >= train_state.total_steps:
            break
    
    pbar.close()
    
    # Final evaluation
    if rank == 0:
        print("\nFinal evaluation...")
    eval_metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank, world_size)
    
    if rank == 0 and eval_metrics:
        print("\nFinal Results:")
        for k, v in sorted(eval_metrics.items()):
            print(f"  {k}: {v:.4f}")
        
        save_checkpoint(config, train_state, is_best=True)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
