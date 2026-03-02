"""Shared utilities for loading HRM models and data for experiments."""

import yaml
import json
from pathlib import Path

import numpy as np
import torch


def load_model_and_data(checkpoint_path: str, data_path: str, device: str = "cuda"):
    """Load model checkpoint and dataset.
    
    Args:
        checkpoint_path: Path to checkpoint.pt file
        data_path: Path to data directory (e.g. data/sudoku-extreme-1k-aug-1000)
        device: Device to load model on
        
    Returns:
        model: Loaded HierarchicalReasoningModel_ACTV1
        cfg: Model config dict
        inputs: Test inputs array [N, 81]
        labels: Test labels array [N, 81]
    """
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load config from YAML file
    config_path = Path(checkpoint_path).parent / "all_config.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    # Load dataset metadata
    dataset_json_path = Path(data_path) / "test" / "dataset.json"
    with open(dataset_json_path, "r") as f:
        dataset_metadata = json.load(f)
    
    # Extract arch config and add required fields from dataset metadata
    cfg = full_config['arch'].copy()
    cfg['batch_size'] = full_config.get('global_batch_size', 768)
    cfg['seq_len'] = dataset_metadata['seq_len']
    cfg['vocab_size'] = dataset_metadata['vocab_size']
    cfg['num_puzzle_identifiers'] = dataset_metadata['num_puzzle_identifiers']
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(cfg).to(device)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle compiled model prefix (_orig_mod.) and loss head wrapper (model.)
    # The checkpoint keys look like: _orig_mod.model.inner.X
    # We need: inner.X
    def fix_key(k):
        k = k.removeprefix("_orig_mod.")  # Remove torch.compile prefix
        k = k.removeprefix("model.")       # Remove ACTLossHead wrapper prefix
        return k
    
    fixed_state_dict = {fix_key(k): v for k, v in checkpoint.items()}
    model.load_state_dict(fixed_state_dict, assign=True)
    
    model.eval()
    
    print(f"Loading dataset from {data_path}...")
    # Load test inputs and labels directly
    test_dir = Path(data_path) / "test"
    inputs = np.load(test_dir / "all__inputs.npy")
    labels = np.load(test_dir / "all__labels.npy")
    
    return model, cfg, inputs, labels


def move_carry_to_device(carry, device):
    """Move all tensors in carry to the specified device."""
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    return carry
