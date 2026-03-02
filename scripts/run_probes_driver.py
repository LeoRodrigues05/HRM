"""Probe Collection Driver for HRM Model.

This script runs the HRM model on a test dataset and collects hidden state
information (z_H and z_L) at each ACT step for linear probe analysis.

Environment Variables:
    PROBE_BATCH_SIZE: Batch size for inference (default: 8)
    HRM_HALT_MAX_STEPS: Maximum ACT steps (default: 8)
    MAX_PROBE_BATCHES: Maximum batches to process (default: 10)
    CPU_ONLY: Set to "1" to force CPU execution

Usage:
    python scripts/run_probes_driver.py
    
    # With custom settings:
    PROBE_BATCH_SIZE=4 MAX_PROBE_BATCHES=20 python scripts/run_probes_driver.py

Output:
    - results/probes/probe_global.pt: Pooled hidden states per step
    - results/probes/probe_local.pt: Per-token hidden states per step
    - results/probes/probe_index.json: Collection metadata
"""
import os
import sys
import json
import logging
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure repo root is on sys.path when executed as `python scripts/run_probes_driver.py`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.probes import ProbeRecorder


# ============================================================================
# Configuration
# ============================================================================

# Default model configuration
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "puzzle_emb_ndim": 0,
    "num_puzzle_identifiers": 1024,
    "H_cycles": 2,
    "L_cycles": 2,
    "H_layers": 2,
    "L_layers": 2,
    "hidden_size": 512,
    "expansion": 2.0,
    "num_heads": 8,
    "pos_encodings": "rope",
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "halt_max_steps": 8,
    "halt_exploration_prob": 0.0,
    "forward_dtype": "bfloat16",
}

# Default dataset path
DEFAULT_DATASET_ROOT = os.path.join("data", "sudoku-extreme-1k-aug-1000")


# ============================================================================
# Main Driver
# ============================================================================

def main() -> None:
    """Main entry point for probe collection."""
    # Parse environment configuration
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "8"))
    halt_steps_env = os.environ.get("HRM_HALT_MAX_STEPS")
    max_batches = int(os.environ.get("MAX_PROBE_BATCHES", "10"))
    cpu_only = os.environ.get("CPU_ONLY", "0") == "1"
    
    device = torch.device("cpu") if cpu_only else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running probe collection on device={device}, batch_size={batch_size}, max_batches={max_batches}")

    # Detect dataset paths and seq_len
    dataset_root = DEFAULT_DATASET_ROOT
    meta_path = os.path.join(dataset_root, "test", "dataset.json")
    inputs_path = os.path.join(dataset_root, "test", "all__inputs.npy")

    if not (os.path.exists(meta_path) and os.path.exists(inputs_path)):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. "
            "Please build it via: python dataset/build_sudoku_dataset.py"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)
    vocab_size = meta.get("vocab_size", 512)

    inp = np.load(inputs_path, mmap_mode="r")
    seq_len_detected = int(inp.shape[1])
    logger.info(f"Dataset: vocab_size={vocab_size}, seq_len={seq_len_detected}")

    # Build model configuration
    config = {
        **DEFAULT_MODEL_CONFIG,
        "batch_size": batch_size,
        "seq_len": seq_len_detected,
        "vocab_size": vocab_size,
    }

    # Apply environment override for halt steps
    if halt_steps_env is not None:
        try:
            config["halt_max_steps"] = int(halt_steps_env)
            logger.info(f"Using halt_max_steps={config['halt_max_steps']} from environment")
        except ValueError:
            logger.warning(f"Invalid HRM_HALT_MAX_STEPS value: {halt_steps_env}, using default")

    # Initialize model
    logger.info("Initializing HRM model...")
    model = HierarchicalReasoningModel_ACTV1(config).to(device)
    model.eval()

    # Initialize dataset
    logger.info(f"Loading dataset from {dataset_root}...")
    ds_cfg = PuzzleDatasetConfig(
        seed=42,
        dataset_path=dataset_root,
        global_batch_size=batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    ds = PuzzleDataset(ds_cfg, split="test")
    dl = DataLoader(ds, batch_size=None, shuffle=False)

    # Initialize probe recorder
    output_dir = os.path.join("results", "probes")
    rec = ProbeRecorder(output_dir=output_dir)
    logger.info(f"Probe data will be saved to {output_dir}")

    # Run inference and collect probes
    processed = 0
    total_steps_recorded = 0
    
    with torch.no_grad():
        for _set_name, batch, _count in dl:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Initialize carry and move tensors to device
            carry = model.initial_carry(batch)
            carry.inner_carry = type(carry.inner_carry)(
                z_H=carry.inner_carry.z_H.to(device),
                z_L=carry.inner_carry.z_L.to(device),
            )
            carry.steps = carry.steps.to(device)
            carry.halted = carry.halted.to(device)
            carry.current_data = {k: v.to(device) for k, v in carry.current_data.items()}

            # ACT loop - record hidden states at each step
            for step in range(model.config.halt_max_steps):
                carry, outputs = model(carry, batch, probe_recorder=None)
                
                # Record with predictions for better label derivation
                preds_step = outputs.get("intermediate_preds_step")
                step_index = int(carry.steps.max().item()) if torch.is_tensor(carry.steps) else step
                
                rec.record_hidden(
                    step_index=step_index,
                    phase="grad",
                    z_H=carry.inner_carry.z_H,
                    z_L=carry.inner_carry.z_L,
                    batch=batch,
                    preds=preds_step
                )
                total_steps_recorded += 1
                
                if torch.all(carry.halted):
                    break
                    
            processed += 1
            if processed % 5 == 0:
                logger.info(f"Processed {processed}/{max_batches} batches...")
                
            if processed >= max_batches:
                break

    # Finalize and save
    logger.info(f"Collected data from {processed} batches, {total_steps_recorded} total step recordings")
    rec.finalize_and_save()
    logger.info(f"Saved probe datasets to {output_dir}")


if __name__ == "__main__":
    main()
