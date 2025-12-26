import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.probes import ProbeRecorder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect dataset paths and seq_len
    dataset_root = os.path.join("data", "sudoku-extreme-1k-aug-1000")
    meta_path = os.path.join(dataset_root, "test", "dataset.json")
    inputs_path = os.path.join(dataset_root, "test", "all__inputs.npy")

    if not (os.path.exists(meta_path) and os.path.exists(inputs_path)):
        raise FileNotFoundError("Dataset not found. Please build it via dataset/build_sudoku_dataset.py.")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    vocab_size = meta.get("vocab_size", 512)

    inp = np.load(inputs_path, mmap_mode="r")
    seq_len_detected = int(inp.shape[1])

    # Conservative batch size to avoid OOM (override with PROBE_BATCH_SIZE)
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "8"))
    # Limit ACT steps if needed (override with HRM_HALT_MAX_STEPS)
    halt_steps_env = os.environ.get("HRM_HALT_MAX_STEPS")
    # Limit number of batches processed to avoid OOM (override with MAX_PROBE_BATCHES)
    max_batches = int(os.environ.get("MAX_PROBE_BATCHES", "10"))
    # Optional CPU-only run to avoid GPU OOM (set CPU_ONLY=1)
    cpu_only = os.environ.get("CPU_ONLY", "0") == "1"

    # Model config: match dataset seq_len, disable puzzle embeddings to avoid length mismatch
    config = {
        "batch_size": batch_size,
        "seq_len": seq_len_detected,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1024,
        "vocab_size": vocab_size,
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

    # Apply env override for halt steps
    if halt_steps_env is not None:
        try:
            config["halt_max_steps"] = int(halt_steps_env)
        except Exception:
            pass

    # Optional CPU-only execution
    if cpu_only:
        device = torch.device("cpu")

    model = HierarchicalReasoningModel_ACTV1(config).to(device)
    model.eval()

    # Dataset config (test mode)
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

    rec = ProbeRecorder(output_dir=os.path.join("results", "probes"))

    processed = 0
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

            # ACT loop
            for _ in range(model.config.halt_max_steps):
                carry, outputs = model(carry, batch, probe_recorder=None)
                # Record with predictions for better labels
                preds_step = outputs.get("intermediate_preds_step")
                # Use current z_H/z_L from carry.inner_carry
                rec.record_hidden(step_index=int(carry.steps.max().item()) if torch.is_tensor(carry.steps) else 0,
                                  phase="grad",
                                  z_H=carry.inner_carry.z_H,
                                  z_L=carry.inner_carry.z_L,
                                  batch=batch,
                                  preds=preds_step)
                if torch.all(carry.halted):
                    break
            processed += 1
            if processed >= max_batches:
                break

    rec.finalize_and_save()
    print("Saved probe datasets to results/probes")


if __name__ == "__main__":
    main()
