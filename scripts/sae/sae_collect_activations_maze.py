#!/usr/bin/env python3
"""Maze sibling of ``sae_collect_activations.py``.

Runs forward passes on N puzzles through the HRM-Maze model, recording z_H
(and optionally z_L) at each ACT step. Saves flattened activations plus
metadata for downstream SAE training and linear-probe analysis (Phases 3, 5).

Defaults are tuned for the Maze-30x30-hard checkpoint shipped under
``checkpoints/sapientinc-hrm-maze-30x30-hard/``:

    seq_len     = 900   (30x30 grid)
    vocab_size  = 6     (pad + ``# SGo``)
    hidden_size = 512
    halt_max_steps = 16

Activations are stored as ``float16`` by default to keep the on-disk tensor
manageable: ``[N, 16, 900, 512]`` = ~14 GB float32 / ~7 GB float16 at N=500.

Output (under ``--output_dir``)
-------------------------------
    activations_zH.pt   dict with:
        'z_H': [N, halt_max_steps, seq_len, hidden_size] (float16/32)
        'inputs': [N, seq_len] uint8/long token IDs
        'labels': [N, seq_len] uint8/long token IDs
        'puzzle_ids': [N]
        'per_step_accuracy': [N, halt_max_steps]
        'per_step_preds': [N, halt_max_steps, seq_len]
        'n_puzzles', 'max_steps', 'seq_len', 'hidden_dim'
    activations_zL.pt   same shape, if --also_collect_zL is passed.

Usage
-----
    python scripts/sae/sae_collect_activations_maze.py --n_puzzles 500
    python scripts/sae/sae_collect_activations_maze.py --n_puzzles 5 \
        --output_dir /tmp/sae_smoke_maze
"""

import os
import sys
import argparse
import logging
import time
from typing import Any, Dict, Optional

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _patch_attention_for_cpu,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "sapientinc-hrm-maze-30x30-hard")


# ═══════════════════════════════════════════════════════════════════════════
# Model / Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data(device: torch.device, ckpt_dir: str):
    """Load checkpoint and create test data loader (batch_size=1).

    The checkpoint directory must contain ``all_config.yaml`` (or
    ``config.yaml``) and a checkpoint file named either ``checkpoint`` or
    ``checkpoint.pt``. The Maze HF checkpoint uses the bare ``checkpoint``
    filename.
    """
    config_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_meta = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__, batch_size=1,
        vocab_size=test_meta.vocab_size, seq_len=test_meta.seq_len,
        num_puzzle_identifiers=test_meta.num_puzzle_identifiers, causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt_filename: Optional[str] = None
    for cand in ("checkpoint", "checkpoint.pt"):
        if os.path.exists(os.path.join(ckpt_dir, cand)):
            ckpt_filename = cand
            break
    if ckpt_filename is None:
        raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir}")

    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_filename),
                      map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(
        k.startswith("_orig_mod.") for k in ck
    ):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(
        k.startswith("_orig_mod.") for k in mk
    ):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, "model"):
        m = m.model

    return m, test_loader, test_meta


# ═══════════════════════════════════════════════════════════════════════════
# Activation collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_all_activations(
    model: HierarchicalReasoningModel_ACTV1,
    test_loader,
    device: torch.device,
    n_puzzles: int,
    seq_len: int,
    hidden_dim: int,
    max_steps: int = 16,
    also_collect_zL: bool = False,
    store_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """Collect z_H (and optionally z_L) at every ACT step for every puzzle."""
    ablator = ActivationAblator(model, device=device)
    cells = seq_len  # task-agnostic alias

    # Materialize batches
    batches = []
    for i, data in enumerate(test_loader):
        if i >= n_puzzles:
            break
        if isinstance(data, (tuple, list)):
            batch = data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        else:
            batch = data
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batches.append(batch)

    logger.info(f"Loaded {len(batches)} puzzles. Running inference...")

    all_z_H = []
    all_z_L = [] if also_collect_zL else None
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []
    all_step_acc = []
    all_step_preds = []

    for pi, batch in enumerate(batches):
        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)

        targets_tok = batch["labels"][:, -cells:]   # [1, seq_len]
        inputs_tok = batch["inputs"][:, -cells:]    # [1, seq_len]
        puzzle_id = batch["puzzle_identifiers"]      # [1]

        z_H_steps = []
        z_L_steps = []
        step_accs = []
        step_preds_list = []

        for step in range(max_steps):
            if step in cache:
                ac = cache[step]
                z_H = ac.z_H_out[:, -cells:, :].squeeze(0).to(store_dtype).cpu()
                z_H_steps.append(z_H)
                if also_collect_zL:
                    z_L = ac.z_L_out[:, -cells:, :].squeeze(0).to(store_dtype).cpu()
                    z_L_steps.append(z_L)

                preds = ac.preds[:, -cells:].cpu()
                acc = (preds.view(-1) == targets_tok.view(-1).cpu()).float().mean().item()
                step_accs.append(acc)
                step_preds_list.append(preds.squeeze(0))
            else:
                # Pad zero if step not reached (early ACT halt)
                z_H_steps.append(torch.zeros(cells, hidden_dim, dtype=store_dtype))
                if also_collect_zL:
                    z_L_steps.append(torch.zeros(cells, hidden_dim, dtype=store_dtype))
                step_accs.append(0.0)
                step_preds_list.append(torch.zeros(cells, dtype=torch.long))

        all_z_H.append(torch.stack(z_H_steps))
        if also_collect_zL:
            all_z_L.append(torch.stack(z_L_steps))
        all_inputs.append(inputs_tok.squeeze(0).cpu())
        all_labels.append(targets_tok.squeeze(0).cpu())
        all_puzzle_ids.append(puzzle_id.cpu())
        all_step_acc.append(step_accs)
        all_step_preds.append(torch.stack(step_preds_list))

        if (pi + 1) % 50 == 0:
            logger.info(f"  {pi+1}/{len(batches)} puzzles processed")

    result = {
        'z_H': torch.stack(all_z_H),
        'inputs': torch.stack(all_inputs),
        'labels': torch.stack(all_labels),
        'puzzle_ids': torch.cat(all_puzzle_ids),
        'per_step_accuracy': torch.tensor(all_step_acc),
        'per_step_preds': torch.stack(all_step_preds),
        'n_puzzles': len(batches),
        'max_steps': max_steps,
        'seq_len': cells,
        'hidden_dim': hidden_dim,
    }
    if also_collect_zL:
        result['z_L'] = torch.stack(all_z_L)

    logger.info(f"Done. z_H shape: {result['z_H'].shape}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect HRM-Maze z_H/z_L activations for SAE / probe training")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--also_collect_zL", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/maze/sae_study")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--store_dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    model, test_loader, test_meta = load_model_and_data(device, ckpt_dir=args.checkpoint_dir)
    logger.info("Model loaded. seq_len=%d vocab_size=%d hidden=%d",
                test_meta.seq_len, test_meta.vocab_size, model.config.hidden_size)

    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32

    t0 = time.time()
    result = collect_all_activations(
        model, test_loader, device,
        n_puzzles=args.n_puzzles,
        seq_len=test_meta.seq_len,
        hidden_dim=model.config.hidden_size,
        max_steps=args.max_steps,
        also_collect_zL=args.also_collect_zL,
        store_dtype=store_dtype,
    )
    elapsed = time.time() - t0
    logger.info(f"Collection took {elapsed:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)

    z_H_path = os.path.join(args.output_dir, "activations_zH.pt")
    if args.also_collect_zL:
        # Split into two files so each can be loaded independently
        z_H_only = {k: v for k, v in result.items() if k != 'z_L'}
        torch.save(z_H_only, z_H_path)
        z_L_path = os.path.join(args.output_dir, "activations_zL.pt")
        z_L_only = {k: (result['z_L'] if k == 'z_H' else v)
                    for k, v in z_H_only.items()}
        # rename 'z_H' field of the z_L file to 'z_L' for clarity
        z_L_only.pop('z_H', None)
        z_L_only['z_L'] = result['z_L']
        torch.save(z_L_only, z_L_path)
        logger.info(f"Saved {z_H_path} ({os.path.getsize(z_H_path)/2**20:.1f} MB) "
                    f"and {z_L_path} ({os.path.getsize(z_L_path)/2**20:.1f} MB)")
    else:
        torch.save(result, z_H_path)
        logger.info(f"Saved {z_H_path} ({os.path.getsize(z_H_path)/2**20:.1f} MB)")
    logger.info(f"  z_H shape: {result['z_H'].shape}")
    logger.info(f"  Mean final-step accuracy: {result['per_step_accuracy'][:, -1].mean():.4f}")


if __name__ == "__main__":
    main()
