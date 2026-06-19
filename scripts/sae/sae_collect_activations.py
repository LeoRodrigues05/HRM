#!/usr/bin/env python3
"""SAE E10: Collect z_H (and optionally z_L) activations for SAE training.

Runs forward passes on N puzzles through the HRM model, recording z_H
at each of the 16 ACT steps. Saves flattened activations plus metadata
for downstream SAE training.

Output
------
  results/sae_study/activations_zH.pt   — dict with:
      'activations': [N_puzzles, 16, 81, 512]
      'metadata': per-puzzle info (inputs, labels, puzzle_ids, per-step accuracy)
  results/sae_study/activations_zL.pt   — same, if --also_collect_zL

Usage
-----
    python scripts/sae/sae_collect_activations.py --n_puzzles 1000
    python scripts/sae/sae_collect_activations.py --n_puzzles 100 --also_collect_zL --device cpu
"""

import os
import sys
import argparse
import logging
import time
from typing import Any, Dict, List, Optional

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

SUDOKU_SIZE = 9
SUDOKU_CELLS = 81
DIGIT_OFFSET = 1


# ═══════════════════════════════════════════════════════════════════════════
# Model / Data loading (matches E8/E9 pattern)
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_checkpoint_file(ckpt_dir: str, ckpt_file: Optional[str]) -> str:
    """Return the path to the weights file inside ckpt_dir.

    If ckpt_file is given, use it. Otherwise prefer checkpoint.pt, then fall back
    to the latest pretrain.py-style `step_<N>` snapshot (highest N).
    """
    if ckpt_file:
        return os.path.join(ckpt_dir, ckpt_file)
    default = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.exists(default):
        return default
    step_files = [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and "." not in f]
    if not step_files:
        raise FileNotFoundError(
            f"No checkpoint.pt or step_<N> file found in {ckpt_dir}")
    latest = max(step_files, key=lambda f: int(f.split("_")[1]))
    return os.path.join(ckpt_dir, latest)


def load_model_and_data(device: torch.device,
                        ckpt_dir: Optional[str] = None,
                        ckpt_file: Optional[str] = None):
    """Load checkpoint and create test data loader (batch_size=1).

    Args:
        ckpt_dir: checkpoint directory. Defaults to the bundled sapientinc model.
        ckpt_file: weights filename inside ckpt_dir. If None, auto-resolves to
            checkpoint.pt or the latest step_<N> snapshot.
    """
    if ckpt_dir is None:
        ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "sapientinc-sudoku-extreme")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(os.path.join(ckpt_dir, "all_config.yaml")):
        config_path = os.path.join(ckpt_dir, "all_config.yaml")
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

    weights_path = _resolve_checkpoint_file(ckpt_dir, ckpt_file)
    logger.info(f"Loading weights from {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
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
    max_steps: int = 16,
    also_collect_zL: bool = False,
) -> Dict[str, Any]:
    """Collect z_H (and optionally z_L) at every ACT step for every puzzle.

    Returns dict with:
        'z_H': [N, steps, 81, D] tensor
        'z_L': [N, steps, 81, D] tensor  (if also_collect_zL)
        'inputs': [N, 81] token IDs
        'labels': [N, 81] token IDs
        'puzzle_ids': [N] puzzle identifiers
        'per_step_accuracy': [N, steps] per-step cell accuracy
        'per_step_preds': [N, steps, 81] predictions at each step
    """
    ablator = ActivationAblator(model, device=device)

    # Collect batches
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

        targets_tok = batch["labels"][:, -SUDOKU_CELLS:]  # [1, 81]
        inputs_tok = batch["inputs"][:, -SUDOKU_CELLS:]   # [1, 81]
        puzzle_id = batch["puzzle_identifiers"]            # [1]

        # Collect z_H at each step
        n_steps_actual = len(cache)
        z_H_steps = []
        z_L_steps = []
        step_accs = []
        step_preds_list = []

        for step in range(max_steps):
            if step in cache:
                ac = cache[step]
                z_H = ac.z_H_out[:, -SUDOKU_CELLS:, :].squeeze(0).float().cpu()  # [81, D]
                z_H_steps.append(z_H)
                if also_collect_zL:
                    z_L = ac.z_L_out[:, -SUDOKU_CELLS:, :].squeeze(0).float().cpu()
                    z_L_steps.append(z_L)

                preds = ac.preds[:, -SUDOKU_CELLS:].cpu()  # [1, 81]
                acc = (preds.view(-1) == targets_tok.view(-1).cpu()).float().mean().item()
                step_accs.append(acc)
                step_preds_list.append(preds.squeeze(0))  # [81]
            else:
                # Pad with zeros if step not reached
                z_H_steps.append(torch.zeros(SUDOKU_CELLS, 512))
                if also_collect_zL:
                    z_L_steps.append(torch.zeros(SUDOKU_CELLS, 512))
                step_accs.append(0.0)
                step_preds_list.append(torch.zeros(SUDOKU_CELLS, dtype=torch.long))

        all_z_H.append(torch.stack(z_H_steps))        # [steps, 81, D]
        if also_collect_zL:
            all_z_L.append(torch.stack(z_L_steps))
        all_inputs.append(inputs_tok.squeeze(0).cpu())  # [81]
        all_labels.append(targets_tok.squeeze(0).cpu())  # [81]
        all_puzzle_ids.append(puzzle_id.cpu())
        all_step_acc.append(step_accs)
        all_step_preds.append(torch.stack(step_preds_list))  # [steps, 81]

        if (pi + 1) % 50 == 0:
            logger.info(f"  {pi+1}/{len(batches)} puzzles processed")

    result = {
        'z_H': torch.stack(all_z_H),                    # [N, steps, 81, D]
        'inputs': torch.stack(all_inputs),                # [N, 81]
        'labels': torch.stack(all_labels),                # [N, 81]
        'puzzle_ids': torch.cat(all_puzzle_ids),          # [N]
        'per_step_accuracy': torch.tensor(all_step_acc),  # [N, steps]
        'per_step_preds': torch.stack(all_step_preds),    # [N, steps, 81]
        'n_puzzles': len(batches),
        'max_steps': max_steps,
        'hidden_dim': 512,
    }
    if also_collect_zL:
        result['z_L'] = torch.stack(all_z_L)              # [N, steps, 81, D]

    logger.info(f"Done. z_H shape: {result['z_H'].shape}")
    return result


def main():
    parser = argparse.ArgumentParser(description="SAE E10: Collect activations for SAE training")
    parser.add_argument("--n_puzzles", type=int, default=1000,
                        help="Number of puzzles to collect activations from")
    parser.add_argument("--max_steps", type=int, default=16,
                        help="Number of ACT steps")
    parser.add_argument("--also_collect_zL", action="store_true",
                        help="Also collect z_L activations")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")
    parser.add_argument("--output_dir", type=str, default="results/sae_study",
                        help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint directory (default: bundled sapientinc model)")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                        help="Weights filename inside checkpoint_dir "
                             "(default: checkpoint.pt or latest step_<N>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load model and data
    model, test_loader, test_meta = load_model_and_data(
        device, ckpt_dir=args.checkpoint_dir, ckpt_file=args.checkpoint_file)
    logger.info("Model loaded.")

    # Collect activations
    t0 = time.time()
    result = collect_all_activations(
        model, test_loader, device,
        n_puzzles=args.n_puzzles,
        max_steps=args.max_steps,
        also_collect_zL=args.also_collect_zL,
    )
    elapsed = time.time() - t0
    logger.info(f"Collection took {elapsed:.1f}s")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    z_H_path = os.path.join(args.output_dir, "activations_zH.pt")
    torch.save(result, z_H_path)
    file_size_mb = os.path.getsize(z_H_path) / (1024 * 1024)
    logger.info(f"Saved z_H activations to {z_H_path} ({file_size_mb:.1f} MB)")
    logger.info(f"  z_H shape: {result['z_H'].shape}")
    logger.info(f"  per_step_accuracy shape: {result['per_step_accuracy'].shape}")
    logger.info(f"  Mean final-step accuracy: {result['per_step_accuracy'][:, -1].mean():.4f}")

    if args.also_collect_zL:
        logger.info(f"  z_L shape: {result['z_L'].shape}")
        logger.info(f"  (z_L stored in same file)")


if __name__ == "__main__":
    main()
