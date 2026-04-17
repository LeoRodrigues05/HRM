"""scripts/controlled_common.py

Shared utilities for controlled experiment scripts.
Provides: checkpoint auto-detection, model loading, batch extraction,
bootstrap CI, and common argument parsing.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, cast

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from torch import nn
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm_v2.hrm_v2 import HierarchicalReasoningModel_V2
from scripts.core.activation_ablation import (
    ActivationAblator, ACTModel, _patch_attention_for_cpu,
)
from scripts.core.activation_patching import ActivationCache, compute_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint auto-detection
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_CANDIDATES = [
    os.path.join(REPO_ROOT, "checkpoints", "sapientinc-sudoku-extreme", "checkpoint.pt"),
    os.path.join(REPO_ROOT, "Checkpoint_HRM_Sudoku", "Checkpoint_HRM_Sudoku",
                 "Checkpoint_HRM_Sudoku", "checkpoint.pt"),
]


def find_checkpoint() -> str:
    for p in CHECKPOINT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Tried: {CHECKPOINT_CANDIDATES}. "
        "Use --checkpoint to specify path."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_dataloader(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ACTModel, Any, PretrainConfig]:
    """Load model + test dataloader from checkpoint.

    Returns (unwrapped_act_model, test_loader, config).
    """
    config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_meta = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=test_meta.vocab_size,
        seq_len=test_meta.seq_len,
        num_puzzle_identifiers=test_meta.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    # Unwrap to ACT model
    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        if hasattr(m, "model"):
            m = m.model
    if not isinstance(m, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        raise TypeError(f"Expected ACT model, got {type(m)}")

    return cast(ACTModel, m), test_loader, config


# ═══════════════════════════════════════════════════════════════════════════
# Batch extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_batch(item) -> Dict[str, torch.Tensor]:
    if isinstance(item, (tuple, list)):
        if len(item) >= 2 and isinstance(item[1], dict):
            return item[1]
        if isinstance(item[0], dict):
            return item[0]
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported dataloader item: {type(item)}")


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute mean and bootstrap confidence interval."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    rng = np.random.RandomState(seed)
    means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(means, alpha * 100)),
        "ci_upper": float(np.percentile(means, (1 - alpha) * 100)),
        "std": float(np.std(arr)),
        "n": len(arr),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Common argument parser
# ═══════════════════════════════════════════════════════════════════════════

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint.pt (auto-detected if omitted)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode: N=20 puzzles",
    )
    return parser


def resolve_args(args) -> argparse.Namespace:
    """Fill in auto-detected defaults."""
    if args.checkpoint is None:
        args.checkpoint = find_checkpoint()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ═══════════════════════════════════════════════════════════════════════════
# Seeded puzzle collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_puzzles(
    test_loader,
    device: torch.device,
    num_puzzles: int,
    seed: int = 42,
) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    """Collect a deterministic subset of puzzles from the test loader.

    Returns list of (puzzle_idx, batch_dict).
    """
    puzzles = []
    for idx, data in enumerate(test_loader):
        if len(puzzles) >= num_puzzles:
            break
        batch = extract_batch(data)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        puzzles.append((idx, batch))
    return puzzles
