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
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from torch import nn
from pretrain import PretrainConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm_v2.hrm_v2 import HierarchicalReasoningModel_V2
from scripts.core.activation_ablation import (
    ActivationAblator, ACTModel, _patch_attention_for_cpu,
)
from scripts.core.activation_patching import ActivationCache, compute_metrics
from scripts.core.sudoku_sample import collect_indexed_batches, load_puzzle_indices


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

def _create_eval_dataloader(
    config: PretrainConfig,
    *,
    pin_memory: bool,
) -> Tuple[Any, Any]:
    """Create the test dataloader used by analysis scripts.

    ``pretrain.create_dataloader`` defaults to one worker process. In this
    sandbox, torch multiprocessing cannot open its resource-sharing socket, so
    eval analysis defaults to single-process loading. Set
    ``HRM_EVAL_DATALOADER_WORKERS=1`` to opt back into the original worker
    behavior on a normal machine.
    """
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=1,
        rank=0,
        num_replicas=1,
    ), split="test")
    num_workers = int(os.environ.get("HRM_EVAL_DATALOADER_WORKERS", "0"))
    kwargs: Dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs.update({
            "prefetch_factor": 8,
            "persistent_workers": True,
        })
    return DataLoader(dataset, **kwargs), dataset.metadata

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

    test_loader, test_meta = _create_eval_dataloader(config, pin_memory=(device.type == "cuda"))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Puzzle-embedding reconciliation. The per-puzzle embedding table is sized by
    # num_puzzle_identifiers, which for ARC depends on the exact dataset build (the
    # augmentation unique-count of a few borderline puzzles varies with the build
    # machine's file ordering, so a rebuilt dataset can differ from the checkpoint
    # by a handful of rows). Size the model to max(checkpoint, dataset) so every
    # trained weight loads cleanly AND every dataset identifier stays in range; any
    # extra rows keep their fresh init. Sudoku/Maze sizes match, so this is a no-op.
    pe_suffix = "puzzle_emb.weights"
    ckpt_pe_key0 = next((k for k in ckpt if k.endswith(pe_suffix)), None)
    ckpt_pe_rows = int(ckpt[ckpt_pe_key0].shape[0]) if ckpt_pe_key0 is not None else 0
    data_pe_rows = int(test_meta.num_puzzle_identifiers)
    model_pe_rows = max(ckpt_pe_rows, data_pe_rows)
    if ckpt_pe_rows and ckpt_pe_rows != data_pe_rows:
        print(f"[load_model] WARNING puzzle_emb mismatch: checkpoint={ckpt_pe_rows} "
              f"dataset={data_pe_rows}; sizing model to {model_pe_rows} (extra rows keep "
              f"fresh init; identifier indices may be shifted vs training).")

    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=test_meta.vocab_size,
        seq_len=test_meta.seq_len,
        num_puzzle_identifiers=model_pe_rows,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

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

    # Pad the checkpoint's puzzle_emb to the model size, keeping the model's fresh
    # init for any extra rows so all params still load with assign=True.
    pe_key = next((k for k in ckpt if k.endswith(pe_suffix)), None)
    if pe_key is not None and ckpt[pe_key].shape[0] < model_pe_rows:
        model_pe = dict(model_full.state_dict())[pe_key]
        padded = model_pe.clone()
        padded[: ckpt[pe_key].shape[0]] = ckpt[pe_key].to(padded.dtype)
        ckpt[pe_key] = padded

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
    parser.add_argument(
        "--puzzle_indices", type=str, default=None,
        help="JSON manifest/list of dataloader puzzle indices to evaluate",
    )
    parser.add_argument(
        "--save_puzzle_indices", type=str, default=None,
        help="Write the collected dataloader puzzle indices to this JSON manifest",
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
    puzzle_indices_path: Optional[str] = None,
) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    """Collect a deterministic subset of puzzles from the test loader.

    Returns list of (puzzle_idx, batch_dict).
    """
    del seed  # The explicit index manifest is the reproducibility boundary.
    puzzle_indices = (
        load_puzzle_indices(puzzle_indices_path, limit=num_puzzles)
        if puzzle_indices_path else None
    )
    return collect_indexed_batches(
        test_loader,
        device,
        num_puzzles=num_puzzles,
        puzzle_indices=puzzle_indices,
        extract_batch=extract_batch,
    )
