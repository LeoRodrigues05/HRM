#!/usr/bin/env python3
"""E8: Constraint-Specific Linear Probes at Scale.

Trains per-cell linear probes on z_H (and z_L) to predict Sudoku constraint
properties at each ACT step. After training, extracts probe weight vectors
and analyzes their geometric relationships (orthogonality, PCA overlap).

Key targets
-----------
  Binary (per-cell):
    violated_in_row   – 1 if this cell's digit is duplicated in its row
    violated_in_col   – 1 if this cell's digit is duplicated in its column
    violated_in_box   – 1 if this cell's digit is duplicated in its box
    is_naked_single   – 1 if exactly 1 candidate remains (forced cell)
    is_hidden_single_row/col/box – hidden single in that unit
    per_cell_correct   – 1 if cell matches the solution
    is_given           – 1 if cell was a clue

  Regression (per-cell):
    candidate_count   – number of valid candidates (0–9)
    filled_in_row/col/box – how many cells are filled in the unit

  Multiclass (per-cell, 9 classes):
    cell_digit        – the digit at this cell (1–9, 0=empty)

Output
------
  results/e8_constraint_probes/
    probe_weights.pt         – dict mapping target→{W, b, accuracy/r2, in_dim, ...}
    geometric_analysis.json  – pairwise cosine of weight vectors, PCA projections
    sweep_results.csv        – full accuracy table

Usage
-----
    # Quick test (5 puzzles)
    python scripts/e8_constraint_probes.py --n_puzzles 5

    # Full run
    python scripts/e8_constraint_probes.py --n_puzzles 500 --steps 0,4,8,12,15
"""

import os
import sys
import json
import csv
import argparse
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Ensure repo root is on Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _patch_attention_for_cpu
)
from scripts.core.sudoku_sample import (
    collect_indexed_batches,
    load_puzzle_indices,
    save_puzzle_indices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Sudoku constants ─────────────────────────────────────────────────────
SUDOKU_SIZE = 9
SUDOKU_CELLS = 81
DIGIT_OFFSET = 1  # token_id = digit + 1; blank=1

# ── Probe targets ────────────────────────────────────────────────────────
BINARY_TARGETS = [
    "per_cell_correct",
    "is_given",
    "is_empty",
    "violated_in_row",
    "violated_in_col",
    "violated_in_box",
    "is_naked_single",
    "is_hidden_single_row",
    "is_hidden_single_col",
    "is_hidden_single_box",
]

REGRESSION_TARGETS = [
    "candidate_count",
    "filled_in_row",
    "filled_in_col",
    "filled_in_box",
]

MULTICLASS_TARGETS = [
    "cell_digit",       # 10 classes: 0-9
]

ALL_TARGETS = BINARY_TARGETS + REGRESSION_TARGETS + MULTICLASS_TARGETS


# ═══════════════════════════════════════════════════════════════════════════
# Label derivation (self-contained, no dependency on utils/probes.py)
# ═══════════════════════════════════════════════════════════════════════════

def derive_per_cell_labels(
    preds: torch.Tensor,       # [B, 81] predicted token ids
    targets: torch.Tensor,     # [B, 81] ground truth token ids
    inputs: torch.Tensor,      # [B, 81] original input token ids
) -> Dict[str, torch.Tensor]:
    """Derive constraint labels for every cell. Returns dict of [B,81] tensors."""
    B = preds.shape[0]
    labels: Dict[str, torch.Tensor] = {}

    # Convert token ids → digits  (0 = empty, 1-9 = filled)
    digits = (preds.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    target_digits = (targets.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    input_digits = (inputs.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)

    grid = digits.view(B, SUDOKU_SIZE, SUDOKU_SIZE)

    # ── Simple per-cell binary ────────────────────────────────────────
    labels["per_cell_correct"] = (digits == target_digits).int()
    labels["is_given"] = (input_digits != 0).int()
    labels["is_empty"] = (digits == 0).int()

    # ── Per-cell violation in row/col/box ─────────────────────────────
    violated_row = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    violated_col = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    violated_box = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)

    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            d = grid[:, r, c]  # [B]
            nonzero = d > 0     # [B]

            # row check: count occurrences of d in row r
            row_count = (grid[:, r, :] == d.unsqueeze(1)).sum(dim=1)  # [B]
            violated_row[:, idx] = (nonzero & (row_count > 1)).int()

            # col check
            col_count = (grid[:, :, c] == d.unsqueeze(1)).sum(dim=1)
            violated_col[:, idx] = (nonzero & (col_count > 1)).int()

            # box check
            br, bc = (r // 3) * 3, (c // 3) * 3
            box = grid[:, br:br+3, bc:bc+3].reshape(B, 9)
            box_count = (box == d.unsqueeze(1)).sum(dim=1)
            violated_box[:, idx] = (nonzero & (box_count > 1)).int()

    labels["violated_in_row"] = violated_row
    labels["violated_in_col"] = violated_col
    labels["violated_in_box"] = violated_box

    # ── Cell digit (multiclass) ───────────────────────────────────────
    labels["cell_digit"] = digits.int()  # 0-9

    # ── Candidate-based features ──────────────────────────────────────
    # Pre-compute used digits per unit
    used_row = torch.zeros(B, SUDOKU_SIZE, SUDOKU_SIZE + 1, dtype=torch.bool)
    used_col = torch.zeros(B, SUDOKU_SIZE, SUDOKU_SIZE + 1, dtype=torch.bool)
    used_box = torch.zeros(B, SUDOKU_SIZE, SUDOKU_SIZE + 1, dtype=torch.bool)

    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            d = grid[:, r, c]
            b = (r // 3) * 3 + (c // 3)
            for dd in range(1, SUDOKU_SIZE + 1):
                mask = (d == dd)
                used_row[:, r, dd] |= mask
                used_col[:, c, dd] |= mask
                used_box[:, b, dd] |= mask

    candidate_count = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    is_naked_single = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    candidate_set = torch.zeros(B, SUDOKU_CELLS, SUDOKU_SIZE, dtype=torch.bool)

    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            blank = (grid[:, r, c] == 0)
            b = (r // 3) * 3 + (c // 3)
            allowed = ~(used_row[:, r, 1:SUDOKU_SIZE+1] |
                        used_col[:, c, 1:SUDOKU_SIZE+1] |
                        used_box[:, b, 1:SUDOKU_SIZE+1])  # [B, 9]
            cands = allowed & blank.unsqueeze(1)
            candidate_set[:, idx, :] = cands
            cc = cands.sum(dim=1)
            candidate_count[:, idx] = cc.int()
            is_naked_single[:, idx] = (blank & (cc == 1)).int()

    labels["candidate_count"] = candidate_count
    labels["is_naked_single"] = is_naked_single

    # ── Hidden singles ────────────────────────────────────────────────
    hs_row = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    hs_col = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    hs_box = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)

    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            blank = (grid[:, r, c] == 0)
            if not blank.any():
                continue
            cell_cands = candidate_set[:, idx, :]  # [B,9]
            b = (r // 3) * 3 + (c // 3)

            for d in range(SUDOKU_SIZE):
                has = cell_cands[:, d]
                if not has.any():
                    continue

                # row
                others_row = torch.zeros(B, dtype=torch.bool)
                for cc in range(SUDOKU_SIZE):
                    if cc != c:
                        others_row |= candidate_set[:, r * SUDOKU_SIZE + cc, d]
                hs_row[:, idx] |= (has & ~others_row).int()

                # col
                others_col = torch.zeros(B, dtype=torch.bool)
                for rr in range(SUDOKU_SIZE):
                    if rr != r:
                        others_col |= candidate_set[:, rr * SUDOKU_SIZE + c, d]
                hs_col[:, idx] |= (has & ~others_col).int()

                # box
                br, bc2 = (r // 3) * 3, (c // 3) * 3
                others_box = torch.zeros(B, dtype=torch.bool)
                for dr in range(3):
                    for dc in range(3):
                        rr2, cc2 = br + dr, bc2 + dc
                        if rr2 != r or cc2 != c:
                            others_box |= candidate_set[:, rr2 * SUDOKU_SIZE + cc2, d]
                hs_box[:, idx] |= (has & ~others_box).int()

    labels["is_hidden_single_row"] = hs_row
    labels["is_hidden_single_col"] = hs_col
    labels["is_hidden_single_box"] = hs_box

    # ── Filled-in-unit counts ─────────────────────────────────────────
    filled_in_row = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    filled_in_col = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    filled_in_box = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)

    row_fill = (grid != 0).sum(dim=2)  # [B,9]
    col_fill = (grid != 0).sum(dim=1)  # [B,9]

    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            br, bc2 = (r // 3) * 3, (c // 3) * 3
            bf = (grid[:, br:br+3, bc2:bc2+3] != 0).sum(dim=(1, 2))
            filled_in_row[:, idx] = row_fill[:, r]
            filled_in_col[:, idx] = col_fill[:, c]
            filled_in_box[:, idx] = bf

    labels["filled_in_row"] = filled_in_row
    labels["filled_in_col"] = filled_in_col
    labels["filled_in_box"] = filled_in_box

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Linear probe (copied from train_linear_probes.py — self-contained)
# ═══════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def train_binary(X_tr, y_tr, X_va, y_va, epochs=50, lr=1e-2):
    device = X_tr.device
    model = LinearProbe(X_tr.shape[1], 1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        logits = model(X_tr)
        loss = crit(logits, y_tr.float().view(-1, 1))
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        tr_acc = ((torch.sigmoid(model(X_tr)) > 0.5).long().view(-1) == y_tr.long()).float().mean().item()
        va_acc = ((torch.sigmoid(model(X_va)) > 0.5).long().view(-1) == y_va.long()).float().mean().item()
    return model, tr_acc, va_acc


def train_regression(X_tr, y_tr, X_va, y_va, epochs=80, lr=1e-2):
    device = X_tr.device
    model = LinearProbe(X_tr.shape[1], 1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        pred = model(X_tr)
        loss = crit(pred, y_tr.float().view(-1, 1))
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    def _r2(X, y):
        with torch.no_grad():
            p = model(X).view(-1)
            ss_res = ((y.float() - p) ** 2).sum()
            ss_tot = ((y.float() - y.float().mean()) ** 2).sum().clamp(min=1e-8)
            return (1 - ss_res / ss_tot).item()
    return model, _r2(X_tr, y_tr), _r2(X_va, y_va)


def train_multiclass(X_tr, y_tr, X_va, y_va, n_classes=10, epochs=50, lr=1e-2):
    device = X_tr.device
    model = LinearProbe(X_tr.shape[1], n_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        logits = model(X_tr)
        loss = crit(logits, y_tr.long())
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        tr_acc = (model(X_tr).argmax(1) == y_tr.long()).float().mean().item()
        va_acc = (model(X_va).argmax(1) == y_va.long()).float().mean().item()
    return model, tr_acc, va_acc


# ===========================================================================
# Statistics helpers (seed ensemble + leakage-free split)
# ===========================================================================

def mean_ci(values, confidence: float = 0.95):
    """Mean + t-based CI over a small sample (e.g. seeds).

    Returns ``(mean, ci_lower, ci_upper, std)``. Uses Student-t so the interval
    is honest for the handful of seeds we run.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean, 0.0
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    try:
        from scipy import stats as _sps
        tcrit = float(_sps.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except Exception:
        tcrit = 1.96
    half = tcrit * sem
    return mean, mean - half, mean + half, std


def puzzle_disjoint_split(n_rows: int, val_frac: float, seed: int,
                          cells_per_puzzle: int = SUDOKU_CELLS, device="cpu"):
    """Train/val row indices with a PUZZLE-DISJOINT split.

    Rows are grouped by puzzle in contiguous blocks of ``cells_per_puzzle``.
    Splitting at the puzzle level prevents cells from the same puzzle landing in
    both train and val (which leaks puzzle-specific structure and inflates val
    scores). Falls back to a row-level split when there are < 2 puzzles.
    """
    n_puzzles = n_rows // cells_per_puzzle
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if n_puzzles < 2:
        perm = torch.randperm(n_rows, generator=g).to(device)
        n_val = max(1, int(n_rows * val_frac))
        return perm[n_val:], perm[:n_val]
    puzzle_perm = torch.randperm(n_puzzles, generator=g)
    n_val_p = max(1, int(n_puzzles * val_frac))
    val_p, train_p = puzzle_perm[:n_val_p], puzzle_perm[n_val_p:]

    def _rows(pidx):
        base = (pidx.long() * cells_per_puzzle).view(-1, 1)
        offs = torch.arange(cells_per_puzzle).view(1, -1)
        return (base + offs).reshape(-1).to(device)

    return _rows(train_p), _rows(val_p)


# ═══════════════════════════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data(device: torch.device):
    """Load checkpoint and create test data loader (batch_size=1)."""
    ckpt_dir = os.path.join(REPO_ROOT,
        "checkpoints", "sapientinc-sudoku-extreme")
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

    ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"),
                       map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(k.startswith("_orig_mod.") for k in ck):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(k.startswith("_orig_mod.") for k in mk):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    # Unwrap to inner model
    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, "model"):
        m = m.model

    return m, test_loader, test_meta


def collect_activations(
    model: HierarchicalReasoningModel_ACTV1,
    test_loader,
    device: torch.device,
    n_puzzles: int,
    steps_to_record: List[int],
    max_steps: int = 16,
    puzzle_indices_path: Optional[str] = None,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]], int, List[int]]:
    """Run inference and collect z_H per-cell features + constraint labels.

    Returns
    -------
    features : dict  step_str → list of [N_cells, D] tensors (one per puzzle)
    label_bank : dict  target_name → list of [N_cells] tensors
    hidden_dim : int  the D dimension
    """
    ablator = ActivationAblator(model, device=device)

    def _extract_batch(data):
        if isinstance(data, (tuple, list)):
            return data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        return data

    puzzle_indices = (
        load_puzzle_indices(puzzle_indices_path, limit=n_puzzles)
        if puzzle_indices_path else None
    )
    indexed_batches = collect_indexed_batches(
        test_loader,
        device,
        num_puzzles=n_puzzles,
        puzzle_indices=puzzle_indices,
        extract_batch=_extract_batch,
    )
    collected_indices = [idx for idx, _batch in indexed_batches]
    batches = [batch for _idx, batch in indexed_batches]

    logger.info(f"Collected {len(batches)} puzzles. Running inference...")

    # Structures: per-step features and labels
    # z_H has shape [B=1, T, D].  We extract the last 81 positions (puzzle cells).
    features_H: Dict[str, List[torch.Tensor]] = {str(s): [] for s in steps_to_record}
    features_L: Dict[str, List[torch.Tensor]] = {str(s): [] for s in steps_to_record}
    label_bank: Dict[str, List[torch.Tensor]] = {t: [] for t in ALL_TARGETS}
    hidden_dim = 0

    for pi, batch in enumerate(batches):
        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)

        targets_tok = batch["labels"][:, -SUDOKU_CELLS:]  # [1, 81]
        inputs_tok = batch["inputs"][:, -SUDOKU_CELLS:]

        for step in steps_to_record:
            if step not in cache:
                continue
            ac = cache[step]
            # z_H_out is [1, T, D]; take last 81 positions
            z_H = ac.z_H_out[:, -SUDOKU_CELLS:, :].squeeze(0).float()  # [81, D]
            z_L = ac.z_L_out[:, -SUDOKU_CELLS:, :].squeeze(0).float()
            hidden_dim = z_H.shape[-1]

            features_H[str(step)].append(z_H.cpu())
            features_L[str(step)].append(z_L.cpu())

            # Derive labels from predictions at this step
            preds_tok = ac.preds[:, -SUDOKU_CELLS:]  # [1, 81]
            cell_labels = derive_per_cell_labels(preds_tok.cpu(), targets_tok.cpu(), inputs_tok.cpu())
            for tgt in ALL_TARGETS:
                if tgt in cell_labels:
                    label_bank[tgt].append(cell_labels[tgt].squeeze(0))  # [81]

        if (pi + 1) % 50 == 0:
            logger.info(f"  {pi+1}/{len(batches)} puzzles processed")

    logger.info(f"Done. Hidden dim = {hidden_dim}")
    return features_H, features_L, label_bank, hidden_dim, collected_indices


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E8: Constraint-Specific Linear Probes")
    parser.add_argument("--n_puzzles", type=int, default=500)
    parser.add_argument("--steps", type=str, default="0,4,8,12,15",
                        help="Comma-separated ACT steps to probe")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds for the probe ensemble "
                             "(each uses an independent puzzle-disjoint split)")
    parser.add_argument("--puzzle_indices", type=str, default=None,
                        help="JSON manifest/list of dataloader puzzle indices to evaluate")
    parser.add_argument("--save_puzzle_indices", type=str, default=None,
                        help="Write the collected dataloader puzzle indices to this JSON manifest")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/probes/e8_constraint_probes")
    parser.add_argument("--z_level", type=str, default="H",
                        choices=["H", "L", "both"],
                        help="Which hidden state to probe: H, L, or both")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    steps_to_record = [int(s) for s in args.steps.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    model, test_loader, test_meta = load_model_and_data(device)
    logger.info("Model loaded.")

    # ── Collect ───────────────────────────────────────────────────────
    t0 = time.time()
    features_H, features_L, label_bank, hidden_dim, collected_indices = collect_activations(
        model, test_loader, device, args.n_puzzles, steps_to_record, args.max_steps,
        puzzle_indices_path=args.puzzle_indices,
    )
    if args.save_puzzle_indices:
        save_puzzle_indices(
            args.save_puzzle_indices,
            collected_indices,
            metadata={
                "experiment": "E8_constraint_probes",
                "seed": args.seed,
                "num_puzzles": len(collected_indices),
                "source": args.puzzle_indices or "first_n_test_loader",
            },
        )
    logger.info(f"Collection took {time.time()-t0:.1f}s")

    # ── Build datasets & train probes ─────────────────────────────────
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    seeds = [int(s) for s in str(args.seeds).split(",")] if args.seeds else [args.seed]
    logger.info(f"Seed ensemble: {seeds} (puzzle-disjoint splits, val_frac={args.val_frac})")

    z_levels = ["H", "L"] if args.z_level == "both" else [args.z_level]

    results = []  # list of dicts for CSV
    probe_weights = {}  # target → { W, b, val_score, ... }

    for z_name in z_levels:
        features = features_H if z_name == "H" else features_L

        for step_str in sorted(features.keys(), key=int):
            feat_list = features[step_str]
            if not feat_list:
                continue

            # Stack features: [N_puzzles * 81, D]
            X_all = torch.cat(feat_list, dim=0).to(probe_device)
            n_samples = X_all.shape[0]
            n_steps_per_puzzle = len(steps_to_record)

            # Compute step index within the label bank for alignment
            step_idx_within = sorted(features.keys(), key=int).index(step_str)

            for target in ALL_TARGETS:
                lbl_list = label_bank.get(target, [])
                if not lbl_list:
                    continue

                # label_bank has entries for each (puzzle, step) in sequence
                # Each entry is [81]; total entries = n_puzzles * n_steps
                # We need entries corresponding to this step
                step_labels = lbl_list[step_idx_within::n_steps_per_puzzle]
                if not step_labels:
                    continue
                y_all = torch.cat([l.view(-1) for l in step_labels], dim=0).to(probe_device)

                if y_all.shape[0] != X_all.shape[0]:
                    # Alignment mismatch — try to trim
                    min_n = min(X_all.shape[0], y_all.shape[0])
                    X_use = X_all[:min_n]
                    y_use = y_all[:min_n]
                else:
                    X_use = X_all
                    y_use = y_all

                # Skip targets with zero variance
                if y_use.float().std() < 1e-6:
                    continue

                # Determine task type once
                if target in BINARY_TARGETS:
                    task, metric = "binary", "acc"
                elif target in REGRESSION_TARGETS:
                    task, metric = "regression", "r2"
                elif target in MULTICLASS_TARGETS:
                    task, metric = "multiclass", "acc"
                else:
                    continue

                # ── Seed ensemble with PUZZLE-DISJOINT splits ─────────
                seed_tr_scores: List[float] = []
                seed_va_scores: List[float] = []
                rep_probe = None          # first-seed probe kept for geometry
                rep_ntr = rep_nva = 0
                seed_weight_vecs: List[torch.Tensor] = []  # per-seed [in] for binary geometry
                for s_i, seed in enumerate(seeds):
                    tr_idx, va_idx = puzzle_disjoint_split(
                        X_use.shape[0], args.val_frac, seed,
                        cells_per_puzzle=SUDOKU_CELLS, device=probe_device,
                    )
                    X_tr, y_tr = X_use[tr_idx], y_use[tr_idx]
                    X_va, y_va = X_use[va_idx], y_use[va_idx]

                    torch.manual_seed(seed)
                    if task == "binary":
                        probe, tr_score, va_score = train_binary(
                            X_tr, y_tr, X_va, y_va, epochs=args.epochs, lr=args.lr)
                    elif task == "regression":
                        probe, tr_score, va_score = train_regression(
                            X_tr, y_tr, X_va, y_va, epochs=args.epochs, lr=args.lr)
                    else:
                        probe, tr_score, va_score = train_multiclass(
                            X_tr, y_tr, X_va, y_va, n_classes=10,
                            epochs=args.epochs, lr=args.lr)
                    seed_tr_scores.append(tr_score)
                    seed_va_scores.append(va_score)
                    if task == "binary":
                        seed_weight_vecs.append(probe.linear.weight.detach().cpu()[0])  # [in]
                    if s_i == 0:
                        rep_probe = probe
                        rep_ntr, rep_nva = X_tr.shape[0], X_va.shape[0]

                va_mean, va_lo, va_hi, va_std = mean_ci(seed_va_scores)
                tr_mean = float(np.mean(seed_tr_scores))

                results.append({
                    "z_level": z_name,
                    "step": int(step_str),
                    "target": target,
                    "task": task,
                    "metric": metric,
                    "train_score": round(tr_mean, 5),
                    "val_score": round(va_mean, 5),
                    "val_ci_lower": round(va_lo, 5),
                    "val_ci_upper": round(va_hi, 5),
                    "val_std": round(va_std, 5),
                    "n_seeds": len(seeds),
                    "val_scores_per_seed": ";".join(f"{v:.5f}" for v in seed_va_scores),
                    "n_train": rep_ntr,
                    "n_val": rep_nva,
                    "in_dim": int(X_use.shape[1]),
                })

                # Store representative (first-seed) weights for geometry/ablation
                key = f"z_{z_name}_step{step_str}_{target}"
                W = rep_probe.linear.weight.detach().cpu()   # [out, in]
                b = rep_probe.linear.bias.detach().cpu()      # [out]
                probe_weights[key] = {
                    "W": W, "b": b,
                    "task": task, "metric": metric,
                    "val_score": va_mean,
                    "val_ci_lower": va_lo,
                    "val_ci_upper": va_hi,
                    "val_std": va_std,
                    "val_scores_per_seed": seed_va_scores,
                    "n_seeds": len(seeds),
                    "in_dim": int(X_use.shape[1]),
                    "z_level": z_name,
                    "step": int(step_str),
                    "target": target,
                    # per-seed unit weight vectors for ensemble geometry (binary only)
                    "W_per_seed": [w for w in seed_weight_vecs] if seed_weight_vecs else None,
                }

                logger.info(
                    f"  z_{z_name} step {step_str:>2s} {target:28s} "
                    f"{metric}={va_mean:.4f} ±{(va_hi - va_mean):.4f} "
                    f"(train={tr_mean:.4f}, {len(seeds)} seeds)"
                )

    # ── Save results CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    logger.info(f"Saved {len(results)} rows to {csv_path}")

    # ── Save seed-ensemble summary (mean ± 95% CI per probe) ──────────
    summary = {
        f"z_{r['z_level']}_step{r['step']}_{r['target']}": {
            "z_level": r["z_level"], "step": r["step"], "target": r["target"],
            "task": r["task"], "metric": r["metric"],
            "val_mean": r["val_score"], "val_ci_lower": r["val_ci_lower"],
            "val_ci_upper": r["val_ci_upper"], "val_std": r["val_std"],
            "val_per_seed": [float(x) for x in r["val_scores_per_seed"].split(";")],
            "train_mean": r["train_score"], "n_seeds": r["n_seeds"],
            "n_train": r["n_train"], "n_val": r["n_val"],
        }
        for r in results
    }
    with open(os.path.join(args.output_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved probe_summary.json (seed-ensemble mean ± 95% CI)")

    # ── Save probe weights ────────────────────────────────────────────
    torch.save(probe_weights, os.path.join(args.output_dir, "probe_weights.pt"))
    logger.info(f"Saved probe weights ({len(probe_weights)} probes)")

    # ── Provenance ────────────────────────────────────────────────────
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "E8_constraint_probes", {
            "n_puzzles": args.n_puzzles, "steps": args.steps,
            "max_steps": args.max_steps, "epochs": args.epochs, "lr": args.lr,
            "val_frac": args.val_frac, "seeds": seeds, "z_level": args.z_level,
            "split": "puzzle_disjoint",
            "puzzle_indices": args.puzzle_indices,
            "save_puzzle_indices": args.save_puzzle_indices,
            "num_puzzles_collected": len(collected_indices),
        }, repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"Could not write _meta.json: {e}")

    # ── Geometric analysis ────────────────────────────────────────────
    logger.info("Running geometric analysis...")
    geometric = geometric_analysis(probe_weights, args.output_dir)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("E8 CONSTRAINT PROBES — SUMMARY")
    print("=" * 80)
    print(f"{'z':>3s}  {'step':>4s}  {'target':28s}  {'metric':>6s}  {'val':>8s}  {'train':>8s}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (-x["val_score"], x["target"])):
        print(f"{r['z_level']:>3s}  {r['step']:>4d}  {r['target']:28s}  "
              f"{r['metric']:>6s}  {r['val_score']:>8.4f}  {r['train_score']:>8.4f}")

    if geometric.get("constraint_cosines"):
        print("\n── Constraint Direction Cosines (binary probes, same step) ──")
        for entry in geometric["constraint_cosines"]:
            print(f"  cos({entry['probe_a']}, {entry['probe_b']}) = {entry['cosine']:.4f}")

    logger.info(f"All results in {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Geometric analysis of probe weight vectors
# ═══════════════════════════════════════════════════════════════════════════

def geometric_analysis(probe_weights: dict, output_dir: str) -> dict:
    """Analyze pairwise cosine similarity of probe weight vectors and PCA overlap."""
    result: Dict[str, Any] = {}

    # ── Pairwise cosines between constraint targets at the same step ──
    constraint_targets = [
        "violated_in_row", "violated_in_col", "violated_in_box",
        "is_naked_single", "is_hidden_single_row", "is_hidden_single_col",
        "is_hidden_single_box", "per_cell_correct",
    ]
    cosine_entries = []

    # Group by (z_level, step)
    groups: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, val in probe_weights.items():
        tgt = val["target"]
        if tgt in constraint_targets and val["task"] == "binary":
            group_key = f"z_{val['z_level']}_step{val['step']}"
            if group_key not in groups:
                groups[group_key] = {}
            # For binary probe, W is [1, D]. Use the first (only) row.
            groups[group_key][tgt] = val["W"][0]

    for gk, vecs in groups.items():
        targets_sorted = sorted(vecs.keys())
        for i in range(len(targets_sorted)):
            for j in range(i + 1, len(targets_sorted)):
                a, b = targets_sorted[i], targets_sorted[j]
                cos = F.cosine_similarity(vecs[a].unsqueeze(0), vecs[b].unsqueeze(0)).item()
                cosine_entries.append({
                    "group": gk,
                    "probe_a": a,
                    "probe_b": b,
                    "cosine": round(cos, 5),
                })

    result["constraint_cosines"] = cosine_entries

    # ── Seed-ensemble cosines & PCA (mean ± 95% CI across seeds) ──────
    # Uses per-seed weight vectors when available so the reported geometry is
    # robust to probe-training randomness rather than a single-seed snapshot.
    seed_groups: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for key, val in probe_weights.items():
        tgt = val["target"]
        if (tgt in constraint_targets and val["task"] == "binary"
                and val.get("W_per_seed")):
            gk = f"z_{val['z_level']}_step{val['step']}"
            seed_groups.setdefault(gk, {})[tgt] = val["W_per_seed"]  # list of [D]

    ensemble_cosines = []
    ensemble_pca = []
    for gk, vecs in seed_groups.items():
        n_seeds = min(len(v) for v in vecs.values()) if vecs else 0
        if n_seeds < 1:
            continue
        targets_sorted = sorted(vecs.keys())
        # pairwise cosine per seed
        for i in range(len(targets_sorted)):
            for j in range(i + 1, len(targets_sorted)):
                a, b = targets_sorted[i], targets_sorted[j]
                per_seed = [
                    F.cosine_similarity(vecs[a][s].unsqueeze(0),
                                        vecs[b][s].unsqueeze(0)).item()
                    for s in range(n_seeds)
                ]
                m, lo, hi, sd = mean_ci(per_seed)
                ensemble_cosines.append({
                    "group": gk, "probe_a": a, "probe_b": b,
                    "cosine_mean": round(m, 5),
                    "cosine_ci_lower": round(lo, 5),
                    "cosine_ci_upper": round(hi, 5),
                    "cosine_std": round(sd, 5),
                    "n_seeds": n_seeds,
                    "cosine_per_seed": [round(x, 5) for x in per_seed],
                })
        # PCA PC1 explained-variance per seed
        if len(targets_sorted) >= 2:
            pc1_per_seed = []
            for s in range(n_seeds):
                mat = torch.stack([vecs[t][s] for t in targets_sorted], dim=0)
                mat_c = mat - mat.mean(dim=0, keepdim=True)
                try:
                    S = torch.linalg.svdvals(mat_c)
                    ev = (S ** 2) / (S ** 2).sum()
                    pc1_per_seed.append(float(ev[0].item()))
                except Exception:
                    pass
            if pc1_per_seed:
                m, lo, hi, sd = mean_ci(pc1_per_seed)
                ensemble_pca.append({
                    "group": gk, "n_probes": len(targets_sorted),
                    "pc1_explained_mean": round(m, 5),
                    "pc1_explained_ci_lower": round(lo, 5),
                    "pc1_explained_ci_upper": round(hi, 5),
                    "pc1_explained_std": round(sd, 5),
                    "n_seeds": len(pc1_per_seed),
                    "pc1_per_seed": [round(x, 5) for x in pc1_per_seed],
                })
    result["constraint_cosines_ensemble"] = ensemble_cosines
    result["pca_pc1_ensemble"] = ensemble_pca

    # ── PCA of constraint weight vectors at each step ──
    pca_entries = []
    for gk, vecs in groups.items():
        if len(vecs) < 2:
            continue
        mat = torch.stack([vecs[t] for t in sorted(vecs.keys())], dim=0)  # [K, D]
        mat_centered = mat - mat.mean(dim=0, keepdim=True)
        try:
            U, S, _ = torch.svd(mat_centered)
            explained = (S ** 2) / (S ** 2).sum()
            pca_entries.append({
                "group": gk,
                "n_probes": mat.shape[0],
                "singular_values": S.tolist()[:5],
                "explained_variance_ratio": explained.tolist()[:5],
            })
        except Exception:
            pass

    result["pca_of_weight_vectors"] = pca_entries

    # ── Norm of each probe weight vector ──
    norm_entries = []
    for key, val in probe_weights.items():
        W = val["W"]  # [out, D]
        norm_entries.append({
            "key": key,
            "target": val["target"],
            "z_level": val["z_level"],
            "step": val["step"],
            "W_norm": round(float(W.norm().item()), 5),
            "val_score": round(val["val_score"], 5),
        })
    result["weight_norms"] = norm_entries

    # Save
    # Convert tensors to lists for JSON serialization
    json_path = os.path.join(output_dir, "geometric_analysis.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    logger.info(f"Geometric analysis saved to {json_path}")

    return result


if __name__ == "__main__":
    main()
