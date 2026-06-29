"""Shared helpers for ARC-AGI experiments.

Mirrors ``scripts/maze/maze_common.py`` (Maze) and
``scripts/controlled/controlled_common.py`` (Sudoku) but defaults the checkpoint
to the HRM ARC-2 model and exposes ARC-specific prediction metrics defined on
the 30x30 colour-grid encoding (see ``utils/arc_targets.py`` /
``dataset/build_arc_dataset.py``).

Token encoding: PAD=0, EOS=1, colour c in 0..9 -> token c+2. Labels are remapped
by the dataloader so the PAD region is the loss-ignore id -100; ``valid =
label != -100`` therefore selects the grid + EOS markers of the target.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from utils.arc_targets import (
    SEQ_LEN, GRID_SIZE, PAD_ID, EOS_ID, COLOR_OFFSET, NUM_COLORS,
    IGNORE_LABEL_ID, grid_extent, grid_mask, token_to_colour,
    num_distinct_colours,
)

# The ARC-2 checkpoint downloaded from sapientinc/HRM-checkpoint-ARC-2.
# Note the file is named "checkpoint" (no .pt), with all_config.yaml alongside.
ARC_CHECKPOINT = os.path.join(
    REPO_ROOT, "checkpoints", "sapientinc-hrm-arc-2", "checkpoint",
)

ARC_METRIC_KEYS = [
    "token_acc",          # accuracy over all valid (non-ignored) label cells
    "exact_solved",       # all valid cells correct (ARC's exact-match objective)
    "colour_cell_acc",    # accuracy where the label is a colour (token >= 2)
    "eos_acc",            # accuracy where the label is the EOS marker
    "background_acc",     # accuracy where the label colour == 0 (black)
    "shape_correct",      # predicted grid bounding box matches the label's
    "height_correct",     # predicted grid height matches the label's
    "width_correct",      # predicted grid width matches the label's
    "num_colours_correct",  # distinct-colour count matches the label
    "colour_iou",         # mean IoU of per-colour cell sets (pred vs label)
]


def get_puzzle_emb_len(model) -> int:
    inner = getattr(model, "inner", model)
    return int(getattr(inner, "puzzle_emb_len", 0))


# ---------------------------------------------------------------------------
# Flattening helpers (puzzle-emb prefix aware)
# ---------------------------------------------------------------------------

def _flat_int(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.ndim == 2:
            x = x[0]
        return x.detach().to("cpu").to(torch.int64).numpy().reshape(-1)
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.int64, copy=False).reshape(-1)


def _align_pred(pred: np.ndarray, label_len: int) -> np.ndarray:
    """Drop the puzzle-emb prefix from a flat prediction if present."""
    pred = pred.reshape(-1)
    if pred.size > label_len:
        return pred[-label_len:]
    return pred


def cell_positions(puzzle_emb_len: int) -> np.ndarray:
    """Absolute z_H/z_L positions of the 900 grid cells (after the emb prefix)."""
    return np.arange(SEQ_LEN, dtype=np.int64) + puzzle_emb_len


# ---------------------------------------------------------------------------
# ARC prediction metrics
# ---------------------------------------------------------------------------

def _colour_iou(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> float:
    """Mean intersection-over-union of per-colour cell sets over present colours."""
    pcol = token_to_colour(pred)
    lcol = token_to_colour(label)
    colours = np.unique(lcol[(lcol >= 0) & valid])
    if colours.size == 0:
        return 0.0
    ious = []
    for c in colours:
        p = (pcol == c) & valid
        l = (lcol == c) & valid
        union = int((p | l).sum())
        if union == 0:
            continue
        ious.append(int((p & l).sum()) / union)
    return float(np.mean(ious)) if ious else 0.0


def arc_prediction_metrics(pred_flat, label_flat, input_flat=None) -> Dict[str, float]:
    """ARC-specific structural metrics for one 30x30 prediction.

    ``input_flat`` is accepted for signature parity with the maze/sudoku metrics
    (ARC metrics are computed purely from prediction vs label).
    """
    del input_flat
    pred = _flat_int(pred_flat)
    label = _flat_int(label_flat)
    pred = _align_pred(pred, label.size)
    if pred.size != SEQ_LEN or label.size != SEQ_LEN:
        raise ValueError(
            f"ARC metrics expect {SEQ_LEN} cells, got pred={pred.size} label={label.size}")

    valid = label != IGNORE_LABEL_ID
    if not valid.any():
        return {k: 0.0 for k in ARC_METRIC_KEYS}

    total = int(valid.sum())
    correct = (pred == label) & valid

    colour_lab = valid & (label >= COLOR_OFFSET)
    eos_lab = valid & (label == EOS_ID)
    bg_lab = valid & (label == COLOR_OFFSET)  # colour 0 == token 2

    def _acc(mask):
        d = int(mask.sum())
        return float((correct & mask).sum() / d) if d else 0.0

    # Geometry: bounding box / distinct colours from the colour tokens.
    # (Labels use -100 for PAD; grid_extent looks for tokens >= COLOR_OFFSET so
    # the ignore id is naturally excluded.)
    _, _, lh, lw = grid_extent(label)
    _, _, ph, pw = grid_extent(pred)
    n_lab_col = num_distinct_colours(label)
    n_pred_col = num_distinct_colours(pred)

    return {
        "token_acc": float(correct.sum() / total),
        "exact_solved": float(int(correct.sum()) == total),
        "colour_cell_acc": _acc(colour_lab),
        "eos_acc": _acc(eos_lab),
        "background_acc": _acc(bg_lab),
        "shape_correct": float((ph == lh) and (pw == lw)),
        "height_correct": float(ph == lh),
        "width_correct": float(pw == lw),
        "num_colours_correct": float(n_pred_col == n_lab_col),
        "colour_iou": _colour_iou(pred, label, valid),
    }


def arc_batch_metrics(preds, labels, inputs=None) -> Dict[str, float]:
    """Mean ARC metrics over a batch ([B, 900] or [900])."""
    p = preds.detach().to("cpu") if isinstance(preds, torch.Tensor) else np.asarray(preds)
    y = labels.detach().to("cpu") if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    rows = [arc_prediction_metrics(p[i], y[i]) for i in range(p.shape[0])]
    return {k: float(np.mean([r[k] for r in rows])) for k in ARC_METRIC_KEYS}


def maybe_arc_batch_metrics(preds, labels, inputs=None) -> Optional[Dict[str, float]]:
    """Return ARC metrics only for 30x30-shaped tensors (else None)."""
    seq_len = labels.shape[-1] if hasattr(labels, "shape") and len(labels.shape) else None
    if seq_len != SEQ_LEN:
        return None
    return arc_batch_metrics(preds, labels)


# ---------------------------------------------------------------------------
# Per-puzzle global feature dict (for the global linear/MLP probes)
# ---------------------------------------------------------------------------

def arc_global_features(pred_flat, label_flat, input_flat) -> Dict[str, float]:
    """Per-puzzle scalar targets decoded from the mean-pooled answer state.

    Combines the structural metrics above with input/output grid geometry so the
    global probes can ask "does z encode the grid's shape / colour count / whether
    it solved the puzzle".
    """
    metrics = arc_prediction_metrics(pred_flat, label_flat)
    inp = _flat_int(input_flat)
    label = _flat_int(label_flat)
    inp = _align_pred(inp, SEQ_LEN) if inp.size != SEQ_LEN else inp
    _, _, ih, iw = grid_extent(inp)
    _, _, oh, ow = grid_extent(label)
    return {
        "exact_solved": metrics["exact_solved"],
        "shape_correct": metrics["shape_correct"],
        "token_acc": metrics["token_acc"],
        "colour_cell_acc": metrics["colour_cell_acc"],
        "colour_iou": metrics["colour_iou"],
        "input_height": float(ih),
        "input_width": float(iw),
        "output_height": float(oh),
        "output_width": float(ow),
        "num_input_colours": float(num_distinct_colours(inp)),
        "num_output_colours": float(num_distinct_colours(label)),
        "size_preserved": float((ih == oh) and (iw == ow)),
    }
