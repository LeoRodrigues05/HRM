"""Shared helpers for maze experiments.

Mirrors `scripts/controlled/controlled_common.py` but defaults the checkpoint
to the maze-30x30 model and exposes spatial-mask utilities defined on the
30x30 maze grid (puzzle-emb prefix aware).
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.maze_targets import (
    GRID_SIZE, SEQ_LEN, PAD_ID, WALL_ID, FREE_ID, START_ID, GOAL_ID, PATH_ID,
    distance_to_goal, distance_to_start, on_optimal_path, is_wall,
)

MAZE_CHECKPOINT = os.path.join(
    REPO_ROOT, "checkpoints", "sapientinc-hrm-maze-30x30-hard", "checkpoint",
)

MAZE_PATH_TOKENS = (PATH_ID, START_ID, GOAL_ID)
MAZE_METRIC_KEYS = [
    "token_acc",
    "exact_solved",
    "path_cell_acc",
    "path_precision",
    "path_recall",
    "path_f1",
    "path_jaccard",
    "pred_path_length",
    "path_length_ratio",
    "wall_path_rate",
    "connects_start_goal",
    "no_branch",
    "valid_sg_path",
    "valid_optimal_path",
]

MAZE_LAYOUT_TOKENS = (WALL_ID, START_ID, GOAL_ID)
MAZE_LAYOUT_METRIC_KEYS = [
    "target_layout_acc",
    "target_wall_acc",
    "target_endpoint_acc",
    "target_layout_error_rate",
    "source_layout_acc",
    "source_unique_wall_imprint_rate",
    "target_unique_wall_erasure_rate",
    "source_endpoint_imprint_rate",
    "target_endpoint_erasure_rate",
    "toward_source_layout_rate",
    "toward_target_layout_rate",
]


def get_puzzle_emb_len(model) -> int:
    inner = getattr(model, "inner", model)
    return int(getattr(inner, "puzzle_emb_len", 0))


def _flat_label(label: torch.Tensor) -> np.ndarray:
    if label.ndim == 2:
        label = label[0]
    return label.detach().to("cpu").to(torch.uint8).numpy()


def _flat_input(inp: torch.Tensor) -> np.ndarray:
    if inp.ndim == 2:
        inp = inp[0]
    return inp.detach().to("cpu").to(torch.uint8).numpy()


def _flat_int(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.ndim == 2:
            x = x[0]
        return x.detach().to("cpu").to(torch.int64).numpy().reshape(-1)
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.int64, copy=False).reshape(-1)


def cell_positions(puzzle_emb_len: int) -> np.ndarray:
    """Return absolute z_H/z_L positions of the 900 maze cells."""
    return np.arange(SEQ_LEN, dtype=np.int64) + puzzle_emb_len


def mask_on_path(label_flat: np.ndarray, puzzle_emb_len: int) -> List[int]:
    g = label_flat.reshape(GRID_SIZE, GRID_SIZE)
    mask = on_optimal_path(g).reshape(-1)
    cells = np.nonzero(mask)[0]
    return (cells + puzzle_emb_len).tolist()


def mask_off_path(label_flat: np.ndarray, input_flat: np.ndarray, puzzle_emb_len: int) -> List[int]:
    """Passable (non-wall) cells that are NOT on the optimal path."""
    lg = label_flat.reshape(GRID_SIZE, GRID_SIZE)
    ig = input_flat.reshape(GRID_SIZE, GRID_SIZE)
    on = on_optimal_path(lg)
    passable = (ig != WALL_ID) & (ig != PAD_ID)
    mask = passable & (~on)
    cells = np.nonzero(mask.reshape(-1))[0]
    return (cells + puzzle_emb_len).tolist()


def mask_near_token(input_flat: np.ndarray, token_id: int, max_dist: int, puzzle_emb_len: int) -> List[int]:
    """All passable cells within BFS distance <= max_dist of the given token (S or G)."""
    g = input_flat.reshape(GRID_SIZE, GRID_SIZE)
    if token_id == GOAL_ID:
        dist = distance_to_goal(g)
    elif token_id == START_ID:
        dist = distance_to_start(g)
    else:
        raise ValueError(token_id)
    mask = (dist <= max_dist) & (dist >= 0)
    cells = np.nonzero(mask.reshape(-1))[0]
    return (cells + puzzle_emb_len).tolist()


def build_spatial_masks(batch: Dict[str, torch.Tensor], puzzle_emb_len: int,
                        near_dist: int = 5) -> Dict[str, List[int]]:
    """Return dict of spatial-group name -> list of absolute z positions to patch."""
    lab = _flat_label(batch["labels"])
    inp = _flat_input(batch["inputs"])
    return {
        "on_path":  mask_on_path(lab, puzzle_emb_len),
        "off_path": mask_off_path(lab, inp, puzzle_emb_len),
        "near_S":   mask_near_token(inp, START_ID, near_dist, puzzle_emb_len),
        "near_G":   mask_near_token(inp, GOAL_ID,  near_dist, puzzle_emb_len),
    }


def _neighbors(pos: int):
    r, c = divmod(pos, GRID_SIZE)
    if r > 0:
        yield pos - GRID_SIZE
    if r + 1 < GRID_SIZE:
        yield pos + GRID_SIZE
    if c > 0:
        yield pos - 1
    if c + 1 < GRID_SIZE:
        yield pos + 1


def _component_stats(path_mask: np.ndarray) -> Dict[str, int]:
    """Connected-component and degree stats for a flat boolean path mask."""
    path_mask = path_mask.astype(bool, copy=False).reshape(-1)
    path_indices = np.nonzero(path_mask)[0]
    if path_indices.size == 0:
        return {"components": 0, "branch_cells": 0, "dead_end_cells": 0}

    seen = np.zeros(path_mask.size, dtype=bool)
    components = 0
    for start in path_indices:
        if seen[start]:
            continue
        components += 1
        queue = deque([int(start)])
        seen[start] = True
        while queue:
            cur = queue.popleft()
            for nxt in _neighbors(cur):
                if path_mask[nxt] and not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)

    degrees = []
    for pos in path_indices:
        degrees.append(sum(1 for nxt in _neighbors(int(pos)) if path_mask[nxt]))

    return {
        "components": components,
        "branch_cells": int(sum(d > 2 for d in degrees)),
        "dead_end_cells": int(sum(d == 1 for d in degrees)),
    }


def _connected(path_mask: np.ndarray, start_pos: int, goal_pos: int) -> bool:
    path_mask = path_mask.astype(bool, copy=False).reshape(-1)
    if start_pos < 0 or goal_pos < 0:
        return False
    if not path_mask[start_pos] or not path_mask[goal_pos]:
        return False

    seen = np.zeros(path_mask.size, dtype=bool)
    queue = deque([int(start_pos)])
    seen[start_pos] = True
    while queue:
        cur = queue.popleft()
        if cur == goal_pos:
            return True
        for nxt in _neighbors(cur):
            if path_mask[nxt] and not seen[nxt]:
                seen[nxt] = True
                queue.append(nxt)
    return False


def maze_prediction_metrics(pred_flat, label_flat, input_flat=None) -> Dict[str, float]:
    """Maze-specific structural metrics for one 30x30 prediction.

    Token accuracy is dominated by wall/free cells. These metrics also score
    whether the predicted path covers the true optimal path, stays off walls,
    connects S to G, avoids branch points, and has the optimal path length.
    """
    pred = _flat_int(pred_flat)
    label = _flat_int(label_flat)
    if pred.size > label.size:
        pred = pred[-label.size:]
    if pred.size != SEQ_LEN or label.size != SEQ_LEN:
        raise ValueError(f"maze metrics expect {SEQ_LEN} cells, got pred={pred.size} label={label.size}")

    valid = label != -100
    if not valid.any():
        return {k: 0.0 for k in MAZE_METRIC_KEYS}

    total = int(valid.sum())
    correct = (pred == label) & valid
    true_path = valid & np.isin(label, MAZE_PATH_TOKENS)
    pred_path = valid & np.isin(pred, MAZE_PATH_TOKENS)

    tp = int((true_path & pred_path).sum())
    pred_count = int(pred_path.sum())
    true_count = int(true_path.sum())
    union_count = int((true_path | pred_path).sum())

    precision = tp / pred_count if pred_count else 0.0
    recall = tp / true_count if true_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    path_jaccard = tp / union_count if union_count else 0.0

    path_cell_acc = float((correct & true_path).sum() / true_count) if true_count else 0.0
    exact_solved = float(int(correct.sum()) == total)

    wall_path_count = 0
    connects_start_goal = 0.0
    no_branch = 0.0
    valid_sg_path = 0.0
    if input_flat is not None:
        inp = _flat_int(input_flat)
        if inp.size > SEQ_LEN:
            inp = inp[-SEQ_LEN:]
        if inp.size != SEQ_LEN:
            raise ValueError(f"maze input must have {SEQ_LEN} cells, got {inp.size}")
        wall_mask = (inp == WALL_ID) | (inp == PAD_ID)
        wall_path_count = int((pred_path & wall_mask).sum())
        passable_path = pred_path & (~wall_mask)
        start_positions = np.nonzero(inp == START_ID)[0]
        goal_positions = np.nonzero(inp == GOAL_ID)[0]
        start_pos = int(start_positions[0]) if start_positions.size else -1
        goal_pos = int(goal_positions[0]) if goal_positions.size else -1

        stats = _component_stats(passable_path)
        connects_start_goal = float(_connected(passable_path, start_pos, goal_pos))
        no_branch = float(stats["branch_cells"] == 0)
        valid_sg_path = float(bool(connects_start_goal) and wall_path_count == 0 and stats["branch_cells"] == 0)

    path_length_ratio = pred_count / true_count if true_count else 0.0
    valid_optimal_path = float(bool(valid_sg_path) and pred_count == true_count)
    wall_path_rate = wall_path_count / pred_count if pred_count else 0.0

    return {
        "token_acc": float(correct.sum() / total),
        "exact_solved": exact_solved,
        "path_cell_acc": path_cell_acc,
        "path_precision": float(precision),
        "path_recall": float(recall),
        "path_f1": float(f1),
        "path_jaccard": float(path_jaccard),
        "pred_path_length": float(pred_count),
        "path_length_ratio": float(path_length_ratio),
        "wall_path_rate": float(wall_path_rate),
        "connects_start_goal": float(connects_start_goal),
        "no_branch": float(no_branch),
        "valid_sg_path": float(valid_sg_path),
        "valid_optimal_path": float(valid_optimal_path),
    }


def _rate(mask: np.ndarray, hit: np.ndarray) -> float:
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    return float((mask & hit).sum() / denom)


def maze_layout_metrics(pred_flat, target_input_flat, source_input_flat=None) -> Dict[str, float]:
    """Measure whether a prediction preserves the target maze layout.

    For Maze, the fixed puzzle layout is the wall/start/goal pattern. The
    model's output should copy those tokens from the target puzzle while only
    deciding which passable cells are on the path. When source activations are
    patched into a target run, the source-specific rates quantify whether the
    output starts matching source walls/endpoints at absolute coordinates.
    """
    pred = _flat_int(pred_flat)
    target = _flat_int(target_input_flat)
    if pred.size > SEQ_LEN:
        pred = pred[-SEQ_LEN:]
    if target.size > SEQ_LEN:
        target = target[-SEQ_LEN:]
    if pred.size != SEQ_LEN or target.size != SEQ_LEN:
        raise ValueError(f"maze layout metrics expect {SEQ_LEN} cells, got pred={pred.size} target={target.size}")

    target_layout = np.isin(target, MAZE_LAYOUT_TOKENS)
    target_walls = target == WALL_ID
    target_endpoints = (target == START_ID) | (target == GOAL_ID)

    out = {
        "target_layout_acc": _rate(target_layout, pred == target),
        "target_wall_acc": _rate(target_walls, pred == WALL_ID),
        "target_endpoint_acc": _rate(target_endpoints, pred == target),
        "target_layout_error_rate": _rate(target_layout, pred != target),
        "source_layout_acc": 0.0,
        "source_unique_wall_imprint_rate": 0.0,
        "target_unique_wall_erasure_rate": 0.0,
        "source_endpoint_imprint_rate": 0.0,
        "target_endpoint_erasure_rate": 0.0,
        "toward_source_layout_rate": 0.0,
        "toward_target_layout_rate": 0.0,
    }

    if source_input_flat is None:
        return out

    source = _flat_int(source_input_flat)
    if source.size > SEQ_LEN:
        source = source[-SEQ_LEN:]
    if source.size != SEQ_LEN:
        raise ValueError(f"maze source input must have {SEQ_LEN} cells, got {source.size}")

    source_layout = np.isin(source, MAZE_LAYOUT_TOKENS)
    source_unique_walls = (source == WALL_ID) & (target != WALL_ID)
    target_unique_walls = (target == WALL_ID) & (source != WALL_ID)
    source_unique_endpoints = ((source == START_ID) | (source == GOAL_ID)) & (source != target)
    target_unique_endpoints = ((target == START_ID) | (target == GOAL_ID)) & (source != target)
    layout_disagreement = (source != target) & (source_layout | target_layout)

    out.update({
        "source_layout_acc": _rate(source_layout, pred == source),
        "source_unique_wall_imprint_rate": _rate(source_unique_walls, pred == WALL_ID),
        "target_unique_wall_erasure_rate": _rate(target_unique_walls, pred != WALL_ID),
        "source_endpoint_imprint_rate": _rate(source_unique_endpoints, pred == source),
        "target_endpoint_erasure_rate": _rate(target_unique_endpoints, pred != target),
        "toward_source_layout_rate": _rate(layout_disagreement, pred == source),
        "toward_target_layout_rate": _rate(layout_disagreement, pred == target),
    })
    return out


def maze_batch_metrics(preds, labels, inputs=None) -> Dict[str, float]:
    """Mean maze metrics over a batch."""
    p = preds.detach().to("cpu") if isinstance(preds, torch.Tensor) else np.asarray(preds)
    y = labels.detach().to("cpu") if isinstance(labels, torch.Tensor) else np.asarray(labels)
    x = inputs.detach().to("cpu") if isinstance(inputs, torch.Tensor) else None if inputs is None else np.asarray(inputs)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if x is not None and x.ndim == 1:
        x = x.reshape(1, -1)

    rows = []
    for i in range(p.shape[0]):
        inp_i = None if x is None else x[i]
        rows.append(maze_prediction_metrics(p[i], y[i], inp_i))
    return {k: float(np.mean([row[k] for row in rows])) for k in MAZE_METRIC_KEYS}


def maybe_maze_batch_metrics(preds, labels, inputs=None) -> Optional[Dict[str, float]]:
    """Return maze metrics only for 30x30 maze-shaped tensors."""
    seq_len = labels.shape[-1] if hasattr(labels, "shape") and len(labels.shape) else None
    if seq_len != SEQ_LEN:
        return None
    if inputs is None:
        return None
    return maze_batch_metrics(preds, labels, inputs)


def grid_predictions_accuracy(pred_flat: np.ndarray, label_flat: np.ndarray) -> Dict[str, float]:
    """Token-level + path-cell-only accuracy."""
    metrics = maze_prediction_metrics(pred_flat, label_flat)
    return {k: metrics[k] for k in ("token_acc", "path_cell_acc", "exact_solved")}
