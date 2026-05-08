"""Maze target derivation utilities.

Companion to ``utils/probes.py`` for the Maze 30x30-hard task. Provides
BFS-based per-cell and per-puzzle target features that can be used as
linear-probe regression/classification targets, as ablation masks, or as
intervention targets.

Token encoding (matches ``dataset/build_maze_dataset.py``):

    PAD_ID  = 0   # padding
    WALL_ID = 1   # '#'
    FREE_ID = 2   # ' '
    START_ID= 3   # 'S'
    GOAL_ID = 4   # 'G'
    PATH_ID = 5   # 'o' (only present in labels along the optimal path)

Grid is 30x30 (seq_len=900). Inputs contain {WALL, FREE, START, GOAL};
labels additionally mark the optimal path with PATH_ID.

Per-cell features (numpy arrays of shape ``(H, W)``):
    - on_optimal_path(grid_label)  -> bool
    - distance_to_goal(grid_input) -> int32  (UNREACHABLE for walls / unreachable)
    - distance_to_start(grid_input) -> int32
    - is_wall(grid_input)          -> bool
    - is_junction(grid_input)      -> bool
    - is_dead_end(grid_input)      -> bool

Per-puzzle features (scalars):
    - path_length(grid_label)      -> int (count of PATH_ID tokens, 0 if no path)
    - num_dead_ends(grid_input)    -> int
    - frac_solved(grid_pred, grid_label) -> float in [0, 1]

All accessors accept either a flat sequence of length 900 or a 30x30 grid
(numpy array, torch tensor, or bytes). Results are cached per puzzle via
an LRU keyed on the input grid's byte representation, so it is cheap to
call multiple feature functions on the same maze.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Tuple, Union

import numpy as np

try:  # torch is optional at import time
    import torch
    _TorchTensor = torch.Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TorchTensor = ()  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_ID = 0
WALL_ID = 1
FREE_ID = 2
START_ID = 3
GOAL_ID = 4
PATH_ID = 5

GRID_SIZE = 30
SEQ_LEN = GRID_SIZE * GRID_SIZE  # 900

UNREACHABLE = np.iinfo(np.int32).max  # sentinel for BFS distance to walls / unreachable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

GridLike = Union[np.ndarray, "_TorchTensor", bytes, bytearray, memoryview]


def _to_grid(grid: GridLike) -> np.ndarray:
    """Coerce any supported input to a contiguous uint8 grid of shape (30, 30)."""
    if torch is not None and isinstance(grid, torch.Tensor):
        arr = grid.detach().cpu().numpy()
    elif isinstance(grid, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(grid), dtype=np.uint8)
    else:
        arr = np.asarray(grid)

    if arr.ndim == 1:
        if arr.size != SEQ_LEN:
            raise ValueError(f"flat grid must have length {SEQ_LEN}, got {arr.size}")
        arr = arr.reshape(GRID_SIZE, GRID_SIZE)
    elif arr.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"grid must be {GRID_SIZE}x{GRID_SIZE}, got {arr.shape}")

    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _grid_key(grid: np.ndarray) -> bytes:
    """Hashable key for LRU caches."""
    return grid.tobytes()


def _passable(grid: np.ndarray) -> np.ndarray:
    """Boolean mask of cells the agent can stand on (anything that isn't a wall or pad)."""
    return (grid != WALL_ID) & (grid != PAD_ID)


def _find_unique(grid: np.ndarray, token_id: int) -> Tuple[int, int] | None:
    """Return (row, col) of the unique occurrence of ``token_id``, or None."""
    where = np.argwhere(grid == token_id)
    if where.size == 0:
        return None
    r, c = int(where[0, 0]), int(where[0, 1])
    return r, c


# ---------------------------------------------------------------------------
# BFS (cached)
# ---------------------------------------------------------------------------


def _bfs_from(grid_bytes: bytes, src_row: int, src_col: int) -> np.ndarray:
    """4-connected BFS over passable cells. Returns distance grid (int32)."""
    grid = np.frombuffer(grid_bytes, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
    dist = np.full((GRID_SIZE, GRID_SIZE), UNREACHABLE, dtype=np.int32)
    passable = _passable(grid)

    if not passable[src_row, src_col]:
        return dist

    dist[src_row, src_col] = 0
    q: deque[Tuple[int, int]] = deque()
    q.append((src_row, src_col))
    while q:
        r, c = q.popleft()
        d_next = dist[r, c] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and passable[nr, nc] and dist[nr, nc] > d_next:
                dist[nr, nc] = d_next
                q.append((nr, nc))
    return dist


@lru_cache(maxsize=4096)
def _bfs_from_token_cached(grid_bytes: bytes, token_id: int) -> np.ndarray:
    grid = np.frombuffer(grid_bytes, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
    pos = _find_unique(grid, token_id)
    if pos is None:
        return np.full((GRID_SIZE, GRID_SIZE), UNREACHABLE, dtype=np.int32)
    return _bfs_from(grid_bytes, pos[0], pos[1])


# ---------------------------------------------------------------------------
# Per-cell features
# ---------------------------------------------------------------------------


def is_wall(grid_input: GridLike) -> np.ndarray:
    g = _to_grid(grid_input)
    return g == WALL_ID


def on_optimal_path(grid_label: GridLike) -> np.ndarray:
    """Cells along the optimal solution path (PATH_ID + endpoints S/G)."""
    g = _to_grid(grid_label)
    return (g == PATH_ID) | (g == START_ID) | (g == GOAL_ID)


def distance_to_goal(grid_input: GridLike) -> np.ndarray:
    """Shortest-path distance (in steps) from each cell to G; UNREACHABLE if blocked."""
    g = _to_grid(grid_input)
    return _bfs_from_token_cached(_grid_key(g), GOAL_ID).copy()


def distance_to_start(grid_input: GridLike) -> np.ndarray:
    g = _to_grid(grid_input)
    return _bfs_from_token_cached(_grid_key(g), START_ID).copy()


def _free_neighbor_count(grid: np.ndarray) -> np.ndarray:
    """For each cell, number of 4-neighbors that are passable (free / S / G)."""
    passable = _passable(grid).astype(np.int32)
    nbr = np.zeros_like(passable)
    nbr[1:, :] += passable[:-1, :]
    nbr[:-1, :] += passable[1:, :]
    nbr[:, 1:] += passable[:, :-1]
    nbr[:, :-1] += passable[:, 1:]
    return nbr


def is_junction(grid_input: GridLike) -> np.ndarray:
    """Passable cell with > 2 passable neighbors (a branch point)."""
    g = _to_grid(grid_input)
    return _passable(g) & (_free_neighbor_count(g) > 2)


def is_dead_end(grid_input: GridLike) -> np.ndarray:
    """Passable cell (excluding S / G) with exactly 1 passable neighbor."""
    g = _to_grid(grid_input)
    free_only = (g == FREE_ID)
    return free_only & (_free_neighbor_count(g) == 1)


# ---------------------------------------------------------------------------
# Per-puzzle scalars
# ---------------------------------------------------------------------------


def path_length(grid_label: GridLike) -> int:
    """Number of cells on the optimal path (counts PATH_ID + S + G)."""
    g = _to_grid(grid_label)
    return int(((g == PATH_ID) | (g == START_ID) | (g == GOAL_ID)).sum())


def num_dead_ends(grid_input: GridLike) -> int:
    return int(is_dead_end(grid_input).sum())


def num_junctions(grid_input: GridLike) -> int:
    return int(is_junction(grid_input).sum())


def frac_solved(grid_pred: GridLike, grid_label: GridLike) -> float:
    """Fraction of optimal-path cells correctly predicted as path cells.

    Considers a cell "correctly solved" if it lies on the ground-truth optimal
    path (PATH_ID, S, or G) and the prediction also marks it as such.
    """
    pred = _to_grid(grid_pred)
    lab = _to_grid(grid_label)
    truth = (lab == PATH_ID) | (lab == START_ID) | (lab == GOAL_ID)
    if not truth.any():
        return 0.0
    pred_path = (pred == PATH_ID) | (pred == START_ID) | (pred == GOAL_ID)
    return float((truth & pred_path).sum()) / float(truth.sum())


# ---------------------------------------------------------------------------
# Convenience: bulk feature dict for one puzzle
# ---------------------------------------------------------------------------


def all_features(grid_input: GridLike, grid_label: GridLike | None = None) -> dict:
    """Return a dict of every per-cell and per-puzzle feature for one maze."""
    g_in = _to_grid(grid_input)
    feats = {
        "is_wall": is_wall(g_in),
        "distance_to_goal": distance_to_goal(g_in),
        "distance_to_start": distance_to_start(g_in),
        "is_junction": is_junction(g_in),
        "is_dead_end": is_dead_end(g_in),
        "num_dead_ends": num_dead_ends(g_in),
        "num_junctions": num_junctions(g_in),
    }
    if grid_label is not None:
        g_lab = _to_grid(grid_label)
        feats["on_optimal_path"] = on_optimal_path(g_lab)
        feats["path_length"] = path_length(g_lab)
    return feats


__all__ = [
    "PAD_ID", "WALL_ID", "FREE_ID", "START_ID", "GOAL_ID", "PATH_ID",
    "GRID_SIZE", "SEQ_LEN", "UNREACHABLE",
    "is_wall", "on_optimal_path",
    "distance_to_goal", "distance_to_start",
    "is_junction", "is_dead_end",
    "path_length", "num_dead_ends", "num_junctions", "frac_solved",
    "all_features",
]
