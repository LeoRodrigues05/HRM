"""Offline safety test: maze-metric guard must be a no-op for Sudoku (seq_len 81)
and produce metrics for maze (seq_len 900). No GPU needed."""
import numpy as np
import torch
from scripts.maze.maze_common import maybe_maze_batch_metrics, MAZE_METRIC_KEYS, SEQ_LEN

# Sudoku-shaped: labels last dim = 81 -> must return None (no-op)
preds81 = torch.randint(0, 10, (1, 81))
labels81 = torch.randint(0, 10, (1, 81))
r81 = maybe_maze_batch_metrics(preds81, labels81, preds81)
assert r81 is None, f"Sudoku NOT no-op! got {r81}"
print("Sudoku (81): no-op OK ->", r81)

# Maze-shaped: labels last dim = 900 -> must return dict with maze keys
assert SEQ_LEN == 900, SEQ_LEN
preds900 = torch.randint(0, 6, (1, 900))
labels900 = torch.randint(0, 6, (1, 900))
inputs900 = torch.randint(0, 6, (1, 900))
r900 = maybe_maze_batch_metrics(preds900, labels900, inputs900)
assert r900 is not None, "Maze returned None!"
assert set(MAZE_METRIC_KEYS).issubset(r900.keys()), set(MAZE_METRIC_KEYS) - set(r900.keys())
print("Maze (900): metrics OK, keys =", list(r900.keys()))
print("  sample:", {k: round(float(r900[k]), 3) for k in list(MAZE_METRIC_KEYS)[:6]})

# Guard when inputs is None -> None
assert maybe_maze_batch_metrics(preds900, labels900, None) is None
print("inputs=None guard OK")
print("ALL SAFETY CHECKS PASSED")
