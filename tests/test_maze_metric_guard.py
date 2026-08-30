"""CPU-only tests for the Sudoku/Maze metric dispatch guard."""

import torch
from scripts.maze.maze_common import maybe_maze_batch_metrics, MAZE_METRIC_KEYS, SEQ_LEN


def test_sudoku_shape_is_ignored() -> None:
    preds = torch.randint(0, 10, (1, 81))
    labels = torch.randint(0, 10, (1, 81))

    assert maybe_maze_batch_metrics(preds, labels, preds) is None


def test_maze_shape_returns_maze_metrics() -> None:
    assert SEQ_LEN == 900
    preds = torch.randint(0, 6, (1, SEQ_LEN))
    labels = torch.randint(0, 6, (1, SEQ_LEN))
    inputs = torch.randint(0, 6, (1, SEQ_LEN))

    result = maybe_maze_batch_metrics(preds, labels, inputs)

    assert result is not None
    assert set(MAZE_METRIC_KEYS).issubset(result)


def test_missing_inputs_are_ignored() -> None:
    preds = torch.randint(0, 6, (1, SEQ_LEN))
    labels = torch.randint(0, 6, (1, SEQ_LEN))

    assert maybe_maze_batch_metrics(preds, labels, None) is None
