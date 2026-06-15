"""Shared Sudoku puzzle-index helpers for hardened experiment reruns.

The Phase 0 Sudoku standard is that interventions should be comparable on the
same puzzle subset. These helpers persist explicit dataloader indices and load
exactly those indices in scripts that still iterate through the test loader.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


BatchExtractor = Callable[[Any], Dict[str, Any]]


def _normalise_indices(raw: Sequence[Any], limit: Optional[int] = None) -> List[int]:
    indices = [int(i) for i in raw]
    if any(i < 0 for i in indices):
        raise ValueError("Puzzle indices must be non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("Puzzle indices must be unique")
    if limit is not None:
        indices = indices[:limit]
    return indices


def default_puzzle_indices(num_puzzles: int) -> List[int]:
    """Return the deterministic first-N dataloader indices."""
    if num_puzzles < 0:
        raise ValueError("num_puzzles must be non-negative")
    return list(range(num_puzzles))


def load_puzzle_indices(path: str, limit: Optional[int] = None) -> List[int]:
    """Load puzzle indices from either a JSON list or an index manifest."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        raw = data.get("indices", data.get("puzzle_indices"))
    else:
        raw = data
    if raw is None:
        raise ValueError(f"No 'indices' field found in {path}")
    return _normalise_indices(raw, limit=limit)


def save_puzzle_indices(
    path: str,
    indices: Sequence[int],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist explicit puzzle indices as a small JSON manifest."""
    clean = _normalise_indices(indices)
    manifest: Dict[str, Any] = {
        "puzzle_type": "sudoku",
        "selection_policy": "explicit_dataloader_indices",
        "num_puzzles": len(clean),
        "indices": clean,
    }
    if metadata:
        manifest["metadata"] = metadata
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def collect_indexed_batches(
    test_loader,
    device: torch.device,
    *,
    num_puzzles: Optional[int] = None,
    puzzle_indices: Optional[Sequence[int]] = None,
    extract_batch: BatchExtractor,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Collect first-N or explicitly indexed batches from a test loader.

    Returns ``[(dataloader_index, batch), ...]`` in requested index order.
    """
    if puzzle_indices is None:
        if num_puzzles is None:
            raise ValueError("num_puzzles is required when puzzle_indices is not set")
        requested = default_puzzle_indices(num_puzzles)
    else:
        requested = _normalise_indices(
            puzzle_indices,
            limit=num_puzzles if num_puzzles is not None else None,
        )

    if not requested:
        return []

    wanted = set(requested)
    found: Dict[int, Dict[str, Any]] = {}
    max_requested = max(wanted)

    for idx, data in enumerate(test_loader):
        if idx in wanted:
            found[idx] = _to_device(extract_batch(data), device)
            if len(found) == len(wanted):
                break
        if idx >= max_requested and len(found) < len(wanted):
            break

    missing = [idx for idx in requested if idx not in found]
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        raise ValueError(
            f"Could not collect {len(missing)} requested puzzle indices "
            f"(first missing: {preview})"
        )

    return [(idx, found[idx]) for idx in requested]
