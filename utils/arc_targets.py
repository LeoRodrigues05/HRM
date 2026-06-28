"""ARC-AGI target derivation utilities.

Companion to ``utils/maze_targets.py`` (Maze) and the Sudoku constraint
derivations in ``scripts/probes/e8_constraint_probes.py``. Provides per-cell and
per-grid features for the ARC-AGI task that can be used as linear/non-linear
probe targets, as ablation masks, or as intervention targets.

Token encoding (matches ``dataset/build_arc_dataset.py``):

    PAD_ID = 0    # padding outside the actual grid
    EOS_ID = 1    # end-of-grid marker (one row below / one col right of the grid)
    colour c in 0..9  ->  token  c + 2     (so token 2..11)

Grids live on a 30x30 canvas (seq_len = 900). The *actual* grid is a rectangle
of colour tokens (>= 2) whose bottom / right edge is delimited by EOS (== 1);
everything outside is PAD (== 0). Inputs use this encoding directly; labels use
the same encoding except the dataloader has already remapped PAD (0) to the loss
ignore id ``-100`` by the time a batch is seen by analysis code.

Per-cell features (ARC-native):
    input_colour          - 0..9 colour of the input cell (multiclass)
    output_colour         - 0..9 colour of the output (label) cell (multiclass)
    input_is_background   - input colour == 0 (black)
    input_inside_grid     - position lies inside the input grid rectangle
    output_inside_grid    - position lies inside the output grid rectangle
    is_eos                - input token == EOS
    per_cell_correct      - prediction matches the label at this cell
    colour_changed        - output colour != input colour (transform touched cell)
    same_as_input         - output token == input token (copy cell)
    input_component_size  - size of the input cell's 4-connected same-colour blob
    is_object_boundary    - input cell adjacent (4-conn) to a different colour
    num_same_colour_neighbours - 0..4 same-colour 4-neighbours in the input grid

Per-grid features (scalars, broadcastable):
    input_height / input_width   - extent of the input grid rectangle
    output_height / output_width - extent of the output grid rectangle
    size_preserved               - output extent == input extent
    num_input_colours            - distinct colours present in the input grid
    num_output_colours           - distinct colours present in the output grid

All accessors accept a flat sequence of length 900 (numpy or torch).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

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
EOS_ID = 1
COLOR_OFFSET = 2          # token = colour + COLOR_OFFSET
NUM_COLORS = 10           # colours 0..9
GRID_SIZE = 30
SEQ_LEN = GRID_SIZE * GRID_SIZE   # 900
IGNORE_LABEL_ID = -100    # matches models.losses.IGNORE_LABEL_ID


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().to("cpu").numpy()
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.reshape(-1)


def token_to_colour(tokens: np.ndarray) -> np.ndarray:
    """Map tokens to colours 0..9; non-colour tokens (PAD/EOS/ignore) -> -1."""
    tok = tokens.astype(np.int64)
    colour = tok - COLOR_OFFSET
    colour[tok < COLOR_OFFSET] = -1   # PAD, EOS, or ignore (-100) -> -1
    return colour


def grid_mask(tokens: np.ndarray) -> np.ndarray:
    """Boolean [900] mask of cells that hold a colour token (>= COLOR_OFFSET)."""
    return tokens.astype(np.int64) >= COLOR_OFFSET


def grid_extent(tokens: np.ndarray) -> Tuple[int, int, int, int]:
    """Bounding box of the colour region.

    Returns (min_row, min_col, height, width). If there are no colour cells
    (degenerate), returns (0, 0, 0, 0).
    """
    mask = grid_mask(tokens).reshape(GRID_SIZE, GRID_SIZE)
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, 0, 0
    min_r, max_r = int(rows[0]), int(rows[-1])
    min_c, max_c = int(cols[0]), int(cols[-1])
    return min_r, min_c, (max_r - min_r + 1), (max_c - min_c + 1)


def inside_grid_mask(tokens: np.ndarray) -> np.ndarray:
    """Boolean [900] mask of cells inside the grid bounding box (incl. background)."""
    min_r, min_c, h, w = grid_extent(tokens)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    if h > 0 and w > 0:
        out[min_r:min_r + h, min_c:min_c + w] = True
    return out.reshape(-1)


def _neighbors(r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < GRID_SIZE:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < GRID_SIZE:
        yield r, c + 1


def connected_component_sizes(tokens: np.ndarray) -> np.ndarray:
    """For each cell, the size of its 4-connected same-colour component.

    Only colour cells (token >= COLOR_OFFSET) participate; PAD/EOS cells get 0.
    Returns a flat [900] int array.
    """
    colour = token_to_colour(tokens).reshape(GRID_SIZE, GRID_SIZE)
    sizes = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    seen = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if seen[r, c] or colour[r, c] < 0:
                continue
            col = colour[r, c]
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc):
                    if not seen[nr, nc] and colour[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sz = len(comp)
            for (cr, cc) in comp:
                sizes[cr, cc] = sz
    return sizes.reshape(-1)


def object_boundary_mask(tokens: np.ndarray) -> np.ndarray:
    """Colour cells that touch (4-conn) a cell of a *different* colour inside the grid."""
    colour = token_to_colour(tokens).reshape(GRID_SIZE, GRID_SIZE)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if colour[r, c] < 0:
                continue
            for nr, nc in _neighbors(r, c):
                if colour[nr, nc] >= 0 and colour[nr, nc] != colour[r, c]:
                    out[r, c] = True
                    break
    return out.reshape(-1)


def num_same_colour_neighbours(tokens: np.ndarray) -> np.ndarray:
    """Count of 4-connected same-colour neighbours for each colour cell (0..4)."""
    colour = token_to_colour(tokens).reshape(GRID_SIZE, GRID_SIZE)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if colour[r, c] < 0:
                continue
            out[r, c] = sum(
                1 for nr, nc in _neighbors(r, c) if colour[nr, nc] == colour[r, c]
            )
    return out.reshape(-1)


def num_distinct_colours(tokens: np.ndarray) -> int:
    colour = token_to_colour(tokens)
    return int(np.unique(colour[colour >= 0]).size)


# ---------------------------------------------------------------------------
# Per-cell label bank (mirrors the Sudoku derive_per_cell_labels signature)
# ---------------------------------------------------------------------------

PER_CELL_BINARY = [
    "per_cell_correct",
    "input_is_background",
    "input_inside_grid",
    "output_inside_grid",
    "is_eos",
    "colour_changed",
    "same_as_input",
    "is_object_boundary",
]

PER_CELL_REGRESSION = [
    "input_component_size",
    "num_same_colour_neighbours",
]

PER_CELL_MULTICLASS = [
    "input_colour",   # 0..9 (only valid where input is a colour cell)
    "output_colour",  # 0..9 (only valid where output is a colour cell)
]

# Per-grid scalars are broadcast to every cell so they share the probe pipeline.
PER_GRID_REGRESSION = [
    "input_height",
    "input_width",
    "output_height",
    "output_width",
    "num_input_colours",
    "num_output_colours",
]
PER_GRID_BINARY = [
    "size_preserved",
]

ALL_TARGETS = (
    PER_CELL_BINARY + PER_CELL_REGRESSION + PER_CELL_MULTICLASS
    + PER_GRID_REGRESSION + PER_GRID_BINARY
)


def _derive_single(pred: np.ndarray, target: np.ndarray, inp: np.ndarray
                   ) -> Dict[str, np.ndarray]:
    """Derive all per-cell + broadcast per-grid features for one puzzle ([900])."""
    inp = inp.astype(np.int64)
    target = target.astype(np.int64)
    pred = pred.astype(np.int64)

    in_colour = token_to_colour(inp)        # -1 outside grid
    out_colour = token_to_colour(target)    # -1 outside grid / ignore
    in_inside = grid_mask(inp)
    out_inside = grid_mask(target)

    feats: Dict[str, np.ndarray] = {}

    # Binary per-cell
    valid_lab = target != IGNORE_LABEL_ID
    feats["per_cell_correct"] = ((pred == target) & valid_lab).astype(np.int32)
    feats["input_is_background"] = (in_colour == 0).astype(np.int32)
    feats["input_inside_grid"] = in_inside.astype(np.int32)
    feats["output_inside_grid"] = out_inside.astype(np.int32)
    feats["is_eos"] = (inp == EOS_ID).astype(np.int32)
    both_colour = (in_colour >= 0) & (out_colour >= 0)
    feats["colour_changed"] = (both_colour & (in_colour != out_colour)).astype(np.int32)
    feats["same_as_input"] = (both_colour & (in_colour == out_colour)).astype(np.int32)
    feats["is_object_boundary"] = object_boundary_mask(inp).astype(np.int32)

    # Regression per-cell
    feats["input_component_size"] = connected_component_sizes(inp).astype(np.int32)
    feats["num_same_colour_neighbours"] = num_same_colour_neighbours(inp).astype(np.int32)

    # Multiclass per-cell (colour 0..9; cells without a colour -> 0 but masked
    # downstream via the inside-grid selection in the probe collector)
    feats["input_colour"] = np.clip(in_colour, 0, NUM_COLORS - 1).astype(np.int32)
    feats["output_colour"] = np.clip(out_colour, 0, NUM_COLORS - 1).astype(np.int32)

    # Per-grid scalars, broadcast to every cell
    _, _, ih, iw = grid_extent(inp)
    _, _, oh, ow = grid_extent(target)
    ones = np.ones(SEQ_LEN, dtype=np.int32)
    feats["input_height"] = ones * ih
    feats["input_width"] = ones * iw
    feats["output_height"] = ones * oh
    feats["output_width"] = ones * ow
    feats["num_input_colours"] = ones * num_distinct_colours(inp)
    feats["num_output_colours"] = ones * num_distinct_colours(target)
    feats["size_preserved"] = ones * int((ih == oh) and (iw == ow))

    return feats


def derive_per_cell_labels(preds, targets, inputs) -> Dict[str, "object"]:
    """Batch wrapper. Args are [B, 900] (torch or numpy). Returns dict of [B,900].

    Mirrors ``e8_constraint_probes.derive_per_cell_labels`` so the ARC probe
    driver can reuse the same collection logic.
    """
    use_torch = torch is not None and isinstance(preds, torch.Tensor)
    P = _to_batch(preds)
    T = _to_batch(targets)
    I = _to_batch(inputs)
    B = P.shape[0]

    banks: Dict[str, list] = {t: [] for t in ALL_TARGETS}
    for b in range(B):
        f = _derive_single(P[b], T[b], I[b])
        for t in ALL_TARGETS:
            banks[t].append(f[t])

    out: Dict[str, object] = {}
    for t in ALL_TARGETS:
        stacked = np.stack(banks[t], axis=0)   # [B, 900]
        out[t] = torch.from_numpy(stacked) if use_torch else stacked
    return out


def _to_batch(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().to("cpu").numpy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr
