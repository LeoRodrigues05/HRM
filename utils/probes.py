"""Probe Recording Utilities for HRM Hidden State Analysis.

This module provides the `ProbeRecorder` class for capturing hidden states
(z_H and z_L) during model inference to build datasets for linear probe
analysis.

The recorded data enables analysis of:
- Global properties: puzzle-level metrics like is_solved, violation counts
- Local properties: per-cell metrics like correctness, position indices

Typical usage:
    recorder = ProbeRecorder(output_dir="results/probes")
    # During model inference:
    recorder.record_hidden(step_index=0, phase="grad", z_H=z_H, z_L=z_L, batch=batch)
    # After all inference:
    recorder.finalize_and_save()

Output files:
    - probe_global.pt: Pooled (mean) features per step
    - probe_local.pt: Per-token features per step
    - probe_index.json: Metadata about the collection
"""
import os
import json
import logging
from typing import Dict, Optional, List, Any

import torch

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Sudoku token IDs (common across the repository)
PAD_ID = 0
BLANK_ID = 1  # Represents '0' or empty cell
DIGIT_OFFSET = 1  # Token ID = digit + DIGIT_OFFSET (so digit 1 = token 2)

# Sudoku grid dimensions
SUDOKU_SIZE = 9
SUDOKU_CELLS = 81


# ============================================================================
# ProbeRecorder Class
# ============================================================================


class ProbeRecorder:
    """Records hidden states (z_H, z_L) for building linear probe datasets.
    
    This class captures intermediate representations during HRM inference
    to enable post-hoc analysis of what information is encoded in the
    model's hidden states.
    
    The recorder supports:
    - Recording at arbitrary ACT steps and phases (grad/nograd)
    - Automatic label derivation for Sudoku puzzles
    - Global (pooled) and local (per-token) feature extraction
    
    Attributes:
        output_dir: Directory where probe files will be saved
        puzzle_type: Type of puzzle (currently only "sudoku" is supported)
        max_steps: Optional limit on number of steps to record
        hidden_log: List of recorded hidden state entries
        
    Example:
        >>> recorder = ProbeRecorder("results/probes")
        >>> recorder.record_hidden(step_index=0, phase="grad", z_H=z_H, z_L=z_L, batch=batch)
        >>> recorder.finalize_and_save()
    """

    def __init__(
        self, 
        output_dir: str, 
        puzzle_type: str = "sudoku", 
        max_steps: Optional[int] = None
    ):
        """Initialize the ProbeRecorder.
        
        Args:
            output_dir: Directory to save probe files
            puzzle_type: Type of puzzle for label derivation ("sudoku")
            max_steps: Maximum number of ACT steps to record (None = no limit)
        """
        self.output_dir = output_dir
        self.puzzle_type = puzzle_type
        self.max_steps = max_steps
        os.makedirs(self.output_dir, exist_ok=True)

        self.hidden_log: List[Dict[str, Any]] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache for computing per-puzzle deltas like "changed since previous step"
        self._prev_compare_by_id: Dict[int, torch.Tensor] = {}

    def record_hidden(
        self, 
        step_index: int, 
        phase: str, 
        z_H: torch.Tensor, 
        z_L: torch.Tensor, 
        batch: Dict[str, torch.Tensor], 
        preds: Optional[torch.Tensor] = None
    ) -> None:
        """Record hidden states for a single step.
        
        Args:
            step_index: Current ACT step index (0-based)
            phase: Current phase ("grad" or "nograd")
            z_H: High-level hidden states [B, T, D]
            z_L: Low-level hidden states [B, T, D]
            batch: Input batch dictionary with "inputs", "labels", etc.
            preds: Optional model predictions for this step [B, T]
        """
        # Detach and move to CPU to keep memory reasonable
        z_H_cpu = z_H.detach().to("cpu").float()
        z_L_cpu = z_L.detach().to("cpu").float()

        entry = {
            "step": int(step_index),
            "phase": str(phase),
            "z_H_shape": list(z_H_cpu.shape),
            "z_L_shape": list(z_L_cpu.shape),
        }
        # Save references to tensors separately to avoid JSON bloat
        entry["z_H"] = z_H_cpu
        entry["z_L"] = z_L_cpu

        # Also keep relevant inputs for label derivation
        for k in ("inputs", "labels", "puzzle_identifiers"):
            if k in batch:
                entry[k] = batch[k].detach().to("cpu")

        # Optional: model intermediate predictions (token IDs)
        if preds is not None:
            entry["preds"] = preds.detach().to("cpu")

        # Cache previous-step compare tensors for this batch when identifiers are available.
        # We'll compute compare = preds if present else inputs.
        try:
            ids = entry.get("puzzle_identifiers")
            inputs = entry.get("inputs")
            compare = entry.get("preds") if ("preds" in entry) else inputs
            if ids is not None and compare is not None and torch.is_tensor(ids) and torch.is_tensor(compare):
                ids_flat = ids.view(-1).to(torch.int64)
                compare_flat = compare
                if compare_flat.ndim >= 2:
                    compare_flat = compare_flat.reshape(compare_flat.shape[0], -1)
                # Save the last 81 tokens as the Sudoku grid when possible.
                if compare_flat.ndim == 2 and compare_flat.shape[1] >= 81:
                    compare_flat = compare_flat[:, -81:]
                prev_list = []
                for b in range(ids_flat.numel()):
                    pid = int(ids_flat[b].item())
                    prev = self._prev_compare_by_id.get(pid)
                    prev_list.append(prev)
                    self._prev_compare_by_id[pid] = compare_flat[b].detach().clone()
                entry["_prev_compare"] = prev_list
        except Exception:
            pass

        self.hidden_log.append(entry)

    def _derive_labels(self, entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Derive probe labels from the batch entry.
        
        This method computes various labels that linear probes can be trained
        to predict. The labels are puzzle-type specific.
        
        For Sudoku puzzles, the following labels are derived:
        
        **Global Labels (puzzle-level):**
        - is_solved: Binary indicator (1 if all cells match target)
        - pct_filled: Fraction of cells filled (0.0 to 1.0)
        - violated_rows_count: Number of rows with duplicate digits
        - violated_cols_count: Number of columns with duplicate digits  
        - violated_boxes_count: Number of 3x3 boxes with duplicate digits
        - violated_units_total: Sum of all violation counts
        
        **Local Labels (per-cell):**
        - per_cell_correct: Binary per-cell correctness [B, 81]
        - cell_changed_from_input: Binary indicator of cell modification [B, 81]
        - cells_changed_since_prev_step: Binary indicator of step-to-step change [B, 81]
        - row_idx: Row index (0-8) for each cell [B, 81]
        - col_idx: Column index (0-8) for each cell [B, 81]
        - is_forced_cell: Binary indicator if cell has exactly one valid candidate [B, 81]
        
        Args:
            entry: Dictionary containing inputs, labels, preds, and hidden states
            
        Returns:
            Dictionary mapping label names to tensors
        """
        labels: Dict[str, torch.Tensor] = {}
        
        if self.puzzle_type != "sudoku":
            logger.warning(f"Label derivation not implemented for puzzle_type={self.puzzle_type}")
            return labels
            
        # Helper function to extract 81-cell grid from various tensor shapes
        def _as_grid_81(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            """Convert tensor to [B, 81] grid format."""
            if x is None or not torch.is_tensor(x):
                return None
            if x.ndim == 1:
                x = x.view(1, -1)
            if x.ndim == 2:
                if x.shape[1] == SUDOKU_CELLS:
                    return x
                if x.shape[1] > SUDOKU_CELLS:
                    return x[:, -SUDOKU_CELLS:]
                return None
            if x.ndim == 3:
                # Support [B, 9, 9] format
                if x.shape[-2:] == (SUDOKU_SIZE, SUDOKU_SIZE):
                    return x.reshape(x.shape[0], SUDOKU_CELLS)
                # Otherwise flatten tokens dimension if possible
                x2 = x.reshape(x.shape[0], -1)
                if x2.shape[1] >= SUDOKU_CELLS:
                    return x2[:, -SUDOKU_CELLS:]
            return None

        targets = _as_grid_81(entry.get("labels"))
        inputs = _as_grid_81(entry.get("inputs"))
        preds = _as_grid_81(entry.get("preds"))
        compare = preds if preds is not None else inputs
        if compare is None:
            return labels

        B = int(compare.shape[0])

        # ----------------------------------------------------------------
        # Global labels: Puzzle-level metrics
        # ----------------------------------------------------------------
        
        # Per-cell correctness and solvedness (when targets are available)
        if targets is not None:
            try:
                per_cell_correct = (compare == targets).to(torch.int32)
            except Exception:
                per_cell_correct = torch.zeros_like(compare, dtype=torch.int32)
            labels["per_cell_correct"] = per_cell_correct
            try:
                is_solved = torch.all(compare == targets, dim=1).to(torch.int32)
            except Exception:
                is_solved = torch.zeros((B,), dtype=torch.int32)
            labels["is_solved"] = is_solved

        # Percentage of cells filled at this step
        # Treat tokens >= 2 as filled digits; BLANK_ID=1 is empty
        filled = (compare != PAD_ID) & (compare != BLANK_ID)
        pct_filled = filled.float().sum(dim=1) / float(SUDOKU_CELLS)
        labels["pct_filled"] = pct_filled.to(torch.float32)

        # ----------------------------------------------------------------
        # Local labels: Per-cell metrics
        # ----------------------------------------------------------------
        
        # Cell changed from original input grid
        if inputs is not None:
            cell_changed_from_input = ((compare != inputs) & (inputs != PAD_ID)).to(torch.int32)
            labels["cell_changed_from_input"] = cell_changed_from_input

        # Cell changed since previous step (if we cached previous compare tensors)
        prev_list = entry.get("_prev_compare")
        if isinstance(prev_list, list) and len(prev_list) == B and inputs is not None:
            prev = []
            for p in prev_list:
                if p is None:
                    prev.append(torch.full((SUDOKU_CELLS,), PAD_ID, dtype=compare.dtype))
                else:
                    prev.append(p.view(-1)[:SUDOKU_CELLS].to(compare.dtype))
            prev = torch.stack(prev, dim=0)
            cells_changed_since_prev_step = ((compare != prev) & (compare != PAD_ID)).to(torch.int32)
            labels["cells_changed_since_prev_step"] = cells_changed_since_prev_step

        # Row/col indices as per-cell categorical targets (0..8)
        idx = torch.arange(SUDOKU_CELLS, dtype=torch.long)
        row_idx = (idx // SUDOKU_SIZE).view(1, SUDOKU_CELLS).expand(B, SUDOKU_CELLS)
        col_idx = (idx % SUDOKU_SIZE).view(1, SUDOKU_CELLS).expand(B, SUDOKU_CELLS)
        labels["row_idx"] = row_idx
        labels["col_idx"] = col_idx
        
        # Box index (0-8) for each cell
        box_idx = ((row_idx // 3) * 3 + (col_idx // 3))
        labels["box_idx"] = box_idx
        
        # Position within the 3x3 box (0-8)
        position_in_box = ((row_idx % 3) * 3 + (col_idx % 3))
        labels["position_in_box"] = position_in_box

        # ----------------------------------------------------------------
        # Constraint violation counts
        # ----------------------------------------------------------------
        
        # Convert to digits 0..9 where 0 means empty
        digits = (compare.to(torch.int64) - DIGIT_OFFSET).clamp(min=0, max=SUDOKU_SIZE)
        grid = digits.view(B, SUDOKU_SIZE, SUDOKU_SIZE)

        def _count_unit_violations(unit: torch.Tensor) -> torch.Tensor:
            """Count if any digit 1..9 repeats in a unit (ignore zeros)."""
            viol = torch.zeros((B,), dtype=torch.int32)
            for d in range(1, SUDOKU_SIZE + 1):
                c = (unit == d).sum(dim=1)
                viol = viol | (c > 1).to(torch.int32)
            return viol

        violated_rows = torch.stack(
            [_count_unit_violations(grid[:, r, :]) for r in range(SUDOKU_SIZE)], dim=1
        ).sum(dim=1)
        violated_cols = torch.stack(
            [_count_unit_violations(grid[:, :, c]) for c in range(SUDOKU_SIZE)], dim=1
        ).sum(dim=1)
        
        violated_boxes_list = []
        for br in range(3):
            for bc in range(3):
                box = grid[:, br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3].reshape(B, SUDOKU_SIZE)
                violated_boxes_list.append(_count_unit_violations(box))
        violated_boxes = torch.stack(violated_boxes_list, dim=1).sum(dim=1)
        
        labels["violated_rows_count"] = violated_rows.to(torch.int32)
        labels["violated_cols_count"] = violated_cols.to(torch.int32)
        labels["violated_boxes_count"] = violated_boxes.to(torch.int32)
        labels["violated_units_total"] = (violated_rows + violated_cols + violated_boxes).to(torch.int32)

        # ----------------------------------------------------------------
        # Cell state features
        # ----------------------------------------------------------------
        
        # Current digit value at each cell (0 = empty, 1-9 = filled)
        cell_digit = digits.view(B, SUDOKU_CELLS)
        labels["cell_digit"] = cell_digit.to(torch.int32)
        
        # Is the cell currently empty?
        is_empty = (cell_digit == 0).to(torch.int32)
        labels["is_empty"] = is_empty
        
        # Was this cell a given (clue) in the original input?
        if inputs is not None:
            input_digits = (inputs.to(torch.int64) - DIGIT_OFFSET).clamp(min=0, max=SUDOKU_SIZE)
            is_given = (input_digits != 0).to(torch.int32)
            labels["is_given"] = is_given

        # ----------------------------------------------------------------
        # Candidate-based features (per-cell)
        # ----------------------------------------------------------------
        
        # Precompute row/col/box masks of used digits
        used_row = torch.zeros((B, SUDOKU_SIZE, SUDOKU_SIZE + 1), dtype=torch.bool)
        used_col = torch.zeros((B, SUDOKU_SIZE, SUDOKU_SIZE + 1), dtype=torch.bool)
        used_box = torch.zeros((B, SUDOKU_SIZE, SUDOKU_SIZE + 1), dtype=torch.bool)
        
        for r in range(SUDOKU_SIZE):
            for c in range(SUDOKU_SIZE):
                d = grid[:, r, c]
                for dd in range(1, SUDOKU_SIZE + 1):
                    m = d == dd
                    used_row[:, r, dd] |= m
                    used_col[:, c, dd] |= m
                    b = (r // 3) * 3 + (c // 3)
                    used_box[:, b, dd] |= m

        # Per-cell candidate count and related features
        candidate_count = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_naked_single = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_hidden_single_row = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_hidden_single_col = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_hidden_single_box = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        candidate_set = torch.zeros((B, SUDOKU_CELLS, SUDOKU_SIZE), dtype=torch.bool)  # [B, 81, 9]
        
        for r in range(SUDOKU_SIZE):
            for c in range(SUDOKU_SIZE):
                cell_idx = r * SUDOKU_SIZE + c
                cell = grid[:, r, c]
                blank = cell == 0
                b = (r // 3) * 3 + (c // 3)
                
                # Candidates are digits not used in row, col, or box
                allowed = ~(used_row[:, r, 1:SUDOKU_SIZE+1] | 
                           used_col[:, c, 1:SUDOKU_SIZE+1] | 
                           used_box[:, b, 1:SUDOKU_SIZE+1])  # [B, 9]
                
                # Store candidate set for this cell
                candidate_set[:, cell_idx, :] = allowed & blank.unsqueeze(1)
                
                # Candidate count (0 for filled cells)
                cand_count = (allowed & blank.unsqueeze(1)).sum(dim=1)
                candidate_count[:, cell_idx] = cand_count.to(torch.int32)
                
                # Naked single: exactly 1 candidate
                is_naked_single[:, cell_idx] = (blank & (cand_count == 1)).to(torch.int32)
        
        labels["candidate_count"] = candidate_count
        labels["is_naked_single"] = is_naked_single  # Same as is_forced_cell
        
        # ----------------------------------------------------------------
        # Hidden singles detection
        # ----------------------------------------------------------------
        # A hidden single occurs when a digit can only go in one cell within a unit
        
        for r in range(SUDOKU_SIZE):
            for c in range(SUDOKU_SIZE):
                cell_idx = r * SUDOKU_SIZE + c
                b = (r // 3) * 3 + (c // 3)
                blank = grid[:, r, c] == 0
                
                if not blank.any():
                    continue
                
                cell_cands = candidate_set[:, cell_idx, :]  # [B, 9]
                
                # Check each candidate digit
                for d in range(SUDOKU_SIZE):
                    has_cand = cell_cands[:, d]  # [B]
                    if not has_cand.any():
                        continue
                    
                    # Hidden single in row: this is the only cell in the row with this candidate
                    row_cells = [r * SUDOKU_SIZE + cc for cc in range(SUDOKU_SIZE) if cc != c]
                    others_have_in_row = torch.zeros((B,), dtype=torch.bool)
                    for other_idx in row_cells:
                        others_have_in_row |= candidate_set[:, other_idx, d]
                    is_hidden_row = has_cand & ~others_have_in_row
                    is_hidden_single_row[:, cell_idx] |= is_hidden_row.to(torch.int32)
                    
                    # Hidden single in column
                    col_cells = [rr * SUDOKU_SIZE + c for rr in range(SUDOKU_SIZE) if rr != r]
                    others_have_in_col = torch.zeros((B,), dtype=torch.bool)
                    for other_idx in col_cells:
                        others_have_in_col |= candidate_set[:, other_idx, d]
                    is_hidden_col = has_cand & ~others_have_in_col
                    is_hidden_single_col[:, cell_idx] |= is_hidden_col.to(torch.int32)
                    
                    # Hidden single in box
                    box_r, box_c = (r // 3) * 3, (c // 3) * 3
                    box_cells = [
                        (box_r + dr) * SUDOKU_SIZE + (box_c + dc)
                        for dr in range(3) for dc in range(3)
                        if (box_r + dr) != r or (box_c + dc) != c
                    ]
                    others_have_in_box = torch.zeros((B,), dtype=torch.bool)
                    for other_idx in box_cells:
                        others_have_in_box |= candidate_set[:, other_idx, d]
                    is_hidden_box = has_cand & ~others_have_in_box
                    is_hidden_single_box[:, cell_idx] |= is_hidden_box.to(torch.int32)
        
        labels["is_hidden_single_row"] = is_hidden_single_row
        labels["is_hidden_single_col"] = is_hidden_single_col
        labels["is_hidden_single_box"] = is_hidden_single_box
        # Combined: is hidden single in ANY unit
        labels["is_hidden_single"] = ((is_hidden_single_row | is_hidden_single_col | is_hidden_single_box) > 0).to(torch.int32)
        
        # ----------------------------------------------------------------
        # Constraint pressure features (per-cell)
        # ----------------------------------------------------------------
        
        # Count filled cells in each unit for each cell
        filled_in_row = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        filled_in_col = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        filled_in_box = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        
        row_fill_counts = (grid != 0).sum(dim=2)  # [B, 9]
        col_fill_counts = (grid != 0).sum(dim=1)  # [B, 9]
        
        for r in range(SUDOKU_SIZE):
            for c in range(SUDOKU_SIZE):
                cell_idx = r * SUDOKU_SIZE + c
                b = (r // 3) * 3 + (c // 3)
                box_r, box_c = (r // 3) * 3, (c // 3) * 3
                box_fill = (grid[:, box_r:box_r+3, box_c:box_c+3] != 0).sum(dim=(1, 2))
                
                filled_in_row[:, cell_idx] = row_fill_counts[:, r]
                filled_in_col[:, cell_idx] = col_fill_counts[:, c]
                filled_in_box[:, cell_idx] = box_fill
        
        labels["filled_in_row"] = filled_in_row
        labels["filled_in_col"] = filled_in_col
        labels["filled_in_box"] = filled_in_box
        
        # Total constraint pressure: sum of filled cells across all three units (minus double counting)
        # Each cell sees: row + col + box - 2 (the cell itself is counted 3 times if filled)
        constraint_pressure = filled_in_row + filled_in_col + filled_in_box
        labels["constraint_pressure"] = constraint_pressure
        
        # ----------------------------------------------------------------
        # Relative constraint features
        # ----------------------------------------------------------------
        
        # Is this cell the most constrained (fewest candidates) in its row/col/box?
        is_min_cand_in_row = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_min_cand_in_col = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_min_cand_in_box = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        
        # Set filled cells to high candidate count for min comparison
        cand_for_min = candidate_count.clone().float()
        cand_for_min[is_empty == 0] = 99  # Filled cells don't compete
        
        for r in range(SUDOKU_SIZE):
            row_cands = cand_for_min[:, r*SUDOKU_SIZE:(r+1)*SUDOKU_SIZE]  # [B, 9]
            row_min = row_cands.min(dim=1, keepdim=True).values
            row_is_min = (row_cands == row_min) & (row_cands < 99)
            is_min_cand_in_row[:, r*SUDOKU_SIZE:(r+1)*SUDOKU_SIZE] = row_is_min.to(torch.int32)
        
        for c in range(SUDOKU_SIZE):
            col_indices = [r * SUDOKU_SIZE + c for r in range(SUDOKU_SIZE)]
            col_cands = cand_for_min[:, col_indices]  # [B, 9]
            col_min = col_cands.min(dim=1, keepdim=True).values
            col_is_min = (col_cands == col_min) & (col_cands < 99)
            for i, idx in enumerate(col_indices):
                is_min_cand_in_col[:, idx] = col_is_min[:, i].to(torch.int32)
        
        for br in range(3):
            for bc in range(3):
                box_indices = [
                    (br*3 + dr) * SUDOKU_SIZE + (bc*3 + dc)
                    for dr in range(3) for dc in range(3)
                ]
                box_cands = cand_for_min[:, box_indices]  # [B, 9]
                box_min = box_cands.min(dim=1, keepdim=True).values
                box_is_min = (box_cands == box_min) & (box_cands < 99)
                for i, idx in enumerate(box_indices):
                    is_min_cand_in_box[:, idx] = box_is_min[:, i].to(torch.int32)
        
        labels["is_min_cand_in_row"] = is_min_cand_in_row
        labels["is_min_cand_in_col"] = is_min_cand_in_col
        labels["is_min_cand_in_box"] = is_min_cand_in_box
        labels["is_most_constrained"] = ((is_min_cand_in_row | is_min_cand_in_col | is_min_cand_in_box) > 0).to(torch.int32)
        
        # ----------------------------------------------------------------
        # Pointing/Claiming (Locked Candidates) detection
        # ----------------------------------------------------------------
        
        # A cell participates in locked candidates if:
        # - Its candidates in the box are confined to one row/col (pointing)
        # - Its candidates in the row/col are confined to one box (claiming)
        is_pointing_cell = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        is_claiming_cell = torch.zeros((B, SUDOKU_CELLS), dtype=torch.int32)
        
        for br in range(3):
            for bc in range(3):
                box_r, box_c = br * 3, bc * 3
                box_indices = [
                    (box_r + dr, box_c + dc, (box_r + dr) * SUDOKU_SIZE + (box_c + dc))
                    for dr in range(3) for dc in range(3)
                ]
                
                for d in range(SUDOKU_SIZE):
                    # Get cells in this box that have candidate d
                    cand_positions = []
                    for r, c, idx in box_indices:
                        has_d = candidate_set[:, idx, d]  # [B]
                        cand_positions.append((r, c, idx, has_d))
                    
                    # Check if all candidates for d are in same row (pointing to row)
                    for target_r in range(box_r, box_r + 3):
                        in_row = torch.zeros((B,), dtype=torch.bool)
                        out_row = torch.zeros((B,), dtype=torch.bool)
                        for r, c, idx, has_d in cand_positions:
                            if r == target_r:
                                in_row |= has_d
                            else:
                                out_row |= has_d
                        # Pointing: candidates exist in row but not outside
                        pointing = in_row & ~out_row
                        if pointing.any():
                            for r, c, idx, has_d in cand_positions:
                                if r == target_r:
                                    is_pointing_cell[:, idx] |= (pointing & has_d).to(torch.int32)
                    
                    # Check if all candidates for d are in same col (pointing to col)
                    for target_c in range(box_c, box_c + 3):
                        in_col = torch.zeros((B,), dtype=torch.bool)
                        out_col = torch.zeros((B,), dtype=torch.bool)
                        for r, c, idx, has_d in cand_positions:
                            if c == target_c:
                                in_col |= has_d
                            else:
                                out_col |= has_d
                        pointing = in_col & ~out_col
                        if pointing.any():
                            for r, c, idx, has_d in cand_positions:
                                if c == target_c:
                                    is_pointing_cell[:, idx] |= (pointing & has_d).to(torch.int32)
        
        labels["is_pointing_cell"] = is_pointing_cell
        labels["is_claiming_cell"] = is_claiming_cell
        labels["is_locked_candidate"] = ((is_pointing_cell | is_claiming_cell) > 0).to(torch.int32)
        
        # Keep backward compatibility
        labels["is_forced_cell"] = is_naked_single
        
        return labels

    def finalize_and_save(self) -> None:
        """Serialize recorded data to disk.
        
        This method processes all recorded entries, derives labels, and saves
        the resulting probe datasets to disk.
        
        Output files:
            - probe_global.pt: List of dicts with pooled z_H/z_L and labels
            - probe_local.pt: List of dicts with per-token z_H/z_L and labels
            - probe_index.json: Metadata about the collection
        """
        logger.info(f"Processing {len(self.hidden_log)} recorded entries...")
        
        # Build datasets of (hidden, labels)
        global_samples: List[Dict[str, Any]] = []
        local_samples: List[Dict[str, Any]] = []

        for entry in self.hidden_log:
            labels = self._derive_labels(entry)
            z_H = entry["z_H"]  # [B, T, D]
            z_L = entry["z_L"]  # [B, T, D]
            step = entry["step"]
            phase = entry["phase"]

            # Global samples use pooled z_H and z_L (mean over sequence)
            z_H_global = z_H.mean(dim=-2)  # [B, D]
            z_L_global = z_L.mean(dim=-2)  # [B, D]
            global_samples.append({
                "step": step,
                "phase": phase,
                "z_H": z_H_global,
                "z_L": z_L_global,
                "labels": labels,
            })

            # Local samples: keep per-token features to decode fine-grained statuses
            local_samples.append({
                "step": step,
                "phase": phase,
                "z_H": z_H,
                "z_L": z_L,
                "labels": labels,
            })

        # Save torch tensors in lists
        torch.save(global_samples, os.path.join(self.output_dir, "probe_global.pt"))
        torch.save(local_samples, os.path.join(self.output_dir, "probe_local.pt"))

        # Small JSON index
        index = {
            "count": len(self.hidden_log),
            "puzzle_type": self.puzzle_type,
            "files": {
                "global": "probe_global.pt",
                "local": "probe_local.pt",
            }
        }
        with open(os.path.join(self.output_dir, "probe_index.json"), "w") as f:
            json.dump(index, f, indent=2)


# ============================================================================
# Convenience Function for Probe Collection
# ============================================================================

def run_probe_collection(
    model,  # HierarchicalReasoningModel_ACTV1 or similar
    dataloader, 
    device: Optional[torch.device] = None, 
    output_dir: str = "results/probes"
) -> ProbeRecorder:
    """Run probe collection over a dataloader.
    
    This convenience function handles the full probe collection workflow:
    1. Initialize a ProbeRecorder
    2. Run the model through all batches in the dataloader
    3. Record hidden states at each ACT step
    4. Save the collected data to disk
    
    Args:
        model: HRM model to collect probes from (must support probe_recorder argument)
        dataloader: DataLoader providing batches of puzzles
        device: Device to run on (defaults to CUDA if available)
        output_dir: Directory to save probe files
        
    Returns:
        The ProbeRecorder instance with all collected data
        
    Example:
        >>> recorder = run_probe_collection(model, test_loader, output_dir="results/probes")
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    recorder = ProbeRecorder(output_dir=output_dir)
    logger.info(f"Starting probe collection on device={device}")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)
            # Run ACT loop while recording
            for _ in range(model.config.halt_max_steps):
                carry, outputs = model(carry, batch, probe_recorder=recorder)
                if torch.all(carry.halted):
                    break
    recorder.finalize_and_save()
    return recorder
