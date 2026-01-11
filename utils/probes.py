import os
import json
from typing import Dict, Optional, List

import torch


class ProbeRecorder:
    """
    Records hidden states (z_H, z_L) over time and assembles probe datasets
    for linear classifiers/regressors on:
      - Global metrics: is_solved, violated_rows_count, search-state variables
      - Local metrics: per-cell correctness, candidate features, cell-changes
    
    Usage:
      recorder = ProbeRecorder(output_dir)
      model(..., probe_recorder=recorder)
      recorder.finalize_and_save()
    """

    def __init__(self, output_dir: str, puzzle_type: str = "sudoku", max_steps: Optional[int] = None):
        self.output_dir = output_dir
        self.puzzle_type = puzzle_type
        self.max_steps = max_steps
        os.makedirs(self.output_dir, exist_ok=True)

        self.hidden_log: List[Dict] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For computing per-puzzle deltas like "changed since previous step".
        self._prev_compare_by_id: Dict[int, torch.Tensor] = {}

    def record_hidden(self, step_index: int, phase: str, z_H: torch.Tensor, z_L: torch.Tensor, batch: Dict[str, torch.Tensor], preds: Optional[torch.Tensor] = None):
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

    def _derive_labels(self, entry: Dict) -> Dict[str, torch.Tensor]:
        """
        Derive various labels for probing from the batch entry.
        This is puzzle-type specific; implement minimal Sudoku metrics here.
        """
        labels = {}
        if self.puzzle_type == "sudoku":
            # Token conventions (used across repo reports): PAD=0, '0'=1 (blank), digits 1..9 => token_id=digit+1.
            PAD_ID = 0
            BLANK_ID = 1

            def _as_grid_81(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if x is None or not torch.is_tensor(x):
                    return None
                if x.ndim == 1:
                    x = x.view(1, -1)
                if x.ndim == 2:
                    if x.shape[1] == 81:
                        return x
                    if x.shape[1] > 81:
                        return x[:, -81:]
                    return None
                if x.ndim == 3:
                    # Support [B,9,9]
                    if x.shape[-2:] == (9, 9):
                        return x.reshape(x.shape[0], 81)
                    # Otherwise flatten tokens dimension if possible.
                    x2 = x.reshape(x.shape[0], -1)
                    if x2.shape[1] >= 81:
                        return x2[:, -81:]
                return None

            targets = _as_grid_81(entry.get("labels"))
            inputs = _as_grid_81(entry.get("inputs"))
            preds = _as_grid_81(entry.get("preds"))
            compare = preds if preds is not None else inputs
            if compare is None:
                return labels

            B = int(compare.shape[0])

            # Per-cell correctness and solvedness (when targets are available).
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

            # Percentage of cells filled at this step.
            # Treat tokens >= 2 as filled digits; BLANK_ID=1 is empty.
            filled = (compare != PAD_ID) & (compare != BLANK_ID)
            pct_filled = filled.float().sum(dim=1) / 81.0
            labels["pct_filled"] = pct_filled.to(torch.float32)

            # Cell changed from original input grid.
            if inputs is not None:
                cell_changed_from_input = ((compare != inputs) & (inputs != PAD_ID)).to(torch.int32)
                labels["cell_changed_from_input"] = cell_changed_from_input

            # Cell changed since previous step (if we cached previous compare tensors).
            prev_list = entry.get("_prev_compare")
            if isinstance(prev_list, list) and len(prev_list) == B and inputs is not None:
                prev = []
                for p in prev_list:
                    if p is None:
                        prev.append(torch.full((81,), PAD_ID, dtype=compare.dtype))
                    else:
                        prev.append(p.view(-1)[:81].to(compare.dtype))
                prev = torch.stack(prev, dim=0)
                cells_changed_since_prev_step = ((compare != prev) & (compare != PAD_ID)).to(torch.int32)
                labels["cells_changed_since_prev_step"] = cells_changed_since_prev_step

            # Row/col indices as per-cell categorical targets (0..8).
            idx = torch.arange(81, dtype=torch.long)
            row_idx = (idx // 9).view(1, 81).expand(B, 81)
            col_idx = (idx % 9).view(1, 81).expand(B, 81)
            labels["row_idx"] = row_idx
            labels["col_idx"] = col_idx

            # Violation counts: rows/cols/boxes and total violated units.
            # Convert to digits 0..9 where 0 means empty.
            digits = (compare.to(torch.int64) - 1).clamp(min=0, max=9)
            grid = digits.view(B, 9, 9)

            def _count_unit_violations(unit: torch.Tensor) -> torch.Tensor:
                # unit: [B,9]; count whether any digit 1..9 repeats (ignore zeros)
                viol = torch.zeros((B,), dtype=torch.int32)
                for d in range(1, 10):
                    c = (unit == d).sum(dim=1)
                    viol = viol | (c > 1).to(torch.int32)
                return viol

            violated_rows = torch.stack([_count_unit_violations(grid[:, r, :]) for r in range(9)], dim=1).sum(dim=1)
            violated_cols = torch.stack([_count_unit_violations(grid[:, :, c]) for c in range(9)], dim=1).sum(dim=1)
            violated_boxes_list = []
            for br in range(3):
                for bc in range(3):
                    box = grid[:, br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3].reshape(B, 9)
                    violated_boxes_list.append(_count_unit_violations(box))
            violated_boxes = torch.stack(violated_boxes_list, dim=1).sum(dim=1)
            labels["violated_rows_count"] = violated_rows.to(torch.int32)
            labels["violated_cols_count"] = violated_cols.to(torch.int32)
            labels["violated_boxes_count"] = violated_boxes.to(torch.int32)
            labels["violated_units_total"] = (violated_rows + violated_cols + violated_boxes).to(torch.int32)

            # Forced cell: among currently blank cells, does exactly one digit fit by Sudoku constraints?
            # Computed relative to current grid (compare).
            is_forced = torch.zeros((B, 81), dtype=torch.int32)

            # Precompute row/col/box masks of used digits
            used_row = torch.zeros((B, 9, 10), dtype=torch.bool)
            used_col = torch.zeros((B, 9, 10), dtype=torch.bool)
            used_box = torch.zeros((B, 9, 10), dtype=torch.bool)
            for r in range(9):
                for c in range(9):
                    d = grid[:, r, c]
                    for dd in range(1, 10):
                        m = d == dd
                        used_row[:, r, dd] |= m
                        used_col[:, c, dd] |= m
                        b = (r // 3) * 3 + (c // 3)
                        used_box[:, b, dd] |= m

            # Candidate count for each blank cell
            for r in range(9):
                for c in range(9):
                    cell = grid[:, r, c]
                    blank = cell == 0
                    if not blank.any():
                        continue
                    b = (r // 3) * 3 + (c // 3)
                    # available digits are those not used in any of the three units
                    allowed = ~(used_row[:, r, 1:10] | used_col[:, c, 1:10] | used_box[:, b, 1:10])
                    cand_count = allowed.sum(dim=1)  # [B]
                    forced_here = blank & (cand_count == 1)
                    is_forced[:, r * 9 + c] = forced_here.to(torch.int32)

            labels["is_forced_cell"] = is_forced
        return labels

    def finalize_and_save(self):
        """Serialize tensors to .pt and a small index.json to describe the dataset."""
        # Build datasets of (hidden, labels)
        global_samples = []
        local_samples = []

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


def run_probe_collection(model, dataloader, device: Optional[torch.device] = None, output_dir: str = "results/probes"):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    recorder = ProbeRecorder(output_dir=output_dir)

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
