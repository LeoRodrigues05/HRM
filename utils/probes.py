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

        self.hidden_log.append(entry)

    def _derive_labels(self, entry: Dict) -> Dict[str, torch.Tensor]:
        """
        Derive various labels for probing from the batch entry.
        This is puzzle-type specific; implement minimal Sudoku metrics here.
        """
        labels = {}
        if self.puzzle_type == "sudoku":
            # Expect labels to be target tokens (solution). We can compute simple correctness.
            if "labels" in entry:
                targets = entry["labels"]
                # Prefer model predictions if present; else fall back to inputs
                preds = entry.get("preds")
                inputs = entry.get("inputs")
                compare = preds if preds is not None else inputs
                if compare is None:
                    return labels
                # Per-cell correctness (binary)
                try:
                    per_cell_correct = (compare == targets).to(torch.int32)
                except Exception:
                    per_cell_correct = torch.zeros_like(targets, dtype=torch.int32)
                labels["per_cell_correct"] = per_cell_correct

                # Global: solved at step
                try:
                    is_solved = torch.all(compare == targets, dim=(-1, -2)) if compare.ndim == 3 else torch.all(compare == targets, dim=-1)
                except Exception:
                    is_solved = torch.zeros((targets.shape[0],), dtype=torch.int32)
                labels["is_solved"] = is_solved.to(torch.int32)

                # Simple heuristic: rows violated count (requires 9x9 token grid). If unavailable, fallback to zeros.
                try:
                    grid = inputs
                    if grid.ndim == 3:
                        # assume shape (B, S, V) -> not directly a grid; skip
                        violated_rows = torch.zeros((grid.shape[0],), dtype=torch.int32)
                    else:
                        violated_rows = torch.zeros((grid.shape[0],), dtype=torch.int32)
                except Exception:
                    violated_rows = torch.zeros((inputs.shape[0],), dtype=torch.int32)
                labels["violated_rows_count"] = violated_rows

                # Which cells changed since last step: we need prior step in same batch; here we leave placeholder zeros
                changed_mask = torch.zeros_like(per_cell_correct)
                labels["cells_changed"] = changed_mask
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
