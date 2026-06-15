"""Report 3: Time-shift side-by-side report for Maze.

Time-shift = intra-puzzle activation transplant: cache the puzzle's z_H at
`donor_step`, run the same puzzle from scratch and at `recipient_step`
inject the donor activations, continue ACT to step max-1.

Renders an HTML report showing:
  1) The puzzle (input + label + baseline final prediction)
  2) Donor activation marker: which step's z_H is being copied
  3) Per-ACT-step side-by-side: baseline preds vs time-shifted preds

Usage (default = inject converged step 0 -> later step 8):
    python scripts/maze/time_shift_report_maze.py \
        --puzzle_idx 0 --donor_step 0 --recipient_step 8 \
        --output results/maze/reports/time_shift_p0_d0_r8.html
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles,
)
from scripts.controlled.controlled_time_shift import TimeShiftRunner
from scripts.core.activation_ablation import ActivationCache
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, grid_predictions_accuracy,
)
from scripts.maze.maze_render_common import (
    grid_html, grid_classes, html_doc, LEGEND, metrics_table,
)


def _flat(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 2:
        t = t[0]
    return t.detach().to("cpu").to(torch.int64).numpy()


def _slice_preds(preds: np.ndarray, label_len: int) -> np.ndarray:
    if preds.size > label_len:
        return preds[-label_len:]
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--puzzle_idx", type=int, default=0,
                   help="Position in collected puzzles.")
    p.add_argument("--donor_step", type=int, default=0)
    p.add_argument("--recipient_step", type=int, default=8)
    p.add_argument("--transfer_level", choices=["H", "L", "both"], default="H")
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--num_pool", type=int, default=8)
    p.add_argument("--output", type=str,
                   default="results/maze/reports/time_shift_report.html")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[time_shift_report] device={device} donor={args.donor_step} -> recipient={args.recipient_step} level={args.transfer_level}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    runner = TimeShiftRunner(model, device=device)

    puzzles = collect_puzzles(test_loader, device, max(args.num_pool, args.puzzle_idx + 1))
    loader_idx, batch = puzzles[args.puzzle_idx]
    print(f"[time_shift_report] puzzle pool_pos={args.puzzle_idx} loader_idx={loader_idx}")

    # Baseline cache (donor activations live in here too)
    base_cache: Dict[int, ActivationCache] = {}
    runner.run_and_cache_activations(batch, base_cache, max_steps=args.max_steps)

    # Time-shifted run
    _shift_out, shifted_cache = runner.run_with_time_shift(
        batch, base_cache,
        donor_step=args.donor_step,
        recipient_step=args.recipient_step,
        transfer_level=args.transfer_level,
        max_steps=args.max_steps,
    )

    label = _flat(batch["labels"])
    inp = _flat(batch["inputs"])

    base_steps = sorted(base_cache.keys())
    shift_steps = sorted(shifted_cache.keys())

    base_final = _slice_preds(_flat(base_cache[base_steps[-1]].preds), label.size)
    shift_final = _slice_preds(_flat(shifted_cache[shift_steps[-1]].preds), label.size)

    base_m = grid_predictions_accuracy(base_final, label)
    shift_m = grid_predictions_accuracy(shift_final, label)

    body: List[str] = []
    body.append("<h1>Maze — Time-Shift Report</h1>")
    body.append(
        f"<div class='meta'>"
        f"puzzle={loader_idx} &nbsp;|&nbsp; "
        f"donor_step={args.donor_step} → recipient_step={args.recipient_step} &nbsp;|&nbsp; "
        f"transfer_level={args.transfer_level} &nbsp;|&nbsp; "
        f"max_steps={args.max_steps} &nbsp;|&nbsp; "
        f"generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>"
    )
    body.append(LEGEND)

    # Summary table
    body.append("<h2>Final-prediction accuracy</h2>")
    body.append(metrics_table([
        {"run": "baseline", "token_acc": base_m["token_acc"],
         "path_cell_acc": base_m["path_cell_acc"], "exact_solved": int(base_m["exact_solved"])},
        {"run": "time-shifted", "token_acc": shift_m["token_acc"],
         "path_cell_acc": shift_m["path_cell_acc"], "exact_solved": int(shift_m["exact_solved"])},
        {"run": "Δ shift - baseline", "token_acc": shift_m["token_acc"] - base_m["token_acc"],
         "path_cell_acc": shift_m["path_cell_acc"] - base_m["path_cell_acc"],
         "exact_solved": int(shift_m["exact_solved"]) - int(base_m["exact_solved"])},
    ], ["run", "token_acc", "path_cell_acc", "exact_solved"]))

    # Context
    body.append("<h2>Puzzle context</h2><div class='row3'>")
    body.append(grid_html("Input", inp))
    body.append(grid_html("Label", label))
    body.append(grid_html(
        "Baseline final prediction", base_final,
        grid_classes(base_final, label_ids=label),
    ))
    body.append("</div>")

    # Donor highlight (baseline preds at donor step)
    donor_preds = _slice_preds(_flat(base_cache[args.donor_step].preds), label.size)
    body.append(f"<h2>Donor activations</h2>")
    body.append(
        f"<div class='meta'>Donor = baseline z_{args.transfer_level} at step "
        f"<b>{args.donor_step}</b>. The prediction at that step (shown below) is "
        f"copied in by injecting the cached z into the recipient step "
        f"<b>{args.recipient_step}</b> of a fresh forward pass.</div>"
    )
    body.append("<div class='row2'>")
    body.append(grid_html(
        f"Baseline preds @ donor step {args.donor_step}", donor_preds,
        grid_classes(donor_preds, label_ids=label),
    ))
    body.append("</div>")

    # Final side-by-side
    body.append("<h2>Final: baseline vs time-shifted</h2><div class='row2'>")
    body.append(grid_html(
        "Baseline final", base_final,
        grid_classes(base_final, label_ids=label),
    ))
    body.append(grid_html(
        "Time-shifted final (diff vs baseline highlighted)", shift_final,
        grid_classes(shift_final, label_ids=label, prev_ids=base_final),
    ))
    body.append("</div>")

    # Per-step side-by-side
    body.append("<h2>Per-ACT-step: baseline vs time-shifted</h2>")
    body.append(
        "<div class='meta'>Left tile = baseline preds after step s. "
        "Right tile = time-shifted preds after step s. Yellow-diagonal cells "
        "changed vs the baseline at the same step. Red cells disagree with the label.</div>"
    )

    rows = []
    for s in base_steps:
        bp = _slice_preds(_flat(base_cache[s].preds), label.size)
        sp = _slice_preds(_flat(shifted_cache[s].preds), label.size) if s in shifted_cache else bp
        bm = grid_predictions_accuracy(bp, label)
        sm = grid_predictions_accuracy(sp, label)
        marker = ""
        if s == args.donor_step:
            marker = " 📤DONOR"
        if s == args.recipient_step:
            marker += " 📥RECIPIENT"
        body.append(
            f"<h3>Step {s}{marker} &nbsp; base tok={bm['token_acc']:.3f} → shifted tok={sm['token_acc']:.3f}</h3>"
        )
        body.append("<div class='row2'>")
        body.append(grid_html(
            f"Baseline step {s}", bp,
            grid_classes(bp, label_ids=label), cell_size_px=11,
        ))
        body.append(grid_html(
            f"Shifted step {s} (diff vs baseline)", sp,
            grid_classes(sp, label_ids=label, prev_ids=bp), cell_size_px=11,
        ))
        body.append("</div>")
        rows.append({
            "step": s,
            "marker": marker.strip() or "",
            "base_token_acc": bm["token_acc"],
            "shifted_token_acc": sm["token_acc"],
            "Δ_token_acc": sm["token_acc"] - bm["token_acc"],
            "n_cells_diff": int((bp != sp).sum()),
        })

    body.append("<h2>Per-step summary</h2>")
    body.append(metrics_table(rows, [
        "step", "marker", "base_token_acc", "shifted_token_acc",
        "Δ_token_acc", "n_cells_diff",
    ]))

    html = html_doc("Maze — Time-Shift Report", "".join(body))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"[time_shift_report] wrote {args.output}")


if __name__ == "__main__":
    main()
