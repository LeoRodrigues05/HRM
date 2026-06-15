"""Report 1: Intermediate solutions for a Maze test puzzle.

Picks a (random or specified) test puzzle, runs the maze HRM with ACT step
caching, and emits a single self-contained HTML report showing the maze
input, the ground-truth label, and the model's prediction at every ACT step.

Usage:
    python scripts/maze/intermediate_solutions_maze.py \
        --puzzle_idx 7 --output results/maze/reports/intermediate_p7.html
    python scripts/maze/intermediate_solutions_maze.py --random
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles,
)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
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
    """Strip puzzle-emb prefix from preds if present."""
    if preds.size > label_len:
        return preds[-label_len:]
    return preds


def render(puzzle_idx: int, batch, ablator: ActivationAblator, max_steps: int,
           puzzle_emb_len: int) -> str:
    cache: Dict[int, ActivationCache] = {}
    ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
    steps = sorted(cache.keys())
    label = _flat(batch["labels"])
    inp = _flat(batch["inputs"])

    final_preds = _slice_preds(_flat(cache[steps[-1]].preds), label.size)
    final_metrics = grid_predictions_accuracy(final_preds, label)

    body: List[str] = []
    body.append(f"<h1>Maze — Intermediate Solutions</h1>")
    body.append(
        f"<div class='meta'>Puzzle idx: <b>{puzzle_idx}</b> &nbsp;|&nbsp; "
        f"max_steps={max_steps} &nbsp;|&nbsp; "
        f"final token_acc={final_metrics['token_acc']:.4f} &nbsp;|&nbsp; "
        f"path_cell_acc={final_metrics['path_cell_acc']:.4f} &nbsp;|&nbsp; "
        f"exact_solved={int(final_metrics['exact_solved'])} &nbsp;|&nbsp; "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
    )
    body.append(LEGEND)

    # Context: input, label, final prediction
    body.append("<h2>Puzzle context</h2><div class='row3'>")
    body.append(grid_html("Input (S/G/walls)", inp))
    body.append(grid_html("Label (optimal path)", label))
    body.append(grid_html(
        "Final prediction (errors highlighted)", final_preds,
        grid_classes(final_preds, label_ids=label),
    ))
    body.append("</div>")

    # Per-step trajectory
    body.append("<h2>ACT-step trajectory</h2>")
    body.append(
        "<div class='meta'>Each tile is the model's prediction after that ACT step. "
        "Cells highlighted with a yellow diagonal changed since the previous step. "
        "Red cells disagree with the ground-truth label.</div>"
    )
    body.append("<div class='steps'>")
    prev_preds: Optional[np.ndarray] = None
    step_rows: List[Dict[str, object]] = []
    for s in steps:
        preds = _slice_preds(_flat(cache[s].preds), label.size)
        m = grid_predictions_accuracy(preds, label)
        cls = grid_classes(preds, label_ids=label, prev_ids=prev_preds)
        title = (f"Step {s} &nbsp;tok={m['token_acc']:.3f}"
                 f" path={m['path_cell_acc']:.3f}"
                 f" {'✓' if m['exact_solved'] else ''}")
        body.append(grid_html(title, preds, cls, cell_size_px=10))
        step_rows.append({
            "step": s, "token_acc": m["token_acc"],
            "path_cell_acc": m["path_cell_acc"],
            "exact_solved": int(m["exact_solved"]),
        })
        prev_preds = preds
    body.append("</div>")

    # Numeric table
    body.append("<h2>Per-step metrics</h2>")
    body.append(metrics_table(step_rows, ["step", "token_acc", "path_cell_acc", "exact_solved"]))

    return html_doc("Maze — Intermediate Solutions", "".join(body))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--puzzle_idx", type=int, default=None,
                   help="Test-set puzzle index (collection order). Random if not set.")
    p.add_argument("--random", action="store_true", help="Pick a random puzzle.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--num_pool", type=int, default=64,
                   help="How many puzzles to pull from the loader; sampled puzzle must be < this.")
    p.add_argument("--output", type=str,
                   default="results/maze/reports/intermediate_solutions.html")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[intermediate] device={device}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    ablator = ActivationAblator(model, device=device)

    puzzles = collect_puzzles(test_loader, device, args.num_pool)
    print(f"[intermediate] collected {len(puzzles)} candidate puzzles")

    if args.puzzle_idx is None or args.random:
        chosen_pos = rng.randrange(len(puzzles))
    else:
        # Match by the puzzle's idx-in-loader; fall back to position
        chosen_pos = next((i for i, (idx, _) in enumerate(puzzles) if idx == args.puzzle_idx),
                          args.puzzle_idx if args.puzzle_idx < len(puzzles) else 0)
    loader_idx, batch = puzzles[chosen_pos]
    print(f"[intermediate] rendering puzzle (pool_pos={chosen_pos}, loader_idx={loader_idx})")

    html = render(loader_idx, batch, ablator, args.max_steps, pel)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"[intermediate] wrote {args.output}")


if __name__ == "__main__":
    main()
