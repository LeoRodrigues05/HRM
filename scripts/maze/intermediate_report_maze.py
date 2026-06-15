#!/usr/bin/env python3
"""Maze intermediate-solution report: every ACT step for a sample of puzzles.

Runs the HRM-Maze model and renders, for each sampled puzzle, the model's
predicted grid at *every* ACT step (up to 16) next to the input and the
ground-truth optimal path. Samples 5 puzzles the model solves **correctly**
(final-step exact match) and 5 it solves **incorrectly**, with a CORRECT /
INCORRECT badge and per-step path-validity metrics so you can watch the route
form (or fail to).

Output: a single self-contained HTML file.

Usage
  python scripts/maze/intermediate_report_maze.py --device cuda \
      --output results/maze/reports/intermediate_steps_5correct_5incorrect.html
  # CPU (slower): drop --device or pass --device cpu
"""
from __future__ import annotations
import os, sys, argparse
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from scripts.controlled.controlled_common import load_model_and_dataloader, collect_puzzles
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.maze.maze_common import MAZE_CHECKPOINT, get_puzzle_emb_len, maze_prediction_metrics
from scripts.maze.maze_render_common import grid_html, grid_classes, html_doc, LEGEND, metrics_table
from scripts.maze.linear_probes_maze import _flat, _slice_preds

METRIC_KEYS = ["exact_solved", "valid_sg_path", "connects_start_goal", "path_f1", "token_acc"]


def render_puzzle(idx, batch, ablator, max_steps):
    cache: dict = {}
    ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
    inp = _flat(batch["inputs"])
    label = _flat(batch["labels"])
    steps = sorted(cache.keys())

    final = _slice_preds(_flat(cache[steps[-1]].preds), label.size)
    fm = maze_prediction_metrics(final, label, inp)
    correct = bool(fm["exact_solved"])
    badge_cls = "start" if correct else "goal"
    badge = "CORRECT" if correct else "INCORRECT (solved wrong)"

    body = [f"<h2>Puzzle {idx} &nbsp; <span class='{badge_cls}' "
            f"style='padding:2px 10px'>{badge}</span></h2>"]
    body.append(metrics_table([{k: round(float(fm[k]), 4) for k in METRIC_KEYS}], METRIC_KEYS))
    body.append("<div class='row2'>")
    body.append(grid_html("Input (S/G/walls)", inp, grid_classes(inp)))
    body.append(grid_html("Label (optimal path)", label, grid_classes(label)))
    body.append("</div>")

    body.append(f"<h3>Per-ACT-step prediction ({len(steps)} steps; red=wrong cell, "
                "pink=missing path cell, striped=changed vs previous step)</h3>")
    body.append("<div class='steps'>")
    prev = None
    for s in steps:
        preds = _slice_preds(_flat(cache[s].preds), label.size)
        cls = grid_classes(preds, label_ids=label, prev_ids=prev)
        m = maze_prediction_metrics(preds, label, inp)
        title = (f"step {s} &nbsp; exact={int(m['exact_solved'])} "
                 f"sgpath={int(m['valid_sg_path'])} f1={m['path_f1']:.2f}")
        body.append(grid_html(title, preds, cls, cell_size_px=9))
        prev = preds
    body.append("</div>")
    return "".join(body), correct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=MAZE_CHECKPOINT)
    p.add_argument("--n_correct", type=int, default=5)
    p.add_argument("--n_incorrect", type=int, default=5)
    p.add_argument("--max_scan", type=int, default=250,
                   help="Max puzzles to scan to find the requested correct/incorrect counts.")
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/maze/reports/intermediate_steps_5correct_5incorrect.html")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[intermediate] device={device} scanning up to {args.max_scan} puzzles "
          f"for {args.n_correct} correct + {args.n_incorrect} incorrect")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    ablator = ActivationAblator(model, device=device)
    puzzles = collect_puzzles(test_loader, device, args.max_scan, seed=args.seed)

    correct, incorrect = [], []
    for n, (idx, batch) in enumerate(puzzles):
        if len(correct) >= args.n_correct and len(incorrect) >= args.n_incorrect:
            break
        html_p, is_ok = render_puzzle(idx, batch, ablator, args.max_steps)
        if is_ok and len(correct) < args.n_correct:
            correct.append(html_p)
        elif (not is_ok) and len(incorrect) < args.n_incorrect:
            incorrect.append(html_p)
        if (n + 1) % 20 == 0:
            print(f"[intermediate] scanned {n+1} | correct={len(correct)} incorrect={len(incorrect)}")

    head = [f"<h1>Maze — Intermediate ACT-step Solutions</h1>",
            f"<div class='meta'>checkpoint={args.checkpoint} | max_steps={args.max_steps} | "
            f"found {len(correct)} correct + {len(incorrect)} incorrect | "
            f"generated {datetime.now():%Y-%m-%d %H:%M}</div>", LEGEND]
    body = head + [f"<h1 style='color:#1a7f37'>✓ Correctly solved ({len(correct)})</h1>"] + correct
    body += [f"<h1 style='color:#cf222e'>✗ Incorrectly solved ({len(incorrect)})</h1>"] + incorrect
    with open(args.output, "w") as f:
        f.write(html_doc("Maze — Intermediate ACT-step Solutions", "".join(body)))
    print(f"[intermediate] found {len(correct)} correct + {len(incorrect)} incorrect")
    print(f"[intermediate] wrote {args.output}")
    if len(correct) < args.n_correct or len(incorrect) < args.n_incorrect:
        print(f"[intermediate] WARN: wanted {args.n_correct}+{args.n_incorrect}; "
              f"raise --max_scan if short (incorrect mazes are ~7% of the test set).")


if __name__ == "__main__":
    main()
