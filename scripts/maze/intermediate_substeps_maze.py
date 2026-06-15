"""Report 4: Intermediate z_L / z_H sub-states for a Maze test puzzle.

For every ACT step, captures all H_cycles*L_cycles intermediate z_L tensors
and H_cycles intermediate z_H tensors via forward hooks, decodes each via the
LM head, and renders them side-by-side as small maze grids.

IMPORTANT: the LM head was trained to decode z_H only. Decoded z_L grids are
an OFF-DISTRIBUTION projection — use them to track when path/grid structure
emerges in the working memory, not as a correctness measurement.

Usage:
    python scripts/maze/intermediate_substeps_maze.py --random --seed 7 \
        --output results/maze/reports/intermediate_substeps.html
"""
from __future__ import annotations

import os
import sys
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
from scripts.maze.substep_capture import (
    SubStepRecorder, decode_via_lm_head, sublabels,
)


CAVEAT_HTML = (
    "<div class='meta' style='border-left:4px solid #cf222e;padding:6px 10px;"
    "background:#fff5f5;color:#222;'>"
    "<b>Caveat — how z_L is decoded.</b> The maze LM head is trained to read "
    "from <code>z_H</code> only (<code>output = lm_head(z_H)</code>). To "
    "visualize <code>z_L</code> as a grid we apply the same LM head to "
    "<code>z_L</code>; this is an <i>off-distribution projection</i> useful "
    "for seeing whether path/grid structure has emerged in the working "
    "memory, but it is <b>not</b> a correctness measurement. Only the "
    "<code>z_H grad (final)</code> column is the actual model output for that "
    "ACT step."
    "</div>"
)


def _flat(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 2:
        t = t[0]
    return t.detach().to("cpu").to(torch.int64).numpy()


def _slice_preds(preds: np.ndarray, label_len: int) -> np.ndarray:
    if preds.size > label_len:
        return preds[-label_len:]
    return preds


def _puzzle_exact_solved(batch, model, ablator: ActivationAblator,
                         max_steps: int) -> bool:
    """Run a forward pass and report whether the maze is exactly solved.

    Used to pick a genuinely correct vs incorrect puzzle for the success /
    failed reports instead of relying on hard-coded puzzle indices.
    """
    cache: Dict[int, ActivationCache] = {}
    ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
    last = sorted(cache.keys())[-1]
    label = _flat(batch["labels"])
    preds = _slice_preds(_flat(cache[last].preds), label.size)
    return bool(grid_predictions_accuracy(preds, label)["exact_solved"])


def render(puzzle_idx: int, batch, model, ablator: ActivationAblator,
           max_steps: int, puzzle_emb_len: int) -> str:
    H_cycles = int(model.config.H_cycles)
    L_cycles = int(model.config.L_cycles)
    L_per = H_cycles * L_cycles
    H_per = H_cycles

    cache: Dict[int, ActivationCache] = {}
    with SubStepRecorder(model) as rec:
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
        steps = sorted(cache.keys())
        groups = rec.take_groups(len(steps), H_cycles, L_cycles)

    label = _flat(batch["labels"])
    inp = _flat(batch["inputs"])
    final_preds = _slice_preds(_flat(cache[steps[-1]].preds), label.size)
    final_m = grid_predictions_accuracy(final_preds, label)

    l_labels, h_labels = sublabels(H_cycles, L_cycles)

    body: List[str] = []
    body.append("<h1>Maze — Intermediate sub-step activations (z_L / z_H)</h1>")
    body.append(
        f"<div class='meta'>Puzzle idx: <b>{puzzle_idx}</b> &nbsp;|&nbsp; "
        f"max_steps={max_steps} &nbsp;|&nbsp; H_cycles={H_cycles} &nbsp;|&nbsp; "
        f"L_cycles={L_cycles} &nbsp;|&nbsp; per ACT step: {L_per} z_L states "
        f"+ {H_per} z_H states &nbsp;|&nbsp; total: {len(steps)*L_per} z_L / "
        f"{len(steps)*H_per} z_H states &nbsp;|&nbsp; final token_acc="
        f"{final_m['token_acc']:.4f}, exact_solved={int(final_m['exact_solved'])} "
        f"&nbsp;|&nbsp; {datetime.now():%Y-%m-%d %H:%M:%S}</div>"
    )
    body.append(CAVEAT_HTML)
    body.append(LEGEND)

    body.append("<h2>Puzzle context</h2><div class='row3'>")
    body.append(grid_html("Input (S/G/walls)", inp))
    body.append(grid_html("Label (optimal path)", label))
    body.append(grid_html("Final prediction (errors highlighted)", final_preds,
                          grid_classes(final_preds, label_ids=label)))
    body.append("</div>")

    body.append("<h2>Per ACT-step sub-state trajectory</h2>")
    body.append(
        "<div class='meta'>For each ACT step we show the "
        f"{L_per} z_L sub-states (decoded by the LM head as a projection) "
        f"followed by the {H_per} z_H sub-states (LM head trained on these). "
        "The right-most z_H column is what the model actually outputs for "
        "that ACT step.</div>"
    )

    # Track previous final-z_H preds for change highlighting between ACT steps
    prev_final_zH: Optional[np.ndarray] = None
    per_step_summary: List[Dict[str, object]] = []
    cell_px = 8

    for s, grp in zip(steps, groups):
        # Decode all sub-states
        l_preds = [_slice_preds(_flat(decode_via_lm_head(z, model, puzzle_emb_len)),
                                label.size) for z in grp.z_L_substeps]
        h_preds = [_slice_preds(_flat(decode_via_lm_head(z, model, puzzle_emb_len)),
                                label.size) for z in grp.z_H_substeps]
        # Official z_H final from cache (same as h_preds[-1] but trustworthy):
        cache_pred = _slice_preds(_flat(cache[s].preds), label.size)

        body.append(f"<h3>ACT step {s}</h3>")
        body.append("<div class='steps'>")
        # z_L sub-states
        for lbl, p in zip(l_labels, l_preds):
            m = grid_predictions_accuracy(p, label)
            title = f"{lbl}<br>tok={m['token_acc']:.3f} path={m['path_cell_acc']:.3f}"
            body.append(grid_html(title, p, grid_classes(p, label_ids=label),
                                  cell_size_px=cell_px))
        # spacer (separator)
        body.append("<div style='width:8px;border-left:2px dashed #888;'></div>")
        # z_H sub-states
        for lbl, p in zip(h_labels, h_preds):
            m = grid_predictions_accuracy(p, label)
            extra_cls = grid_classes(
                p, label_ids=label,
                prev_ids=prev_final_zH if lbl.endswith("(final)") else None,
            )
            title = f"{lbl}<br>tok={m['token_acc']:.3f} path={m['path_cell_acc']:.3f}"
            body.append(grid_html(title, p, extra_cls, cell_size_px=cell_px))
        body.append("</div>")

        # Sanity: cache_pred should equal h_preds[-1]
        agree = bool(np.array_equal(cache_pred, h_preds[-1])) if h_preds else False
        m_final = grid_predictions_accuracy(h_preds[-1] if h_preds else cache_pred, label)
        per_step_summary.append({
            "step": s,
            "z_L final tok": grid_predictions_accuracy(l_preds[-1], label)["token_acc"]
                              if l_preds else 0.0,
            "z_H final tok": m_final["token_acc"],
            "z_H final path": m_final["path_cell_acc"],
            "exact_solved": int(m_final["exact_solved"]),
            "lm_head==cache": "yes" if agree else "no",
        })
        prev_final_zH = h_preds[-1] if h_preds else prev_final_zH

    body.append("<h2>Per-step summary</h2>")
    body.append(metrics_table(per_step_summary,
        ["step", "z_L final tok", "z_H final tok", "z_H final path",
         "exact_solved", "lm_head==cache"]))

    return html_doc("Maze — Intermediate sub-states", "".join(body))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--puzzle_idx", type=int, default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--num_pool", type=int, default=64)
    p.add_argument("--require", type=str, default="any",
                   choices=["any", "solved", "failed"],
                   help="Scan the pool and pick the first puzzle whose final "
                        "prediction is exactly solved / not solved. Overrides "
                        "--puzzle_idx / --random selection.")
    p.add_argument("--output", type=str,
                   default="results/maze/reports/intermediate_substeps.html")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[intermediate_substeps] device={device}")
    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    ablator = ActivationAblator(model, device=device)

    puzzles = collect_puzzles(test_loader, device, args.num_pool)
    print(f"[intermediate_substeps] collected {len(puzzles)} candidate puzzles")
    if args.require != "any":
        want_solved = args.require == "solved"
        chosen_pos = None
        for i, (idx, batch) in enumerate(puzzles):
            if _puzzle_exact_solved(batch, model, ablator, args.max_steps) == want_solved:
                chosen_pos = i
                print(f"[intermediate_substeps] require={args.require}: "
                      f"selected pool_pos={i} loader_idx={idx}")
                break
        if chosen_pos is None:
            raise SystemExit(
                f"[intermediate_substeps] no puzzle matching require={args.require} "
                f"in pool of {len(puzzles)}; increase --num_pool."
            )
    elif args.puzzle_idx is None or args.random:
        chosen_pos = rng.randrange(len(puzzles))
    else:
        chosen_pos = next((i for i, (idx, _) in enumerate(puzzles) if idx == args.puzzle_idx),
                          args.puzzle_idx if args.puzzle_idx < len(puzzles) else 0)
    loader_idx, batch = puzzles[chosen_pos]
    print(f"[intermediate_substeps] rendering puzzle pool_pos={chosen_pos} loader_idx={loader_idx}")

    html = render(loader_idx, batch, model, ablator, args.max_steps, pel)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"[intermediate_substeps] wrote {args.output}")


if __name__ == "__main__":
    main()
