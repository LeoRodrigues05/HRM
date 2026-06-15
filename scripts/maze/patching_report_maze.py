"""Report 2: Activation patching side-by-side report for Maze.

Picks two test puzzles (source = donor of activations, target = recipient).
Runs:
  - baseline source forward pass (cache all step activations)
  - baseline target forward pass (cache all step preds)
  - patched target run: at the given step(s), replace target's z_H (and/or z_L)
    with source's z_H at the same step(s), then continue ACT to step max-1.

Renders an HTML report showing:
  1) Source maze (input + label + final prediction)
  2) Target maze (input + label + final prediction baseline vs patched)
  3) Per-ACT-step side-by-side: baseline-target preds vs patched-target preds
     with cells highlighted where the patched run diverged.

Usage (default = maximally disruptive: H at steps 0,1,2, full grid):
    python scripts/maze/patching_report_maze.py \
        --src_idx 0 --tgt_idx 1 \
        --patch_level H --patch_steps 0,1,2 \
        --output results/maze/reports/patching_p0_to_p1.html
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles,
)
from scripts.core.activation_patching import ActivationPatcher, ActivationCache
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, grid_predictions_accuracy,
    cell_positions, build_spatial_masks,
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


def _positions_for_group(group: str, tgt_batch, pel: int, near_dist: int) -> List[int]:
    if group == "all":
        return cell_positions(pel).tolist()
    masks = build_spatial_masks(tgt_batch, pel, near_dist=near_dist)
    return masks[group]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--src_idx", type=int, default=0,
                   help="Position in collected puzzles to use as donor (source).")
    p.add_argument("--tgt_idx", type=int, default=1,
                   help="Position in collected puzzles to use as recipient (target).")
    p.add_argument("--patch_level", choices=["H", "L", "both"], default="H")
    p.add_argument("--patch_steps", type=str, default="0,1,2")
    p.add_argument("--patch_group", choices=["all", "on_path", "off_path", "near_S", "near_G"],
                   default="all", help="Spatial subset of target cells to patch (default: all).")
    p.add_argument("--near_dist", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--num_pool", type=int, default=8)
    p.add_argument("--output", type=str,
                   default="results/maze/reports/patching_report.html")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    patch_steps = [int(x) for x in args.patch_steps.split(",") if x.strip()]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[patch_report] device={device} patch_steps={patch_steps} level={args.patch_level} group={args.patch_group}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    patcher = ActivationPatcher(model, device=device)

    puzzles = collect_puzzles(test_loader, device, max(args.num_pool, args.src_idx + 1, args.tgt_idx + 1))
    src_loader_idx, src_batch = puzzles[args.src_idx]
    tgt_loader_idx, tgt_batch = puzzles[args.tgt_idx]
    print(f"[patch_report] source pool_pos={args.src_idx} loader_idx={src_loader_idx}")
    print(f"[patch_report] target pool_pos={args.tgt_idx} loader_idx={tgt_loader_idx}")

    # Cache source activations
    src_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(src_batch, src_cache, max_steps=args.max_steps)

    # Baseline target run
    base_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(tgt_batch, base_cache, max_steps=args.max_steps)

    # Patched target run
    positions = _positions_for_group(args.patch_group, tgt_batch, pel, args.near_dist)
    print(f"[patch_report] patching {len(positions)} positions at steps {patch_steps}")
    _, patched_cache, _patch_valid = patcher.run_with_patching(
        tgt_batch, src_cache,
        patch_level=args.patch_level,
        patch_steps=patch_steps,
        patch_positions=positions,
        max_steps=args.max_steps,
    )

    src_label = _flat(src_batch["labels"]); src_input = _flat(src_batch["inputs"])
    tgt_label = _flat(tgt_batch["labels"]); tgt_input = _flat(tgt_batch["inputs"])

    base_steps = sorted(base_cache.keys())
    patched_steps_keys = sorted(patched_cache.keys())

    src_final = _slice_preds(_flat(src_cache[sorted(src_cache.keys())[-1]].preds), src_label.size)
    base_final = _slice_preds(_flat(base_cache[base_steps[-1]].preds), tgt_label.size)
    patched_final = _slice_preds(_flat(patched_cache[patched_steps_keys[-1]].preds), tgt_label.size)

    src_m = grid_predictions_accuracy(src_final, src_label)
    base_m = grid_predictions_accuracy(base_final, tgt_label)
    patch_m = grid_predictions_accuracy(patched_final, tgt_label)

    body: List[str] = []
    body.append("<h1>Maze — Activation Patching Report</h1>")
    body.append(
        f"<div class='meta'>"
        f"src=puzzle {src_loader_idx} &nbsp;|&nbsp; tgt=puzzle {tgt_loader_idx} &nbsp;|&nbsp; "
        f"patch_level={args.patch_level} &nbsp;|&nbsp; patch_steps={patch_steps} &nbsp;|&nbsp; "
        f"group={args.patch_group} (n_positions={len(positions)}) &nbsp;|&nbsp; "
        f"max_steps={args.max_steps} &nbsp;|&nbsp; "
        f"generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>"
    )
    body.append(LEGEND)

    # Summary table
    body.append("<h2>Final-prediction accuracy</h2>")
    body.append(metrics_table([
        {"run": "source (donor) baseline", "token_acc": src_m["token_acc"],
         "path_cell_acc": src_m["path_cell_acc"], "exact_solved": int(src_m["exact_solved"])},
        {"run": "target baseline (unpatched)", "token_acc": base_m["token_acc"],
         "path_cell_acc": base_m["path_cell_acc"], "exact_solved": int(base_m["exact_solved"])},
        {"run": "target patched", "token_acc": patch_m["token_acc"],
         "path_cell_acc": patch_m["path_cell_acc"], "exact_solved": int(patch_m["exact_solved"])},
        {"run": "Δ patched - baseline", "token_acc": patch_m["token_acc"] - base_m["token_acc"],
         "path_cell_acc": patch_m["path_cell_acc"] - base_m["path_cell_acc"],
         "exact_solved": int(patch_m["exact_solved"]) - int(base_m["exact_solved"])},
    ], ["run", "token_acc", "path_cell_acc", "exact_solved"]))

    # Source puzzle
    body.append("<h2>Source (donor) puzzle</h2><div class='row3'>")
    body.append(grid_html("Source input", src_input))
    body.append(grid_html("Source label", src_label))
    body.append(grid_html("Source final pred", src_final, grid_classes(src_final, label_ids=src_label)))
    body.append("</div>")

    # Target puzzle
    body.append("<h2>Target (recipient) puzzle</h2><div class='row3'>")
    body.append(grid_html("Target input", tgt_input))
    body.append(grid_html("Target label", tgt_label))
    body.append(grid_html(
        "Target baseline final (errors red)", base_final,
        grid_classes(base_final, label_ids=tgt_label),
    ))
    body.append("</div>")

    # Final side-by-side
    body.append("<h2>Final: baseline target vs patched target</h2>")
    body.append("<div class='row2'>")
    body.append(grid_html(
        "Baseline target (unpatched)", base_final,
        grid_classes(base_final, label_ids=tgt_label),
    ))
    body.append(grid_html(
        "Patched target (diff vs baseline highlighted)", patched_final,
        grid_classes(patched_final, label_ids=tgt_label, prev_ids=base_final),
    ))
    body.append("</div>")

    # Per-step side-by-side
    body.append("<h2>Per-ACT-step: baseline vs patched recipient</h2>")
    body.append(
        "<div class='meta'>Left tile = baseline target after step s. "
        "Right tile = patched target after step s, with cells changed vs the "
        "baseline at the same step highlighted (yellow diagonal). "
        "Red cells disagree with the ground-truth label.</div>"
    )

    rows = []
    for s in base_steps:
        base_p = _slice_preds(_flat(base_cache[s].preds), tgt_label.size)
        patched_p = _slice_preds(_flat(patched_cache[s].preds), tgt_label.size)
        bm = grid_predictions_accuracy(base_p, tgt_label)
        pm = grid_predictions_accuracy(patched_p, tgt_label)
        is_patched_step = s in patch_steps
        flag = " ⚡PATCHED" if is_patched_step else ""
        body.append(f"<h3>Step {s}{flag} &nbsp; base tok={bm['token_acc']:.3f} → patched tok={pm['token_acc']:.3f}</h3>")
        body.append("<div class='row2'>")
        body.append(grid_html(
            f"Baseline step {s}", base_p,
            grid_classes(base_p, label_ids=tgt_label), cell_size_px=11,
        ))
        body.append(grid_html(
            f"Patched step {s} (diff vs baseline)", patched_p,
            grid_classes(patched_p, label_ids=tgt_label, prev_ids=base_p),
            cell_size_px=11,
        ))
        body.append("</div>")
        n_diff = int((base_p != patched_p).sum())
        rows.append({
            "step": s, "patched_here": "yes" if is_patched_step else "",
            "base_token_acc": bm["token_acc"], "patched_token_acc": pm["token_acc"],
            "Δ_token_acc": pm["token_acc"] - bm["token_acc"],
            "n_cells_diff": n_diff,
        })

    body.append("<h2>Per-step summary</h2>")
    body.append(metrics_table(rows, [
        "step", "patched_here", "base_token_acc", "patched_token_acc",
        "Δ_token_acc", "n_cells_diff",
    ]))

    html = html_doc("Maze — Activation Patching Report", "".join(body))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"[patch_report] wrote {args.output}")


if __name__ == "__main__":
    main()
