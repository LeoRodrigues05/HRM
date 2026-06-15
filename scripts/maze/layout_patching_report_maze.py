"""Maze layout-drift report for activation patching.

This diagnostic tests the Sudoku-style question: when z_H from a source puzzle
is patched into a target puzzle, does the target output start copying the
source puzzle layout?  For Maze, layout means wall/start/goal tokens. Path
tokens are not considered layout because they are the model's solution.

Default run:
    python scripts/maze/layout_patching_report_maze.py \
        --src_idx 2 --tgt_idx 3 --patch_level H --patch_steps 1 \
        --output results/maze/reports/layout_patching_p2_to_p3_H_step1.html
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
from scripts.core.activation_patching import ActivationPatcher, ActivationCache
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, cell_positions, build_spatial_masks,
    maze_prediction_metrics, maze_layout_metrics,
)
from scripts.maze.maze_render_common import (
    grid_html, grid_classes, html_doc, LEGEND, metrics_table,
)
from utils.maze_targets import WALL_ID, START_ID, GOAL_ID


EXTRA_CSS = """
<style>
td.layout_disagreement { box-shadow: inset 0 0 0 2px #8250df; }
td.source_imprint { box-shadow: inset 0 0 0 2px #0969da; }
td.target_erased { box-shadow: inset 0 0 0 2px #bf8700; }
td.layout_changed { outline: 2px solid #cf222e; outline-offset: -2px; }
.miniLegend span { display:inline-block; padding:2px 8px; margin-right:6px;
                   border:1px solid #999; font-size:0.85em; }
.miniLegend .layout_disagreement { box-shadow: inset 0 0 0 2px #8250df; }
.miniLegend .source_imprint { box-shadow: inset 0 0 0 2px #0969da; }
.miniLegend .target_erased { box-shadow: inset 0 0 0 2px #bf8700; }
.miniLegend .layout_changed { outline: 2px solid #cf222e; outline-offset: -2px; }
</style>
"""

EXTRA_LEGEND = """
<div class='miniLegend'>
  <span class='layout_disagreement'>source/target layout differ</span>
  <span class='source_imprint'>patched matches source layout</span>
  <span class='target_erased'>target wall/endpoint erased</span>
  <span class='layout_changed'>layout token changed vs baseline</span>
</div>
"""


def _flat(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 2:
        t = t[0]
    return t.detach().to("cpu").to(torch.int64).numpy().reshape(-1)


def _slice_preds(preds: np.ndarray, label_len: int) -> np.ndarray:
    preds = preds.reshape(-1)
    if preds.size > label_len:
        return preds[-label_len:]
    return preds


def _layout_mask(x: np.ndarray) -> np.ndarray:
    return (x == WALL_ID) | (x == START_ID) | (x == GOAL_ID)


def _positions_for_group(group: str, tgt_batch, pel: int, near_dist: int) -> List[int]:
    if group == "all":
        return cell_positions(pel).tolist()
    masks = build_spatial_masks(tgt_batch, pel, near_dist=near_dist)
    return masks[group]


def _target_input_classes(target_input: np.ndarray, source_input: np.ndarray) -> List[str]:
    classes = grid_classes(target_input)
    disagree = (source_input != target_input) & (_layout_mask(source_input) | _layout_mask(target_input))
    for i, flag in enumerate(disagree):
        if flag:
            classes[i] = f"{classes[i]} layout_disagreement"
    return classes


def _patched_layout_classes(
    patched: np.ndarray,
    baseline: np.ndarray,
    target_label: np.ndarray,
    target_input: np.ndarray,
    source_input: np.ndarray,
) -> List[str]:
    classes = grid_classes(patched, label_ids=target_label, prev_ids=baseline)
    source_layout = _layout_mask(source_input)
    target_layout = _layout_mask(target_input)
    layout_disagree = (source_input != target_input) & (source_layout | target_layout)
    source_imprint = layout_disagree & (patched == source_input) & (patched != target_input)
    target_wall_erased = (target_input == WALL_ID) & (source_input != WALL_ID) & (patched != WALL_ID)
    target_endpoint_erased = (
        ((target_input == START_ID) | (target_input == GOAL_ID))
        & (source_input != target_input)
        & (patched != target_input)
    )
    layout_changed = (patched != baseline) & (source_layout | target_layout)

    for i in range(patched.size):
        extra = []
        if source_imprint[i]:
            extra.append("source_imprint")
        if target_wall_erased[i] or target_endpoint_erased[i]:
            extra.append("target_erased")
        if layout_changed[i]:
            extra.append("layout_changed")
        if extra:
            classes[i] = f"{classes[i]} {' '.join(extra)}"
    return classes


def _metric_row(prefix: str, struct: Dict[str, float], layout: Dict[str, float]) -> Dict[str, object]:
    return {
        "run": prefix,
        "exact": int(struct["exact_solved"]),
        "path_f1": struct["path_f1"],
        "valid_sg": int(struct["valid_sg_path"]),
        "target_layout_acc": layout["target_layout_acc"],
        "source_wall_imprint": layout["source_unique_wall_imprint_rate"],
        "target_wall_erasure": layout["target_unique_wall_erasure_rate"],
        "toward_source_layout": layout["toward_source_layout_rate"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--src_idx", type=int, default=2)
    p.add_argument("--tgt_idx", type=int, default=3)
    p.add_argument("--patch_level", choices=["H", "L", "both"], default="H")
    p.add_argument("--patch_steps", type=str, default="1")
    p.add_argument("--patch_group", choices=["all", "on_path", "off_path", "near_S", "near_G"],
                   default="all")
    p.add_argument("--near_dist", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--num_pool", type=int, default=8)
    p.add_argument("--output", type=str,
                   default="results/maze/reports/layout_patching_report.html")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    patch_steps = [int(x) for x in args.patch_steps.split(",") if x.strip()]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[layout_patch] device={device} level={args.patch_level} steps={patch_steps} group={args.patch_group}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    patcher = ActivationPatcher(model, device=device)

    puzzles = collect_puzzles(test_loader, device,
                              max(args.num_pool, args.src_idx + 1, args.tgt_idx + 1))
    src_loader_idx, src_batch = puzzles[args.src_idx]
    tgt_loader_idx, tgt_batch = puzzles[args.tgt_idx]
    print(f"[layout_patch] source pool_pos={args.src_idx} loader_idx={src_loader_idx}")
    print(f"[layout_patch] target pool_pos={args.tgt_idx} loader_idx={tgt_loader_idx}")

    src_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(src_batch, src_cache, max_steps=args.max_steps)

    base_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(tgt_batch, base_cache, max_steps=args.max_steps)

    positions = _positions_for_group(args.patch_group, tgt_batch, pel, args.near_dist)
    _, patched_cache, patch_valid = patcher.run_with_patching(
        tgt_batch, src_cache,
        patch_level=args.patch_level,
        patch_steps=patch_steps,
        patch_positions=positions,
        max_steps=args.max_steps,
        verify=True,
    )

    src_label = _flat(src_batch["labels"])
    src_input = _flat(src_batch["inputs"])
    tgt_label = _flat(tgt_batch["labels"])
    tgt_input = _flat(tgt_batch["inputs"])

    steps = sorted(base_cache.keys())
    src_steps = sorted(src_cache.keys())
    patched_steps_keys = sorted(patched_cache.keys())

    src_final = _slice_preds(_flat(src_cache[src_steps[-1]].preds), src_label.size)
    base_final = _slice_preds(_flat(base_cache[steps[-1]].preds), tgt_label.size)
    patched_final = _slice_preds(_flat(patched_cache[patched_steps_keys[-1]].preds), tgt_label.size)

    base_struct = maze_prediction_metrics(base_final, tgt_label, tgt_input)
    patch_struct = maze_prediction_metrics(patched_final, tgt_label, tgt_input)
    base_layout = maze_layout_metrics(base_final, tgt_input, src_input)
    patch_layout = maze_layout_metrics(patched_final, tgt_input, src_input)
    src_struct = maze_prediction_metrics(src_final, src_label, src_input)

    body: List[str] = []
    body.append(EXTRA_CSS)
    body.append("<h1>Maze - z_H patch layout-drift diagnostic</h1>")
    body.append(
        f"<div class='meta'>src=puzzle {src_loader_idx} | tgt=puzzle {tgt_loader_idx} | "
        f"patch_level={args.patch_level} | patch_steps={patch_steps} | "
        f"group={args.patch_group} (n_positions={len(positions)}) | "
        f"max_steps={args.max_steps} | generated {datetime.now():%Y-%m-%d %H:%M:%S}</div>"
    )
    body.append(
        "<div class='meta'>Layout drift is measured on wall/start/goal tokens only. "
        "A source-imprint event means the patched target output matches the source "
        "layout token at an absolute grid coordinate where source and target layouts differ.</div>"
    )
    body.append(LEGEND)
    body.append(EXTRA_LEGEND)

    body.append("<h2>Final metrics</h2>")
    rows = [
        _metric_row("source baseline", src_struct, maze_layout_metrics(src_final, src_input)),
        _metric_row("target baseline", base_struct, base_layout),
        _metric_row("target patched", patch_struct, patch_layout),
        {
            "run": "delta patched - baseline",
            "exact": int(patch_struct["exact_solved"]) - int(base_struct["exact_solved"]),
            "path_f1": patch_struct["path_f1"] - base_struct["path_f1"],
            "valid_sg": int(patch_struct["valid_sg_path"]) - int(base_struct["valid_sg_path"]),
            "target_layout_acc": patch_layout["target_layout_acc"] - base_layout["target_layout_acc"],
            "source_wall_imprint": patch_layout["source_unique_wall_imprint_rate"] - base_layout["source_unique_wall_imprint_rate"],
            "target_wall_erasure": patch_layout["target_unique_wall_erasure_rate"] - base_layout["target_unique_wall_erasure_rate"],
            "toward_source_layout": patch_layout["toward_source_layout_rate"] - base_layout["toward_source_layout_rate"],
        },
    ]
    body.append(metrics_table(rows, [
        "run", "exact", "path_f1", "valid_sg", "target_layout_acc",
        "source_wall_imprint", "target_wall_erasure", "toward_source_layout",
    ]))

    body.append("<h2>Puzzle context</h2><div class='row3'>")
    body.append(grid_html("Source input", src_input))
    body.append(grid_html("Target input (layout disagreement outlined)", tgt_input,
                          _target_input_classes(tgt_input, src_input)))
    body.append(grid_html("Target label", tgt_label))
    body.append("</div>")

    body.append("<h2>Final target outputs</h2><div class='row2'>")
    body.append(grid_html("Target baseline final", base_final,
                          grid_classes(base_final, label_ids=tgt_label)))
    body.append(grid_html("Target patched final", patched_final,
                          _patched_layout_classes(
                              patched_final, base_final, tgt_label, tgt_input, src_input,
                          )))
    body.append("</div>")

    body.append("<h2>Per-step layout diagnostics</h2>")
    body.append(
        "<div class='meta'>The patched grid is highlighted against the baseline "
        "prediction from the same ACT step. Layout metrics are computed against "
        "the target input and source input at absolute coordinates.</div>"
    )

    step_rows = []
    for s in steps:
        base_p = _slice_preds(_flat(base_cache[s].preds), tgt_label.size)
        patch_p = _slice_preds(_flat(patched_cache[s].preds), tgt_label.size)
        bs = maze_prediction_metrics(base_p, tgt_label, tgt_input)
        ps = maze_prediction_metrics(patch_p, tgt_label, tgt_input)
        bl = maze_layout_metrics(base_p, tgt_input, src_input)
        pl = maze_layout_metrics(patch_p, tgt_input, src_input)
        layout_mask = _layout_mask(tgt_input) | _layout_mask(src_input)
        changed_layout = int(((patch_p != base_p) & layout_mask).sum())
        is_patched = s in patch_steps
        body.append(
            f"<h3>Step {s}{' PATCHED' if is_patched else ''} | "
            f"path_f1 {bs['path_f1']:.3f} -> {ps['path_f1']:.3f} | "
            f"source wall imprint {bl['source_unique_wall_imprint_rate']:.3f} -> "
            f"{pl['source_unique_wall_imprint_rate']:.3f}</h3>"
        )
        body.append("<div class='row2'>")
        body.append(grid_html(f"Baseline step {s}", base_p,
                              grid_classes(base_p, label_ids=tgt_label), cell_size_px=10))
        body.append(grid_html(f"Patched step {s}", patch_p,
                              _patched_layout_classes(
                                  patch_p, base_p, tgt_label, tgt_input, src_input,
                              ),
                              cell_size_px=10))
        body.append("</div>")
        step_rows.append({
            "step": s,
            "patched_here": "yes" if is_patched else "",
            "delta_path_f1": ps["path_f1"] - bs["path_f1"],
            "target_layout_acc": pl["target_layout_acc"],
            "source_wall_imprint": pl["source_unique_wall_imprint_rate"],
            "target_wall_erasure": pl["target_unique_wall_erasure_rate"],
            "toward_source_layout": pl["toward_source_layout_rate"],
            "changed_layout_cells": changed_layout,
        })

    body.append("<h2>Step summary</h2>")
    body.append(metrics_table(step_rows, [
        "step", "patched_here", "delta_path_f1", "target_layout_acc",
        "source_wall_imprint", "target_wall_erasure", "toward_source_layout",
        "changed_layout_cells",
    ]))

    if patch_valid:
        val_rows = []
        for s, vals in sorted(patch_valid.items()):
            val_rows.append({"step": s, **vals})
        body.append("<h2>Patch verification</h2>")
        body.append(metrics_table(val_rows, [
            "step", "pre_diff_H", "post_diff_H", "pre_diff_L", "post_diff_L",
        ]))

    html = html_doc("Maze - Layout Patching Diagnostic", "".join(body))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"[layout_patch] wrote {args.output}")


if __name__ == "__main__":
    main()
