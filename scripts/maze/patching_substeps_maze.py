"""Report 5: Activation patching with z_L sub-step visualization.

Same flow as `patching_report_maze.py` but at each ACT step we render the
H_cycles*L_cycles intermediate z_L tensors (decoded via the LM head) for both
the baseline-target and patched-target runs, side-by-side.

CAVEAT: z_L decoded via the LM head is an OFF-DISTRIBUTION projection — the
LM head was trained on z_H only. Treat decoded z_L grids as a qualitative
view of the working memory's contents, not a correctness measurement.

Usage (default = maximally disruptive: H+L at step 1, full grid):
    python scripts/maze/patching_substeps_maze.py \
        --src_idx 0 --tgt_idx 1 \
        --patch_level both --patch_steps 1 \
        --output results/maze/reports/patching_substeps.html
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional

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
from scripts.maze.substep_capture import (
    SubStepRecorder, decode_via_lm_head, sublabels,
)


CAVEAT_HTML = (
    "<div class='meta' style='border-left:4px solid #cf222e;padding:6px 10px;"
    "background:#fff5f5;color:#222;'>"
    "<b>Caveat — how z_L is decoded.</b> The maze LM head is trained on "
    "<code>z_H</code> only. To visualize <code>z_L</code> we apply the same "
    "LM head; this is an off-distribution projection useful for tracking "
    "whether/where the working memory carries puzzle-specific structure, but "
    "it is <b>not</b> a correctness measurement."
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


def _positions_for_group(group: str, tgt_batch, pel: int, near_dist: int) -> List[int]:
    if group == "all":
        return cell_positions(pel).tolist()
    masks = build_spatial_masks(tgt_batch, pel, near_dist=near_dist)
    return masks[group]


def _run_with_substeps(patcher: ActivationPatcher, model, batch, *,
                       patched: bool = False, src_cache=None,
                       patch_level=None, patch_steps=None, patch_positions=None,
                       max_steps: int = 16):
    cache: Dict[int, ActivationCache] = {}
    with SubStepRecorder(model) as rec:
        if patched:
            _, cache, _ = patcher.run_with_patching(
                batch, src_cache,
                patch_level=patch_level,
                patch_steps=patch_steps,
                patch_positions=patch_positions,
                max_steps=max_steps,
            )
        else:
            patcher.run_and_cache_activations(batch, cache, max_steps=max_steps)
        steps = sorted(cache.keys())
        groups = rec.take_groups(len(steps),
                                 int(model.config.H_cycles),
                                 int(model.config.L_cycles))
    return cache, steps, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--src_idx", type=int, default=0)
    p.add_argument("--tgt_idx", type=int, default=1)
    p.add_argument("--patch_level", choices=["H", "L", "both"], default="both")
    p.add_argument("--patch_steps", type=str, default="1")
    p.add_argument("--patch_group", choices=["all", "on_path", "off_path", "near_S", "near_G"],
                   default="all")
    p.add_argument("--near_dist", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--num_pool", type=int, default=8)
    p.add_argument("--show_zH", action="store_true",
                   help="Also render z_H sub-states (default: z_L only).")
    p.add_argument("--cell_px", type=int, default=7,
                   help="Cell size in pixels for sub-step thumbnails.")
    p.add_argument("--output", type=str,
                   default="results/maze/reports/patching_substeps.html")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    patch_steps = [int(x) for x in args.patch_steps.split(",") if x.strip()]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[patch_substeps] device={device} patch_steps={patch_steps} "
          f"level={args.patch_level} group={args.patch_group}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    patcher = ActivationPatcher(model, device=device)

    puzzles = collect_puzzles(test_loader, device,
                              max(args.num_pool, args.src_idx + 1, args.tgt_idx + 1))
    src_loader_idx, src_batch = puzzles[args.src_idx]
    tgt_loader_idx, tgt_batch = puzzles[args.tgt_idx]
    print(f"[patch_substeps] source pool_pos={args.src_idx} loader_idx={src_loader_idx}")
    print(f"[patch_substeps] target pool_pos={args.tgt_idx} loader_idx={tgt_loader_idx}")

    # 1. Source baseline (we just need its cache for patching donor states)
    src_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(src_batch, src_cache, max_steps=args.max_steps)

    # 2. Baseline target with sub-step capture
    base_cache, base_steps, base_groups = _run_with_substeps(
        patcher, model, tgt_batch, patched=False, max_steps=args.max_steps,
    )

    # 3. Patched target with sub-step capture
    positions = _positions_for_group(args.patch_group, tgt_batch, pel, args.near_dist)
    print(f"[patch_substeps] patching {len(positions)} positions at steps {patch_steps}")
    patched_cache, patched_steps, patched_groups = _run_with_substeps(
        patcher, model, tgt_batch, patched=True, src_cache=src_cache,
        patch_level=args.patch_level, patch_steps=patch_steps,
        patch_positions=positions, max_steps=args.max_steps,
    )

    H_cycles = int(model.config.H_cycles); L_cycles = int(model.config.L_cycles)
    l_labels, h_labels = sublabels(H_cycles, L_cycles)
    L_per = H_cycles * L_cycles

    tgt_label = _flat(tgt_batch["labels"]); tgt_input = _flat(tgt_batch["inputs"])
    src_label = _flat(src_batch["labels"]); src_input = _flat(src_batch["inputs"])
    src_final = _slice_preds(_flat(src_cache[sorted(src_cache.keys())[-1]].preds), src_label.size)
    base_final = _slice_preds(_flat(base_cache[base_steps[-1]].preds), tgt_label.size)
    patched_final = _slice_preds(_flat(patched_cache[patched_steps[-1]].preds), tgt_label.size)

    src_m = grid_predictions_accuracy(src_final, src_label)
    base_m = grid_predictions_accuracy(base_final, tgt_label)
    patch_m = grid_predictions_accuracy(patched_final, tgt_label)

    body: List[str] = []
    body.append("<h1>Maze — Activation Patching (z_L sub-states)</h1>")
    body.append(
        f"<div class='meta'>src=puzzle {src_loader_idx} &nbsp;|&nbsp; "
        f"tgt=puzzle {tgt_loader_idx} &nbsp;|&nbsp; patch_level={args.patch_level} "
        f"&nbsp;|&nbsp; patch_steps={patch_steps} &nbsp;|&nbsp; "
        f"group={args.patch_group} (n_positions={len(positions)}) &nbsp;|&nbsp; "
        f"H={H_cycles} L={L_cycles} → {L_per} z_L sub-states per ACT step "
        f"&nbsp;|&nbsp; {datetime.now():%Y-%m-%d %H:%M:%S}</div>"
    )
    body.append(CAVEAT_HTML)
    body.append(LEGEND)

    # Summary
    body.append("<h2>Final-prediction accuracy</h2>")
    body.append(metrics_table([
        {"run": "source (donor) baseline", "token_acc": src_m["token_acc"],
         "path_cell_acc": src_m["path_cell_acc"], "exact_solved": int(src_m["exact_solved"])},
        {"run": "target baseline", "token_acc": base_m["token_acc"],
         "path_cell_acc": base_m["path_cell_acc"], "exact_solved": int(base_m["exact_solved"])},
        {"run": "target patched", "token_acc": patch_m["token_acc"],
         "path_cell_acc": patch_m["path_cell_acc"], "exact_solved": int(patch_m["exact_solved"])},
        {"run": "Δ patched - baseline", "token_acc": patch_m["token_acc"] - base_m["token_acc"],
         "path_cell_acc": patch_m["path_cell_acc"] - base_m["path_cell_acc"],
         "exact_solved": int(patch_m["exact_solved"]) - int(base_m["exact_solved"])},
    ], ["run", "token_acc", "path_cell_acc", "exact_solved"]))

    # Donor + recipient context
    body.append("<h2>Source (donor) puzzle</h2><div class='row3'>")
    body.append(grid_html("Source input", src_input))
    body.append(grid_html("Source label", src_label))
    body.append(grid_html("Source final pred", src_final, grid_classes(src_final, label_ids=src_label)))
    body.append("</div>")
    body.append("<h2>Target (recipient) puzzle</h2><div class='row3'>")
    body.append(grid_html("Target input", tgt_input))
    body.append(grid_html("Target label", tgt_label))
    body.append(grid_html("Target baseline final", base_final,
                          grid_classes(base_final, label_ids=tgt_label)))
    body.append("</div>")

    body.append("<h2>Final: baseline vs patched target</h2><div class='row2'>")
    body.append(grid_html("Baseline target", base_final,
                          grid_classes(base_final, label_ids=tgt_label)))
    body.append(grid_html("Patched target (diff vs baseline)", patched_final,
                          grid_classes(patched_final, label_ids=tgt_label, prev_ids=base_final)))
    body.append("</div>")

    # Per-step z_L sub-states: baseline (top row) vs patched (bottom row)
    body.append("<h2>Per ACT-step z_L sub-states: baseline (top) vs patched (bottom)</h2>")
    body.append(
        "<div class='meta'>Each row shows the "
        f"{L_per} z_L sub-states (decoded via LM head) at that ACT step. "
        "In the patched row, cells changed vs the baseline at the same "
        "sub-state are highlighted (yellow diagonal). The "
        "<code>⚡PATCHED</code> tag marks ACT steps where the carry was "
        "overridden at the post-reset boundary.</div>"
    )

    per_step_rows = []
    for s in base_steps:
        bgrp = base_groups[s]
        pgrp = patched_groups[s] if s < len(patched_groups) else base_groups[s]
        is_patched_step = s in patch_steps
        flag = " ⚡PATCHED" if is_patched_step else ""

        # Decode sub-states
        base_l = [_slice_preds(_flat(decode_via_lm_head(z, model, pel)), tgt_label.size)
                  for z in bgrp.z_L_substeps]
        patch_l = [_slice_preds(_flat(decode_via_lm_head(z, model, pel)), tgt_label.size)
                   for z in pgrp.z_L_substeps]

        body.append(f"<h3>ACT step {s}{flag}</h3>")
        # Baseline row
        body.append("<div class='steps'>")
        body.append(f"<div style='align-self:center;font-weight:700;color:#444;"
                    f"writing-mode:vertical-rl;transform:rotate(180deg);'>BASELINE</div>")
        for lbl, p in zip(l_labels, base_l):
            m = grid_predictions_accuracy(p, tgt_label)
            title = f"{lbl}<br>tok={m['token_acc']:.3f}"
            body.append(grid_html(title, p, grid_classes(p, label_ids=tgt_label),
                                  cell_size_px=args.cell_px))
        body.append("</div>")
        # Patched row
        body.append("<div class='steps'>")
        body.append(f"<div style='align-self:center;font-weight:700;color:#cf222e;"
                    f"writing-mode:vertical-rl;transform:rotate(180deg);'>PATCHED</div>")
        for lbl, bp, pp in zip(l_labels, base_l, patch_l):
            m = grid_predictions_accuracy(pp, tgt_label)
            n_diff = int((bp != pp).sum())
            title = f"{lbl}<br>tok={m['token_acc']:.3f} Δ={n_diff}"
            body.append(grid_html(title, pp,
                                  grid_classes(pp, label_ids=tgt_label, prev_ids=bp),
                                  cell_size_px=args.cell_px))
        body.append("</div>")

        # Optional z_H row
        if args.show_zH:
            base_h = [_slice_preds(_flat(decode_via_lm_head(z, model, pel)), tgt_label.size)
                      for z in bgrp.z_H_substeps]
            patch_h = [_slice_preds(_flat(decode_via_lm_head(z, model, pel)), tgt_label.size)
                       for z in pgrp.z_H_substeps]
            body.append("<div class='steps'>")
            body.append("<div style='align-self:center;font-weight:700;color:#666;"
                        "writing-mode:vertical-rl;transform:rotate(180deg);'>z_H</div>")
            for lbl, bp, pp in zip(h_labels, base_h, patch_h):
                m = grid_predictions_accuracy(pp, tgt_label)
                n_diff = int((bp != pp).sum())
                title = f"{lbl}<br>base→patched tok={m['token_acc']:.3f} Δ={n_diff}"
                body.append(grid_html(title, pp,
                                      grid_classes(pp, label_ids=tgt_label, prev_ids=bp),
                                      cell_size_px=args.cell_px))
            body.append("</div>")

        # Numeric per-substate diff counts
        per_step_rows.append({
            "step": s, "patched_here": "yes" if is_patched_step else "",
            **{f"zL[{i}] Δcells": int((base_l[i] != patch_l[i]).sum())
               for i in range(len(base_l))},
        })

    body.append("<h2>Per-step z_L diff (cells differing between baseline and patched)</h2>")
    sub_headers = ["step", "patched_here"] + [f"zL[{i}] Δcells" for i in range(L_per)]
    body.append(metrics_table(per_step_rows, sub_headers))

    html = html_doc("Maze — Patching z_L sub-states", "".join(body))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"[patch_substeps] wrote {args.output}")


if __name__ == "__main__":
    main()
