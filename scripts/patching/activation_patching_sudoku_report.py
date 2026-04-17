#!/usr/bin/env python3
"""scripts/activation_patching_sudoku_report.py

Generate a colored HTML Sudoku report for an activation patching experiment.

This mirrors the coloring scheme used by:
- sudoku_report_colored_ai.py
- result_metrics_sudoku.py (token->char convention)

It reads the patching YAML produced by scripts/activation_patching.py and renders:
- Target input (givens)
- Target baseline intermediate predictions at selected steps
- Target patched intermediate predictions at selected steps

Patched grids are colored with a diagonal split overlay on cells that changed
relative to the baseline grid at the same step.

Usage:
  python scripts/activation_patching_sudoku_report.py \
    --results_yaml results/activation_patching_e2e/patch_s0_t1_both.yaml \
    --output_html results/activation_patching_e2e/activation_patching_report.html

Notes:
- Requires access to the dataset referenced by the checkpoint config.
- Uses the target puzzle index in the YAML to fetch the matching puzzle from the test split.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from pretrain import PretrainConfig, create_dataloader


def id2num(i: int) -> str:
    """HRM Sudoku token map used across repo reports.

    - tokens 2..10 -> '1'..'9'
    - everything else (PAD=0, '0'=1, ignore=-100, etc) -> '.'
    """
    if 2 <= i <= 10:
        return str(i - 1)
    return "."


def to_chars(vec: List[int] | np.ndarray | torch.Tensor) -> List[str]:
    if torch.is_tensor(vec):
        arr = vec.detach().to("cpu").view(-1).to(torch.int64).numpy()
    else:
        arr = np.asarray(vec, dtype=np.int64).reshape(-1)
    return [id2num(int(x)) for x in arr]


def to_rows(vec81: List[str]) -> List[List[str]]:
    return [vec81[r * 9 : (r + 1) * 9] for r in range(9)]


def compute_violations(chars81: List[str]) -> List[bool]:
    """True at i if that filled cell violates row/col/box uniqueness."""
    viol = [False] * 81

    # rows
    for r in range(9):
        vals = [chars81[r * 9 + c] for c in range(9) if chars81[r * 9 + c] != "."]
        for c in range(9):
            ch = chars81[r * 9 + c]
            if ch != "." and vals.count(ch) > 1:
                viol[r * 9 + c] = True

    # cols
    for c in range(9):
        col_vals = [chars81[r * 9 + c] for r in range(9) if chars81[r * 9 + c] != "."]
        for r in range(9):
            ch = chars81[r * 9 + c]
            if ch != "." and col_vals.count(ch) > 1:
                viol[r * 9 + c] = True

    # boxes
    for br in range(3):
        for bc in range(3):
            idxs = [(br * 3 + rr) * 9 + (bc * 3 + cc) for rr in range(3) for cc in range(3)]
            box_vals = [chars81[i] for i in idxs if chars81[i] != "."]
            for i in idxs:
                ch = chars81[i]
                if ch != "." and box_vals.count(ch) > 1:
                    viol[i] = True

    return viol


def compute_classes(curr_chars: List[str], given_chars: List[str], prev_chars: Optional[List[str]] = None) -> List[str]:
    """Match sudoku_report_colored_ai.py coloring priority.

    Priority:
      changed_ok / changed_bad (half yellow) > given (blue) > ok/bad > blank

    "changed" is defined relative to prev_chars (when provided).
    """
    viol = compute_violations(curr_chars)
    classes: List[str] = []
    for i in range(81):
        ch = curr_chars[i]
        given = (given_chars[i] != ".")
        changed = (prev_chars is not None and ch != prev_chars[i] and ch != ".")

        if changed:
            cls = "changed_ok" if not viol[i] else "changed_bad"
        elif given:
            cls = "given"
        elif ch == ".":
            cls = "blank"
        else:
            cls = "ok" if not viol[i] else "bad"
        classes.append(cls)

    return classes


def table_html(title: str, arr81_chars: List[str], classes: Optional[List[str]] = None) -> str:
    rows = to_rows(arr81_chars)
    h = ["<div class='gridBlock'>", f"<div class='gridTitle'>{title}</div>", "<table class='sgrid'>"]
    for r in range(9):
        h.append("<tr>")
        for c in range(9):
            ch = rows[r][c]
            cls = classes[r * 9 + c] if classes else ""
            borders = []
            if r in (2, 5):
                borders.append("bb")
            if c in (2, 5):
                borders.append("br")
            cls = (cls + " " + " ".join(borders)).strip()
            h.append(f"<td class='{cls}'>{ch}</td>")
        h.append("</tr>")
    h.append("</table>")
    h.append("</div>")
    return "\n".join(h)


def extract_batch(item: Any) -> Dict[str, torch.Tensor]:
    """Extract batch dict from dataloader yields.

    Observed formats in this repo:
    - (set_name, batch_dict, global_batch_size)
    - [set_name, batch_dict, global_batch_size]
    - batch_dict
    """
    if isinstance(item, (tuple, list)):
        if len(item) == 3 and isinstance(item[1], dict):
            return item[1]
        if len(item) == 2 and isinstance(item[1], dict):
            return item[1]
        if len(item) == 1 and isinstance(item[0], dict):
            return item[0]
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported dataloader item type/shape: {type(item)} repr={repr(item)[:200]}")


def load_target_puzzle_from_config(config: PretrainConfig, target_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
    dl, _meta = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=1,
        rank=0,
        world_size=1,
    )

    for i, item in enumerate(dl):
        if i > target_idx:
            break
        batch = extract_batch(item)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if i == target_idx:
            return batch

    raise IndexError(f"Target puzzle idx={target_idx} not found in test split")


def main() -> None:
    ap = argparse.ArgumentParser(description="HTML report for activation patching Sudoku experiments")
    ap.add_argument("--results_yaml", required=True, help="Path to patch_*.yaml output from activation_patching.py")
    ap.add_argument("--output_html", required=True, help="Output HTML path")
    ap.add_argument("--device", default="cpu", help="Device for dataset tensors (cpu is fine)")
    ap.add_argument("--steps", default=None, help="Comma-separated steps to render. Default: steps present in YAML 'step_outputs'.")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    with open(args.results_yaml, "r") as f:
        results = yaml.safe_load(f)

    cfg = results.get("config", {})
    checkpoint = cfg.get("checkpoint")
    target_idx = int(cfg.get("target_puzzle_idx"))

    if not checkpoint:
        raise ValueError("results_yaml is missing config.checkpoint")

    config_path = os.path.join(os.path.dirname(checkpoint), "all_config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    target_batch = load_target_puzzle_from_config(config, target_idx=target_idx, device=device)
    givens = to_chars(target_batch["inputs"])

    labels = None
    if "labels" in target_batch:
        labels = to_chars(target_batch["labels"])

    step_outputs: Dict[str, Dict[str, Any]] = results.get("step_outputs", {}) or {}
    if not step_outputs:
        raise ValueError(
            "No step_outputs found in results_yaml. "
            "Re-run scripts/activation_patching.py (newer version) to generate per-step baseline/patched predictions."
        )

    available_steps = sorted(int(k) for k in step_outputs.keys())
    if args.steps is None:
        steps = available_steps
    else:
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip() != ""]

    # CSS copied from sudoku_report_colored_ai.py
    css = """
    <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; color:#222; }
    h1 { margin: 8px 0 0 0; }
    h2 { margin: 18px 0 6px 0; }
    .meta { color:#555; margin-bottom: 16px; }
    .row3 { display:grid; grid-template-columns: repeat(3, max-content); gap:20px; align-items:start; }
    .row2 { display:grid; grid-template-columns: repeat(2, max-content); gap:20px; align-items:start; }
    .gridBlock { display:block; }
    .sgrid { border-collapse:collapse; margin:6px 0 12px 0; }
    .sgrid td { width:22px; height:22px; text-align:center; border:1px solid #777; padding:2px 4px; position:relative; }
    .sgrid td.br { border-right:2px solid #111; }
    .sgrid td.bb { border-bottom:2px solid #111; }
    .gridTitle { font-weight:700; margin:4px 0; }

    /* Darker, more distinct fills */
    .given   { background:#9AB0FF; font-weight:700; }
    .ok      { background:#57B97B; color:#101; }
    .bad     { background:#E86B6B; color:#101; }
    .blank   { color:#667; }

    /* Changed cells: diagonal split yellow/green or yellow/red */
    .changed_ok {
      background-image: linear-gradient(135deg, #FFE15A 50%, #57B97B 50%);
      background-color: #57B97B;
    }
    .changed_bad {
      background-image: linear-gradient(135deg, #FFE15A 50%, #E86B6B 50%);
      background-color: #E86B6B;
    }

    .legend { margin: 10px 0 16px 0; color:#444; }
    .legend span { display:inline-block; padding:2px 8px; margin-right:8px; border:1px solid #999; }
    </style>
    """

    legend = """
    <div class='legend'>
      <span class='given'>given</span>
      <span class='ok'>ok</span>
      <span class='bad'>bad</span>
      <span class='changed_ok'>changed_ok</span>
      <span class='changed_bad'>changed_bad</span>
      <span class='blank'>blank</span>
    </div>
    """

    sections: List[str] = []

    # Header / metadata
    meta_lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Results YAML: {args.results_yaml}",
        f"Checkpoint: {checkpoint}",
        f"Target puzzle idx: {target_idx}",
        f"Patch level: {cfg.get('patch_level')}",
        f"Patch steps: {cfg.get('patch_steps')}",
        f"Patch positions: {cfg.get('patch_positions')}",
        f"Max steps: {cfg.get('max_steps')}",
    ]

    # Puzzle context (Input + Label if present)
    ctx = ["<h2>Target Puzzle</h2>", "<div class='row3'>"]
    ctx.append(table_html("Input", givens))
    if labels is not None:
        ctx.append(table_html("Label", labels))
    else:
        ctx.append("<div class='gridBlock'><div class='gridTitle'>Label</div><div>(not available)</div></div>")
    # Placeholder third column keeps a consistent 3-up layout
    ctx.append("<div class='gridBlock'></div>")
    ctx.append("</div>")
    sections.append("\n".join(ctx))

    # Steps
    sections.append("<h2>Baseline vs Patched (per step)</h2>")
    for s in steps:
        key = str(s)
        if key not in step_outputs:
            continue
        so = step_outputs[key]
        base_preds = to_chars(so.get("baseline_preds"))
        patched_preds = to_chars(so.get("patched_preds"))

        base_classes = compute_classes(base_preds, givens, prev_chars=None)
        patched_classes = compute_classes(patched_preds, givens, prev_chars=base_preds)

        sec = [f"<h3>Step {s}</h3>", "<div class='row2'>"]
        sec.append(table_html("Baseline target preds", base_preds, base_classes))
        sec.append(table_html("Patched target preds (changes vs baseline highlighted)", patched_preds, patched_classes))
        sec.append("</div>")
        sections.append("\n".join(sec))

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Activation Patching Sudoku Report</title>{css}</head>
<body>
<h1>Activation Patching Sudoku Report</h1>
<div class='meta'>{'<br/>'.join(meta_lines)}</div>
{legend}
{''.join(sections)}
</body></html>
"""

    out = Path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
