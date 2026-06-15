"""Shared HTML rendering helpers for Maze 30x30 reports.

Mirrors the Sudoku reporting style in `scripts/core/activation_patching.py`
(grid + color legend + change highlighting). Token encoding follows
`utils/maze_targets.py`:

    PAD=0  WALL=1  FREE=2  START=3  GOAL=4  PATH=5
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from utils.maze_targets import (
    GRID_SIZE, SEQ_LEN, PAD_ID, WALL_ID, FREE_ID, START_ID, GOAL_ID, PATH_ID,
)

TOKEN_CHAR = {
    PAD_ID: ".",
    WALL_ID: "█",
    FREE_ID: " ",
    START_ID: "S",
    GOAL_ID: "G",
    PATH_ID: "•",
}


def to_chars(arr: Sequence[int]) -> List[str]:
    return [TOKEN_CHAR.get(int(v), "?") for v in arr]


def to_ids(arr: Sequence[int]) -> List[int]:
    return [int(v) for v in arr]


CSS = """
<style>
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       background:#fafafa; color:#222; margin:16px; }
h1 { margin: 8px 0 4px 0; }
h2 { margin: 18px 0 6px 0; }
h3 { margin: 12px 0 4px 0; font-size: 1.0em; }
.meta { color:#555; margin-bottom: 12px; font-size: 0.9em; }
.legend span { display:inline-block; padding:2px 8px; margin-right:6px;
               border:1px solid #999; font-size:0.85em; }
.row2 { display:grid; grid-template-columns: repeat(2, max-content);
        gap:24px; align-items:start; }
.row3 { display:grid; grid-template-columns: repeat(3, max-content);
        gap:24px; align-items:start; }
.steps { display:flex; flex-wrap:wrap; gap:18px; align-items:start; }
.gridTitle { font-weight:700; margin:4px 0; font-size:0.9em; }
table.mgrid { border-collapse:collapse; }
table.mgrid td {
  width:12px; height:12px; padding:0; text-align:center;
  font-size:9px; line-height:12px;
  border:1px solid #ddd;
}
/* Cell base styles by token */
td.wall   { background:#222; color:#222; }
td.free   { background:#fff; color:#ccc; }
td.start  { background:#1a7f37; color:#fff; font-weight:700; }
td.goal   { background:#cf222e; color:#fff; font-weight:700; }
td.path   { background:#fbe88a; color:#7a5a00; font-weight:700; }
td.pad    { background:#eee; color:#bbb; }
/* Correctness overlays for label-aware rendering */
td.bad         { background:#ff6b6b !important; color:#fff; }
td.missing     { background:#fff1f1 !important; color:#cf222e; }
/* Changed cells vs prev: diagonal gradient (matches Sudoku report style) */
td.changed_ok  { background-image: linear-gradient(135deg, #ffe15a 50%, #fbe88a 50%); }
td.changed_bad { background-image: linear-gradient(135deg, #ffe15a 50%, #ff6b6b 50%); color:#fff; }
table.metrics { border-collapse:collapse; margin:6px 0 12px 0; }
table.metrics th, table.metrics td { border:1px solid #999; padding:3px 8px; font-size:0.9em; }
table.metrics th { background:#eee; }
details > summary { cursor:pointer; }
</style>
"""

LEGEND = """
<div class='legend'>
  <span class='wall'>wall</span>
  <span class='free'>free</span>
  <span class='start'>S start</span>
  <span class='goal'>G goal</span>
  <span class='path'>• path</span>
  <span class='bad'>wrong</span>
  <span class='missing'>missing</span>
  <span class='changed_ok'>changed</span>
</div>
"""


def _base_class(token_id: int) -> str:
    return {
        PAD_ID: "pad",
        WALL_ID: "wall",
        FREE_ID: "free",
        START_ID: "start",
        GOAL_ID: "goal",
        PATH_ID: "path",
    }.get(int(token_id), "free")


def grid_classes(
    pred_ids: Sequence[int],
    *,
    label_ids: Optional[Sequence[int]] = None,
    prev_ids: Optional[Sequence[int]] = None,
) -> List[str]:
    """Return per-cell CSS class list.

    - Base class from `pred_ids` (wall/free/start/goal/path/pad).
    - If `label_ids` given: cells where pred != label get `bad`;
      cells where label is a PATH cell but pred is FREE get `missing`.
    - If `prev_ids` given and pred differs from prev: add `changed_ok` or `changed_bad`
      (relative to label correctness if available, else default to changed_ok).
    """
    n = len(pred_ids)
    classes: List[str] = []
    for i in range(n):
        p = int(pred_ids[i])
        base = _base_class(p)
        wrong = label_ids is not None and int(label_ids[i]) != p
        if label_ids is not None and int(label_ids[i]) in (PATH_ID, START_ID, GOAL_ID) and p == FREE_ID:
            base = "missing"
            wrong = True
        elif wrong:
            base = "bad"
        if prev_ids is not None and int(prev_ids[i]) != p:
            base = "changed_bad" if wrong else "changed_ok"
        classes.append(base)
    return classes


def grid_html(title: str, ids: Sequence[int], classes: Optional[Sequence[str]] = None,
              cell_size_px: Optional[int] = None) -> str:
    """Render a single 30x30 maze grid as an HTML table."""
    chars = [TOKEN_CHAR.get(int(v), "?") for v in ids]
    style = ""
    if cell_size_px is not None:
        style = (f" style='width:{cell_size_px}px;height:{cell_size_px}px;"
                 f"font-size:{max(7, cell_size_px - 3)}px;line-height:{cell_size_px}px'")
    parts = [f"<div><div class='gridTitle'>{title}</div>", "<table class='mgrid'>"]
    for r in range(GRID_SIZE):
        parts.append("<tr>")
        for c in range(GRID_SIZE):
            i = r * GRID_SIZE + c
            cls = classes[i] if classes is not None else _base_class(ids[i])
            ch = chars[i] if chars[i] != " " else "&nbsp;"
            parts.append(f"<td class='{cls}'{style}>{ch}</td>")
        parts.append("</tr>")
    parts.append("</table></div>")
    return "".join(parts)


def metrics_table(rows: List[Dict[str, object]], headers: Sequence[str]) -> str:
    h = ["<table class='metrics'><tr>"]
    h += [f"<th>{x}</th>" for x in headers]
    h.append("</tr>")
    for row in rows:
        h.append("<tr>")
        for k in headers:
            v = row.get(k, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            h.append(f"<td>{v}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def html_doc(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>"
        f"{CSS}</head><body>{body}</body></html>"
    )
