#!/usr/bin/env python3
# sudoku_report.py  — colored validation + change tracking

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# ---- CLI / paths ----
OUT_HTML = sys.argv[1] if len(sys.argv) > 1 else "Sudoku_Reports/sudoku_eval_report_colored.html"
NPZ_PATH = "Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/step_0_all_preds.npz"
NUM_EXAMPLES = 5  # how many puzzles to include

# ---------- helpers ----------
def id2num(i: int) -> str:
    # HRM token map: 2..10 -> '1'..'9', everything else -> '.'
    if 2 <= i <= 10:
        return str(i - 1)
    return "."

def to_rows(vec81):
    return [vec81[r * 9 : (r + 1) * 9] for r in range(9)]

def table_html(title, arr81_chars, classes=None):
    rows = to_rows(arr81_chars)
    h = [f"<div class='gridTitle'>{title}</div><table class='sgrid'>"]
    for r in range(9):
        h.append("<tr>")
        for c in range(9):
            ch = arr81_chars[r * 9 + c]
            cls = classes[r * 9 + c] if classes else ""
            borders = []
            if r in (2, 5): borders.append("bb")
            if c in (2, 5): borders.append("br")
            cls = (cls + " " + " ".join(borders)).strip()
            h.append(f"<td class='{cls}'>{ch}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "\n".join(h)

# ---- Sudoku validation / styling ----
def compute_violations(chars81):
    """Return list[bool] length 81; True if the filled cell violates row/col/box."""
    viol = [False] * 81

    # rows
    for r in range(9):
        vals = [chars81[r*9+c] for c in range(9) if chars81[r*9+c] != "."]
        for c in range(9):
            ch = chars81[r*9+c]
            if ch != "." and vals.count(ch) > 1:
                viol[r*9+c] = True

    # cols
    for c in range(9):
        col_vals = [chars81[r*9+c] for r in range(9) if chars81[r*9+c] != "."]
        for r in range(9):
            ch = chars81[r*9+c]
            if ch != "." and col_vals.count(ch) > 1:
                viol[r*9+c] = True

    # 3x3 boxes
    for br in range(3):
        for bc in range(3):
            idxs = [(br*3+rr)*9 + (bc*3+cc) for rr in range(3) for cc in range(3)]
            box_vals = [chars81[i] for i in idxs if chars81[i] != "."]
            for i in idxs:
                ch = chars81[i]
                if ch != "." and box_vals.count(ch) > 1:
                    viol[i] = True

    return viol

def compute_classes(curr_chars, given_chars, prev_chars=None):
    """
    Build CSS classes per cell with priority:
      changed (yellow) > given (blue) > ok (green) / bad (red) > blank (dim)
    """
    viol = compute_violations(curr_chars)
    classes = []
    for i in range(81):
        ch = curr_chars[i]
        given = (given_chars[i] != ".")
        changed = (prev_chars is not None and ch != prev_chars[i])
        if changed and ch != ".":      # only highlight meaningful changes
            cls = "changed"
        elif given:
            cls = "given"
        elif ch == ".":
            cls = "blank"
        else:
            cls = "ok" if not viol[i] else "bad"
        classes.append(cls)
    return classes

# ---------- page assembly ----------
def render_one(i, inputs, labels, pred, interm=None):
    givens       = [id2num(int(x)) for x in inputs[i]]
    label_chars  = [id2num(int(x)) for x in labels[i]]
    pred_chars   = [id2num(int(x)) for x in pred[i]]

    pred_classes = compute_classes(pred_chars, givens, prev_chars=None)

    sec = []
    sec.append(f"<h3>Puzzle {i+1}</h3>")
    sec.append("<div class='row3'>")
    sec.append(table_html("Input", givens))                      # plain (givens already bolded via CSS if you want)
    sec.append(table_html("Prediction", pred_chars, pred_classes))
    sec.append(table_html("Label", label_chars))
    sec.append("</div>")

    # Intermediate playback: color + change tracking between successive steps
    if interm is not None:
        S = interm.shape[0]
        sec.append("<details><summary>Show intermediate steps</summary>")
        sec.append("<div class='steps'>")

        prev_chars = None
        for s in range(S):
            step = interm[s, i]
            # Support [81,V] or [81] per step
            if step.ndim == 2:
                step = step.argmax(-1)
            step_chars = [id2num(int(x)) for x in (step if step.ndim == 1 else step.squeeze())]

            classes = compute_classes(step_chars, givens, prev_chars=prev_chars)
            sec.append(table_html(f"Step {s+1}/{S}", step_chars, classes))
            prev_chars = step_chars

        sec.append("</div></details>")
    return "\n".join(sec)

def main():
    z = np.load(NPZ_PATH, allow_pickle=True)
    inputs = z["inputs"]
    labels = z["labels"]
    pred   = z["logits"].argmax(-1) if "logits" in z else z["pred"]

    # normalize intermediate steps (either [S,N,81] or [S,N,81,V])
    interm = None
    for k in ("intermediate_logits", "intermediate_preds"):
        if k in z:
            v = z[k]
            if (v.ndim == 4 and v.shape[-1] > 1) or (v.ndim == 3):
                interm = v
            break

    N = min(NUM_EXAMPLES, len(pred))
    sections = [render_one(i, inputs, labels, pred, interm=interm) for i in range(N)]

    css = """
    <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; color:#222; }
    h1 { margin: 8px 0 0 0; }
    .meta { color:#666; margin-bottom: 16px; }
    .row3 { display:grid; grid-template-columns: repeat(3, max-content); gap:20px; align-items:start; }
    .sgrid { border-collapse:collapse; margin:6px 0 12px 0; }
    .sgrid td { width:22px; height:22px; text-align:center; border:1px solid #bbb; padding:2px 4px; }
    .sgrid td.br { border-right:2px solid #111; }
    .sgrid td.bb { border-bottom:2px solid #111; }
    .gridTitle { font-weight:700; margin:4px 0; }

    /* Coloring priority enforced in Python:
       changed (yellow) > given (blue) > ok (green) / bad (red) > blank (gray)
    */
    .given   { background:#e6f0ff; font-weight:700; }  /* Blue */
    .changed { background:#fff6bf; }                   /* Yellow */
    .ok      { background:#e9f8ee; }                   /* Green  */
    .bad     { background:#fde8e8; }                   /* Red    */
    .blank   { color:#bbb; }                           /* Dim    */

    details { margin-top: 6px; }
    .steps { display:grid; grid-template-columns: repeat(5, max-content); gap:18px; }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sudoku Eval Report</title>{css}</head>
<body>
<h1>Sudoku Evaluation Report</h1>
<div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Source: {NPZ_PATH}</div>
{''.join(sections)}
</body></html>
"""
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML} — open it locally or download it.")

if __name__ == "__main__":
    main()
