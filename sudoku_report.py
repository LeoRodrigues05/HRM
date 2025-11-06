#!/usr/bin/env python3
# sudoku_report.py
import os, json
import numpy as np
from pathlib import Path
from datetime import datetime

NPZ_PATH = "Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/eval_with_steps.npz"
OUT_HTML = "sudoku_eval_report.html"
NUM_EXAMPLES = 5  # how many puzzles to include

# --- token helpers (HRM Sudoku) ---
def id2num(i: int) -> str:
    if 2 <= i <= 10:  # tokens map 2..10 -> '1'..'9'
        return str(i - 1)
    return "."

def to_rows(vec81):
    return [vec81[r*9:(r+1)*9] for r in range(9)]

def style_cell(inp_ch, pred_ch, label_ch):
    # Input givens bold, blanks dim; correctness colored on prediction
    base = []
    if inp_ch == ".":
        base.append("blank")
    else:
        base.append("given")
    if label_ch != ".":
        base.append("ok" if pred_ch == label_ch else "bad")
    return " ".join(base)

def table_html(title, arr81_chars, classes=None):
    rows = to_rows(arr81_chars)
    h = [f"<div class='gridTitle'>{title}</div><table class='sgrid'>"]
    for r in range(9):
        h.append("<tr>")
        for c in range(9):
            ch = arr81_chars[r*9+c]
            cls = classes[r*9+c] if classes else ""
            borders = []
            if r in (2,5): borders.append("bb")
            if c in (2,5): borders.append("br")
            cls = (cls + " " + " ".join(borders)).strip()
            h.append(f"<td class='{cls}'>{ch}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "\n".join(h)

def render_one(i, inputs, labels, pred, interm=None):
    inp_chars   = [id2num(int(x)) for x in inputs[i]]
    label_chars = [id2num(int(x)) for x in labels[i]]
    pred_chars  = [id2num(int(x)) for x in pred[i]]

    classes = []
    for k in range(81):
        classes.append(style_cell(inp_chars[k], pred_chars[k], label_chars[k]))

    sec = []
    sec.append(f"<h3>Puzzle {i}</h3>")
    # Row of 3 tables
    sec.append("<div class='row3'>")
    sec.append(table_html("Input", inp_chars))
    sec.append(table_html("Prediction", pred_chars, classes))
    sec.append(table_html("Label", label_chars))
    sec.append("</div>")

    # Optional intermediate playback
    if interm is not None:
        # interm: [S, N, 81] or [S, N, 81, V] -> pick argmax if last dim is V
        S = interm.shape[0]
        sec.append("<details><summary>Show intermediate steps</summary>")
        sec.append("<div class='steps'>")
        for s in range(S):
            step = interm[s, i]
            if step.ndim == 2:  # [81, V]
                step = step.argmax(-1)
            step_chars = [id2num(int(x)) for x in (step if step.ndim == 1 else step.squeeze())]
            sec.append(table_html(f"Step {s+1}/{S}", step_chars))
        sec.append("</div></details>")
    return "\n".join(sec)

def main():
    z = np.load(NPZ_PATH, allow_pickle=True)
    inputs = z["inputs"]
    labels = z["labels"]
    if "logits" in z:
        pred = z["logits"].argmax(-1)
    else:
        pred = z["pred"]
    interm = None
    # if you saved intermediate steps, one of these keys might exist:
    for k in ("intermediate_logits", "intermediate_preds"):
        if k in z:
            v = z[k]
            # normalize to [S, N, 81] if possible
            if v.ndim == 4 and v.shape[-1] > 32:     # [S,N,81,V]
                v = v  # handled during render via argmax
            elif v.ndim == 3:                        # [S,N,81]
                pass
            else:
                v = None
            interm = v
            break

    N = min(NUM_EXAMPLES, len(pred))
    sections = []
    for i in range(N):
        sections.append(render_one(i, inputs, labels, pred, interm=interm))

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
    .given { font-weight:700; }
    .blank { color:#bbb; }
    .ok { background:#e9f8ee; }
    .bad { background:#fde8e8; }
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
    Path(OUT_HTML).write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML} — open it locally or download it.")

if __name__ == "__main__":
    main()
