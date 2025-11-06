import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

# ---------- Config ----------
NPZ_PATH = "Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/eval_outputs_rank0.npz"
NUM_EXAMPLES = 5  # how many puzzles to display
SHOW_INPUT_HIGHLIGHTS = True  # bold givens, dim blanks
# ----------------------------


def id2num(i: int) -> str:
    """Map HRM token id -> display char."""
    if 2 <= i <= 10:
        return str(i - 1)  # 2..10 -> '1'..'9'
    return "."  # PAD/EOS/other shown as '.'


def row_vals(arr, r):
    return [int(x) for x in arr[r * 9 : (r + 1) * 9]]


def styled_cell(val: str, *, style: str | None = None) -> str:
    return f"[{style}]{val}[/]" if style else val


def show_table_inputs(inp: np.ndarray, console: Console):
    """Pretty-print the input grid with givens bold and blanks dim."""
    t = Table(show_header=False, box=box.SIMPLE_HEAVY, padding=(0, 1))
    for r in range(9):
        row = []
        vals = row_vals(inp, r)
        for c, v in enumerate(vals):
            ch = id2num(v)
            style = None
            if SHOW_INPUT_HIGHLIGHTS:
                if ch == ".":
                    style = "dim"
                else:
                    style = "bold"
            row.append(styled_cell(ch, style=style))
        t.add_row(*_with_box_separators(row))
        if r in (2, 5):
            t.add_section()
    console.print(t)


def show_table_pred_vs_label(pred: np.ndarray, label: np.ndarray, console: Console):
    """Color-code prediction vs label: green = correct, red = wrong."""
    t = Table(show_header=False, box=box.SIMPLE_HEAVY, padding=(0, 1))
    correct = 0
    total = 81
    for r in range(9):
        prow = row_vals(pred, r)
        lrow = row_vals(label, r)
        row_cells = []
        for c, (pv, lv) in enumerate(zip(prow, lrow)):
            pch = id2num(pv)
            lch = id2num(lv)
            ok = (pch == lch)
            if lch == ".":
                # If label has '.', treat as don't-care (shouldn't happen for solved labels)
                style = None
            else:
                style = "green bold" if ok else "red"
                correct += int(ok)
            row_cells.append(styled_cell(pch, style=style))
        t.add_row(*_with_box_separators(row_cells))
        if r in (2, 5):
            t.add_section()
    acc = correct / total
    console.print(t)
    console.print(f"[bold]Cell accuracy:[/] {acc:.3%}  (correct {correct}/{total})")


def _with_box_separators(cells: list[str]) -> list[str]:
    """Insert a vertical box separator between 3x3 blocks by returning cells as-is;
    the Table box style handles borders. This helper is kept for future per-cell separators."""
    return cells


def main():
    z = np.load(NPZ_PATH, allow_pickle=True)
    inputs = z["inputs"]                # [N, 81]
    labels = z["labels"]                # [N, 81]
    pred = z["logits"].argmax(-1) if "logits" in z else z["pred"]  # [N, 81]

    console = Console()

    n = min(NUM_EXAMPLES, len(pred))
    for i in range(n):
        console.rule(f"[bold blue]Sudoku {i}")
        console.print("[bold]Input[/]")
        show_table_inputs(inputs[i], console)
        console.print("[bold]Prediction (green=correct, red=wrong)[/]")
        show_table_pred_vs_label(pred[i], labels[i], console)
        console.print("[bold]Label[/]")
        show_table_inputs(labels[i], console)
        console.print()


if __name__ == "__main__":
    main()
