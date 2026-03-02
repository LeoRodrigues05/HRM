#!/usr/bin/env python3
"""
easy_puzzle_analysis.py - Analyze HRM model on easy Sudoku puzzles (few missing cells)

This script creates easy Sudoku puzzles from complete solutions and evaluates
the HRM model's ability to solve them, capturing intermediate steps.

The HRM model was trained on EXTREME difficulty Sudoku puzzles (50+ missing cells).
This analysis tests whether it can generalize to trivially easy puzzles (1-5 missing cells).
"""

import sys
import os
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Token mapping: HRM uses 2..10 -> '1'..'9', else '.'
def id2num(i: int) -> str:
    if 2 <= i <= 10:
        return str(i - 1)
    return "."

def num2id(ch: str) -> int:
    """Convert sudoku digit (1-9) or '.' to HRM token id"""
    if ch in '123456789':
        return int(ch) + 1  # '1' -> 2, '2' -> 3, ..., '9' -> 10
    return 1  # '.' or '0' -> 1 (blank token)


def create_easy_puzzles_from_solution(solution: np.ndarray, num_missing: int = 1, num_variants: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create easy puzzles by removing `num_missing` cells from a complete solution.
    Returns list of (input, label) pairs.
    """
    puzzles = []
    flat_solution = solution.flatten()
    
    for _ in range(num_variants):
        # Randomly select cells to blank out
        blank_indices = np.random.choice(81, size=num_missing, replace=False)
        
        puzzle_input = flat_solution.copy()
        puzzle_input[blank_indices] = 0  # 0 represents blank
        
        puzzles.append((puzzle_input, flat_solution.copy()))
    
    return puzzles


def get_valid_solutions() -> List[np.ndarray]:
    """Generate or load valid Sudoku solutions."""
    # Create some valid solutions by loading from dataset or generating
    solutions = []
    
    # Simple valid solution (row-shifted pattern)
    base = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    shifts = [0, 3, 6, 1, 4, 7, 2, 5, 8]  # Valid sudoku shift pattern
    solution = np.array([np.roll(base, -s) for s in shifts]).flatten()
    solutions.append(solution)
    
    # Another valid solution (different pattern)
    base2 = np.array([5, 3, 4, 6, 7, 2, 1, 9, 8])
    solution2 = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]).flatten()
    solutions.append(solution2)
    
    # Third solution
    solution3 = np.array([
        [9, 8, 7, 6, 5, 4, 3, 2, 1],
        [6, 5, 4, 3, 2, 1, 9, 8, 7],
        [3, 2, 1, 9, 8, 7, 6, 5, 4],
        [8, 9, 6, 7, 4, 5, 2, 1, 3],
        [7, 4, 5, 2, 1, 3, 8, 9, 6],
        [2, 1, 3, 8, 9, 6, 7, 4, 5],
        [5, 7, 9, 4, 6, 8, 1, 3, 2],
        [4, 6, 8, 1, 3, 2, 5, 7, 9],
        [1, 3, 2, 5, 7, 9, 4, 6, 8]
    ]).flatten()
    solutions.append(solution3)
    
    return solutions


def convert_to_hrm_format(puzzles: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Convert puzzles to HRM input format."""
    inputs = []
    labels = []
    
    for inp, lbl in puzzles:
        # Convert to HRM token IDs
        inp_ids = np.array([num2id(str(x) if x > 0 else '.') for x in inp], dtype=np.int32)
        lbl_ids = np.array([num2id(str(x)) for x in lbl], dtype=np.int32)
        
        inputs.append(inp_ids)
        labels.append(lbl_ids)
    
    return {
        'inputs': np.array(inputs),
        'labels': np.array(labels),
        'puzzle_identifiers': np.zeros(len(inputs), dtype=np.int32)  # Dummy identifiers
    }


def load_model(checkpoint_path: str):
    """Load the HRM model from checkpoint."""
    config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    # Create a minimal metadata object
    class MockMetadata:
        vocab_size = 11  # 0-10 tokens
        seq_len = 81
        num_puzzle_identifiers = 1000000  # Large enough
        total_groups = 1
        mean_puzzle_examples = 1
    
    train_state = init_train_state(config, MockMetadata(), world_size=1)
    
    # Load weights
    try:
        train_state.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cuda"), assign=True
        )
    except:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v 
             for k, v in torch.load(checkpoint_path, map_location="cuda").items()},
            assign=True
        )
    
    train_state.model.eval()
    return train_state, config


def run_inference_with_steps(model, batch: Dict[str, torch.Tensor], max_steps: int = 16):
    """Run model inference and capture all intermediate steps."""
    with torch.inference_mode():
        # Initialize carry
        carry = model.initial_carry(batch)
        
        intermediate_preds = []
        
        for step_idx in range(max_steps):
            # Run one step
            carry, _, metrics, preds, all_finish = model(
                carry=carry, 
                batch=batch, 
                return_keys=["logits", "intermediate_preds_step"]
            )
            
            if "intermediate_preds_step" in preds:
                intermediate_preds.append(preds["intermediate_preds_step"].cpu())
            
            if all_finish:
                break
        
        # Get final logits
        final_logits = preds.get("logits", None)
        if final_logits is not None:
            final_preds = final_logits.argmax(-1).cpu()
        else:
            final_preds = None
        
        if intermediate_preds:
            intermediate_preds = torch.stack(intermediate_preds, dim=0)
        else:
            intermediate_preds = None
            
        return final_preds, intermediate_preds, len(intermediate_preds) if intermediate_preds is not None else 0


def compute_violations(chars81: List[str]) -> List[bool]:
    """True at i if that filled cell violates row/col/box uniqueness."""
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
    # boxes
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
    """Compute CSS classes for cells."""
    viol = compute_violations(curr_chars)
    classes = []
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


def table_html(title, arr81_chars, classes=None, highlight_missing=None):
    """Generate HTML table for a Sudoku grid."""
    h = [f"<div class='gridTitle'>{title}</div><table class='sgrid'>"]
    for r in range(9):
        h.append("<tr>")
        for c in range(9):
            idx = r * 9 + c
            ch = arr81_chars[idx]
            cls = classes[idx] if classes else ""
            
            # Highlight originally missing cells
            if highlight_missing and idx in highlight_missing:
                cls += " highlight"
            
            borders = []
            if r in (2, 5): borders.append("bb")
            if c in (2, 5): borders.append("br")
            cls = (cls + " " + " ".join(borders)).strip()
            h.append(f"<td class='{cls}'>{ch}</td>")
        h.append("</tr>")
    h.append("</table>")
    return "\n".join(h)


def render_puzzle_analysis(idx: int, input_ids: np.ndarray, label_ids: np.ndarray, 
                           pred_ids: np.ndarray, intermediate: np.ndarray = None,
                           num_steps: int = 0):
    """Render HTML for one puzzle analysis."""
    givens = [id2num(int(x)) for x in input_ids]
    label_chars = [id2num(int(x)) for x in label_ids]
    pred_chars = [id2num(int(x)) for x in pred_ids]
    
    # Find missing cell indices
    missing_indices = [i for i, ch in enumerate(givens) if ch == "."]
    num_missing = len(missing_indices)
    
    # Check if prediction is correct
    is_correct = (pred_ids == label_ids).all()
    
    pred_classes = compute_classes(pred_chars, givens, prev_chars=None)
    
    status_class = "success" if is_correct else "failure"
    status_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    
    sec = []
    sec.append(f"<div class='puzzle-container {status_class}'>")
    sec.append(f"<h3>Puzzle {idx+1} - {num_missing} missing cell{'s' if num_missing > 1 else ''} - <span class='status {status_class}'>{status_text}</span></h3>")
    sec.append(f"<p class='info'>Missing positions: {missing_indices} | Total steps taken: {num_steps}</p>")
    
    sec.append("<div class='row3'>")
    sec.append(table_html("Input (Given)", givens, highlight_missing=set(missing_indices)))
    sec.append(table_html("Model Prediction", pred_chars, pred_classes, highlight_missing=set(missing_indices)))
    sec.append(table_html("Ground Truth", label_chars, highlight_missing=set(missing_indices)))
    sec.append("</div>")
    
    # Show analysis of missing cells
    sec.append("<div class='analysis'>")
    sec.append("<h4>Missing Cell Analysis:</h4>")
    sec.append("<ul>")
    for mi in missing_indices:
        row, col = mi // 9, mi % 9
        box = (row // 3) * 3 + (col // 3)
        expected = label_chars[mi]
        predicted = pred_chars[mi]
        match = "✓" if expected == predicted else "✗"
        sec.append(f"<li>Cell ({row+1},{col+1}) [Box {box+1}]: Expected <b>{expected}</b>, Got <b>{predicted}</b> {match}</li>")
    sec.append("</ul>")
    sec.append("</div>")
    
    # Intermediate steps
    if intermediate is not None and len(intermediate) > 0:
        sec.append("<details><summary>Show intermediate steps ({} steps)</summary>".format(len(intermediate)))
        sec.append("<div class='steps'>")
        prev_chars = None
        for s in range(len(intermediate)):
            step = intermediate[s]
            if step.ndim == 2:
                step = step.argmax(-1)
            step_chars = [id2num(int(x)) for x in step]
            classes = compute_classes(step_chars, givens, prev_chars=prev_chars)
            sec.append(table_html(f"Step {s+1}", step_chars, classes, highlight_missing=set(missing_indices)))
            prev_chars = step_chars
        sec.append("</div></details>")
    
    sec.append("</div>")
    return "\n".join(sec), is_correct


def generate_report(results: List[Dict], output_path: str):
    """Generate the full HTML report."""
    css = """
    <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; color:#222; max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1 { margin: 8px 0 0 0; color: #333; }
    h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
    .meta { color:#555; margin-bottom: 16px; }
    .summary { background: #e8f4e8; padding: 15px; border-radius: 8px; margin: 20px 0; }
    .summary.mixed { background: #fff3e0; }
    .summary h3 { margin-top: 0; }
    
    .puzzle-container { border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin: 20px 0; }
    .puzzle-container.success { border-color: #4caf50; background: #f8fff8; }
    .puzzle-container.failure { border-color: #f44336; background: #fff8f8; }
    
    .status { font-weight: bold; padding: 2px 8px; border-radius: 4px; }
    .status.success { background: #4caf50; color: white; }
    .status.failure { background: #f44336; color: white; }
    
    .info { color: #666; font-size: 0.9em; }
    .analysis { background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 10px 0; }
    .analysis ul { margin: 5px 0; }
    
    .row3 { display:grid; grid-template-columns: repeat(3, max-content); gap:20px; align-items:start; }
    .sgrid { border-collapse:collapse; margin:6px 0 12px 0; }
    .sgrid td { width:28px; height:28px; text-align:center; border:1px solid #777; padding:2px 4px; position:relative; font-size: 16px; }
    .sgrid td.br { border-right:2px solid #111; }
    .sgrid td.bb { border-bottom:2px solid #111; }
    .gridTitle { font-weight:700; margin:4px 0; }

    /* Cell styling */
    .given   { background:#9AB0FF; font-weight:700; }
    .ok      { background:#57B97B; color:#101; }
    .bad     { background:#E86B6B; color:#101; }
    .blank   { color:#667; }
    .highlight { box-shadow: inset 0 0 0 2px #ff9800; }

    .changed_ok {
      background-image: linear-gradient(135deg, #FFE15A 50%, #57B97B 50%);
      background-color: #57B97B;
    }
    .changed_bad {
      background-image: linear-gradient(135deg, #FFE15A 50%, #E86B6B 50%);
      background-color: #E86B6B;
    }

    details { margin-top: 6px; }
    summary { cursor: pointer; font-weight: bold; color: #1976d2; }
    .steps { display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, max-content)); gap:18px; margin-top: 10px; }
    
    .legend { display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-box { width: 20px; height: 20px; border: 1px solid #777; }
    </style>
    """
    
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    sections = []
    for r in results:
        html, _ = render_puzzle_analysis(
            r['idx'], r['input'], r['label'], r['pred'],
            r.get('intermediate'), r.get('num_steps', 0)
        )
        sections.append(html)
    
    summary_class = "mixed" if 0 < correct < total else ""
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Easy Sudoku Analysis - HRM Model</title>{css}</head>
<body>
<h1>🧩 Easy Sudoku Analysis Report</h1>
<div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: HRM (Hierarchical Reasoning Model)</div>

<div class="summary {summary_class}">
<h3>Summary</h3>
<p><b>Total Puzzles:</b> {total}</p>
<p><b>Correct:</b> {correct} ({100*correct/total:.1f}%)</p>
<p><b>Incorrect:</b> {total - correct} ({100*(total-correct)/total:.1f}%)</p>
<p>This analysis tests the HRM model on <b>easy</b> Sudoku puzzles with only 1-3 missing cells.</p>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-box" style="background:#9AB0FF;"></div> Given (input cell)</div>
    <div class="legend-item"><div class="legend-box" style="background:#57B97B;"></div> Correct prediction</div>
    <div class="legend-item"><div class="legend-box" style="background:#E86B6B;"></div> Incorrect/violation</div>
    <div class="legend-item"><div class="legend-box" style="box-shadow: inset 0 0 0 2px #ff9800;"></div> Originally missing cell</div>
</div>

<h2>Detailed Results</h2>
{''.join(sections)}

<h2>Observations</h2>
<div class="analysis">
<p>This report demonstrates the HRM model's performance on trivially easy Sudoku puzzles where only 1-3 cells are missing.
Even with most of the puzzle already solved, the model needs to correctly identify the missing digits through:</p>
<ul>
<li><b>Row constraint:</b> Each row must contain digits 1-9 exactly once</li>
<li><b>Column constraint:</b> Each column must contain digits 1-9 exactly once</li>
<li><b>Box constraint:</b> Each 3x3 box must contain digits 1-9 exactly once</li>
</ul>
<p>For puzzles with only 1 missing cell, the answer is deterministic and can be found by elimination from any constraint.</p>
</div>

</body></html>
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--output", type=str, 
                        default="Sudoku_Reports/easy_puzzle_analysis.html")
    parser.add_argument("--num-puzzles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Loading model...")
    train_state, config = load_model(args.checkpoint)
    model = train_state.model
    
    print("Generating easy puzzles...")
    solutions = get_valid_solutions()
    
    # Generate puzzles with varying difficulty (1-3 missing cells)
    all_puzzles = []
    
    # 1 missing cell
    for sol in solutions:
        all_puzzles.extend(create_easy_puzzles_from_solution(sol, num_missing=1, num_variants=2))
    
    # 2 missing cells  
    for sol in solutions:
        all_puzzles.extend(create_easy_puzzles_from_solution(sol, num_missing=2, num_variants=2))
    
    # 3 missing cells
    for sol in solutions:
        all_puzzles.extend(create_easy_puzzles_from_solution(sol, num_missing=3, num_variants=2))
    
    # Limit to requested number
    all_puzzles = all_puzzles[:args.num_puzzles]
    
    print(f"Running inference on {len(all_puzzles)} puzzles...")
    
    # Convert to HRM format
    batch_data = convert_to_hrm_format(all_puzzles)
    
    results = []
    
    # Process puzzles one at a time to capture steps
    for i in range(len(all_puzzles)):
        batch = {
            'inputs': torch.tensor(batch_data['inputs'][i:i+1]).cuda(),
            'labels': torch.tensor(batch_data['labels'][i:i+1]).cuda(),
            'puzzle_identifiers': torch.tensor(batch_data['puzzle_identifiers'][i:i+1]).cuda()
        }
        
        pred, intermediate, num_steps = run_inference_with_steps(model, batch, max_steps=16)
        
        if pred is None:
            print(f"Warning: No prediction for puzzle {i}")
            continue
        
        is_correct = (pred[0].numpy() == batch_data['labels'][i]).all()
        
        results.append({
            'idx': i,
            'input': batch_data['inputs'][i],
            'label': batch_data['labels'][i],
            'pred': pred[0].numpy(),
            'intermediate': intermediate[:, 0].numpy() if intermediate is not None else None,
            'num_steps': num_steps,
            'correct': is_correct
        })
        
        status = "✓" if is_correct else "✗"
        num_missing = (batch_data['inputs'][i] == 1).sum()
        print(f"  Puzzle {i+1}: {num_missing} missing cells - {status}")
    
    print(f"\nGenerating report...")
    generate_report(results, args.output)
    
    # Print summary
    correct = sum(1 for r in results if r['correct'])
    print(f"\nSummary: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")


if __name__ == "__main__":
    main()
