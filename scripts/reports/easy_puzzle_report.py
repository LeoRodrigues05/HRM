#!/usr/bin/env python3
"""
easy_puzzle_report.py - Generate a report showing HRM model on easy Sudoku puzzles

The HRM model was trained on EXTREME difficulty Sudoku puzzles (50+ missing cells).
This report creates easy puzzles (1-5 missing cells) from solutions in the test set
and shows how the model performs on them, including intermediate steps.

Key Finding: The model struggles with easier puzzles despite being trained on harder ones,
demonstrating that it doesn't generalize well to simpler constraint satisfaction tasks.
"""

import sys
import os
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Configuration
NPZ_PATH = "data/step_0_all_preds.npz"
CHECKPOINT_PATH = "checkpoints/sapientinc-sudoku-extreme/checkpoint.pt"
CONFIG_PATH = "checkpoints/sapientinc-sudoku-extreme/all_config.yaml"


def id2num(i: int) -> str:
    """HRM token map: 2..10 -> '1'..'9', else '.'"""
    if 2 <= i <= 10:
        return str(i - 1)
    return "."


def num2id(n: int) -> int:
    """Convert digit 1-9 to HRM token id (2-10), 0 -> 1 (blank)"""
    if 1 <= n <= 9:
        return n + 1
    return 1  # blank


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


def create_easy_puzzle(solution_ids: np.ndarray, num_missing: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    """Create an easy puzzle by blanking out num_missing cells from a complete solution."""
    puzzle = solution_ids.copy()
    blank_indices = rng.choice(81, size=num_missing, replace=False).tolist()
    for idx in blank_indices:
        puzzle[idx] = 1  # blank token
    return puzzle, blank_indices


def load_model_and_run_easy_puzzles(easy_puzzles: List[Dict], max_steps: int = 16):
    """Load model and run inference on easy puzzles."""
    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from pretrain import PretrainConfig, init_train_state, create_dataloader
    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    # We need train metadata for vocab size etc
    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1, 
        global_batch_size=config.global_batch_size, rank=0, world_size=1
    )
    
    # Create model
    train_state = init_train_state(config, train_metadata, world_size=1)
    
    # Load weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    try:
        train_state.model.load_state_dict(checkpoint, assign=True)
    except:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()},
            assign=True
        )
    
    train_state.model.eval()
    model = train_state.model
    
    results = []
    
    for puzzle_data in easy_puzzles:
        # Create batch
        batch = {
            'inputs': torch.tensor(puzzle_data['input'][None, :], dtype=torch.int32).cuda(),
            'labels': torch.tensor(puzzle_data['label'][None, :], dtype=torch.int32).cuda(),
            'puzzle_identifiers': torch.tensor([0], dtype=torch.int32).cuda()
        }
        
        with torch.inference_mode():
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            intermediate_preds = []
            
            for step_idx in range(max_steps):
                # The ACTLossHead wrapper requires return_keys argument
                carry, loss, metrics, outputs, all_finish = model(
                    carry=carry, 
                    batch=batch,
                    return_keys=["logits", "intermediate_preds_step"]
                )
                
                if "intermediate_preds_step" in outputs:
                    step_pred = outputs["intermediate_preds_step"].cpu().numpy()[0]
                else:
                    # Fall back to computing from logits
                    step_pred = outputs["logits"].argmax(-1).cpu().numpy()[0]
                intermediate_preds.append(step_pred)
                
                if all_finish:
                    break
            
            # Final prediction
            final_pred = outputs["logits"].argmax(-1).cpu().numpy()[0]
        
        puzzle_data['pred'] = final_pred
        puzzle_data['intermediate'] = np.array(intermediate_preds)
        puzzle_data['num_steps'] = len(intermediate_preds)
        puzzle_data['correct'] = (final_pred == puzzle_data['label']).all()
        results.append(puzzle_data)
    
    return results


def render_puzzle_html(idx: int, puzzle_data: Dict) -> str:
    """Render HTML for one puzzle analysis."""
    input_ids = puzzle_data['input']
    label_ids = puzzle_data['label']
    pred_ids = puzzle_data['pred']
    intermediate = puzzle_data.get('intermediate')
    missing_indices = puzzle_data['missing_indices']
    
    givens = [id2num(int(x)) for x in input_ids]
    label_chars = [id2num(int(x)) for x in label_ids]
    pred_chars = [id2num(int(x)) for x in pred_ids]
    
    num_missing = len(missing_indices)
    is_correct = puzzle_data['correct']
    
    # Count correct missing cells
    correct_missing = sum(1 for mi in missing_indices if pred_ids[mi] == label_ids[mi])
    
    # Count how many GIVEN cells were incorrectly changed by the model
    given_indices = [i for i in range(81) if i not in missing_indices]
    given_changed = sum(1 for gi in given_indices if pred_ids[gi] != input_ids[gi])
    
    pred_classes = compute_classes(pred_chars, givens, prev_chars=None)
    
    status_class = "success" if is_correct else "failure"
    status_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    
    sec = []
    sec.append(f"<div class='puzzle-container {status_class}'>")
    sec.append(f"<h3>Puzzle {idx+1} - {num_missing} missing cell{'s' if num_missing > 1 else ''} - <span class='status {status_class}'>{status_text}</span></h3>")
    sec.append(f"<p class='info'>Missing positions: {missing_indices} | Missing cells correct: {correct_missing}/{num_missing}</p>")
    
    # Highlight given cells changed issue
    if given_changed > 0:
        sec.append(f"<p class='warning'>⚠️ <b>WARNING:</b> Model changed {given_changed} given (input) cells that should have been preserved!</p>")
    
    sec.append("<div class='row3'>")
    sec.append(table_html("Input (Given)", givens, highlight_missing=set(missing_indices)))
    sec.append(table_html("Model Prediction", pred_chars, pred_classes, highlight_missing=set(missing_indices)))
    sec.append(table_html("Ground Truth", label_chars, highlight_missing=set(missing_indices)))
    sec.append("</div>")
    
    # Analysis of missing cells
    sec.append("<div class='analysis'>")
    sec.append("<h4>Missing Cell Analysis:</h4>")
    sec.append("<ul>")
    for mi in missing_indices:
        row, col = mi // 9, mi % 9
        box = (row // 3) * 3 + (col // 3)
        expected = label_chars[mi]
        predicted = pred_chars[mi]
        match = "✓" if expected == predicted else "✗"
        sec.append(f"<li>Cell ({row+1},{col+1}) [Row {row+1}, Col {col+1}, Box {box+1}]: Expected <b>{expected}</b>, Got <b>{predicted}</b> {match}</li>")
    sec.append("</ul>")
    sec.append("</div>")
    
    # Show which given cells were changed (if any)
    if given_changed > 0:
        sec.append("<div class='analysis warning-box'>")
        sec.append(f"<h4>⚠️ Given Cells Changed ({given_changed} cells):</h4>")
        sec.append("<p>The model incorrectly modified these input cells that should have been preserved:</p>")
        sec.append("<ul>")
        count = 0
        for gi in given_indices:
            if pred_ids[gi] != input_ids[gi]:
                row, col = gi // 9, gi % 9
                sec.append(f"<li>Cell ({row+1},{col+1}): Input was <b>{givens[gi]}</b>, Model changed to <b>{pred_chars[gi]}</b></li>")
                count += 1
                if count >= 10:
                    remaining = given_changed - count
                    if remaining > 0:
                        sec.append(f"<li>... and {remaining} more cells</li>")
                    break
        sec.append("</ul>")
        sec.append("</div>")
    
    # Intermediate steps
    if intermediate is not None and len(intermediate) > 0:
        sec.append(f"<details><summary>Show all {len(intermediate)} intermediate steps</summary>")
        sec.append("<div class='steps'>")
        prev_chars = None
        for s in range(len(intermediate)):
            step = intermediate[s]
            step_chars = [id2num(int(x)) for x in step]
            classes = compute_classes(step_chars, givens, prev_chars=prev_chars)
            
            # Check which missing cells are correct at this step
            correct_at_step = sum(1 for mi in missing_indices if step[mi] == label_ids[mi])
            # Check given cells changed at this step
            given_changed_step = sum(1 for gi in given_indices if step[gi] != input_ids[gi])
            step_title = f"Step {s+1} ({correct_at_step}/{num_missing}✓, {given_changed_step}⚠)"
            
            sec.append(table_html(step_title, step_chars, classes, highlight_missing=set(missing_indices)))
            prev_chars = step_chars
        sec.append("</div></details>")
    
    sec.append("</div>")
    return "\n".join(sec)


def generate_report(results: List[Dict], output_path: str):
    """Generate the full HTML report."""
    css = """
    <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; color:#222; max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1 { margin: 8px 0 0 0; color: #333; }
    h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 8px; margin-top: 30px; }
    .meta { color:#555; margin-bottom: 16px; }
    
    .summary { background: #fff3e0; padding: 15px; border-radius: 8px; margin: 20px 0; border: 1px solid #ffcc80; }
    .summary.good { background: #e8f5e9; border-color: #a5d6a7; }
    .summary.bad { background: #ffebee; border-color: #ef9a9a; }
    .summary h3 { margin-top: 0; }
    
    .key-finding { background: linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ff9800; }
    .key-finding h3 { margin-top: 0; color: #e65100; }
    
    .critical-issue { background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #d32f2f; }
    .critical-issue h3 { margin-top: 0; color: #c62828; }
    
    .puzzle-container { border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin: 20px 0; }
    .puzzle-container.success { border-color: #4caf50; background: #f8fff8; }
    .puzzle-container.failure { border-color: #f44336; background: #fff8f8; }
    
    .status { font-weight: bold; padding: 2px 8px; border-radius: 4px; }
    .status.success { background: #4caf50; color: white; }
    .status.failure { background: #f44336; color: white; }
    
    .info { color: #666; font-size: 0.9em; }
    .warning { color: #d32f2f; font-size: 0.9em; background: #ffebee; padding: 5px 10px; border-radius: 4px; margin: 5px 0; }
    .analysis { background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 10px 0; }
    .analysis ul { margin: 5px 0; }
    .warning-box { background: #fff3e0; border-left: 3px solid #ff9800; }
    
    .row3 { display:grid; grid-template-columns: repeat(3, max-content); gap:20px; align-items:start; }
    .sgrid { border-collapse:collapse; margin:6px 0 12px 0; }
    .sgrid td { width:28px; height:28px; text-align:center; border:1px solid #777; padding:2px 4px; position:relative; font-size: 16px; }
    .sgrid td.br { border-right:2px solid #111; }
    .sgrid td.bb { border-bottom:2px solid #111; }
    .gridTitle { font-weight:700; margin:4px 0; font-size: 12px; }

    .given   { background:#9AB0FF; font-weight:700; }
    .ok      { background:#57B97B; color:#101; }
    .bad     { background:#E86B6B; color:#101; }
    .blank   { color:#667; }
    .highlight { box-shadow: inset 0 0 0 3px #ff9800; }

    .changed_ok { background-image: linear-gradient(135deg, #FFE15A 50%, #57B97B 50%); }
    .changed_bad { background-image: linear-gradient(135deg, #FFE15A 50%, #E86B6B 50%); }

    details { margin-top: 6px; }
    summary { cursor: pointer; font-weight: bold; color: #1976d2; }
    .steps { display:grid; grid-template-columns: repeat(auto-fill, minmax(180px, max-content)); gap:12px; margin-top: 10px; }
    
    .legend { display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-box { width: 20px; height: 20px; border: 1px solid #777; }
    
    .stats-table { border-collapse: collapse; margin: 15px 0; }
    .stats-table td, .stats-table th { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    .stats-table th { background: #f5f5f5; }
    </style>
    """
    
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Calculate how many puzzles had given cells changed
    def count_given_changed(r):
        given_indices = [i for i in range(81) if i not in r['missing_indices']]
        return sum(1 for gi in given_indices if r['pred'][gi] != r['input'][gi])
    
    puzzles_with_given_changed = sum(1 for r in results if count_given_changed(r) > 0)
    total_given_changed = sum(count_given_changed(r) for r in results)
    
    # Group by difficulty
    by_difficulty = {}
    for r in results:
        n = len(r['missing_indices'])
        by_difficulty.setdefault(n, []).append(r)
    
    sections = [render_puzzle_html(i, r) for i, r in enumerate(results)]
    
    summary_class = "bad" if correct < total // 2 else ("good" if correct == total else "")
    
    # Stats table
    stats_rows = []
    for n in sorted(by_difficulty.keys()):
        puzzles = by_difficulty[n]
        n_correct = sum(1 for p in puzzles if p['correct'])
        n_given_changed = sum(1 for p in puzzles if count_given_changed(p) > 0)
        stats_rows.append(f"<tr><td>{n} missing</td><td>{len(puzzles)}</td><td>{n_correct}</td><td>{100*n_correct/len(puzzles):.0f}%</td><td>{n_given_changed}</td></tr>")
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Easy Sudoku Analysis - HRM Model</title>{css}</head>
<body>
<h1>🧩 Easy Sudoku Analysis: Does HRM Generalize?</h1>
<div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: HRM (Hierarchical Reasoning Model)</div>

<div class="key-finding">
<h3>🔑 Key Finding</h3>
<p>The HRM model was trained exclusively on <b>EXTREME difficulty</b> Sudoku puzzles with ~55 missing cells on average.
This analysis tests whether the model can generalize to <b>trivially easy</b> puzzles with only 1-5 missing cells.</p>
<p><b>Result:</b> The model achieves <b>{100*correct/total:.0f}%</b> accuracy on easy puzzles ({correct}/{total} correct).
This suggests the model {"has learned generalizable constraint reasoning" if correct > total * 0.8 else "struggles to generalize to simpler problems despite mastering harder ones"}.</p>
</div>

<div class="critical-issue">
<h3>🚨 Critical Issue: Model Changes Given Cells</h3>
<p>The model is not just failing to fill in missing cells - it's <b>actively changing cells that were provided as input</b>!</p>
<ul>
<li><b>{puzzles_with_given_changed}/{total}</b> puzzles ({100*puzzles_with_given_changed/total:.0f}%) had given cells incorrectly modified</li>
<li><b>{total_given_changed}</b> total given cells were changed across all puzzles</li>
</ul>
<p><b>Why this happens:</b> The model outputs predictions for all 81 cells, not just the missing ones.
During training on extreme puzzles (~55 missing cells), the model learned to predict most cells.
When presented with easy puzzles (1-5 missing cells), it still tries to "solve" all cells,
overwriting the given inputs with its predictions.</p>
</div>

<div class="summary {summary_class}">
<h3>Summary Statistics</h3>
<table class="stats-table">
<tr><th>Difficulty</th><th>Puzzles</th><th>Correct</th><th>Accuracy</th><th>Given Changed</th></tr>
{''.join(stats_rows)}
<tr style="font-weight: bold;"><td>Total</td><td>{total}</td><td>{correct}</td><td>{100*correct/total:.0f}%</td><td>{puzzles_with_given_changed}</td></tr>
</table>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-box" style="background:#9AB0FF;"></div> Given (input cell)</div>
    <div class="legend-item"><div class="legend-box" style="background:#57B97B;"></div> Correct prediction</div>
    <div class="legend-item"><div class="legend-box" style="background:#E86B6B;"></div> Incorrect/violation</div>
    <div class="legend-item"><div class="legend-box" style="box-shadow: inset 0 0 0 3px #ff9800;"></div> Originally missing cell</div>
</div>

<h2>Detailed Results</h2>
{''.join(sections)}

<h2>Methodology</h2>
<div class="analysis">
<p>Easy puzzles were created by:</p>
<ol>
<li>Taking complete Sudoku solutions from the test set (ground truth labels)</li>
<li>Randomly blanking out 1-5 cells to create trivially easy puzzles</li>
<li>Running the HRM model for 16 ACT steps (same as training)</li>
<li>Comparing the final prediction to the known solution</li>
</ol>
<p>For a puzzle with only 1 missing cell, there is exactly one valid digit that satisfies all three constraints
(row, column, and 3x3 box). This is the simplest possible Sudoku reasoning task.</p>
</div>

<h2>Observations</h2>
<div class="analysis">
<p>Looking at the intermediate steps reveals how the model approaches these puzzles:</p>
<ul>
<li><b>Iterative refinement:</b> The model often starts with incorrect guesses and refines over steps</li>
<li><b>Constraint violations:</b> Some predictions violate basic Sudoku rules (duplicates in row/col/box)</li>
<li><b>Training distribution shift:</b> The model may be overfitting to patterns seen in extreme puzzles</li>
</ul>
</div>

</body></html>
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze HRM on easy Sudoku puzzles")
    parser.add_argument("--output", type=str, default="Sudoku_Reports/easy_puzzle_analysis.html")
    parser.add_argument("--num-puzzles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-missing", type=int, default=5)
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    print("Loading pre-computed predictions...")
    z = np.load(NPZ_PATH, allow_pickle=True)
    labels = z['labels']  # These are complete solutions
    intermediate = z['intermediate_preds']  # Shape: [16, N, 81]
    
    print(f"Found {len(labels)} puzzles in dataset")
    print(f"Intermediate steps shape: {intermediate.shape}")
    
    # Create easy puzzles from the solutions
    print(f"\nCreating {args.num_puzzles} easy puzzles (1-{args.max_missing} missing cells)...")
    
    easy_puzzles = []
    
    # Select random solutions
    selected_indices = rng.choice(len(labels), size=min(args.num_puzzles * 2, len(labels)), replace=False)
    
    for i, sol_idx in enumerate(selected_indices):
        if len(easy_puzzles) >= args.num_puzzles:
            break
            
        solution = labels[sol_idx]
        
        # Determine number of missing cells (cycle through 1-5)
        num_missing = (i % args.max_missing) + 1
        
        # Create easy puzzle
        easy_input, missing_indices = create_easy_puzzle(solution, num_missing, rng)
        
        easy_puzzles.append({
            'input': easy_input,
            'label': solution,
            'missing_indices': missing_indices,
            'original_idx': sol_idx
        })
    
    print(f"Created {len(easy_puzzles)} easy puzzles")
    
    # Run model on easy puzzles
    print("\nLoading model and running inference...")
    results = load_model_and_run_easy_puzzles(easy_puzzles, max_steps=16)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, args.output)
    
    # Print summary
    correct = sum(1 for r in results if r['correct'])
    print(f"\n{'='*50}")
    print(f"SUMMARY: {correct}/{len(results)} easy puzzles solved correctly ({100*correct/len(results):.1f}%)")
    print(f"{'='*50}")
    
    for r in results:
        num_missing = len(r['missing_indices'])
        status = "✓" if r['correct'] else "✗"
        print(f"  Puzzle (original #{r['original_idx']}, {num_missing} missing): {status}")


if __name__ == "__main__":
    main()
