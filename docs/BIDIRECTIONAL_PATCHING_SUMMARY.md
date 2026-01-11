# Bidirectional Activation Patching - Implementation Summary

## Changes Made

### 1. **Source Puzzle Display in HTML Reports**
   - Updated `make_colored_html_report()` function signature to accept optional `source_input` and `source_labels` parameters
   - HTML reports now display the source puzzle at the beginning (input and labels)
   - Applied to both forward and inverse experiment reports

### 2. **Forward Experiment Enhancements**
   - Added "direction" field to context dictionary: `"forward (source→target)"`
   - Forward experiment YAML files now have `_forward` suffix (e.g., `patch_s0_t250_both_forward.yaml`)
   - Forward experiment HTML reports now have `_forward` suffix (e.g., `activation_patching_report_forward.html`)
   - Source puzzle data is passed to HTML report for display

### 3. **Inverse Experiment Implementation**
   - New inverse experiment loop patches **target activations into source puzzle** (opposite direction)
   - For each `--num_runs`, runs the model on source batch with target's cached activations
   - Computes metrics, stepwise comparisons, and intermediate predictions
   - Generates separate YAML results with `_inverse` suffix
   - Generates separate HTML report with `_inverse` suffix
   - In the inverse report, the "source puzzle display" now shows the target puzzle (since we're patching target→source)

### 4. **Activation Caches**
   - Updated activation cache structure to include both `patched_forward` and `patched_inverse` keys
   - Allows post-experiment analysis of both directions

### 5. **File Naming Convention**
   - **Forward YAML**: `patch_sX_tY_LEVEL_forward.yaml`
   - **Inverse YAML**: `patch_sX_tY_LEVEL_inverse.yaml`
   - **Forward HTML**: `activation_patching_report_forward.html`
   - **Inverse HTML**: `activation_patching_report_inverse.html`
   - **Activation Cache**: `activations_sX_tY.pt` (contains both forward and inverse caches)

## How It Works

### Forward Experiment (Original)
```
Source Puzzle → Extract Activations → Use as Patch on Target Puzzle
Target's z_H/z_L get replaced with Source's z_H/z_L at specified steps
Result: Does target puzzle follow source's solution when using source's reasoning?
```

### Inverse Experiment (New)
```
Target Puzzle → Extract Activations → Use as Patch on Source Puzzle
Source's z_H/z_L get replaced with Target's z_H/z_L at specified steps
Result: Does source puzzle follow target's solution when using target's reasoning?
```

## Example Commands

### Basic Bidirectional Patching (z_H and z_L both)
```bash
cd /home/ubuntu/HRM
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 250 \
    --patch_level both \
    --num_runs 5 \
    --report_html activation_patching_report.html \
    --output_dir results/activation_patching_e2e
```

### Patch Only z_H (Global/Controller Stream)
```bash
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 250 \
    --patch_level H \
    --num_runs 5 \
    --report_html activation_patching_report.html \
    --output_dir results/activation_patching_e2e
```

### Patch Only z_L (Local/Per-cell Stream)
```bash
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 250 \
    --patch_level L \
    --num_runs 5 \
    --report_html activation_patching_report.html \
    --output_dir results/activation_patching_e2e
```

### Patch Specific Steps Only
```bash
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 250 \
    --patch_level both \
    --patch_steps 1,2,3 \
    --num_runs 5 \
    --report_html activation_patching_report.html \
    --output_dir results/activation_patching_e2e
```

## Output Files

After running the script, the `--output_dir` will contain:

```
results/activation_patching_e2e/
├── patch_s0_t250_both_forward.yaml          # Forward experiment results
├── patch_s0_t250_both_inverse.yaml          # Inverse experiment results
├── activation_patching_report_forward.html  # Forward report with source puzzle display
├── activation_patching_report_inverse.html  # Inverse report with target puzzle display
└── activations_s0_t250.pt                   # Cached activations for both directions
```

## Key Metrics in Reports

### Forward Experiment
- **baseline_accuracy_run0**: Accuracy of target puzzle on its own
- **patched_accuracy_run0**: Accuracy of target puzzle when using source's reasoning
- **direction**: "forward (source→target)"
- **Source Puzzle Display**: Shows source puzzle's input and labels

### Inverse Experiment
- **baseline_accuracy_run0**: Accuracy of source puzzle on its own
- **patched_accuracy_run0**: Accuracy of source puzzle when using target's reasoning
- **direction**: "inverse (target→source)"
- **Source Puzzle Display**: Shows target puzzle's input and labels (since we're patching target into source)

## Interpretation Guide

**High Forward patched_accuracy + Low Inverse patched_accuracy**
→ Source's reasoning is more "correct" than target's; target benefits from source's activations but source doesn't benefit from target's

**High Forward patched_accuracy + High Inverse patched_accuracy**
→ Both puzzles' reasoning streams are similarly effective; they encode similar information

**Low Forward patched_accuracy + High Inverse patched_accuracy**
→ Target's reasoning is more "correct"; target doesn't benefit from source but source benefits from target

**Low Both**
→ The two puzzles' reasoning streams are quite different; little cross-puzzle influence
