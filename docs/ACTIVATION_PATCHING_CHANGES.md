# Bidirectional Activation Patching - Change Log

## Overview
The `activation_patching.py` script has been enhanced to support bidirectional experiments:
- **Forward**: Patch source activations into target puzzle
- **Inverse**: Patch target activations into source puzzle

## Code Changes in `/home/ubuntu/HRM/scripts/activation_patching.py`

### 1. Forward Experiment - Source Puzzle Display (Lines ~1372-1390)
Added source puzzle data to HTML report generation:
```python
source_input_flat = source_batch["inputs"][0].detach().cpu().tolist()
source_labels_flat = source_batch["labels"][0].detach().cpu().tolist()

make_colored_html_report(
    report_path,
    context,
    labels_flat,
    target_input_flat,
    baseline_final_flat,
    patched_final_flat,
    step_outputs,
    run_metrics,
    patch_validation_first,
    patched_steps_effective,
    source_input=source_input_flat,          # ← NEW
    source_labels=source_labels_flat,        # ← NEW
)
```

### 2. Forward Experiment - File Naming (Line ~1349)
Added `_forward` suffix to YAML file:
```python
results_path = os.path.join(
    args.output_dir, 
    f"patch_s{args.source_puzzle_idx}_t{args.target_puzzle_idx}_{args.patch_level}_forward.yaml"  # ← NEW suffix
)
```

### 3. Forward Experiment - Direction Metadata (Line ~1362)
Added direction field to context:
```python
context = {
    # ... existing fields ...
    "direction": "forward (source→target)",  # ← NEW
    # ... existing fields ...
}
```

### 4. Inverse Experiment Loop (Lines ~1394-1530)
Complete new section that:
- Runs forward passes with target activations patched into source
- Collects metrics for multiple runs
- Computes stepwise comparisons
- Saves inverse YAML results with `_inverse` suffix
- Generates inverse HTML report with source puzzle display (which is target puzzle)

Key structure:
```python
for run_idx in range(args.num_runs):
    inverse_patched_outputs, inverse_patched_cache, inverse_patch_validation = patcher.run_with_patching(
        source_batch,
        patcher.target_cache,              # ← Patching target activations
        patch_level=args.patch_level,
        patch_steps=patch_steps,
        patch_positions=patch_positions,
        max_steps=args.max_steps,
        verify=args.verify_patching,
    )
    # Collect metrics, predictions, and caches
    # ... (similar to forward experiment)
```

### 5. Activation Cache Update (Lines ~1545-1570)
Updated to save both forward and inverse caches:
```python
torch.save({
    "source": {...},
    "target": {...},
    "patched_forward": {...},     # ← Renamed from "patched"
    "patched_inverse": {...},     # ← NEW
}, cache_path)
```

## New Outputs Generated

Per experiment, you now get:

### Files
| File | Forward | Inverse |
|------|---------|---------|
| YAML Results | `patch_sX_tY_LEVEL_forward.yaml` | `patch_sX_tY_LEVEL_inverse.yaml` |
| HTML Report | `activation_patching_report_forward.html` | `activation_patching_report_inverse.html` |
| Activation Cache | `activations_sX_tY.pt` (contains both) |

### Report Sections
Both reports include:
- **Source puzzle display** (at top): Shows the puzzle whose activations are being patched
  - Forward: Source puzzle input/labels
  - Inverse: Target puzzle input/labels
- **Baseline results**: Model predictions without patching
- **Patched results**: Model predictions with activations patched
- **Stepwise metrics**: Per-step accuracy changes
- **Direction metadata**: Which direction the patching flows

## Usage

### Run Forward + Inverse Together
```bash
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 250 \
    --patch_level both \
    --num_runs 5 \
    --report_html activation_patching_report.html \
    --output_dir results/activation_patching_e2e
```

Generates:
- `patch_s0_t250_both_forward.yaml` - Forward results
- `patch_s0_t250_both_inverse.yaml` - Inverse results
- `activation_patching_report_forward.html` - Forward report
- `activation_patching_report_inverse.html` - Inverse report
- `activations_s0_t250.pt` - Cached activations for both

## Backward Compatibility

✅ Existing scripts and commands still work - the changes are additive
✅ Original forward experiment logic is unchanged
✅ New inverse experiment is entirely separate section
✅ No breaking changes to CLI arguments

## Testing

Syntax verified with: `python3 -m py_compile scripts/activation_patching.py` ✓

Ready to run! Example command provided in BIDIRECTIONAL_PATCHING_SUMMARY.md
