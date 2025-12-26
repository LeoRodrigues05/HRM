# Activation Patching for HRM

This directory contains scripts for running activation patching experiments on the Hierarchical Reasoning Model (HRM).

## Overview

Activation patching is a causal intervention technique where we:
1. Run a **source puzzle** through the model and cache its activations (z_H, z_L)
2. Run a **target puzzle** through the model normally (baseline)
3. Run the **target puzzle** again, but replace (patch) its activations with those from the source puzzle at specific steps/layers
4. Compare the performance change to understand what information is encoded in the activations

The key insight is that even when two puzzles require different actions (e.g., filling row 1 vs row 3 in Sudoku), we can test whether activations from one puzzle are useful for solving another, revealing what abstract representations the model learns.

## Scripts

### 1. `activation_patching.py`

Run a single activation patching experiment.

**Usage:**
```bash
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 1 \
    --patch_level both \
    --patch_steps 0,1,2 \
    --max_steps 8 \
    --output_dir results/activation_patching
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--source_puzzle_idx`: Index of source puzzle in test set
- `--target_puzzle_idx`: Index of target puzzle in test set
- `--patch_level`: Which activations to patch (`H`, `L`, or `both`)
- `--patch_steps`: Which reasoning steps to patch (comma-separated, e.g., `0,1,2`). Use `all` to patch all steps
- `--patch_positions`: Which sequence positions to patch (comma-separated). Use `all` to patch all positions
- `--max_steps`: Maximum number of reasoning steps
- `--output_dir`: Directory to save results
- `--device`: Device to use (`cuda` or `cpu`)

**Output:**
- YAML file with metrics (accuracy before/after patching, accuracy change)
- PyTorch file with cached activations
- Console output with detailed analysis

### 2. `batch_activation_patching.py`

Run multiple activation patching experiments systematically.

**Usage:**
```bash
python scripts/batch_activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt \
    --num_puzzles 10 \
    --patch_levels H,L,both \
    --patch_steps_configs all,0,1,2 \
    --mode pairwise \
    --output_dir results/activation_patching_batch
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--num_puzzles`: Number of puzzles to test from test set
- `--patch_levels`: Comma-separated list of patch levels to test
- `--patch_steps_configs`: Comma-separated list of step configurations
- `--mode`: Experiment mode:
  - `pairwise`: Test all pairs of puzzles (comprehensive but slow)
  - `one_to_many`: Test first puzzle against all others (faster)
- `--max_steps`: Maximum reasoning steps
- `--output_dir`: Output directory
- `--device`: Device to use

**Output:**
- Individual result files for each experiment
- `summary.json` with aggregated results
- Console output with statistics

## Understanding the Results

### Key Metrics

1. **Source Accuracy**: How well the model solves the source puzzle
2. **Target Baseline Accuracy**: How well the model solves the target puzzle normally
3. **Target Patched Accuracy**: How well the model solves the target puzzle with patched activations
4. **Accuracy Change**: `patched - baseline` (positive = patching helped, negative = patching hurt)

### Interpretation

**Large Negative Accuracy Change**: The patched activations disrupted the target puzzle solution, suggesting that:
- The activations encode puzzle-specific information
- Different puzzles require different representations at that layer/step

**Small Accuracy Change (near zero)**: The patched activations had minimal impact, suggesting:
- The activations at that layer/step encode general information
- The model can recover from the intervention

**Positive Accuracy Change**: The patched activations improved performance (rare), suggesting:
- The source puzzle activations captured useful general strategies
- The target puzzle was harder than the source

### Example Experiment

**Scenario**: Test if row-detection activations transfer between puzzles

```bash
# Patch only the H-level (high-level reasoning) at step 2
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt \
    --source_puzzle_idx 5 \
    --target_puzzle_idx 12 \
    --patch_level H \
    --patch_steps 2 \
    --output_dir results/row_detection_test
```

If accuracy drops significantly, it suggests that step 2 of the H-level encodes puzzle-specific row information that doesn't transfer.

## Advanced Usage

### Position-Specific Patching

Patch only specific sequence positions (e.g., first row tokens in Sudoku):

```bash
python scripts/activation_patching.py \
    --checkpoint <checkpoint> \
    --source_puzzle_idx 0 \
    --target_puzzle_idx 1 \
    --patch_level both \
    --patch_positions 0,1,2,3,4,5,6,7,8 \
    --output_dir results/first_row_patch
```

### Step-by-Step Analysis

Test each reasoning step individually:

```bash
for step in 0 1 2 3 4 5; do
    python scripts/activation_patching.py \
        --checkpoint <checkpoint> \
        --source_puzzle_idx 0 \
        --target_puzzle_idx 1 \
        --patch_level both \
        --patch_steps $step \
        --output_dir results/step_by_step
done
```

### Layer Ablation

Test which level (H or L) is more important:

```bash
# Test H-level only
python scripts/activation_patching.py --patch_level H ...

# Test L-level only
python scripts/activation_patching.py --patch_level L ...

# Test both levels
python scripts/activation_patching.py --patch_level both ...
```

## Implementation Details

### Model Architecture

The HRM model has two levels of activations:
- **z_H**: High-level reasoning states (updated every H_cycle)
- **z_L**: Low-level reasoning states (updated every L_cycle)

Both are tensors of shape `[batch_size, seq_len, hidden_size]`.

### Patching Mechanism

The `ActivationPatcher` class:
1. Caches activations during a forward pass
2. During patched forward pass, replaces the inner carry activations before each step
3. Allows selective patching by layer (H/L), step, and position

### Memory Considerations

- Activations are cached on CPU to save GPU memory
- Use smaller batch sizes (typically 1) for patching experiments
- Consider reducing `max_steps` if memory is limited

## Citation

If you use this code in your research, please cite the original HRM paper and mention the activation patching technique.

## Troubleshooting

**Issue**: Out of memory errors
- Solution: Reduce `max_steps`, patch fewer positions, or use CPU

**Issue**: Different puzzle shapes
- Solution: The code handles variable sequence lengths automatically

**Issue**: No checkpoint found
- Solution: Ensure you've trained a model first using `pretrain.py`

**Issue**: Accuracy changes are all near zero
- Solution: Try patching at different steps or levels; some layers may be more important than others
