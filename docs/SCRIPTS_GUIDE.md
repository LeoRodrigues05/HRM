# HRM Bash Scripts Guide

This document describes all bash scripts available for working with the Hierarchical Reasoning Model (HRM).

## Quick Reference

| Operation | Script |
|-----------|--------|
| Initial Setup | `Initialize_HRM_Repo.sh` |
| Linear Probes | `scripts/probe_commands.sh` |
| Activation Patching | `scripts/run_activation_patching_examples.sh` |
| Single Inference | `scripts/single_inference.sh` |
| Full Evaluation | `Result_Generator_HRM.sh` |

---

## 1. Initial Setup (`Initialize_HRM_Repo.sh`)

**Purpose**: Set up the repository from scratch with all dependencies.

**What it does**:
1. Clones the HRM repository
2. Creates Python 3.10 virtual environment using `uv`
3. Installs CUDA 12.6 (if not present)
4. Installs PyTorch and FlashAttention
5. Installs all Python requirements
6. Builds the Sudoku dataset
7. Runs initial evaluation

**Usage**:
```bash
./Initialize_HRM_Repo.sh
```

**Prerequisites**:
- Linux system with NVIDIA GPU
- `git`, `curl`, `wget`
- Root access for CUDA installation (optional)

---

## 2. Linear Probes (`scripts/probe_commands.sh`)

**Purpose**: Train and evaluate linear probes on model activations to understand what information is encoded at different layers/steps.

**What it does**:
1. Activates virtual environment
2. Builds probe training dataset (if needed)
3. Collects activations from model on test puzzles
4. Trains linear probes on collected activations
5. Generates analysis of probe results

**Usage**:
```bash
cd /home/ubuntu/HRM
./scripts/probe_commands.sh
```

**Key Configuration**:
- `CHECKPOINT`: Model checkpoint path
- `DATA_DIR`: Dataset directory
- `PROBE_OUTPUT_DIR`: Where to save probe results

**Output**:
- Probe training logs
- Accuracy metrics per layer/step
- Analysis HTML/CSV files

---

## 3. Activation Patching (`scripts/run_activation_patching_examples.sh`)

**Purpose**: Run activation patching experiments to understand which activations causally influence model predictions.

**What it does**:
1. Runs 7 example patching experiments:
   - Full patching (both z_H and z_L, all steps)
   - H-level only (global reasoning stream)
   - L-level only (local cell stream)
   - Early steps (1-3)
   - Late steps (5-7)
   - Single step (step 3)
   - Position-specific (first row)
2. Generates forward AND inverse experiments for each
3. Creates HTML reports with colored Sudoku grids
4. Runs multiple times (default: 5) for stability verification

**Usage**:
```bash
cd /home/ubuntu/HRM
./scripts/run_activation_patching_examples.sh
```

**Customization**:
```bash
# Edit these variables at the top of the script:
CHECKPOINT="path/to/checkpoint.pt"
OUTPUT_DIR="results/activation_patching_examples"
NUM_RUNS=5
```

**Output per experiment**:
- `*_forward.yaml`: Forward patching results (source→target)
- `*_inverse.yaml`: Inverse patching results (target→source)
- `*_report.html`: HTML report for forward experiment
- `*_report_inverse.html`: HTML report for inverse experiment
- `activations_*.pt`: Cached activations

**Individual Experiment**:
```bash
python scripts/activation_patching.py \
    --checkpoint Checkpoint_HRM_Sudoku/.../checkpoint.pt \
    --source_puzzle_idx 111 \
    --target_puzzle_idx 220 \
    --patch_level both \
    --patch_steps 1,2,3,4,5,6,7 \
    --num_runs 5 \
    --report_html activation_patching_report.html \
    --output_dir results/my_experiment
```

---

## 4. Single Inference (`scripts/single_inference.sh`)

**Purpose**: Run inference on individual puzzles and visualize predictions.

**What it does**:
1. Loads model checkpoint
2. Runs inference on specified puzzles
3. Displays puzzle, prediction, and ground truth grids
4. Computes accuracy metrics (overall, unknown cells, known cells)
5. Saves results to YAML

**Usage**:
```bash
cd /home/ubuntu/HRM

# Run on first 10 puzzles
./scripts/single_inference.sh

# Run on specific puzzle
./scripts/single_inference.sh --puzzle_idx 42

# Run on first 20 puzzles
./scripts/single_inference.sh --num_puzzles 20

# Custom checkpoint and output
./scripts/single_inference.sh \
    --checkpoint path/to/checkpoint.pt \
    --output_dir results/my_inference
```

**Options**:
| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | Model checkpoint path | `Checkpoint_HRM_Sudoku/.../checkpoint.pt` |
| `--data_dir` | Dataset directory | `data/sudoku-extreme-1k-aug-1000` |
| `--output_dir` | Output directory | `results/single_inference` |
| `--num_puzzles` | Number of puzzles to run | 10 |
| `--puzzle_idx` | Specific puzzle index | (runs num_puzzles if not set) |

**Output**:
- Console: Printed Sudoku grids and accuracy metrics
- File: `inference_results.yaml` with detailed results

---

## 5. Full Evaluation (`Result_Generator_HRM.sh`)

**Purpose**: Run comprehensive evaluation on the full test set and generate reports.

**What it does**:
1. Runs distributed evaluation using `evaluate.py`
2. Converts results to NPZ format
3. Generates colored Sudoku report HTML
4. Computes aggregate metrics

**Usage**:
```bash
cd /home/ubuntu/HRM
./Result_Generator_HRM.sh
```

**Note**: This script runs evaluation asynchronously and waits for completion.

---

## Script Locations Summary

```
HRM/
├── Initialize_HRM_Repo.sh          # Initial setup
├── Result_Generator_HRM.sh         # Full evaluation
└── scripts/
    ├── probe_commands.sh           # Linear probes workflow
    ├── run_activation_patching_examples.sh  # Activation patching examples
    └── single_inference.sh         # Single puzzle inference
```

---

## Common Patterns

### Activating the Environment
All scripts auto-activate the virtual environment if it exists:
```bash
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
```

### Checkpoint Path
Most scripts use this checkpoint path:
```bash
CHECKPOINT="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt"
```

### Running from Repository Root
Always run scripts from the repository root:
```bash
cd /home/ubuntu/HRM
./scripts/some_script.sh
```
