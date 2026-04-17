# Query 02: Tighten Existing Experiments

## Objective
Re-run all existing MI experiments (E1, E2b, E5, E9) with proper single-variable controls, larger sample sizes, confidence intervals, and matched random baselines. Also add the missing z_L ablation experiment. This converts the current exploratory findings into publication-quality evidence.

---

## Prompt to Execute

Copy and paste the following into the AI assistant:

---

### QUERY START

I need to tighten the existing MI experiments in `/home/leo.rodrigues/HRM/` for publication. Each experiment needs larger N, single-variable controls, and proper statistics. Please do the following, creating new scripts where needed and modifying existing ones:

#### 1. Controlled z_H Ablation — Revised E1 (`scripts/controlled_ablation_zH.py`)

Create a new script (based on `scripts/batch_ablation_1k.py`) that:

**Design**: For each of 16 steps (0–15), run a SEPARATE ablation experiment where z_H is zeroed at ONLY that one step. All other steps run normally. This is the single-variable version of E1.

**Requirements**:
- N ≥ 5,000 puzzles (use the full test set or a consistent random subset with `--seed 42`)
- For EACH puzzle, run:
  - 1 baseline forward pass (no ablation) → record per-step accuracy
  - 16 single-step ablation runs (ablate step 0 only, step 1 only, ..., step 15 only) → record per-step accuracy
  - 1 all-steps ablation run (ablate every step) → record final accuracy
- Record per-puzzle: `{puzzle_id, baseline_accuracy, ablated_accuracy[step], delta_accuracy[step], baseline_step_accuracies[16], ablated_step_accuracies[16]}`
- Compute aggregate: mean Δaccuracy per ablated step, with 95% CI (bootstrap, 1000 resamples)
- Save: `results/controlled_ablation_zH/per_puzzle.jsonl`, `results/controlled_ablation_zH/aggregate.json`

**Key plot**: Δaccuracy vs. ablated step — a single clean curve with error bars showing when z_H matters most.

Additionally, create a **z_L ablation complement** — same design but ablating z_L instead of z_H at each step:
- This proves z_L is recoverable (small Δ expected since z_L gets reset from z_H + input)
- Save to `results/controlled_ablation_zL/`

#### 2. Extended Freeze Experiment — Revised E2b (`scripts/controlled_freeze.py`)

Create a new script (based on `scripts/batch_freeze_h.py`) that:

**Design**: Freeze z_H (or z_L) after step k, for every k in {0, 1, 2, ..., 15}. From step k onward, z_H is fixed (no updates from H_level); z_L continues updating normally.

**Requirements**:
- N ≥ 1,000 puzzles (consistent set, `--seed 42`)
- For EACH puzzle, run:
  - 1 baseline (no freeze) → record 16-step accuracy trajectory
  - 16 freeze-z_H runs (freeze after step 0, 1, ..., 15) → record accuracy trajectories
  - 16 freeze-z_L runs (freeze after step 0, 1, ..., 15) → record accuracy trajectories (CONTROL)
- Record per-puzzle results with both z_H and z_L freeze for direct comparison
- Compute aggregate: mean accuracy drop per freeze step, with 95% CI
- Save: `results/controlled_freeze/per_puzzle.jsonl`, `results/controlled_freeze/aggregate.json`

**Key plots**:
- z_H freeze vs z_L freeze at same step: shows asymmetry (z_H freeze hurts more)
- Final accuracy vs. freeze step: shows the 3-phase convergence pattern (rapid → refine → converge)

#### 3. Full Time-Shift Transfer Matrix — Revised E5 (`scripts/controlled_time_shift.py`)

Create a new script (based on `scripts/batch_time_shift.py`) that:

**Design**: For the SAME puzzle, inject z_H from step `source` into step `target`, for all pairs (source, target) where source ≠ target.

**Requirements**:
- N ≥ 500 puzzles (consistent set, `--seed 42`)
- For EACH puzzle:
  - Run baseline forward pass → cache z_H at every step [0..15]
  - For each (source_step, target_step) pair: run modified forward pass where z_H at target_step is replaced with cached z_H from source_step → record final accuracy
- This produces a 16×16 transfer matrix per puzzle
- Compute aggregate: mean transfer matrix with 95% CI per cell
- Save: `results/controlled_time_shift/per_puzzle.jsonl`, `results/controlled_time_shift/transfer_matrix.json`

**Key plot**: 16×16 heatmap showing Δaccuracy for each (source → target) transfer. Upper triangle = future→past (should help), lower triangle = past→future (should hurt). Diagonal = no transfer (baseline).

#### 4. Statistical Directed Ablation — Revised E9 (`scripts/controlled_directed_ablation.py`)

Create a new script (based on `scripts/e9_directed_ablation.py`) that:

**Design**: Project out probe directions from z_H, with proper statistical controls.

**Requirements**:
- N ≥ 500 puzzles (consistent set, `--seed 42`)
- Load probe weights from E8 (`results/e8_constraint_probes/probe_weights.pt`)
- For EACH probe direction (row, col, box, per_cell_correct, is_given, cell_digit):
  - Run ablation: project out that single direction from z_H at all steps
  - Run 10 matched random controls: project out a RANDOM unit vector from z_H at all steps (different random vector each time, same norm as probe direction)
- Record per-puzzle:
  - Direction-specific Δaccuracy (e.g., ablate row direction → Δrow_violations, Δcol_violations, Δbox_violations, Δcell_accuracy)
  - Random control Δaccuracy (mean ± std across 10 random vectors)
- Statistical tests:
  - Paired t-test: probe direction vs. random baseline (per puzzle, then aggregate)
  - Bootstrap CI on the difference
  - Report Cohen's d effect size
- **Multi-direction ablation**: Also try projecting out 2 and 3 probe directions simultaneously (the full ~3-d constraint subspace from E8). If the 3-d subspace ablation has a larger effect than any single direction, it suggests the info is distributed across the subspace.
- Save: `results/controlled_directed_ablation/per_puzzle.jsonl`, `results/controlled_directed_ablation/aggregate.json`, `results/controlled_directed_ablation/statistical_tests.json`

**Key plots**:
- Bar chart: Δaccuracy per direction + random baseline, with error bars
- Specificity matrix: [ablated_direction × violation_type] with significance stars
- Multi-direction ablation effect: 1-d vs 2-d vs 3-d subspace ablation

#### 5. Master Runner Script (`scripts/run_all_controlled_experiments.py`)

Create a script that:
- Runs all 4 experiments above in sequence
- Uses consistent puzzle set (same seed, same N)
- Log timing and results summary
- Can be run with `--quick` flag for testing (N=20 per experiment)

#### 6. Unified Plotting Script (`scripts/plot_controlled_experiments.py`)

Create a publication-quality plotting script that:
- Reads all results from `results/controlled_*/`
- Generates all key figures with consistent styling (matplotlib, LaTeX labels, serif font)
- All plots must have error bars (95% CI)
- Save PNG + PDF to `results/controlled_experiments_plots/`
- Figures:
  1. z_H vs z_L ablation comparison (Δacc per step, both on same axes)
  2. Freeze z_H vs freeze z_L (final acc vs freeze step)
  3. Time-shift 16×16 heatmap
  4. Directed ablation bar chart with random baseline
  5. Multi-direction subspace ablation
  6. Combined 2×3 figure for paper

#### Important Implementation Notes

- Base all new scripts on the existing infrastructure — study these files first:
  - `scripts/batch_ablation_1k.py` — ablation framework (model loading, forward pass hooks)
  - `scripts/batch_freeze_h.py` — freeze mechanism
  - `scripts/batch_time_shift.py` — time-shift mechanism
  - `scripts/e9_directed_ablation.py` — directed ablation with probe weights
  - `scripts/activation_ablation.py` — `ActivationAblator` class, `ACTModel` wrapper
- Use the SAME consistent puzzle subset across ALL experiments (define once, reuse)
- All experiments should support `--device cpu` and `--device cuda`
- Default checkpoint path should be auto-detected from `Checkpoint_HRM_Sudoku/` or configurable via `--checkpoint`

### QUERY END

---

## Expected Outcomes

1. Four new controlled experiment scripts in `scripts/`
2. Master runner and unified plotter
3. Results in `results/controlled_*/` directories
4. Publication-quality figures with error bars in `results/controlled_experiments_plots/`

## Success Criteria

- E1 single-step ablation shows monotonically increasing importance for later steps
- z_L ablation shows <5% effect (confirming z_L is recoverable)
- Freeze z_H > freeze z_L asymmetry confirmed with statistical significance
- Transfer matrix shows clear upper/lower triangle pattern
- Directed ablation shows probe directions indistinguishable from random (p > 0.05)
- Multi-direction ablation (3-d subspace) may show slightly larger effect than single direction
- All plots have 95% CI error bars
-
-----------------------------------------------

Perfect thank you for doing that, now I would like for you to run the second script, i.e., #file:02_tighten_experiments.md . Make sure to be accurate, at the of running the experiments let me know what you could be doing wrong, in addition, for the time poaching experiment, I want you to tighten it by keeping one step consistent and then testing it. In addition, you can find the  The rest of the query is as below: 

### QUERY START

I need to tighten the existing MI experiments in `/home/leo.rodrigues/HRM/` for publication. Each experiment needs larger N, single-variable controls, and proper statistics. Please do the following, creating new scripts where needed and modifying existing ones:

#### 1. Controlled z_H Ablation — Revised E1 (`scripts/controlled_ablation_zH.py`)

Create a new script (based on `scripts/batch_ablation_1k.py`) that:

**Design**: For each of 16 steps (0–15), run a SEPARATE ablation experiment where z_H is zeroed at ONLY that one step. All other steps run normally. This is the single-variable version of E1.

**Requirements**:
- N ≥ 5,000 puzzles (use the full test set or a consistent random subset with `--seed 42`)
- For EACH puzzle, run:
  - 1 baseline forward pass (no ablation) → record per-step accuracy
  - 16 single-step ablation runs (ablate step 0 only, step 1 only, ..., step 15 only) → record per-step accuracy
  - 1 all-steps ablation run (ablate every step) → record final accuracy
- Record per-puzzle: `{puzzle_id, baseline_accuracy, ablated_accuracy[step], delta_accuracy[step], baseline_step_accuracies[16], ablated_step_accuracies[16]}`
- Compute aggregate: mean Δaccuracy per ablated step, with 95% CI (bootstrap, 1000 resamples)
- Save: `results/controlled_ablation_zH/per_puzzle.jsonl`, `results/controlled_ablation_zH/aggregate.json`

**Key plot**: Δaccuracy vs. ablated step — a single clean curve with error bars showing when z_H matters most.

Additionally, create a **z_L ablation complement** — same design but ablating z_L instead of z_H at each step:
- This proves z_L is recoverable (small Δ expected since z_L gets reset from z_H + input)
- Save to `results/controlled_ablation_zL/`

#### 2. Extended Freeze Experiment — Revised E2b (`scripts/controlled_freeze.py`)

Create a new script (based on `scripts/batch_freeze_h.py`) that:

**Design**: Freeze z_H (or z_L) after step k, for every k in {0, 1, 2, ..., 15}. From step k onward, z_H is fixed (no updates from H_level); z_L continues updating normally.

**Requirements**:
- N ≥ 1,000 puzzles (consistent set, `--seed 42`)
- For EACH puzzle, run:
  - 1 baseline (no freeze) → record 16-step accuracy trajectory
  - 16 freeze-z_H runs (freeze after step 0, 1, ..., 15) → record accuracy trajectories
  - 16 freeze-z_L runs (freeze after step 0, 1, ..., 15) → record accuracy trajectories (CONTROL)
- Record per-puzzle results with both z_H and z_L freeze for direct comparison
- Compute aggregate: mean accuracy drop per freeze step, with 95% CI
- Save: `results/controlled_freeze/per_puzzle.jsonl`, `results/controlled_freeze/aggregate.json`

**Key plots**:
- z_H freeze vs z_L freeze at same step: shows asymmetry (z_H freeze hurts more)
- Final accuracy vs. freeze step: shows the 3-phase convergence pattern (rapid → refine → converge)

#### 3. Full Time-Shift Transfer Matrix — Revised E5 (`scripts/controlled_time_shift.py`)

Create a new script (based on `scripts/batch_time_shift.py`) that:

**Design**: For the SAME puzzle, inject z_H from step `source` into step `target`, for all pairs (source, target) where source ≠ target.

**Requirements**:
- N ≥ 500 puzzles (consistent set, `--seed 42`)
- For EACH puzzle:
  - Run baseline forward pass → cache z_H at every step [0..15]
  - For each (source_step, target_step) pair: run modified forward pass where z_H at target_step is replaced with cached z_H from source_step → record final accuracy
- This produces a 16×16 transfer matrix per puzzle
- Compute aggregate: mean transfer matrix with 95% CI per cell
- Save: `results/controlled_time_shift/per_puzzle.jsonl`, `results/controlled_time_shift/transfer_matrix.json`

**Key plot**: 16×16 heatmap showing Δaccuracy for each (source → target) transfer. Upper triangle = future→past (should help), lower triangle = past→future (should hurt). Diagonal = no transfer (baseline).

#### 4. Statistical Directed Ablation — Revised E9 (`scripts/controlled_directed_ablation.py`)

Create a new script (based on `scripts/e9_directed_ablation.py`) that:

**Design**: Project out probe directions from z_H, with proper statistical controls.

**Requirements**:
- N ≥ 500 puzzles (consistent set, `--seed 42`)
- Load probe weights from E8 (`results/e8_constraint_probes/probe_weights.pt`)
- For EACH probe direction (row, col, box, per_cell_correct, is_given, cell_digit):
  - Run ablation: project out that single direction from z_H at all steps
  - Run 10 matched random controls: project out a RANDOM unit vector from z_H at all steps (different random vector each time, same norm as probe direction)
- Record per-puzzle:
  - Direction-specific Δaccuracy (e.g., ablate row direction → Δrow_violations, Δcol_violations, Δbox_violations, Δcell_accuracy)
  - Random control Δaccuracy (mean ± std across 10 random vectors)
- Statistical tests:
  - Paired t-test: probe direction vs. random baseline (per puzzle, then aggregate)
  - Bootstrap CI on the difference
  - Report Cohen's d effect size
- **Multi-direction ablation**: Also try projecting out 2 and 3 probe directions simultaneously (the full ~3-d constraint subspace from E8). If the 3-d subspace ablation has a larger effect than any single direction, it suggests the info is distributed across the subspace.
- Save: `results/controlled_directed_ablation/per_puzzle.jsonl`, `results/controlled_directed_ablation/aggregate.json`, `results/controlled_directed_ablation/statistical_tests.json`

**Key plots**:
- Bar chart: Δaccuracy per direction + random baseline, with error bars
- Specificity matrix: [ablated_direction × violation_type] with significance stars
- Multi-direction ablation effect: 1-d vs 2-d vs 3-d subspace ablation

#### 5. Master Runner Script (`scripts/run_all_controlled_experiments.py`)

Create a script that:
- Runs all 4 experiments above in sequence
- Uses consistent puzzle set (same seed, same N)
- Log timing and results summary
- Can be run with `--quick` flag for testing (N=20 per experiment)

#### 6. Unified Plotting Script (`scripts/plot_controlled_experiments.py`)

Create a publication-quality plotting script that:
- Reads all results from `results/controlled_*/`
- Generates all key figures with consistent styling (matplotlib, LaTeX labels, serif font)
- All plots must have error bars (95% CI)
- Save PNG + PDF to `results/controlled_experiments_plots/`
- Figures:
  1. z_H vs z_L ablation comparison (Δacc per step, both on same axes)
  2. Freeze z_H vs freeze z_L (final acc vs freeze step)
  3. Time-shift 16×16 heatmap
  4. Directed ablation bar chart with random baseline
  5. Multi-direction subspace ablation
  6. Combined 2×3 figure for paper

#### Important Implementation Notes

- Base all new scripts on the existing infrastructure — study these files first:
  - `scripts/batch_ablation_1k.py` — ablation framework (model loading, forward pass hooks)
  - `scripts/batch_freeze_h.py` — freeze mechanism
  - `scripts/batch_time_shift.py` — time-shift mechanism
  - `scripts/e9_directed_ablation.py` — directed ablation with probe weights
  - `scripts/activation_ablation.py` — `ActivationAblator` class, `ACTModel` wrapper
- Use the SAME consistent puzzle subset across ALL experiments (define once, reuse)
- All experiments should support `--device cpu` and `--device cuda`
- Default checkpoint path should be auto-detected from `Checkpoint_HRM_Sudoku/` or configurable via `--checkpoint`

### QUERY END

