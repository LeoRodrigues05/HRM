# Sudoku Experiment Controls Report

Generated from the completed Sudoku artifacts under `results/`, `scripts/`,
and `docs/` on 2026-06-08.

## Executive Summary

The Sudoku experiment suite has moved from exploratory mechanistic checks to a
partly hardened Phase 0 suite. The strongest controls are now in the controlled
intervention runs: each puzzle has its own clean baseline, interventions are
reported as paired deltas, per-puzzle rows are saved, and aggregate results use
bootstrap confidence intervals. Probe and directed-ablation results were also
hardened with puzzle-disjoint validation, five probe seeds, random-direction
controls, paired tests, effect sizes, and provenance metadata.

The strongest completed controls are:

| Control | Where it appears | Why it matters |
|---|---|---|
| Per-puzzle clean baseline | `results/controlled/*/per_puzzle.jsonl`, `results/sae_study/causal_ablation/per_puzzle.jsonl` | Prevents puzzle difficulty from being confounded with intervention effect. |
| Paired intervention deltas | Controlled ablation, freeze, time-shift, directed ablation, SAE causal ablation | Measures effect relative to the same puzzle's clean run. |
| Bootstrap 95% CIs | `results/controlled/*/aggregate.json`, updated `results/baseline_comparison/HRM_eval.json` | Quantifies puzzle-sampling uncertainty. |
| H vs L stream controls | `results/controlled/ablation/*`, `results/controlled/freeze/aggregate.json`, patching runs | Tests whether effects are specific to the high-level stream rather than generic hidden-state damage. |
| Random-direction controls | `results/controlled/directed_ablation/analysis.json`, older `results/directed_ablation/*` | Tests whether probe directions are causally special versus arbitrary subspaces. |
| Subspace ablations | `results/controlled/directed_ablation/analysis.json` | Checks whether row/col/box directions only matter jointly. |
| Multiple probe seeds | `results/probes/e8_constraint_probes/probe_summary.json` | Separates stable decodability from one lucky split. |
| Puzzle-disjoint probe split | `results/probes/e8_constraint_probes/_meta.json` | Prevents cells from the same Sudoku puzzle leaking between train and validation. |
| Architecture baselines | `results/baseline_comparison/*_eval.json` | Tests whether recurrence, not just HRM-specific hierarchy, explains performance. |
| SAE hyperparameter sweep | `results/sae_study/sweep_results.csv` | Avoids basing the SAE claim on a single dictionary size/L1 setting. |
| Run provenance | `_meta.json` in baseline, E8, directed-ablation, and SAE-sweep directories | Records git SHA, dirty state, command line, GPU, seed/config parameters. |

The main remaining gaps are uneven provenance coverage for some controlled
aggregate directories, older baseline files that still lack CI fields, and the
SAE causal aggregate on disk still being weaker than the controlled suite even
though the current script has been upgraded to compute bootstrap summaries and
paired tests.

## Completed Experiments And Controls

### 1. Baseline And Recurrence Comparison

Artifacts:

- `results/baseline_comparison/HRM_eval.json`
- `results/baseline_comparison/{PT,RNN,SRNN,UT}_best_eval.json`
- checkpoint sweeps such as `PT_step*_eval.json`, `RNN_step*_eval.json`,
  `SRNN_step*_eval.json`, and `UT_step*_eval.json`
- `results/metrics/difficulty_stratified/stratified_summary.json`
- `results/metrics/hamming.csv`, `results/hamming_multi/hamming_per_puzzle.csv`

Controls used:

- Architecture controls: HRM is compared to recurrent and non-recurrent
  baselines trained/evaluated on the Sudoku task. The saved best-checkpoint
  comparison includes HRM, Plain Transformer, Vanilla/Recurrent RNN style
  baselines, Standard RNN/GRU, and Universal Transformer variants.
- Per-step metrics: evaluation tracks cell accuracy, puzzle accuracy,
  unknown-cell accuracy, Hamming distance, and row/col/box violations for each
  ACT step.
- Task-specific validity metrics: Sudoku-specific row/col/box violation counts
  are tracked, preventing conclusions from relying on accuracy alone.
- Puzzle-level uncertainty: the updated HRM rerun stores bootstrap CIs per
  step. Example: HRM step 15 cell accuracy is 0.803 with CI [0.787, 0.819] on
  500 puzzles.
- Cross-puzzle hidden-state patching control in `HRM_eval.json`: 100 source
  target pairs, patching at even steps. Cross-puzzle patching is strongly
  damaging from step 2 onward, while step 0 is a null control with zero delta.
- Difficulty stratification: the difficulty-stratified run reports separate
  easy/medium/hard curves, showing recurrence helps across difficulty bands
  instead of only on easy puzzles.

Rigor status:

- Strong as an architectural and metric control.
- Mixed statistical strength across saved files: the newest HRM evaluation has
  CIs and `_meta.json`, but several older baseline JSONs still report mean and
  std only.

### 2. Controlled z_H / z_L Ablation

Artifacts:

- `results/controlled/ablation/zH/aggregate.json`
- `results/controlled/ablation/zH/per_puzzle.jsonl`
- `results/controlled/ablation/zH_1000/zH/aggregate.json`
- `results/controlled/ablation/zL_extended/zL/aggregate.json`
- historical: `results/ablation/batch_ablation_zH/aggregate_stats.json`

Controls used:

- Clean run per puzzle before any ablation.
- Single-variable interventions: ablate one stream at a time, either `z_H` or
  `z_L`.
- Single-step ablations for each step 0-15, plus an all-steps ablation.
- Per-puzzle paired deltas saved in JSONL.
- Bootstrap CIs in aggregate JSON.
- H/L comparison as an internal control. On the scaled controlled artifacts,
  all-step `z_H` and `z_L` ablations are both large, but their single-step
  profiles differ sharply. For example, `z_H` step 15 ablation is much more
  damaging than early ablation, while individual `z_L` step ablations are near
  zero even though all-steps `z_L` ablation is large.
- Historical ablation also tracked confidence, entropy, cells fixed, cells
  broken, cells changed, and decreased/unchanged/increased puzzle buckets.

Representative hardened results:

- `z_H` all-steps ablation, N=5000: mean delta accuracy -0.205, CI
  [-0.210, -0.200].
- `z_H` single-step step 0, N=5000: -0.078, CI [-0.083, -0.073].
- `z_H` single-step step 15, N=5000: -0.276, CI [-0.282, -0.271].
- `z_L_extended` all-steps ablation, N=1000: -0.193, CI [-0.203, -0.183].

Rigor status:

- Strong for `z_H` and the extended `z_L` run.
- Do not use `results/controlled/ablation/zL/aggregate.json` as the main `z_L`
  result: that file is N=20 and should be treated as smoke-test/exploratory.

### 3. Controlled Freeze

Artifacts:

- `results/controlled/freeze/aggregate.json`
- `results/controlled/freeze/per_puzzle.jsonl`
- historical: `results/freeze/freeze_h/aggregate_stats.json`

Controls used:

- Clean run per puzzle.
- Freeze-after-k intervention for every k=0-15.
- H/L stream comparison: freeze `z_H` while `z_L` continues, and freeze `z_L`
  while `z_H` continues.
- Bootstrap CIs over puzzle-level deltas.
- Historical run additionally tracked hurt/helped/unchanged puzzle counts and
  mean broken cells.

Representative hardened results:

- Freeze `z_H` at k=0, N=1000: delta -0.091, CI [-0.099, -0.083].
- Freeze `z_H` at k=5, N=1000: delta -0.015, CI [-0.019, -0.011].
- Freeze `z_H` at k=11-15: approximately null.
- Freeze `z_L` at most k: CIs generally cross or hug zero, showing that the
  freeze effect is specific to the high-level trajectory.

Rigor status:

- Strong. This is a good control against interpreting `z_H` as a static plan:
  freezing early hurts, freezing late does not.

### 4. Controlled Time-Shift

Artifacts:

- `results/controlled/time_shift/aggregate.json`
- `results/controlled/time_shift/per_puzzle.jsonl`
- `results/controlled/time_shift/pairs.json`
- historical: `results/time_shift/aggregate_stats.json`

Controls used:

- Same-puzzle transfer, avoiding cross-puzzle identity confounds.
- Clean run per puzzle.
- Fixed-recipient sweep and fixed-donor sweep. The controlled run fixes
  recipient step 2 for a donor sweep and donor step 10 for a recipient sweep.
- Transfer pairs are explicitly saved.
- Bootstrap CIs for each donor->recipient pair.
- Directionality control: future-to-past and past-to-future transfers can be
  compared.
- Historical run tracked boosted/hurt/unchanged puzzle counts and fixed/broken
  cell counts.

Representative hardened results:

- Baseline N=500: mean accuracy 0.803, CI [0.787, 0.818].
- Adjacent transfers are small or null.
- Later-to-earlier transfers can slightly improve accuracy. Example: 9->2
  mean delta +0.0129, CI [+0.0071, +0.0193].
- Transferring a mid-step state into late readout steps is mildly harmful:
  10->14 mean delta -0.0058, CI [-0.0100, -0.0017].

Rigor status:

- Strong for the progressive-refinement claim, with effects correctly reported
  as small.

### 5. Cross-Puzzle Activation Patching

Artifacts:

- `results/baseline_comparison/HRM_eval.json` activation patching section
- `results/patching/activation_patching_*/*.yaml`
- `results/patching/activation_patching_*/*.html`
- scripts: `scripts/bash/run_activation_patching_examples.sh`,
  `scripts/bash/run_activation_patching_experiments2.sh`

Controls used:

- Clean target baseline and patched target comparison.
- Forward and inverse patching: source->target and target->source.
- Multiple repeated runs for stability in the scripted examples (`num_runs=5`).
- Stream specificity: patch `z_H`, `z_L`, or both.
- Temporal specificity: early steps, late steps, all steps, and single-step
  variants.
- Spatial specificity: all positions, first row, first two rows, first subgrid,
  and row-masked/modified-input variants.
- Patch validation metrics in YAML: saved pre/post differences confirm whether
  the intended stream was actually overwritten.

Representative results:

- In `HRM_eval.json`, cross-puzzle hidden-state patching at step 2 causes mean
  delta -0.665 with CI [-0.702, -0.627] over 100 pairs, growing to -0.720 at
  step 14.
- Representative both-stream patching for puzzle 111->220 drops target accuracy
  from 0.704 to 0.049.
- Representative row-masked `z_H` patching can be highly damaging, while a
  corresponding `z_L` row-masked example is null.

Rigor status:

- Useful as mechanistic and qualitative support.
- The aggregate `HRM_eval.json` patching is stronger than the individual YAML
  examples. The individual examples are controlled case studies, not
  population-level statistical evidence.

### 6. E8 Linear Constraint Probes

Artifacts:

- `results/probes/e8_constraint_probes/probe_summary.json`
- `results/probes/e8_constraint_probes/sweep_results.csv`
- `results/probes/e8_constraint_probes/geometric_analysis.json`
- `results/probes/e8_constraint_probes/probe_weights.pt`
- `results/probes/e8_constraint_probes/_meta.json`

Controls used:

- Five-seed probe ensemble: seeds 0-4.
- Puzzle-disjoint train/validation split, recorded in `_meta.json`.
- Fixed validation fraction and explicit steps: 0, 4, 8, 12, 15.
- Both `z_H` and `z_L` probed in the completed run.
- Multiple target families: structural labels such as cell digit/is-given and
  dynamic labels such as row/col/box violations and per-cell correctness.
- Train and validation scores saved, making overfitting visible.
- CI over seeds for validation accuracy.
- Geometric stability controls: cosine similarities and PCA explained variance
  are computed across the seed ensemble.

Representative hardened results:

- `z_H` row-violation probe improves from 0.741 at step 0 to 0.883 at step 15.
- `z_H` cell digit is high from step 0: 0.985 at step 0 and 0.989 at step 15.
- `z_H` violation directions are highly aligned at step 15: row/box cosine
  mean about 0.970, row/col about 0.933, box/col about 0.955.
- PCA of the four correctness/violation directions gives step-15 PC1 explained
  mean about 0.954 with CI [0.948, 0.961].

Rigor status:

- Strong for decodability and geometry. The puzzle-disjoint split closes a major
  leakage concern from earlier single-split probe runs.

### 7. Directed Ablation Of Probe Directions

Artifacts:

- `results/controlled/directed_ablation/analysis.json`
- `results/controlled/directed_ablation/per_direction_results.json`
- `results/controlled/directed_ablation/directed_ablation_table.md`
- `results/controlled/directed_ablation/_meta.json`
- historical: `results/directed_ablation/e9_directed_ablation/aggregate_results.json`
- nonlinear control: `results/directed_ablation/e9b_nonlinear/statistical_tests.json`

Controls used:

- Probe directions are compared to random directions rather than to zero.
- Ten random-direction controls per probe direction in the controlled run.
- Per-puzzle paired comparison: each probe-direction delta is compared to the
  same puzzle's average random-control delta.
- Paired t-test and paired Wilcoxon test.
- Cohen's d effect sizes.
- Bootstrap CIs for probe and random-control deltas.
- Sudoku violation deltas are tracked in addition to cell accuracy.
- Multi-direction subspace ablation: row+col 2D and row+col+box 3D.
- Nonlinear-probe directed ablation serves as an additional control against
  "linear probes are too weak" objections.

Representative hardened results:

- Row violation direction: +0.04 percentage points, CI [-0.70, +0.76],
  t-test p=0.616, Wilcoxon p=0.167, Cohen's d=-0.023.
- Col violation direction: -0.13 pp, CI [-0.80, +0.60], d=-0.050.
- Box violation direction: -0.11 pp, CI [-0.83, +0.56], d=-0.048.
- Correctness: +0.03 pp, CI [-0.69, +0.77], d=-0.026.
- Is-given: +0.31 pp, CI [-0.58, +1.17], d=+0.014.
- Random controls pooled: +0.20 pp.
- Row+col+box 3D subspace: +0.05 pp, CI [-0.73, +0.83].

Rigor status:

- Strong. This is the cleanest control supporting "readout does not equal
  causal use."

### 8. SAE Sweep And Causal Ablation

Artifacts:

- `results/sae_study/sweep_results.csv`
- `results/sae_study/_meta.json`
- `results/sae_study/sae_d*_l*_features.json`
- `results/sae_study/sae_d*_l*_log.json`
- `results/sae_study/feature_analysis/*.json`
- `results/sae_study/causal_ablation/aggregate.json`
- `results/sae_study/causal_ablation/per_puzzle.jsonl`

Controls used:

- Dictionary-size and sparsity sweep: d in {1024, 2048, 4096, 8192} and L1 in
  {0.003, 0.01, 0.03}.
- Reconstruction/sparsity frontier metrics: reconstruction loss, L1 loss,
  alive/dead features, mean sparsity, L0, mean activation, training time.
- Fixed seed for sweep.
- Feature controls in causal ablation: top SAE features versus random SAE
  features.
- Probe-direction and random-direction controls in the same causal-ablation
  framework.
- Per-puzzle clean baselines and per-puzzle intervention rows.
- Statistical tests in aggregate: SAE top features versus random features,
  SAE features versus probe directions, and probe directions versus random
  directions.

Representative completed results:

- Sweep covers all 12 dict-size/L1 configurations.
- d=2048, L1=0.01 has final reconstruction loss about 2.44e-4 and L0 about
  705.
- Causal aggregate, N=300 puzzles: top SAE feature ablations mean delta -0.036;
  random SAE feature ablations mean delta -0.034; p=0.431 for top versus random.
- SAE feature ablations are more damaging than probe-direction ablations
  (p approximately 3.23e-28), but top SAE features are not more damaging than
  random SAE features.

Rigor status:

- Stronger than the original single-SAE claim because the sweep exists.
- Still not as hardened as controlled directed ablation in the saved aggregate:
  the completed `aggregate.json` reports std and t-tests but does not yet expose
  the same full CI/Wilcoxon/effect-size table style for every condition. The
  current script has been upgraded toward that standard, but the artifact on
  disk should be regenerated before treating it as final paper evidence.

### 9. Difficulty-Stratified Dynamics

Artifacts:

- `results/metrics/difficulty_stratified/stratified_summary.json`
- `results/metrics/difficulty_stratified/per_puzzle.json`

Controls used:

- Stratifies Sudoku puzzles into easy, medium, hard groups.
- Reports per-step cell accuracy, puzzle accuracy, Hamming distance, and total
  violations within each group.
- Checks whether recurrence-driven gains are robust across difficulty rather
  than driven by easy puzzles.

Representative results:

- Easy puzzles: cell accuracy improves from 0.817 to 0.985, puzzle accuracy
  from 0.129 to 0.943.
- Medium puzzles: cell accuracy improves from 0.691 to 0.850, puzzle accuracy
  from 0.000 to 0.552.
- Hard puzzles: cell accuracy improves from 0.603 to 0.705, puzzle accuracy
  from 0.000 to 0.216.

Rigor status:

- Good diagnostic consistency check, but currently mean/std only in the saved
  summary.

## Cross-Experiment Consistency Controls

The most important rigor feature is not any single experiment; it is the
triangulation across intervention types.

| Claim | Controls supporting it |
|---|---|
| Recurrence matters for Sudoku | Architecture baselines, per-step HRM/UT/RNN curves, Hamming and violation reductions, difficulty-stratified curves. |
| z_H is a dynamic state, not static metadata | Single-step ablation, all-steps ablation, freeze-after-k, same-puzzle time-shift. |
| z_H and z_L play different roles | H/L ablation and H/L freeze controls, H/L patching case studies, both-state E8 probes. |
| z_H is puzzle-specific | Cross-puzzle patching, forward/inverse patching, target-baseline versus patched-target comparisons. |
| Decodable features are not necessarily causal | Five-seed E8 probes plus controlled directed ablation against random directions and subspace controls. |
| Sparse features do not localize the whole mechanism | SAE sweep plus top-feature versus random-feature causal controls. |

## Provenance Audit

Completed `_meta.json` provenance files found:

- `results/baseline_comparison/_meta.json`
- `results/probes/e8_constraint_probes/_meta.json`
- `results/controlled/directed_ablation/_meta.json`
- `results/sae_study/_meta.json`

These record git SHA `a34e5172d06559e854aae0b005c796e7cab4734e`, dirty-worktree
status, timestamp, Python/platform, GPU (`NVIDIA RTX 5000 Ada Generation`), argv,
and parameters. This is good, but uneven: several controlled aggregate
directories do not currently have their own `_meta.json` files on disk.

## Remaining Rigor Gaps

1. Regenerate older baseline-comparison JSONs with the current CI-aware
   evaluator so every model has the same bootstrap CI schema as `HRM_eval.json`.
2. Regenerate SAE causal ablation with the current script so `aggregate.json`
   includes bootstrap CIs, Wilcoxon p-values, and effect sizes for all key
   comparisons.
3. Add or regenerate `_meta.json` for controlled ablation, freeze, and time-shift
   directories. The scripts support this pattern, but the completed artifacts on
   disk are uneven.
4. Persist a shared Sudoku puzzle-index manifest next to final controlled runs
   and point every intervention/probe/baseline command at it. The helper exists
   in `scripts/core/sudoku_sample.py`; making the manifest explicit would remove
   any ambiguity about sample alignment.
5. Treat individual patching YAML/HTML examples as qualitative case studies
   unless they are rerun as population-level paired experiments with CIs.
6. Where tables quote historical ablation, freeze, time-shift, or difficulty
   stratified means, prefer the hardened controlled artifacts when they exist.

## Bottom Line

The Sudoku suite now has a defensible control stack for the core mechanistic
claims: paired clean baselines, H/L stream controls, temporal controls, random
direction/feature controls, seeded puzzle-disjoint probes, and bootstrapped
uncertainty in the controlled experiments. The claims that are currently most
rigorous are the causal importance/dynamics of `z_H`, the contrast with `z_L`,
and the "readout does not imply causal use" result. The SAE causal story and
some architecture-baseline comparisons are directionally supported, but their
saved artifacts should be regenerated into the same Phase 0 statistical format
before they are treated as final conference-paper numbers.
