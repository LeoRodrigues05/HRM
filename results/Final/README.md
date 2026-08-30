# Final — curated paper-backing results

Curated snapshot of the results that directly back the cross-task HRM
interpretability paper (Sudoku · Maze · ARC). Small artifacts only: aggregate
JSONs, provenance (`_meta.json`), summary CSVs, paper figures, and the two HTML
reports.

Raw activation dumps (`*.pt`) and per-example records (`per_puzzle.jsonl`,
`per_pair.jsonl`, `per_direction_results.json`) are not in the repository at all —
see [`docs/EXPERIMENTS.md §6`](../../docs/EXPERIMENTS.md) for the command that
regenerates each. The per-task folders (`results/{Sudoku,Maze,ARC}/`) remain the
source of truth for the aggregates mirrored here.

## Layout

- `figures/`  — the canonical destination for every manuscript figure. This
  includes the cross-task/ARC figures and the supporting Sudoku/Maze panels;
  there is no separate root-level `paper/figures` or `figure_final` tree.
- `reports/`  — `sudoku_report.html`, `maze_report.html`.
- `sudoku/`, `maze/`, `arc/` — curated aggregates, mirroring each experiment's
  path under its task folder (source of truth).

## What each finding draws on

- **F1 (iterative refinement, task-dependent depth & locus)**
  - Sudoku: `sudoku/controlled/ablation/{zH,zL}`, `controlled/freeze`,
    `controlled/time_shift`, `patching/patching_full_steps`.
  - Maze: `maze/hardened/ablation_controlled/{zH,zL}`, `freeze_controlled`,
    `patching_full_steps`, `step_dynamics`.
  - ARC: `arc/freeze`, `arc/patching_full_steps`.
- **F2 (probe readout ≠ causal)**
  - `*/directed_ablation`, `*/probe_geometry`, Maze `linear_probes`/`mlp_probes`,
    ARC `hardened/linear_probes`. Figures: `fig_*_readout_vs_causal`,
    `fig_arc_decodability_by_step_z_H`.
- **F3 (deeply distributed)**
  - `*/sae_study/causal_ablation`, Sudoku `controlled/causal_subspace`,
    ARC `causal_subspace`. Figure: `fig_crosstask_sae_ladder`,
    `fig_arc_distributed_computation`.
- **Supporting**: Sudoku `baseline_comparison`, `metrics` (Hamming curve),
  `localizability/{hrm_1step,ut_1step}`, `sae_study/bptt_study/*` (BPTT-vs-stock).

Provenance (git SHA, seed, N, GPU) is in each `_meta.json`.

Regeneration commands and the distinction between CPU-only and GPU-backed
figure generators are documented in `docs/EXPERIMENTS.md`.
