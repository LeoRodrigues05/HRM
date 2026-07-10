# Final — curated paper-backing results

Curated snapshot of the results that directly back the cross-task HRM
interpretability paper (Sudoku · Maze · ARC). Small artifacts only: aggregate
JSONs, provenance (`_meta.json`), summary CSVs, paper figures, and the two HTML
reports. Raw activation dumps (`*.pt`, ~16 GB) and per-puzzle JSONL are **not**
copied here — they live in the per-task folders (`results/{Sudoku,Maze,ARC}/`).

## Layout

- `figures/`  — cross-task paper figures (`fig_crosstask_*`) + ARC paper figures.
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
