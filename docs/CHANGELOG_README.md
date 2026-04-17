# HRM Repository — Changelog & Experiment Log

This document records **every change** made to the [Hierarchical Reasoning Model (HRM)](https://github.com/LeoRodrigues05/HRM) repository, listed in chronological order, together with the scientific significance of each change.

> **Repository:** `https://github.com/LeoRodrigues05/HRM`
> **Original Authors:** One (Sapient AI) — core HRM architecture
> **Interpretability Work:** Leo Raphael Rodrigues (LeoRodrigues05)
> **Last Updated:** 2026-03-02

---

## Table of Contents

1. [Change Log (Chronological)](#change-log-chronological)
   - [Phase 1 — Original Release (Jul 2025)](#phase-1--original-release-jul-2025)
   - [Phase 2 — Community Contributions (Jul–Sep 2025)](#phase-2--community-contributions-julsep-2025)
   - [Phase 3 — Intermediate-Step Visualisation (Nov 2025)](#phase-3--intermediate-step-visualisation-nov-2025)
   - [Phase 4 — Linear Probes (Dec 2025)](#phase-4--linear-probes-dec-2025)
   - [Phase 5 — Activation Patching (Dec 2025)](#phase-5--activation-patching-dec-2025)
   - [Phase 6 — Repo Reorganisation & Bidirectional Patching (Jan 2026)](#phase-6--repo-reorganisation--bidirectional-patching-jan-2026)
   - [Phase 7 — Paper Replication Experiments (Jan 2026)](#phase-7--paper-replication-experiments-jan-2026)
   - [Phase 8 — Extended Probes, Ablations & Presentation (Feb–Mar 2026)](#phase-8--extended-probes-ablations--presentation-febmar-2026)
2. [Key Scientific Findings (Cumulative)](#key-scientific-findings-cumulative)
3. [Repository Structure](#repository-structure)
4. [Large Files (excluded from Git)](#large-files-excluded-from-git)

---

## Change Log (Chronological)

### Phase 1 — Original Release (Jul 2025)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `caa00bb` | 2025-07-09 | One | **Add git submodules** — added ARC-AGI, ARC-AGI-2, and ConceptARC benchmark datasets as submodules under `dataset/raw-data/`. |
| `bd62227` | 2025-07-09 | One | **Initial public release** — 24 files, +3 375 lines. Core HRM architecture (`models/hrm/hrm_act_v1.py`), training (`pretrain.py`), evaluation (`evaluate.py`), dataset builders for Sudoku/Maze/ARC, configs, visualiser, and the original README with architecture diagram, quick-start guide, and citation. |
| `171e2fc` | 2025-07-21 | One | **Post-release polish** — added Apache 2.0 `LICENSE`; expanded README; bug-fixes in `build_maze_dataset.py`, `build_sudoku_dataset.py`, `evaluate.py`; improvements to `models/layers.py` and `pretrain.py`. |

**Significance:** Establishes the complete HRM codebase — a 27M-parameter hierarchical recurrent transformer with two-level reasoning (z_H for abstract planning, z_L for detailed computation), Adaptive Computation Time (ACT) halting, and support for Sudoku, Maze, and ARC tasks.

---

### Phase 2 — Community Contributions (Jul–Sep 2025)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `237887b` | 2025-07-27 | Liam Norman | **Fix typo in README.md** (PR #6). |
| `bafeb56` | 2025-07-29 | One | Merge PR #6. |
| `9f0e53d` | 2025-08-02 | btoo | **BibTeX syntax highlighting** — changed citation code block to `bibtex` for proper rendering (PR #17). |
| `c8be3bb` | 2025-08-03 | One | Merge PR #17. |
| `55d0a2a` | 2025-08-31 | WC-william | **Added Discord info** — community link `https://discord.gg/sapient` added to README. |
| `72f7b58` | 2025-09-09 | raincchio | **Update layers.py** — removed an incorrect comment. |

**Significance:** External community engagement; minor quality-of-life fixes to docs and code.

---

### Phase 3 — Intermediate-Step Visualisation (Nov 2025)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `8563ccc` | 2025-11-06 | LeoRodrigues05 | **Intermediate steps & HTML report generation** — modified `pretrain.py` and `hrm_act_v1.py` to capture per-step predictions; added `pretty_sudoku_print.py` (terminal pretty-printer) and `sudoku_report.py` (HTML report with red-marked incorrect cells); generated first evaluation report `sudoku_eval_report.html`. |
| `46ef717` | 2025-11-11 | LeoRodrigues05 | **Coloured Sudoku reports & repo setup script** — added `sudoku_report_colored_ai.py` for colour-coded reports with AI annotations; created `Initialize_HRM_Repo.sh` for one-command setup; organised reports under `Sudoku_Reports/`. |
| `9437957` | 2025-11-11 | LeoRodrigues05 | **Hamming distance analysis** — added `result_metrics_sudoku.py` and `result_metrics_sudoku_multiple.py` for computing step-wise Hamming distance between intermediate predictions and ground truth; created `Result_Generator_HRM.sh` pipeline; generated convergence plots (`hamming_mean_std.png`, `hamming_overlay.png`, `hamming_small_multiples.png`) and `results/metrics/hamming.csv`. |

**Significance:** First interpretability work — the ability to visualise every intermediate reasoning step of the model. Hamming distance analysis provides the first quantitative evidence that predictions converge to ground truth across ACT steps.

---

### Phase 4 — Linear Probes (Dec 2025)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `d816543` | 2025-12-09 | LeoRodrigues05 | **Linear probe framework** — created `utils/probes.py` (`ProbeRecorder` class to hook into model and capture z_H/z_L activations); added `scripts/generate_probes.py`, `scripts/train_linear_probes.py`, `scripts/run_probes_driver.py`, `scripts/probe_commands.sh`; documented all experiments in `docs/PROBE_RUNLOG.md`; saved probe artefacts (`probe_global.pt`, `probe_index.json`); created analysis notebook `results/probes/probe_results.ipynb`. |

**Significance:** Infrastructure to test **functional specialisation** of z_H vs z_L via linear classifiers trained on activations. Initial finding: "little evidence" of clean separation — subsequent experiments (E8, E9) later showed constraint info is decodable from z_H at ~90% accuracy but distributed rather than localised. The framework enabled all later probe experiments.

---

### Phase 5 — Activation Patching (Dec 2025)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `ef5240f` | 2025-12-26 | LeoRodrigues05 | **First draft activation patching** — created `scripts/activation_patching.py` (cache z_H/z_L from source puzzle, patch into target, measure accuracy change); `scripts/analyze_activation_patching.py` (post-hoc analysis); `scripts/batch_activation_patching.py` (systematic batch experiments); `scripts/run_activation_patching_examples.sh`; comprehensive `docs/ACTIVATION_PATCHING_README.md`. |
| `e188d0a` | 2025-12-26 | Leo Raphael Rodrigues | **Merge PR #1: feature/linear-probes** — merged Phases 4+5 into main. |

**Significance:** Causal intervention framework — swap activations between different puzzles to test what information z_H and z_L encode. Supports per-step, per-layer, and per-position patching. This is the core tool for all later causal experiments.

---

### Phase 6 — Repo Reorganisation & Bidirectional Patching (Jan 2026)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `e4cc9c8` | 2026-01-11 | LeoRodrigues05 | **Major reorganisation + bidirectional patching** — (39 files, +31 659 lines). Moved docs to `docs/`, bash scripts to `scripts/bash/`. Added bidirectional (forward + inverse) activation patching to `scripts/activation_patching.py` (+1 393 lines); created `scripts/activation_patching_sudoku_report.py` (Sudoku-specific HTML); added `scripts/sweep_linear_probes.py` (hyperparameter sweep); enhanced `scripts/train_linear_probes.py` and `utils/probes.py`. New docs: `docs/ACTIVATION_PATCHING_CHANGES.md`, `docs/BIDIRECTIONAL_PATCHING_SUMMARY.md`, `docs/SCRIPTS_GUIDE.md`. Saved results across 5 directories (e2e, experiments2, rowmasked, s0_t250, s111_t220). |
| `ae205e6` | 2026-01-11 | Leo Raphael Rodrigues | **Merge PR #2: leo-mech-interp-changes** — merged Phase 6 into main. |

**Significance:** Key finding — **z_H patching causes catastrophic accuracy drop (−56%), while z_L patching has minimal effect (−1.2%)**. This confirms z_H carries the "plan" and z_L is a transient "scratch pad". Bidirectional patching enables testing both directions of causal influence.

---

### Phase 7 — Paper Replication Experiments (Jan 2026)

| Commit | Date | Author | Summary |
|--------|------|--------|---------|
| `cdf9b27` | 2026-01-23 | LeoRodrigues05 | **Paper claim verification suite** — created `experiments/paper_replication/` package with 5 experiments: `exp1_easy_hard_analysis.py` (easy vs hard performance), `exp2_grokking_analysis.py` (grokking claims), `exp3_step_dynamics.py` (iterative refinement), `exp4_specialization_probes.py` (z_H/z_L specialisation), `exp5_activation_patching.py` (causal importance); `run_all_experiments.py` orchestrator; updated `.gitignore` extensively. Generated publication-quality plots and `EXPERIMENT_SUMMARY.md`. |

**Significance:** Systematic verification of paper claims against 422 786 test puzzles:

| Claim | Experiment | Verdict |
|-------|-----------|---------|
| HRM struggles with easy puzzles | E1 | **CONTRADICTED** — 90.2% accuracy on hardest vs 73.6% on easiest |
| Training exhibits grokking | E2 | **INCONCLUSIVE** — needs training-time checkpoints |
| Predictions refine iteratively | E3 | **VERIFIED** — accuracy improves 67.9% → 83.7% across 16 steps |
| z_H and z_L have distinct roles | E4 | **VERIFIED** — z_H encodes global violations, z_L encodes per-cell properties |
| z_H carries causal importance | E5 | **VERIFIED** — z_H patching = −56.4% accuracy, z_L = −4.1% |

---

### Phase 8 — Extended Probes, Ablations & Presentation (Feb–Mar 2026)

These changes are **uncommitted** at the time of writing. They represent the latest round of experiments.

#### Modified Files
| File | Change |
|------|--------|
| `experiments/paper_replication/__init__.py` | Minor import updates |
| `experiments/paper_replication/config.py` | Added new experiment configurations |
| `experiments/paper_replication/results/grokking_analysis/results.json` | Updated grokking analysis results |
| `results/probes/probe_global.pt` | Re-generated probes (50.6 MB → 3.9 MB) |
| `scripts/activation_patching.py` | Minor enhancements |
| `scripts/run_probes_driver.py` | Extended probe collection pipeline (+168/−40 lines) |
| `scripts/sweep_linear_probes.py` | Extended sweep capabilities (+99 lines) |
| `scripts/train_linear_probes.py` | Major expansion (+406 lines) — multi-target probes |
| `utils/probes.py` | Major overhaul (+731 lines) — constraint-specific probe recording |

#### New Scripts
| Script | Purpose |
|--------|---------|
| `scripts/activation_ablation.py` | Single-puzzle activation ablation (zero out z_H/z_L at specific steps) |
| `scripts/activation_cross_step_transfer.py` | Transfer activations across time-steps within same puzzle |
| `scripts/batch_ablation_1k.py` | Batch z_H ablation over ~1 000+ puzzles |
| `scripts/batch_activation_ablation.py` | Batch z_H + z_L ablation experiments |
| `scripts/batch_freeze_h.py` | Freeze z_H after step k; measure accuracy decay |
| `scripts/batch_time_shift.py` | Time-shift patching: inject future/past z_H into current step |
| `scripts/debug_easy_puzzle.py` | Debug tool for analysing easy-puzzle failures |
| `scripts/e8_constraint_probes.py` | Constraint-specific probes (row/col/box violation decoding from z_H) |
| `scripts/e9_directed_ablation.py` | Directed ablation: project out probe directions from z_H, measure causal effect |
| `scripts/easy_puzzle_analysis.py` | Analysis of easy-puzzle performance patterns |
| `scripts/easy_puzzle_report.py` | Generate HTML reports for easy puzzle analysis |
| `scripts/plot_ablation_results.py` | Plot ablation results (step trajectories, single-step impact) |
| `scripts/plot_e8_e9.py` | Generate all E8/E9 figures (probe accuracy, cosine heatmaps, PCA, specificity matrix) |
| `scripts/plot_presentation.py` | Generate presentation-quality plots for all experiments |
| `scripts/quick_cosine_check.py` | Quick diagnostic: z_H cosine similarity across ACT steps |
| `scripts/run_ablation_experiments.py` | Orchestrator for all ablation experiments |

#### New Experiment Results
| Directory | Content |
|-----------|---------|
| `results/ablation_plots/` | Publication plots for ablation experiments |
| `results/ablation_zH_single_step/` | Single-step z_H ablation (multiple puzzles × steps) |
| `results/activation_ablation_1k/` | Batch ablation over 1 000 puzzles |
| `results/activation_ablation_single_p572/` | Single-puzzle deep ablation study |
| `results/batch_ablation_zH/` | Batch z_H ablation aggregate results |
| `results/e8_constraint_probes/` | Constraint probe results (row/col/box violation accuracy) |
| `results/e8_e9_plots/` | All E8 & E9 figures (9 publication figures) |
| `results/e9_directed_ablation/` | Directed ablation results (ablation vs random baselines) |
| `results/freeze_h/` | Freeze-z_H experiments (accuracy matrix, per-puzzle results) |
| `results/hamming_multi/hamming_per_puzzle.csv` | Per-puzzle Hamming distance data |
| `results/presentation_plots/` | 9 presentation-quality combined plots (PDF + PNG) |
| `results/time_shift/` | Time-shift patching results (future→past, past→future) |

#### New Documentation
| Document | Content |
|----------|---------|
| `docs/E8_E9_FINDINGS_SUMMARY.md` | E8 constraint probes (~90% accuracy decoding violations from z_H) & E9 directed ablation (negligible causal effect — readout ≠ computation) |
| `docs/EXPERIMENT_SUMMARY_AND_SLIDES.md` | Complete 12-slide presentation covering all experiments E1–E7b + diagnostics |

#### New Paper Replication Scripts
| File | Purpose |
|------|---------|
| `experiments/paper_replication/analyze_grokking_checkpoints.py` | Analyse grokking from training checkpoints |
| `experiments/paper_replication/exp6_latent_trajectories.py` | Latent z_H trajectory analysis (norm, rotation, PCA) |
| `experiments/paper_replication/exp7_segment_loss_scaling.py` | Segment-level loss scaling experiments |
| `experiments/paper_replication/train_for_grokking.py` | Train model from scratch to study grokking |
| `experiments/paper_replication/run_grokking_experiment.sh` | Automated grokking training driver |

#### New Model & Config Files
| File | Purpose |
|------|---------|
| `models/hrm_v2/` | HRM v2 model architecture |
| `config/arch/hrm_v2.yaml` | HRM v2 architecture config |
| `config/cfg_pretrain_v2.yaml` | HRM v2 pretraining config |
| `pretrain_v2.py` | HRM v2 training script |

**Significance:** This phase completes the full interpretability study with the following key findings:

- **E2b (Freeze z_H):** Freezing z_H at step 0 causes −9.0% accuracy loss; freezing at step 8 causes only −0.5% — most planning occurs in steps 0–4.
- **E5 (Time-Shift):** Future z_H → past = +3.5% boost; past z_H → future = −4.0% degradation — z_H quality improves monotonically.
- **E8 (Constraint Probes):** Row/col/box violations are linearly decodable from z_H at ~90% accuracy; three violation directions are geometrically distinct (pairwise cosines 0.60–0.73).
- **E9 (Directed Ablation):** Projecting out probe directions from z_H produces negligible effect (Δaccuracy < 1%) — **readout ≠ computation**. The constraint-solving mechanism is distributed across a higher-dimensional subspace.
- **Diagnostics:** z_H lives on a constant-norm manifold (‖z_H‖ ≈ 205); rotation up to 30° between first and last step; directional evolution, not magnitude change.

---

## Key Scientific Findings (Cumulative)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **z_H is the "plan", z_L is the "scratch pad"** | z_L patching = −1.2% effect; z_H patching = −56% (catastrophic) |
| 2 | **z_H is progressively refined, not static** | Freeze@step0 = −9.0%; Freeze@step8 = −0.5%; most refinement in steps 0–4 |
| 3 | **Future z_H > Past z_H (monotonic improvement)** | Time-shift: future→past = +3.5% boost; past→future = −4.0% degradation |
| 4 | **z_H evolves on a constant-norm manifold** | ‖z_H‖ ≈ 205 always; rotation up to 30° between first and last step |
| 5 | **Constraint info is decodable but distributed** | E8 probes: ~90% accuracy; E9 ablation: Δaccuracy < 1% (readout ≠ computation) |
| 6 | **HRM performs better on hard puzzles** | 90.2% on hardest vs 73.6% on easiest (contradicts paper claim) |
| 7 | **Binary puzzle states are perfectly separable** | is_solved, is_forced probes achieve 100% accuracy |
| 8 | **Iterative refinement verified** | Accuracy improves 67.9% → 83.7% across 16 ACT steps |

---

## Repository Structure

```
HRM/
├── README.md                       # Original project README
├── CHANGELOG_README.md             # This file — complete change log
├── LICENSE                         # Apache 2.0
├── requirements.txt                # Python dependencies
│
├── models/                         # Model definitions
│   ├── hrm/hrm_act_v1.py          # HRM v1 with ACT (original)
│   ├── hrm_v2/                     # HRM v2 architecture
│   ├── layers.py                   # Shared transformer layers
│   ├── losses.py                   # Loss functions
│   ├── sparse_embedding.py         # Sparse embedding module
│   └── common.py                   # Model utilities
│
├── config/                         # Training & architecture configs
│   ├── arch/hrm_v1.yaml
│   ├── arch/hrm_v2.yaml
│   ├── cfg_pretrain.yaml
│   └── cfg_pretrain_v2.yaml
│
├── dataset/                        # Dataset builders
│   ├── build_sudoku_dataset.py
│   ├── build_maze_dataset.py
│   ├── build_arc_dataset.py
│   ├── common.py
│   └── raw-data/                   # ARC submodules
│
├── scripts/                        # All experiment scripts
│   ├── activation_patching.py      # Core activation patching
│   ├── activation_ablation.py      # Zero ablation
│   ├── batch_ablation_1k.py        # Batch ablation (1k puzzles)
│   ├── batch_freeze_h.py           # Freeze z_H experiments
│   ├── batch_time_shift.py         # Time-shift patching
│   ├── e8_constraint_probes.py     # Constraint-specific probes
│   ├── e9_directed_ablation.py     # Directed ablation
│   ├── train_linear_probes.py      # Linear probe training
│   ├── sweep_linear_probes.py      # Probe hyperparameter sweep
│   ├── plot_*.py                   # Various plot generators
│   └── bash/                       # Shell scripts
│
├── experiments/paper_replication/   # Paper claim verification
│   ├── exp1_easy_hard_analysis.py
│   ├── exp2_grokking_analysis.py
│   ├── exp3_step_dynamics.py
│   ├── exp4_specialization_probes.py
│   ├── exp5_activation_patching.py
│   ├── exp6_latent_trajectories.py
│   ├── exp7_segment_loss_scaling.py
│   ├── train_for_grokking.py
│   └── results/                    # Experiment outputs
│
├── utils/
│   └── probes.py                   # ProbeRecorder + constraint probe utilities
│
├── docs/                           # Documentation
│   ├── ACTIVATION_PATCHING_README.md
│   ├── ACTIVATION_PATCHING_CHANGES.md
│   ├── BIDIRECTIONAL_PATCHING_SUMMARY.md
│   ├── E8_E9_FINDINGS_SUMMARY.md
│   ├── EXPERIMENT_SUMMARY_AND_SLIDES.md
│   ├── PROBE_RUNLOG.md
│   └── SCRIPTS_GUIDE.md
│
├── results/                        # All experiment results
│   ├── ablation_plots/
│   ├── ablation_zH_single_step/
│   ├── activation_ablation_1k/
│   ├── activation_patching_*/
│   ├── e8_constraint_probes/
│   ├── e8_e9_plots/
│   ├── e9_directed_ablation/
│   ├── freeze_h/
│   ├── hamming_multi/
│   ├── presentation_plots/
│   ├── probes/
│   └── time_shift/
│
├── Sudoku_Reports/                 # HTML evaluation reports
├── pretrain.py                     # Training script (v1)
├── pretrain_v2.py                  # Training script (v2)
├── evaluate.py                     # Evaluation script
├── puzzle_dataset.py               # Dataset loader
├── puzzle_visualizer.html          # Interactive visualiser
├── sudoku_report_colored_ai.py     # Coloured Sudoku report generator
├── result_metrics_sudoku.py        # Single-puzzle Hamming metrics
└── result_metrics_sudoku_multiple.py # Multi-puzzle Hamming metrics
```

---

## Large Files (excluded from Git)

The following files exceed 1 GB and are **not pushed** to the repository. They are archived separately in `LARGE_FILES_ZIP/`.

| File | Size | Description |
|------|------|-------------|
| `cuda_12_6.run` | 4.2 GB | CUDA 12.6 installer binary |
| `Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/step_0_all_preds.0` | 5.1 GB | Full prediction dump for all puzzles at step 0 |

Additionally, the `.gitignore` excludes all `.pt`, `.npy`, `.ckpt`, `.pth`, `.run` files and `Checkpoint_*/` directories to keep the repository size manageable:

| Category | Example Paths | Approx. Size |
|----------|---------------|-------------|
| Checkpoint v1 | `Checkpoint_HRM_Sudoku/` | 6.1 GB |
| Checkpoint v2 | `Checkpoint_HRM_v2_Sudoku/` | 235 MB |
| Training data | `data/sudoku-extreme-1k-aug-1000/*.npy` | 1.3 GB |
| Grokking checkpoints | `experiments/paper_replication/results/grokking_checkpoints_full/` | 6.2 GB |
| Probe weights | `results/probes/probe_global.pt`, `probe_local.pt` | 208 MB |
| CUDA installer | `cuda_12_6.run` | 4.2 GB |
