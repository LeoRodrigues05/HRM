# Experiments & Scripts Guide

This document describes the mechanistic-interpretability (MI) experiments in this
fork of the [Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734)
repository, and the scripts used to reproduce them.

All commands assume:

```bash
cd <repo-root>
source .venv/bin/activate                 # created by scripts/bash/Initialize_HRM_Repo.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
```

The default Sudoku checkpoint is
[`checkpoints/sapientinc-sudoku-extreme/checkpoint.pt`](../checkpoints/sapientinc-sudoku-extreme/)
and the default test set is `data/sudoku-extreme-1k-aug-1000/`.

---

## 0. Setup & data

| Step | Command | Notes |
|------|---------|-------|
| Bootstrap env, CUDA, FA, deps, data | `bash scripts/bash/Initialize_HRM_Repo.sh` | One-shot setup. Skip parts via `SKIP_CUDA=1`, `SKIP_FA=1`, `SKIP_DATA=1`, `SKIP_APT=1`. |
| Build small Sudoku set | `python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000` | ~1k puzzles × 1k augmentations. |
| Build full Sudoku set | `python dataset/build_sudoku_dataset.py` | Full version. |
| Build Maze 30×30 hard | `python dataset/build_maze_dataset.py` | 1k examples. |
| Build ARC-1 / ARC-2 | `python dataset/build_arc_dataset.py [--dataset-dirs ...]` | Requires `git submodule update --init --recursive`. |
| Pretrained checkpoints | `huggingface-cli download sapientinc/HRM-checkpoint-sudoku-extreme --local-dir checkpoints/sapientinc-sudoku-extreme` | Same for ARC-2 and Maze checkpoints. |

---

## 1. Training

| Model | Config | Launch |
|-------|--------|--------|
| HRM (Sudoku, 1k) | `config/cfg_pretrain.yaml` | `OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0` |
| HRM (Sudoku, full) | same | `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned` |
| HRM (Maze 30×30) | same | `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0` |
| HRM (ARC-1 / ARC-2) | same | `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py [data_path=data/arc-2-aug-1000]` |
| Universal Transformer | `config/cfg_pretrain_universal_transformer.yaml` | `sbatch scripts/bash/train_universal_transformer.sh` (or run the inner `python pretrain.py --config-name ...`) |
| Vanilla RNN | `config/cfg_pretrain_vanilla_rnn.yaml` | `sbatch scripts/bash/train_vanilla_rnn.sh` |
| Plain Transformer | `config/cfg_pretrain_plain_transformer.yaml` | `sbatch scripts/bash/train_plain_transformer.sh` |
| Standard RNN (GRU) | `config/cfg_pretrain_standard_rnn.yaml` | `sbatch scripts/bash/train_standard_rnn.sh` |
| HRM v2 (sparse attn + constraint head) | `config/cfg_pretrain_v2.yaml` | `python pretrain_v2.py` (see `models/hrm_v2/README.md`) |

The SLURM headers in the `train_*.sh` scripts can be ignored when running locally:
each script auto-activates `.venv` (or a conda env named `${HRM_CONDA_ENV:-hrm}`)
and then runs `python pretrain.py --config-name <cfg_name>`.

---

## 2. Evaluation

```bash
# Full distributed eval (writes per-puzzle predictions for downstream analysis)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py \
    checkpoint=checkpoints/sapientinc-sudoku-extreme/checkpoint.pt
```

Side-by-side comparison of HRM vs all baselines:

```bash
bash scripts/bash/eval_all_baselines.sh        # 5-model comparison @ best ckpts
bash scripts/bash/eval_baselines.sh            # multi-checkpoint sweep
```

`scripts/analysis/evaluate_baselines.py` computes per-step cell/puzzle accuracy,
Hamming distance, and row/col/box violation counts. Outputs land under
`results/Sudoku/baseline_comparison/`.

ARC submission scoring uses [`arc_eval.ipynb`](../arc_eval.ipynb) on the JSON
predictions produced by `evaluate.py`.

---

## 3. Mechanistic-interpretability experiments

The MI suite probes what z<sub>H</sub> (high-level state) and z<sub>L</sub>
(low-level state) encode and which parts of them are *causal* for solving
Sudoku. All experiments share `scripts/core/activation_ablation.py` /
`scripts/core/activation_patching.py` as the underlying intervention engine.

| ID | Question | Driver script | Outputs |
|----|----------|---------------|---------|
| **E1** | Is z<sub>H</sub> necessary? Which steps matter? | `scripts/ablation/batch_ablation_1k.py` | `results/Sudoku/ablation/batch_ablation_zH/` |
| **E2** | Is z<sub>H</sub> a static plan or progressive refinement? | `scripts/freeze/batch_freeze_h.py` | `results/Sudoku/freeze/` |
| **E3 / E3b** | Does z<sub>H</sub> carry puzzle-specific info? | `scripts/core/activation_patching.py` (use `scripts/bash/run_activation_patching_examples.sh`) | `results/Sudoku/patching/` |
| **E4** | What is decodable from z<sub>H</sub> / z<sub>L</sub>? | `scripts/probes/sweep_linear_probes.py`, `scripts/probes/train_linear_probes.py` | `results/Sudoku/probes/` |
| **E5** | How does z<sub>H</sub> evolve across steps? | `scripts/time_shift/batch_time_shift.py` | `results/Sudoku/time_shift/` |
| **E7b** | Convergence of predictions across steps | `scripts/reports/result_metrics_sudoku.py` | `results/Sudoku/metrics/`, `results/Sudoku/hamming_multi/` |
| **E8** | Linear constraint probes (row/col/box violations, naked singles, …) | `scripts/probes/e8_constraint_probes.py` | `results/Sudoku/probes/e8_constraint_probes/` |
| **E9** | Directed ablation along E8 directions (causal validation) | `scripts/directed_ablation/e9_directed_ablation.py` | `results/Sudoku/directed_ablation/` |
| **E9b** | MLP / non-linear probes & their directed ablation | `scripts/probes/nonlinear_probes.py`, `scripts/directed_ablation/e9b_nonlinear_directed_ablation.py` | `results/Sudoku/directed_ablation/e9b_nonlinear/` |
| **E10** | Sparse autoencoder study on z<sub>H</sub> (features + causal ablation) | `scripts/sae/sae_train.py`, `scripts/sae/sae_analyze_features.py`, `scripts/sae/sae_causal_ablation.py`, `scripts/sae/sae_sweep.py`, `scripts/sae/sae_plot.py` | `results/Sudoku/sae_study/` |
| Maze | Maze variants of E1/E2/E5/E8/E9/E10 | `scripts/sae/sae_collect_activations_maze.py`, `scripts/maze/` | `results/Maze/` |
| ARC | ARC-AGI replication of E4/E8/E9/E9b + H2/A/H4 (ARC-native features) | `scripts/arc/slurm_arc_suite.sbatch` (see `archive/plans/PLAN_ARC_experiments.md`) | `results/ARC/arc/` |
| Controlled | Single-variable, larger-N replications of E1/E2/E5/E9 with stats | `scripts/controlled/run_all_controlled_experiments.py` | `results/Sudoku/controlled/` |

The ARC suite uses the HRM ARC-2 checkpoint
(`checkpoints/sapientinc-hrm-arc-2/checkpoint`) and `data/arc-2-aug-1000`
(rebuild with `seed=42 num_aug=1000` to align the puzzle-identifier embeddings).
ARC-native probe features live in [`utils/arc_targets.py`](../utils/arc_targets.py)
and ARC structural metrics in [`scripts/arc/arc_common.py`](../scripts/arc/arc_common.py).
Run end-to-end with `sbatch -p gpu --gres=gpu:1 scripts/arc/slurm_arc_suite.sbatch`.

### Paper-replication baseline (Phase 7)

The earlier paper-replication suite lives under
[`experiments/paper_replication/`](../experiments/paper_replication/) and can be
run end-to-end with:

```bash
python experiments/paper_replication/run_all_experiments.py
```

Per-experiment scripts: `exp1_easy_hard_analysis.py`, `exp2_grokking_analysis.py`,
`exp3_step_dynamics.py`, `exp4_specialization_probes.py`,
`exp5_activation_patching.py`, `exp6_latent_trajectories.py`,
`exp7_segment_loss_scaling.py`. Outputs are written under
`experiments/paper_replication/results/<exp>/`.

### Convenience wrappers (bash)

| Script | Purpose |
|--------|---------|
| `scripts/bash/probe_commands.sh` | End-to-end probe collection + linear probe training. |
| `scripts/bash/run_activation_patching_examples.sh` | Seven canonical patching scenarios (full / H-only / L-only / early / late / single-step / position-specific). |
| `scripts/bash/run_activation_patching_experiments2.sh` | Larger sweep of patching configurations. |
| `scripts/bash/run_zl_at_scale.sh` | z<sub>L</sub>-only patching at scale. |
| `scripts/bash/single_inference.sh` | Run inference on one (or N) puzzles and pretty-print. |
| `scripts/bash/Result_Generator_HRM.sh` | Async full eval → NPZ → metrics → coloured HTML report. |

### Plotting & figures

```bash
# Task-level result figures
python scripts/plotting/plot_baseline_comparison.py \
    --results_dir results/Sudoku/baseline_comparison
python scripts/plotting/plot_maze_figures.py

# Canonical manuscript figures -> results/Final/figures
python scripts/plotting/generate_paper_figures.py
python scripts/plotting/generate_supporting_paper_figures.py
python scripts/plotting/plot_crosstask_causal_ladder.py
python scripts/plotting/plot_maze_causal_evidence.py
python scripts/plotting/plot_readout_vs_causality.py
python scripts/plotting/plot_zh_trajectories.py
```

Older exploratory/presentation plotters are not part of the manuscript
regeneration path; see the cleanup review list in the repository handoff before
retaining them in a release bundle.

---

## 4. Key file map

```
config/                       Hydra configs (HRM, baselines, HRM v2)
models/
  hrm/hrm_act_v1.py           HRM with ACT (z_H/z_L hidden states + probe hooks)
  hrm_v2/                     Sparse-attention + constraint-head variant
  baselines/                  Vanilla RNN, GRU, Plain/Universal Transformer
dataset/
  build_sudoku_dataset.py     Sudoku-Extreme builder
  build_maze_dataset.py       Maze 30x30 builder
  build_arc_dataset.py        ARC-1 / ARC-2 builder
puzzle_dataset.py             Shared dataloader used by pretrain.py / evaluate.py
pretrain.py                   Generic training entrypoint (Hydra)
pretrain_v2.py                HRM v2 training entrypoint
evaluate.py                   Distributed evaluation + prediction dumping
scripts/
  core/                       Activation patching / ablation engine
  ablation/   freeze/   patching/   time_shift/    Per-experiment drivers
  probes/                     Linear, sweep, and constraint probes (E4/E8)
  directed_ablation/          E9 / E9b directed ablation
  sae/                        E10 sparse-autoencoder study
  controlled/                 Tighter, single-variable controlled re-runs
  analysis/                   Evaluation, difficulty-stratified, trajectory tools
  plotting/                   Paper / presentation figure generation
  reports/                    Sudoku HTML / metric reports
  bash/                       Convenience wrappers (Init, train_*, eval_*, etc.)
utils/
  probes.py                   ProbeRecorder + per-cell label derivation
  functions.py                Misc helpers (norms, masks, ...)
  maze_targets.py             Maze-specific evaluation helpers
results/                      All experiment outputs (gitignored except committed plots)
experiments/paper_replication/  Phase-7 replication of the original paper claims
tests/                        CPU-only regression and safety checks
```

---

## 5. Reproducing the paper figures

After running E1/E2/E5/E8/E9/E10 (and their controlled / maze variants):

```bash
python scripts/plotting/generate_paper_figures.py
python scripts/plotting/generate_supporting_paper_figures.py
python scripts/plotting/plot_crosstask_causal_ladder.py
python scripts/plotting/plot_maze_causal_evidence.py
python scripts/plotting/plot_readout_vs_causality.py
ls results/Final/figures/*.pdf
```

The committed `results/Final/figures/*.pdf` files are the exact assets used in
the manuscript; rebuilding them requires the corresponding `results/...`
artefacts. `plot_maze_patching_example.py` additionally needs the Maze checkpoint
and a GPU because it selects and reruns a qualitative intervention example.

---

## 6. Artefacts not in this repository (regenerate locally)

Every number reported in the paper is backed by a small aggregate committed here —
typically `aggregate.json`, `analysis.json`, or `aggregate_stats.json` next to where
the raw dump used to live, plus a `_meta.json` recording git SHA, seed, N, and GPU.

Three classes of artefact are deliberately absent, all fully regenerable.

### 6.1 Model weights and datasets → Hugging Face

| Artefact | Source |
|---|---|
| Released HRM checkpoints (Sudoku / Maze / ARC-2) | `sapientinc/*` — see [README §2](../README.md) |
| Sudoku baselines (Recurrent / Universal / Plain Transformer, GRU) | this project's HF model repo |
| `arc2-adapted-evalonly/step_7391` | this project's HF model repo |
| `bptt_study/{hrm_stock,hrm_bptt}` | this project's HF model repo |
| Trained SAE weights (`sae_d*_l1*.pt`) | this project's HF model repo |
| `data/arc-2-evalonly`, `data/sudoku-extreme-1k-aug-1000` | this project's HF dataset repo |

The two datasets are published rather than left to `dataset/build_*.py` because the
adapted ARC checkpoint's `puzzle_emb` is bound to the exact build ordering — see §7.

### 6.2 Activation caches and SAE sweep snapshots → regenerate on a GPU

| Artefact | Size | Regenerate with |
|---|---|---|
| `results/Sudoku/sae_study/activations_zH.pt` | 1.7 GB | `python scripts/sae/sae_collect_activations.py` |
| `results/Maze/sae_study/activations_zH.pt` | 5.6 GB | `python scripts/sae/sae_collect_activations_maze.py` |
| `results/ARC/arc/sae_study/activations_zH.pt` | 4.2 GB | `python scripts/arc/sae_collect_activations_arc.py` |
| `results/*/sae_study/sae_d*_epoch*.pt` | ~3.9 GB | `python scripts/sae/sae_sweep.py` (per-epoch checkpoints) |
| `results/Maze/hardened/probe_geometry/probe_weights.pt` | 1.5 MB | `python scripts/maze/linear_probes_maze.py --save_probe_weights` |
| `results/Maze/hardened/trajectory_pca/trajectory_pca.npz` | 60 KB | `python scripts/maze/trajectory_pca_maze.py` |

### 6.3 Per-example dumps and prediction files → regenerate from the drivers

Raw per-example records were removed to keep the repository clonable. Each row's
aggregate is committed; re-run the driver only if you need per-puzzle detail.

| Removed artefact | Committed aggregate that replaces it | Regenerate with |
|---|---|---|
| `checkpoints/**/*_all_preds.*` (~28 GB) | `results/Sudoku/baseline_comparison/*_eval.json` | `evaluate.py checkpoint=<ckpt>` |
| `results/Sudoku/ablation/batch_ablation_zH/results_per_puzzle.jsonl` | `aggregate_stats.json` | `scripts/ablation/batch_ablation_1k.py` |
| `results/Sudoku/hamming_multi/hamming_per_puzzle.csv` | `hamming_*.png` in the same directory | `scripts/reports/result_metrics_sudoku.py` |
| `results/Sudoku/controlled/**/per_puzzle.jsonl` | `aggregate.json` per experiment | `scripts/controlled/run_all_controlled_experiments.py` |
| `results/*/sae_study/**/per_puzzle.jsonl` | `aggregate.json` per experiment | `scripts/sae/sae_causal_ablation.py` |
| `results/*/directed_ablation/per_direction_results.json` | `analysis.json` (all reported deltas, CIs, p-values) | `scripts/directed_ablation/e9_directed_ablation.py` |
| `results/Maze/hardened/**/per_{puzzle,pair}.jsonl` | `aggregate.json` per experiment | `scripts/maze/run_all_maze_hardened.sh` |
| `results/Maze/reports/*.html` (qualitative dumps) | `results/Final/reports/maze_report.html` | `scripts/maze/intermediate_report_maze.py` |

---

## 7. ARC-AGI-2: puzzle-embedding adaptation, not fine-tuning

The ARC-AGI-2 model used for the representation-level analysis is **not** a
conventional fine-tune of the released checkpoint. `scripts/arc/adapt_puzzle_emb.sh`
freezes the reasoning core (`lr=1e-9`) and trains **only** the puzzle embedding
(`puzzle_emb_lr=1e-2`, 2000 epochs, `global_batch_size=96`), starting from
`checkpoints/sapientinc-hrm-arc-2/checkpoint`.

This matters for reproduction. HRM identifies each puzzle by an index into a learned
`puzzle_emb` table, and that index is assigned at dataset-build time. The released
checkpoint's table is therefore meaningless against any other build of the data, which
is why the released checkpoint scores ~0 on a locally rebuilt ARC-2 set. Adaptation
re-fits the table to *our* build while leaving the reasoning weights untouched, so the
mechanism under study is still the released model's.

Two consequences:

- The ARC-2 dataset must be built with `seed=42, num_aug=1000`. A different seed or
  augmentation count reorders the puzzle identifiers and invalidates the adapted
  embedding. This is why the built dataset is published rather than only its builder.
- Do **not** rebuild `data/arc-2-evalonly` between adaptation and analysis.

```bash
bash scripts/arc/adapt_puzzle_emb.sh evalonly   # 120 eval tasks (used in the paper)
bash scripts/arc/adapt_puzzle_emb.sh full       # all 1120 tasks, ~10x slower
```

Recovered accuracy is verified with `scripts/arc/measure_arc_accuracy.py` before the
interpretability suite runs; `RUN_SUITE=1` chains the two behind an accuracy gate.
