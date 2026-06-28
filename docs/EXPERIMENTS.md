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
`results/baseline_comparison/`.

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
| **E1** | Is z<sub>H</sub> necessary? Which steps matter? | `scripts/ablation/batch_ablation_1k.py` | `results/ablation/batch_ablation_zH/` |
| **E2** | Is z<sub>H</sub> a static plan or progressive refinement? | `scripts/freeze/batch_freeze_h.py` | `results/freeze/` |
| **E3 / E3b** | Does z<sub>H</sub> carry puzzle-specific info? | `scripts/core/activation_patching.py` (use `scripts/bash/run_activation_patching_examples.sh`) | `results/patching/` |
| **E4** | What is decodable from z<sub>H</sub> / z<sub>L</sub>? | `scripts/probes/sweep_linear_probes.py`, `scripts/probes/train_linear_probes.py` | `results/probes/` |
| **E5** | How does z<sub>H</sub> evolve across steps? | `scripts/time_shift/batch_time_shift.py` | `results/time_shift/` |
| **E7b** | Convergence of predictions across steps | `scripts/reports/result_metrics_sudoku.py` | `results/metrics/`, `results/hamming_multi/` |
| **E8** | Linear constraint probes (row/col/box violations, naked singles, …) | `scripts/probes/e8_constraint_probes.py` | `results/probes/e8_constraint_probes/` |
| **E9** | Directed ablation along E8 directions (causal validation) | `scripts/directed_ablation/e9_directed_ablation.py` | `results/directed_ablation/` |
| **E9b** | MLP / non-linear probes & their directed ablation | `scripts/probes/nonlinear_probes.py`, `scripts/directed_ablation/e9b_nonlinear_directed_ablation.py` | `results/directed_ablation/nonlinear/` |
| **E10** | Sparse autoencoder study on z<sub>H</sub> (features + causal ablation) | `scripts/sae/sae_train.py`, `scripts/sae/sae_analyze_features.py`, `scripts/sae/sae_causal_ablation.py`, `scripts/sae/sae_sweep.py`, `scripts/sae/sae_plot.py` | `results/sae_study/` |
| Maze | Maze variants of E1/E2/E5/E8/E9/E10 | `scripts/sae/sae_collect_activations_maze.py`, `results/maze/...` | `results/maze/` |
| ARC | ARC-AGI replication of E4/E8/E9/E9b + H2/A/H4 (ARC-native features) | `scripts/arc/slurm_arc_suite.sbatch` (see `docs/PLAN_ARC_experiments.md`) | `results/arc/` |
| Controlled | Single-variable, larger-N replications of E1/E2/E5/E9 with stats | `scripts/controlled/run_all_controlled_experiments.py` | `results/controlled/` |

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
python scripts/plotting/plot_baseline_comparison.py --results_dir results/baseline_comparison
python scripts/plotting/plot_e8_e9.py
python scripts/plotting/plot_ablation_results.py
python scripts/plotting/plot_zh_trajectories.py
python scripts/plotting/generate_paper_figures.py     # rebuilds paper/figures/*.pdf
```

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
```

---

## 5. Reproducing the paper figures

After running E1/E2/E5/E8/E9/E10 (and their controlled / maze variants):

```bash
python scripts/plotting/generate_paper_figures.py
ls paper/figures/*.pdf
```

The committed `paper/figures/*.pdf` files are the exact assets used in the
manuscript; rebuilding them requires the corresponding `results/...` artefacts.

---

## 6. Large binary artefacts (gitignored — regenerate locally)

All *text* results (JSON / JSONL / YAML / Markdown) and figures (PDF / PNG) are
committed. The large binary caches are **gitignored** (`.pt` / `.npy` / `.npz`)
because they exceed GitHub's 100 MB/file limit and are fully regenerable. None of
these are needed for the ARC-AGI reproduction. To rebuild them:

| Artefact (gitignored) | Size | Regenerate with |
|---|---|---|
| `results/{maze/,}sae_study/activations_zH.pt` | 2.5–8.5 GB | `scripts/sae/sae_collect_activations.py` (maze: `sae_collect_activations_maze.py`) |
| `results/{maze/,}sae_study/sae_d*_l1*.pt` | ~32 MB each | `scripts/sae/sae_train.py` / `sae_sweep.py` |
| `results/maze/hardened/probe_geometry/probe_weights.pt` | 1.5 MB | `scripts/maze/linear_probes_maze.py --save_probe_weights` |
| `results/maze/hardened/trajectory_pca/trajectory_pca.npz` | 60 KB | `scripts/maze/` trajectory-PCA step |
| `data/`, `checkpoints/` | — | submodule build + HF download (see [README §2](../README.md)) |
