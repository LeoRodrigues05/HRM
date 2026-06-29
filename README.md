# Tell Me Why: Mechanistic Interpretability of the Hierarchical Reasoning Model

A research fork of the
[Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734)
codebase, extended with a full mechanistic-interpretability (MI) suite that
asks **what z<sub>H</sub> and z<sub>L</sub> encode**, **which steps of the ACT
recurrence matter**, and **which directions in latent space are *causally*
responsible for solving Sudoku** — not just statistically correlated with the
solution.

The original HRM is a 27 M-parameter recurrent transformer with two
interdependent modules — a high-level state z<sub>H</sub> (slow, abstract
planning) and a low-level state z<sub>L</sub> (fast, per-cell computation) —
iterated for up to 16 ACT steps with a Q-learning halting head. With only 1 000
training examples it nearly solves Sudoku-Extreme, the 30×30 hard maze
distribution, and is competitive on ARC-AGI-2.

![](./assets/hrm.png)

## What this repository adds on top of upstream HRM

- **Baselines**: Vanilla RNN, GRU, Plain Transformer, and Universal Transformer
  trained on the same Sudoku-Extreme data (`models/baselines/`,
  `config/cfg_pretrain_*.yaml`, `scripts/bash/train_*.sh`).
- **HRM v2** with sparse constraint-aware attention and an explicit
  constraint-satisfaction head (`models/hrm_v2/`, `pretrain_v2.py`,
  `config/cfg_pretrain_v2.yaml`).
- **Activation patching / ablation engine** (`scripts/core/`) used by every
  intervention experiment.
- **Interpretability experiment drivers** (`scripts/{ablation,freeze,patching,
  time_shift,probes,directed_ablation,sae,controlled}/`) covering ablation,
  freeze, time-shift, linear / non-linear / sparse-autoencoder probes, and
  directed ablation along discovered directions.
- **Plotting and report tooling** (`scripts/plotting/`, `scripts/reports/`).
- **Paper-replication scripts** (`experiments/paper_replication/`) that
  re-derive every claim in the original HRM paper from the released
  checkpoint.

A complete experiment ↔ script ↔ output map lives in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Open follow-ups and architectural
extensions are catalogued in [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md).

---

## Replication in three steps

### 1. Bootstrap the environment

Linux + NVIDIA GPU (Ampere or newer recommended) is assumed.

```bash
git clone <this-repo-url> HRM && cd HRM
git submodule update --init --recursive
bash scripts/bash/Initialize_HRM_Repo.sh        # uv venv + CUDA 12.6 + PyTorch + FA + deps + Sudoku data
source .venv/bin/activate
```

Useful skip flags for the bootstrap script:

```bash
SKIP_CUDA=1 SKIP_APT=1 bash scripts/bash/Initialize_HRM_Repo.sh   # already have CUDA + sudo-less env
SKIP_FA=1   bash scripts/bash/Initialize_HRM_Repo.sh               # skip flash-attn (CPU-only / unsupported GPU)
SKIP_DATA=1 bash scripts/bash/Initialize_HRM_Repo.sh               # do not rebuild the Sudoku dataset
```

For Hopper GPUs install FlashAttention 3 instead:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install && cd ../..
```

(Optional) experiment tracking:

```bash
wandb login
```

### 2. Get the model checkpoints and data

The interpretability experiments are designed to run against the released
pretrained HRM checkpoints:

```bash
mkdir -p checkpoints
huggingface-cli download sapientinc/HRM-checkpoint-sudoku-extreme \
    --local-dir checkpoints/sapientinc-sudoku-extreme
huggingface-cli download sapientinc/HRM-checkpoint-maze-30x30-hard \
    --local-dir checkpoints/sapientinc-hrm-maze-30x30-hard
huggingface-cli download sapientinc/HRM-checkpoint-ARC-2 \
    --local-dir checkpoints/sapientinc-hrm-arc-2     # only if you want ARC results
```

Datasets are reproducible from source:

```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000              # used by every interpretability script
python dataset/build_maze_dataset.py                  # for maze MI experiments
python dataset/build_arc_dataset.py                   # ARC-1 (default dirs); requires submodules
# ARC-2 is built by the ARC wrapper scripts directly (explicit ARC-AGI-2 paths) —
# see docs/PLAN_ARC_path_A_embedding_adaptation.md; no manual build needed.
```

You can re-train HRM (and the four baselines) yourself instead — exact commands
and configs are listed in [`docs/EXPERIMENTS.md §1`](docs/EXPERIMENTS.md#1-training).

### 3. Run the experiment suite

End-to-end orchestrators (run any subset):

```bash
# Paper-claims replication on the released HRM checkpoint
python experiments/paper_replication/run_all_experiments.py

# Single-variable, larger-N controlled re-runs of E1 / E2 / E5 / E9
python scripts/controlled/run_all_controlled_experiments.py

# 5-model baseline comparison (HRM vs Vanilla RNN, GRU, Plain & Universal Transformer)
bash scripts/bash/eval_all_baselines.sh

# Constraint probes (E8) and directed ablation (E9)
python scripts/probes/e8_constraint_probes.py
python scripts/directed_ablation/e9_directed_ablation.py

# Sparse-autoencoder study (E10)
python scripts/sae/sae_train.py
python scripts/sae/sae_analyze_features.py
python scripts/sae/sae_causal_ablation.py

# Regenerate every committed paper figure
python scripts/plotting/generate_paper_figures.py
```

A per-experiment table (`E1…E10`, maze variants, controlled runs) with the
exact driver script and output directory is in
[`docs/EXPERIMENTS.md §3`](docs/EXPERIMENTS.md#3-mechanistic-interpretability-experiments).

---

## Project layout (short version)

```
config/                Hydra configs (HRM, HRM v2, baselines)
models/                HRM (`hrm/`), HRM v2 (`hrm_v2/`), baselines (`baselines/`)
dataset/               Sudoku / Maze / ARC dataset builders + raw-data submodules
pretrain.py            Generic Hydra training entrypoint
pretrain_v2.py         HRM v2 training entrypoint
evaluate.py            Distributed evaluation + per-puzzle prediction dump
arc_eval.ipynb         ARC submission scoring notebook
puzzle_dataset.py      Shared dataloader

scripts/
  core/                Activation-patching / ablation engine
  ablation/  freeze/  patching/  time_shift/    Per-experiment intervention drivers
  probes/              Linear, sweep, MLP, and constraint-specific probes
  directed_ablation/   E9 / E9b directed ablation along probe directions
  sae/                 E10 sparse-autoencoder study (train, analyse, causal)
  controlled/          Tighter, single-variable controlled re-runs of E1/E2/E5/E9
  analysis/            Evaluation, difficulty-stratified, trajectory tools
  plotting/            Paper / presentation figure generators
  reports/             Sudoku HTML and metric reports
  bash/                Convenience wrappers (Init, train_*, eval_*, single_inference, ...)

experiments/paper_replication/   Phase-7 replication of the original HRM claims

docs/
  EXPERIMENTS.md       Full experiment + script catalogue (this is the long one)
  FUTURE_WORK.md       Open follow-ups / extensions

paper/                 LaTeX source of the accompanying manuscript (gitignored except figures/)
```

The directories `data/`, `outputs/`, `wandb/`, `logs/`, `checkpoints/`, and the
LaTeX build artefacts are intentionally gitignored — they are large and
fully reproducible from the steps above.

---

## Notes & caveats

- Small-sample (1k) Sudoku training shows a typical ±2 % accuracy variance.
  For Sudoku-Extreme the late-stage Q-learning loss can become unstable;
  early-stop once train accuracy reaches 100 %.
- All `scripts/bash/*.sh` wrappers source `.venv/bin/activate` if it exists and
  fall back to a conda environment named `${HRM_CONDA_ENV:-hrm}` otherwise.
  No paths are hard-coded to a specific user — every script `cd`s to
  `git rev-parse --show-toplevel` (or `pwd`) before running.
- Multi-GPU training uses `torchrun --nproc-per-node 8`; single-GPU training
  works by replacing the launcher with plain `python` and lowering
  `global_batch_size`.

## Citation

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}
}
```

This fork additionally accompanies an interpretability manuscript whose source
lives under `paper/` (kept out of the public repository); the committed
`paper/figures/*.pdf` files are the exact assets used in the manuscript.

## License

See [LICENSE](LICENSE).
