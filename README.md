# Dissecting Hierarchical Reasoning Models: A Mechanistic Study

Code and results for a mechanistic-interpretability study of the
[Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734) across
**Sudoku-Extreme**, **Maze-Hard**, and **ARC-AGI-2**.

HRM is a 27M-parameter recurrent encoder with two interacting Transformer
modules: a slow high-level state z<sub>H</sub> updated once per cycle, and a
fast low-level state z<sub>L</sub> updated several times per cycle, iterated for
up to 16 ACT segments. It reasons in latent space, so its intermediate steps are
never verbalized — which is exactly what makes it hard to audit.

This repository asks three questions and ships the code and aggregates behind
each answer:

| | Question | Finding |
|---|---|---|
| **RQ1** | What functional roles do z<sub>H</sub> and z<sub>L</sub> play? | HRM iteratively refines a puzzle-specific solution state. The locus is task-dependent: z<sub>H</sub> on Sudoku, z<sub>L</sub> on Maze. |
| **RQ2** | Do linearly decodable directions carry the computation? | No. Task variables decode at ~90%, yet ablating those probe directions is indistinguishable from random controls. |
| **RQ3** | Is there a compact, causally dominant feature set? | No. SAE ablations bite harder than probe directions, but the top-50 features are no more important than 50 random ones — including under full BPTT training. |

![](./assets/hrm.png)

---

## Reproducing the results

Three tiers, cheapest first. Most readers want Tier 1.

### Tier 1 — redraw every paper figure (CPU, ~1 minute, no downloads)

Every reported number is committed as a small `aggregate.json` / `analysis.json`,
so the figures rebuild without checkpoints or datasets.

```bash
git clone https://github.com/LeoRodrigues05/HRM && cd HRM
pip install -r requirements.txt
bash reproduce_figures.sh
```

Writes 23 PDFs and their PNGs to `results/Final/figures/`. Each generator prints
the values it plots, so you can diff them against the paper directly — e.g. the
cross-task ladder prints Sudoku `Top-50 SAE=-3.63pp  Random-50 SAE=-3.42pp`,
matching Table 15.

### Tier 2 — re-run the analyses (GPU, hours)

Needs the checkpoints and datasets. Everything comes from Hugging Face.

```bash
git submodule update --init --recursive       # ARC raw data

# Released HRM checkpoints the analysis runs against
huggingface-cli download sapientinc/HRM-checkpoint-sudoku-extreme \
    --local-dir checkpoints/sapientinc-sudoku-extreme
huggingface-cli download sapientinc/HRM-checkpoint-maze-30x30-hard \
    --local-dir checkpoints/sapientinc-hrm-maze-30x30-hard
huggingface-cli download sapientinc/HRM-checkpoint-ARC-2 \
    --local-dir checkpoints/sapientinc-hrm-arc-2

# Models trained for this paper (baselines, ARC-adapted, stock/BPTT pair, SAEs,
# probe weights)
huggingface-cli download LeoRodrigues05/hrm-interpretability-checkpoints \
    --local-dir checkpoints/paper

# Datasets bound to those checkpoints (see "A note on ARC-AGI-2" below)
huggingface-cli download --repo-type dataset \
    LeoRodrigues05/hrm-interpretability-data --local-dir data

# Maze data comes from upstream
huggingface-cli download --repo-type dataset sapientinc/maze-30x30-hard-1k \
    --local-dir data/maze-30x30-hard-1k
```

Then run whichever suite you need:

```bash
# Sudoku: single-variable controlled re-runs of the interventions (Finding 1)
python scripts/controlled/run_all_controlled_experiments.py

# Maze: the hardened variants used in the paper
bash scripts/maze/run_all_maze_hardened.sh

# ARC-AGI-2: probes + SAE study
bash scripts/arc/run_arc_end_to_end.sh full

# Probes and directed ablation (Finding 2)
python scripts/probes/e8_constraint_probes.py
python scripts/directed_ablation/e9_directed_ablation.py

# Sparse autoencoders (Finding 3)
python scripts/sae/sae_collect_activations.py     # writes a multi-GB cache
python scripts/sae/sae_train.py
python scripts/sae/sae_causal_ablation.py

# 5-model baseline comparison (Figure 1, Table 4)
bash scripts/bash/eval_all_baselines.sh
```

Results land in `results/{Sudoku,Maze,ARC}/`, alongside the committed
aggregates. [`docs/EXPERIMENTS.md §3`](docs/EXPERIMENTS.md) maps every
experiment ID in the paper to its driver script and output directory.

### Tier 3 — retrain from scratch (multi-GPU, days)

```bash
bash scripts/bash/Initialize_HRM_Repo.sh    # uv venv + CUDA + PyTorch + FA + data
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000
```

Training commands for HRM, the four baselines, and the stock/BPTT pair are in
[`docs/EXPERIMENTS.md §1`](docs/EXPERIMENTS.md).

---

## A note on ARC-AGI-2

The ARC model is **not** a fine-tune. HRM identifies each puzzle by an index into
a learned `puzzle_emb` table assigned at dataset-build time, so the released
checkpoint scores ~0 against any locally rebuilt ARC-2 set.
`scripts/arc/adapt_puzzle_emb.sh` re-fits *only* that table (`puzzle_emb_lr=1e-2`)
with the reasoning core frozen (`lr=1e-9`), leaving the mechanism under study
untouched.

This is why `data/arc-2-evalonly` is published rather than rebuilt: it must be the
exact build the embedding was fitted against. Do not rebuild it between adaptation
and analysis. Full rationale in [`docs/EXPERIMENTS.md §7`](docs/EXPERIMENTS.md).

---

## Layout

```
models/
  hrm/hrm_act_v1.py     HRM with ACT; `one_step_grad` toggles the BPTT variant
  baselines/            Recurrent (vanilla_rnn.py) / Universal Transformer,
                        Plain Transformer, GRU (standard_rnn.py)
  hrm_v2/               Sparse-attention + constraint-head variant (exploratory)
  sae.py                Sparse autoencoder (L1 and top-k)
scripts/
  core/                 Activation patching / ablation engine used by every
                        intervention experiment
  ablation/ freeze/ patching/ time_shift/    Per-experiment drivers (Finding 1)
  controlled/           Larger-N, single-variable re-runs -- what the paper cites
  probes/               Linear, sweep, MLP, constraint probes (Finding 2)
  directed_ablation/    Ablation along probe directions vs random controls
  sae/                  SAE train / analyse / causal ablation (Finding 3)
  maze/ arc/            Task-specific variants of the above
  plotting/             Manuscript figure generators
  analysis/ reports/    Evaluation, trajectories, HTML reports
config/                 Hydra configs; arch/hrm_v1{,_bptt}.yaml differ by one line
dataset/                Sudoku / Maze / ARC builders + raw-data submodules
results/Final/          Curated paper-backing aggregates, reports, figures
results/{Sudoku,Maze,ARC}/   Per-experiment aggregates and provenance
experiments/paper_replication/   Replication of the *original* HRM paper's claims
```

`results/Final/` is the curated set behind the manuscript; its
[`README`](results/Final/README.md) maps each finding to the artefacts
supporting it. Every result directory carries a `_meta.json` recording the git
SHA, seed, sample size, and GPU of the run that produced it.

Two things are intentionally absent, both regenerable and both documented with
their exact rebuild command in [`docs/EXPERIMENTS.md §6`](docs/EXPERIMENTS.md):
large binaries (on Hugging Face) and raw per-example records (`per_puzzle.jsonl`
and friends — the committed aggregate is the reported artefact).

---

## Notes

- Driver scripts default to writing under `results/{Sudoku,Maze,ARC}/`, the same
  tree as the committed aggregates. Pass `--output_dir` to keep a run separate.
- Small-sample (1k) Sudoku training shows ~±2% accuracy variance; the late-stage
  Q-learning loss can destabilize, so early-stop once train accuracy hits 100%.
- `scripts/bash/*.sh` activate `.venv` if present, else a conda env named
  `${HRM_CONDA_ENV:-hrm}`. Nothing is hard-coded to a machine: scripts resolve
  the repo root via `git rev-parse --show-toplevel`.
- Multi-GPU training uses `torchrun --nproc-per-node 8`; single-GPU works by
  substituting plain `python` and lowering `global_batch_size`.
- SLURM headers in `scripts/**/*.sbatch` and `train_*.sh` can be ignored when
  running locally — each script also works when invoked directly.

## Citation

The upstream model:

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

## Acknowledgements

This repository is a research fork of the
[Hierarchical Reasoning Model](https://github.com/sapientinc/HRM) codebase by
Sapient Intelligence, and uses their released Sudoku-Extreme, Maze-Hard, and
ARC-2 checkpoints.

Parts of the interpretability tooling, the release cleanup, and the
documentation in this repository were developed with assistance from
[Claude](https://claude.com/claude-code) (Anthropic). All experimental design,
analysis, and reported results are the authors'.

## License

See [LICENSE](LICENSE).
