# ARC-AGI Mechanistic-Interpretability Suite — Plan

Replicates the Sudoku/Maze MI experiments on the HRM **ARC-2** checkpoint
(`sapientinc/HRM-checkpoint-ARC-2`). The HRM architecture and intervention
engine are task-agnostic, so the suite reuses the existing
`scripts/core/` ablation/patching primitives and the controlled/maze drivers;
the ARC-specific work is (a) a native feature bank and (b) ARC structural
metrics.

## 0. Status

| Component | File | State |
|-----------|------|-------|
| ARC-2 checkpoint | `checkpoints/sapientinc-hrm-arc-2/checkpoint` | ✅ downloaded |
| ARC-AGI-2 raw data | `dataset/raw-data/ARC-AGI-2/data` | ✅ present (1000 train / 120 eval) |
| Dataset build | `data/arc-2-aug-1000/` | ⏳ building (`seed=42 num_aug=1000`) |
| Native features | `utils/arc_targets.py` | ✅ |
| ARC metrics / checkpoint | `scripts/arc/arc_common.py` | ✅ |
| Linear + MLP probes (E4/E8/E9b) | `scripts/arc/arc_linear_probes.py` | ✅ |
| Directed ablation (E9) | `scripts/arc/directed_ablation_arc.py` | ✅ |
| Policy A/H4 + causal subspace (H2) | `scripts/analysis/*` (`--task arc`) | ✅ registered |
| Figures | `scripts/arc/plot_arc_figures.py` | ✅ |
| Orchestrator | `scripts/arc/slurm_arc_suite.sbatch` | ✅ |

## 1. Why a new feature bank

Sudoku features (row/col/box violations, naked singles) and maze features
(on-path, walls, distance-to-goal) do not exist on ARC. ARC is an
input→output grid-transformation task on a 30×30 colour canvas. The native
feature families in `utils/arc_targets.py` are:

- **Grid geometry** — `input_height/width`, `output_height/width`,
  `size_preserved`, `(in/out)_inside_grid`, `is_eos`. Tests whether z encodes
  the grid's bounding box and the EOS-delimited shape.
- **Colour structure** — `input_colour`/`output_colour` (10-way multiclass),
  `num_(in/out)put_colours`, `input_is_background`, `colour_iou`.
- **Object structure** — `input_component_size` (4-connected same-colour blob),
  `is_object_boundary`, `num_same_colour_neighbours`. Tests object-level
  representations.
- **The transformation** — `per_cell_correct`, `colour_changed`,
  `same_as_input`. Tests whether z encodes the input→output edit itself.

Token encoding (matches `dataset/build_arc_dataset.py`): PAD=0, EOS=1,
colour `c∈0..9 → token c+2`. Labels arrive with the PAD region remapped to the
loss-ignore id `-100`, so `valid = label != -100` selects the grid + EOS.

## 2. Experiment map (Sudoku/Maze → ARC)

| ID | Question | ARC driver | Output |
|----|----------|-----------|--------|
| **E4/E8** | What is linearly decodable from z_H/z_L, per ACT step? | `arc_linear_probes.py --probe_type linear` | `results/arc/hardened/linear_probes/` |
| **E9b** | Non-linear decodability (MLP − linear delta) | `arc_linear_probes.py --probe_type mlp` | `results/arc/hardened/linear_probes_mlp/` |
| **E9** | Are probe directions *causal*? (project out, measure damage vs random) | `directed_ablation_arc.py` | `results/arc/hardened/directed_ablation/` |
| **H2** | Minimal causal subspace rank | `causal_subspace.py --task arc` | `results/arc/causal_subspace/` |
| **A** | Policy-improvement operator across steps | `policy_improvement.py --task arc` | `results/arc/policy_improvement/` |
| **H4** | H vs L contribution to value gain | `policy_decomposition.py --task arc` | `results/arc/policy_decomposition/` |

Protocol parity with Sudoku/Maze hardening: per-`(stream, step, target)`
independent probe, **puzzle-disjoint** train/val split, seed ensemble with
95% t-CI, majority baseline + headroom. Regression targets report R² but are
flagged non-reportable (unreliable on near-constant targets).

ARC structural metrics (`arc_common.arc_prediction_metrics`):
`token_acc, exact_solved, colour_cell_acc, eos_acc, background_acc,
shape_correct, height/width_correct, num_colours_correct, colour_iou`.
The policy `value` for ARC is `colour_cell_acc` (dense), with `exact_solved`
tracked separately (ARC's exact-match objective is sparse on ARC-2).

## 3. Running

Prerequisites:

```bash
# data (one-shot, slow due to num_aug=1000 augmentation hashing)
python dataset/build_arc_dataset.py     # after setting ARC-2 dirs in DataProcessConfig
#   dataset_dirs=["dataset/raw-data/ARC-AGI-2/data"], output_dir="data/arc-2-aug-1000"
# checkpoint
huggingface-cli download sapientinc/HRM-checkpoint-ARC-2 \
    --local-dir checkpoints/sapientinc-hrm-arc-2
```

> The dataset MUST be rebuilt with `seed=42, num_aug=1000` so the ~1.05M
> puzzle-identifier embeddings align with the checkpoint's `puzzle_emb.weights`
> (eval tasks share IDs with their training demonstrations; the loader assigns
> the checkpoint's embedding table by shape).

Full suite (GPU):

```bash
sbatch -p gpu --gres=gpu:1 scripts/arc/slurm_arc_suite.sbatch
# or smaller: N=50 sbatch ... scripts/arc/slurm_arc_suite.sbatch
```

CPU smoke (a handful of puzzles, two steps, two seeds):

```bash
python scripts/arc/arc_linear_probes.py \
    --num_puzzles 6 --steps 0,1 --epochs 5 --positions_per_sample 64 \
    --seeds 0,1 --device cpu --save_probe_weights \
    --output_dir /tmp/arc_probe_smoke
python scripts/arc/directed_ablation_arc.py \
    --num_puzzles 6 --device cpu \
    --probe_weights /tmp/arc_probe_smoke/probe_weights.pt \
    --output_dir /tmp/arc_e9_smoke
```

## 4. Dependencies between stages

`arc_linear_probes.py --save_probe_weights` produces
`probe_weights.pt` (per-binary-feature unit directions with
`stream`/`W_mean`/`val_acc_mean`/`target`), consumed by both
`directed_ablation_arc.py` (E9) and `causal_subspace.py --task arc` (H2). Run
the linear probes first. All other stages are independent.

## 5. Not ported (and why)

- **E10 (SAE study)** — heavy, separate training run; can be added later via
  `scripts/sae/sae_collect_activations_*` with an ARC collector.
- **E3 activation patching across puzzles** — the cross-puzzle patching engine is
  task-agnostic; an ARC wrapper analogous to the maze patching study is a
  follow-up, not part of this core suite.
