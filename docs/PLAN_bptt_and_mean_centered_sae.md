# Plan: BPTT-vs-stock HRM + mean-centered SAE for the feature-sparsity problem

## Context

Our mechanistic-interpretability fork of HRM (Wang et al. 2025, [arXiv:2506.21734]) trains
Sparse Autoencoders (SAEs) on the high-level state `z_H` to discover interpretable features.
Two findings motivate this work:

1. **Features are extremely distributed / many are dead.** On the current checkpoint the
   d=2048 / l1=0.01 SAE shows **704/2048 (~34%) dead features** on held-out data and **L0 ≈ 705
   active features per input** (`results/sae_study/sae_d2048_l10.01_features.json`,
   `feature_analysis/dead_feature_analysis.json`). No feature reaches |corr| > 0.5 with any
   constraint target — the code is dense and non-monosemantic.

2. **There are two plausible, independent causes**, and we will address both:
   - **Model side:** HRM is trained with a **one-step gradient approximation**
     ([hrm_act_v1.py:189-211](models/hrm/hrm_act_v1.py#L189-L211) — all H/L cycles except the
     last run under `torch.no_grad()`, and the carry is `.detach()`ed at
     [L221](models/hrm/hrm_act_v1.py#L221)). The hypothesis: gradients that never flow through
     the recurrence produce representations that are diffuse/entangled. We test this by training
     an HRM with **back-propagation through time (BPTT)** enabled and comparing the SAE-feature
     statistics of the two models.
   - **SAE side:** [arXiv:2605.31518] "On the Relationship Between Activation Outliers and Feature
     Death in SAEs" (Simon, Adams, Zou) shows dimension-level activation outliers
     (γ = ‖μ‖/‖σ‖) shift pre-activations at init so features anti-aligned with the mean are
     permanently negative → dead. Crucially they show a **learnable bias recovers these features
     too slowly at high γ** — which is exactly our situation: our SAE *already* has a learnable
     `pre_bias` ([sae.py:52,95](models/sae.py#L52)) yet still shows 34% deadness. Their fix is to
     **mean-center activations before SAE training**. This is a small, low-risk change.

**Decisions locked in** (from clarifying questions): within-step BPTT; retrain both stock and
BPTT HRM from scratch under identical configs; evaluate the mean-centering SAE fix as a
controlled A/B on the *existing* checkpoint's activations now (decoupled from the HRM retrain).

## Literature framing & our contribution

- **SAE foundations:** Bricken et al. 2023 (pre-encoder/decoder bias — our `pre_bias`),
  Gao et al. 2024 ([arXiv:2406.04093], TopK SAEs + dead-feature resampling — our
  `reinitialize_dead_features`, [sae.py:201](models/sae.py#L201)). We already implement both
  standard mitigations, so our residual deadness is *not* explained by missing them — it points
  to the activation-outlier mechanism of [arXiv:2605.31518].
- **Our novel angle:** the outlier/feature-death literature is evaluated on *feed-forward*
  transformer/vision/protein activations. HRM's `z_H` is a **recurrent, iteratively-refined**
  state. Two contributions: (a) test whether the γ-outlier → death relationship holds on a
  recurrent reasoning state and whether mean-centering recovers features there; (b) ask whether
  the *training objective* (one-step grad vs BPTT) changes the activation-outlier geometry (γ)
  and hence downstream feature death — a model-side lever the SAE literature does not consider.
  This connects our two tracks into a single story: *distributed/dead SAE features can stem from
  both the encoded model's training dynamics and the SAE's handling of activation shift.*

## Track 1 — Stock vs BPTT HRM (within-step BPTT)

**Precedent already in repo:** the baselines support `one_step_grad: false` for full BPTT
([vanilla_rnn.py:234-240](models/baselines/vanilla_rnn.py#L234-L240); configs
`config/arch/recurrent_transformer_standalone.yaml`, `universal_transformer_standalone.yaml`).
HRM itself has no such flag yet — we add one mirroring this pattern.

**Implementation:**
1. Add `one_step_grad: bool = True` to `HierarchicalReasoningModel_ACTV1Config`
   ([hrm_act_v1.py:31-57](models/hrm/hrm_act_v1.py#L31)).
2. In `_Inner.forward` ([hrm_act_v1.py:188-211](models/hrm/hrm_act_v1.py#L188)), branch on the
   flag: when `one_step_grad=True` keep current behavior; when `False`, run the full
   `H_cycles × L_cycles` loop **without** the `torch.no_grad()` wrapper so every segment is
   differentiated (the carry is still detached at [L221](models/hrm/hrm_act_v1.py#L221), so
   gradients stop at ACT-step boundaries — this is within-step BPTT). Keep the
   probe-recorder hooks intact.
3. New arch config `config/arch/hrm_v1_bptt.yaml` = copy of `hrm_v1.yaml` + `one_step_grad: false`.
4. New top-level config `config/cfg_pretrain_hrm_bptt.yaml` mirroring the stock Sudoku-Extreme
   settings, pointing `arch: hrm_v1_bptt`, distinct `checkpoint_path`.
5. SLURM scripts `scripts/bash/train_hrm_stock.sh` and `train_hrm_bptt.sh`, modeled on
   [train_vanilla_rnn.sh](scripts/bash/train_vanilla_rnn.sh) but using
   `torchrun --nproc-per-node 4` (4×A100 DDP) and the same seed for both runs.

**Prerequisites:** `data/` and `checkpoints/` are currently empty. Build the dataset first via
the Sudoku builder in `dataset/` (the stock config expects `data/sudoku-extreme-1k-aug-1000`).

**Time estimates (4×A100 node, DDP).** Anchored to the repo's own real budget: the baseline
SLURM scripts request **24 h on 1 GPU for 40 000 epochs** on this exact dataset
([train_vanilla_rnn.sh] `--time=23:59:00`, `cfg_pretrain_vanilla_rnn.yaml` `epochs=40000`). HRM
Sudoku-Extreme uses `epochs≈20000` (`docs/EXPERIMENTS.md`) but does more work per step (H+L
hierarchy + the ACT target-Q double forward at [hrm_act_v1.py:295](models/hrm/hrm_act_v1.py#L295)).

| Run | 1×A100 | 4×A100 (DDP, ~3.5× speedup) |
|-----|--------|------------------------------|
| **Stock HRM** (one-step grad) | ~12–24 h | **~4–8 h** |
| **BPTT HRM** (within-step) | ~30–60 h | **~10–18 h** |

BPTT multiplier ≈ 2–3×: backward now flows through ~4× the layers (all H/L cycles vs one), and
~3–4× higher activation memory may force a smaller per-GPU micro-batch + gradient accumulation,
costing additional wall-clock. These are order-of-magnitude figures — **pin them down with a
200-step calibration run** (tqdm/wandb already report steps/sec; extrapolate against
`total_steps = epochs · total_groups · mean_puzzle_examples / global_batch_size`,
[pretrain.py:174](pretrain.py#L174)) before committing the full job.

**Risk:** within-step BPTT may be less stable late in training (the repo already warns Q-learning
can destabilize). Mitigations: same gradient clipping/LR as stock, early-stop at 100% train acc,
checkpoint every eval.

## Track 2 — Mean-centered SAE (controlled A/B, existing activations)

**Faithful, minimal change** — store the empirical activation mean as a fixed (non-learnable)
buffer subtracted in `encode` and added back in `decode`, so the encode/decode interface still
works on *raw* activations (important: `scripts/sae/sae_causal_ablation.py` calls
`sae.encode(z_H)` / `sae.decode(h)` on raw `z_H`).

**Implementation:**
1. In `SparseAutoencoder.__init__` ([sae.py:39](models/sae.py#L39)) register
   `act_mean` buffer (zeros, shape `[input_dim]`) and add a `set_mean(mean)` method. Subtract
   `act_mean` in `encode` ([sae.py:95](models/sae.py#L95)) and add it back in `decode`
   ([sae.py:108](models/sae.py#L108)), *in addition to* the existing `pre_bias` (keep `pre_bias`
   for the learnable residual; centering removes the dominant shift up front). `TopKSparseAutoencoder`
   inherits this automatically.
2. In `sae_train.py` ([train_sae](scripts/sae/sae_train.py#L76)) add `--center_mean` flag:
   compute `mean = activations.mean(0)` once, call `sae.set_mean(mean)`, and persist `act_mean`
   plus a `center_mean` flag in the saved checkpoint `config` dict
   ([sae.py: torch.save block](scripts/sae/sae_train.py#L236)).
3. Optionally log γ = ‖μ‖/‖σ‖ for the activation set (per [arXiv:2605.31518]) into the features
   JSON — gives us the predictor the paper validates and a number to report.

**Controlled A/B protocol** (decoupled from Track 1, runs on existing
`results/sae_study/activations_zH.pt`):
- Reuse [sae_sweep.py](scripts/sae/sae_sweep.py) to run `{baseline, mean-centered}` ×
  `dict_size∈{1024,2048,4096}` × `l1∈{0.003,0.01,0.03}`, same seed.
- Compare via `get_feature_stats` ([sae.py:165](models/sae.py#L165)): **dead_count, L0,
  recon loss, mean_sparsity**, plus specialization (`sae_analyze_features.py`) and γ.
- Hypothesis: mean-centering drops dead_count sharply (paper claims it "eliminates outlier-induced
  death") at equal/better reconstruction.
- This is fast (existing SAE runs train in ~1 min on GPU per `_features.json`), so no SLURM needed —
  a single GPU interactive run suffices.

## Files to create / modify (summary)

- **Modify:** `models/hrm/hrm_act_v1.py` (one_step_grad flag), `models/sae.py` (act_mean buffer +
  set_mean), `scripts/sae/sae_train.py` (--center_mean), optionally `scripts/sae/sae_sweep.py`
  (sweep the flag).
- **Create:** `config/arch/hrm_v1_bptt.yaml`, `config/cfg_pretrain_hrm_bptt.yaml`,
  `scripts/bash/train_hrm_stock.sh`, `scripts/bash/train_hrm_bptt.sh`.

## Verification

- **Track 1 code:** unit-sanity — instantiate HRM with `one_step_grad=False`, run one forward,
  assert the inner H/L-cycle activations now `requires_grad` (negate the existing assertion at
  [hrm_act_v1.py:207](models/hrm/hrm_act_v1.py#L207)) and that `loss.backward()` populates grads
  on H_level/L_level params from earlier cycles. Then a 200-step calibration run per model to
  confirm convergence + record steps/sec for the time estimate.
- **Track 1 result:** after both models hit target train acc, run
  `sae_collect_activations.py` on each, train an identical SAE on each, compare dead_count/L0/γ.
- **Track 2 code:** assert round-trip on raw activations is unchanged in interface
  (`decode(encode(x))` close to current behavior when `act_mean=0`); with centering on, assert
  γ of centered activations ≈ 0 and dead_count drops in the A/B sweep.
- **End-to-end story:** one figure/table crossing (stock vs BPTT) × (baseline vs mean-centered
  SAE) on dead_count / L0 / specialization.

## Open items / risks

- Dataset + a stock checkpoint must be (re)built; `data/` and `checkpoints/` are empty now.
- Across-step BPTT (full unroll through ACT steps) is explicitly **out of scope** for now
  (needs gradient checkpointing; deferred).
- Time figures are anchored estimates — calibrate before launching full jobs.

Sources: [arXiv:2605.31518](https://arxiv.org/abs/2605.31518),
[arXiv:2406.04093](https://arxiv.org/abs/2406.04093),
[arXiv:2506.21734](https://arxiv.org/abs/2506.21734).
