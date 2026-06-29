# Implementation Plan — H1, H2, H4: Why HRM's computation is distributed, what its causal code is, and where the hierarchy earns its keep

**Audience:** an engineer/LLM implementing this in the HRM repo. Follow it literally;
do **not** redesign the interfaces named here — they are verified against the codebase
(file:line references given). When in doubt, re-read the cited source file. Ask before
inventing a new interface.

**Companion doc:** `docs/PLAN_policy_improvement_experiments.md` (Experiments A & D, already
implemented in `scripts/analysis/policy_improvement.py`). H4 extends that work; reuse its
helpers, do not duplicate them.

---

## 0. Scientific framing (why these three)

The existing paper's central results on Sudoku are **negative/distributed**: constraint info is
linearly decodable from `z_H` (~88%) but ablating those probe directions does nothing
(`results/directed_ablation/`), and no sparse SAE subset localizes the computation
(`results/sae_study/causal_ablation/`). "We looked and couldn't localize it" does not clear an
A\* bar. These three experiments convert the negative into a positive, mechanistic story:

- **H1 — Distributedness is *caused by* the truncated 1-step gradient.** Hypothesis: a model
  trained with full BPTT is *more localizable* (probe directions become causal, minimal causal
  subspace shrinks) than the 1-step-gradient model. If true, distributedness is a **training-
  induced law of truncated-gradient latent reasoners**, not an HRM quirk. This is the headline
  pivot.
- **H2 — The causal code is low-rank, redundant, and *rotated away* from the readable basis.**
  We have two endpoints only (readable directions = 0% causal; full `z_H` = −20% causal) and
  nothing in between. H2 finds the **minimal causal subspace** and measures its rank, its
  damage-vs-rank curve (distributed vs localized), and its alignment to probe/SAE bases. Turns
  "spread out" into "rank-r, redundant, and not where probes look."
- **H4 — The hierarchy localizes the policy-improvement operator.** Decompose each ACT step's
  policy improvement into the **H-update** vs the **L-cycles** contribution, and show the
  equally-accurate flat baseline lacks a comparably separable handle. Makes the policy story
  (Exps A/D) **HRM-specific** instead of "generic recurrence."

**Dependency order:** implement **H2 first** (it builds the reusable minimal-causal-subspace
tool), then **H1** (its scorecard *consumes* the H2 tool), then **H4** (independent). The SLURM
orchestrator runs them in that order.

**Scope:** primary task is **Sudoku** (where the distributedness findings live). Maze is a
secondary replication where noted. ARC is explicitly **out of scope** for this plan (later work).

---

## 1. Ground-truth facts about the codebase (DO NOT re-derive)

### 1.1 Model, loaders, per-step cache (verified)
- **HRM loader (v1/v2 only):** `from scripts.controlled.controlled_common import
  load_model_and_dataloader, collect_puzzles, bootstrap_ci, find_checkpoint, extract_batch`
  - `model, test_loader, config = load_model_and_dataloader(checkpoint_path, device)` returns the
    **unwrapped** `HierarchicalReasoningModel_ACTV1` (has `.inner`, `.config`, `.initial_carry`).
    It reads `all_config.yaml`/`config.yaml` next to the checkpoint, so it works on **any HRM
    checkpoint dir** (this is what makes H1 checkpoint-agnostic). It **raises** on non-HRM models
    (`controlled_common.py:151`) — the Universal Transformer baseline needs a separate loader
    (see §1.6).
  - `collect_puzzles(test_loader, device, num_puzzles, seed=42, puzzle_indices_path=None)`
    → `List[(puzzle_idx:int, batch:Dict[str,Tensor])]`; `batch` has `inputs`, `labels`,
    `puzzle_identifiers`, each `[1, seq_len]` on `device`.
  - `bootstrap_ci(list_of_floats)` → `{"mean","ci_lower","ci_upper","std","n"}`.
- **Forward + cache + ablation engine:** `from scripts.core.activation_ablation import
  ActivationAblator, ActivationCache`
  - `ablator = ActivationAblator(model, device=device)`
  - `ablator.run_and_cache_activations(batch, cache: Dict[int,ActivationCache], max_steps=16)`
    fills `cache[s]` for `s=0..max_steps-1` (halting is step-count based, so **all steps always
    present**; `activation_ablation.py:208`).
  - `ablator.run_with_ablation(batch, ablate_level="H"|"L"|"both", ablate_steps=None,
    ablate_positions=None, max_steps=16, ablation_value=0.0)` → `(final_outputs, ablated_cache,
    ablation_info)`. **Zeroing** ablation. `ablate_steps=None` = all steps. This is the
    **full-`z_H` ablation reference** for H2 (`ablate_level="H"`, all steps, value 0.0).
- **`ActivationCache` fields (per step `s`)** — torch tensors:
  - `.logits` `[1, seq_len, vocab]` — **already answer-aligned**: model does
    `lm_head(z_H)[:, puzzle_emb_len:]` so the puzzle-emb prefix is **already stripped**
    (`hrm_act_v1.py:222`). `logits[0,j]` aligns 1:1 with `labels[0,j]`. May be bf16 → `.float()`.
  - `.preds` `[1, seq_len]` = `logits.argmax(-1)`.
  - `.z_H`, `.z_L` `[1, puzzle_emb_len+seq_len, hidden]` — step **input** states (carry into step).
  - `.z_H_out`, `.z_L_out` — step **output** states (what `lm_head` reads is `z_H_out`).
  - `.q_halt_logits`, `.q_continue_logits` `[1]`; `.step` int.

### 1.2 Architecture constants (verified from `all_config.yaml`)
Both Sudoku and Maze HRM checkpoints: `H_cycles=2, L_cycles=2, H_layers=4, L_layers=4,
hidden_size=512, num_heads=8, expansion=4, halt_max_steps=16, pos_encodings=rope,
puzzle_emb_ndim=512`. Therefore **`puzzle_emb_len = ceil(512/512) = 1`** (call it `pel`). Sudoku
`seq_len=81` (+pel ⇒ `z_H` is `[1,82,512]`); Maze `seq_len=900` (`[1,901,512]`). Sudoku loss is
`stablemax_cross_entropy`; this does not matter for eval (we read logits directly).

### 1.3 The HRM inner forward (verified, `hrm_act_v1.py:188-227`) — critical for H4
Within **one ACT step**, with `H_cycles=L_cycles=2`, the update schedule is:
```python
input_embeddings = model.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])  # [1, pel+seq, D]
cos_sin = model.inner.rotary_emb() if hasattr(model.inner, "rotary_emb") else None
seq_info = dict(cos_sin=cos_sin)
z_H, z_L = carry.z_H, carry.z_L           # = cache[s].z_H, cache[s].z_L
with torch.no_grad():
    for _H in range(H_cycles):            # 2
        for _L in range(L_cycles):        # 2
            if not (_H == H_cycles-1 and _L == L_cycles-1):
                z_L = model.inner.L_level(z_L, z_H + input_embeddings, **seq_info)
        if not (_H == H_cycles-1):
            z_H = model.inner.H_level(z_H, z_L, **seq_info)
# "1-step grad" tail (we run under no_grad for analysis):
z_L = model.inner.L_level(z_L, z_H + input_embeddings, **seq_info)
z_H = model.inner.H_level(z_H, z_L, **seq_info)         # this z_H == cache[s].z_H_out
logits = model.inner.lm_head(z_H)[:, pel:]              # == cache[s].logits
```
Net effect per step: `z_L` updated 4×, `z_H` updated 2×. **`lm_head` reads `z_H` only** — `z_L`
influences the policy only *through* the next `H_level` call. `model.inner.H_level(a, b)` computes
`layers(rms_norm(a + b))` (input-injection add; `hrm_act_v1.py:92-99`). The reasoning modules are
`model.inner.H_level` and `model.inner.L_level` (`HierarchicalReasoningModel_ACTV1ReasoningModule`).

### 1.4 Subspace / direction ablation engine (verified, `e9_directed_ablation.py:154-242`) — for H2
```python
from scripts.directed_ablation.e9_directed_ablation import DirectionalAblator, load_model_and_data
abl = DirectionalAblator(model, device=device)
final_out, cache = abl.run_with_directional_ablation(
    batch,
    direction=torch.zeros(D),            # ignored when direction_matrix is given, but pass a [D] tensor
    ablate_level="H",                    # "H" or "L"
    ablate_steps=None,                   # all steps
    max_steps=16,
    direction_matrix=Q,                  # [K, D] -> projects out the K-dim subspace span(Q)
)
```
It orthonormalizes via `torch.linalg.qr(direction_matrix.T)` then applies
`z' = z - Q Qᵀ z` at every step (`e9_directed_ablation.py:181-216`). **Single direction:** pass
`direction=[D]` and `direction_matrix=None`. This is the exact engine H2 needs — do not rewrite it.

### 1.5 Probe & SAE artifacts (verified to exist) — for H1, H2
- Probe weights (Sudoku, current 1-step HRM): `results/probes/e8_constraint_probes/probe_weights.pt`
  — dict `{key: {"W":[out,in], "b":[out], "val_score":float, "z_level":"H"|"L", "step":int,
  "target":str, "W_per_seed":[...], ...}}` (`e8_constraint_probes.py:716-734`). Per-target
  direction extraction pattern is `select_best_directions(...)` (`e9_directed_ablation.py:313`).
- Probe trainer + label derivation (reusable, importable):
  `from scripts.probes.e8_constraint_probes import derive_per_cell_labels, train_binary,
  LinearProbe, CONSTRAINT_DIRECTIONS` (the constraint target list is in `e9` as
  `CONSTRAINT_DIRECTIONS`). `derive_per_cell_labels(preds[B,81], targets[B,81], inputs[B,81])`
  → dict of `[B,81]` label tensors. **Sudoku-only** (81-cell grid logic).
- SAE study: checkpoints `results/sae_study/sae_d2048_l10.01.pt` (the paper config), activation
  bank `results/sae_study/activations_zH.pt` (**2.6 GB**, `[N, steps, 81, D]`), maze bank
  `results/maze/sae_study/activations_zH.pt`. SAE classes:
  `from models.sae import SparseAutoencoder, TopKSparseAutoencoder` with `.encode`, `.decode`,
  `.dict_size`, `.input_dim`. SAE feature ablator: `SAEFeatureAblator` in
  `scripts/sae/sae_causal_ablation.py:174`.
- **Checkpoint-specificity (critical for H1):** probe weights and SAEs are only valid for the
  exact model whose activations they were fit on. For any **new** checkpoint (HRM-BPTT), you must
  **re-collect activations + retrain probes (and SAE if used)** on that checkpoint. The existing
  `e8`/`e9`/`sae_causal_ablation` scripts **hardcode** the Sudoku checkpoint in their local
  `load_model_and_data()` — do **not** reuse those entry points for other checkpoints. H1's
  scorecard script (§3) is self-contained and takes `--checkpoint`.

### 1.6 Universal Transformer baseline (verified, `models/baselines/universal_transformer.py`) — for H1 Track B and H4 flat comparison
- Class `UniversalTransformerModel` with `.inner` (`UniversalTransformer_Inner`), single recurrent
  state **`z`** (no `z_H`/`z_L`). InnerCarry is `UniversalTransformerInnerCarry(z=[B,T,D])`.
- Config flag **`one_step_grad`** (`universal_transformer.py:68`): `True` (default) = HRM-style
  truncated gradient (no-grad on all but last iteration); `False` = full BPTT through all
  `num_iterations`. Inner forward `universal_transformer.py:215-248`; readout
  `lm_head(z)[:, pel:]` (`:245`); per-iteration block is `_apply_shared_block(z, input_emb,
  cos_sin)` (`:208-213`).
- **Verified checkpoint pair (same arch `num_shared_layers=4`, `hidden=768`, same data
  `sudoku-extreme-1k-aug-1000`):**
  - `checkpoints/baselines/universal_transformer/` — `one_step_grad=True` (default),
    `num_iterations=4`, `halt_max_steps=16`. **The 1-step-gradient model.** Note: checkpoints are
    per-step files (`step_26040`, …) — use the latest `step_*` file (largest number) that is a
    plain state-dict, not a `*_all_preds.*` file.
  - `checkpoints/baselines/universal_transformer_standalone/` — `one_step_grad=false`,
    `num_iterations=16`, `halt_max_steps=1`. **The full-BPTT model.**
  - (Also `recurrent_transformer_standalone/` = `VanillaRNNModel`, `one_step_grad=false`,
    `num_iterations=16` — a second BPTT point if wanted; different class, lower priority.)
- **Caveat to record in results:** the UT pair differs in *both* gradient regime and recurrence
  layout (4 ACT steps × 4 iters vs 1 ACT step × 16 iters), so it is a **strong-but-not-airtight**
  control. The airtight control is HRM-1step vs HRM-BPTT (identical arch); run both, lead with
  HRM-BPTT when it lands, use the UT pair for the result available **now**.

### 1.7 Compute environment (HARD constraints)
- Login node has **no usable GPU**. All real runs via SLURM: `sbatch -p gpu --gres=gpu:1`, env
  `conda run --no-capture-output -n hrm python -u ...`, `--device cuda`. Per-user limits:
  `--mem ≤ 64G`, `-n ≤ 12`. Mirror `scripts/analysis/slurm_policy_improvement.sbatch`.
- CPU is only for an import/1-puzzle/`max_steps=2` smoke to verify the code path and write path.
  A full 900-token maze forward on CPU exceeds minutes — never do full CPU runs.
- `results/sae_study/activations_zH.pt` is 2.6 GB: load on the GPU node, and **subsample**
  (e.g. ≤ 200k rows) before any SVD/PCA — do not SVD the whole bank.

---

## 2. Experiment H2 — Minimal causal subspace (implement FIRST)

**Question:** what is the smallest subspace of `z_H` whose ablation reproduces the full-`z_H`
ablation damage, how does damage scale with rank (distributed vs localized), and is that causal
subspace aligned with the directions probes/SAEs read?

### 2.1 Files to create
1. `scripts/analysis/causal_subspace.py` — compute (importable `find_causal_subspace` + CLI).
2. `scripts/analysis/plot_causal_subspace.py` — figures from the JSON.
(SLURM is shared, §6.)

Outputs (Sudoku → `results/controlled/causal_subspace/`, Maze → `results/maze/causal_subspace/`):
- `subspace_curve.json` — damage vs rank for each direction-ordering.
- `alignment.json` — projection energy of probe/SAE bases into the causal subspace.
- `_meta.json` (via `scripts.core.provenance.write_meta`, in try/except).
- Figures → `results/reports/causal_subspace_figures/`.

### 2.2 Candidate direction pools (build these `[K, D]` bases, all unit-orthonormalized)
For the chosen `z_level="H"` and the model under test:
1. **PCA pool** — primary. Collect `z_H_out` over `N_pca` puzzles × the analyzed steps (reuse
   `ActivationAblator.run_and_cache_activations`; take `cache[s].z_H_out[0, pel:]` → `[seq,D]`,
   cast float, mask `labels!=-100` for Sudoku take all 81 cells). Stack to `[M, D]`, subsample
   `M ≤ 200k`, center, run `torch.pca_lowrank` or `torch.linalg.svd` on the covariance → ordered
   orthonormal directions `V[D, D]` (columns = PCs by variance).
2. **Random pool** — `R` independent random orthonormal bases (QR of Gaussian `[D,D]`), for the
   control curve (mean ± CI across the `R` draws).
3. **Probe+SAE pool** — the readable directions: stack the per-target probe vectors (from
   `probe_weights.pt`, pattern of `select_best_directions`) and, optionally, top-SAE decoder
   columns (`sae.decoder.weight` columns for the top-k features by firing rate, as in
   `select_top_features`). Orthonormalize. Used for the **alignment** analysis and as one ordering
   in the damage curve.

### 2.3 Damage-vs-rank curve (the core measurement)
Reference damage: run full-`z_H` zeroing once per puzzle via
`ActivationAblator.run_with_ablation(batch, ablate_level="H", ablate_steps=None,
ablation_value=0.0)`; `Δfull = value_ablated − value_baseline` (use the **task value**, §2.6).
Expected `Δfull ≈ −0.19…−0.20` cell-acc on Sudoku (sanity).

For each ordering ∈ {PCA-top, PCA-bottom, random (×R), probe+SAE} and each rank
`r ∈ {1,2,4,8,16,32,64,128,256,512}` (clip to ≤ D):
- Build `Q = pool[:r]` → `[r, D]`.
- For each puzzle: `final_out, cache = abl.run_with_directional_ablation(batch, direction=zeros(D),
  ablate_level="H", ablate_steps=None, max_steps=16, direction_matrix=Q)`; compute
  `Δacc_r = value(cache[last].preds) − value_baseline`.
- Aggregate `Δacc_r` over puzzles with `bootstrap_ci`.

Record per ordering: the curve `{r: ci}`, and the **minimal causal rank** `r*(frac)` = smallest
`r` with `mean(Δacc_r) ≤ frac · Δfull` for `frac ∈ {0.5, 0.9}` (more negative = more damage; use
`≤` on signed deltas). **Interpretation:** PCA-top reaching `Δfull` at small `r*` and well below
the random curve ⇒ low-rank causal code; random curve ≈ linear in `r` ⇒ redundant/distributed;
a sharp knee ⇒ localized.

### 2.4 Alignment analysis (causal basis vs readable basis)
Let `Qc = PCA-top[:r*]` (the causal subspace at `frac=0.9`). For each readable basis `B` (probe
matrix; SAE-top matrix), compute **projection energy** `‖QcᵀB‖_F² / ‖B‖_F²` ∈ [0,1] (fraction of
the readable directions' variance lying inside the causal subspace) and the **principal angles**
(`torch.linalg.svd(Qcᵀ B̂)` singular values = cos of principal angles). Low projection energy ⇒
"the model uses a basis rotated away from what probes read" (the H2 headline). Report a random-`B`
control (Gaussian directions) for calibration.

### 2.5 Optional: gradient-attribution ordering (stretch, mark clearly optional)
A 4th ordering: top singular directions of `∂loss/∂z_H` at a step (loss = CE on `labels`). Needs a
grad-enabled single-step forward (replicate §1.3 tail **with grad**, on one step, `z_H.requires_
grad_(True)`). If autograd-through-`H_level` is fiddly, **skip** — the PCA/random/probe orderings
already deliver the result. Do not block H2 on this.

### 2.6 Task value (identical to the policy plan — reuse)
`from scripts.analysis.policy_improvement import task_value` if exported; else replicate:
Maze → `maze_prediction_metrics(preds_row, label_row, input_row)["valid_sg_path"]` (NOT token_acc);
Sudoku → `compute_metrics(preds[1,seq], labels[1,seq])["accuracy"]`. Always mask `labels==-100`.
`from scripts.maze.maze_common import MAZE_CHECKPOINT, maze_prediction_metrics`;
`from scripts.core.activation_patching import compute_metrics`.

### 2.7 CLI (`causal_subspace.py`)
`--task {sudoku,maze}` (default sudoku), `--checkpoint` (default per task: Sudoku
`find_checkpoint()`, Maze `MAZE_CHECKPOINT`), `--num_puzzles 300`, `--n_pca 200`,
`--ranks 1,2,4,8,16,32,64,128,256,512`, `--n_random_bases 5`, `--probe_weights
results/probes/e8_constraint_probes/probe_weights.pt`, `--sae_path
results/sae_study/sae_d2048_l10.01.pt` (optional; skip alignment-SAE if absent), `--z_level H`,
`--max_steps 16`, `--device cuda`, `--output_dir`, `--seed 42`.

### 2.8 Acceptance (H2)
- `Δfull` on Sudoku ≈ −0.19…−0.20 (else investigate before trusting curves).
- `r*(0.9)` is reported for PCA-top and random; PCA-top `r*` < random `r*` is the expected sign.
- Projection energy of probe basis into `Qc` is reported with a random-`B` control.
- Maze replication runs (value = valid_sg_path) but is secondary.

---

## 3. Experiment H1 — Localizability vs training regime (BPTT vs 1-step)

**Question:** is the distributed code a consequence of truncated-gradient training? Build a
per-checkpoint **localizability scorecard** and compare across training regimes.

### 3.1 Files to create
1. `scripts/analysis/localizability_scorecard.py` — self-contained, `--checkpoint`-parameterized
   HRM scorecard (this is the deliverable that runs the moment HRM-BPTT lands).
2. `scripts/analysis/baseline_localizability.py` — single-state (`z`) scorecard for the Universal
   Transformer pair (Track B, available now). Mirrors (1) but for `UniversalTransformerModel`.
3. `scripts/analysis/plot_localizability.py` — bar/curve comparison across checkpoints.

Outputs: `results/localizability/<tag>/scorecard.json` where `<tag>` ∈
{`hrm_1step`, `hrm_bptt`, `ut_1step`, `ut_bptt`}; comparison fig → `results/reports/
localizability_figures/`.

### 3.2 The scorecard (per checkpoint) — metrics, in increasing tooling cost
Compute on **the checkpoint's own activations** (never cross-checkpoint probes/SAE):
1. **`probe_decodability`** — train constraint probes on this model's `z_H` and report mean val
   accuracy (the readout strength). Reuse `derive_per_cell_labels` + `train_binary` from `e8`;
   collect activations via `ActivationAblator.run_and_cache_activations` then `cache[s].z_H_out[0,
   -81:]` (Sudoku). Steps `{0,4,8,12,15}`, 5-seed puzzle-disjoint split (reuse
   `puzzle_disjoint_split` from `e8`).
2. **`probe_causal_gap`** — mean `Δacc` from ablating each trained probe direction **minus** the
   mean `Δacc` from random unit directions (same count). Single-direction ablation uses
   `DirectionalAblator.run_with_directional_ablation(batch, direction=w, ablate_level="H",
   ablate_steps=None, max_steps=16, direction_matrix=None)` — note the method is
   `run_with_directional_ablation` (with the "-al-"); the differently-named
   `run_with_direction_ablation` belongs to `SAEFeatureAblator`, not `DirectionalAblator`.
   `probe_causal_gap ≈ 0` ⇒ readable ≠ causal (current HRM); clearly negative ⇒ readable *is*
   causal.
3. **`min_causal_rank`** `r*(0.9)` and **`subspace_linearity`** — call H2's
   `find_causal_subspace(model, ...)` (import from `causal_subspace.py`). Smaller `r*` and more
   knee-shaped (less linear) curve ⇒ more localized. **This is the portable core metric** (no
   probes/SAE needed).
4. **(optional) `sae_top_vs_random_gap`** — only if an SAE has been trained on *this* checkpoint;
   otherwise emit `null`. Do not retrain an SAE inside the scorecard by default.

Each metric stored with CI and the inputs used. Add `meta`: checkpoint path, `one_step_grad` /
training regime (read from config if present), arch, `n_puzzles`, seeds.

### 3.3 H1 hypothesis / expected scorecard deltas (BPTT vs 1-step)
`probe_causal_gap` more negative (probe dirs become causal), `min_causal_rank` smaller,
`subspace_linearity` lower (knee appears), under **BPTT**. `probe_decodability` may stay high in
both (decodability ≠ causality is the existing finding). Report the signed deltas with CIs.

### 3.4 Two tracks
- **Track A (primary, airtight, HRM):** run scorecard on `hrm_1step` = current
  `checkpoints/sapientinc-sudoku-extreme` **now** (establishes the baseline numbers and exercises
  the whole pipeline), and on `hrm_bptt` = the awaited full-BPTT HRM checkpoint when it lands
  (just pass `--checkpoint <path>`). The comparison is then one `plot_localizability.py` call.
  **Wire `--checkpoint` so nothing else changes when the BPTT path arrives.**
- **Track B (available now, UT pair):** run `baseline_localizability.py` on `ut_1step`
  (`checkpoints/baselines/universal_transformer`, latest `step_*`) vs `ut_bptt`
  (`checkpoints/baselines/universal_transformer_standalone`). This needs the single-state harness
  (§3.5). Record the §1.6 caveat in `_meta`.

### 3.5 Single-state harness for the UT (Track B)
The existing ablators read `inner_carry.z_H`/`.z_L`; the UT has only `z`. Implement a small
`UTActivationHarness` in `baseline_localizability.py` (do **not** edit the model). It must:
- Load the UT via a local loader mirroring `controlled_common.load_model_and_dataloader` but
  accepting `UniversalTransformerModel` (use `utils.functions.load_model_class(config.arch.name)`
  and `pretrain.PretrainConfig`; reuse the `_orig_mod` prefix handling verbatim).
- Roll out the ACT loop calling `model.inner(...)` per step (signature returns
  `(new_carry, logits, (q_halt, q_continue))`, `universal_transformer.py:215`), caching `z`
  (= `carry.z` in) and `z_out` (= returned `new_carry.z`) and `logits` per step — the UT analogue
  of `run_and_cache_activations`.
- Provide `run_with_subspace_ablation(batch, Q[K,D])` that projects `z' = z - QQᵀz` on the carried
  `z` **before** each `model.inner` call (mirror the math in `e9_directed_ablation.py:201-216` but
  on the single `z`). With `one_step_grad=False, halt_max_steps=1`, the rollout is a single ACT
  step internally unrolling `num_iterations` — that's fine; cache the one step.
Then compute the same scorecard metrics on `z` (probes on `z_out[:, -81:]`, subspace via PCA of
`z`). `probe_decodability`, `probe_causal_gap`, `min_causal_rank`, `subspace_linearity` are all
defined identically on `z`.

### 3.6 CLI / acceptance (H1)
- `localizability_scorecard.py`: `--checkpoint` (required for non-default), `--task sudoku`,
  `--num_puzzles 300`, `--probe_steps 0,4,8,12,15`, `--seeds 0,1,2,3,4`, `--ranks ...`,
  `--device cuda`, `--output_dir results/localizability/<tag>`, `--tag <tag>`.
- Acceptance: `hrm_1step` reproduces existing numbers — `probe_decodability` ≈ 0.88 (violation
  probes), `probe_causal_gap` ≈ 0 (within CI of random), `Δfull` ≈ −0.20. UT scorecards finish and
  emit all four metrics with CIs. The plot shows the 1-step vs BPTT deltas with CIs.

---

## 4. Experiment H4 — Where the hierarchy earns its keep (H-update vs L-cycles)

**Question (B):** within each ACT step, does the **H-update** or the **L-cycles** perform the
policy improvement? **Question (flat):** does the equally-accurate Universal Transformer have a
comparably separable two-timescale structure, or is HRM's separability the architectural payoff?

### 4.1 Files to create
1. `scripts/analysis/policy_decomposition.py` — HRM per-step H-vs-L advantage decomposition.
2. `scripts/analysis/two_timescale_baseline.py` — emergent-timescale analysis on the UT.
3. `scripts/analysis/plot_policy_decomposition.py` — figures.

Outputs: Sudoku → `results/controlled/policy_decomposition/`, Maze → `results/maze/
policy_decomposition/`; UT → `results/baseline_comparison/two_timescale/`; figs →
`results/reports/policy_decomposition_figures/`.

### 4.2 Per-step H/L decomposition (the core causal result) — reuse the cache, replicate §1.3
For each puzzle, run `ablator.run_and_cache_activations(batch, cache, max_steps=16)`. For each
step `s`, you have `z_H_in = cache[s].z_H`, `z_L_in = cache[s].z_L`,
`z_H_out = cache[s].z_H_out`, and `logits_full = cache[s].logits`. Precompute
`input_embeddings = model.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])`
and `cos_sin` once per puzzle (§1.3). Build three within-step readouts (all under `torch.no_grad`,
operate on `[1, pel+seq, D]`, strip `[:, pel:]` only at `lm_head`):
- **π_in** = `softmax(lm_head(z_H_in)[:, pel:])` — policy entering the step.
- **π_full** = `softmax(logits_full)` — true policy after the full step (sanity: equals the
  recomputed full rollout of §1.3 from `z_H_in,z_L_in`; assert max-abs logit diff < 1e-2).
- **π_Lfrozen** = recompute the step with **`z_L` held at `z_L_in`** (do the two `H_level` updates
  only, never call `L_level`): `zH = H_level(z_H_in, z_L_in, **seq_info)` then
  `zH = H_level(zH, z_L_in, **seq_info)`; `π_Lfrozen = softmax(lm_head(zH)[:, pel:])`. This is the
  counterfactual "L-module contributes nothing new this step."

Per valid cell (mask `labels!=-100`), with `y` the solution token, define log-likelihood of the
solution `logp(π) = log π[y]`, and decompose the step's policy improvement:
```
Δtotal_s = mean_valid( logp(π_full) - logp(π_in) )      # total per-step improvement
ΔH_s     = mean_valid( logp(π_Lfrozen) - logp(π_in) )   # attributable to the H-update (stale L)
ΔL_s     = mean_valid( logp(π_full) - logp(π_Lfrozen) ) # extra from live L-cycles
# identity check: ΔH_s + ΔL_s ≈ Δtotal_s  (exact by construction; assert |resid|<1e-4)
```
Also record the **fraction of the improvement carried by H**: `frac_H_s = ΔH_s / (ΔH_s+ΔL_s)`
when `|Δtotal_s|>eps`. Aggregate per step with `bootstrap_ci` over puzzles, split solved/failed
(final-step exact), for both tasks.

**Expected (the HRM-specific claim):** the policy improvement is dominated by the **H-update**
(`frac_H` near 1) — the readout state — while the L-cycles set up the proposal `z_L` that the
H-update consumes. Quantifying this *is* the hierarchy's mechanistic payoff. (Maze, being a
1-step solver, should show one big `ΔH` at step ~1; Sudoku spreads ΔH across steps — ties to A/D.)

### 4.3 Optional fine-grained within-step trajectory (stretch)
Instrument the full §1.3 rollout to emit `lm_head` after **each** `H_level` update (2 per step) →
a micro-policy `logp_true` curve at H-update resolution. Nice for a figure; not required for the
headline. Mark optional.

### 4.4 Flat-model comparison (`two_timescale_baseline.py`)
Goal: show the UT lacks a separable two-timescale handle. Using the §3.5 `UTActivationHarness` on
`checkpoints/baselines/universal_transformer` (the 1-step UT, matched recurrence to HRM):
- Collect per-iteration states `z^(i)` (`i=0..num_iterations`) by instrumenting `_apply_shared_
  block` calls within one ACT step (replicate `universal_transformer.py:226-241`, capturing `z`
  after each block application).
- **Timescale separability metrics** (compare to HRM's `z_H` vs `z_L`): (i) per-iteration change
  rate `‖z^(i+1)-z^(i)‖ / ‖z^(i)‖` — HRM has two distinct rates (slow `z_H`, fast `z_L`); test
  whether the UT shows one rate or a split. (ii) Project `z^(i)` onto PC1–2 of the trajectory and
  measure whether updates decompose into two roughly-orthogonal velocity components (slow drift +
  fast oscillation) vs a single component. (iii) Optionally, can a linear readout separate
  "slow-subspace" vs "fast-subspace" of `z` the way `z_H`/`z_L` are separated in HRM?
- **Deliverable framing:** a quantitative statement — "HRM exposes two ablation-separable
  timescales; the UT's single `z` does not decompose into comparable slow/fast subspaces" — or, if
  the UT *does* show an emergent split, the equally-interesting "recurrence learns the timescale
  split HRM hard-codes." Either is a positive result; report whichever the data shows.

This part is higher-variance; implement after §4.2 lands. Keep it modular.

### 4.5 CLI / acceptance (H4)
- `policy_decomposition.py`: `--task {sudoku,maze}`, `--checkpoint` (default per task),
  `--num_puzzles 500`, `--max_steps 16`, `--device cuda`, `--output_dir`, `--eps 0.01`.
- Acceptance: the identity `ΔH_s+ΔL_s≈Δtotal_s` holds (assert); `π_full` matches recomputed full
  rollout (assert <1e-2); final-step value ≈ baselines (Sudoku 0.80, Maze 0.93); `frac_H`
  reported per step with CIs. UT timescale metrics finish and produce the comparison figure.

---

## 5. Reusable helpers (import, don't duplicate)
- `scripts.analysis.policy_improvement`: `task_value`, and its aggregation patterns (look at the
  file before writing H4; reuse `per_step` → `bootstrap_ci` aggregation and the solved/failed
  split). 486 lines — read it first.
- `scripts.controlled.controlled_common`: `load_model_and_dataloader`, `collect_puzzles`,
  `bootstrap_ci`, `find_checkpoint`, `extract_batch`.
- `scripts.core.activation_ablation`: `ActivationAblator`, `ActivationCache`.
- `scripts.directed_ablation.e9_directed_ablation`: `DirectionalAblator`, `select_best_directions`.
- `scripts.probes.e8_constraint_probes`: `derive_per_cell_labels`, `train_binary`, `LinearProbe`,
  `puzzle_disjoint_split`, `CONSTRAINT_DIRECTIONS`/`BINARY_TARGETS`.
- `scripts.sae.sae_causal_ablation`: `SAEFeatureAblator`, `select_top_features`.
- `scripts.maze.maze_common`: `MAZE_CHECKPOINT`, `maze_prediction_metrics`.
- `scripts.core.activation_patching`: `compute_metrics`.
- `scripts.core.provenance`: `write_meta(output_dir, experiment, params, repo_root=REPO_ROOT)`.
- `scripts.core.sudoku_sample`: `collect_indexed_batches`, `load_puzzle_indices`,
  `save_puzzle_indices` (for shared puzzle-index manifests / cross-experiment comparability).

Every new script starts with the standard header:
```python
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
```

---

## 6. SLURM orchestration (`scripts/analysis/slurm_h1_h2_h4.sbatch`)
Mirror `scripts/analysis/slurm_policy_improvement.sbatch` (header `-N 1 -n 8 --mem=64G
-t 06:00:00 -o logs/%x_%j.out -e logs/%x_%j.err`, `#SBATCH -J hrm_h124`,
`set -euo pipefail`, `cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"`,
`export PYTHONPATH="$PWD:${PYTHONPATH:-}"`, `conda run --no-capture-output -n hrm python -u ...`).
Run order (respecting the dependency):
1. H2: `causal_subspace.py --task sudoku` (writes `subspace_curve.json` consumed by H1).
2. H1: `localizability_scorecard.py --tag hrm_1step` (current HRM); then
   `baseline_localizability.py --tag ut_1step` and `--tag ut_bptt`. **Add an `if [ -n "$HRM_BPTT_CKPT" ]`
   guard** that also runs `--checkpoint "$HRM_BPTT_CKPT" --tag hrm_bptt` when that env var is set
   (so the BPTT run is one env-var away).
3. H4: `policy_decomposition.py --task sudoku` then `--task maze`; then `two_timescale_baseline.py`.
4. Plots: the three `plot_*.py`.
Provide an `N="${N:-300}"` override and an `HRM_BPTT_CKPT="${HRM_BPTT_CKPT:-}"` hook.
Submit: `sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_h1_h2_h4.sbatch`.

---

## 7. Validation, smoke tests, acceptance
1. **Import/arg check (login node, CPU):** `conda run -n hrm python
   scripts/analysis/causal_subspace.py --help` (redirect with `>|`; the profile sets
   `noclobber`). Same for each new script.
2. **CPU smoke (tiny):** each compute script with `--task sudoku --num_puzzles 1 --device cpu
   --max_steps 2 --output_dir /tmp/<exp>_smoke` (for H2 also `--ranks 1,2`). Must produce a valid
   JSON with the documented keys. If it exceeds ~4 min, kill it — imports + first write verified is
   enough; submit the GPU job for real numbers.
3. **GPU job:** confirm every JSON + figure is written and logs end with a "done" line.
4. **Numeric sanity (must hold or stop and report):**
   - H2: `Δfull(z_H zero)` ≈ −0.19…−0.20 on Sudoku; subspace deltas are monotone-ish in `r`;
     `Δacc` at `r=D` ≈ `Δfull` (projecting out the full space ≈ zeroing, up to the mean component).
   - H1 `hrm_1step`: `probe_decodability` ≈ 0.88 (violations); `probe_causal_gap` ≈ 0 within CI.
   - H4: `ΔH+ΔL ≈ Δtotal` (assert); `π_full` ≈ recomputed rollout (assert <1e-2); final value ≈
     0.80 Sudoku / 0.93 Maze.
5. **Cross-checks to write into the results READMEs:**
   - H1: state the four scorecard metrics for every checkpoint tag with CIs, and the BPTT−1step
     (and UT BPTT−1step) deltas; conclude for/against "training regime causes distributedness."
   - H2: state `r*(0.5)`, `r*(0.9)` and probe-basis projection energy; one sentence tying the
     low projection energy to Finding 3's "readout ≠ causation."
   - H4: state per-step `frac_H` (H-update share of the improvement) for both tasks, and the UT
     timescale-separability verdict.

---

## 8. Gotchas / accuracy caveats (read before coding)
- **`logits`/`preds` are already puzzle-emb-stripped** (`lm_head(z_H)[:, pel:]`). Only `z_H`/`z_L`
  carry the `pel`-length prefix; strip `[:, pel:]` **only** when you call `lm_head` yourself
  (H2 PCA reads `z_H_out[:, pel:]`; H4 readouts strip after `lm_head`). `pel=1` here but read it
  from `model.inner.puzzle_emb_len` — do not hardcode.
- **Cast bf16 → float32** before any softmax/SVD/PCA.
- **Mask `labels == -100`** for every per-cell mean/flag.
- **Maze value = `valid_sg_path`** (not token_acc); maze metrics need the `inputs` row; Sudoku
  doesn't. The Sudoku-specific `[:, -81:]` slicing in `e9`/`sae` does **not** apply to maze —
  for maze use the full sequence and the `labels!=-100` mask.
- **`direction_matrix` is `[K, D]`** (rows = directions); the engine QR-orthonormalizes `.T`
  internally — pass raw rows, don't pre-transpose.
- **Probes/SAE are checkpoint-specific** — never run `e8`/`e9`/`sae_causal_ablation`'s hardcoded
  loaders on a non-Sudoku-extreme checkpoint; H1's scorecard is self-contained and takes
  `--checkpoint`.
- **UT has no `z_H`/`z_L`** — Track B and the flat comparison need the §3.5 single-state harness;
  do not feed a UT into `ActivationAblator`/`DirectionalAblator` (they will `AttributeError` on
  `.z_H`).
- **UT checkpoint files** are per-step (`step_26040`, …) plus `*_all_preds.*` dumps — load the
  largest plain `step_*` state-dict, not a preds file.
- **2.6 GB activation bank** — subsample before SVD; never SVD the full `[N,steps,81,512]`.
- **One forward pass per puzzle** serves H4 (read three readouts from one cache + cheap H_level
  recomputes); H2 needs one `run_with_directional_ablation` per (ordering, rank, puzzle) — keep
  `--num_puzzles` modest (300) and ranks log-spaced.
- **Eval-only. Do not retrain anything** (the HRM-BPTT checkpoint is produced elsewhere; this plan
  only consumes it). The UT BPTT checkpoint already exists.
- **Determinism:** use fixed `--num_puzzles` (deterministic loader order) or a shared
  `--puzzle_indices` manifest (`scripts.core.sudoku_sample`) if you want cell-for-cell
  comparability with A/D and the probe/SAE runs.

---

## 9. What to hand back
- The new scripts: `causal_subspace.py`, `plot_causal_subspace.py`,
  `localizability_scorecard.py`, `baseline_localizability.py`, `plot_localizability.py`,
  `policy_decomposition.py`, `two_timescale_baseline.py`, `plot_policy_decomposition.py`,
  `slurm_h1_h2_h4.sbatch`.
- The JSON outputs + figures listed per experiment.
- Three short results READMEs under `results/reports/`:
  `localizability_README.md`, `causal_subspace_README.md`, `policy_decomposition_README.md`,
  each with the numbers + CIs and the cross-check paragraph from §7.5.
- A note recording the §1.6 UT caveat and a placeholder line for the HRM-BPTT result
  ("run `HRM_BPTT_CKPT=<path> sbatch ...` when the checkpoint lands").
