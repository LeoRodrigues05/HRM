# Plan — HRM Mechanistic Interpretability → Conference (AAAI track)

**Status:** draft for refinement
**Hard deadline:** 2026-07-10 · **Target submit:** 2026-07-06 (keeps a 4-day buffer)
**Today:** 2026-06-06
**Compute:** 1× RTX 5000 Ada (32 GB), single GPU, via SLURM. Login node `lo-01` has **no usable GPU** (old driver, no `nvidia-smi`) — all training/benchmarks must run as SLURM jobs on a compute node.
**Current paper:** `docs/paper/` — ICML *workshop* draft, Sudoku-only, 3 findings. Must be reworked to a multi-domain, statistically-hardened conference paper.

> **Guiding principle for this revision:** before adding *new* breadth (Maze/ARC) or the *bold* training-regime experiment, we first **re-run and harden the existing Sudoku results** so every number in the paper has a script, a seed, and a confidence interval. This is **Phase 0** below and is the prerequisite "extra term" before jumping in.

---

## Phase 0 — Re-run & harden the existing Sudoku results (the prerequisite term)

### 0.1 Audit: what is solid vs. what must be re-run

| Result (paper ref) | Source on disk | N | CIs / stats present? | Verdict |
|---|---|---|---|---|
| `zH`/`zL` per-step ablation (F1, `tab` + `fig:causal_evidence`) | `results/controlled/ablation/{zH,zL}/aggregate.json` | 5000 | **Yes** (bootstrap CI) | **Solid** — surface CIs in figure |
| Freeze-after-k (F1, `tab:freeze`) | `results/controlled/freeze/aggregate.json` | 1000 | **Yes** (CI) | **Solid** — table omits CIs; add them |
| Time-shift (F1, §timeshift) | `results/controlled/time_shift/aggregate.json` | 500 | Yes (aggregate) | **Mostly solid** — verify + surface CI |
| Baseline 5-model recurrence (F1, `tab:main_results`) | `results/baseline_comparison/`, `results/metrics/hamming.csv` | 500 | std only | **Re-run** with bootstrap CI per step |
| **Linear probes** (F2, `tab:probes`) | `results/probes/e8_constraint_probes/sweep_results.csv` | single split (32400/8100) | **No CIs, no seeds, single split** | **RE-RUN** |
| **Probe geometry / cosines / PCA** (F2, `tab:cosines`, "0.6% of 512-d", "3-dim", "57% var") | `results/probes/e8_constraint_probes/geometric_analysis.json` | single fit | **No CIs / no seeds** | **RE-RUN** (seed ensemble) |
| Cross-puzzle patching (F2, −53.1 / −61.7) | `results/patching/` | 500 | point estimates | **Re-run** with CI |
| **Directed ablation, linear** (F3, `tab:directed_ablation`) | `results/directed_ablation/e9_directed_ablation/aggregate_results.json` | **200** | **No p-value, no Cohen's d on disk** | **RE-RUN** (see 0.2) |
| Directed ablation, nonlinear (F3, app) | `results/directed_ablation/e9b_nonlinear/statistical_tests.json` | — | Yes (p, d) | OK as control |
| **SAE causal ablation** (F3, `tab:sae_causal`) | `results/sae_study/causal_ablation/aggregate.json` | 300 | t-test only; std ≈16% | **Re-run** + **sweep** (0.3) |
| **SAE feature dictionary** (F3, `tab:sae_top_features`) | `results/sae_study/sae_d2048_l10.01_*` | single config | **Only d=2048, λ=0.01 exists** | **SWEEP** required |

### 0.2 Specific "made-up / unverifiable" metrics to regenerate

1. **`tab:directed_ablation` (F3) N mismatch + phantom statistics.**
   - Paper text and caption say **"500 sample puzzles"**; the only artifact on disk is `n_puzzles: 200`.
   - The table reports **p-values** (0.408, 0.004, 0.365, 0.209) and the text cites **Cohen's d = −0.13**. Neither appears anywhere in `e9_directed_ablation/` (only the *nonlinear* `e9b` has saved `p_value`/`cohens_d`). **Provenance is missing** → treat these numbers as unverified.
   - **Action:** re-run E9 (linear directed ablation) at **N = 500**, and have the script persist, per direction: `mean_delta`, bootstrap 95% CI, **paired Wilcoxon p**, **Cohen's d**, plus the matched random-control distribution. Overwrite the table from the JSON.

2. **Linear probe accuracies (`tab:probes`).** Single train/val split, one seed, no error bars. **Action:** retrain with **≥5 seeds** (or k-fold), report mean ± 95% CI per (step, target). Use a **puzzle-disjoint** train/val split (current split may leak cells from the same puzzle across train/val).

3. **Probe geometry numbers** ("3-dimensional subspace", "PC1 ≈ 77%", "0.6% of 512-d", cosine table). Single fit. **Action:** recompute across the seed ensemble; report CI on cosines and on explained-variance.

4. **"60% reduction in violations (73.6 → 29.8)" / Hamming curve.** `results/metrics/hamming.csv` has mean+std but the violation reduction is a single run. **Action:** attach bootstrap CI; confirm the 73.6 and 29.8 endpoints from a saved artifact.

5. **PCA trajectory variance (57%) and directional metrics** (F1, `fig:zh_trajectories`). Single run. **Action:** report over seeds / multiple puzzle samples with CI.

### 0.3 SAE: close the single-config hole (F3's weakest point)
- A reviewer's first objection to "SAEs don't localize" is "you used one bad SAE."
- **Action:** sweep dictionary size **d ∈ {1024, 2048, 4096, 8192}** × L1 **λ ∈ {0.003, 0.01, 0.03}** on `zH` (`scripts/sae/sae_sweep.py`). Report the reconstruction-vs-sparsity frontier and **re-run the top-50 vs random-50 causal ablation per config** with bootstrap CIs. Claim only holds if "distributed" survives the whole sweep.

### 0.4 Statistical-rigor standard (apply to **every** number that lands in a table/figure)
- **95% bootstrap CIs** everywhere (reuse `bootstrap_ci` from `scripts/controlled/controlled_common.py`).
- **Effect size (Cohen's d)** and **paired Wilcoxon** for every "X vs control" comparison.
- **≥3 seeds** for anything involving training or probe fitting; **≥5** for probes (cheap).
- **One shared, seeded puzzle set** reused across E1/E2/E5/E8/E9/E10 so numbers are comparable cell-by-cell (per `docs/FUTURE_WORK.md` §E). Persist it once; load everywhere.
- Write a `_meta.json` next to every results dir (git SHA, seed, N, config, GPU).

> Phase 0 is **eval-only** (forward passes + probe fits). It is cheap (hours, not days) and can run on the single GPU interleaved with Maze. **Do this first.**

---

## Compute reality & training-time estimates

### Measured anchors (from `wandb/`, baseline trainings on the cluster)
| Run | Model | Steps | Wall-clock | s/step |
|---|---|---|---|---|
| `8aqsf25p` | Plain Transformer (Sudoku, seq 81, single-pass) | 52,080 | 27,356 s (7.6 h) | **0.525** |
| `owi05dke` | Standard RNN (Sudoku) | 52,080 | 13,323 s (3.7 h) | **0.256** |

(GPU count for these runs was not recorded in the wandb metadata — treat as single-GPU-representative but confirm with the benchmark below.)

### Cost model for HRM
- **One-step gradient (current/"baseline" HRM):** per optimizer step (deep supervision = 1 segment) the model does `H_cycles×L_cycles = 4` inner block-applies under `torch.no_grad()` + **1** grad L-update + **1** grad H-update; backward flows through only that last segment (`models/hrm/hrm_act_v1.py` L188–221). Activation memory ≈ a couple of block-applies → cheap. Per-step cost ≈ a single-pass transformer of the same width.
- **Full BPTT:** unroll **all 16 segments × 4 inner = 64** block-applies **with gradient**; store activations for all of them. Backward dominates.
  - **Compute:** ≈ **5–12× slower per step** than one-step grad.
  - **Memory:** ≈ **16–32× activation memory**. On 32 GB this forces *at least one of*: gradient checkpointing (≈1.3–2× compute, bounds memory), batch-size cut (768 → 64–128, which adds proportional steps for the same epoch budget), or **truncated BPTT** through the last K segments (K∈{4,8}) — the recommended middle ground.

### Estimates (single RTX 5000 Ada, one seed, same epoch budget ≈ 52k steps)
| Config | Task | Est. s/step | Est. wall-clock / seed | Notes |
|---|---|---|---|---|
| HRM one-step grad (**baseline retrain**) | Sudoku (seq 81) | 0.5–0.8 | **~8–12 h** | cheap; 3 seeds ≈ 1.5–2 days |
| HRM one-step grad | Maze (seq 900) | ~2–4 | **~1.5–3 days** | attention O(L²) dominates |
| HRM **truncated BPTT** K=4 | Sudoku | ~2–4 | **~1.5–3 days** | grad-checkpoint, bs ~256 |
| HRM **full BPTT** K=16 | Sudoku | ~3–6 | **~2.5–5 days** | grad-checkpoint + bs cut |
| HRM **full BPTT** K=16 | Maze | heavy | **~5–10 days** | memory wall at seq 900 — **likely infeasible by 07-06** |

### Recommendation (given 1 GPU + deadline)
- **Do the BPTT experiment on Sudoku, not Maze.** Sudoku (seq 81) is ~1–2 orders of magnitude cheaper per step; the mechanism claim ("more gradient horizon → more localizable representations") does not require the longer-horizon Maze.
- Prefer **truncated BPTT K=4 and K=8** vs **one-step grad** (3 points on the "gradient horizon" axis) over a single all-or-nothing full-BPTT run. Cheaper, multi-seed-able, and a *cleaner* scientific story (monotonic trend).
- Spend the Maze GPU budget on the **eval-only MI parity suite** (`scripts/maze/*`, hours each) + the **no-halt convergence control**, not on Maze training.
- **Benchmark before committing.** Replace the table above with measured `s/step` + peak memory by running ~200 steps of each config on the compute node:
  ```bash
  # one-step grad (baseline) — measure s/step + peak mem
  OMP_NUM_THREADS=8 python pretrain.py \
      data_path=data/sudoku-extreme-1k-aug-1000 \
      epochs=50 eval_interval=1000 global_batch_size=384 \
      +run_name=bench_onestep
  # truncated BPTT K=4 (after the grad-horizon flag is added to the model)
  #   arch.bptt_segments=4  global_batch_size=256  +grad_checkpoint=true
  ```
  Watch `nvidia-smi` peak memory; if OOM, halve `global_batch_size` and enable checkpointing.

---

## Headline new experiment — does the one-step gradient cause distributed representations?

**Hypothesis (from `07discussion.tex`):** HRM's representations are distributed (F3) *because* the one-step gradient approximation provides no per-feature sparsity pressure. More gradient horizon → more localizable (probe/SAE-ablatable) representations.

**Design.**
- **Independent variable:** gradient horizon K ∈ {1 (current), 4, 8, (16 if compute allows)} via truncated BPTT.
- **Controls:** identical data, seed set, optimizer, epoch budget; only K changes. Match final task accuracy as closely as possible (report it — if BPTT trains better/worse, that itself is a finding).
- **Dependent variables (the MI suite, re-run per K):** (a) directed-ablation effect of probe directions vs random (does it become *non*-random?); (b) SAE top-50 vs random-50 causal gap; (c) probe-weight subspace dimensionality.
- **Decision gate (end of Phase 2):** if K=4/8 already show no localization shift, **stop** — report the negative trend (still publishable: "distribution is not just a truncation artifact"). If they shift, push to K=16 on Sudoku only.
- **Seeds:** ≥2 per K (≥3 if time). **Task:** Sudoku.

---

## Reworked schedule (Phase 0 prepended; submit 07-06)

| Week | Dates | Focus | Exit criteria |
|---|---|---|---|
| **W0/W1** | Jun 6–12 | **Phase 0 harden** (probes, E9, SAE sweep kickoff, shared seed set, CI utils). **Benchmark** one-step vs K=4 on the compute node. **Launch** Sudoku one-step + K=4 BPTT trainings (longest pole). | Every F2/F3 number re-generated with CI + saved provenance; measured s/step & peak mem; BPTT trainings running. |
| **W2** | Jun 13–19 | **Maze parity** (eval-only MI suite at N≥500 + CIs) + **no-halt control**. Finish SAE sweep (Sudoku). Continue/append BPTT K=8. | Maze F1/F2/F3 at N≥500 w/ CIs; SAE sweep table; maze convergence claim controlled. |
| **W3** | Jun 20–26 | **Mechanism analysis** (run MI suite on each trained K-model). **Freeze all experiments after this week.** Begin rewrite (method + F1). | Grad-horizon result decided (in or cut); all experimental numbers final. |
| **W4** | Jun 27–Jul 3 | **Write + figures.** Rework all `docs/paper/` sections to multi-domain; regenerate every figure with CIs; new cross-domain + SAE-sweep + grad-horizon panels. `_meta.json` everywhere. Internal review (claim→figure→script 1:1). | Complete draft; every claim traceable to a script + CI. |
| **W5 (buffer)** | Jul 4–10 | **Polish + submit.** Address review, appendix tables, anonymized code release, arXiv build. **Submit 07-06**, hard stop 07-10. | Camera-ready PDF + supplementary + code link. |

---

## Paper rework checklist (`docs/paper/`)
- Switch off the ICML *workshop* style → target conference main-track style/length; promote appendix method content into the body.
- `00abs`/`01intro`: re-pitch as **multi-domain** (Sudoku + Maze, ARC if it lands). Add grad-horizon finding as a 4th contribution if it survives the decision gate.
- `03prelim`: add Maze (and ARC) task/metric definitions alongside Sudoku.
- `04finding1`: add Maze columns; add the **fast-convergence / task-dependent depth** subsection **with the no-halt control**.
- `05finding2`: probe tables with **CIs + seeds**; add Maze probes and `zL` probes; fix the geometry numbers.
- `06finding3`: replace single-SAE result with the **sweep**; fix `tab:directed_ablation` (N=500, real p/d/CI); add the **grad-horizon** result as the mechanistic cause.
- `07discussion`: move the "why distributed" paragraph from *hypothesis* to *result*; drop the single-task limitation.

---

## Risks & cut list (cut top-down if behind)
1. **Full BPTT K=16 / any Maze training** → drop; keep truncated K∈{4,8} on Sudoku only.
2. **ARC breadth** → drop to a single F1 figure; Sudoku + Maze is the minimum for "generalizes".
3. **`zL` probes / baseline-model MI** → appendix-only.
4. **Grad-horizon experiment** → if benchmark shows >3 days/seed even truncated, keep "why distributed" as a hypothesis and lean on multi-domain + SAE sweep for the conference bar.

## Open questions to confirm (affect scope)
- **Exact AAAI (or AAAI-track/workshop) CfP date & page limit?** The named file is `plan_aaai.md`, but the AAAI 2026 main-track deadline has passed and AAAI 2027 abstracts are ~Aug 2026 — confirm the precise venue/deadline so W4–W5 length and the page budget are right.
- Can we get **>1 GPU** or longer queue limits for the BPTT trainings? That single fact decides whether full-BPTT and multi-seed are in scope.
- Is retraining HRM with ≥3 seeds acceptable, or is the budget 1 seed/config?
