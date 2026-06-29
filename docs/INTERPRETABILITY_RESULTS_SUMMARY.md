# HRM Interpretability Suite — Results Summary (Sudoku · Maze · ARC-AGI-2)

A short, cross-task index of every interpretability experiment we ran, with headline
numbers read from the on-disk result JSONs. Sudoku is the paper's main task; Maze is
the parity replication; ARC-AGI-2 is the third-domain extension this project added on
the **Path-A adapted checkpoint** (`checkpoints/arc2-adapted-evalonly/step_7391` —
reasoning core *frozen*, `puzzle_emb` re-fit). Unless noted, all checkpoints are the
released HRM weights and every interventional number is a 95%-CI bootstrap mean.

> **Read ARC numbers as representation-level only.** The released ARC checkpoint's
> `puzzle_emb` table (1,045,829×512) cannot be aligned to any rebuildable dataset
> (filesystem-order-dependent build, no released identifiers), so the adapted model
> recovers per-cell accuracy (~90%) but **0% exact-grid / voting accuracy**. ARC
> probe/ablation results characterize the frozen *reasoning core*; they are not a
> solving benchmark. See `docs/PLAN_ARC_path_A_embedding_adaptation.md`.

---

## 1. Coverage matrix

| Experiment family | Sudoku | Maze | ARC-AGI-2 |
|---|:--:|:--:|:--:|
| Baseline-vs-recurrence accuracy (per-step) | ✅ | ✅ | — |
| Per-step `z_H` / `z_L` ablation | ✅ | ✅ | partial¹ |
| **Freezing** (`z_H` frozen after step *k*) | ✅ | ✅ | ❌ **not run** |
| Time-shift transplant | ✅ | ✅ | ❌ not run |
| **Cross-puzzle / activation patching** | ✅ | ✅ | ❌ **not run** |
| Linear probes (decodability) | ✅ | ✅ | ✅ |
| MLP (non-linear) probes | ✅ | ✅ | ✅ |
| Probe geometry (PCA / cosine) | ✅ | ✅ | partial² |
| Directed ablation (probe dirs vs random) | ✅ | ✅ | ✅ |
| Causal subspace (low-rank ablation) | ✅ | ✅ | ✅ |
| **SAE feature analysis + causal ablation** | ✅ | ✅ | ❌ **not run** |
| Policy improvement (value-by-step) | ✅ | ✅ | ✅ |
| Policy decomposition | ✅ | ✅ | ✅ |
| Trajectory PCA / directional convergence | ✅ | ✅ | partial³ |
| Single-shot / voting accuracy diagnostics | ✅ | ✅ | ✅ |

¹ ARC has full-`z_H`/`z_L` directed ablation but no per-step zero-ablation sweep.
² ARC stored `probe_weights.pt` but no published geometry analysis.
³ ARC has the causal-subspace alignment curve (a directional-convergence proxy) but no PCA-trajectory figure.

**Bottom line on the three the question asked about:** **SAE, freezing, and
cross-puzzle patching were _not_ run for ARC.** They exist for Sudoku and Maze.

---

## 2. Sudoku (main task) — `results/controlled/`, `results/sae_study/`, `results/probes/`

- **Recurrence drives accuracy.** HRM cell 80.3%, puzzle 45.4%; non-recurrent
  baselines (Plain Transformer, GRU) stay >30 mismatched cells. Hierarchy is *not*
  what wins raw accuracy (Recurrent Transformer ties), but it yields an intervenable
  `z_H` (paper Tables 1–2, Fig 1).
- **`z_H` is causally essential, progressively.** All-steps `z_H` ablation **−18.0%**
  cell acc; per-step rises **step0 −6.7% → step15 −26.1%**. `z_L` all-steps **−19.3%**
  but no single step matters → `z_H` carries the evolving solution, `z_L` is a buffer.
- **Freezing reveals progressive refinement.** Freeze `z_H` after step *k*:
  **k0 −9.1% → k5 −1.5% → k8 −0.4% → k12 −0.2%.** Most work done in steps 5–8.
- **Cross-puzzle patching is destructive.** Patching `z_H` from a donor: steps 1–3
  **−53.1%**, steps 5–7 **−61.7%**; post-patch grid is rewritten to the donor's
  constraints → `z_H` is puzzle-specific (paper §5.1, Fig 4).
- **Linear probes decode constraints (~90%).** Cell digit 99.0%, is-given 97.3%,
  row/col/box violation 88.8–90.1%, correctness 83.5% at step 15; +~15% from step 0
  (paper Table 4). **MLP gain <1pp** → linearly accessible.
- **But readout ≠ causal.** Directed ablation of probe directions: all |Δacc| <1%,
  indistinguishable from random (e.g. per-cell-correct probe Δ +0.03%, p≈0.57).
- **SAE: computation is distributed.** SAE causal ablation — top-50 features
  **−3.6%**, random-50 **−3.4%**, probe dirs **+0.2%**, random dirs **+0.6%**; full
  `z_H` ablation (−18%) ≫ SAE (−3.6%) ≫ probe (≈0). No mono-semantic feature.
- **Trajectories converge.** Solved puzzles cos>0.99 by step 8, ‖Δz_H‖→0; failed
  oscillate (‖Δz_H‖≈1.5) — "spurious attractor" (paper Fig 3, Fig 8).

## 3. Maze (parity replication) — `results/maze/`

- **The metric artifact.** All-steps `z_H` ablation moves **token acc only −0.8%** but
  **valid start→goal path −32.5%** — token accuracy hides the damage path-validity
  reveals. `z_L` path Δ **−8.5%**.
- **Maze `z_H` is load-bearing only at readout.** Freeze-after-*k* ≈ **0% at every k**
  (k0 −0.02%, k8 +0.01%, k15 0.0) — *not* progressively refined like Sudoku.
  Cross-puzzle patching confirms: **step15 −26%**, but **step8 +2%** (replaceable
  mid-trajectory, decisive only at the final readout).
- **Probes decode geometry.** on-optimal-path 98.8% (base 64%), is-wall / is-free
  100%, is-junction 84% at step 15.
- **Slightly more probe-direction causality than Sudoku.** Directed ablation of the
  on-optimal-path direction: Δvalid-path **−2.8%** vs random +0.08%, **significant** —
  the one place a single probe direction has a real (if small) causal effect.
- **SAE distributed too.** top-50 path Δ **−3.0%**, random-50 **−3.3%**, probe dirs
  −0.2%, random +0.2%.
- **Policy value climbs.** value-by-step **0.234 → 0.93** over 16 steps.

## 4. ARC-AGI-2 (this project, frozen-core Path-A) — `results/arc/`

- **Accuracy / diagnostics** (`diagnostics/`): token 90.9%, colour-cell 90.2%, but
  **exact_solved ≈ 2%, shape_correct 0%, TTA-voting pass@K = 0%** across K∈{1,2,10,
  100,1000}. The frozen-core + orphaned-`puzzle_emb` ceiling — no exact-grid match
  possible. Convergence trend over adaptation checkpoints: token/colour plateau early,
  exact stays ~0.
- **Linear probes** (`hardened/linear_probes/`): per-cell-correct 91.6% (base 84.4%),
  **output-colour 88.6% (base 11.1%)**, input-colour 81.4%, colour-changed 97.2%,
  input/output-inside-grid ~99.8%, is-EOS 99.8% at step 15. Features are highly
  decodable from both `z_H` and `z_L`.
- **MLP probes** (`hardened/linear_probes_mlp/`): small gains on per-cell/colour
  targets (linearly accessible); large nominal MLP gains on exact_solved/shape_correct
  are degenerate (one-class ~99% baselines) and not informative.
- **Directed ablation** (`hardened/directed_ablation/`): probe directions move
  colour-cell acc by **≤0.55%** — input/output-inside-grid −0.55% and colour-changed
  −0.40% are significant; per-cell-correct −0.05% is not; random-control band ±0.06%.
  `z_L` directed ablation (`directed_ablation_zL/`) mostly n.s. → **readout ≠ causal**,
  same dissociation as Sudoku/Maze.
- **Causal subspace** (`causal_subspace/`): ablating the **top-4 PCA subspace of `z_H`
  = −9.28%** colour-cell acc (min causal rank 4), while the full *readable* subspace
  is only −0.53% and subspace-linearity ≈ 0.005 → causal mass is **low-rank and
  distributed**, not along readable probe directions.
- **Policy improvement / decomposition** (`policy_improvement/`, `policy_decomposition/`):
  value-by-step **0.824 → 0.911**, rising then plateauing by step ~2–3 (n_solved 12/500).

**ARC gaps (not run):** SAE feature analysis + causal ablation; freezing
(`z_H`-after-*k*); cross-puzzle / activation patching; time-shift transplant;
trajectory-PCA; probe-geometry. The first three are the direct analogues of the
Sudoku/Maze experiments the question asked about.

---

## 5. Cross-task takeaways

1. **Readout ≠ causally relevant** holds on all three tasks: features decode at
   80–100% yet ablating those directions barely moves task accuracy
   (Sudoku <1%, Maze −2.8% at most, ARC ≤0.55%).
2. **Computation is deeply distributed** (full state ≫ SAE ≫ probe ≈ random for
   Sudoku/Maze; rank-4 subspace ≫ probe ≈ random for ARC).
3. **Task-dependent depth.** Sudoku `z_H` is *progressively refined* (freeze damage
   decays smoothly); Maze `z_H` is *load-bearing only at the final readout* (flat
   freeze curve, patch damage only at step 15). ARC's frozen core sits closer to the
   Sudoku pattern at the representation level (early plateau by step ~2–3).
4. **Metric choice matters** (Maze token-acc −0.8% vs path −32.5%; ARC token 91% vs
   exact-grid 0%) — per-token metrics overstate competence.

## 6. Where to find the figures

- Sudoku/Maze paper figures: `docs/ICMLWorkshop2026_HRM_Interpretability_Final.pdf`;
  regenerators in `scripts/maze/plot_consolidated_figures.py`,
  `scripts/plotting/`, `scripts/analysis/plot_*`.
- ARC representation-level figures (paper-style): `results/reports/arc_paper_figures/`
  via `scripts/arc/plot_arc_paper_figures.py` (Fig-5 readout-vs-causal, Fig-10
  distributed-computation, decodability-by-step).
- ARC diagnostics figures (accuracy / voting / convergence):
  `results/reports/arc_figures_step7391/` via `scripts/arc/plot_arc_new_figures.py`.
