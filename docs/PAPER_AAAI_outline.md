# Paragraph-by-Paragraph Outline — HRM Cross-Task Mechanistic Study (AAAI-26)

**Working title:** *What Hierarchical Reasoning Models Compute: A Cross-Task Mechanistic Study*

**Reframe goal.** Promote the Sudoku-only ICML-workshop paper
(`docs/ICMLWorkshop2026_HRM_Interpretability_Final.pdf`) to a full AAAI-26 submission that keeps
the **original three-finding spine** but validates each finding across **Sudoku (main task),
Maze, and ARC-AGI-2**, with one genuinely new mechanistic insight: *the locus/schedule of `z_H`
refinement is task-dependent* (Sudoku refines progressively; Maze's `z_H` is load-bearing only at
the readout).

**Target format (AAAI-26 main technical track).** ≤ **7 pages** technical content *including all
figures and tables*, **unlimited references**, two-column AAAI style (AAAI-26 Author Kit), plus an
**optional technical appendix** that reviewers are *not obligated to read* — so every load-bearing
claim, number, and figure must sit inside the 7 pages, and everything else (per-step tables,
derivations, the ARC reproducibility deep-dive) goes to the appendix. 7 pages × two columns is
tight for 3 findings × 3 tasks → the figure budget is ~5 combined multi-task panels, and prose is
ruthlessly topic-sentence-first.

**Writing principles** (Nanda, *Highly Opinionated Advice on How to Write ML Papers*; Foerster,
*How to ML Paper*): one cohesive narrative of ≤3 concrete claims; **first sentence of every
paragraph is its topic sentence** (skimming only first sentences should reconstruct the argument);
**funnel** abstract & intro; a **"money" Figure 1** on page 1; claims calibrated to evidence
strength; red-team the weak points in Limitations; converge *qualitatively distinct* evidence
(probe + ablation + SAE) on each claim.

Legend below: each bullet = one paragraph; **bold lead** = its topic sentence; `→` = the
data/figure it draws on; **(NEW)** = depends on an ARC experiment still to be run (SAE / freezing /
cross-puzzle patching).

---

## Figure & table budget (in-text, fits the 7 pages)

| # | Figure | Panels | Source artifacts |
|---|---|---|---|
| Fig 1 | Money figure: HRM schematic + 3-task readout≠causal teaser | 2 rows | schematic + probe/ablation JSONs |
| Fig 2 | F1 — per-step `z_H` ablation + freeze-after-k, 3 tasks | a,b | `controlled/ablation`, `*/freeze*`, ARC freeze **(NEW)** |
| Fig 3 | F1 — cross-puzzle patching, 3 tasks (replaceable mid-traj, decisive at readout) | 1 | maze `patching_full_steps`, Sudoku patching, ARC **(NEW)** |
| Fig 4 | F2 — readout vs causal (probe acc bars + directed-ablation Δ w/ random band), 3 tasks | a,b | `*/linear_probes`, `*/directed_ablation`, `arc_paper_figures` |
| Fig 5 | F3 — distributed computation ladder: full-state ≫ SAE ≫ probe ≈ random, 3 tasks | 1 | `*/sae_study/causal_ablation`, `causal_subspace`, ARC SAE **(NEW)** |
| Tbl 1 | Architectures/params + per-task headline accuracy | — | paper Tbl 1–2; ARC diagnostics |
| Tbl 2 | Probe decodability vs MLP gain, key targets × 3 tasks | — | probe summaries |

Appendix figures: trajectory-PCA, directional-convergence, probe-geometry, ARC voting/accuracy,
SAE dead-feature analysis, full per-step tables.

---

## Title + Abstract

- **Title:** *What Hierarchical Reasoning Models Compute: A Cross-Task Mechanistic Study.*
  Retains "mechanistic study"; "cross-task" signals the scope upgrade over the workshop version.
- **Abstract** (single ~180-word funnel paragraph). **Reasoning models increasingly refine a
  *latent* state rather than emit token-by-token rationales, and HRM is a strong, tiny (27M-param)
  exemplar across Sudoku, Maze, and ARC-AGI.** → gap: what HRM actually computes, and whether that
  mechanism is task-general, is unknown. → what we do: a mechanistic study across **three** tasks
  combining linear/MLP probing, directed and SAE-based causal ablation, freezing, and cross-puzzle
  patching. → findings: (F1) `z_H` holds an iteratively refined solution state whose *refinement
  depth is task-dependent*; (F2) features decode at 80–100% yet ablating them is causally inert —
  *decodable ≠ used*; (F3) computation is *deeply distributed* — no SAE/probe subset rivals
  full-state ablation. → implication: probing alone overstates mechanism in latent reasoners;
  causal validation is necessary. Close on the methodological takeaway.

## 1. Introduction (5 paragraphs + contributions; Figure 1)

- **P1 — Context.** **Machine reasoning is shifting from verbalized chain-of-thought to iterative
  refinement of a continuous latent state.** Motivate latent reasoning; introduce HRM as
  hierarchical, recurrent, parameter-light, and empirically strong on Sudoku/Maze/ARC. → cite HRM,
  Coconut, recurrent-depth, Ren&Liu.
- **P2 — Gap.** **Strong benchmark numbers do not reveal the underlying algorithm.** Prior HRM
  analysis is partial (Ren&Liu: fixed points / PCA modes; Knoop&Kamradt: ARC ablations) and never
  asks whether *decodable* features are *causally used*, nor whether the mechanism holds across
  tasks. State three RQs: (RQ1) how is work divided between `z_H`/`z_L`? (RQ2) what do the latents
  encode? (RQ3) are decoded features the ones the model causally uses?
- **P3 — Our work + claims.** **We mechanistically dissect HRM across Sudoku, Maze, and
  ARC-AGI-2.** State F1/F2/F3 crisply, each carrying the cross-task qualifier; flag the new
  contrast (refinement is progressive on Sudoku but readout-concentrated on Maze).
- **P3.5 — Evidence preview.** **Each claim rests on qualitatively distinct, converging evidence.**
  F1: per-step ablation + freezing + cross-puzzle patching + trajectory convergence; F2: probing
  vs directed ablation; F3: SAE causal ablation + low-rank causal-subspace ablation.
- **P4 — Impact.** **Our most transferable result is a cautionary, causally-validated finding:
  high probe accuracy overstates what a latent reasoner computes.** Plus: the first cross-task
  mechanistic characterization of HRM. Forward-ref Discussion.
- **Contributions (4 bullets):** (i) cross-task validation of the three findings (Sudoku+Maze+ARC);
  (ii) the *task-dependent refinement-depth* contrast (new mechanistic insight); (iii) the
  probe/SAE-vs-full-state causal dissociation replicated on three tasks; (iv) an honest
  representation-level ARC study, with a reproducibility caveat for transductive checkpoints.
- **Figure 1 (money figure).** Top row: HRM `z_H`/`z_L` segment loop feeding the three task grids.
  Bottom row: a 3-mini-panel teaser of F2 — probe accuracy bars (high) beside directed-ablation Δ
  (≈0 with a random-control band). A reader should grasp the thesis from Fig 1 + caption alone.

## 2. Related Work (positioning, not a survey)

- **P1 — Latent-space reasoning.** **A line of work replaces token-level CoT with computation in a
  latent state.** Coconut/Heima/recurrent-depth/Universal-Transformer; HRM as the hierarchical
  recurrent instance we study.
- **P2 — Mechanistic interpretability tools.** **Our toolkit — probing, causal ablation, and sparse
  autoencoders — is standard, but we use it to test causal relevance, not just decodability.**
  Belinkov (probing); Ravfogel/Elazar (amnesic/null-space); Cunningham/Templeton (SAEs).
- **P3 — HRM-specific prior work.** **Two prior studies touch HRM but leave our questions open.**
  Ren&Liu (structured-guesser, fixed points) and Knoop&Kamradt (ARC reproduction); we add causal
  relevance, the two-timescale division of labour, and cross-task breadth.

## 3. Background & Setup

- **P1 — HRM architecture.** **HRM couples a fast low-level state `z_L` and a slow high-level state
  `z_H` updated on different timescales.** `N`=2 segments × `T` low-level steps, ACT halting up to
  16, one-step-gradient training; index segments by recursive step `t∈{0,…,15}`. Details → App.
- **P2 — Tasks & metrics.** **We study three constraint-style reasoning tasks chosen to expose
  different solution structures.** Sudoku-Extreme (cell accuracy / constraint violations; main
  task), Maze-Hard (valid start→goal path — a structural metric the token metric hides), ARC-AGI-2
  (per-cell colour accuracy). One sentence each on why the task is diagnostic.
- **P3 — ARC adaptation caveat (short; forward-ref Limitations).** **For ARC we use the released
  checkpoint with the reasoning core frozen and the per-puzzle embedding re-fit (Path A), so ARC
  results are representation-level, not exact-grid solving.** Full reason in Limitations/appendix.

## 4. Finding 1 — HRM iteratively refines a solution state in `z_H`, with task-dependent depth

- **P1 — Claim.** **`z_H` carries an iteratively refined solution while `z_L` acts as a working
  buffer, but the *depth* of refinement is task-dependent.**
- **P2 — Per-step ablation.** **Zeroing `z_H` becomes increasingly destructive over reasoning
  steps, while no single `z_L` step matters.** Sudoku: step-0 −6.7% → step-15 −26.1% (all-steps
  −18.0%); `z_L` only in aggregate (−19.3%). → `controlled/ablation`, Fig 2a.
- **P3 — Metric artifact (Maze).** **On Maze, ablating `z_H` barely moves token accuracy (−0.8%)
  yet destroys path validity (−32.5%) — per-token metrics hide the real damage.** Motivates
  structural metrics. → `maze/hardened/ablation_controlled`.
- **P4 — Freezing reveals the schedule (the new contrast).** **Freezing `z_H` after step k exposes
  *where* the work happens, and it differs sharply by task.** Sudoku decays smoothly (k0 −9.1% →
  k8 −0.4%, work done by ~step 8); **Maze is flat (~0% at every k) → `z_H` is load-bearing only at
  the final readout**; **ARC freeze curve (NEW)** placed on this spectrum. → `controlled/freeze`,
  `maze/hardened/freeze_controlled`, `results/arc/freeze` **(NEW)**, Fig 2b.
- **P5 — Cross-puzzle patching.** **Transplanting a donor `z_H` confirms it is puzzle-specific, and
  confirms the same schedule.** Sudoku catastrophic (steps 1–3 −53%, 5–7 −62%); Maze decisive only
  at readout (+2% at step 8, −26% at step 15); **ARC patching (NEW)**. → maze
  `patching_full_steps`, Sudoku `patching`, `results/arc/patching_full_steps` **(NEW)**, Fig 3.
- **P6 — Two-timescale + convergence.** **Solved trajectories converge to a fixed point while
  failed ones oscillate, and `z_H` is the persistent intervenable state.** cos>0.99, ‖Δz_H‖→0
  (solved) vs ‖Δz_H‖≈1.5 (failed). → directional-convergence / trajectory-PCA (appendix figure).
- **P7 — Takeaway.** **Across all three tasks `z_H` is the refined plan, but its refinement
  schedule is task-shaped — progressive for Sudoku, readout-concentrated for Maze.**

## 5. Finding 2 — Probe readout ≠ causally relevant

- **P1 — Claim.** **Task features are highly decodable from `z_H`, yet ablating those very
  directions is causally inert: decodable ≠ used.**
- **P2 — Probes decode (all 3 tasks).** **Linear probes recover task structure far above chance on
  every task.** Sudoku constraints ~88–90% (correctness 83.5%); Maze geometry ~99% (on-path 98.8%
  vs 64% base); ARC per-cell 91.6% & output-colour 88.6% (vs 11%). → `*/linear_probes`, Tbl 2,
  Fig 4a.
- **P3 — Linearly accessible.** **MLP probes add <1pp over linear, so the information is linearly
  present and a linear ablation is a fair causal test.** → MLP probe summaries, Tbl 2.
- **P4 — Directed ablation: probe ≈ random.** **Removing probe directions changes task accuracy
  within ±1%, indistinguishable from random controls.** Sudoku |Δ|<1% (per-cell probe +0.03%, n.s.);
  ARC ≤0.55% (random band ±0.06%). → `*/directed_ablation`, `arc_paper_figures`, Fig 4b.
- **P5 — The honest exception (Maze).** **One direction bucks the pattern: the Maze
  on-optimal-path direction has a small but significant causal effect (−2.8% vs +0.08% random) —
  we report it as the ceiling, not the rule.** (Red-team built in.) → `maze/.../directed_ablation`.
- **P6 — Takeaway.** **High probe accuracy is not evidence of causal use; causal validation is
  necessary — the paper's most transferable methodological result.**

## 6. Finding 3 — Computation is deeply distributed

- **P1 — Claim.** **No small, identifiable feature set carries HRM's computation; it is spread
  across the high-dimensional `z_H`.**
- **P2 — SAE causal ablation (3 tasks; ARC NEW).** **Ablating the top-50 SAE features is no more
  damaging than ablating 50 random features, and both are far weaker than full-state ablation.**
  Sudoku −3.6% vs −3.4% (full −18%); Maze −3.0% vs −3.3%; **ARC (NEW)**. No mono-semantic feature.
  → `*/sae_study/causal_ablation`, `results/arc/sae_study` **(NEW)**, Fig 5.
- **P3 — Causal ladder / subspace.** **A causal ladder orders the evidence: full-state ≫ SAE ≫
  probe ≈ random.** For ARC the causal mass concentrates in a *rank-4* subspace (−9.3%) that is
  *not* the readable probe directions. → `causal_subspace`, `fig_arc_distributed_computation`,
  Fig 5.
- **P4 — Takeaway.** **Computation is distributed and redundant — consistent with truncated
  one-step-gradient training that exerts no per-feature sparsity pressure.**

## 7. Discussion

- **P1 — What HRM is.** **HRM is a constraint-aware iterative refiner that holds a puzzle-specific,
  distributed solution state in `z_H`; hierarchy is an interpretability-friendly inductive bias,
  not the raw-accuracy driver** (a flat recurrent transformer matches it on Sudoku).
- **P2 — Methodological lesson.** **Probing-only interpretability overstates mechanism and should be
  paired with causal ablation** — a lesson that generalizes beyond HRM to latent reasoners.
- **P3 — Cross-task lesson.** **Mechanism is shared but the schedule is task-dependent, so
  single-task mechanistic interpretability can mislead about depth and locus.**

## 8. Limitations & Conclusion

- **P1 — ARC reproducibility caveat (negative result, as a note).** **The released ARC checkpoint
  is ~95% an orphaned transductive `puzzle_emb`, so we study a frozen-core Path-A adaptation at the
  representation level and do not report ARC solving accuracy.** The augmentation build is
  filesystem-order dependent and no prebuilt dataset/identifiers were released, so the released
  seed (42) cannot realign it; usable accuracy needs full retraining (issue-#12 precedent). Details
  → appendix.
- **P2 — Other limitations.** **Three caveats bound our claims.** Single SAE config (d=2048,
  λ=0.01); `z_L` causal content only partially probed; ARC's frozen core recovers per-cell colour
  but not output *shape* (per-cell ≠ exact grid).
- **P3 — Conclusion.** **Across three tasks, HRM refines a distributed, puzzle-specific solution
  state in `z_H` whose decodable features overstate what it causally uses.** Restate F1/F2/F3 + the
  methodological takeaway; one forward-looking sentence (sparsity-promoting training; `z_L` causal
  study).

## Appendix (supplementary; non-load-bearing)

Full HRM architecture/training; per-task dataset construction; **ARC Path-A + reproducibility
deep-dive** (orphaned `puzzle_emb`, build non-determinism, issue-#12 retraining precedent); all
per-step ablation tables; freeze/patching full tables; SAE config + dead-feature analysis; probe
geometry/PCA; ARC voting/accuracy diagnostics; extra figures.

---

### Status of evidence

- **Ready now** (numbers in `docs/INTERPRETABILITY_RESULTS_SUMMARY.md`): all of F2; F1 per-step
  ablation, metric-artifact, Sudoku/Maze freezing & patching, convergence; F3 Sudoku/Maze SAE +
  ARC causal-subspace.
- **To be produced (NEW) before the draft is complete:** ARC SAE causal ablation, ARC freezing,
  ARC cross-puzzle patching — the three experiments that bring ARC to parity in Fig 2b, Fig 3,
  Fig 5. Drivers mirror the Maze implementations; run via `srun` on `ws-l3-019`.
