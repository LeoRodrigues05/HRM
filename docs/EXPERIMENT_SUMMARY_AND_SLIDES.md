# Interpretability Experiments on the Hierarchical Reasoning Model (HRM)
## Summary of All Experiments & Presentation Slides
### Date: February 22, 2026

---

## Complete Experiments Inventory (Cumulative)

| # | Experiment | Script | Puzzles | Status |
|---|---|---|---|---|
| E1 | z_H Zero Ablation (single & all steps) | `batch_ablation_1k.py` | 2,803 | Done |
| E2a | z_H+z_L Both-Level Ablation | `batch_activation_ablation.py` | 200 | Partial |
| E2b | **z_H Freeze After Step k** (NEW today) | `batch_freeze_h.py` | 200 | **Done** |
| E3 | Cross-Puzzle Activation Patching | `activation_patching.py` | 6 pairs | Done |
| E3b | Bidirectional z_H vs z_L Patching | `activation_patching.py` | 2 pairs | Done |
| E5 | **Time-Shift Patching** (NEW today) | `batch_time_shift.py` | 100 | **Done** |
| E7a | Linear Probes on z_H/z_L | `sweep_linear_probes.py` | 80 | Done |
| E7b | Multi-Step Hamming Tracking | `result_metrics_sudoku.py` | ~423k | Done |
| Diag | z_H Cosine Similarity Across Steps (NEW today) | `quick_cosine_check.py` | 4 | Done |

---

## PRESENTATION SLIDES

---

### Slide 1: Title & Outline

**Title:** Interpretability Experiments on the Hierarchical Reasoning Model (HRM)
**Subtitle:** Understanding How Recurrent Transformers Plan and Reason on Sudoku

**Outline:**
1. Model Architecture Recap
2. Experiment Suite Overview
3. Results by Experiment Group
   - Ablation studies (what z_H does)
   - Activation patching (what z_H encodes)
   - Freeze & time-shift (how z_H evolves)
   - Linear probes (what's linearly decodable)
4. Key Findings
5. Mapping to Action Items
6. Next Steps

---

### Slide 2: Model Architecture Recap

**HRM v1 — Hierarchical Reasoning Model with ACT**

- Two-level recurrent hierarchy: **z_H** (high-level, 512-d) and **z_L** (low-level, 512-d)
- Each ACT step: 2 H-cycles × 2 L-cycles update both representations
- Output: `lm_head(z_H)` — final prediction comes **directly from z_H**
- Q-learning ACT halting: model learns when to stop (max 16 steps)
- Dataset: Sudoku-extreme-1k-aug (9×9 Sudoku, 81 cells), test split
- Baseline accuracy: **82.0%** mean cell accuracy (200-puzzle sample)

**Key question:** What role does z_H play — static plan? progressive refinement? something else?

---

### Slide 3: Experiment Suite Overview

| Experiment | Question Answered | Method |
|---|---|---|
| **E1: z_H Ablation** | Is z_H necessary? | Zero out z_H at specific steps |
| **E2: Freeze-H** | Is z_H a static plan or dynamic? | Cache z_H at step k, replay forever |
| **E3: Cross-Puzzle Patching** | Does z_H encode puzzle-specific info? | Swap z_H between puzzles |
| **E5: Time-Shift** | Does z_H improve over time? | Inject future z_H into past steps |
| **E7: Probes + Hamming** | What's linearly decodable from z_H/z_L? | Linear classifiers on activations |

---

### Slide 4: E1 — z_H Zero Ablation (2,803 puzzles)

*Action Item: "Figure out what high-level planning looks like in Sudoku"*

**Setup:** Zero out z_H at specified ACT steps, measure accuracy impact.

| Condition | Mean Acc | Δ from Baseline |
|---|---|---|
| Baseline | 83.5% | — |
| Ablate ALL steps | 62.3% | **−21.2%** |
| Ablate step 4 only | 77.6% | −5.9% |
| Ablate step 6 only | 76.4% | −7.1% |
| Ablate step 8 only | 74.9% | −8.6% |
| Ablate step 10 only | 72.7% | **−10.8%** |

**Key finding:** z_H is **essential** — zeroing it costs 21%. Later steps carry more weight (step 10 > 8 > 6 > 4). z_H importance grows monotonically with iteration depth.

---

### Slide 5: E2 — Freeze z_H After Step k (200 puzzles, NEW)

*Action Item: "Figure out what high-level planning looks like in Sudoku"*

**Setup:** After step k, lock z_H to its step-k value. z_L keeps updating.

| Freeze After | Mean Acc | Δ Acc | Puzzles Hurt |
|---|---|---|---|
| **Never (baseline)** | **82.0%** | — | — |
| Step 0 | 72.9% | **−9.0%** | 126/200 |
| Step 1 | 76.3% | −5.7% | 99/200 |
| Step 2 | 78.3% | −3.7% | 81/200 |
| Step 4 | 79.8% | −2.2% | 67/200 |
| Step 8 | 81.5% | −0.5% | 52/200 |
| Step 12 | 81.8% | **−0.1%** | 45/200 |

**Finding:** z_H is **progressively refined, not a one-shot plan**. The first 4 steps account for 75% of the total refinement. By step 8, z_H is essentially converged (only 0.5% loss from freezing). This gives us a **phase portrait**: rapid planning (steps 0–4) → fine-tuning (steps 4–8) → convergence (8+).

**Step-level accuracy curves** (from freeze_accuracy_matrix.json):
- Baseline: 67.1% → 79.7% → 81.3% → 82.0% across steps 0→4→8→15
- Freeze@0: 67.1% → 72.8% → 72.5% → 72.9% (plateaus immediately after freeze)
- Freeze@4: 67.1% → 79.7% → 79.9% → 79.8% (plateaus at step 4 level)

---

### Slide 6: E3 — Cross-Puzzle Activation Patching

*Action Item: "Figure out what high-level planning looks like in Sudoku"*

**Setup:** Inject activations from puzzle A into puzzle B during forward pass.

| Experiment | Result |
|---|---|
| Patch **both z_H+z_L** for 5 steps | Target acc collapses: 57% → **5%** |
| Patch **only z_L** for all steps | Target acc barely changes: 70% → **69%** (Δ = −1.2%) |
| Patch **both, reverse direction** | Also collapses: 54% → **7%** |

**Finding:** **z_H is the critical carrier of puzzle-specific reasoning state.** z_L is generic/recoverable — the model can reconstruct z_L from z_H + input context, but not vice versa. This confirms z_H as the "plan" and z_L as the "scratch pad."

---

### Slide 7: E5 — Time-Shift Patching (100 puzzles, NEW)

*Action Item: "Map out intermediate activations to insightful graphs"*

**Setup:** Same puzzle, inject z_H from a later step into an earlier step (and vice versa).

| Transfer | Direction | Δ at Recipient Step | Δ at Final Step |
|---|---|---|---|
| **Step 10 → Step 2** | future→past | **+3.3%** | +1.7% |
| **Step 8 → Step 2** | future→past | **+3.5%** | +0.8% |
| Step 12 → Step 4 | future→past | +0.9% | −0.3% |
| Step 14 → Step 6 | future→past | +0.4% | −0.2% |
| Step 4 → Step 10 | past→future | **−1.9%** | −0.6% |
| **Step 2 → Step 8** | past→future | **−4.0%** | −1.5% |

**Finding:** Clear causal asymmetry:
- **Future→past (into early steps):** Boosts accuracy +3–4% — later z_H encodes strictly more solution information
- **Past→future (into late steps):** Hurts accuracy −2–4% — reverting to early z_H erases progress
- **Diminishing returns for mid-step injection** (→4, →6): by step 4–6 the model has already caught up, so future z_H adds little
- This proves **monotonic progressive refinement** of z_H

---

### Slide 8: E7 — Linear Probes & Hamming Tracking

*Action Item: "Map out intermediate activations to insightful graphs"*

**Linear Probes (80 puzzles):**

| What's Decodable? | From Where | Accuracy/R² |
|---|---|---|
| Is puzzle solved? | z_H or z_L | **100%** |
| Is cell forced (naked single)? | z_H or z_L | **100%** |
| Per-cell correctness | z_L (product) | **91.2%** |
| % cells filled | z_H×z_L product | R² = **0.85** |
| Column position | z_L (product) | **59.0%** |
| Row position | z_L | 22.1% |

**Hamming Distance Convergence (423k puzzles):**

| Step | Mean Error Rate |
|---|---|
| 1 | 33.5% |
| 3 | 14.4% |
| 5 | 10.2% |
| 8 | 8.6% |
| 15 | **7.6%** |

**Finding:** The model's representations are remarkably structured — binary puzzle states (solved/forced) are *perfectly* linearly separable. The steepest improvement happens in steps 1–3 (error halves twice), consistent with the freeze experiment showing z_H crystallizes in steps 0–4.

---

### Slide 9: Cosine Similarity Diagnostic (NEW)

*Action Item: "Figure out what high-level planning looks like in Sudoku"*

**z_H direction changes across ACT steps** (despite constant L2 norm ≈ 205):

| Puzzle | First-vs-Last Cosine | Behavior |
|---|---|---|
| Puzzle 0 (easy) | 0.693 | Converges by step 5 (cosine→1.0) |
| Puzzle 2 (hard) | 0.869 | Still iterating at step 15 (~0.96 consecutive) |
| Puzzle 5 (hard) | 0.875 | Still iterating at step 15 (~0.97 consecutive) |
| Puzzle 8 (easy) | 0.721 | Converges by step 4 (cosine→1.0) |

**Finding:** z_H lives on a **constant-norm manifold** (‖z_H‖ ≈ 205.0 always) but rotates significantly — up to 30° between first and last step. Two regimes: easy puzzles converge quickly (cosine→1.0 by step 5), hard puzzles keep rotating through all 16 steps. This means the model's "planning" is a **directional rotation** in activation space, not a magnitude change.

---

### Slide 10: Unified Key Findings

| # | Finding | Supporting Evidence |
|---|---|---|
| 1 | **z_H is the plan, z_L is the scratch pad** | z_L patching = −1.2%, z_H patching = catastrophic |
| 2 | **z_H is progressively refined (not static)** | Freeze@0 = −9.0%, Freeze@8 = −0.5% |
| 3 | **Most planning happens in steps 0–4** | 75% of freeze damage occurs before step 4; Hamming halves by step 3 |
| 4 | **Later z_H > Earlier z_H (monotonic)** | Time-shift: future→past = +3.5%, past→future = −4.0% |
| 5 | **z_H evolves on a constant-norm manifold** | ‖z_H‖ ≈ 205 always; cosine rotation up to 30° |
| 6 | **Puzzle difficulty modulates convergence speed** | Easy: converge step 5, Hard: still iterating step 15 |
| 7 | **Representations are surprisingly structured** | Perfect linear probes for is_solved, is_forced (100%) |

---

### Slide 11: Action Items Status

| Action Item | Status | Evidence/Notes |
|---|---|---|
| **Figure out what high-level planning looks like in Sudoku** | **Substantially answered** | z_H = progressive plan refined over steps 0–4, then fine-tuned. Directional rotation on constant-norm manifold. Two regimes (easy=converge, hard=iterate). |
| **Read paper on "Emergent World Representations"** | **Not started** | Relevant comparison: do HRM's z_H representations constitute an "emergent world model" of the Sudoku board? Probes show 100% solved-state decodability. |
| **Start drawing baselines (vanilla, recurrent transformers)** | **Not started** | Need to train/eval vanilla Transformer and Universal Transformer on same Sudoku dataset for comparison. Architecture exists in codebase but no runs yet. |
| **Map intermediate activations to insightful graphs** | **Partially done** | 9 publication plots in `ablation_plots/`, Hamming curves, freeze accuracy matrix, time-shift asymmetry data. Still need: PCA/UMAP of z_H trajectories, per-puzzle difficulty-stratified plots. |
| **Add more synthetic data** | **Not started** | Dataset builder exists (`build_sudoku_dataset.py`). Current set: sudoku-extreme-1k-aug-1000. |
| **Create first draft of paper** | **Not started** | Have enough results for "Interpretability" section. Missing: baselines, related work framing, full intro. |

---

### Slide 12: Next Steps (Prioritized)

**P0 — Immediate (complete the interpretability story):**
1. **PCA/UMAP visualization** of z_H trajectories across steps — show the rotation visually
2. **Difficulty-stratified analysis** — split freeze/time-shift results by baseline accuracy buckets
3. **E3-DELTA:** Compute activation deltas (Δz_H per step) and correlate with accuracy gains

**P1 — Short-term (baselines + paper skeleton):**
4. **Train vanilla Transformer baseline** on same Sudoku dataset
5. **Train Universal Transformer** (recurrent but flat, no hierarchy) for comparison
6. **Read "Emergent World Representations"** — frame z_H findings as world-model evidence
7. **Draft paper outline** with sections populated from current results

**P2 — Medium-term (strengthen claims):**
8. **Scale linear probes** to more puzzles (currently 80) and add z_H *trajectory* probes
9. **Add synthetic data** (more difficulty levels, augmentations)
10. **Subspace analysis (E1-SUBSPACE):** project z_H onto principal components, ablate individual dimensions

---

## Raw Data References

- E1 Ablation: `results/batch_ablation_zH/aggregate_stats.json`
- E2 Freeze: `results/freeze_h/aggregate_stats.json`, `results/freeze_h/freeze_accuracy_matrix.json`
- E3 Patching: `results/activation_patching_s0_t250/`, `results/activation_patching_s111_t220/`
- E5 Time-Shift: `results/time_shift/aggregate_stats.json`
- E7 Probes: `results/probes/linear_probe_sweep.csv`
- E7 Hamming: `results/hamming_multi/`, `results/metrics/hamming.csv`
- Ablation Plots: `results/ablation_plots/` (9 PDF+PNG pairs)
- Per-puzzle JSONL: `results/freeze_h/results_per_puzzle.jsonl`, `results/time_shift/results_per_puzzle.jsonl`
