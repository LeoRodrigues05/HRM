# HRM Mechanistic Interpretability — Final Push to Publication

## Project Summary

This repository contains mechanistic interpretability (MI) research on the **Hierarchical Reasoning Model (HRM)** ([Wang et al., 2025; arXiv:2506.21734](https://arxiv.org/abs/2506.21734)), a 27M-parameter hierarchical recurrent transformer that solves 9×9 Sudoku puzzles in a single forward pass. The model uses two interdependent modules — a high-level module (`z_H`, slow, abstract planning) and a low-level module (`z_L`, fast, detailed computation) — iterated over 16 "deep supervision" segments of 2 H-cycles × 2 L-cycles each.

The MI work investigates: **Why does HRM work so well? What do z_H and z_L encode? Where are constraints stored? How does the hierarchy enable reasoning?**

---

## Current Experiment Inventory

### Phase 7 — Paper Replication (Complete, committed)

| Exp | Name | Script | N | Key Finding |
|-----|------|--------|---|-------------|
| E1 | Easy/Hard Analysis | `experiments/paper_replication/exp1_easy_hard_analysis.py` | 423k | 90.2% hard, 73.6% easy (**contradicts** paper claim) |
| E3 | Step Dynamics | `experiments/paper_replication/exp3_step_dynamics.py` | 423k | 67.9% → 83.7% accuracy across 16 steps (**verified**) |
| E4 | Specialization Probes | `experiments/paper_replication/exp4_specialization_probes.py` | 423k | z_H and z_L encode distinct information (**verified**) |
| E5 | Activation Patching | `experiments/paper_replication/exp5_activation_patching.py` | 6 pairs | z_H patch = −56.4%, z_L = −4.1% (**verified**) |
| E6 | Latent Trajectories | `experiments/paper_replication/exp6_latent_trajectories.py` | — | Norm, rotation, PCA evolution |
| E7 | Segment Loss Scaling | `experiments/paper_replication/exp7_segment_loss_scaling.py` | — | Loss scaling dynamics |

### Phase 8 — Extended Analysis (Uncommitted, latest work)

| Exp | Name | Script | N | Key Finding | Quality |
|-----|------|--------|---|-------------|---------|
| E1-ext | z_H Zero Ablation | `scripts/batch_ablation_1k.py` | 2,803 | −21.2% all-steps; step 10 > 8 > 6 > 4 | ⚠️ Multi-variable ablation |
| E2b | Freeze z_H | `scripts/batch_freeze_h.py` | 200 | 3 phases: rapid→refine→converge | ⚠️ Small N |
| E5-ext | Time-Shift Patching | `scripts/batch_time_shift.py` | 100 | Future→past +3.5%, past→future −4% | ⚠️ Small N |
| E8 | Constraint Probes | `scripts/e8_constraint_probes.py` | 500 | ~90% probe accuracy, 3-d subspace in 512-d z_H | ✅ Good |
| E9 | Directed Ablation | `scripts/e9_directed_ablation.py` | 200 | <1% causal effect (**critical negative result**: readout ≠ computation) | ⚠️ Small N, needs stats |

### Infrastructure

- Bidirectional activation patching framework (`scripts/activation_patching.py`)
- Linear probe pipeline (`scripts/train_linear_probes.py`, `sweep_linear_probes.py`)
- Batch experiment orchestration (ablation, freeze, time-shift)
- 9+ publication-quality plots (PNG/PDF) in `results/e8_e9_plots/`, `results/presentation_plots/`
- HTML Sudoku visualization reports

---

## What's Missing — 4 Major Gaps

### Gap A: Baseline Comparisons (Critical)

**No baseline implementations exist** in the codebase. Every MI paper needs to demonstrate the model under study is special. Required:

1. **Vanilla RNN** — Same ~27M params, same Sudoku data, same training. Goal: show premature convergence (validates HRM's hierarchical convergence claim).
2. **Universal Transformer** — Same compute budget, recurrent but flat (no H/L split). Proves hierarchy (not just recurrence) is the differentiator.
3. **(Optional) Deep Transformer** — 8/16 layers. The HRM paper shows this saturates; replicate.

Compare on: puzzle accuracy, Hamming convergence, constraint satisfaction, probe accuracy.

→ **See [01_run_baselines.md](01_run_baselines.md) for the executable query.**

### Gap B: Tighter Ablation Methodology

Current ablations vary multiple factors simultaneously. For publication, need single-variable controlled ablations with larger N and confidence intervals.

- E1: Single-step ablation at each step independently, N≥5,000
- E2b: N≥1,000, add z_L freeze control, all steps 0–15
- E5: Full transfer matrix, N≥500
- E9: N≥500, proper statistical tests, multi-direction ablation
- New E1b: z_L ablation (complement to E1)

→ **See [02_tighten_experiments.md](02_tighten_experiments.md) for the executable query.**

### Gap C: Sparse Autoencoder (SAE) Study

E9 showed linear probes find readout features, not computational features. SAE can find overcomplete sparse features that may be causally relevant.

- Train SAE on z_H activations (dictionary sizes 1024–4096)
- Analyze feature specialization per constraint type
- Causal validation: ablate individual SAE features → compare to E9 probe directions
- Key prediction: SAE features should show higher causal specificity than linear probes

→ **See [03_sae_study.md](03_sae_study.md) for the executable query.**

### Gap D: Statistical Rigor & Presentation

- Error bars (95% CI) on all plots
- Effect sizes (Cohen's d) for all comparisons
- Consistent puzzle set across all experiments
- Random controls for every ablation
- Reproducibility (reported random seeds)

→ **See [04_polish_and_figures.md](04_polish_and_figures.md) for the executable query.**

---

## Recommended Publication Narrative

### Story Arc

1. **Introduction**: HRM achieves near-perfect Sudoku with 27M params. How does it work?
2. **z_H is the planmaker**: Ablation (E1) + freeze (E2b) + time-shift (E5) → z_H carries a progressive, improving plan
3. **z_H encodes constraints**: Linear probes (E8) → 90% accuracy decoding row/col/box violations
4. **But probes find readout, not computation**: Directed ablation (E9) → <1% causal effect
5. **SAE reveals true computational features**: E10 → sparse features with higher causal specificity
6. **HRM vs. flat alternatives**: Baseline comparison → hierarchy is essential for reasoning
7. **Emergent dimensionality hierarchy**: Brain correspondence (PR analysis from original paper)

### Target Venues

- **NeurIPS** (mechanistic interpretability track)
- **ICLR** (representation learning)
- **ICML**
- **TMLR** (faster review cycle)

### Working Title

*"Mechanistic Interpretability of Hierarchical Recurrent Reasoning: How the HRM Solves Sudoku"*

---

## Implementation Phases

### Phase 1: Baselines (Week 1–2)
1. Implement vanilla RNN baseline model class
2. Implement Universal Transformer baseline
3. Create training configs and adapt `pretrain.py`
4. Train both on Sudoku-Extreme
5. Evaluate: accuracy + Hamming + patching
6. Compare results to HRM

### Phase 2: Tighten Existing Experiments (Week 2–3, parallel with Phase 1)
7. Re-run E1 with single-step controlled ablation (N=5000)
8. Re-run E2b with N=1000, add z_L freeze
9. Re-run E5 with full transfer matrix (N=500)
10. Re-run E9 with N=500, add statistical tests
11. New z_L ablation experiment
12. Add CI/error bars to all plots

### Phase 3: SAE Study (Week 3–4)
13. Collect z_H activations (≥1000 puzzles, all 16 steps)
14. Implement SAE (or integrate `sae-lens`)
15. Train SAE with hyperparameter sweep
16. Analyze SAE feature specialization
17. Causal ablation of SAE features vs. E9 probe directions

### Phase 4: Polish & Write (Week 4–5)
18. Publication-quality figures with error bars
19. Paper draft (methodology, results, introduction, related work)
20. Internal review

---

## Key Files Reference

### Architecture & Training
| File | Purpose |
|------|---------|
| `models/hrm/hrm_act_v1.py` | HRM v1 model (z_H, z_L, ACT halting) |
| `models/layers.py` | Reusable Transformer blocks (attention, SwiGLU, RoPE) |
| `models/common.py` | Weight initialization, shared utils |
| `models/losses.py` | Loss heads (StableMax CE, ACT Q-learning) |
| `pretrain.py` | Training loop (cosine LR, deep supervision) |
| `config/arch/hrm_v1.yaml` | Architecture config |
| `config/cfg_pretrain.yaml` | Training hyperparameters |

### Existing Experiment Scripts
| File | Experiment |
|------|-----------|
| `scripts/batch_ablation_1k.py` | E1: z_H zero ablation |
| `scripts/batch_freeze_h.py` | E2b: Freeze z_H at step k |
| `scripts/batch_time_shift.py` | E5: Time-shift patching |
| `scripts/e8_constraint_probes.py` | E8: Constraint-specific linear probes |
| `scripts/e9_directed_ablation.py` | E9: Directed ablation of probe directions |
| `scripts/activation_patching.py` | Core patching framework |
| `scripts/activation_ablation.py` | Zero-out ablation framework |

### Visualization
| File | Purpose |
|------|---------|
| `scripts/plot_e8_e9.py` | E8/E9 publication figures |
| `scripts/plot_presentation.py` | Combined presentation plots |
| `scripts/plot_ablation_results.py` | Ablation trajectory visualization |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Focus on Sudoku only | Deepest existing analysis, clearest story |
| Use HRM v1 for main results | v2 adds structural biases (sparse attention, GNN) that confound MI findings |
| SAE on z_H first, z_L second | z_H is where distributed features reside (higher PR, more interesting) |
| Target ≥1,000 puzzles per experiment | Balance of statistical rigor and compute budget |
| Vanilla RNN + Universal Transformer as baselines | Minimum set to prove hierarchy matters |

---

## Executable Query Files

| File | Purpose | Dependencies |
|------|---------|-------------|
| [`01_run_baselines.md`](01_run_baselines.md) | Implement & train RNN + UT baselines, run comparative experiments | None |
| [`02_tighten_experiments.md`](02_tighten_experiments.md) | Re-run E1/E2b/E5/E9 with proper controls & larger N | None (parallel with 01) |
| [`03_sae_study.md`](03_sae_study.md) | SAE training & causal validation on z_H | Depends on E8 probe weights |
| [`04_polish_and_figures.md`](04_polish_and_figures.md) | Publication figures with error bars, paper-ready output | Depends on 01–03 results |
