# E8 & E9 Experiment Findings: Constraint Probes and Causal Validation

## Executive Summary

**E8 (Constraint-Specific Linear Probes)** demonstrates that z_H linearly encodes rich
constraint information — row, column, and box violations are decodable at ~90% accuracy
from a single linear layer, and this accuracy improves across ACT steps as the model
refines its solution. The three violation directions are geometrically distinct (pairwise
cosines 0.60–0.73) and anti-correlated with the correctness direction (cosines −0.55 to
−0.59).

**E9 (Directed Ablation)** reveals that projecting out these probe directions from z_H
produces negligible causal effects (Δaccuracy < 1%, Δviolations < 0.3 per puzzle),
indistinguishable from random direction ablations. This is a critical negative result:
**the probe directions are readout features, not computational features**. The model's
constraint-solving mechanism is distributed across a higher-dimensional subspace that
cannot be captured by a single linear direction per constraint type.

---

## E8: Constraint-Specific Linear Probes

### Setup
- **Scale**: 500 puzzles × 81 cells = 40,500 per-cell samples (32,400 train / 8,100 val)
- **ACT Steps**: 0, 4, 8, 12, 15
- **Probe architecture**: Single linear layer (512 → 1 for binary, 512 → 10 for digit)
- **Targets**: 6 constraint types — per_cell_correct, is_given, violated_in_{row,col,box}, cell_digit
- **Training**: SGD with momentum, 60 epochs, lr=0.01, 80/20 train/val split

### Finding 1: z_H Linearly Encodes Constraint Information

| Target | Step 0 | Step 4 | Step 8 | Step 12 | Step 15 | Δ(0→15) |
|---|---|---|---|---|---|---|
| cell_digit | 98.5% | 98.9% | 99.0% | 99.2% | 99.0% | +0.5% |
| is_given | 99.6% | 95.5% | 96.4% | 96.5% | 97.3% | −2.3% |
| violated_in_row | 74.1% | 88.0% | 89.0% | 88.8% | **89.9%** | **+15.8%** |
| violated_in_col | 74.5% | 87.7% | 88.3% | 88.9% | **90.1%** | **+15.6%** |
| violated_in_box | 74.2% | 86.9% | 88.0% | 88.6% | **88.8%** | **+14.6%** |
| per_cell_correct | 71.1% | 80.7% | 83.2% | 83.1% | **83.5%** | **+12.4%** |

**Interpretation**: The model's high-level hidden state z_H becomes progressively more
informative about constraint violations as computation proceeds. The biggest jump occurs
between step 0 and step 4 (~14% improvement), with diminishing returns thereafter.
This aligns with the ACT step-utilization finding from E5: early steps build the
constraint representation, later steps refine it.

Structural features (cell_digit, is_given) are decodable from step 0 — these are
directly available from the input embedding. Dynamic features (violations, correctness)
require iterative reasoning and emerge across steps.

### Finding 2: Constraint Directions Are Geometrically Distinct

Pairwise cosine similarities between probe weight vectors (step 15):

| | row | col | box | correct |
|---|---|---|---|---|
| **row** | 1.00 | 0.67 | 0.69 | −0.55 |
| **col** | 0.67 | 1.00 | 0.73 | −0.59 |
| **box** | 0.69 | 0.73 | 1.00 | −0.55 |
| **correct** | −0.55 | −0.59 | −0.55 | 1.00 |

**Interpretation**:
- **Violation triplet (row/col/box)**: Moderately correlated (0.67–0.73), reflecting
  shared "something is wrong" semantics. But each has a unique component — they are NOT
  the same direction in 512-d space.
- **Correctness vs violations**: Consistently anti-correlated (−0.55 to −0.59). The
  model encodes "correct" and "violated" in roughly opposing directions, as expected from
  the task structure.
- **Not orthogonal**: If constraint types were fully independent, cosines would be ~0.
  The 0.6–0.7 range suggests the model uses a partially shared "constraint health"
  subspace with type-specific perturbations.

### Finding 3: Constraint Information Lives in a Low-Dimensional Subspace

PCA of the 4 constraint probe weight vectors (row/col/box/correct):

| Step | PC1 explained | PC2 explained | PC3 explained | Effective dim |
|---|---|---|---|---|
| 0 | 79.0% | 12.4% | 8.6% | ~1.5 |
| 4 | 75.1% | 13.5% | 11.4% | ~1.8 |
| 8 | 73.8% | 13.8% | 12.4% | ~1.9 |
| 12 | 74.9% | 14.9% | 10.2% | ~1.7 |
| 15 | 76.9% | 13.0% | 10.1% | ~1.6 |

The 4 constraint directions span approximately a **3-dimensional subspace** of the 512-d
z_H space. PC1 captures ~75–79% of variance (the shared "constraint health" axis), with
PC2 and PC3 capturing the type-specific differences. This extreme compression (3/512 = 0.6%
of available dimensions) suggests the model dedicates a tiny fraction of z_H capacity to
constraint tracking, with the rest used for other computational purposes.

### Finding 4: Weight Vector Norms Are Uniform

| Target | ‖W‖ (step 15) |
|---|---|
| violated_in_row | 1.20 |
| violated_in_col | 1.20 |
| violated_in_box | 1.16 |
| per_cell_correct | 1.11 |
| is_given | 2.55 |
| cell_digit | 3.91 |

The three violation probes have nearly identical norms (~1.2), suggesting they use similar
amounts of signal in z_H. The `is_given` probe has a larger norm (2.55) — it's reading a
stronger structural signal. `cell_digit` has the largest norm (3.91) because it's a
10-class multiclass probe that must separate more categories.

---

## E9: Directed Ablation (Causal Validation)

### Setup
- **Scale**: 200 puzzles
- **Method**: For each probe direction ŵ, apply z'_H = z_H − (z_H · ŵ)ŵ at every ACT step
- **Directions tested**: violated_in_row, violated_in_col, violated_in_box,
  per_cell_correct, is_given, plus 3 random control directions
- **Metrics**: Change in row/col/box violations, change in cell accuracy, cells broken

### Finding 5: Single-Direction Ablation Has Negligible Causal Effect

| Direction ablated | Δaccuracy | Δrow viols | Δcol viols | Δbox viols | Cells broken |
|---|---|---|---|---|---|
| violated_in_row | −0.09% | −0.04 | −0.15 | −0.07 | 2.5 |
| violated_in_col | −0.59% | +0.08 | −0.03 | +0.23 | 2.7 |
| violated_in_box | −0.23% | +0.14 | −0.11 | +0.16 | 2.5 |
| per_cell_correct | +0.72% | −0.22 | −0.29 | −0.31 | 2.0 |
| is_given | +0.20% | +0.02 | −0.08 | +0.02 | 2.7 |
| random_control_0 | −0.44% | +0.16 | −0.10 | +0.10 | 2.4 |
| random_control_1 | +0.07% | −0.18 | −0.41 | −0.25 | 1.9 |
| random_control_2 | +0.95% | −0.16 | −0.36 | −0.28 | 1.8 |

**Interpretation**: All accuracy changes are within ±1%, and violation count changes are
within ±0.4 — well within noise. Critically, the constraint directions produce no larger
effects than random controls. This rules out the hypothesis that a single linear direction
per constraint type is the model's primary computational mechanism.

### Finding 6: No Selective Specificity

| Direction | Specificity Score | Interpretation |
|---|---|---|
| violated_in_row | 0.14 | Well below chance (0.33) |
| violated_in_col | 0.08 | Near zero |
| violated_in_box | 0.40 | Slightly above chance |

For row and col directions, the specificity is *below* the 0.33 baseline — ablating the
"row" direction does not preferentially break row consistency. This means the directions
found by probes are not the directions the model *uses* to enforce constraints.

### Finding 7: Symmetry Between Broken and Fixed Cells

| Direction | Mean cells broken | Mean cells fixed |
|---|---|---|
| violated_in_row | 2.50 | 2.43 |
| violated_in_col | 2.66 | 2.18 |
| violated_in_box | 2.55 | 2.36 |
| random_control_0 | 2.44 | 2.08 |
| random_control_1 | 1.92 | 1.98 |
| random_control_2 | 1.82 | 2.59 |

The number of cells broken ≈ cells fixed for all directions, including randoms. This
pattern is consistent with **perturbation in a non-causally-relevant direction**: the
ablation slightly jostles the representation, randomly flipping some cells correct/incorrect
without systematic specificity.

---

## Joint Interpretation: What This Tells Us About z_H

### The Readout ≠ Computation Dissociation

E8 and E9 together reveal a fundamental dissociation:

1. **z_H is information-rich** — linear probes can decode constraint violations, cell
   correctness, and digit identity with 84–99% accuracy.
2. **This information is not stored in single directions** — removing any one direction
   has no measurable effect on the model's constraint-solving behavior.

This is the classic **"readout ≠ computation"** dissociation from neuroscience: just
because information can be linearly decoded from a neural population doesn't mean that
specific neurons/directions are *used* by downstream computation to drive behavior.

### Distributed Representation Hypothesis

The most parsimonious explanation: the model uses a **distributed, redundant encoding**
of constraint information across many dimensions of z_H. Removing 1 of 512 dimensions
(0.2% of the space) simply doesn't remove enough information to impair function. The
model likely encodes each constraint through a **subspace** of 10–50+ dimensions, with
significant redundancy that makes it robust to single-direction perturbation.

### Implications for E10 (Sparse Dictionary Learning)

This directly motivates the SAE (Sparse Autoencoder) approach:

- **Why linear probes fail causally**: They find the *maximum-variance* encoding direction,
  not the *functional* direction. These can differ in distributed representations.
- **What SAEs can reveal**: By decomposing z_H into a learned overcomplete dictionary,
  SAEs can find **sparse, interpretable features** that may correspond to functional units
  the model actually uses — even if these are nonlinear combinations of the original
  dimensions.
- **Key prediction**: SAE features should show higher causal specificity than linear probe
  directions when ablated individually, because they're learned to capture the model's
  own feature structure rather than being externally imposed.

---

## Experimental Details

| Parameter | E8 | E9 |
|---|---|---|
| Puzzles | 500 | 200 |
| Per-cell samples | 40,500 | 200 × 81 = 16,200 |
| GPU time | 258s (~4.3 min) | 562s (~9.4 min) |
| Hardware | NVIDIA RTX A6000 (49GB) | NVIDIA RTX A6000 (49GB) |
| Script | scripts/e8_constraint_probes.py | scripts/e9_directed_ablation.py |
| Output | results/e8_constraint_probes/ | results/e9_directed_ablation/ |
