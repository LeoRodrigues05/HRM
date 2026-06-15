# The Maze Story, in Contrast to Sudoku

*Mechanistic interpretation of the high-level recurrent state `z_H` in HRM, and why
the maze conclusion depended entirely on using the right metric.*

All numbers below come from the **hardened** re-runs (path-validity metrics, raised N,
bootstrap 95% CIs) under `results/maze/hardened/` and the Sudoku controlled suite under
`results/controlled/`. Both tasks use the **identical** HRM checkpoint architecture
(`H_cycles=2, L_cycles=2, halt_max_steps=16, hidden=512, H_layers=L_layers=4`); the only
difference is the *task*, not the network.

---

## 0. The bug that almost produced a false finding

The maze controlled-ablation and time-shift scripts originally summarised damage with
**token accuracy** only. On a 30×30 maze the output grid is ~900 cells dominated by walls
and empty corridor; the solution path is ~110 cells. A model can completely destroy the
path and still score ~99% token accuracy, because it gets all the walls and most empty
cells right.

> **z_H all-steps ablation, maze, N=200**
> token accuracy: **−0.8%** [−1.1, −0.6]  ← looks negligible
> valid start→goal path: **−32.5%** [−39.5, −26.0]  ← catastrophic

The earlier, token-only measurement said "ablating `z_H` barely changes the maze output,
so `z_H` is not causally important for maze." **That conclusion was a metric artifact.**
Once we score the thing the task is actually about — *is the predicted path a valid route
from start to goal?* — `z_H` turns out to be causally critical, just as in Sudoku.

The fix (now in `controlled_ablation.py` and `controlled_time_shift.py`, mirroring the
already-correct `controlled_freeze.py`) records `valid_sg_path`, `valid_optimal_path`,
`connects_start_goal`, `path_f1`, `exact_solved`, etc. alongside token accuracy. It is a
no-op for Sudoku (guarded on `labels.shape[-1] == 900`), so nothing in the Sudoku results
changed.

---

## 1. How each task is solved over ACT steps

**Maze — solved almost immediately, then flat** (`step_dynamics`, N=500):

| ACT step | exact | valid start→goal path | connects start↔goal |
|---:|---:|---:|---:|
| 0 | 0.166 | 0.234 | 0.350 |
| 1 | 0.720 | 0.880 | 0.892 |
| 2 | 0.754 | 0.924 | 0.934 |
| … | … | … | … |
| 15 | 0.750 | 0.930 | 0.944 |

A single recurrent step takes the maze from "mostly-right cells but no globally valid
route" (valid path 23%) to "globally valid route" (88%). Steps 2–15 add essentially
nothing. The maze is a **shallow** computation: one or two iterations suffice.

**Sudoku — refined gradually across all steps.** Accuracy climbs smoothly across the full
16-step budget; every additional step contributes. The Sudoku computation is **deep** and
iterative.

---

## 2. Is `z_H` causally important? Yes — in both tasks

**All-steps ablation (zero `z_H` at every step):**

| | Sudoku (N=5000) | Maze (N=200) |
|---|---:|---:|
| headline metric | exact/token **−20.5%** [−21.0, −20.0] | valid path **−32.5%** [−39.5, −26.0] |
| token accuracy | −20.5% | −0.8% (hidden!) |

`z_H` carries solution-bearing state in **both** tasks. The contrast the field expected
("`z_H` matters for Sudoku but not maze") does not survive a correct metric.

---

## 3. *Where* `z_H` matters differs sharply — the real contrast

Single-step ablation (zero `z_H` at exactly one step, leave the rest intact):

**Sudoku — DISTRIBUTED / accumulating** (token Δ, N=5000):

```
step  0: -7.8%   step  4: -6.0%   step  8: -8.7%
step 12: -14.5%  step 14: -21.0%  step 15: -27.6%
```

Every step's `z_H` matters, and the damage grows monotonically toward the end. `z_H` is a
**long-horizon working memory**: it accumulates the evolving solution, so corrupting it at
*any* step throws away real progress.

**Maze — CONCENTRATED at the readout** (valid-path Δ, N=200):

```
steps 0–13: ~-2%  (CIs cross 0 — individually dispensable)
step 14:    -6.5%
step 15:    -38.5%  [-46, -32]   ≈ the whole all-steps effect
```

Only the **final** `z_H` matters. Because the input maze is re-injected into `z_L` on every
inner step, the model can *rebuild* its high-level solution from scratch each step — so any
single intermediate `z_H` is disposable. What cannot be damaged is the **last** `z_H`,
because that is the tensor `lm_head` reads to produce the answer.

> **Mechanistic reading:** for the maze, `z_H` behaves as an **output/solution buffer**
> that holds the current answer; for Sudoku, `z_H` behaves as an **iterative scratchpad**
> that accumulates the answer over many steps. Same architecture, two different roles,
> selected by how much sequential computation the task actually requires.

---

## 4. `z_H` vs `z_L`: who carries the load

All-steps ablation, valid-path / exact metric:

| | `z_H` ablation | `z_L` ablation |
|---|---:|---:|
| **Maze** (valid path) | **−32.5%** | −8.5% |
| **Sudoku** (exact/token) | −20.5% | −19.3% |

In the **maze**, `z_H` dominates (`z_H` ≫ `z_L`): the high-level state is where the route
lives. In **Sudoku**, the two streams matter almost equally (`z_H` ≈ `z_L`): the
computation is genuinely two-stream and distributed.

---

## 5. Cross-puzzle activation patching  *(hardened re-run, N=100 pairs)*

Prediction from §3: patching a *foreign* puzzle's `z_H` at steps 4/8/12 should have a
**small** effect on the path, because the model re-derives `z_H` from the (unchanged)
re-injected input before readout. This is consistent with — not contradictory to — the
finding that `z_H` is causal at the final step. A direct test would patch at step 15.

That is exactly what the hardened re-run shows:

| patched stream | valid path Δ at step 4 | step 8 | step 12 | exact solved Δ |
|---|---:|---:|---:|---:|
| `z_H` | +0.0% [−4.0, +4.0] | +2.0% [−1.0, +6.0] | +2.0% [−1.0, +6.0] | −1% to −2% |
| `z_L` | −45.0% [−55.0, −35.0] | −46.0% [−56.0, −36.0] | −46.0% [−56.0, −36.0] | −74.0% [−82.0, −65.0] |

So the patching result is **asymmetric**: early/mid `z_H` is replaceable, while cross-puzzle
`z_L` at the same steps is highly disruptive. This matches the readout-buffer story:
foreign `z_H` at steps 4/8/12 is overwritten by later input-conditioned computation, but
foreign `z_L` interferes with the pathway that rebuilds the route representation.

The controlled time-shift re-run is also null on path validity (N=80, recipient step 2):
all donor→recipient transfers have valid-path Δ = 0.0% [0.0, 0.0]. This reinforces the
same point: intermediate maze states are easy to recover from when the original input is
still being re-injected.

## 6. Freeze  *(hardened re-run, N=100)*

Sudoku freeze (`z_H` held constant from step k onward) costs −9.1% at k=0, decaying to
−1.5% by k=5 — freezing early is expensive because it blocks accumulation. The maze
prediction: freezing should be cheap once the solution has appeared, since the path is
already present by step ~2.

The hardened maze freeze confirms this:

| freeze stream | k=0 valid path Δ | k=1 | k=2 | k≥4 |
|---|---:|---:|---:|---:|
| `z_H` | −4.0% [−9.0, +1.0] | 0.0% | +1.0% [0.0, +3.0] | 0.0% |
| `z_L` | −1.0% [−4.0, +2.0] | 0.0% | 0.0% | 0.0% |

Token accuracy is essentially unchanged throughout (largest `z_H` freeze Δ is −0.05% at
k=0). Freezing therefore behaves unlike Sudoku: once the maze route is available, the
future recurrent trajectory can be clamped without measurable path damage.

## 7. Linear probes  *(hardened re-run, 5 seeds — submitted)*

The hardened 5-seed maze probe rerun has been submitted as Slurm job `140089`, writing to
`results/maze/hardened/linear_probes`. The older single-seed probe run already showed that
`z_H` and `z_L` linearly decode on-optimal-path and wall structure at high accuracy, but it
lives under `results/maze/linear_probes` and should not be treated as the hardened result.

Decodability is *readout*, not *causation*; §2–6 supply the causal evidence.

---

## 8. The story in one paragraph

HRM solves a maze in essentially **one** recurrent step and then idles, whereas it solves
Sudoku by **iterating** for its full step budget. In both cases the high-level state `z_H`
is causally essential — destroying it wrecks the answer — but its *role* is task-dependent:
in Sudoku `z_H` is an iterative working memory whose contents accumulate across every step,
so ablating any step does graded damage that compounds toward the end; in the maze `z_H` is
an output buffer that holds the (quickly-found) route, so only the final step's `z_H` is
load-bearing while earlier copies are reconstructable from the re-injected input. The reason
this contrast was nearly missed is purely methodological: on a wall-dominated 900-cell grid,
token accuracy is blind to whether the path is valid, and it reported the maze ablation as a
harmless −0.8% when the path-validity damage was −32.5%. The same recurrent machinery adapts
its use of depth to the task — and you only see it if you measure the task's actual objective.
