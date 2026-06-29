# Implementation Plan — Experiments A & D: HRM recursion as a policy-improvement operator

**Audience:** an engineer/LLM implementing this in the HRM repo. Follow it literally; do
not redesign the interfaces named here — they are verified against the codebase.

## 0. Goal & scientific framing

Asadulaev et al. 2025 ("Your Latent Reasoning is Secretly a Policy Improvement Operator",
arXiv 2511.16886) argue **theoretically** that each recursive step of a TRM is a policy-
improvement operator: with a reference policy `π̂` (pre-reasoning logits) and an improved
policy `π⁺` (post-reasoning logits), the advantage `A(s,a)=log π⁺(a|s) − log π̂(a|s)` makes a
step "help" iff the ground-truth token beats the average advantage: `A(s,y*) > E_a[A(s,a)]`
(their Eq. 18). They prove this at **training time** (Prop. 4.1) and never measure it at
inference, and they remove "dead compute" via a training change (DIS) to cut forward passes
~18×.

We hold the complementary asset: a trained HRM + per-step intervention tooling. This plan
measures the policy-improvement signal **at inference**, per step, on Sudoku and Maze:

- **Experiment A — Policy-improvement curves.** Show HRM's per-step log-likelihood of the
  solution rises and test their Eq. 18 inequality per cell/step. This is the inference-time
  evidence they lack, and it upgrades our Finding 1 from "refinement happens" to "each step
  is a measurable policy-improvement operator."
- **Experiment D — Dead compute & early halting.** Use the same per-step signal to identify
  "dead" steps and measure the accuracy-vs-compute curve (the inference-time, no-retraining
  analog of their 18×). Cross-check against (i) our existing per-step ablation results and
  (ii) HRM's own ACT halting head.

**Expected contrast (our prior results):** Sudoku improves across all 16 steps (every step
alive); Maze is solved by ACT step ~1 then idles (steps ~2–15 are dead compute). A & D should
make this quantitative in the paper's policy-improvement language.

---

## 1. Ground-truth facts about the codebase (DO NOT re-derive; rely on these)

### 1.1 Model & per-step cache
- Load with: `from scripts.controlled.controlled_common import load_model_and_dataloader, collect_puzzles, bootstrap_ci, extract_batch`
  - `model, test_loader, config = load_model_and_dataloader(checkpoint_path, device)` → `model`
    is the **unwrapped** `HierarchicalReasoningModel_ACTV1` (has `.inner`, `.config`,
    `.initial_carry`).
  - `collect_puzzles(test_loader, device, num_puzzles, seed=42, puzzle_indices_path=None)`
    → `List[(puzzle_idx:int, batch:Dict[str,Tensor])]`. `batch` has `inputs`, `labels`,
    `puzzle_identifiers`, each `[1, seq_len]` already on `device`.
  - `bootstrap_ci(list_of_floats)` → `{"mean","ci_lower","ci_upper","std","n"}`.
- Forward+cache with: `from scripts.core.activation_ablation import ActivationAblator, ActivationCache`
  - `ablator = ActivationAblator(model, device=device)`
  - `cache: Dict[int, ActivationCache] = {}; ablator.run_and_cache_activations(batch, cache, max_steps=16)`
    runs ACT steps `0..max_steps-1` and fills `cache[s]` for each step `s`.
- **`ActivationCache` fields (per step `s`)** — all torch tensors unless noted:
  - `.logits`  → `[1, seq_len, vocab]`, **already answer-aligned** (the model does
    `lm_head(z_H)[:, puzzle_emb_len:]`, so the puzzle-embedding prefix is ALREADY stripped;
    `logits[0, j]` aligns 1:1 with `labels[0, j]`). May be bf16 → cast `.float()` before softmax.
  - `.preds` → `[1, seq_len]` = `logits.argmax(-1)`.
  - `.z_H`, `.z_L` → step **input** states `[1, puzzle_emb_len+seq_len, hidden]`.
  - `.z_H_out`, `.z_L_out` → step **output** states (same shape).
  - `.q_halt_logits`, `.q_continue_logits` → `[1]` scalars (ACT halting head).
  - `.step` → int.

### 1.2 Critical identity (why we do NOT need lm_head surgery)
In `run_and_cache_activations`, step `t`'s input state equals step `t−1`'s output state:
`cache[t].z_H == cache[t-1].z_H_out` (reset_carry is a no-op for non-halted steps). Since the
readout is `logits = lm_head(z_H_out)`, the paper's **within-step pre/post-reasoning policies
collapse to consecutive cached logits**:
```
π̂_t (pre-reasoning at step t)  = softmax(lm_head(z_H_in_t))  = softmax(cache[t-1].logits)  (t≥1)
π⁺_t (post-reasoning at step t) = softmax(lm_head(z_H_out_t)) = softmax(cache[t].logits)
```
So **all advantages for t≥1 come from cached logits alone**. The only exception is step 0,
whose reference is the learned init `H_init` (see §3.4, optional).

### 1.3 Checkpoints
- Maze:   `from scripts.maze.maze_common import MAZE_CHECKPOINT`  (= `checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint`)
- Sudoku: `from scripts.controlled.controlled_common import find_checkpoint`; `find_checkpoint()` → `checkpoints/sapientinc-sudoku-extreme/checkpoint.pt`

### 1.4 Per-step task "value" metric (task-specific)
- Maze: `from scripts.maze.maze_common import maze_prediction_metrics`
  `maze_prediction_metrics(pred_flat_900, label_flat_900, input_flat_900)` → dict incl.
  `valid_sg_path`, `exact_solved`, `token_acc` (all per-puzzle floats). Use **`valid_sg_path`**
  as the primary value (NOT token_acc — token_acc is the metric artifact we already fixed).
- Sudoku: `from scripts.core.activation_patching import compute_metrics`
  `compute_metrics(pred[1,seq], label[1,seq])` → `{"accuracy","correct","total_positions"}`.
  Value = `accuracy` (cell accuracy); `exact = (correct == total_positions)`.
- Labels use ignore index `-100`. Always mask `valid = labels != -100`.

### 1.5 Compute environment (HARD constraint)
- The login node has **no usable GPU**. All runs go through SLURM: `sbatch -p gpu --gres=gpu:1`,
  env `conda run --no-capture-output -n hrm python -u ...`. Model on `--device cuda`.
- CPU is too slow for the 900-token maze forward (a single full puzzle exceeds minutes). Do
  **not** attempt full runs on the login node; use a 1-puzzle/`max_steps=2` CPU smoke only to
  check imports + the write path, then submit the real job.
- Mirror the existing sbatch pattern in `scripts/maze/slurm_directed_ablation.sbatch`
  (`-N 1 -n 8 --mem=64G -t 02:00:00 -o logs/maze/%x_%j.out`). Per-user limits: keep `--mem ≤ 64G`
  and `-n ≤ 12`.

---

## 2. Files to create

1. `scripts/analysis/policy_improvement.py` — the compute script (one script, `--task {maze,sudoku}`).
2. `scripts/analysis/plot_policy_improvement.py` — figures from the JSON outputs.
3. `scripts/analysis/slurm_policy_improvement.sbatch` — runs both tasks.

Outputs:
- `results/<task>/policy_improvement/aggregate.json`
- `results/<task>/policy_improvement/per_puzzle.jsonl`
- `results/<task>/policy_improvement/_meta.json` (via `scripts.core.provenance.write_meta` if available; else skip in a try/except)
- Figures into `results/reports/policy_improvement_figures/`.
Here `<task>` ∈ {`maze`, `sudoku`} → use `results/maze/...` and `results/controlled/...` respectively.

---

## 3. Per-step quantities (the math) — compute these per puzzle

For a puzzle with steps `s = 0..T-1` (T=16), let `valid = (labels[0] != -100)` (a boolean over
`seq_len` cells), `y = labels[0]` (the solution token per cell). For each step `s`:

```
L_s   = cache[s].logits[0].float()         # [seq_len, vocab]
logp_s = log_softmax(L_s, dim=-1)          # [seq_len, vocab]
p_s    = softmax(L_s, dim=-1)              # [seq_len, vocab]
```

### 3.1 Value of the solution (the "did it improve" signal)
- `logp_true_s` = mean over `valid` cells of `logp_s[j, y[j]]`.  (mean log-likelihood of y*)
- `p_true_s`    = mean over `valid` cells of `p_s[j, y[j]]`.       (mean prob of y*)
- `value_s`     = task value at step s (maze `valid_sg_path` / sudoku cell `accuracy`), computed
  from `cache[s].preds`.  `exact_s` likewise.

### 3.2 Per-step advantage (their A_t) — for s ≥ 1
Reference `π̂ = p_{s-1}` (= `softmax(cache[s-1].logits)`), improved `π⁺ = p_s`.
```
A_s(j, a)      = logp_s[j, a] - logp_{s-1}[j, a]               # [seq_len, vocab]
adv_true_s     = mean_valid( A_s[j, y[j]] )                    # advantage on the ground-truth token
# Expectation of advantage under the reference policy π̂ (matches their derivation):
E_adv_s(j)     = sum_a p_{s-1}[j, a] * A_s[j, a]   = -KL(p_{s-1} || p_s)[j]   # ≤ 0, per cell
frac_eq18_s    = mean_valid( 1[ A_s[j, y[j]] > E_adv_s[j] ] )  # THE Eq.18 test (fraction of cells)
kl_prev_s      = mean_valid( KL(p_{s-1}[j] || p_s[j]) )        # = -mean(E_adv_s); "compute done this step"
```
Notes:
- `KL(p_{s-1}||p_s)[j] = sum_a p_{s-1}[j,a]*(logp_{s-1}[j,a]-logp_s[j,a])`. So `E_adv_s = -KL`.
  Compute `kl_prev_s` directly and set `E_adv_s = -kl` per cell (cheaper, identical).
- Use a numerically safe softmax/log_softmax (torch's are fine). Everything in float32.

### 3.3 Sharpening / margin (optional, cheap, nice for plots)
- `margin_s` = mean over valid cells of `(top1 logit − top2 logit)` of `L_s[j]` (confidence).

### 3.4 Step-0 advantage (OPTIONAL — the only place lm_head is needed)
Reference for step 0 is the learned init `H_init`, not a previous step. To get π̂_0:
```
pel   = model.inner.puzzle_emb_len
zHin0 = cache[0].z_H                                   # [1, pel+seq_len, hidden]
L0_pre = model.inner.lm_head(zHin0)[:, pel:].float()  # strip prefix -> [1, seq_len, vocab]
```
Then `A_0` uses `π̂ = softmax(L0_pre[0])`, `π⁺ = p_0`. If this is fiddly, SKIP step 0 advantage
(report advantages for s≥1 only); `logp_true_0` still anchors the value curve. Mark in output
whether step-0 advantage was computed.

---

## 4. Experiment A — Policy-improvement curves (what to record)

Per puzzle, store a `per_step` list of dicts: `{step, logp_true, p_true, value, exact,
adv_true, frac_eq18, kl_prev, margin, q_halt, q_continue}` (adv/frac/kl absent or null at s=0
unless §3.4 done).

Aggregate (`aggregate.json`): for each step, `bootstrap_ci` over puzzles of every scalar above.
Also aggregate **separately for solved vs failed puzzles** (split by final `exact`/`valid_sg_path`):
their guarantee should hold strongly on solved puzzles; failed puzzles may show flat/negative
advantage — this ties to our PCA "convergence vs wandering" result.

**Claims A should let us make (acceptance-style):**
- A1: `logp_true_s` (and `value_s`) increase across steps on Sudoku at every step; on Maze they
  jump by step ~1 then plateau.
- A2: `frac_eq18_s` (fraction of cells satisfying Eq. 18) is high (≫ chance) during the "alive"
  phase — direct inference-time confirmation of their training-time guarantee.
- A3: `kl_prev_s` is large early then → 0 (Maze fast, Sudoku gradual) — per-step "compute done".

---

## 5. Experiment D — Dead compute & early halting (what to record)

All three parts reuse the **same single cache per puzzle** (no extra forward passes).

### 5.1 D1 — Dead-step identification
A step `s≥1` is "alive" if it materially improves the solution. Record per puzzle:
- `d_logp_s = logp_true_s − logp_true_{s-1}` (per-step value gain), and `kl_prev_s` (§3.2).
- `n_alive(eps)` = number of steps with `d_logp_s > eps` (default `eps = 0.01`); also a KL-based
  variant `kl_prev_s > eps_kl` (default `eps_kl = 1e-3`). Report both; they should agree.
Aggregate: mean `n_alive` per task. Expectation: Maze ≈ 1–2 alive; Sudoku ≈ all.
**Triangulation (report in the doc, no new compute):** compare `n_alive` to our existing
per-step **ablation** nulls (`results/maze/hardened/ablation_controlled/zH` shows maze steps
0–13 ≈ 0 effect = dead; `results/controlled/ablation/zH` shows Sudoku every step matters). A and
the ablation should label the same steps dead/alive — forward-side and causal-side agreement.

### 5.2 D2 — Early-halt accuracy-vs-compute curve (the inference-time "18×")
Because each step's readout is cached, "halt after step s" prediction = `cache[s].preds`.
Per puzzle, for s = 0..T-1 compute `value_at_halt_s = value(cache[s].preds)`.
Aggregate `value_at_halt_s` per step (bootstrap_ci). Then define, per task, the **compute
needed for near-final quality**:
```
final = value_at_halt_{T-1}
s_star(tau) = min s such that value_at_halt_s >= final - tau     # tau default 0.01 (1 point)
act_step_budget = s_star + 1                                     # ACT steps actually needed
speedup = T / (s_star + 1)
```
Report `s_star` and `speedup` for `tau ∈ {0.0, 0.01, 0.02}`, both as the **population mean of
per-puzzle s_star** and as the **s_star of the population mean curve** (report both; they
differ). Expectation: Maze `s_star≈0–1` (speedup ~8–16×), Sudoku `s_star≈14–15` (speedup ~1×).
This is the inference-time, no-retraining analog of their 18×, and it shows the speedup is
**task-dependent** (their headline number is task-specific too — they note Sudoku resists DIS).

### 5.3 D3 — Does HRM's ACT halting already "know" the dead compute?
From cached `q_halt_logits`/`q_continue_logits`, the step ACT *would* halt at is
`s_act = min s such that q_halt_logits_s > q_continue_logits_s` (or `T-1` if never). Per puzzle
record `s_act`. Aggregate the distribution of `s_act` per task and compare to `s_star` (D2):
- If `s_act ≈ s_star`: HRM's learned halting already tracks policy-improvement saturation.
- If `s_act ≫ s_star` (esp. Maze): HRM wastes compute its own halting could cut — exactly the
  "dead compute" their DIS removes, observable in our model without retraining.

---

## 6. Main script skeleton (`scripts/analysis/policy_improvement.py`)

Mirror `scripts/maze/eval_step_dynamics_maze.py` structure. Critical, error-prone pieces shown.

```python
import os, sys, json, time, argparse
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
import numpy as np, torch
import torch.nn.functional as F
from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles, bootstrap_ci, find_checkpoint)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.core.activation_patching import compute_metrics
from scripts.maze.maze_common import MAZE_CHECKPOINT, maze_prediction_metrics

def task_value(preds_row, label_row, input_row, task):
    """preds_row/label_row/input_row: 1-D int tensors/arrays of length seq_len."""
    if task == "maze":
        m = maze_prediction_metrics(preds_row, label_row, input_row)
        return {"value": m["valid_sg_path"], "exact": m["exact_solved"], "token_acc": m["token_acc"]}
    # sudoku
    p = preds_row.reshape(1, -1); y = label_row.reshape(1, -1)
    cm = compute_metrics(torch.as_tensor(p), torch.as_tensor(y))
    return {"value": cm["accuracy"], "exact": float(cm["correct"] == cm["total_positions"]),
            "token_acc": cm["accuracy"]}

def per_step_policy_stats(cache, labels_row, inputs_row, task, compute_step0_adv, model):
    steps = sorted(cache.keys())
    y = labels_row.long()                          # [seq_len]
    valid = (y != -100)
    yv = y.clamp(min=0)                             # safe gather index; only read at valid cells
    # precompute logp/p per step
    logps, ps, preds = {}, {}, {}
    for s in steps:
        L = cache[s].logits[0].float()             # [seq_len, vocab]
        logps[s] = F.log_softmax(L, dim=-1)
        ps[s]    = logps[s].exp()
        preds[s] = cache[s].preds[0].long()
    rows = []
    for s in steps:
        logp_true = logps[s].gather(-1, yv[:, None]).squeeze(-1)      # [seq_len]
        tv = task_value(preds[s].cpu().numpy(), labels_row.cpu().numpy(),
                        None if inputs_row is None else inputs_row.cpu().numpy(), task)
        row = {"step": s,
               "logp_true": float(logp_true[valid].mean()),
               "p_true": float(ps[s].gather(-1, yv[:,None]).squeeze(-1)[valid].mean()),
               "value": tv["value"], "exact": tv["exact"], "token_acc": tv["token_acc"],
               "q_halt": float(cache[s].q_halt_logits.item()),
               "q_continue": float(cache[s].q_continue_logits.item())}
        ref = None
        if s >= 1:
            ref = (logps[s-1], ps[s-1])
        elif compute_step0_adv:
            pel = model.inner.puzzle_emb_len
            L0pre = model.inner.lm_head(cache[0].z_H)[:, pel:].float()[0]
            ref = (F.log_softmax(L0pre, -1), F.softmax(L0pre, -1))
        if ref is not None:
            logp_ref, p_ref = ref
            A = logps[s] - logp_ref                                   # [seq_len, vocab]
            kl = (p_ref * (logp_ref - logps[s])).sum(-1)              # KL(ref||cur) per cell, ≥0
            A_true = A.gather(-1, yv[:,None]).squeeze(-1)             # advantage on y*
            E_adv = -kl                                              # E_{a~ref}[A] per cell
            row.update({
                "adv_true": float(A_true[valid].mean()),
                "kl_prev": float(kl[valid].mean()),
                "frac_eq18": float((A_true[valid] > E_adv[valid]).float().mean()),
            })
        rows.append(row)
    return rows
```

Driver: load model+loader (task-dispatched checkpoint), `collect_puzzles`, loop puzzles,
`ablator.run_and_cache_activations(batch, cache, max_steps=args.max_steps)`, build rows, write
`per_puzzle.jsonl`. Then aggregate per step with `bootstrap_ci` over puzzles (overall + solved
+ failed splits), compute D2 `value_at_halt` curve / `s_star` / `speedup` and D3 `s_act`
distribution, write `aggregate.json`. Include args: `--task`, `--checkpoint` (default per task),
`--num_puzzles` (default 500), `--max_steps 16`, `--device cuda`, `--output_dir`,
`--compute_step0_adv` (default True), `--eps 0.01`, `--tau 0.01`.

**Inputs row for maze value:** pass `batch["inputs"][0]` (length seq_len; maze metrics need it
for `valid_sg_path`). For sudoku pass `None`.

---

## 7. Plot script (`scripts/analysis/plot_policy_improvement.py`)

Read both `results/maze/policy_improvement/aggregate.json` and
`results/controlled/policy_improvement/aggregate.json`. Use matplotlib (Agg backend). Echo every
plotted value to stdout (accuracy discipline — mirror `scripts/maze/plot_consolidated_figures.py`).
Figures → `results/reports/policy_improvement_figures/`:

- `figA1_value_logp.{pdf,png}` — twin panel: per-step `logp_true` (left) and `value` (right),
  Sudoku vs Maze. Shows rising vs saturating.
- `figA2_eq18.{pdf,png}` — per-step `frac_eq18` with CI, both tasks + a 0.5 chance line. Title:
  "Fraction of cells satisfying the policy-improvement inequality (Asadulaev Eq. 18)".
- `figA3_kl.{pdf,png}` — per-step `kl_prev` (compute done per step), both tasks (Maze spikes at
  step 1 then ~0; Sudoku spread out).
- `figD2_earlyhalt.{pdf,png}` — `value_at_halt_s` vs step (ACT-step budget), both tasks, with a
  vertical marker at `s_star(0.01)` and the implied `speedup` annotated.
- `figD3_halting.{pdf,png}` — histogram/box of `s_act` (ACT's own halt step) vs `s_star`, per task.

---

## 8. SLURM (`scripts/analysis/slurm_policy_improvement.sbatch`)

```bash
#!/usr/bin/env bash
#SBATCH -J hrm_polimp
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs
N="${N:-500}"
conda run --no-capture-output -n hrm python -u scripts/analysis/policy_improvement.py \
  --task maze   --num_puzzles "$N" --output_dir results/maze/policy_improvement      --device cuda
conda run --no-capture-output -n hrm python -u scripts/analysis/policy_improvement.py \
  --task sudoku --num_puzzles "$N" --output_dir results/controlled/policy_improvement --device cuda
```
Submit: `sbatch -p gpu --gres=gpu:1 scripts/analysis/slurm_policy_improvement.sbatch`.

---

## 9. Validation, smoke test, acceptance criteria

1. **Import + arg check (login node, CPU):**
   `conda run -n hrm python scripts/analysis/policy_improvement.py --help` (use `>|` to redirect;
   the profile sets `noclobber`).
2. **CPU smoke (tiny):** `--task maze --num_puzzles 1 --max_steps 2 --device cpu
   --output_dir /tmp/pi_smoke`. It will be slow but must produce a valid `aggregate.json` with
   the per-step keys. (If it exceeds ~4 min, kill it; the import + first-step write is enough to
   verify the code path — submit the GPU job for real numbers.)
3. **Submit GPU job;** confirm `aggregate.json` + `per_puzzle.jsonl` written and logs end with a
   "done" line.
4. **Sanity checks on results (must hold or investigate):**
   - `p_true` ∈ [0,1], `frac_eq18` ∈ [0,1]; `kl_prev ≥ 0`; `logp_true ≤ 0`.
   - `value` at the final step ≈ our known baselines: Maze `valid_sg_path ≈ 0.93`
     (`results/maze/hardened/step_dynamics/aggregate.json`), Sudoku cell-acc ≈ 0.82.
   - `value_at_halt` is monotone-ish increasing; `value_at_halt_{T-1} == value_{T-1}`.
   - Maze: `n_alive` small (≈1–2), `s_star(0.01)` small; Sudoku: `n_alive` large, `s_star` large.
   - `frac_eq18` during the alive phase should be clearly > 0.5 on solved puzzles.
5. **Cross-check (write one paragraph in the results README):** the steps A/D label "dead" must
   match the steps our per-step **ablation** finds null (maze 0–13; sudoku none). If they
   disagree, something is wrong — stop and report.

---

## 10. Gotchas / accuracy caveats (read before coding)

- **Do NOT slice puzzle_emb off `logits`/`preds`** — already stripped (`lm_head(z_H)[:, pel:]`).
  Only `z_H`/`z_L` carry the prefix; that matters only for the optional §3.4 step-0 lm_head call.
- **Cast logits to float32** before softmax (checkpoints may be bf16).
- **Mask `labels == -100`** everywhere a per-cell mean/flag is taken; never include ignore cells.
- **Maze value must be `valid_sg_path`, not `token_acc`** (token_acc is the metric artifact).
- **Maze `maze_prediction_metrics` needs `inputs`** (for connectivity); Sudoku does not.
- **`s_star` two ways**: per-puzzle-then-average vs from the mean curve — report both; they
  differ because min() is nonlinear.
- **Failed puzzles**: report solved/failed splits; the Eq. 18 guarantee is expected to hold on
  solved puzzles and may fail on the "wandering" ones (consistent with our PCA finding).
- **One forward pass per puzzle** serves A and D; do not add extra passes for the early-halt
  curve (read it from the cache).
- **Determinism**: pass a fixed `--num_puzzles`; `collect_puzzles` is deterministic in loader
  order. (Optional: reuse a shared puzzle-index manifest if you want cell-for-cell comparability
  with other experiments — not required here.)
- **Do not retrain anything.** This is eval-only. (Their DIS is a training change; out of scope.)

---

## 11. What to hand back

- The 3 new files, the 2 `aggregate.json`s, the 5 figures.
- A short `results/reports/policy_improvement_README.md` stating, with numbers + CIs: per-step
  `frac_eq18` (A2), the Maze-vs-Sudoku `n_alive` and `s_star`/`speedup` (D1/D2), the `s_act`
  vs `s_star` comparison (D3), and the one-paragraph triangulation against the existing
  per-step ablation results.
```
