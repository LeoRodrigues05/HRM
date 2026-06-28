# Plan — Path A: Recover a solving ARC‑2 model by re‑fitting `puzzle_emb`

**Audience:** an implementing LLM/engineer with repo access. This is fully
self‑contained. Follow it top to bottom. Do **not** improvise around the
"Invariants" — they are the correctness boundary.

> **Status — all code applied, run the wrapper:**
> - `pretrain.py` has `load_checkpoint` + core‑only loading in `init_train_state`.
> - `config/cfg_pretrain.yaml` has the required null stubs for Hydra overrides
>   (`checkpoint_path`, `project_name`, `run_name`, `load_checkpoint`).
> - The eval‑only raw symlink is **created** (`data/raw-arc2-evalonly/evaluation`).
> - A one‑shot wrapper exists: **`scripts/arc/adapt_puzzle_emb.sh`**.
>
> **Quick‑start on a fresh GPU node:**
> ```bash
> cd /home/leo.rodrigues/HRM
>
> # Recommended: GBS=384 for A100/H100 (~4× fewer steps → ~1h total)
> GBS=384 bash scripts/arc/adapt_puzzle_emb.sh evalonly
>
> # On RTX 5000 Ada (32 GB) GBS=96 is safe (~3.5h total):
> bash scripts/arc/adapt_puzzle_emb.sh evalonly
>
> # If exact_solved still rising when done, double the budget:
> EPOCHS=4000 bash scripts/arc/adapt_puzzle_emb.sh evalonly
> ```
> The wrapper: builds the dataset if missing → trains `puzzle_emb` with frozen
> core → verifies with `measure_arc_accuracy.py`. It skips the dataset build if
> `data/arc-2-evalonly/` already exists, so re‑runs are cheap.
>
> **Known gotchas already fixed in this codebase (do not re‑introduce):**
> - `adam_atan2_pytorch` asserts `lr > 0` → use `lr=1e-9` not `lr=0`.
> - The training‑loop eval iterates ALL ~165 k augmented test examples × 16 halt
>   steps ≈ 3 h per cycle; wrapper sets `EVAL_INT=EPOCHS` to skip intermediate
>   evals — `measure_arc_accuracy.py` does a fast 100‑puzzle spot‑check instead.
> - `config/cfg_pretrain.yaml` must have null stubs for every key you override
>   from the CLI; missing keys trigger a Hydra struct error.

---

## 0. Background (read once)

HRM's ARC model is **transductive**: the input grid alone does not specify the
task; the transformation rule lives in a **learned per‑puzzle embedding**
(`puzzle_emb`). At eval, each test grid looks up its puzzle's embedding by an
integer **identifier index** assigned at dataset‑build time.

Problem we are solving: the published checkpoint
(`checkpoints/sapientinc-hrm-arc-2/checkpoint`, `puzzle_emb` = **1,045,829** rows)
was trained against one specific dataset build. Rebuilding
`data/arc-2-aug-1000` on a different machine yields a slightly different
identifier set (we get 1,045,833–1,045,838; off by 4–9) because the augmentation
unique‑count of a few borderline small‑grid puzzles depends on file‑traversal
order/RNG. The result: the published embeddings are misaligned to our build, and
measured **`exact_solved = 0/100`** even though the trained reasoning core loads
fine (`token_acc ≈ 0.43`). We cannot bit‑reproduce the authors' build, and the HF
repo ships no dataset/identifier mapping.

**The fix (Path A):** keep the checkpoint's **reasoning core frozen**, throw away
its `puzzle_emb`, and **re‑learn `puzzle_emb` on OUR build**. This is exactly the
transductive mechanism HRM is designed around. The repo already uses **two
separate optimizers** — a sparse SignSGD optimizer for `puzzle_emb` and AdamAtan2
for everything else — so freezing the core is just `lr=0` on the core optimizer
while `puzzle_emb_lr > 0`.

**Why it must work in principle:** within a *single consistent build*, an eval
task's demo augmentations (in the `train` split) and its test augmentations (in
the `test` split) share the *same* identifier strings, hence the same embedding
rows. So training embeddings on the demos (train split) makes them available to
the test queries (test split) with the correct index — no dependence on the
authors' build at all.

---

## 1. Invariants (do not violate)

1. **Never load the checkpoint's `puzzle_emb.weights`.** It is the wrong size and
   wrong indexing. Drop it; keep the model's fresh, trainable embedding table.
2. **Effectively freeze the core via `lr=1e-9` (not `lr=0`).** `adam_atan2_pytorch`
   asserts `lr > 0`; `1e-9` is numerically equivalent to frozen. `puzzle_emb` is
   trained by its own sparse SignSGD optimizer at `puzzle_emb_lr`.
3. **Use the SAME dataset for adaptation and for the analysis suite.** Do not
   rebuild between them. Identifier indices must stay constant.
4. **The arch config must match the checkpoint.** `config/arch/hrm_v1.yaml`
   already matches the ARC‑2 checkpoint (H_cycles=2, L_cycles=2, H_layers=4,
   L_layers=4, hidden_size=512, num_heads=8, halt_max_steps=16,
   puzzle_emb_ndim=512, pos_encodings=rope, loss=stablemax_cross_entropy). Do not
   change it.
5. **Do not touch** `utils/arc_targets.py`, the `scripts/arc/*` analysis scripts,
   or the `arc_prediction_metrics`. They are already correct.

---

## 2. Step 1 — Add init‑checkpoint loading to `pretrain.py`

`pretrain.py` currently always starts from random init. Add the ability to load a
checkpoint's **core** weights.

### 2a. Add a config field

In `pretrain.py`, in the `PretrainConfig` dataclass (around line 64, next to
`checkpoint_path: Optional[str] = None`), add:

```python
    load_checkpoint: Optional[str] = None  # init core weights from this checkpoint (puzzle_emb is re-learned)
```

### 2b. Load core weights in `init_train_state`

In `init_train_state` (around line 172), **after**
`model, optimizers, optimizer_lrs = create_model(...)` and **before** the
`return TrainState(...)`, insert:

```python
    # Path A: initialise the reasoning core from a pretrained checkpoint, but KEEP
    # the freshly-initialised puzzle_emb table (we re-learn embeddings for THIS
    # dataset's identifier indexing on the frozen core).
    if config.load_checkpoint is not None:
        sd = torch.load(config.load_checkpoint, map_location="cuda", weights_only=False)
        model_keys = list(model.state_dict().keys())
        m_has = any(k.startswith("_orig_mod.") for k in model_keys)
        c_has = any(k.startswith("_orig_mod.") for k in sd)
        if m_has and not c_has:
            sd = {f"_orig_mod.{k}": v for k, v in sd.items()}
        elif c_has and not m_has:
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        # Drop the checkpoint's puzzle_emb so our fresh, trainable table is kept.
        sd = {k: v for k, v in sd.items() if not k.endswith("puzzle_emb.weights")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        miss_non_pe = [k for k in missing if "puzzle_emb" not in k]
        print(f"[load_checkpoint] init core from {config.load_checkpoint}: "
              f"loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[load_checkpoint] non-puzzle_emb missing (should be empty): {miss_non_pe}")
        assert not unexpected, f"unexpected checkpoint keys: {unexpected[:8]}"
        assert not miss_non_pe, f"core weights failed to load: {miss_non_pe[:8]}"
```

**Expected when it runs:** `unexpected = 0`, and the only `missing` keys contain
`puzzle_emb` (the fresh table + its two non‑persistent buffers `local_weights`,
`local_ids`). If `non-puzzle_emb missing` is non‑empty, the arch does not match —
stop and fix the arch before continuing.

(`torch` is already imported in `pretrain.py`.)

---

## 3. Step 2 — Dataset

Two options. **Use 3a (eval‑only) — it is ~8–10× cheaper and sufficient**, because
the interpretability suite only analyses the 120 evaluation tasks. Use 3b only if
you specifically want the model to also carry the 1000 training tasks.

### 3a. (Recommended) Build an eval‑only dataset

Only the 120 evaluation tasks; every training example is then a relevant eval
task, so embedding adaptation converges far faster.

```bash
cd /home/leo.rodrigues/HRM
mkdir -p data/raw-arc2-evalonly
ln -sfn "$PWD/dataset/raw-data/ARC-AGI-2/data/evaluation" data/raw-arc2-evalonly/evaluation
PYTHONPATH=$PWD:$PWD/dataset /home/leo.rodrigues/miniconda3/envs/hrm/bin/python -u -c "
import sys; sys.path.insert(0,'dataset')
import build_arc_dataset as b
b.convert_dataset(b.DataProcessConfig(
    dataset_dirs=['data/raw-arc2-evalonly'],
    output_dir='data/arc-2-evalonly', seed=42, num_aug=1000))
print('BUILD DONE')
"
```

The builder is already memory‑lean (streams to disk; peak < ~15 GB, fits the
32 GB SLURM cap) and takes ~5–7 min. `DATA=data/arc-2-evalonly` below.

> Subdir naming matters: the builder routes the subdir literally named
> `evaluation` to the `test` split (its demos also go to `train`). Keep that name.

### 3b. (Alternative, faithful/full) Use the full build

`data/arc-2-aug-1000` may already exist from earlier. If not, build it:

```bash
PYTHONPATH=$PWD:$PWD/dataset /home/leo.rodrigues/miniconda3/envs/hrm/bin/python -u -c "
import sys; sys.path.insert(0,'dataset')
import build_arc_dataset as b
b.convert_dataset(b.DataProcessConfig(
    dataset_dirs=['dataset/raw-data/ARC-AGI-2/data'],
    output_dir='data/arc-2-aug-1000', seed=42, num_aug=1000))
print('BUILD DONE')
"
```

`DATA=data/arc-2-aug-1000`. This trains embeddings for all 1120 tasks → ~10×
more steps to reach the same eval accuracy.

---

## 4. Step 3 — Run the adaptation (frozen core, train `puzzle_emb`)

Single GPU, no `torchrun` needed. Set `lr=0` (freeze core) and keep
`puzzle_emb_lr=1e-2` (its schedule is constant: `lr_min_ratio=1.0`).

```bash
cd /home/leo.rodrigues/HRM
PY=/home/leo.rodrigues/miniconda3/envs/hrm/bin/python
DATA=data/arc-2-evalonly                       # from Step 3a (or data/arc-2-aug-1000)
CKPT=checkpoints/sapientinc-hrm-arc-2/checkpoint
OUT=checkpoints/arc2-adapted

WANDB_MODE=offline OMP_NUM_THREADS=8 PYTHONPATH=$PWD $PY pretrain.py \
    data_path=$DATA \
    load_checkpoint=$CKPT \
    checkpoint_path=$OUT \
    lr=1e-9 \
    puzzle_emb_lr=1e-2 \
    puzzle_emb_weight_decay=0.1 \
    lr_warmup_steps=200 \
    global_batch_size=96 \
    epochs=2000 \
    eval_interval=200 \
    checkpoint_every_eval=True \
    project_name=arc2_adapt run_name=run1
```

### Sizing / monitoring (important)

- `tqdm` prints `total_steps` at start: `total_steps ≈ epochs * train_groups *
  mean_puzzle_examples / global_batch_size`. Treat **steps**, not epochs, as the
  budget. `epochs` here is a misnomer (one "epoch" ≈ `train_groups * mean` ≈ a few
  thousand examples, a small fraction of the data).
- **Watch `eval/exact_accuracy`** each eval cycle. With `WANDB_MODE=offline` it is
  logged locally; the simplest reliable signal is to run the verifier (Step 4) on
  the latest saved checkpoint while/after training (`checkpoint_every_eval=True`
  writes `step_<N>` files in `$OUT`).
- **Budget expectation:** embeddings need multiple updates per identifier.
  Start with the command above; if `exact_solved` is still rising, increase
  `epochs` (e.g. 2000 → 8000). Eval‑only typically needs far fewer steps than the
  full set. Expect this to be a real run (tens of minutes to a few hours), but far
  cheaper than the 24 h full train.
- **GPU memory:** `global_batch_size=96` is a safe single‑GPU start. Raise toward
  192/384 if VRAM allows (faster); lower to 48 if you OOM. (This is GPU VRAM, not
  the 32 GB host‑RAM cap.)
- **Optional speedup:** keep `torch.compile` on (default). The loader in Step 2b
  handles the `_orig_mod.` prefix either way. Only set `DISABLE_COMPILE=1` if
  compilation causes trouble.

Output: `checkpoints/arc2-adapted/step_<N>` (model state_dict) plus
`all_config.yaml` (written by `pretrain.py`). The latest `step_<N>` is the adapted
checkpoint.

---

## 5. Step 4 — Verify accuracy recovered

Point the existing verifier at the adapted checkpoint. (It reads `all_config.yaml`
next to the checkpoint to get arch + `data_path`; `pretrain.py` saved it there.)

```bash
ADAPTED=$(ls -t checkpoints/arc2-adapted/step_* | head -1)
PYTHONPATH=$PWD $PY scripts/arc/measure_arc_accuracy.py \
    --checkpoint "$ADAPTED" --num_puzzles 100 --device cuda
```

**Decision gate:**
- `exact_solved` clearly **> 0** (ideally tens of %, plus high `colour_cell_acc`,
  `shape_correct`) → success. Proceed to Step 5.
- Still ~0 after a generous budget → see Step 7 fallback.

> Single‑shot per‑augmentation exact‑match is naturally below the published 40.3 %
> (that number uses test‑time‑augmentation voting in `arc_eval.ipynb`). A healthy
> result is a clear jump from 0 to a double‑digit percentage, with `shape_correct`
> and `colour_cell_acc` rising together.

---

## 6. Step 5 — Wire the adapted checkpoint into the suite

The analysis scripts default `--checkpoint` to `arc_common.ARC_CHECKPOINT`. Make
that point at the adapted checkpoint so the whole suite uses the solving model.

Edit `scripts/arc/arc_common.py`:

```python
# was: points at the published (misaligned) checkpoint
# ARC_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "sapientinc-hrm-arc-2", "checkpoint")
ARC_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "arc2-adapted", "step_<N>")  # adapted; set <N> to the verified step
```

Then run the suite (it will use `$DATA` via the adapted checkpoint's
`all_config.yaml`):

```bash
bash scripts/arc/run_arc_end_to_end.sh full     # build step is skipped (dataset exists)
```

If you used the eval‑only dataset, the suite analyses the 120 evaluation tasks —
exactly the set we want to interpret.

---

## 7. Pitfalls & fallbacks

**Pitfalls**
- Forgetting to drop `puzzle_emb.weights` from the checkpoint → size‑mismatch
  `RuntimeError`. (Invariant 1.)
- `lr` not 0 → you fine‑tune the core, defeating "use only the checkpoint" and
  risking drift. Confirm the printed core lr is 0.
- Rebuilding the dataset between adaptation and analysis → identifier indices
  change, embeddings misalign again. (Invariant 3.) Build once, reuse.
- Arch mismatch → `non-puzzle_emb missing` non‑empty in Step 2b output. Stop and
  reconcile arch with the checkpoint's `all_config.yaml`.
- `wandb` prompting/login: `WANDB_MODE=offline` (or `disabled`) avoids it.

**Fallbacks if Step 4 stays ~0**
1. Train longer (more `epochs`) and/or raise `puzzle_emb_lr` to `2e-2`.
2. Confirm the verifier isn't the bottleneck: dump one prediction grid vs label
   (a non‑solving model still produces a structured but wrong grid; a pipeline bug
   produces constant/empty output).
3. As a last resort, unfreeze the core *slightly* (`lr=1e-5`) for a short run —
   this departs from "checkpoint‑only" but can recover solving if the frozen core
   is too rigid. Document if used.
4. If solving cannot be recovered, fall back to **Path B** (input‑feature probes
   only: color/shape/object/geometry decodability are valid without solving;
   drop per‑cell‑correct / policy‑value / solving‑damage results).

---

## 8. Quick reference

| Item | Value |
|------|-------|
| Published checkpoint (core source) | `checkpoints/sapientinc-hrm-arc-2/checkpoint` |
| Arch config (already matches) | `config/arch/hrm_v1.yaml` |
| Dataset (recommended) | `data/arc-2-evalonly` (Step 3a) |
| Dataset (full alt) | `data/arc-2-aug-1000` (Step 3b) |
| Adapted checkpoint out | `checkpoints/arc2-adapted/step_<N>` |
| Python | `/home/leo.rodrigues/miniconda3/envs/hrm/bin/python` (conda env `hrm`) |
| Code change | `pretrain.py`: add `load_checkpoint` field + core‑only load in `init_train_state` |
| Freeze knobs | `lr=0`, `puzzle_emb_lr=1e-2` |
| Verify | `scripts/arc/measure_arc_accuracy.py --checkpoint <adapted>` |
| Suite entry | `scripts/arc/run_arc_end_to_end.sh full` (after Step 5 edit) |

**One‑line summary of the method:** load the checkpoint's reasoning core, freeze
it (`lr=0`), discard its `puzzle_emb`, and re‑learn `puzzle_emb` on our own build
(`puzzle_emb_lr>0`) so the embeddings align with our identifier indexing — making
the model solve ARC again using only the released checkpoint.
