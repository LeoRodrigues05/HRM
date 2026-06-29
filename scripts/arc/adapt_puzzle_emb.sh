#!/usr/bin/env bash
# Path A — recover a solving ARC-2 model from the released checkpoint by re-fitting
# puzzle_emb on OUR dataset build, with the reasoning core FROZEN (lr=0).
# See docs/PLAN_ARC_path_A_embedding_adaptation.md for the full rationale.
#
# Run ON A GPU NODE:
#   bash scripts/arc/adapt_puzzle_emb.sh evalonly      # 120 eval tasks (recommended, cheap)
#   bash scripts/arc/adapt_puzzle_emb.sh full           # all 1120 tasks (faithful, ~10x slower)
#
# Tunables (env): OUT, GBS (global_batch_size), EPOCHS, EVAL_INT, PELR (puzzle_emb_lr)
# Steps budget is empirical: watch the verifier's exact_solved; raise EPOCHS if still rising.
set -uo pipefail

cd /home/leo.rodrigues/HRM
PY=/home/leo.rodrigues/miniconda3/envs/hrm/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Skip pretrain.py's built-in eval. That eval walks ALL ~165k augmented test
# examples × halt_max_steps forward passes = HOURS, and (before the fix) it gated
# the checkpoint save. We verify with measure_arc_accuracy.py (100-puzzle spot
# check) instead. pretrain.py now saves the checkpoint BEFORE eval regardless.
export HRM_SKIP_EVAL="${HRM_SKIP_EVAL:-1}"

MODE="${1:-evalonly}"
CKPT=checkpoints/sapientinc-hrm-arc-2/checkpoint
OUT="${OUT:-checkpoints/arc2-adapted-$MODE}"
# With eval skipped, each "eval cycle" is just a cheap checkpoint save, so EVAL_INT
# gives periodic crash-safe checkpoints. Default = EPOCHS/4 (4 saves). It MUST divide
# EPOCHS evenly (pretrain.py asserts this).
GBS="${GBS:-96}"; EPOCHS="${EPOCHS:-2000}"; PELR="${PELR:-1e-2}"; LR="${LR:-1e-9}"
EVAL_INT="${EVAL_INT:-$((EPOCHS / 4))}"

if [ "$MODE" = full ]; then
    DATA=data/arc-2-aug-1000; RAW="dataset/raw-data/ARC-AGI-2/data"
elif [ "$MODE" = evalonly ]; then
    DATA=data/arc-2-evalonly
    mkdir -p data/raw-arc2-evalonly
    ln -sfn "$PWD/dataset/raw-data/ARC-AGI-2/data/evaluation" data/raw-arc2-evalonly/evaluation
    RAW=data/raw-arc2-evalonly
else
    echo "usage: $0 [evalonly|full]"; exit 1
fi

if [ ! -f "$CKPT" ]; then echo "ERROR missing $CKPT"; exit 2; fi

# 1) Build the dataset once (memory-lean; ~5-7 min) if absent.
if [ ! -f "$DATA/test/dataset.json" ]; then
    echo "[adapt] building $DATA (seed=42 num_aug=1000) ..."
    PYTHONPATH="$PWD:$PWD/dataset" $PY -u -c "
import sys; sys.path.insert(0,'dataset')
import build_arc_dataset as b
b.convert_dataset(b.DataProcessConfig(dataset_dirs=['$RAW'], output_dir='$DATA', seed=42, num_aug=1000))
print('BUILD DONE', flush=True)
" || { echo '[adapt] build failed'; exit 3; }
else
    echo "[adapt] dataset $DATA already present — reusing (do NOT rebuild between adapt and analysis)"
fi

# 2) Adaptation: frozen core (lr=0), train puzzle_emb (puzzle_emb_lr>0).
echo "[adapt] training puzzle_emb on $DATA (core frozen) -> $OUT"
$PY pretrain.py \
    data_path="$DATA" \
    load_checkpoint="$CKPT" \
    checkpoint_path="$OUT" \
    lr="$LR" \
    puzzle_emb_lr="$PELR" \
    puzzle_emb_weight_decay=0.1 \
    lr_warmup_steps=200 \
    global_batch_size="$GBS" \
    epochs="$EPOCHS" \
    eval_interval="$EVAL_INT" \
    checkpoint_every_eval=True \
    project_name=arc2_adapt run_name="$MODE"

# 3) Verify recovered accuracy on the latest adapted checkpoint.
ADAPTED=$(ls -t "$OUT"/step_* 2>/dev/null | head -1)
if [ -z "$ADAPTED" ]; then echo "[adapt] no checkpoint produced in $OUT"; exit 4; fi
DIAG="${DIAG:-results/arc/diagnostics}"
echo "[adapt] verifying $ADAPTED"
$PY scripts/arc/measure_arc_accuracy.py --checkpoint "$ADAPTED" --num_puzzles 100 \
    --device cuda --output_dir "$DIAG"

# 4) Optionally chain into the full interpretability suite, GATED on recovered
#    accuracy. RUN_SUITE=1 enables it; GATE is the minimum exact_solved required
#    (default 0.02 — a clear jump from the broken 0.0). Running the suite on a
#    non-solving model wastes hours, hence the gate.
ACC=$($PY -c "import json;print(json.load(open('$DIAG/arc_accuracy.json'))['exact_solved'])" 2>/dev/null || echo 0)
echo "[adapt] exact_solved=$ACC  (checkpoint: $ADAPTED)"
if [ "${RUN_SUITE:-0}" = "1" ]; then
    GATE="${GATE:-0.02}"
    PASS=$($PY -c "print(1 if float('$ACC') >= float('$GATE') else 0)" 2>/dev/null || echo 0)
    if [ "$PASS" = "1" ]; then
        echo "[adapt] exact_solved=$ACC >= gate $GATE -> launching full suite on adapted checkpoint"
        ARC_CKPT="$ADAPTED" bash scripts/arc/run_arc_end_to_end.sh full
        echo "[adapt] FULL PIPELINE COMPLETE (adapt + verify + suite)."
    else
        echo "[adapt] exact_solved=$ACC < gate $GATE -> NOT running suite (model not solving)."
        echo "[adapt] Raise the budget (EPOCHS=4000) or PELR=2e-2 and rerun, then re-check."
    fi
else
    echo "[adapt] done. To run the suite on this checkpoint:"
    echo "        ARC_CKPT='$ADAPTED' bash scripts/arc/run_arc_end_to_end.sh full"
    echo "        (or re-run this wrapper with RUN_SUITE=1 to auto-chain next time)"
fi
