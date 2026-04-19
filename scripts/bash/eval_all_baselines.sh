#!/bin/bash
# Evaluate best checkpoint for all 5 baseline architectures.
# Produces one JSON per model in results/baseline_comparison/.
#
# Models: HRM, Vanilla RNN, Universal Transformer, Plain Transformer, Standard RNN
# Best checkpoints identified from prior multi-checkpoint sweeps.
set -e

OUTDIR="results/baseline_comparison"
DEVICE="cuda"
N_PUZZLES=500
MAX_STEPS=16

echo "=============================================="
echo "  Evaluating all 5 models (best checkpoints)"
echo "  N=$N_PUZZLES puzzles, max_steps=$MAX_STEPS"
echo "  $(date)"
echo "=============================================="

# ── 1. HRM (reference) ──
echo ""
echo ">>> [1/5] HRM"
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/sapientinc-sudoku-extreme/checkpoint.pt \
    --model_name HRM \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device $DEVICE \
    --output_dir $OUTDIR \
    --max_steps $MAX_STEPS

# ── 2. Vanilla RNN (best = step_20832) ──
echo ""
echo ">>> [2/5] Vanilla RNN (step_20832)"
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/vanilla_rnn/step_20832 \
    --model_name RNN_best \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device $DEVICE \
    --output_dir $OUTDIR \
    --max_steps $MAX_STEPS

# ── 3. Universal Transformer (best = step_20832) ──
echo ""
echo ">>> [3/5] Universal Transformer (step_20832)"
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/universal_transformer/step_20832 \
    --model_name UT_best \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device $DEVICE \
    --output_dir $OUTDIR \
    --max_steps $MAX_STEPS

# ── 4. Plain Transformer (best = step_15624) ──
echo ""
echo ">>> [4/5] Plain Transformer (step_15624)"
python scripts/analysis/evaluate_baselines.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/PlainTransformerModel strange-nightingale/step_15624" \
    --model_name PT_best \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device $DEVICE \
    --output_dir $OUTDIR \
    --max_steps $MAX_STEPS

# ── 5. Standard RNN (best = step_41664) ──
echo ""
echo ">>> [5/5] Standard RNN (step_41664)"
python scripts/analysis/evaluate_baselines.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/StandardRNNModel ecstatic-honeybee/step_41664" \
    --model_name SRNN_best \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device $DEVICE \
    --output_dir $OUTDIR \
    --max_steps $MAX_STEPS

# ── Summary ──
echo ""
echo "=============================================="
echo "  Summary of results"
echo "=============================================="
python -c "
import json, os

outdir = '$OUTDIR'
models = [
    ('HRM',      'HRM_eval.json',      15),
    ('RNN',      'RNN_best_eval.json',  15),
    ('UT',       'UT_best_eval.json',   15),
    ('PT',       'PT_best_eval.json',   0),
    ('SRNN',     'SRNN_best_eval.json', 0),
]
print(f'{\"Model\":>8s}  {\"Cell%\":>7s}  {\"Puzzle%\":>8s}  {\"Hamming\":>8s}  {\"RowViol\":>8s}  {\"ColViol\":>8s}  {\"BoxViol\":>8s}')
print('-' * 68)
for name, fname, step in models:
    fpath = os.path.join(outdir, fname)
    if not os.path.exists(fpath):
        print(f'{name:>8s}  MISSING: {fname}')
        continue
    d = json.load(open(fpath))
    m = d['per_step_metrics'][str(step)]
    print(f'{name:>8s}  {m[\"cell_accuracy\"][\"mean\"]:7.4f}  {m[\"puzzle_accuracy\"][\"mean\"]:8.4f}  '
          f'{m[\"hamming_distance\"][\"mean\"]:8.1f}  {m[\"row_violations\"][\"mean\"]:8.1f}  '
          f'{m[\"col_violations\"][\"mean\"]:8.1f}  {m[\"box_violations\"][\"mean\"]:8.1f}')
"

echo ""
echo "=== Done at $(date) ==="
