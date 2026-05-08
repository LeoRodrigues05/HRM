#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=logs/eval_baselines_%j.out
#SBATCH --error=logs/eval_baselines_%j.err

mkdir -p logs results/baseline_comparison

# Activate environment (prefer .venv, fallback to conda env named "hrm")
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${HRM_CONDA_ENV:-hrm}"
fi
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export DISABLE_COMPILE=1

N_PUZZLES=500
N_PATCHING=100

echo "=== Starting multi-checkpoint baseline evaluation at $(date) ==="

# ─── 1. HRM (reference, fully trained) ───
echo ">>> [1/9] Evaluating HRM (reference)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/sapientinc-sudoku-extreme/checkpoint.pt \
    --model_name HRM \
    --n_puzzles $N_PUZZLES \
    --n_patching_pairs $N_PATCHING \
    --device cuda \
    --output_dir results/baseline_comparison

# ─── 2. Universal Transformer – multiple checkpoints ───
# Best checkpoint (peak accuracy ~step 20k based on training curves)
echo ">>> [2/9] Evaluating UT @ step_15624 (pre-peak)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/universal_transformer/step_15624 \
    --model_name UT_step15624 \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device cuda \
    --output_dir results/baseline_comparison

echo ">>> [3/9] Evaluating UT @ step_20832 (BEST)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/universal_transformer/step_20832 \
    --model_name UT_best \
    --n_puzzles $N_PUZZLES \
    --n_patching_pairs $N_PATCHING \
    --device cuda \
    --output_dir results/baseline_comparison

echo ">>> [4/9] Evaluating UT @ step_31248 (diverging)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/universal_transformer/step_31248 \
    --model_name UT_step31248 \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device cuda \
    --output_dir results/baseline_comparison

echo ">>> [5/9] Evaluating UT @ step_41664 (diverged)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/universal_transformer/step_41664 \
    --model_name UT_step41664 \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device cuda \
    --output_dir results/baseline_comparison

# ─── 3. Vanilla RNN – multiple checkpoints ───
echo ">>> [6/9] Evaluating RNN @ step_10416 (early)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/vanilla_rnn/step_10416 \
    --model_name RNN_step10416 \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device cuda \
    --output_dir results/baseline_comparison

echo ">>> [7/9] Evaluating RNN @ step_20832 (BEST)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/vanilla_rnn/step_20832 \
    --model_name RNN_best \
    --n_puzzles $N_PUZZLES \
    --n_patching_pairs $N_PATCHING \
    --device cuda \
    --output_dir results/baseline_comparison

echo ">>> [8/9] Evaluating RNN @ step_26040 (late peak)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/vanilla_rnn/step_26040 \
    --model_name RNN_step26040 \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device cuda \
    --output_dir results/baseline_comparison

echo ">>> [9/9] Evaluating RNN @ step_31248 (diverged)..."
python scripts/analysis/evaluate_baselines.py \
    --checkpoint checkpoints/baselines/vanilla_rnn/step_31248 \
    --model_name RNN_step31248 \
    --n_puzzles $N_PUZZLES \
    --skip_patching \
    --device cuda \
    --output_dir results/baseline_comparison

# ─── 4. Generate comparison plots ───
echo ">>> Generating comparison plots..."
python scripts/plotting/plot_baseline_comparison.py \
    --results_dir results/baseline_comparison

echo "=== Done at $(date) ==="
