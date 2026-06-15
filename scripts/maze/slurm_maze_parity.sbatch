#!/usr/bin/env bash
#SBATCH -J maze_parity
# NOTE: partition is intentionally NOT hardcoded — pass it on submit, e.g.
#   sbatch -p <your_gpu_partition> --gres=gpu:1 scripts/maze/slurm_maze_parity.sbatch
# (adjust --gres to however your partition exposes GPUs).
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/maze/%x_%j.out
#SBATCH -e logs/maze/%x_%j.err
#
# Maze parity suite — hardened replication of the Sudoku MI probes/SAE on Maze.
#   [1] Linear probes (per-(stream,step,target), puzzle-disjoint, 5-seed CIs)
#   [2] MLP probes on z_H AND z_L (+ linear baseline + delta per probe)
#   [3] SAE study on maze z_H: collect (float32) + d x l1 sweep (Sudoku grid)
# Every probe target is a PATH-VALIDITY metric / maze structural feature — no
# token-accuracy targets. Submit from repo root:  sbatch scripts/maze/slurm_maze_parity.sbatch
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-/home/leo.rodrigues/HRM}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # conda nvcc activate.d trips set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs/maze results/maze/hardened results/maze/sae_study

CKPT="${CKPT:-checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint}"
NUM_PUZZLES="${NUM_PUZZLES:-1000}"
STEPS="${STEPS:-0,1,2,4,8,15}"
SEEDS="${SEEDS:-0,1,2,3,4}"
SAE_PUZZLES="${SAE_PUZZLES:-300}"

python -c "import torch;print('[env] cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "############ [1/3] Maze LINEAR probes (N=$NUM_PUZZLES, steps=$STEPS, 5-seed, puzzle-disjoint) ############"
python -u scripts/maze/linear_probes_maze.py \
  --checkpoint "$CKPT" --num_puzzles "$NUM_PUZZLES" --max_steps 16 \
  --steps "$STEPS" --probe_type linear --seeds "$SEEDS" \
  --device cuda --train_device cuda \
  --output_dir results/maze/hardened/linear_probes

echo "############ [2/3] Maze MLP probes on z_H and z_L (N=$NUM_PUZZLES, steps=$STEPS, 5-seed) ############"
python -u scripts/maze/linear_probes_maze.py \
  --checkpoint "$CKPT" --num_puzzles "$NUM_PUZZLES" --max_steps 16 \
  --steps "$STEPS" --probe_type mlp --seeds "$SEEDS" \
  --device cuda --train_device cuda \
  --output_dir results/maze/hardened/mlp_probes

echo "############ [3/3] Maze SAE study on z_H: collect (N=$SAE_PUZZLES, float32) + d x l1 sweep ############"
python -u scripts/sae/sae_collect_activations_maze.py \
  --n_puzzles "$SAE_PUZZLES" --max_steps 16 --store_dtype float32 --device cuda \
  --output_dir results/maze/sae_study
python -u scripts/sae/sae_sweep.py \
  --activations_path results/maze/sae_study/activations_zH.pt \
  --activation_level z_H \
  --dict_sizes 1024,2048,4096,8192 --l1_coeffs 0.003,0.01,0.03 \
  --device cuda --output_dir results/maze/sae_study

echo "############ MAZE PARITY SUITE DONE ($(date)) ############"
