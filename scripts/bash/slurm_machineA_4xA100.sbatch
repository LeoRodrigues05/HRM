#!/bin/bash
#SBATCH --job-name=machineA
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=48:00:00
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --output=logs/machineA_%x_%j.out
#SBATCH --error=logs/machineA_%x_%j.err
# Partition/QOS above match the real 4xA100 jobs (see train_hrm_bptt.sh). This
# cluster needs no --account. NOTE: cscc-gpu-qos caps a user at 4 GPUs total, so
# the gpu:4 request below is exactly at the cap -- separately-submitted phases
# will QUEUE (not co-run), which is fine. A100s here are 40 GB: if training OOMs,
# lower the batch with GBS=384 (see train_hrm_bptt.sh).
#
# Batch version of the "Machine A - 4xA100 node" interactive workflow.
# Replaces `salloc -N1 --gres=gpu:4 --cpus-per-task=64 --mem=230G --time=48:00:00`
# (the #SBATCH block above requests exactly that allocation) plus the manual
# `conda activate hrm`.
#
# Phases (select with PHASE=..., default "all"):
#   A1  Maze baselines           - 4 archs concurrent x 2 seeds (whole node, sequential seeds)
#   A2  Sudoku baseline ensemble - 4 archs concurrent x 3 seeds (whole node, sequential seeds)
#   A3  Heavy eval-MI            - many 1-GPU jobs packed 4-at-a-time across the node
#
# Usage:
#   sbatch scripts/bash/slurm_machineA_4xA100.sbatch                 # all phases
#   PHASE=A1 sbatch scripts/bash/slurm_machineA_4xA100.sbatch        # just Maze baselines
#   PHASE=A3 sbatch --time=12:00:00 scripts/bash/slurm_machineA_4xA100.sbatch
#   PHASE="A1 A3" sbatch scripts/bash/slurm_machineA_4xA100.sbatch   # subset (space-separated)
#
# RECOMMENDED: submit A1, A2, A3 as three separate jobs rather than one 48h job.
# Each of A1/A2 can itself fill the wall-clock, and a crash in training should
# not take down the eval phase. sbatch defaults to --export=ALL, so PHASE and
# any EPOCHS/GBS overrides propagate from your submitting shell.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs

PHASE="${PHASE:-all}"
TS="$(date +%Y%m%d_%H%M%S)"
NGPU=4

want() { case " ${PHASE} " in *" all "*|*" $1 "*) return 0;; *) return 1;; esac; }

# --- activate environment once (prefer .venv, fallback to conda env "hrm") ---
# Relax -u: conda activate.d cuda hooks reference unbound vars (NVCC_PREPEND_FLAGS).
set +u
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${HRM_CONDA_ENV:-hrm}"
fi
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # A3 scripts under scripts/** import repo-root modules
export DISABLE_COMPILE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[machineA] $(date) on $(hostname)  PHASE=${PHASE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

################################################################################
# A1 - Maze baselines: 4 archs concurrent (one/GPU) x 2 seeds, seeds run in turn.
# Reuses the cluster-agnostic concurrent runner (it self-activates too).
################################################################################
if want A1; then
    echo "==================== [A1] Maze baselines ===================="
    for s in 0 1; do
        echo "[A1] Maze baselines wave seed=${s} @ $(date)"
        TASK=maze SEED=${s} bash scripts/bash/train_baselines_concurrent.sh
    done
    # Straggler-via-DDP alternative (uncomment to use instead of the seed-0 wave):
    #   MODEL=universal_transformer TASK=maze SEED=0 bash scripts/bash/train_baseline_ddp.sh
fi

################################################################################
# A2 - Sudoku baseline seed ensemble (R6). Base configs are cfg_pretrain_<model>
# (NO _sudoku / _maze suffix), so this cannot use train_baselines_concurrent.sh
# (which hardcodes the _<task> suffix). Run the raw loop: 4 models one/GPU,
# concurrently, for seeds 0..2 sequentially.
################################################################################
if want A2; then
    echo "==================== [A2] Sudoku baseline ensemble ===================="
    for s in 0 1 2; do
        echo "[A2] Sudoku wave seed=${s} @ $(date)"
        g=0
        for m in vanilla_rnn standard_rnn plain_transformer universal_transformer; do
            log="logs/baseline_sudoku_${m}_seed${s}_${TS}.log"
            echo "[A2] GPU${g} <- cfg_pretrain_${m} seed=${s}  (log: ${log})"
            CUDA_VISIBLE_DEVICES=$g python pretrain.py --config-name cfg_pretrain_${m} seed=$s \
                > "${log}" 2>&1 &
            g=$((g+1))
        done
        wait
        echo "[A2] seed=${s} wave done @ $(date)"
    done
fi

################################################################################
# A3 - Heavy eval-MI. Each item below is an independent (possibly multi-step)
# chain; train->causal dependencies are expressed with && inside one chain so
# they stay on the same GPU. A tiny pool keeps all 4 GPUs busy.
################################################################################
if want A3; then
    echo "==================== [A3] Heavy eval-MI ===================="
    declare -a POOL_PID=()   # POOL_PID[gpu] = pid currently using that gpu (or empty)

    # pool_submit <label> <command-string>: block until a GPU frees, then launch.
    # Uses `wait -n` (blocks on job completion) instead of a poll/sleep loop.
    pool_submit() {
        local label="$1" cmd="$2" g pid free_gpu=-1
        while [ "${free_gpu}" -lt 0 ]; do
            for g in $(seq 0 $((NGPU-1))); do
                pid="${POOL_PID[$g]:-}"
                if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                    free_gpu=$g
                    break
                fi
            done
            [ "${free_gpu}" -lt 0 ] && wait -n
        done
        local log="logs/a3_${label}_gpu${free_gpu}_${TS}.log"
        echo "[A3] GPU${free_gpu} <- ${label}  (log: ${log})"
        CUDA_VISIBLE_DEVICES=$free_gpu bash -c "$cmd" > "$log" 2>&1 &
        POOL_PID[$free_gpu]=$!
    }

    MAZE_CKPT=checkpoints/sapientinc-hrm-maze-30x30-hard/checkpoint
    ARC_CKPT=checkpoints/arc2-adapted-evalonly/step_7391
    MAZE_ACT=results/Maze/sae_study/activations_zH.pt
    ARC_ACT=results/ARC/arc/sae_study/activations_zH.pt

    # --- Maze R2: mean-ablation (writes zH_mean/ zL_mean/ alongside the zero run)
    pool_submit maze_r2_mean "python scripts/controlled/controlled_ablation.py \
        --checkpoint ${MAZE_CKPT} \
        --ablation_mode mean --num_puzzles 1000 \
        --output_dir results/Maze/hardened/ablation_controlled"

    # --- Maze R3: TopK sweep + causal (one chain per k), then dense-L1 recon-control
    for k in 16 32 64; do
        pool_submit "maze_topk${k}" "python scripts/sae/sae_train.py --activation topk --k ${k} --dict_size 2048 \
            --activations_path ${MAZE_ACT} --output_dir results/Maze/sae_study/topk \
         && python scripts/maze/sae_causal_ablation_maze.py \
            --sae_path results/Maze/sae_study/topk/sae_d2048_topk${k}.pt \
            --checkpoint ${MAZE_CKPT} \
            --activations_path ${MAZE_ACT} --n_puzzles 300 --top_k 50 --n_random_features 50 \
            --output_dir results/Maze/sae_study/causal_ablation_topk${k}"
    done
    pool_submit maze_dense "python scripts/maze/sae_causal_ablation_maze.py \
        --sae_path results/Maze/sae_study/sae_d2048_l10.01.pt \
        --checkpoint ${MAZE_CKPT} \
        --activations_path ${MAZE_ACT} --n_puzzles 300 --top_k 50 --n_random_features 50 \
        --output_dir results/Maze/sae_study/causal_ablation"

    # --- ARC R3: TopK sweep + causal (one chain per k), then dense-L1 recon-control
    for k in 16 32 64; do
        pool_submit "arc_topk${k}" "python scripts/sae/sae_train.py --activation topk --k ${k} --dict_size 2048 \
            --activations_path ${ARC_ACT} --output_dir results/ARC/arc/sae_study/topk \
         && python scripts/arc/sae_causal_ablation_arc.py \
            --sae_path results/ARC/arc/sae_study/topk/sae_d2048_topk${k}.pt \
            --checkpoint ${ARC_CKPT} \
            --activations_path ${ARC_ACT} --n_puzzles 150 --top_k 50 --n_random_features 50 \
            --output_dir results/ARC/arc/sae_study/causal_ablation_topk${k}"
    done
    pool_submit arc_dense "python scripts/arc/sae_causal_ablation_arc.py \
        --sae_path results/ARC/arc/sae_study/sae_d2048_l10.01.pt \
        --checkpoint ${ARC_CKPT} \
        --activations_path ${ARC_ACT} --n_puzzles 150 --top_k 50 --n_random_features 50 \
        --output_dir results/ARC/arc/sae_study/causal_ablation"

    echo "[A3] all chains submitted; waiting for the pool to drain @ $(date)"
    wait
    echo "[A3] eval-MI complete @ $(date)"
fi

echo "[machineA] all requested phases finished @ $(date)"
