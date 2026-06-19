#!/bin/bash
#SBATCH --job-name=hrm_sae_ab
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:1
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/hrm_sae_ab_%j.out
#SBATCH --error=logs/hrm_sae_ab_%j.err

# Post step (recommended order #3): for BOTH trained HRMs (stock + bptt),
# collect z_H activations and run the SAE A/B sweep (baseline vs mean-centered)
# across dict_size x l1. Produces the (stock|bptt) x (baseline|mean-centered)
# table on dead_count / L0 / recon / gamma.

set -eo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
source scripts/bash/_activate_env.sh
export DISABLE_COMPILE=1

N_PUZZLES="${N_PUZZLES:-1000}"

for name in hrm_stock hrm_bptt; do
    CKPT_DIR="checkpoints/bptt_study/${name}"
    OUT="results/sae_study/bptt_study/${name}"
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "WARN: ${CKPT_DIR} not found — skipping ${name}"
        continue
    fi
    mkdir -p "${OUT}"
    echo "=== ${name}: collecting activations ==="
    python scripts/sae/sae_collect_activations.py \
        --n_puzzles "${N_PUZZLES}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${OUT}"

    echo "=== ${name}: SAE A/B sweep (baseline vs mean-centered) ==="
    python scripts/sae/sae_sweep.py \
        --compare_centering \
        --activations_path "${OUT}/activations_zH.pt" \
        --output_dir "${OUT}" \
        --dict_sizes 1024,2048,4096 \
        --l1_coeffs 0.003,0.01,0.03
done

echo "Post step complete at $(date). Compare results/sae_study/bptt_study/*/sweep_results.csv"
