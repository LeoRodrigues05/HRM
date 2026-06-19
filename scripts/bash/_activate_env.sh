#!/bin/bash
# Shared Python-environment activation for all CIAI cluster jobs.
# Sourced (not executed) by the job scripts.
#
# Resolution order (first match wins):
#   1. $HRM_ENV_SETUP  — a full activation command you export before submitting,
#                        e.g.  export HRM_ENV_SETUP="source /home/me/envs/hrm/bin/activate"
#   2. ./.venv         — uv/virtualenv created by scripts/bash/Initialize_HRM_Repo.sh
#   3. ./.conda        — local conda env (per the CIAI docs' PATH trick)
#   4. conda activate $HRM_CONDA_ENV  (default name: hrm)
#   5. fall back to whatever `python` is already on PATH
#
# If none of these is correct for your setup, just export HRM_ENV_SETUP.

if [ -n "${HRM_ENV_SETUP:-}" ]; then
    echo "[env] using HRM_ENV_SETUP"
    eval "${HRM_ENV_SETUP}"
elif [ -f ".venv/bin/activate" ]; then
    echo "[env] sourcing ./.venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -d ".conda/bin" ]; then
    echo "[env] using ./.conda on PATH"
    export PATH="${PWD}/.conda/bin:$PATH"
elif command -v conda >/dev/null 2>&1 || [ -n "${CONDA_EXE:-}" ]; then
    echo "[env] conda activate ${HRM_CONDA_ENV:-hrm}"
    # Initialise conda for this (often non-login) shell.
    CONDA_BASE="$(conda info --base 2>/dev/null || dirname "$(dirname "${CONDA_EXE:-}")")"
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)"
    fi
    conda activate "${HRM_CONDA_ENV:-hrm}"
else
    echo "[env] WARNING: no .venv/.conda/conda found; using system python on PATH"
fi

echo "[env] python = $(command -v python || echo MISSING)"
