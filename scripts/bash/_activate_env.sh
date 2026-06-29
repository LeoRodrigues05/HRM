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
#
# On CIAI, conda is not on PATH by default (it lives behind `module load
# anaconda3` at /apps/local/anaconda3). _ensure_conda() makes it available even
# in non-login batch/srun shells.

_ensure_conda() {
    command -v conda >/dev/null 2>&1 && return 0
    [ -n "${CONDA_EXE:-}" ] && return 0
    # Try the module system (init lmod first; the `module` fn is absent in
    # non-login shells).
    for _init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh \
                 /usr/share/Modules/init/bash; do
        [ -f "${_init}" ] && source "${_init}" 2>/dev/null && break
    done
    if command -v module >/dev/null 2>&1; then
        module load anaconda3 2>/dev/null || true
    fi
    command -v conda >/dev/null 2>&1 && return 0
    # Fall back to the known cluster install path.
    [ -x /apps/local/anaconda3/bin/conda ] && export PATH="/apps/local/anaconda3/bin:$PATH"
}

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
elif _ensure_conda; [ -n "$(command -v conda)" ] || [ -n "${CONDA_EXE:-}" ]; then
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
