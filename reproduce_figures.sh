#!/usr/bin/env bash
# Regenerate every manuscript figure from the committed result aggregates.
#
#   bash reproduce_figures.sh
#
# CPU-only, no downloads, ~1 minute. Needs only the Tier-1 dependencies in
# requirements.txt. Output overwrites results/Final/figures/*.{pdf,png}.
#
# This verifies the reported numbers, not the pipeline that produced them --
# for that, see "Tier 2" in the README.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

GENERATORS=(
    scripts/plotting/generate_paper_figures.py
    scripts/plotting/generate_supporting_paper_figures.py
    scripts/plotting/plot_crosstask_causal_ladder.py
    scripts/plotting/plot_readout_vs_causality.py
    scripts/plotting/plot_maze_causal_evidence.py
)

fail=0
for g in "${GENERATORS[@]}"; do
    printf '=== %s\n' "$g"
    if python "$g"; then :; else echo "    FAILED: $g"; fail=$((fail + 1)); fi
done

echo
echo "figures in results/Final/figures: $(ls results/Final/figures/*.pdf 2>/dev/null | wc -l) PDFs"
if [ "$fail" -ne 0 ]; then
    echo "$fail generator(s) failed"; exit 1
fi

# plot_maze_patching_example.py is deliberately excluded: it re-runs a
# qualitative intervention and therefore needs the Maze checkpoint and a GPU.
echo "OK - all generators reproduced their figures."
