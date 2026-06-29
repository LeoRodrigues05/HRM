#!/usr/bin/env bash
# Wait for the ARC-2 dataset build to finish, then run a tiny CPU smoke of the
# ARC MI suite. Writes progress to logs/arc_smoke.log. Internal use / scratch.
set -uo pipefail
cd /home/leo.rodrigues/HRM
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/home/leo.rodrigues/miniconda3/envs/hrm/bin/python
LOG=logs/arc_smoke.log
SMOKE_OUT=/tmp/arc_smoke
: > "$LOG"

echo "[$(date +%T)] waiting for BUILD DONE..." >> "$LOG"
for i in $(seq 1 720); do   # up to ~6h (720 * 30s)
    if grep -q "BUILD DONE" logs/arc2_build.log 2>/dev/null; then
        echo "[$(date +%T)] build done after ~$((i*30))s wait" >> "$LOG"
        break
    fi
    if ! pgrep -f build_arc_dataset >/dev/null 2>&1 && ! grep -q "BUILD DONE" logs/arc2_build.log 2>/dev/null; then
        echo "[$(date +%T)] ERROR build process died without BUILD DONE" >> "$LOG"
        tail -5 logs/arc2_build.log >> "$LOG"
        exit 2
    fi
    sleep 30
done

echo "[$(date +%T)] === dataset metadata ===" >> "$LOG"
for split in train test; do
    echo "-- $split --" >> "$LOG"
    cat data/arc-2-aug-1000/$split/dataset.json >> "$LOG" 2>&1 || echo "no $split meta" >> "$LOG"
    echo >> "$LOG"
done

echo "[$(date +%T)] === linear probe smoke (CPU) ===" >> "$LOG"
$PY -u scripts/arc/arc_linear_probes.py \
    --num_puzzles 6 --steps 0,1 --epochs 4 --positions_per_sample 48 \
    --seeds 0,1 --device cpu --train_device cpu --save_probe_weights \
    --output_dir "$SMOKE_OUT/probe" >> "$LOG" 2>&1
echo "[$(date +%T)] linear probe exit=$?" >> "$LOG"

echo "[$(date +%T)] === directed ablation smoke (CPU) ===" >> "$LOG"
$PY -u scripts/arc/directed_ablation_arc.py \
    --num_puzzles 4 --max_steps 8 --n_random_controls 3 --device cpu \
    --probe_weights "$SMOKE_OUT/probe/probe_weights.pt" \
    --output_dir "$SMOKE_OUT/e9" >> "$LOG" 2>&1
echo "[$(date +%T)] directed ablation exit=$?" >> "$LOG"

echo "[$(date +%T)] === policy decomposition smoke (CPU) ===" >> "$LOG"
$PY -u scripts/analysis/policy_decomposition.py \
    --task arc --num_puzzles 4 --max_steps 8 --device cpu \
    --output_dir "$SMOKE_OUT/h4" >> "$LOG" 2>&1
echo "[$(date +%T)] policy decomposition exit=$?" >> "$LOG"

echo "[$(date +%T)] === figures smoke ===" >> "$LOG"
$PY -u scripts/arc/plot_arc_figures.py \
    --results_dir "$SMOKE_OUT/.." --output_dir "$SMOKE_OUT/figs" >> "$LOG" 2>&1 || true

echo "[$(date +%T)] SMOKE COMPLETE" >> "$LOG"
ls -R "$SMOKE_OUT" >> "$LOG" 2>&1
