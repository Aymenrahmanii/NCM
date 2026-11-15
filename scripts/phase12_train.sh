#!/usr/bin/env bash
# Phase 12 chain: SAMSum arms for the XSum NCM and bart-base-xsum, one process per run.
#   nohup setsid bash scripts/phase12_train.sh < /dev/null >> results/phase12_train.log 2>&1 &
# Finished runs are skipped (results/phase12_runs.csv). Touch results/phase12.STOP to pause
# after the current run; delete it and relaunch to resume.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src NCM_ROOT=third_party/ncm-pr5-pr4
runs=("ncm zero 0" "bart zero 0")
for seed in 0 1 2; do
    for arm in lora8_qv lora16_all full; do
        runs+=("bart $arm $seed" "ncm $arm $seed")
    done
done
for r in "${runs[@]}"; do
    if [ -e results/phase12.STOP ]; then echo "=== STOP file found, pausing ($(date '+%F %T')) ==="; exit 0; fi
    set -- $r
    echo "=== $1 $2 seed $3 ($(date '+%F %T')) ==="
    .venv/bin/python -m ncm_lab.finetune.train --model "$1" --arm "$2" --seed "$3" \
        2>&1 | grep -vE "Warning|warn\(" || echo "FAILED: $r"
done
echo "=== chain done ($(date '+%F %T')) ==="
