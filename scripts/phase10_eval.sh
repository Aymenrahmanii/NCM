#!/usr/bin/env bash
# Phase 10 evaluation chain: quality (MAUVE etc.), GPU latency, Pareto plot.
#   nohup setsid bash scripts/phase10_eval.sh < /dev/null >> results/phase10_eval.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for cmd in "quality" "latency --device cuda" "plot"; do
    echo "=== $cmd ($(date '+%F %T')) ==="
    # shellcheck disable=SC2086
    .venv/bin/python -m ncm_lab.scale.evaluate $cmd || { echo "FAILED: $cmd"; exit 1; }
done
echo "=== eval done ($(date '+%F %T')) ==="
