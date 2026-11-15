#!/usr/bin/env bash
# Phase 10 training chain, one process per run so GPU memory never carries over.
# Rerun to resume; finished runs are skipped. `touch artifacts/scale/STOP` stops it cleanly
# (the current run checkpoints and exits); `rm artifacts/scale/STOP` before resuming.
#
#   bash scripts/phase10_train.sh            # first seed of everything
#   bash scripts/phase10_train.sh seeds      # extra seeds for gen25m and ar25m
#   bash scripts/phase10_train.sh phase11    # Phase 11 generator variants (100M positions)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
STOP=artifacts/scale/STOP

step() {
  if [ -e "$STOP" ]; then echo "STOP file present, chain halted"; exit 0; fi
  echo "=== $* ($(date '+%F %T')) ==="
  $PY -m ncm_lab.scale.train "$@" || { echo "FAILED: $*"; exit 1; }
}

if [ "${1:-main}" = "phase11" ]; then
  # Phase 11 generator variants at half budget, with an attention baseline at the same budget
  for m in attn nocross conv linear; do
    step gen --size gen25m --mixer "$m" --tokens 100e6 --micro-batch 16
  done
elif [ "${1:-main}" = "seeds" ]; then
  for s in 1 2; do
    step gen --size gen25m --seed "$s" --micro-batch 32
    step ar --size ar25m --seed "$s"
  done
else
  step tokenizer
  step gen --size gen12m --micro-batch 32
  step gen --size gen25m --micro-batch 32
  for size in ar2m ar3m ar7m ar25m; do step ar --size "$size"; done
  step ar --size ar35m --micro-batch 32
fi
[ -e "$STOP" ] || echo "=== chain done ($(date '+%F %T')) ==="
