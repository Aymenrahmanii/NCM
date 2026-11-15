#!/usr/bin/env bash
# Waits for the Phase 12 chain, then: Phase 11 variants, Phase 11 speed (GPU, CPU),
# Phase 10 extra seeds. Each stage stops the queue if it fails or hits a STOP file.
#   nohup setsid bash scripts/queue_after_phase12.sh < /dev/null >> results/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until grep -qE "chain done|STOP file found" results/phase12_train.log 2>/dev/null; do sleep 120; done
grep -q "chain done" results/phase12_train.log || { echo "Phase 12 paused, queue not started"; exit 0; }
echo "=== queue start ($(date '+%F %T')) ==="
bash scripts/phase10_train.sh phase11 || exit 1
[ -e artifacts/scale/STOP ] && exit 0
for dev in "cuda --repeats 20" "cpu --repeats 5"; do
  echo "=== mixer speed $dev ($(date '+%F %T')) ==="
  # shellcheck disable=SC2086
  .venv/bin/python -m ncm_lab.scale.mixer_speed --device $dev || { echo "FAILED: speed $dev"; exit 1; }
done
bash scripts/phase10_train.sh seeds || exit 1
echo "=== queue done ($(date '+%F %T')) ==="
