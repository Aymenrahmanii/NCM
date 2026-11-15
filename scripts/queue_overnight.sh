#!/usr/bin/env bash
# Everything after Phase 12, one GPU job at a time, in priority order:
#   1. Phase 11 variants (100M)       2. validity check (official-style tokenizer)
#   3. evaluation of 1-2 (+ ceiling, ablations)   4. Phase 11 speed
#   5. Phase 10 extra seeds and their evaluation
#   nohup setsid bash scripts/queue_overnight.sh < /dev/null >> results/queue.log 2>&1 &
# Stops at the first failure or STOP file (artifacts/scale/STOP); rerun to resume, finished
# training runs are skipped (evaluations re-run).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
until grep -qE "chain done|STOP file found" results/phase12_train.log 2>/dev/null; do sleep 120; done
grep -q "chain done" results/phase12_train.log || { echo "Phase 12 paused, queue not started"; exit 0; }
stopped() { [ -e artifacts/scale/STOP ] && { echo "STOP file, queue halted"; exit 0; }; }
ev() {
  stopped
  echo "=== evaluate $* ($(date '+%F %T')) ==="
  $PY -m ncm_lab.scale.evaluate "$@" 2>&1 | grep -vE "Loading weights|WARNING clustering"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: evaluate $*"; exit 1; }
}
echo "=== queue start ($(date '+%F %T')) ==="
bash scripts/phase10_train.sh phase11 || exit 1; stopped
bash scripts/phase10_train.sh validity || exit 1; stopped
V2=gen25m-attn-100m-tokv2-tied_s0
ev quality --tag _phase11 --runs gen25m-attn-100m_s0 gen25m-nocross-100m_s0 \
  gen25m-conv-100m_s0 gen25m-linear-100m_s0 "$V2"
ev ceiling --gen-run "$V2"
ev ablate
for dev in "cuda --repeats 20" "cpu --repeats 5"; do
  stopped
  echo "=== mixer speed $dev ($(date '+%F %T')) ==="
  # shellcheck disable=SC2086
  $PY -m ncm_lab.scale.mixer_speed --device $dev || { echo "FAILED: speed $dev"; exit 1; }
done
bash scripts/phase10_train.sh seeds || exit 1
ev quality --tag _seeds --runs gen25m_s1 gen25m_s2 ar25m_s1 ar25m_s2
echo "=== queue done ($(date '+%F %T')) ==="
