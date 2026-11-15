#!/usr/bin/env bash
# After queue_after_phase12.sh: decoding ablations, Phase 11 variant quality, extra-seed quality.
#   nohup setsid bash scripts/queue_eval.sh < /dev/null >> results/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until grep -qE "queue done|FAILED|STOP file|paused" results/queue.log 2>/dev/null; do sleep 120; done
grep -q "queue done" results/queue.log || { echo "training queue did not finish, eval queue not started"; exit 0; }
ev() {
  echo "=== evaluate $* ($(date '+%F %T')) ==="
  .venv/bin/python -m ncm_lab.scale.evaluate "$@" 2>&1 | grep -vE "Loading weights|WARNING clustering" \
    || true
}
ev ablate
ev quality --runs gen25m-attn-100m_s0 gen25m-nocross-100m_s0 gen25m-conv-100m_s0 gen25m-linear-100m_s0 --tag _phase11
ev quality --runs gen25m_s1 gen25m_s2 ar25m_s1 ar25m_s2 --tag _seeds
echo "=== eval queue done ($(date '+%F %T')) ==="
