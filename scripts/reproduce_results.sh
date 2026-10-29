#!/usr/bin/env bash
# Re-create every number in results/ (GPU jobs run strictly one at a time on a 6 GB GPU).
#   bash scripts/reproduce_results.sh                 # all stages
#   bash scripts/reproduce_results.sh phase4 phase5   # some stages
#   bash scripts/reproduce_results.sh limits          # the limits suite (~1-2 h)
# Needs: scripts/download_data.sh, scripts/download_ncm.sh, trained toy models (phase 2),
# and for phase5 a running Ollama with qwen2.5:3b (bash scripts/ollama.sh serve).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
STAGES=${*:-phase2 phase3 phase4 phase5 limits tests}

for stage in $STAGES; do
  echo "===== $stage ($(date +%H:%M:%S))"
  case $stage in
    phase2-train)
      python -m ncm_lab.toy_ncm.data
      python -m ncm_lab.toy_ncm.tokenizer
      python -m ncm_lab.toy_ncm.generator
      python -m ncm_lab.toy_ncm.ar_baseline ;;
    phase2) python -m ncm_lab.toy_ncm.evaluate ;;
    phase3) python scripts/phase3_eval.py --n 50 ;;
    phase4)
      python -m ncm_lab.bench.run --task xsum --n 500 --precision fp16
      python -m ncm_lab.bench.run --task wmt14 --n 500 --precision fp16
      python -m ncm_lab.bench.plot ;;
    phase5)
      python -m ncm_lab.rag.experiments compression --n 40
      python -m ncm_lab.rag.experiments multilingual --n 40
      python -m ncm_lab.agents.run_tasks ;;
    limits) python -m ncm_lab.limits.run all ;;  # needs phase4 outputs and Ollama
    tests) python -m pytest -q -s -m gpu ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "===== done ($(date +%H:%M:%S))"
