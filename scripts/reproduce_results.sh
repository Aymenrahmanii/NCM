#!/usr/bin/env bash
# Re-create every number in results/ (GPU jobs run strictly one at a time on a 6 GB GPU).
#   bash scripts/reproduce_results.sh                 # all stages
#   bash scripts/reproduce_results.sh phase4 phase5   # some stages
#   bash scripts/reproduce_results.sh limits          # the limits suite (~1-2 h)
#   bash scripts/reproduce_results.sh phase10-train   # ~24 GPU-hours, resumable
# Needs: scripts/download_data.sh, scripts/download_ncm.sh, trained toy models (phase 2),
# and for phase5 a running Ollama with qwen2.5:3b (bash scripts/ollama.sh serve).
# Phases 8, 9 and 12 use the upstream checkout with the KV-cache fix (NCM_ROOT below).
# The Phase 10 judge scores through Ollama on the host; see src/ncm_lab/scale/judge.py.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export NCM_ROOT=${NCM_ROOT:-third_party/ncm-pr5-pr4}
STAGES=${*:-phase2 phase3 phase4 phase5 limits phase8 phase9 phase10 phase11 phase12 tests}

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
    phase8)
      python -m ncm_lab.bench.cache_parity
      python -m ncm_lab.bench.interleaved --task xsum
      python -m ncm_lab.bench.interleaved --task wmt14
      python -m ncm_lab.bench.significance ;;
    phase9)
      python -m ncm_lab.bench.long_outputs quality
      python -m ncm_lab.bench.long_outputs mauve
      for n in 128 256 512 1024; do python -m ncm_lab.bench.long_outputs speed --length "$n"; done
      python -m ncm_lab.bench.long_outputs plot ;;
    phase10-train)
      python -m ncm_lab.scale.data
      bash scripts/phase10_train.sh
      bash scripts/phase10_train.sh seeds ;;
    phase10)
      python -m ncm_lab.toy_ncm.diagnose
      python -m ncm_lab.scale.evaluate quality
      python -m ncm_lab.scale.evaluate latency --device cuda
      python -m ncm_lab.scale.evaluate plot
      python -m ncm_lab.scale.evaluate ceiling
      python -m ncm_lab.scale.evaluate ablate
      python -m ncm_lab.scale.evaluate quality --runs gen25m_s1 gen25m_s2 ar25m_s1 ar25m_s2 --tag _seeds
      python -m ncm_lab.scale.judge items ;;  # then judge score/summary next to Ollama
    phase11-train) bash scripts/phase10_train.sh phase11 ;;
    phase11)
      python -m ncm_lab.scale.mixer_speed --device cuda
      python -m ncm_lab.scale.mixer_speed --device cpu --repeats 5
      python -m ncm_lab.scale.evaluate quality --tag _phase11 --runs \n        gen25m-attn-100m_s0 gen25m-nocross-100m_s0 gen25m-conv-100m_s0 gen25m-linear-100m_s0 ;;
    phase12)  # SAMSum CSVs: hf download knkarthick/samsum --repo-type dataset --local-dir data/samsum/raw
      python scripts/phase12_gate.py
      python scripts/phase12_check_loss.py
      bash scripts/phase12_train.sh
      python -m ncm_lab.finetune.train --rescore ;;
    tests) python -m pytest -q -s -m gpu ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "===== done ($(date +%H:%M:%S))"
