#!/usr/bin/env bash
# Phase 3: official NCM code, checkpoints and eval data (only the tasks this project uses).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

[ -d third_party/ncm ] || git clone https://github.com/HazyResearch/ncm.git third_party/ncm
pip install -q -r third_party/ncm/requirements.txt

# one pattern per call: passing several patterns to one --include silently dropped some
for task in xsum wmt14 sudoku; do
  hf download hazyresearch/ncm-checkpoints --local-dir artifacts --include "$task/*"
done
for pattern in "xsum/*" "wmt14/*" "sudoku/*"; do
  hf download hazyresearch/ncm-datasets --repo-type dataset --local-dir data --include "$pattern"
done
du -sh artifacts/xsum artifacts/wmt14 artifacts/sudoku
