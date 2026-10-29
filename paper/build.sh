#!/usr/bin/env bash
# Build paper/main.pdf. Tables/figures from the limits suite appear once they exist:
#   python -m ncm_lab.limits.report   (writes paper/tables/*.tex and results/limits_*.png)
# For arXiv, `bash paper/build.sh --arxiv` also writes paper/arxiv.tar.gz with every file it
# needs in one folder (arXiv does not allow ../ paths).
set -euo pipefail
cd "$(dirname "$0")"
run() { pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log || { tail -30 build.log; exit 1; }; }
run && bibtex main > /dev/null && run && run
echo "wrote paper/main.pdf"

if [ "${1:-}" = "--arxiv" ]; then
  out=$(mktemp -d)
  sed 's|\\graphicspath{{../results/}}|\\graphicspath{{figures/}}|; s|../results/limits_|figures/limits_|g' \
    main.tex > "$out/main.tex"
  mkdir -p "$out/figures" "$out/tables"
  cp refs.bib main.bbl "$out/"
  cp tables/*.tex "$out/tables/" 2>/dev/null || true
  cp ../results/limits_*.png ../results/phase4_*_fp16.png "$out/figures/" 2>/dev/null || true
  tar -czf arxiv.tar.gz -C "$out" .
  echo "wrote paper/arxiv.tar.gz"
fi
