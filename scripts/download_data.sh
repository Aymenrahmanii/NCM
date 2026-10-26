#!/usr/bin/env bash
# Recreate the small datasets that are not in Git.
set -e
cd "$(dirname "$0")/.."
mkdir -p data

if [ ! -f data/tinyshakespeare.txt ]; then
  curl -sL -o data/tinyshakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi
echo "data/tinyshakespeare.txt: $(wc -c < data/tinyshakespeare.txt) bytes"
