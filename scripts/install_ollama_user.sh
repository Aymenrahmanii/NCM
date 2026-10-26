#!/usr/bin/env bash
# Install Ollama into ~/.local/ollama without sudo (the official installer needs sudo).
# Then: ~/.local/ollama/bin/ollama serve   (or: bash scripts/ollama.sh serve)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
pip install -q zstandard
mkdir -p ~/.local/ollama
curl -fL -o /tmp/ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst
python - <<'EOF'
import os, tarfile, zstandard
with open("/tmp/ollama.tar.zst", "rb") as f:
    with zstandard.ZstdDecompressor().stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(os.path.expanduser("~/.local/ollama"), filter="tar")
EOF
rm /tmp/ollama.tar.zst
~/.local/ollama/bin/ollama --version || true
