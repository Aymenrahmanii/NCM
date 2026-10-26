#!/usr/bin/env bash
# Run the user-space Ollama (installed by scripts/install_ollama_user.sh).
#   bash scripts/ollama.sh serve          # start the server (keep this terminal open)
#   bash scripts/ollama.sh pull qwen2.5:3b
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
exec "$HOME/.local/ollama/bin/ollama" "$@"
