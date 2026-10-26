#!/usr/bin/env bash
# Run a command inside the project venv from any shell: bash scripts/run.sh python scripts/x.py
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec "$@"
