#!/usr/bin/env bash
# Launch the TensorLBM B/S platform server.
# Usage: ./start.sh [--port 8000] [--host 0.0.0.0]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure platform deps are installed
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Add tensorlbm to PYTHONPATH
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Starting TensorLBM Platform at http://${HOST}:${PORT}"
exec uvicorn backend.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --reload-dir "$SCRIPT_DIR/backend" \
    --app-dir "$SCRIPT_DIR"
