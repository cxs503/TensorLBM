#!/usr/bin/env bash
# Launch the TensorLBM B/S platform server.
# Usage: ./start.sh [--port 8000] [--host 0.0.0.0] [--prod]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure platform deps are installed
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Add tensorlbm to PYTHONPATH
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
PROD="${PROD:-false}"

# Default: 4 workers (matches TENSORLBM_MAX_WORKERS=4)
# In production, scale workers based on CPU cores
if [ "$PROD" = "true" ]; then
    WORKERS="${WORKERS:-4}"
    echo "Starting TensorLBM Platform (PRODUCTION) at http://${HOST}:${PORT} with ${WORKERS} workers"
    exec uvicorn backend.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --app-dir "$SCRIPT_DIR" \
        --loop uvloop \
        --http httptools \
        --no-access-log
else
    echo "Starting TensorLBM Platform (DEV) at http://${HOST}:${PORT}"
    exec uvicorn backend.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --reload-dir "$SCRIPT_DIR/backend" \
        --app-dir "$SCRIPT_DIR"
fi
