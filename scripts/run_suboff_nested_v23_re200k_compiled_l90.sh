#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_compiled_reynolds_pilot.sh" \
  v23 200000 "$@"
