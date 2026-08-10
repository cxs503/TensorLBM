#!/usr/bin/env bash
set -euo pipefail

# Direct wall-exchange A/B partner for v23: all corrected-boundary, collision,
# memory and Reynolds settings are inherited unchanged; only y2/L is restored.
export TENSORLBM_STRESS_EXCHANGE_DISTANCE=8.4375
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_compiled_reynolds_pilot.sh" \
  v29 200000 "$@"
