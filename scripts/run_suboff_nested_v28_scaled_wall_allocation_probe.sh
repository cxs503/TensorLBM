#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TENSORLBM_CAMPAIGN_GENERATION=v28
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_STRESS_EXCHANGE_DISTANCE=8.4375
exec "$root/scripts/run_suboff_nested_v17_chunked_allocation_probe.sh" "$@"
