#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TENSORLBM_CAMPAIGN_GENERATION=v22
export TENSORLBM_COMPILE_NATURAL_KBC=1
exec "$root/scripts/run_suboff_nested_v17_chunked_allocation_probe.sh" "$@"
