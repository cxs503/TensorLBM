#!/usr/bin/env bash
set -euo pipefail

export TENSORLBM_CAMPAIGN_GENERATION=${TENSORLBM_CAMPAIGN_GENERATION:-v5}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_flat_plate_v4_equivalent_level.sh" "$@"
