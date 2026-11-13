#!/usr/bin/env bash
# Reproducibility script for XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S (Round 2).
# Usage: bash xval/repro_xval_nebb_bodyforce_r2.sh
# Produces: xval/xval_nebb_bodyforce_driver_r2.json (all numbers, zero hand-copy)
#
# Environment: CPU, float64 (FP64FP64), single process, eager backend.
# Requires: Python 3.11+, torch, numpy, TensorLBM src/ on PYTHONPATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S (Round 2) ==="
echo "Repo: ${REPO_ROOT}"
echo "Python: $(python3 --version 2>&1)"
echo ""

python3 "${REPO_ROOT}/xval/xval_nebb_bodyforce_driver_r2.py"

echo ""
echo "=== Verdict ==="
python3 -c "
import json
from pathlib import Path
res = json.loads(Path('${REPO_ROOT}/xval/xval_nebb_bodyforce_driver_r2.json').read_text())
v = res['verdict']
print(f'  pointwise err (NY=10): {v[\"tensorlbm_pointwise_err_ny10\"]:.4%}')
print(f'  pointwise err (NY=20): {v[\"tensorlbm_pointwise_err_ny20\"]:.4%}')
print(f'  owner threshold (15%):  {\"PASS\" if v[\"pass_owner_15pct\"] else \"FAIL\"}')
print(f'  ecosystem threshold (3%): {\"PASS\" if v[\"pass_ecosystem_3pct\"] else \"FAIL\"}')
print(f'  no-slip enforced:       {v[\"no_slip_enforced\"]}')
"

echo ""
echo "Artifact: ${REPO_ROOT}/xval/xval_nebb_bodyforce_driver_r2.json"
echo "Done."
