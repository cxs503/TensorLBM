#!/bin/bash
# CASE-MCMP-SPURIOUS 一键复现脚本
# 用法: bash runs/repro_mcmp_spurious.sh

set -e
cd "$(dirname "$0")/.."

echo "=== CASE-MCMP-SPURIOUS Reproduction ==="
echo "Running 2D spurious current benchmark (quick)..."

python -c "
import sys, json
sys.path.insert(0, 'src')
from tensorlbm.verification.spurious_current import run_static_droplet, METHODS

results = {}
for method_name in ['baseline_bgk_lattice', 'bgk_geometric']:
    cfg = METHODS[method_name]
    print(f'Running {method_name}...')
    run = run_static_droplet(
        G12=2.5, n=96, radius=24.0, width=4.0,
        steps=1500, collision=cfg['collision'],
        force=cfg['force'], forcing=cfg.get('forcing', 'guo'),
        stop_tol=2e-3, check_every=100,
        keep_fields=False,
    )
    results[method_name] = {
        'u_spurious': run['u_max'],
        'u_rms': run['u_rms'],
        'converged': run['converged'],
        'steps': run['steps_run'],
    }
    print(f'  u_spurious = {run[\"u_max\"]:.6e}')

with open('artifacts/mcmp_spurious_2d_quick.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to artifacts/mcmp_spurious_2d_quick.json')
"

echo ""
echo "=== Verdict ==="
python -c "
import json
with open('artifacts/mcmp_spurious_2d_quick.json') as f:
    data = json.load(f)
for method, r in data.items():
    status = 'PASS' if r['u_spurious'] < 1e-3 else 'FAIL'
    print(f'{method}: u_spurious={r[\"u_spurious\"]:.6e} -> {status} (<1e-3)')
"

echo ""
echo "=== Done ==="
