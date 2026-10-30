#!/bin/bash
# Run SDAA sweep for startup_poiseuille
cd /root/tp-agent/work/m49

python -c "
import sys, time, json
from pathlib import Path
REPO = Path('.').resolve()
BENCH = REPO / 'benchmarks'
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / 'src'))
import torch
from compile_route import ensure_tensorlbm_importable
ensure_tensorlbm_importable()
import importlib.util
run_py = BENCH / 'verified' / 'startup_poiseuille' / 'run.py'
spec = importlib.util.spec_from_file_location('sp', run_py)
sp_mod = importlib.util.module_from_spec(spec)
sp_mod.DEVICE = torch.device('sdaa')
spec.loader.exec_module(sp_mod)

results = {}
for H in [15, 29]:
    t0 = time.time()
    r = sp_mod.run_case(H=H, tau=0.8, u_max_target=0.03, nx=16, out_path=None, seed=0, compile_mode='eager')
    elapsed = time.time() - t0
    results[f'H{H}'] = {
        'centerline_err_max_over_umax_pct': r['centerline_err_max_over_umax_pct'],
        'profile_l2_max_pct': r['profile_l2_max_pct'],
        'profile_pointwise_max_over_umax_pct': r['profile_pointwise_max_over_umax_pct'],
        'steps': r['t_end'],
        'elapsed_s': elapsed,
        'profile_table': r['profile_table'],
        'centerline_table_every10': r['centerline_table_every10'],
        'mass_drift_pct': r['mass_drift_pct'],
        'finite': r['finite'],
    }
    print(f'H={H}: center={r[\"centerline_err_max_over_umax_pct\"]:.4f}% l2={r[\"profile_l2_max_pct\"]:.4f}% ptp={r[\"profile_pointwise_max_over_umax_pct\"]:.4f}% steps={r[\"t_end\"]} elapsed={elapsed:.1f}s')

with open('artifacts/sdaa_results_raw.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to artifacts/sdaa_results_raw.json')
" 2>&1 | tee artifacts/sdaa_sweep.log
