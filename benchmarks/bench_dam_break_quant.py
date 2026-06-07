#!/usr/bin/env python3
"""Quantitative dam-break benchmark across all TensorLBM models."""
import json, math, time, sys
from pathlib import Path
import torch
sys.path.insert(0, '/home/jsyc/TensorLBM/src')
from tensorlbm.dam_break_3d import DamBreak3DConfig, run_dam_break_3d, _KOSHIZUKA_FRONT

# Reference data
REF_T, REF_X = zip(*_KOSHIZUKA_FRONT)

# Common geometry for all models
BASE = dict(nx=120, ny=50, nz=50, dam_width=50, fill_height=49,
            n_steps=300, output_interval=30, gravity=8e-5,
            device='cpu', overwrite=True)

CONFIGS = {
    "CG-BGK (2:1)":       dict(model='cg', tau=1.0, rho_heavy=1.0, rho_light=0.5, A=0.005, collision='bgk', C_s=0.0, use_guo=False, free_slip_y=False, hydrostatic_init=False),
    "CG-MRT+SGS+Guo (10:1)": dict(model='cg', tau=0.8, rho_heavy=1.0, rho_light=0.1, A=0.005, collision='mrt_smag', C_s=0.1, use_guo=True, free_slip_y=True, hydrostatic_init=True),
    "SC-BGK (2:1)":       dict(model='sc', tau=1.0, rho_heavy=1.0, rho_light=0.5, G_sc=0.9, collision='bgk', C_s=0.0, use_guo=False, free_slip_y=False, hydrostatic_init=False),
    "SC-MRT+SGS+Guo":     dict(model='sc', tau=0.8, rho_heavy=1.0, rho_light=0.5, G_sc=0.9, collision='mrt_smag', C_s=0.1, use_guo=True, free_slip_y=True, hydrostatic_init=True),
    "FE (5:1)":           dict(model='fe', tau=1.0, rho_heavy=1.0, rho_light=0.2, free_slip_y=False),
    "Free-Surface":        dict(model='fs', tau=0.8, rho_heavy=1.0, rho_light=0.5, free_slip_y=True),
}

def compute_error(sim_t, sim_x):
    """RMSE between simulation and reference (interpolating ref at sim times)."""
    import numpy as np
    if len(sim_t) < 2:
        return float('nan'), 0
    ref_interp = np.interp(sim_t, REF_T, REF_X)
    rmse = np.sqrt(np.mean((np.array(sim_x) - ref_interp)**2))
    # Also compute max error
    max_err = np.max(np.abs(np.array(sim_x) - ref_interp))
    return rmse, max_err

print("=" * 80)
print("TensorLBM 3D Dam-Break Quantitative Benchmark")
print("=" * 80)
print(f"Domain: {BASE['nx']}×{BASE['ny']}×{BASE['nz']}, {BASE['n_steps']} steps")
print(f"Reference: Koshizuka & Oka (1996)")
print()

results = []
for name, cfg in CONFIGS.items():
    print(f"--- {name} ---")
    config = DamBreak3DConfig(**{**BASE, **cfg}, run_name=name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", ""))
    
    t0 = time.time()
    try:
        run_dir = run_dam_break_3d(config)
        elapsed = time.time() - t0
        
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        diags = meta['diagnostics']
        
        sim_t = [d['t_star'] for d in diags]
        sim_x = [d['x_star'] for d in diags]
        mean_rhos = [d.get('mean_rho', float('nan')) for d in diags]
        
        rmse, max_err = compute_error(sim_t, sim_x)
        
        # Stability check
        nan_free = all(not math.isnan(d.get('mean_rho', 0)) for d in diags)
        
        results.append({
            'name': name,
            'steps': len(sim_t),
            'final_t': sim_t[-1] if sim_t else 0,
            'final_X': sim_x[-1] if sim_x else 0,
            'RMSE': rmse,
            'max_err': max_err,
            'mean_rho_final': mean_rhos[-1] if mean_rhos else float('nan'),
            'time_s': elapsed,
            'stable': nan_free,
        })
        
        print(f"  Time: {elapsed:.1f}s, front at t*={sim_t[-1]:.3f}: X*={sim_x[-1]:.3f}")
        print(f"  RMSE={rmse:.4f}, max_err={max_err:.4f}, stable={nan_free}")
        if mean_rhos:
            print(f"  mean_rho: start={mean_rhos[0]:.4f} end={mean_rhos[-1]:.4f}")
        
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({'name': name, 'steps': 0, 'final_t': 0, 'final_X': 0,
                        'RMSE': float('nan'), 'max_err': float('nan'),
                        'mean_rho_final': float('nan'), 'time_s': 0, 'stable': False})

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Model':<30s} {'RMSE':>8s} {'MaxErr':>8s} {'Final X*':>8s} {'Time(s)':>8s} {'Stable':>8s}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<30s} {r['RMSE']:>8.4f} {r['max_err']:>8.4f} {r['final_X']:>8.3f} {r['time_s']:>8.1f} {str(r['stable']):>8s}")

# Best model
valid = [r for r in results if r['stable'] and not math.isnan(r['RMSE'])]
if valid:
    best = min(valid, key=lambda r: r['RMSE'])
    print(f"\nBest model (lowest RMSE): {best['name']} — RMSE={best['RMSE']:.4f}")
