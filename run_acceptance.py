#!/usr/bin/env python3
"""Quick acceptance run - reduced parameters for control-node timing."""
import sys, json, math, time
sys.path.insert(0, 'src')

import torch
torch.set_num_threads(2)

from tensorlbm.free_surface_tension.verification import (
    young_laplace_2d, young_laplace_3d, droplet_oscillation,
    marangoni_case, spurious_case, mass_conservation,
    power_law_slope, linear_fit,
)
from tensorlbm.free_surface_tension.csf import laplace_pressure, rayleigh_period

t0 = time.time()
results = {"cases": [], "criteria": {}}

# (a) 2D Young-Laplace: 3 radii
print("=== (a) 2D Young-Laplace ===")
radii_2d = [10.0, 16.0, 24.0]
laplace_2d_results = []
for R in radii_2d:
    r = young_laplace_2d(R, sigma=0.05, tau=0.9, steps=(500, 1500, 3000))
    err = r['metrics']['dp_rel_err']
    print(f"  R={R}: dp={r['metrics']['dp']:.6f} ref={r['metrics']['dp_ref']:.6f} err={err:.4%}")
    laplace_2d_results.append(r)
results["cases"].append({"name": "young_laplace_2d", "radii": radii_2d, "results": laplace_2d_results})
max_err_2d = max(abs(r['metrics']['dp_rel_err']) for r in laplace_2d_results)
results["criteria"]["laplace_2d_3radii_lt_3pct"] = {"max_rel_err": max_err_2d, "threshold": 0.03, "pass": max_err_2d < 0.03}
print(f"  Max 2D error: {max_err_2d:.4%}: {'PASS' if max_err_2d < 0.03 else 'FAIL'}")

# (b) 3D Young-Laplace: 3 radii (smaller, fewer steps)
print("\n=== (b) 3D Young-Laplace ===")
radii_3d = [6.0, 8.0, 10.0]
laplace_3d_results = []
for R in radii_3d:
    r = young_laplace_3d(R, sigma=0.05, tau=0.9, steps=(200, 600, 1200))
    err = r['metrics']['dp_rel_err']
    print(f"  R={R}: dp={r['metrics']['dp']:.6f} ref={r['metrics']['dp_ref']:.6f} err={err:.4%}")
    laplace_3d_results.append(r)
results["cases"].append({"name": "young_laplace_3d", "radii": radii_3d, "results": laplace_3d_results})
max_err_3d = max(abs(r['metrics']['dp_rel_err']) for r in laplace_3d_results)
results["criteria"]["laplace_3d_3radii_lt_5pct"] = {"max_rel_err": max_err_3d, "threshold": 0.05, "pass": max_err_3d < 0.05}
print(f"  Max 3D error: {max_err_3d:.4%}: {'PASS' if max_err_3d < 0.05 else 'FAIL'}")

# (c) Droplet oscillation: 3 sigma values
print("\n=== (c) Droplet oscillation ===")
sigma_values = [0.03, 0.05, 0.08]
osc_results = []
for sigma in sigma_values:
    r = droplet_oscillation(sigma=sigma, radius=12.0, L=56, eps=0.06, tau=0.9, periods=2.5, sample=3)
    err = r['metrics']['period_rel_err']
    print(f"  sigma={sigma}: T_sim={r['metrics']['period']:.2f} T_ref={r['metrics']['period_ref']:.2f} err={err:.4%} crossings={r['metrics']['n_zero_crossings']}")
    osc_results.append(r)
results["cases"].append({"name": "droplet_oscillation", "sigma_values": sigma_values, "results": osc_results})
max_osc_err = max(abs(r['metrics']['period_rel_err']) for r in osc_results)
results["criteria"]["oscillation_3sigma_lt_5pct"] = {"max_rel_err": max_osc_err, "threshold": 0.05, "pass": max_osc_err < 0.05}
print(f"  Max oscillation error: {max_osc_err:.4%}: {'PASS' if max_osc_err < 0.05 else 'FAIL'}")

# (d) Marangoni: 5 Ma values
print("\n=== (d) Marangoni sweep ===")
mu = (0.9 - 0.5) / 3.0
h1 = 16.0
Ma_targets = [0.005, 0.01, 0.02, 0.05, 0.1]
marangoni_results = []
for Ma in Ma_targets:
    dsdx = 2.0 * Ma * mu**2 / h1**2
    r = marangoni_case(dsdx=dsdx, Ly=32, Lx=4, tau=0.9, steps=4000, sample=200)
    u_i = r['metrics']['u_interface']
    print(f"  Ma={Ma}: dsdx={dsdx:.2e} u_i={u_i:.6f}")
    marangoni_results.append({"Ma_target": Ma, "dsdx": dsdx, "u_interface": u_i})
results["cases"].append({"name": "marangoni", "Ma_targets": Ma_targets, "results": marangoni_results})
Ma_list = [m['Ma_target'] for m in marangoni_results]
u_list = [m['u_interface'] for m in marangoni_results]
a, b, r2 = linear_fit(Ma_list, u_list)
results["criteria"]["marangoni_linear_R2_gt_0.95"] = {"R2": r2, "slope": a, "intercept": b, "threshold": 0.95, "pass": r2 > 0.95}
print(f"  R²={r2:.4f}: {'PASS' if r2 > 0.95 else 'FAIL'}")

# (e) Spurious current convergence: 4 grid levels
print("\n=== (e) Spurious current convergence ===")
radii_sp = [8.0, 12.0, 18.0, 24.0]
spurious_results = []
for R in radii_sp:
    r = spurious_case(radius=R, sigma=0.05, tau=0.9, steps=3000, sample=1000)
    print(f"  R={R}: umax={r['metrics']['umax']:.2e}")
    spurious_results.append(r)
results["cases"].append({"name": "spurious_convergence", "radii": radii_sp, "results": spurious_results})
h_list = [1.0/R for R in radii_sp]
u_list_sp = [r['metrics']['umax'] for r in spurious_results]
slope, intercept, r2_sp = power_law_slope(h_list, u_list_sp)
results["criteria"]["spurious_order_ge_1.8"] = {"order": slope, "R2": r2_sp, "threshold": 1.8, "pass": slope >= 1.8}
print(f"  Order={slope:.2f} (R²={r2_sp:.4f}): {'PASS' if slope >= 1.8 else 'FAIL'}")

# (f) Mass conservation
print("\n=== (f) Mass conservation ===")
mass_r = mass_conservation(steps=1000, sigma=0.05, radius=12.0, L=56, tau=0.9, sample=50)
mass_loss = mass_r['metrics']['mass_loss']
results["cases"].append({"name": "mass_conservation", "result": mass_r})
results["criteria"]["mass_loss_lt_0.1pct"] = {"mass_loss": mass_loss, "threshold": 0.001, "pass": mass_loss < 0.001}
print(f"  Mass loss: {mass_loss:.2e}: {'PASS' if mass_loss < 0.001 else 'FAIL'}")

# Summary
print(f"\n=== Summary ({time.time()-t0:.0f}s) ===")
all_pass = all(v.get('pass', False) for v in results['criteria'].values())
for k, v in results['criteria'].items():
    print(f"  {k}: {'PASS' if v.get('pass', False) else 'FAIL'}")
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

with open('runs/m11185_acceptance.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/m11185_acceptance.json")
