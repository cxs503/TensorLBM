#!/usr/bin/env python3
"""Part 1: Fast 2D cases only."""
import sys, json, math, time
sys.path.insert(0, 'src')
import torch; torch.set_num_threads(2)
from tensorlbm.free_surface_tension.verification import (
    young_laplace_2d, droplet_oscillation, marangoni_case,
    spurious_case, mass_conservation, power_law_slope, linear_fit,
)

t0 = time.time()
results = {}

# (a) 2D Young-Laplace: 3 radii
print("=== (a) 2D Young-Laplace ===")
radii_2d = [10.0, 16.0, 24.0]
lap2d = []
for R in radii_2d:
    r = young_laplace_2d(R, sigma=0.05, tau=0.9, steps=(500, 1500, 3000))
    err = r['metrics']['dp_rel_err']
    print(f"  R={R}: err={err:.4%} umax={r['metrics']['umax']:.6f}")
    lap2d.append(r)
results["laplace_2d"] = lap2d
max_err_2d = max(abs(r['metrics']['dp_rel_err']) for r in lap2d)
print(f"  Max: {max_err_2d:.4%} {'PASS' if max_err_2d < 0.03 else 'FAIL'}")

# (c) Oscillation: 3 sigma
print("\n=== (c) Oscillation ===")
sigmas = [0.03, 0.05, 0.08]
osc = []
for s in sigmas:
    r = droplet_oscillation(sigma=s, radius=12.0, L=56, eps=0.06, tau=0.9, periods=2.5, sample=3)
    err = r['metrics']['period_rel_err']
    print(f"  σ={s}: T={r['metrics']['period']:.1f} ref={r['metrics']['period_ref']:.1f} err={err:.4%} n_cross={r['metrics']['n_zero_crossings']}")
    osc.append(r)
results["oscillation"] = osc
max_osc = max(abs(r['metrics']['period_rel_err']) for r in osc)
print(f"  Max: {max_osc:.4%} {'PASS' if max_osc < 0.05 else 'FAIL'}")

# (d) Marangoni: 5 Ma
print("\n=== (d) Marangoni ===")
mu = (0.9 - 0.5) / 3.0; h1 = 16.0
Ma_targets = [0.005, 0.01, 0.02, 0.05, 0.1]
mar = []
for Ma in Ma_targets:
    dsdx = 2.0 * Ma * mu**2 / h1**2
    r = marangoni_case(dsdx=dsdx, Ly=32, Lx=4, tau=0.9, steps=4000, sample=200)
    print(f"  Ma={Ma}: u_i={r['metrics']['u_interface']:.6f}")
    mar.append({"Ma": Ma, "u_i": r['metrics']['u_interface']})
results["marangoni"] = mar
a, b, r2 = linear_fit([m['Ma'] for m in mar], [m['u_i'] for m in mar])
print(f"  R²={r2:.4f} {'PASS' if r2 > 0.95 else 'FAIL'}")

# (e) Spurious: 4 grids
print("\n=== (e) Spurious ===")
radii_sp = [8.0, 12.0, 18.0, 24.0]
spu = []
for R in radii_sp:
    r = spurious_case(radius=R, sigma=0.05, tau=0.9, steps=3000, sample=1000)
    print(f"  R={R}: umax={r['metrics']['umax']:.2e}")
    spu.append(r)
results["spurious"] = spu
h_list = [1.0/R for R in radii_sp]
u_list = [r['metrics']['umax'] for r in spu]
slope, _, r2s = power_law_slope(h_list, u_list)
print(f"  Order={slope:.2f} {'PASS' if slope >= 1.8 else 'FAIL'}")

# (f) Mass
print("\n=== (f) Mass ===")
mr = mass_conservation(steps=1000, sigma=0.05, radius=12.0, L=56, tau=0.9, sample=50)
ml = mr['metrics']['mass_loss']
print(f"  Loss={ml:.2e} {'PASS' if ml < 0.001 else 'FAIL'}")
results["mass"] = mr

print(f"\nDone in {time.time()-t0:.0f}s")
with open('runs/m11185_part1.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
