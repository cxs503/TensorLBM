#!/usr/bin/env python3
"""Fast acceptance: 2D Laplace, Marangoni, Spurious, Mass."""
import sys, json, time
sys.path.insert(0, 'src')
import torch; torch.set_num_threads(2)
from tensorlbm.free_surface_tension.verification import (
    young_laplace_2d, marangoni_case, spurious_case, mass_conservation,
    power_law_slope, linear_fit,
)

t0 = time.time()
results = {"criteria": {}}

# (a) 2D Young-Laplace: 3 radii
print("=== (a) 2D Young-Laplace ===")
radii_2d = [10.0, 16.0, 24.0]
lap2d = []
for R in radii_2d:
    r = young_laplace_2d(R, sigma=0.05, tau=0.9, steps=(500, 1500, 3000))
    err = r['metrics']['dp_rel_err']
    print(f"  R={R}: err={err:.4%}")
    lap2d.append({"R": R, "dp": r['metrics']['dp'], "ref": r['metrics']['dp_ref'], "err": err})
max_err_2d = max(abs(x['err']) for x in lap2d)
results["laplace_2d"] = lap2d
results["criteria"]["laplace_2d"] = {"max_err": max_err_2d, "pass": max_err_2d < 0.03}
print(f"  Max: {max_err_2d:.4%} {'PASS' if max_err_2d < 0.03 else 'FAIL'}\n")

# (d) Marangoni: 5 Ma
print("=== (d) Marangoni ===")
mu = (0.9 - 0.5) / 3.0; h1 = 16.0
Ma_targets = [0.005, 0.01, 0.02, 0.05, 0.1]
mar = []
for Ma in Ma_targets:
    dsdx = 2.0 * Ma * mu**2 / h1**2
    r = marangoni_case(dsdx=dsdx, Ly=32, Lx=4, tau=0.9, steps=4000, sample=200)
    u_i = r['metrics']['u_interface']
    print(f"  Ma={Ma}: u_i={u_i:.6f}")
    mar.append({"Ma": Ma, "u_i": u_i})
a, b, r2 = linear_fit([m['Ma'] for m in mar], [m['u_i'] for m in mar])
results["marangoni"] = mar
results["criteria"]["marangoni"] = {"R2": r2, "pass": r2 > 0.95}
print(f"  R²={r2:.4f} {'PASS' if r2 > 0.95 else 'FAIL'}\n")

# (e) Spurious: 4 grids
print("=== (e) Spurious ===")
radii_sp = [8.0, 12.0, 18.0, 24.0]
spu = []
for R in radii_sp:
    r = spurious_case(radius=R, sigma=0.05, tau=0.9, steps=3000, sample=1000)
    umax = r['metrics']['umax']
    print(f"  R={R}: umax={umax:.2e}")
    spu.append({"R": R, "umax": umax})
h_list = [1.0/R for R in radii_sp]
u_list = [x['umax'] for x in spu]
slope, _, r2s = power_law_slope(h_list, u_list)
results["spurious"] = spu
results["criteria"]["spurious"] = {"order": slope, "pass": slope >= 1.8}
print(f"  Order={slope:.2f} {'PASS' if slope >= 1.8 else 'FAIL'}\n")

# (f) Mass
print("=== (f) Mass ===")
mr = mass_conservation(steps=1000, sigma=0.05, radius=12.0, L=56, tau=0.9, sample=50)
ml = mr['metrics']['mass_loss']
results["mass"] = {"loss": ml}
results["criteria"]["mass"] = {"loss": ml, "pass": ml < 0.001}
print(f"  Loss={ml:.2e} {'PASS' if ml < 0.001 else 'FAIL'}\n")

print(f"Done in {time.time()-t0:.0f}s")
with open('runs/m11185_fast.json', 'w') as f:
    json.dump(results, f, indent=2)
