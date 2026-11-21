#!/usr/bin/env python3
"""Minimal verification: single radius cases that run fast."""
import sys, json, time
sys.path.insert(0, 'src')
import torch; torch.set_num_threads(2)
from tensorlbm.free_surface_tension.verification import (
    young_laplace_2d, mass_conservation, linear_fit,
)
from tensorlbm.free_surface_tension import (
    CapillarySolver, SolverConfig, tanh_droplet_phi, laplace_pressure,
)

t0 = time.time()
results = {"summary": {}, "cases": []}

# Quick 2D Laplace: single radius
print("=== Quick 2D Young-Laplace (R=8) ===")
r = young_laplace_2d(8.0, sigma=0.05, tau=0.9, steps=(300, 600))
err = r['metrics']['dp_rel_err']
print(f"  dp={r['metrics']['dp']:.6f} ref={r['metrics']['dp_ref']:.6f} err={err:.4%}")
results["cases"].append({
    "name": "young_laplace_2d_quick",
    "R": 8.0,
    "dp": r['metrics']['dp'],
    "ref": r['metrics']['dp_ref'],
    "rel_err": err,
})
results["summary"]["laplace_2d_single"] = {"err": err, "pass": abs(err) < 0.03}

# Quick mass conservation
print("\n=== Quick mass conservation ===")
mr = mass_conservation(steps=500, sigma=0.05, radius=10.0, L=40, tau=0.9, sample=50)
ml = mr['metrics']['mass_loss']
print(f"  Mass loss: {ml:.2e}")
results["cases"].append({
    "name": "mass_conservation_quick",
    "steps": 500,
    "mass_loss": ml,
})
results["summary"]["mass_quick"] = {"loss": ml, "pass": ml < 0.001}

# Marangoni: single case
print("\n=== Quick Marangoni ===")
from tensorlbm.free_surface_tension.verification import marangoni_case
mu = (0.9 - 0.5) / 3.0; h1 = 16.0
Ma = 0.02
dsdx = 2.0 * Ma * mu**2 / h1**2
r = marangoni_case(dsdx=dsdx, Ly=32, Lx=4, tau=0.9, steps=2000, sample=100)
u_i = r['metrics']['u_interface']
print(f"  Ma={Ma}: u_i={u_i:.6f}")
results["cases"].append({
    "name": "marangoni_quick",
    "Ma": Ma,
    "u_interface": u_i,
})

# Spurious: single case
print("\n=== Quick spurious current ===")
from tensorlbm.free_surface_tension.verification import spurious_case
r = spurious_case(radius=10.0, sigma=0.05, tau=0.9, steps=2000, sample=500)
umax = r['metrics']['umax']
print(f"  R=10: umax={umax:.2e}")
results["cases"].append({
    "name": "spurious_quick",
    "R": 10.0,
    "umax": umax,
})

print(f"\nDone in {time.time()-t0:.0f}s")
with open('runs/m11185_quick.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved runs/m11185_quick.json")
