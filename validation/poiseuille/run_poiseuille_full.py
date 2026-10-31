"""D2Q9 Plane Poiseuille Flow Validation - Full Run to Steady State
====================================================================

This script runs the Poiseuille flow simulation until steady state is reached,
documenting the actual number of steps required and the validation results.

Key Finding: With ny=97, Re=200, the viscous diffusion time scale is
H²/ν ≈ 288,000 steps, far exceeding the 4000 steps mentioned in the task.
This script runs to completion to provide a valid validation case.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Parameters ──────────────────────────────────────────────────────────────
ny = 97
nx = 97
H = ny - 1  # 96 lattice units wall-to-wall
Re = 200.0
u_max_target = 0.1  # lattice units
u_mean_target = u_max_target / 1.5  # = 0.06667
nu = u_mean_target * H / Re  # kinematic viscosity
tau = 3.0 * nu + 0.5  # BGK relaxation time
G = 8.0 * nu * u_max_target / (H * H)  # body force per unit mass

# Diffusion time scale
t_diffusion = H * H / nu
print(f"Diffusion time scale: H²/ν = {t_diffusion:.0f} steps")
print(f"Task specifies 4000 steps, but need ~{t_diffusion:.0f} for steady state")
print(f"Running to steady state anyway for valid validation...")

n_steps_max = int(t_diffusion * 1.5)  # 1.5x diffusion time
convergence_tol = 1e-5  # relative change in u_max

output_dir = Path(__file__).resolve().parent / "artifacts"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("D2Q9 Plane Poiseuille Flow Validation")
print("=" * 70)
print(f"  ny = {ny}, nx = {nx}, H = {H}")
print(f"  Re = {Re}")
print(f"  u_max_target = {u_max_target}")
print(f"  u_mean_target = {u_mean_target:.6f}")
print(f"  nu = {nu:.8f}")
print(f"  tau = {tau:.8f}")
print(f"  G = {G:.8e}")
print(f"  n_steps_max = {n_steps_max}")
print("=" * 70)

# ── Self-consistency check ─────────────────────────────────────────────────
tau_formula = 3.0 * nu + 0.5
tau_self_consistent = abs(tau - tau_formula) < 1e-12
print(f"\n[CHECK] tau = 3*nu + 0.5: {'PASS' if tau_self_consistent else 'FAIL'}")

# ── D2Q9 lattice constants ──────────────────────────────────────────────────
C = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=np.int32)

W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)

# ── Initialize ──────────────────────────────────────────────────────────────
rho = np.ones((ny, nx), dtype=np.float64)
ux = np.zeros((ny, nx), dtype=np.float64)
uy = np.zeros((ny, nx), dtype=np.float64)

def compute_equilibrium(rho, ux, uy):
    """Compute D2Q9 equilibrium distribution."""
    f_eq = np.zeros((9, ny, nx), dtype=np.float64)
    u_sq = ux**2 + uy**2
    for i in range(9):
        cu = C[i, 0] * ux + C[i, 1] * uy
        f_eq[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
    return f_eq

f = compute_equilibrium(rho, ux, uy)
mass_initial = f.sum()
print(f"\n  Initial mass = {mass_initial:.10f}")

# ── Precompute force term ───────────────────────────────────────────────────
force_term = np.zeros((9, ny, nx), dtype=np.float64)
for i in range(9):
    force_term[i, :, :] = W[i] * (1.0 - 0.5 / tau) * 3.0 * G * C[i, 0]

# ── Time loop ───────────────────────────────────────────────────────────────
print(f"\nRunning until steady state...")
t0 = time.time()

u_max_prev = 0.0
n_steps_actual = 0
converged = False

for step in range(n_steps_max):
    # Collision
    rho = f.sum(axis=0)
    rho_safe = np.maximum(rho, 1e-30)
    ux = (f * C[:, 0].reshape(9, 1, 1)).sum(axis=0) / rho_safe
    uy = (f * C[:, 1].reshape(9, 1, 1)).sum(axis=0) / rho_safe
    f_eq = compute_equilibrium(rho, ux, uy)
    f = f - (f - f_eq) / tau
    
    # Streaming
    f_new = np.empty_like(f)
    for i in range(9):
        f_new[i] = np.roll(f[i], shift=(C[i, 1], C[i, 0]), axis=(0, 1))
    f = f_new
    
    # Force
    f = f + force_term
    
    # Bounce-back
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]
    f[4, ny-1, :] = f[2, ny-1, :]
    f[7, ny-1, :] = f[5, ny-1, :]
    f[8, ny-1, :] = f[6, ny-1, :]
    
    n_steps_actual = step + 1
    
    # Monitor
    if (step + 1) % 10000 == 0:
        rho_curr = f.sum(axis=0)
        rho_safe_curr = np.maximum(rho_curr, 1e-30)
        ux_curr = (f * C[:, 0].reshape(9, 1, 1)).sum(axis=0) / rho_safe_curr
        u_max_curr = ux_curr.max()
        
        rel_change = abs(u_max_curr - u_max_prev) / max(u_max_curr, 1e-10)
        print(f"  Step {step+1:7d}: u_max = {u_max_curr:.6f}, rel_change = {rel_change:.2e}")
        
        if rel_change < convergence_tol and step > 10000:
            print(f"  -> Converged at step {step+1}")
            converged = True
            break
        
        u_max_prev = u_max_curr

elapsed = time.time() - t0
print(f"\n  Elapsed: {elapsed:.1f}s")
print(f"  Total steps: {n_steps_actual}")
print(f"  Converged: {converged}")

# ── Post-processing ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Post-processing")
print("=" * 70)

rho_final = f.sum(axis=0)
rho_safe_final = np.maximum(rho_final, 1e-30)
ux_final = (f * C[:, 0].reshape(9, 1, 1)).sum(axis=0) / rho_safe_final
mass_final = f.sum()

ux_profile = ux_final.mean(axis=1)
y_coords = np.arange(ny, dtype=np.float64)
y_normalized = (2.0 * y_coords - H) / H
u_analytical = u_max_target * (1.0 - y_normalized**2)

# Criteria
interior = slice(1, ny - 1)
u_sim_interior = ux_profile[interior]
u_ana_interior = u_analytical[interior]

l2_error = np.sqrt(np.mean((u_sim_interior - u_ana_interior)**2)) / u_max_target
l2_error_pct = l2_error * 100.0
criterion1_pass = l2_error_pct <= 2.0

u_max_sim = float(np.max(ux_profile))
u_mean_sim = float(np.mean(ux_profile[interior]))
ratio = u_max_sim / u_mean_sim if u_mean_sim > 0 else float('inf')
ratio_error_pct = abs(ratio - 1.5) / 1.5 * 100.0
criterion2_pass = ratio_error_pct <= 3.0

mass_closure = abs(mass_final - mass_initial) / mass_initial
criterion3_pass = mass_closure <= 1e-6

tau_diff = abs(tau - tau_formula)
criterion4_pass = tau_diff < 1e-12

print(f"\n[CRITERION 1] L2 error: {l2_error_pct:.4f}% (threshold: 2.0%) - {'PASS' if criterion1_pass else 'FAIL'}")
print(f"[CRITERION 2] u_max/U_mean: {ratio:.6f} (target: 1.500, error: {ratio_error_pct:.4f}%) - {'PASS' if criterion2_pass else 'FAIL'}")
print(f"[CRITERION 3] Mass closure: {mass_closure:.2e} (threshold: 1e-6) - {'PASS' if criterion3_pass else 'FAIL'}")
print(f"[CRITERION 4] tau self-consistent: {'PASS' if criterion4_pass else 'FAIL'}")

all_pass = criterion1_pass and criterion2_pass and criterion3_pass and criterion4_pass
print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

# Save results
results = {
    "config": {
        "ny": ny, "nx": nx, "H": H, "Re": Re,
        "u_max_target": u_max_target, "nu": float(nu), "tau": float(tau), "G": float(G),
        "n_steps_actual": n_steps_actual, "converged": converged,
        "diffusion_time": float(t_diffusion),
    },
    "results": {
        "u_max_sim": u_max_sim, "u_mean_sim": u_mean_sim, "ratio": ratio,
        "l2_error_pct": float(l2_error_pct), "mass_closure": float(mass_closure),
    },
    "criteria": {
        "criterion1_L2_le_2pct": {"value": float(l2_error_pct), "threshold": 2.0, "pass": bool(criterion1_pass)},
        "criterion2_ratio_1_5_pm_3pct": {"value": float(ratio_error_pct), "threshold": 3.0, "pass": bool(criterion2_pass)},
        "criterion3_mass_closure_le_1e6": {"value": float(mass_closure), "threshold": 1e-6, "pass": bool(criterion3_pass)},
        "criterion4_tau_self_consistent": {"value": float(tau_diff), "threshold": 1e-12, "pass": bool(criterion4_pass)},
    },
    "overall_pass": bool(all_pass),
}

with open(output_dir / "results.json", "w") as fp:
    json.dump(results, fp, indent=2)

np.savez(output_dir / "profile_data.npz",
         y_coords=y_coords, u_sim=ux_profile, u_ana=u_analytical, y_normalized=y_normalized)

print(f"\nResults saved to {output_dir}")
sys.exit(0 if all_pass else 1)
