#!/usr/bin/env python3
"""XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C runner.

Cross-validates the **variable-density equilibrium** where density depends
linearly on concentration:

    rho(C) = rho_L + C * (rho_H - rho_L)
    f_i^eq = w_i * rho(C) * [1 + 3(e_i.u) + 9/2(e_i.u)^2 - 3/2(u.u)]

Key physics insight:
- For variable-density Poiseuille with constant kinematic viscosity nu,
  the steady-state equation is: d2u/dy2 = -g/nu (independent of rho)
- The velocity profile is the SAME as constant-density Poiseuille
- The test validates that the LBM with variable-density equilibrium
  recovers this profile correctly.

Two approaches tested:
  A) Fixed density field (prescribed concentration, equilibrium uses rho_field)
  B) Evolving density (equilibrium uses actual rho from distributions)

Usage
-----
    python run_xval.py                # full predeclared ladder
    python run_xval.py --quick        # smoke subset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DTYPE = np.float64

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=DTYPE)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = DTYPE(1.0 / 3.0)


def equilibrium_var_density(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    u2 = ux * ux + uy * uy
    return W[:, None, None] * rho[None] * (
        DTYPE(1.0) + DTYPE(3.0) * cu + DTYPE(4.5) * cu * cu - DTYPE(1.5) * u2[None]
    )


def equilibrium_const_density(ux: np.ndarray, uy: np.ndarray, rho0: float = 1.0) -> np.ndarray:
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    u2 = ux * ux + uy * uy
    return W[:, None, None] * DTYPE(rho0) * (
        DTYPE(1.0) + DTYPE(3.0) * cu + DTYPE(4.5) * cu * cu - DTYPE(1.5) * u2[None]
    )


def macroscopic_with_force(f: np.ndarray, fx: np.ndarray, fy: np.ndarray | None = None):
    rho = f.sum(axis=0)
    rho_safe = np.where(rho > 0, rho, DTYPE(1.0))
    if fy is None:
        fy = np.zeros_like(fx)
    ux = ((f * CX[:, None, None]).sum(axis=0) + DTYPE(0.5) * fx) / rho_safe
    uy = ((f * CY[:, None, None]).sum(axis=0) + DTYPE(0.5) * fy) / rho_safe
    return rho, ux, uy


def analytic_poiseuille(y: np.ndarray, H: float, g: float, nu: float) -> np.ndarray:
    return DTYPE(g) / (DTYPE(2.0) * DTYPE(nu)) * y * (DTYPE(H) - y)


def density_from_concentration(C: np.ndarray, rho_L: float, rho_H: float) -> np.ndarray:
    return DTYPE(rho_L) + C * DTYPE(rho_H - rho_L)


def run_lbm_channel(
    ny: int, nx: int, tau: float, g: float,
    C_field: np.ndarray | None,
    rho_L: float = 1.0, rho_H: float = 1.0,
    use_var_density: bool = True,
    density_mode: str = "fixed",  # "fixed" or "evolving"
    steps_cap: int = 30000,
    check_every: int = 50,
    steady_tol: float = 1e-9,
    steady_window: int = 200,
) -> dict:
    """
    density_mode:
      - "fixed": equilibrium uses prescribed rho_field from C
      - "evolving": equilibrium uses actual rho from distributions
    """
    ny_fluid = ny - 2
    omega = DTYPE(1.0) / DTYPE(tau)
    
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    
    if use_var_density and C_field is not None:
        rho_field = density_from_concentration(C_field, rho_L, rho_H)
    else:
        rho_field = np.ones((ny, nx), dtype=DTYPE)
    
    # Body force (acceleration mode: F = g)
    fx = np.zeros((ny, nx), dtype=DTYPE)
    fx[1:-1, :] = DTYPE(g)
    fy = np.zeros((ny, nx), dtype=DTYPE)
    
    ux = np.zeros((ny, nx), dtype=DTYPE)
    uy = np.zeros((ny, nx), dtype=DTYPE)
    
    # Initialize: use rho_field for initial equilibrium
    if use_var_density:
        f = equilibrium_var_density(rho_field, ux, uy)
    else:
        f = equilibrium_const_density(ux, uy, rho0=1.0)
    f[:, solid] = DTYPE(0.0)
    
    u_hist = []
    steps = 0
    t0 = time.perf_counter()
    converged = False
    diverged = False
    
    while steps < steps_cap:
        rho_curr, ux_curr, uy_curr = macroscopic_with_force(f, fx, fy)
        
        if not np.isfinite(ux_curr).all() or not np.isfinite(rho_curr).all():
            diverged = True
            break
        
        # Choose density for equilibrium
        if use_var_density and density_mode == "fixed":
            feq = equilibrium_var_density(rho_field, ux_curr, uy_curr)
        elif use_var_density and density_mode == "evolving":
            feq = equilibrium_var_density(rho_curr, ux_curr, uy_curr)
        else:
            feq = equilibrium_const_density(ux_curr, uy_curr, rho0=1.0)
        
        f_post = f - omega * (f - feq)
        
        # Guo forcing
        cu = CX[:, None, None] * ux_curr[None] + CY[:, None, None] * uy_curr[None]
        cf = CX[:, None, None] * fx[None] + CY[:, None, None] * fy[None]
        uf = ux_curr * fx + uy_curr * fy
        src = W[:, None, None] * (DTYPE(1.0) - DTYPE(0.5) * omega) * (
            (cf - uf[None]) / CS2 + cu * cf / CS2**2
        )
        f_post = f_post + src
        f_post[:, solid] = DTYPE(0.0)
        
        f_new = np.empty_like(f)
        for i in range(9):
            f_new[i] = np.roll(f_post[i], (int(CY[i]), int(CX[i])), axis=(0, 1))
            ys = (np.arange(ny) - int(CY[i])) % ny
            xs = (np.arange(nx) - int(CX[i])) % nx
            src_solid = solid[np.ix_(ys, xs)] & ~solid
            f_new[i][src_solid] = f_post[OPP[i]][src_solid]
        f_new[:, solid] = DTYPE(0.0)
        
        f = f_new
        steps += 1
        
        if steps % check_every == 0:
            _, ux_curr, _ = macroscopic_with_force(f, fx, fy)
            u_max = float(np.abs(ux_curr[1:-1, :]).max())
            u_hist.append(ux_curr[1:-1, :].copy())
            
            if len(u_hist) > steady_window // check_every:
                prev = u_hist[-1 - steady_window // check_every]
                cur = u_hist[-1]
                if u_max > 0:
                    rel_change = float(np.abs(cur - prev).max()) / u_max
                    if rel_change < steady_tol:
                        converged = True
                        break
    
    elapsed = time.perf_counter() - t0
    rho_final, ux_final, uy_final = macroscopic_with_force(f, fx, fy)
    
    ux_fluid = ux_final[1:-1, :]
    ux_profile = ux_fluid.mean(axis=1)
    y_wall = np.arange(ny_fluid, dtype=DTYPE) + DTYPE(0.5)
    
    rho_fluid = rho_final[1:-1, :]
    rho_initial = rho_field[1:-1, :]
    rho_drift = float(np.abs(rho_fluid.sum() - rho_initial.sum()) / rho_initial.sum())
    
    cs = math.sqrt(CS2)
    ma = float(np.abs(ux_fluid).max()) / cs if np.isfinite(ux_fluid).all() else float('nan')
    
    return {
        "steps": steps, "converged": converged, "diverged": diverged,
        "elapsed_s": elapsed,
        "mlups": ny * nx * steps / elapsed / 1e6 if elapsed > 0 else 0,
        "ux_profile": ux_profile, "ux_fluid": ux_fluid, "y_wall": y_wall,
        "rho_drift": rho_drift, "ma": ma,
        "rho_min": float(rho_final[1:-1, :].min()) if np.isfinite(rho_final).all() else float('nan'),
        "rho_max": float(rho_final[1:-1, :].max()) if np.isfinite(rho_final).all() else float('nan'),
        "ny_fluid": ny_fluid, "nx": nx,
    }


def make_step_concentration(ny: int, nx: int, interface_row: int) -> np.ndarray:
    C = np.zeros((ny, nx), dtype=DTYPE)
    C[interface_row:, :] = DTYPE(1.0)
    return C


def run_case(name, ny, nx, tau, g, rho_L, rho_H, interface_frac, use_var_density, nu, density_mode="fixed"):
    ny_fluid = ny - 2
    interface_row = 1 + int(interface_frac * ny_fluid)
    C_field = make_step_concentration(ny, nx, interface_row)
    
    result = run_lbm_channel(
        ny=ny, nx=nx, tau=tau, g=g, C_field=C_field,
        rho_L=rho_L, rho_H=rho_H, use_var_density=use_var_density,
        density_mode=density_mode,
    )
    
    H = float(ny_fluid)
    y = result["y_wall"]
    u_ana = analytic_poiseuille(y, H, g, nu)
    u_lbm = result["ux_profile"]
    u_max_ana = float(np.abs(u_ana).max())
    
    if u_max_ana > 0 and np.isfinite(u_lbm).all():
        l2_error = float(np.sqrt(np.mean((u_lbm - u_ana)**2)) / u_max_ana)
        max_error = float(np.abs(u_lbm - u_ana).max() / u_max_ana)
    else:
        l2_error = float('nan')
        max_error = float('nan')
    
    result.update({
        "u_analytic": u_ana, "l2_error_pct": l2_error * 100, "max_error_pct": max_error * 100,
        "u_max_ana": u_max_ana,
        "u_max_lbm": float(np.abs(u_lbm).max()) if np.isfinite(u_lbm).all() else float('nan'),
        "case_name": name, "rho_L": rho_L, "rho_H": rho_H,
        "interface_frac": interface_frac, "nu": nu, "use_var_density": use_var_density,
        "density_mode": density_mode,
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    
    TAU = 0.8
    NU = (TAU - 0.5) / 3.0
    CS = math.sqrt(1.0 / 3.0)
    G = 1e-5
    
    cases = [
        # Control: constant density
        dict(name="C1_const_density", ny=66, nx=8, tau=TAU, g=G,
             rho_L=1.0, rho_H=1.0, interface_frac=0.5, use_var_density=False, nu=NU, density_mode="fixed"),
        # Variable density, fixed density mode
        dict(name="C2_var_density_ratio1.1_fixed", ny=66, nx=8, tau=TAU, g=G,
             rho_L=1.0, rho_H=1.1, interface_frac=0.5, use_var_density=True, nu=NU, density_mode="fixed"),
        # Variable density, evolving density mode
        dict(name="C3_var_density_ratio1.1_evolving", ny=66, nx=8, tau=TAU, g=G,
             rho_L=1.0, rho_H=1.1, interface_frac=0.5, use_var_density=True, nu=NU, density_mode="evolving"),
        # Higher density ratio, fixed
        dict(name="C4_var_density_ratio2_fixed", ny=66, nx=8, tau=TAU, g=G,
             rho_L=1.0, rho_H=2.0, interface_frac=0.5, use_var_density=True, nu=NU, density_mode="fixed"),
        # Higher density ratio, evolving
        dict(name="C5_var_density_ratio2_evolving", ny=66, nx=8, tau=TAU, g=G,
             rho_L=1.0, rho_H=2.0, interface_frac=0.5, use_var_density=True, nu=NU, density_mode="evolving"),
        # Resolution sensitivity
        dict(name="C6_var_density_ratio2_evolving_N130", ny=130, nx=8, tau=TAU, g=G,
             rho_L=1.0, rho_H=2.0, interface_frac=0.5, use_var_density=True, nu=NU, density_mode="evolving"),
    ]
    
    if args.quick:
        cases = [c for c in cases if c["name"] in ("C1_const_density", "C2_var_density_ratio1.1_fixed", "C3_var_density_ratio1.1_evolving")]
    
    records = []
    for case in cases:
        print(f"\n=== {case['name']}  rho_H/rho_L={case['rho_H']/case['rho_L']:.2f} [{case['density_mode']}]", flush=True)
        rec = run_case(**case)
        records.append(rec)
        status = "DIVERGED" if rec.get("diverged") else ("CONV" if rec["converged"] else "CAP")
        print(f"    steps={rec['steps']} [{status}] "
              f"L2={rec['l2_error_pct']:.4f}% max={rec['max_error_pct']:.4f}% "
              f"Ma={rec['ma']:.4f} drift={rec['rho_drift']:.2e} "
              f"u_ana={rec['u_max_ana']:.6g} u_lbm={rec['u_max_lbm']:.6g}", flush=True)
    
    results = {"tau": TAU, "nu": NU, "cs": CS, "g": G, "records": records}
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for rec in records:
        if np.isnan(rec["max_error_pct"]):
            verdict = "DIVERGED"
        elif rec["max_error_pct"] < 15.0:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        print(f"  {rec['case_name']}: L2={rec['l2_error_pct']:.4f}% max={rec['max_error_pct']:.4f}% [{verdict}]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
