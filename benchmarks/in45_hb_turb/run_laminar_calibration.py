#!/usr/bin/env python3
"""Laminar calibration: HB channel flow vs analytical solutions.

Tests 3 values of flow index n ∈ {0.5, 0.8, 1.0} with various yield stresses.
Compares LBM simulation against exact HB Poiseuille profiles.

This validates the Papanastasiou-regularised HB collision kernel before
proceeding to turbulent cases.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.hb_channel_analytical import (
    HBChannelParams,
    hb_velocity_profile,
    hb_mean_velocity,
    hb_wall_shear_rate,
    metzner_reed_reynolds_number,
    friction_factor_channel,
    dodge_metzner_laminar,
)


def run_laminar_hb_2d(
    H: int,
    nx: int,
    tau_y: float,
    K: float,
    n: float,
    G: float,
    m_pap: float = 500.0,
    max_iter: int = 10000,
    tol: float = 1e-6,
) -> dict:
    """Run 2D laminar HB channel flow with D2Q9.

    Uses the existing yield_stress_lbm infrastructure.
    """
    from tensorlbm.d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
    from tensorlbm.solver import stream
    from tensorlbm.yield_stress_rheology import herschel_bulkley_viscosity
    from tensorlbm.powerlaw import central_difference, strain_rate_shear_rate_2d

    ny = H + 2
    dtype = torch.float64
    device = "cpu"

    # Initialize with parabolic profile
    y = torch.arange(ny, dtype=dtype).view(ny, 1)
    h = H / 2.0
    # Initial guess: Newtonian profile
    nu_guess = K * (G * h / 2) ** (n - 1) if n != 1 else K
    nu_guess = max(nu_guess, 1e-4)
    u_init = G / (2 * nu_guess) * (y - 1) * (ny - 1 - y + 1)
    u_init = torch.clamp(u_init, min=0.0)
    u_init = u_init.expand(ny, nx).clone()

    # Normalize to safe lattice velocity
    u_max = u_init.max().item()
    if u_max > 0.05:
        u_init *= 0.05 / u_max

    ux = u_init
    uy = torch.zeros(ny, nx, dtype=dtype)
    rho = torch.ones(ny, nx, dtype=dtype)

    f = equilibrium(rho, ux, uy)

    opp = OPPOSITE
    c = C.to(dtype=dtype)
    w = W.to(dtype=dtype)

    # Time loop
    u_prev = ux.clone()
    converged = False

    for it in range(max_iter):
        # Compute macroscopic
        rho, ux, uy = macroscopic(f)

        # Compute shear rate
        gamma = strain_rate_shear_rate_2d(ux, uy)

        # Compute effective viscosity (Papanastasiou HB)
        gamma_clamped = torch.clamp(gamma, min=1e-12)
        mu_eff = K * torch.pow(gamma_clamped, n - 1.0) + \
                 tau_y * (1.0 - torch.exp(-m_pap * gamma_clamped)) / gamma_clamped
        nu_eff = mu_eff  # rho = 1
        nu_eff = torch.clamp(nu_eff, min=1e-5, max=0.3)

        # Relaxation time field
        tau_field = 0.5 + 3.0 * nu_eff

        # BGK collision with variable tau
        feq = equilibrium(rho, ux, uy)
        tau_3d = tau_field.unsqueeze(0)
        f = f - (f - feq) / tau_3d

        # Streaming
        f = stream(f)

        # Bounce-back walls
        for i in range(9):
            j = opp[i]
            f[i, 0, :] = f[j, 0, :]
            f[i, -1, :] = f[j, -1, :]

        # Guo body force
        cx = c[:, 0].view(9, 1, 1)
        factor = (1.0 - 0.5 / tau_field)
        for i in range(9):
            f[i] += float(w[i]) * float(cx[i, 0, 0]) * G * factor

        # Convergence check
        if it % 200 == 0 and it > 0:
            rho_new, ux_new, _ = macroscopic(f)
            du = (ux_new - u_prev).abs().max().item()
            u_prev = ux_new.clone()
            if du < tol:
                converged = True
                break

    # Final profile
    rho, ux_final, uy_final = macroscopic(f)
    u_profile = ux_final[:, nx // 2].numpy()  # centerline cut

    # Analytical solution
    params = HBChannelParams(tau_y=tau_y, K=K, n=n, G=G, h=H / 2.0)
    y_anal, u_anal = hb_velocity_profile(params, n_points=ny - 2)
    u_anal_full = np.zeros(ny)
    u_anal_full[1:-1] = u_anal

    # Normalize for comparison
    u_max_anal = u_anal.max() if u_anal.max() > 0 else 1.0
    u_sim_norm = u_profile[1:-1] / max(u_profile.max(), 1e-15)
    u_anal_norm = u_anal / u_max_anal

    # L2 error
    l2_error = np.sqrt(np.mean((u_sim_norm - u_anal_norm) ** 2))

    # Mean velocity
    U_mean_sim = float(u_profile[1:-1].mean())
    U_mean_anal = hb_mean_velocity(params)

    return {
        "n": n,
        "tau_y": tau_y,
        "K": K,
        "G": G,
        "H": H,
        "iterations": it + 1,
        "converged": converged,
        "l2_error": l2_error,
        "U_mean_sim": U_mean_sim,
        "U_mean_anal": U_mean_anal,
        "U_mean_rel_error": abs(U_mean_sim - U_mean_anal) / max(abs(U_mean_anal), 1e-15),
        "u_profile": u_profile.tolist(),
        "u_analytical": u_anal_full.tolist(),
    }


def run_calibration_campaign():
    """Run the full laminar calibration campaign."""
    results = []

    # Parameters
    H = 32  # channel height
    nx = 64
    K = 0.02
    G = 1e-5

    # Test matrix: 3 values of n, multiple tau_y
    n_values = [0.5, 0.8, 1.0]
    tau_y_values = [0.0, 1e-5, 5e-5, 1e-4]

    print("=" * 70)
    print("LAMINAR HB CHANNEL CALIBRATION")
    print("=" * 70)
    print(f"Grid: H={H}, nx={nx}")
    print(f"K={K}, G={G}")
    print(f"n values: {n_values}")
    print(f"tau_y values: {tau_y_values}")
    print()

    for n in n_values:
        for tau_y in tau_y_values:
            # Check if flow exists
            params = HBChannelParams(tau_y=tau_y, K=K, n=n, G=G, h=H / 2.0)
            U_anal = hb_mean_velocity(params)
            if U_anal <= 0:
                print(f"  n={n}, tau_y={tau_y:.0e}: NO FLOW (below yield)")
                results.append({
                    "n": n, "tau_y": tau_y, "K": K, "G": G,
                    "status": "no_flow", "l2_error": None,
                })
                continue

            print(f"  n={n}, tau_y={tau_y:.0e}...", end=" ", flush=True)
            t0 = time.time()

            try:
                res = run_laminar_hb_2d(
                    H=H, nx=nx, tau_y=tau_y, K=K, n=n, G=G,
                    max_iter=8000, tol=1e-6,
                )
                elapsed = time.time() - t0
                res["elapsed_s"] = elapsed
                res["status"] = "ok" if res["converged"] else "not_converged"
                print(f"L2={res['l2_error']:.4f} U_mean_err={res['U_mean_rel_error']:.4f} "
                      f"({elapsed:.1f}s)")
                results.append(res)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {e} ({elapsed:.1f}s)")
                results.append({
                    "n": n, "tau_y": tau_y, "K": K, "G": G,
                    "status": "error", "error": str(e),
                })

    return results


def compute_metzner_reed_sweep():
    """Compute Re_MR and friction factor for a range of parameters.

    This generates the data for the friction factor collapse plot.
    """
    print("\n" + "=" * 70)
    print("METZNER-REED GENERALIZED REYNOLDS NUMBER SWEEP")
    print("=" * 70)

    rho = 1.0
    h = 16.0  # half-height

    # Sweep over n and Re_MR
    n_values = [0.5, 0.7, 0.8, 1.0, 1.2]
    Re_MR_targets = np.logspace(1, 4, 30)

    results = []
    for n in n_values:
        for Re_MR in Re_MR_targets:
            # For laminar: f = 24/Re_MR
            f_lam = dodge_metzner_laminar(n) / Re_MR

            # For turbulent: Dodge-Metzner correlation
            f_turb = None
            if Re_MR > 3000:  # approximate transition
                from tensorlbm.hb_channel_analytical import dodge_metzner_turbulent
                f_turb = dodge_metzner_turbulent(Re_MR, n)

            results.append({
                "n": n,
                "Re_MR": float(Re_MR),
                "f_laminar": float(f_lam),
                "f_turbulent": float(f_turb) if f_turb else None,
                "f_lam_times_Re": float(f_lam * Re_MR),
            })

    return results


def main():
    """Main entry point."""
    out_dir = REPO / "benchmarks" / "in45_hb_turb" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Laminar calibration
    cal_results = run_calibration_campaign()

    # 2. Metzner-Reed sweep
    mr_results = compute_metzner_reed_sweep()

    # 3. Save results
    output = {
        "schema": "in45-hb-turb-laminar-calibration-v1",
        "calibration": cal_results,
        "metzner_reed_sweep": mr_results,
        "summary": {
            "n_cases_run": sum(1 for r in cal_results if r.get("status") == "ok"),
            "n_cases_no_flow": sum(1 for r in cal_results if r.get("status") == "no_flow"),
            "n_cases_error": sum(1 for r in cal_results if r.get("status") == "error"),
            "max_l2_error": max(
                (r.get("l2_error", 0) or 0 for r in cal_results if r.get("status") == "ok"),
                default=0
            ),
            "mean_U_mean_rel_error": np.mean([
                r.get("U_mean_rel_error", 0)
                for r in cal_results if r.get("status") == "ok"
            ]) if any(r.get("status") == "ok" for r in cal_results) else None,
        },
    }

    out_path = out_dir / "laminar_calibration.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Cases run: {output['summary']['n_cases_run']}")
    print(f"Cases no flow: {output['summary']['n_cases_no_flow']}")
    print(f"Cases error: {output['summary']['n_cases_error']}")
    if output['summary']['max_l2_error']:
        print(f"Max L2 error: {output['summary']['max_l2_error']:.4f}")
    if output['summary']['mean_U_mean_rel_error'] is not None:
        print(f"Mean U_mean relative error: {output['summary']['mean_U_mean_rel_error']:.4f}")

    return output


if __name__ == "__main__":
    main()
