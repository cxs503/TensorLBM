#!/usr/bin/env python3
"""Diagnostic: reproduce CAC-G divergence at step 3 and analyze root cause."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Use the repo root's src directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.cac_lbm import (
    CACSim,
    diagnostics,
    init_phi_layers,
    iso_gradient_3d,
    iso_laplacian_3d,
    shifted_neighbors,
    chemical_potential,
    lambda_phi,
    normal_from_gradient,
    f_source,
    g_force,
    f_equilibrium,
    g_equilibrium,
    macroscopic_g,
    rho_of_phi,
    CS2,
)

torch.set_num_threads(1)


def run_diagnostic(
    nz: int = 64,
    ny: int = 16,
    nx: int = 16,
    steps: int = 10,
    dtype: torch.dtype = torch.float64,
):
    """Run a few steps and print detailed diagnostics."""
    shape = (nz, ny, nx)
    width = 4.0
    y0 = float(ny) / 2.0

    phi0 = init_phi_layers(shape, width, y0, dtype=dtype)

    sim = CACSim(
        shape,
        rho_l=1.0,
        rho_g=0.1,
        mu_l=0.1,
        mu_g=0.01,
        sigma=0.01,
        width=width,
        mobility=0.05,
        mu_law="harmonic",
        p0=0.01,
        gravity=(0.0, 0.0, -1e-5),
        gravity_mode="rho",
        axes="wpp",
        device="cpu",
        dtype=dtype,
    )
    sim.initialize(phi0)

    print("=" * 80)
    print("CAC-G Divergence Diagnostic")
    print("=" * 80)
    print(f"Grid: {nz}x{ny}x{nx}, dtype={dtype}")
    print(f"tau_f = {sim.tau_f:.4f}")
    print(f"w0 = {sim.w0:.6f}")
    print()

    # Initial state
    d = diagnostics(sim)
    print(f"Step 0 (init):")
    print(f"  phi range: [{d['phi_min_fluid']:.6e}, {d['phi_max_fluid']:.6e}]")
    print(f"  u_max: {d['u_max_fluid']:.6e}")
    print(f"  nan_f={d['nan_f']}, nan_g={d['nan_g']}")
    print(f"  phi_integral={d['phi_integral_fluid']:.6e}")
    print()

    for step in range(1, steps + 1):
        # Pre-step diagnostics
        phi = sim.f.sum(dim=0)
        rho = rho_of_phi(phi, sim.rho_l, sim.rho_g)
        nb = shifted_neighbors(phi, axes=sim.axes)
        grad = iso_gradient_3d(nb, sim.device)
        mu_phi = chemical_potential(phi, nb, sim.beta, sim.k)
        lam = lambda_phi(phi, sim.width)
        n_hat = normal_from_gradient(*grad)

        phi_min = float(phi[sim.fluid_mask].min().item())
        phi_max = float(phi[sim.fluid_mask].max().item())
        lam_min = float(lam[sim.fluid_mask].min().item())
        lam_max = float(lam[sim.fluid_mask].max().item())

        print(f"Step {step} (pre-collision):")
        print(f"  phi range: [{phi_min:.6e}, {phi_max:.6e}]")
        print(f"  lambda range: [{lam_min:.6e}, {lam_max:.6e}]")

        fm = sim.fluid_mask
        rho_safe = torch.where(fm, rho, torch.ones_like(rho))
        tau_g = 0.5 + 3.0 * sim.mu_l / rho_safe
        inv_tau_g = torch.where(fm, 1.0 / tau_g, torch.zeros_like(tau_g)).unsqueeze(0)

        if sim.gravity_mode == "rho":
            rho_fm = rho * fm.to(rho.dtype)
            F = (
                mu_phi * grad[0] + sim.gravity[0] * rho_fm,
                mu_phi * grad[1] + sim.gravity[1] * rho_fm,
                mu_phi * grad[2] + sim.gravity[2] * rho_fm,
            )
        else:
            F = (
                mu_phi * grad[0] + sim.gravity[0] * fm,
                mu_phi * grad[1] + sim.gravity[1] * fm,
                mu_phi * grad[2] + sim.gravity[2] * fm,
            )

        u, p, u_star = macroscopic_g(sim.g, rho, grad, sim.drho, F, sim.w0)

        phi_u_now = tuple(phi * u[a] for a in range(3))
        phi_u_dt = tuple(phi_u_now[a] - sim.phi_u_prev[a] for a in range(3))

        fsrc = f_source(phi_u_dt, lam, n_hat, sim.tau_f, sim.wv, sim.cf)
        gsrc = g_force(F, u, grad, sim.drho, inv_tau_g, sim.wv, sim.cf)

        fsrc_max = float(fsrc.abs().max().item())
        gsrc_max = float(gsrc.abs().max().item())
        feq_max = float(f_equilibrium(phi, u, sim.wv, sim.cf).abs().max().item())
        geq_max = float(g_equilibrium(p, rho, u, sim.wv, sim.cf, sim.w0).abs().max().item())

        print(f"  u_max: {max(float(u[a].abs().max().item()) for a in range(3)):.6e}")
        print(f"  fsrc_max: {fsrc_max:.6e}, gsrc_max: {gsrc_max:.6e}")
        print(f"  feq_max: {feq_max:.6e}, geq_max: {geq_max:.6e}")

        if fsrc_max > feq_max * 0.1:
            print(f"  WARNING: fsrc_max is {fsrc_max/feq_max*100:.1f}% of feq_max!")
        if gsrc_max > geq_max * 0.1:
            print(f"  WARNING: gsrc_max is {gsrc_max/geq_max*100:.1f}% of geq_max!")

        sim.step()

        d = diagnostics(sim)
        print(f"  post-step: phi=[{d['phi_min_fluid']:.6e}, {d['phi_max_fluid']:.6e}], "
              f"u_max={d['u_max_fluid']:.6e}, nan_f={d['nan_f']}, nan_g={d['nan_g']}")

        if d['nan_f'] > 0 or d['nan_g'] > 0:
            print(f"  *** DIVERGENCE at step {step} ***")
            break

        print()

    return sim


if __name__ == "__main__":
    sim = run_diagnostic(nz=64, ny=16, nx=16, steps=10, dtype=torch.float64)
