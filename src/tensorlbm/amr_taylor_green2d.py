"""Taylor-Green vortex decay with 2-level AMR (D2Q9) — flux conservation verification.

Physical background
-------------------
The Taylor-Green vortex is an exact solution of the incompressible Navier-Stokes
equations in 2D periodic domain. The velocity field decays exponentially:

    u(x,y,t) = -U0 cos(kx) sin(ky) exp(-2 nu k^2 t)
    v(x,y,t) = +U0 sin(kx) cos(ky) exp(-2 nu k^2 t)

where k = 2*pi/L, nu = (tau - 0.5)/3 (lattice units), and the kinetic energy
decays as E(t) = E0 * exp(-4 nu k^2 t).

This module runs the TG vortex on:
1. A uniform grid (reference)
2. A coarse grid with a fine patch covering the vortex core (AMR)

and compares the kinetic energy decay curves to verify that AMR does not
introduce spurious dissipation or conservation errors at the coarse-fine
interface.

Verification criteria
---------------------
- L2 error of kinetic energy decay curve (AMR vs uniform) <= 2%
- Mass drift over the simulation <= 1e-10
- Momentum conservation (net momentum stays zero) <= 1e-12

Scope
-----
* D2Q9 BGK collision, periodic boundary conditions
* 2-level structured AMR with ratio 2, Filippova-Hanel interface exchange
* Low Mach number (U0 <= 0.05) to ensure incompressible regime
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream


@dataclass
class TGAMRConfig:
    """Configuration for Taylor-Green AMR verification."""

    n: int = 64  # coarse grid size (n x n)
    re: float = 100.0  # Reynolds number Re = U0*L/nu
    u0: float = 0.05  # initial velocity amplitude
    steps: int = 5000  # number of coarse time steps
    record_every: int = 50  # record kinetic energy every N steps
    # AMR patch specification (coarse-grid coordinates)
    patch_y0: int = 16
    patch_y1: int = 48
    patch_x0: int = 16
    patch_x1: int = 48
    use_fh: bool = True  # use Filippova-Hanel interpolation


def taylor_green_init(
    n: int, u0: float, device: torch.device | str = "cpu", dtype=torch.float32
) -> torch.Tensor:
    """Initialize Taylor-Green vortex on periodic n x n grid."""
    k = 2.0 * math.pi / n
    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dtype),
        torch.arange(n, device=device, dtype=dtype),
        indexing="ij",
    )
    ux = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy = u0 * torch.sin(k * x) * torch.cos(k * y)
    rho = torch.ones((n, n), device=device, dtype=dtype)
    return equilibrium(rho, ux, uy)


def kinetic_energy(f: torch.Tensor) -> float:
    """Compute domain-averaged kinetic energy 0.5 * <u^2 + v^2>."""
    _, ux, uy = macroscopic(f)
    return float(0.5 * (ux**2 + uy**2).mean().item())


def total_mass(f: torch.Tensor) -> float:
    """Compute total mass (sum of density)."""
    rho, _, _ = macroscopic(f)
    return float(rho.sum().item())


def total_momentum(f: torch.Tensor) -> tuple[float, float]:
    """Compute total momentum (sum of rho*u, rho*v)."""
    rho, ux, uy = macroscopic(f)
    return float((rho * ux).sum().item()), float((rho * uy).sum().item())


def run_uniform(n: int, re: float, u0: float, steps: int, record_every: int) -> dict:
    """Run Taylor-Green on uniform grid (reference)."""
    device = torch.device("cpu")
    f = taylor_green_init(n, u0, device)
    L = float(n)
    nu = u0 * L / re
    tau = 0.5 + 3.0 * nu

    energies = []
    masses = []
    times = []

    for step in range(steps + 1):
        if step % record_every == 0:
            energies.append(kinetic_energy(f))
            masses.append(total_mass(f))
            times.append(float(step))
        if step < steps:
            f = collide_bgk(f, tau)
            f = stream(f)

    return {
        "times": times,
        "energies": energies,
        "masses": masses,
        "tau": tau,
        "nu": nu,
    }


def run_amr(cfg: TGAMRConfig) -> dict:
    """Run Taylor-Green with 2-level AMR.

    Uses a simple manual 2-level approach: coarse grid + one fine patch
    with 2:1 subcycling and Filippova-Hanel interface exchange.
    """
    from .adaptive_refinement import (
        _coarse_to_fine_2d,
        _fh_coarse_to_fine_2d,
        _fine_to_coarse_2d,
        _fh_fine_to_coarse_2d,
    )

    device = torch.device("cpu")
    n = cfg.n
    f_coarse = taylor_green_init(n, cfg.u0, device)
    L = float(n)
    nu = cfg.u0 * L / cfg.re
    tau_coarse = 0.5 + 3.0 * nu
    tau_fine = (tau_coarse - 0.5) * 2.0 + 0.5  # convective scaling

    # Extract coarse patch and create fine patch
    py0, py1 = cfg.patch_y0, cfg.patch_y1
    px0, px1 = cfg.patch_x0, cfg.patch_x1
    f_coarse_patch = f_coarse[:, py0:py1, px0:px1].clone()

    if cfg.use_fh:
        f_fine = _fh_coarse_to_fine_2d(f_coarse_patch, tau_coarse, tau_fine, ratio=2)
    else:
        f_fine = _coarse_to_fine_2d(f_coarse_patch, ratio=2)

    energies = []
    masses_coarse = []
    times = []

    # Main time loop: coarse takes 1 step, fine takes 2 sub-steps
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0:
            e = kinetic_energy(f_coarse)
            m_c = total_mass(f_coarse)
            energies.append(e)
            masses_coarse.append(m_c)
            times.append(float(step))

        if step < cfg.steps:
            # --- Coarse step ---
            f_coarse = collide_bgk(f_coarse, tau_coarse)
            f_coarse_post = f_coarse.clone()  # save for interface interpolation
            f_coarse = stream(f_coarse)

            # --- Fine sub-steps (2:1 subcycling) ---
            for sub in range(2):
                # Inject coarse boundary into fine patch halo
                # Time interpolation: sub=0 sees t^n, sub=1 sees t^{n+1}
                alpha = (sub + 1) / 2.0
                f_parent_blend = (1.0 - alpha) * f_coarse_patch + alpha * f_coarse[:, py0:py1, px0:px1]

                if cfg.use_fh:
                    f_boundary = _fh_coarse_to_fine_2d(f_parent_blend, tau_coarse, tau_fine, ratio=2)
                else:
                    f_boundary = _coarse_to_fine_2d(f_parent_blend, ratio=2)

                # Apply boundary to fine patch edges (1-cell halo)
                ny_f, nx_f = f_fine.shape[1], f_fine.shape[2]
                f_fine[:, 0, :] = f_boundary[:, 0, :]
                f_fine[:, -1, :] = f_boundary[:, -1, :]
                f_fine[:, :, 0] = f_boundary[:, :, 0]
                f_fine[:, :, -1] = f_boundary[:, :, -1]

                # Fine collide + stream
                f_fine = collide_bgk(f_fine, tau_fine)
                f_fine = stream(f_fine)

            # --- Restrict fine back to coarse ---
            f_restricted = _fine_to_coarse_2d(f_fine, ratio=2)
            f_coarse[:, py0:py1, px0:px1] = f_restricted
            f_coarse_patch = f_coarse[:, py0:py1, px0:px1].clone()

    return {
        "times": times,
        "energies": energies,
        "masses_coarse": masses_coarse,
        "tau_coarse": tau_coarse,
        "tau_fine": tau_fine,
        "nu": nu,
    }


def compare_energy_decay(
    uniform_result: dict, amr_result: dict
) -> dict:
    """Compare kinetic energy decay between uniform and AMR runs."""
    e_uni = uniform_result["energies"]
    e_amr = amr_result["energies"]
    n = min(len(e_uni), len(e_amr))

    # L2 relative error
    errors = [(e_amr[i] - e_uni[i]) ** 2 for i in range(n)]
    l2_error = math.sqrt(sum(errors) / n)
    e_max = max(abs(e) for e in e_uni)
    l2_relative = l2_error / e_max if e_max > 0 else l2_error

    # Pointwise max relative error
    max_rel_error = 0.0
    for i in range(n):
        if abs(e_uni[i]) > 1e-15:
            rel = abs(e_amr[i] - e_uni[i]) / abs(e_uni[i])
            max_rel_error = max(max_rel_error, rel)

    return {
        "l2_absolute": l2_error,
        "l2_relative": l2_relative,
        "max_relative_error": max_rel_error,
        "n_points": n,
    }


def run_verification(
    n: int = 64,
    re: float = 100.0,
    u0: float = 0.05,
    steps: int = 5000,
    record_every: int = 50,
) -> dict:
    """Run full TG AMR verification: uniform reference + AMR + comparison."""
    uniform = run_uniform(n, re, u0, steps, record_every)
    cfg = TGAMRConfig(
        n=n, re=re, u0=u0, steps=steps, record_every=record_every
    )
    amr = run_amr(cfg)
    comparison = compare_energy_decay(uniform, amr)

    # Mass conservation check
    mass_drift_coarse = abs(amr["masses_coarse"][-1] - amr["masses_coarse"][0])
    mass_drift_relative = (
        mass_drift_coarse / amr["masses_coarse"][0]
        if amr["masses_coarse"][0] > 0
        else mass_drift_coarse
    )

    return {
        "uniform": uniform,
        "amr": amr,
        "comparison": comparison,
        "mass_drift_coarse": mass_drift_coarse,
        "mass_drift_relative": mass_drift_relative,
        "config": {
            "n": n,
            "re": re,
            "u0": u0,
            "steps": steps,
            "record_every": record_every,
        },
    }
