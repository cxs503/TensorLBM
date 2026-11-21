"""Density-ratio reachability probes for the two-PDF partial-density
colour-gradient (Latva-Kokko / Rothman-Keller) model.

What this module is
-------------------
The validated two-PDF partial-density colour-gradient model
(:mod:`tensorlbm.color_gradient_two_pdf`) was cross-validated only at
**equal densities** (``rho = 1``, density ratio 1 -- the documented
restriction of the partial-density family, as in the LANL MF-LBM reference).

This module adds the **density-ratio** dimension: the two species carry
*different* pure-phase partial densities,

    heavy phase (phi = +1):  rho_a = rho_a0,  rho_b = 0
    light phase (phi = -1):  rho_a = 0,       rho_b = rho_b0
    density ratio           = rho_a0 / rho_b0

so that the total density in a pure region equals the physical density of
that phase.  The model equations are unchanged (the BGK collision, the
Guo forcing, the colour-gradient perturbation and the recolouring all act
on the partial densities); only the *initialisation* and the *diagnostics*
are new.

The probes provided here answer the reachability question for a
water/air-like density ratio (``rho_a0 / rho_b0 = 1000``):

1. **Static stratification** -- heavy fluid below, light fluid above, no
   initial velocity, no body force.  A stable model keeps the interface
   flat (no drift), keeps every partial density non-negative, keeps the
   colour field inside ``[-1, 1]`` and produces no spurious current
   (``u_max`` stays at round-off level).

2. **Static droplet** -- a heavy droplet suspended in a light bath (or the
   reverse), quiescent.  A stable model keeps the droplet spherical
   (radius drift small) and the interface sharp.

Both probes are *feasibility* tests: they do not claim quantitative
accuracy, they answer "does the model survive a 1000:1 density contrast
without blowing up, mixing or generating spurious motion?".

Additive: nothing in ``color_gradient_two_pdf`` / ``multiphase`` is
modified.
"""

from __future__ import annotations

import torch

from .boundary.boundaries import bounce_back_cells
from .color_gradient_two_pdf import (
    colorgradient_two_pdf_step,
    two_pdf_macros,
)
from .d2q9 import equilibrium
from .solver import stream

__all__ = [
    "init_stratified_ratio",
    "init_droplet_ratio",
    "run_stratified_probe",
    "run_droplet_probe",
    "spurious_current",
]


# ---------------------------------------------------------------------------
# Initialisation with a density ratio
# ---------------------------------------------------------------------------


def init_stratified_ratio(
    ny: int,
    nx: int,
    rho_a0: float,
    rho_b0: float,
    width: float = 2.0,
    dev: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """tanh colour profile with a density ratio ``rho_a0 / rho_b0``.

    Lower half = species A (heavy, ``phi = +1``, partial density
    ``rho_a0``); upper half = species B (light, ``phi = -1``, partial
    density ``rho_b0``).  Returns ``(f, g)`` equilibrium PDFs.
    """
    if rho_a0 <= 0.0 or rho_b0 <= 0.0:
        raise ValueError(f"pure-phase densities must be positive (got {rho_a0}, {rho_b0})")
    dt = torch.float64
    if dev is None:
        dev = torch.device("cpu")
    j = torch.arange(ny, dtype=dt, device=dev).view(ny, 1)
    half = ny // 2
    frac_a = 0.5 * (1.0 - torch.tanh((j - half) / max(width, 1e-6))).expand(ny, nx).contiguous()
    z = torch.zeros((ny, nx), dtype=dt, device=dev)
    rho_a = frac_a * rho_a0
    rho_b = (1.0 - frac_a) * rho_b0
    return equilibrium(rho_a, z, z, dev), equilibrium(rho_b, z, z, dev)


def init_droplet_ratio(
    n: int,
    radius: float,
    rho_a0: float,
    rho_b0: float,
    width: float = 2.0,
    dev: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A heavy droplet (species A) in a light bath (species B).

    ``frac_a`` is the heavy-phase fraction (1 inside the droplet, 0 in the
    bath); the partial densities are ``rho_a = frac_a * rho_a0`` and
    ``rho_b = (1 - frac_a) * rho_b0``.  Returns ``(f, g)`` equilibrium PDFs.
    """
    if rho_a0 <= 0.0 or rho_b0 <= 0.0:
        raise ValueError(f"pure-phase densities must be positive (got {rho_a0}, {rho_b0})")
    dt = torch.float64
    if dev is None:
        dev = torch.device("cpu")
    ys = torch.arange(n, dtype=dt, device=dev).view(n, 1)
    xs = torch.arange(n, dtype=dt, device=dev).view(1, n)
    rr = torch.sqrt((xs - n / 2.0 + 0.5) ** 2 + (ys - n / 2.0 + 0.5) ** 2)
    frac_a = 0.5 * (1.0 - torch.tanh((rr - radius) / max(width, 1e-6)))
    z = torch.zeros((n, n), dtype=dt, device=dev)
    rho_a = frac_a * rho_a0
    rho_b = (1.0 - frac_a) * rho_b0
    return equilibrium(rho_a, z, z, dev), equilibrium(rho_b, z, z, dev)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def spurious_current(f: torch.Tensor, g: torch.Tensor) -> float:
    """Maximum lattice speed ``|u|`` of the mixture (spurious-current proxy)."""
    _ra, _rb, _rho, _phi, ux, uy = two_pdf_macros(f, g)
    return float(torch.sqrt(ux**2 + uy**2).max().item())


def _droplet_radius(f: torch.Tensor, g: torch.Tensor) -> float:
    """Equivalent radius of the heavy phase (area / pi), in cells."""
    rho_a, _rb, _rho, _phi, _ux, _uy = two_pdf_macros(f, g)
    rho_a0 = float(rho_a.max().item())
    if rho_a0 <= 0:
        return 0.0
    area = float(rho_a.sum().item()) / rho_a0
    return (area / 3.141592653589793) ** 0.5


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def run_stratified_probe(
    ny: int,
    nx: int,
    rho_a0: float,
    rho_b0: float,
    nu_a: float,
    nu_b: float,
    A: float,
    beta: float,
    steps: int,
    width: float = 2.0,
    dev: torch.device | None = None,
    sample_every: int = 0,
) -> dict:
    """Static stratification probe at a given density ratio.

    Returns a dict with stability metrics: ``stable`` (bool), ``nan_step``,
    ``u_max`` (final spurious current), ``u_max_hist``, ``phi_range``,
    ``min_rho_a``, ``min_rho_b``, ``mass_drift_a_per_step``,
    ``mass_drift_b_per_step``, ``interface_drift_cells``.
    """
    if dev is None:
        dev = torch.device("cpu")
    f, g = init_stratified_ratio(ny, nx, rho_a0, rho_b0, width, dev)
    m_a0, m_b0 = float(f.sum()), float(g.sum())
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True

    # Record the initial interface position (first row where phi < -0.5).
    _ra0, _rb0, _rho0, phi0, _ux0, _uy0 = two_pdf_macros(f, g)
    j_first_b0 = _first_light_row(phi0)

    u_max_hist = []
    nan_step = None
    for step in range(1, steps + 1):
        f, g = colorgradient_two_pdf_step(
            f, g,
            nu_a=nu_a, nu_b=nu_b, A=A, beta=beta,
            gx=0.0, solid_mask=wall,
        )
        f = bounce_back_cells(stream(f), wall)
        g = bounce_back_cells(stream(g), wall)
        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all()):
            nan_step = step
            break
        if sample_every and step % sample_every == 0:
            u_max_hist.append((step, spurious_current(f, g)))

    if nan_step is not None:
        return {
            "stable": False, "nan_step": nan_step,
            "ny": ny, "nx": nx, "rho_a0": rho_a0, "rho_b0": rho_b0,
            "density_ratio": rho_a0 / rho_b0,
            "nu_a": nu_a, "nu_b": nu_b, "A": A, "beta": beta, "steps": steps,
        }

    rho_a, rho_b, _rho, phi, _ux, _uy = two_pdf_macros(f, g)
    j_first_b1 = _first_light_row(phi)
    return {
        "stable": True,
        "nan_step": None,
        "ny": ny, "nx": nx,
        "rho_a0": rho_a0, "rho_b0": rho_b0,
        "density_ratio": rho_a0 / rho_b0,
        "nu_a": nu_a, "nu_b": nu_b, "A": A, "beta": beta, "steps": steps,
        "u_max": spurious_current(f, g),
        "u_max_hist": u_max_hist,
        "phi_range": [float(phi.min().item()), float(phi.max().item())],
        "min_rho_a": float(rho_a.min().item()),
        "min_rho_b": float(rho_b.min().item()),
        "mass_drift_a_per_step": abs(float(f.sum()) - m_a0) / steps,
        "mass_drift_b_per_step": abs(float(g.sum()) - m_b0) / steps,
        "interface_drift_cells": (j_first_b1 - j_first_b0)
        if (j_first_b0 is not None and j_first_b1 is not None) else None,
    }


def run_droplet_probe(
    n: int,
    radius: float,
    rho_a0: float,
    rho_b0: float,
    nu_a: float,
    nu_b: float,
    A: float,
    beta: float,
    steps: int,
    width: float = 2.0,
    dev: torch.device | None = None,
    sample_every: int = 0,
) -> dict:
    """Static droplet probe at a given density ratio.

    Returns a dict with ``stable``, ``nan_step``, ``u_max``, ``radius_final``,
    ``radius_drift_rel``, ``phi_range``, ``min_rho_a``, ``min_rho_b``,
    ``mass_drift_a_per_step``, ``mass_drift_b_per_step``.
    """
    if dev is None:
        dev = torch.device("cpu")
    f, g = init_droplet_ratio(n, radius, rho_a0, rho_b0, width, dev)
    m_a0, m_b0 = float(f.sum()), float(g.sum())
    r0 = _droplet_radius(f, g)

    u_max_hist = []
    nan_step = None
    for step in range(1, steps + 1):
        f, g = colorgradient_two_pdf_step(
            f, g,
            nu_a=nu_a, nu_b=nu_b, A=A, beta=beta,
            gx=0.0,
        )
        f = stream(f)
        g = stream(g)
        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all()):
            nan_step = step
            break
        if sample_every and step % sample_every == 0:
            u_max_hist.append((step, spurious_current(f, g)))

    if nan_step is not None:
        return {
            "stable": False, "nan_step": nan_step,
            "n": n, "radius": radius,
            "rho_a0": rho_a0, "rho_b0": rho_b0,
            "density_ratio": rho_a0 / rho_b0,
            "nu_a": nu_a, "nu_b": nu_b, "A": A, "beta": beta, "steps": steps,
        }

    rho_a, rho_b, _rho, phi, _ux, _uy = two_pdf_macros(f, g)
    r1 = _droplet_radius(f, g)
    return {
        "stable": True,
        "nan_step": None,
        "n": n, "radius": radius,
        "rho_a0": rho_a0, "rho_b0": rho_b0,
        "density_ratio": rho_a0 / rho_b0,
        "nu_a": nu_a, "nu_b": nu_b, "A": A, "beta": beta, "steps": steps,
        "u_max": spurious_current(f, g),
        "u_max_hist": u_max_hist,
        "radius_final": r1,
        "radius_drift_rel": abs(r1 - r0) / r0 if r0 > 0 else None,
        "phi_range": [float(phi.min().item()), float(phi.max().item())],
        "min_rho_a": float(rho_a.min().item()),
        "min_rho_b": float(rho_b.min().item()),
        "mass_drift_a_per_step": abs(float(f.sum()) - m_a0) / steps,
        "mass_drift_b_per_step": abs(float(g.sum()) - m_b0) / steps,
    }


def _first_light_row(phi: torch.Tensor) -> int | None:
    """First row index where ``phi < -0.5`` (the light phase), or None."""
    col = phi[:, phi.shape[1] // 2]
    mask = (col < -0.5).to(torch.int64)
    if not bool(mask.any()):
        return None
    return int(torch.argmax(mask).item())
