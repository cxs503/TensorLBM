"""Audit helpers for the LNG membrane-tank sloshing campaign.

Two independent checks that do not depend on the impact theory:

``gci``                Roache's grid-convergence index from a 3-grid family at a
                       fixed refinement ratio ``r`` (with the observed order of
                       accuracy and the standard safety factor ``Fs = 1.25``).
``energy_budget``      mechanical-energy closure of the free-surface LBM run:
                       the change of ``E = KE + PE`` over a window is compared
                       with the work done by the excitation and the viscous
                       dissipation measured from the non-equilibrium stress,
                       ``eps = (tau - 1/2) / (2 c_s^2 tau^2) * P_neq:P_neq /
                       rho`` (see :func:`viscous_dissipation_2d`).

Both are reported as numbers in the campaign report; neither is fitted.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "richardson_observed_order",
    "gci",
    "gci_family",
    "viscous_dissipation_2d",
    "mechanical_energy",
    "energy_budget",
]

CS2 = 1.0 / 3.0


def richardson_observed_order(f_coarse: float, f_mid: float, f_fine: float, r: float) -> float:
    """Observed order of accuracy ``p`` from three grids (``r`` = refinement ratio).

    ``p = ln(|(f_mid - f_fine)/(f_coarse - f_mid)|) / ln(r)``.  Returns ``nan``
    when the family is not monotone (``|f_coarse - f_mid|`` ~ 0 or the ratio is
    not positive), which is itself a reported outcome.
    """
    if r <= 1.0:
        raise ValueError("refinement ratio must exceed 1")
    d_coarse = abs(f_mid - f_coarse)
    d_fine = abs(f_fine - f_mid)
    if d_coarse < 1e-30 or d_fine < 1e-30:
        return float("nan")
    ratio = d_coarse / d_fine
    if ratio <= 0.0:
        return float("nan")
    return float(math.log(ratio) / math.log(r))


def gci(f_coarse: float, f_mid: float, f_fine: float, r: float, fs: float = 1.25) -> dict:
    """Fine-grid GCI (Roache 1994) plus the Richardson-extrapolated value.

    Returns ``{"p_obs", "f_exact", "gci_fine", "gci_fine_pct", "monotone"}``.
    For a non-monotone family ``gci_fine`` is evaluated with ``|p| = 1`` (the
    conservative convention used when no observed order exists) and
    ``monotone = False`` is reported.
    """
    p = richardson_observed_order(f_coarse, f_mid, f_fine, r)
    d_coarse = abs(f_mid - f_coarse)
    d_fine = abs(f_fine - f_mid)
    monotone = (not math.isnan(p)) and p > 0.0 and d_fine < d_coarse
    p_use = abs(p) if monotone else 1.0
    denom = r**p_use - 1.0
    relative = abs((f_fine - f_mid) / f_fine) if f_fine != 0.0 else float("nan")
    value = fs * relative / denom
    f_exact = (
        f_fine + (f_fine - f_mid) / denom if monotone else float("nan")
    )
    return {
        "p_obs": p if monotone else float("nan"),
        "p_used": p_use,
        "f_exact": f_exact,
        "gci_fine": value,
        "gci_fine_pct": 100.0 * value,
        "monotone": monotone,
        "r": r,
    }


def gci_family(values: list[float], r: float, fs: float = 1.25) -> dict:
    """``gci`` on ``[coarse, mid, fine]``."""
    if len(values) != 3:
        raise ValueError("need exactly three grid values")
    return gci(values[0], values[1], values[2], r, fs)


def _lattice_constants(device=None):
    """D2Q9 velocity set / weights -- imported, never re-declared.

    The ordering must match :mod:`tensorlbm.d2q9` exactly (the solver's
    populations are ordered by ``C``); a private copy of the velocity set would
    silently mis-pair ``f_q`` with ``c_q`` and produce a spurious non-equilibrium
    stress.
    """
    from ..d2q9 import C as C_REPO
    from ..d2q9 import W as W_REPO

    c = C_REPO.to(device=device if device is not None else "cpu", dtype=torch.float32)
    w = W_REPO.to(device=device if device is not None else "cpu", dtype=torch.float32)
    return c[:, 0].clone(), c[:, 1].clone(), w.clone()


def viscous_dissipation_2d(f: torch.Tensor, rho: torch.Tensor, u: torch.Tensor, tau: float):
    """Discrete viscous dissipation rate per cell (lattice units) [energy/step].

    For BGK the non-equilibrium stress is ``P_neq = -2 rho c_s^2 tau S`` with
    ``S`` the strain-rate tensor, so the dissipation rate

        eps = 2 rho nu S:S = nu P_neq:P_neq / (2 rho c_s^4 tau^2)

    is evaluated exactly from the populations (no gradient stencils).  Returns a
    per-cell tensor (only fluid cells are meaningful; solid cells are ~0).
    """
    w = _lattice_constants(f.device)[2]
    cx, cy, _ = _lattice_constants(f.device)
    f_eq = _equilibrium(rho, u)
    d = f - f_eq
    cxx = (cx * cx).view(9, 1, 1)
    cyy = (cy * cy).view(9, 1, 1)
    cxy = (cx * cy).view(9, 1, 1)
    pxx = (d * cxx).sum(dim=0)
    pyy = (d * cyy).sum(dim=0)
    pxy = (d * cxy).sum(dim=0)
    sq = pxx * pxx + pyy * pyy + 2.0 * pxy * pxy
    nu = (tau - 0.5) / 3.0
    return nu * sq / (2.0 * rho.clamp(min=1e-9) * CS2**2 * tau**2)


def _equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Equilibrium populations from the repository's D2Q9 kernel (single source)."""
    from ..d2q9 import equilibrium as _eq

    return _eq(rho, u[..., 0], u[..., 1])


def macroscopic_velocity(f: torch.Tensor, rho: torch.Tensor, fluid_mask: torch.Tensor):
    """Momentum/rho velocity field on the fluid mask (zeros elsewhere)."""
    cx, cy, _ = _lattice_constants(f.device)
    mom_x = (f * cx.view(9, 1, 1)).sum(dim=0)
    mom_y = (f * cy.view(9, 1, 1)).sum(dim=0)
    safe = rho.clamp(min=1e-9)
    ux = torch.where(fluid_mask, mom_x / safe, torch.zeros_like(mom_x))
    uy = torch.where(fluid_mask, mom_y / safe, torch.zeros_like(mom_y))
    return torch.stack([ux, uy], dim=-1)


def mechanical_energy(
    f: torch.Tensor,
    content: torch.Tensor,
    fluid_mask: torch.Tensor,
    *,
    g: float,
    rho_liquid: float = 1.0,
) -> dict[str, float]:
    """KE and PE of the liquid (lattice units; energies per unit depth).

    ``KE = sum 1/2 rho u^2`` over the liquid/gas cells; ``PE = sum rho_liquid g
    y content`` with ``content`` the sub-cell liquid inventory (cell-centre
    height above the tank bottom).  The gas is at rest and massless in the
    Körner model, so it contributes neither term.
    """
    rho = f.sum(dim=0)
    u = macroscopic_velocity(f, rho, fluid_mask)
    ke = float((0.5 * rho * (u[..., 0] ** 2 + u[..., 1] ** 2)).sum())
    ny = content.shape[0]
    y = torch.arange(ny, dtype=content.dtype, device=content.device).view(-1, 1) - 0.5
    pe = float((rho_liquid * g * y.clamp(min=0.0) * content).sum())
    return {"ke": ke, "pe": pe, "total": ke + pe}


def energy_budget(
    e_start: dict[str, float],
    e_end: dict[str, float],
    *,
    work_excitation: float,
    dissipated: float,
    e_ref: float,
) -> dict[str, float]:
    """Closure residual of ``dE = W_exc - W_diss`` normalised by ``e_ref``."""
    if e_ref <= 0.0:
        raise ValueError("e_ref must be positive")
    delta_e = e_end["total"] - e_start["total"]
    residual = delta_e - (work_excitation - dissipated)
    return {
        "delta_e": delta_e,
        "work_excitation": work_excitation,
        "dissipated": dissipated,
        "residual": residual,
        "residual_relative": abs(residual) / e_ref,
        "closure_unclosed": abs(residual) / e_ref,
        "ke_start": e_start["ke"],
        "pe_start": e_start["pe"],
        "ke_end": e_end["ke"],
        "pe_end": e_end["pe"],
        "e_ref": e_ref,
    }