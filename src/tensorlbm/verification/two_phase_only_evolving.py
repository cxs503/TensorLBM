"""Allen-Cahn-evolved two-phase-only arm (M2P-E) for the layered Poiseuille anchor.

Anchor
------
``XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY`` (OSS row ``oss_solutions`` 274,
``lssfau/walberla`` / ``GAP-LBM-MCMP-POISEUILLE``).  Round 1 of this anchor
(``xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/``) validated a *frozen* phase
indicator and declared the resulting gap in its own report: *"Nothing here
validates interface dynamics, coalescence or any flow with a non-zero normal
velocity at the interface."*  The OSS row's borrowable piece is a phase-field
(Allen-Cahn) layered initialisation, so the frozen indicator is a
simplification of that route, not the route itself.

This module closes that gap for the same benchmark: the phase indicator is
**advected and relaxed by an Allen-Cahn field**,

.. math::

    \partial_t \phi + \nabla\cdot(\phi\,u) = -M\,\mu, \qquad
    \mu = \beta\,\phi(\phi^2-1) - \kappa\,\nabla^2\phi ,

solved by the library's already-verified kernel
:func:`tensorlbm.allen_cahn_lbm.step_allen_cahn_field` (7-point Laplacian,
central-difference flux form, ``no_flux`` at the wall rows).  The viscosity is
recomputed from ``C = (1 + \phi)/2`` at every step, so the interface position
and width are *outputs* of the run rather than inputs.

What is reused (no physics re-implemented)
------------------------------------------
``tensorlbm.allen_cahn_lbm.free_energy_parameters`` / ``step_allen_cahn_field``
(phase-field transport), ``tensorlbm.d2q9.equilibrium`` /
``tensorlbm.solver.stream`` / ``tensorlbm.boundary.boundaries.bounce_back_cells``
(LBM chain) and ``tensorlbm.multiphase.guo_force_term`` (Guo forcing).  The
LBM operator chain below is the same eight lines as round 1's
``two_phase_only_step``; it is repeated here instead of imported so that the
round-1 study directory stays byte-frozen.

Two arms are provided:

``run_two_phase_only_evolving(..., evolve=True)``
    M2P-E, the evolving-interface arm.
``run_two_phase_only_evolving(..., evolve=False)``
    the round-1 model (frozen indicator) driven through the *same* code path,
    which is the control for the degradation check (round-3 ``E7``).

The analytic reference is the same two-layer Stokes solution as round 1:
:func:`two_layer_analytic` is imported lazily from the round-1 module so that
there is exactly one implementation of the reference in the repository.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..allen_cahn_lbm import free_energy_parameters, step_allen_cahn_field
from ..boundary.boundaries import bounce_back_cells
from ..d2q9 import equilibrium, macroscopic
from ..multiphase import guo_force_term
from ..solver import stream

__all__ = [
    "ac_parameters",
    "indicator_from_phase_field",
    "phase_field_from_indicator",
    "ac_shear_translation",
    "interface_diagnostics",
    "run_two_phase_only_evolving",
]


# ---------------------------------------------------------------------------
# Phase-field <-> indicator mapping and parameters
# ---------------------------------------------------------------------------


def ac_parameters(width: float, sigma: float = 0.1):
    """Allen-Cahn ``(beta, kappa)`` whose equilibrium width is ``width`` cells.

    ``sigma`` sets the free-energy scale (and hence the relaxation rate through
    ``M * sigma``); the equilibrium thickness ``sqrt(2 kappa / beta)`` does not
    depend on it.  A stratified channel has a *planar* interface, so the
    capillary (Korteweg) force is identically zero and no surface-tension force
    term is applied - only the double-well/gradient relaxation and the
    advective flux enter.
    """
    return free_energy_parameters(sigma=sigma, W=width)


def indicator_from_phase_field(phi: torch.Tensor) -> torch.Tensor:
    """``C = (1 + phi)/2``; ``phi = -tanh((y - y_int)/w)`` gives round 1's ``C``."""
    return 0.5 * (1.0 + phi)


def phase_field_from_indicator(C: torch.Tensor) -> torch.Tensor:
    """Inverse map ``phi = 2C - 1``."""
    return 2.0 * C - 1.0


def mix_viscosity(C: torch.Tensor, nu_lo: float, nu_hi: float, rule: str) -> torch.Tensor:
    """Node-local viscosity from the indicator (same three rules as round 1)."""
    if rule == "linear":
        return C * nu_lo + (1.0 - C) * nu_hi
    if rule == "geometric":
        return torch.exp(C * math.log(nu_lo) + (1.0 - C) * math.log(nu_hi))
    if rule == "harmonic":
        return 1.0 / (C / nu_lo + (1.0 - C) / nu_hi)
    raise ValueError(f"unknown viscosity rule {rule!r}")


# ---------------------------------------------------------------------------
# Verification of the Allen-Cahn advection term against an exact solution
# ---------------------------------------------------------------------------


def _wall_mask(ny: int, nx: int, device) -> torch.Tensor:
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    return mask


def _wavy_interface(
    ys: torch.Tensor,
    xs: torch.Tensor,
    *,
    y_c: float,
    amplitude: float,
    width: float,
    nx: int,
) -> torch.Tensor:
    """``phi = -tanh((y - y_int(x))/width)`` with a periodic ``y_int(x)``."""
    y_int = y_c + amplitude * torch.sin(2.0 * math.pi * xs / nx)
    return -torch.tanh((ys - y_int) / width)


def ac_shear_translation(
    ny: int,
    nx: int,
    *,
    gamma: float = 0.01,
    amplitude: float = 1.6,
    width: float = 2.0,
    sigma: float = 0.1,
    mobility: float = 0.0,
    steps: int = 200,
    dt: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Verify the Allen-Cahn *advection* term against an exact solution.

    The velocity field is the linear shear ``u_x = gamma (y - y_c)``, ``u_y = 0``
    (solenoidal), and the initial interface position ``y_int(x)`` is periodic in
    ``x`` with amplitude ``amplitude`` (cells).  Because the AC relaxation is
    switched off (``mobility = 0``) and ``div u = 0``, the pure-advection problem

        ``d phi/dt + div(phi u) = 0``

    has the exact solution ``phi(x, y, t) = phi0((x - gamma (y - y_c) t) mod nx, y)``
    - the interface is carried along the characteristic without distortion.
    The returned ``l2_rel`` is measured against that exact field on the interior
    rows, so it isolates the advection discretisation (7-point stencil, central
    fluxes, explicit Euler) with no LBM in the loop, and no reference of the
    solvers' own making.

    A second-order advection scheme must show an observed order close to 2 under
    simultaneous space/time refinement; the caller reports whichever value comes
    out (round 3, check ``E6``).
    """
    torch.set_grad_enabled(False)
    y_c = 0.5 * (ny - 1)
    ys = torch.arange(ny, dtype=dtype).view(1, ny, 1)
    xs = torch.arange(nx, dtype=dtype).view(1, 1, nx)
    phi = _wavy_interface(ys, xs, y_c=y_c, amplitude=amplitude, width=width, nx=nx)
    ux = gamma * (ys - y_c) * torch.ones_like(phi)
    uy = torch.zeros_like(phi)
    uz = torch.zeros_like(phi)
    p = ac_parameters(width, sigma)
    phi_initial = phi.clone()
    budget0 = float(phi[:, 1:-1, :].sum())
    l1_0 = float(phi[:, 1:-1, :].abs().sum())
    for _ in range(steps):
        phi = step_allen_cahn_field(
            phi,
            beta=p.beta,
            kappa=p.kappa,
            mobility=mobility,
            dt=dt,
            u=(ux, uy, uz),
            boundary="periodic",
        )
    total_t = steps * dt
    shift = gamma * (ys - y_c) * total_t
    exact = _wavy_interface(
        ys, (xs - shift) % nx, y_c=y_c, amplitude=amplitude, width=width, nx=nx
    )
    fluid = slice(1, ny - 1)
    diff = (phi - exact)[:, fluid, :]
    denom = float(torch.linalg.norm(exact[:, fluid, :]))
    return {
        "config": {
            "ny": ny, "nx": nx, "gamma": gamma, "amplitude": amplitude,
            "width": width, "sigma": sigma, "mobility": mobility,
            "steps": steps, "dt": dt, "dtype": str(dtype),
            "total_time": total_t,
        },
        "l2_rel": float(torch.linalg.norm(diff)) / denom if denom else float("nan"),
        "max_abs_err": float(diff.abs().max()),
        # Warm-start budget: a wavy interface integrates to ~0 over a symmetric
        # channel, so the *absolute* mean is the wrong denominator; the L1 norm
        # of the initial field is used instead and the check is a conservation
        # check (``E6`` companion), not a physics band.
        "phase_budget_rel": abs(float(phi[:, fluid, :].sum()) - float(phi_initial[:, fluid, :].sum()))
        / (l1_0 if l1_0 else float("nan")),
        "phi_range": [float(phi.min()), float(phi.max())],
        "finite": bool(torch.isfinite(phi).all()),
        "initial_amplitude": amplitude,
        "max_advective_cfl": float(
            (gamma * max(abs(y_c - 1.0), abs(ny - 2 - y_c)) * dt)
        ),
        "phi_initial": phi_initial,
    }


# ---------------------------------------------------------------------------
# Phase-field interface diagnostics
# ---------------------------------------------------------------------------


def _crossing(y: np.ndarray, v: np.ndarray, level: float) -> float | None:
    d = v - level
    s = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if s.size == 0:
        return None
    i = int(s[0])
    dv = d[i + 1] - d[i]
    if dv == 0:
        return float(y[i])
    return float(y[i] - d[i] * (y[i + 1] - y[i]) / dv)


def interface_diagnostics(profile: np.ndarray, y_int: float, first_row: int = 1) -> dict:
    """Centre (``C = 1/2`` crossing) and 10-90 width of a monotone ``C(y)``.

    ``first_row`` is the lattice row of ``profile[0]``: pass ``1`` when the
    caller has stripped the two wall rows, ``0`` for an all-row profile.
    ``centre_shift`` is the distance of the crossing from the analytic interface
    plane ``y_int`` and is what round-3 check ``E3`` gates.
    """
    p = np.asarray(profile, dtype=np.float64)
    y = np.arange(len(p), dtype=np.float64) + float(first_row)
    lo = float(np.median(p[: max(2, len(p) // 10)]))
    hi = float(np.median(p[-max(2, len(p) // 10) :]))
    out = {
        "plateau_below": lo,
        "plateau_above": hi,
        "centre": None,
        "width_10_90": None,
        "centre_shift": None,
        "amplitude_retained": abs(hi - lo),
    }
    if abs(hi - lo) < 1e-6:
        return out
    # Normalise so that the crossing is always looked for at level 0.5.
    norm = (p - lo) / (hi - lo)
    centre = _crossing(y, norm, 0.5)
    y10 = _crossing(y, norm, 0.1)
    y90 = _crossing(y, norm, 0.9)
    out["centre"] = centre
    out["width_10_90"] = None if (y10 is None or y90 is None) else abs(y90 - y10)
    out["centre_shift"] = None if centre is None else abs(centre - y_int)
    return out


# ---------------------------------------------------------------------------
# Coupled arm: two-phase-only with an Allen-Cahn-evolved indicator
# ---------------------------------------------------------------------------


def _two_layer_analytic():
    """Load the packaged round-1 reference; no source checkout is required."""
    from . import two_phase_only

    return two_phase_only


def run_two_phase_only_evolving(
    ny: int,
    nx: int = 8,
    *,
    G: float = 4e-6,
    nu_lo: float = 0.5,
    nu_hi: float = 0.25,
    width: float = 2.0,
    width_init: float | None = None,
    rule: str = "harmonic",
    sigma: float = 0.1,
    mobility: float = 1e-2,
    dt_ac: float = 1.0,
    evolve: bool = True,
    max_steps: int = 40_000,
    stop_tol: float = 1e-6,
    check_every: int = 500,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Two-layer Poiseuille with an Allen-Cahn-evolved phase field.

    ``evolve=False`` freezes the indicator (round-1 model) but drives the run
    through the same code path, so the two arms differ *only* in whether the
    phase field is advanced.

    ``width`` is the Allen-Cahn equilibrium thickness ``W`` of the double well;
    ``width_init`` (default: ``width``) is the thickness of the tanh profile the
    layering is initialised with - the OSS route takes the layering from the
    initial ``C``, so a mismatch between the two is a legitimate initialisation
    and is used by the round-3 width-mismatch arm.

    The phase field is advanced with the velocity of the previous LBM step
    (explicit coupling, as in the library's own coupled stepper), and the
    viscosity field - hence ``tau`` - is recomputed from the new indicator every
    step, so neither the interface position nor the width is an input after
    ``t = 0``.
    """
    torch.set_grad_enabled(False)
    device = torch.device(device)
    r1 = _two_layer_analytic()
    y_int = 0.5 * (ny - 1)

    width_init = width if width_init is None else width_init
    ys = torch.arange(ny, dtype=dtype, device=device).view(ny, 1).expand(ny, nx)
    C = 0.5 * (1.0 - torch.tanh((ys - y_int) / width_init))
    phi = phase_field_from_indicator(C).unsqueeze(0)  # (1, ny, nx)
    wall = _wall_mask(ny, nx, device)
    wall3 = wall.unsqueeze(0)
    p = ac_parameters(width, sigma)

    nu = mix_viscosity(C, nu_lo, nu_hi, rule)
    tau = 0.5 + 3.0 * nu
    rho0 = torch.ones_like(C, dtype=dtype)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    force_x = rho0 * G

    sol = r1.two_layer_analytic(ny, y_int, G, nu_lo, nu_hi)
    u_ana = r1.analytic_profile(sol, ny)

    C0_profile = C.mean(dim=1).double().cpu().numpy()
    diag0 = interface_diagnostics(C0_profile, y_int, first_row=0)
    budget0 = float(C[1:-1, :].sum())
    mass0 = float(rho0[1:-1, :].sum())

    history: list[dict] = []
    prev = None
    steps = 0
    converged = False
    t0 = time.time()
    for step in range(1, max_steps + 1):
        if evolve:
            rho_now, ux_now, uy_now = macroscopic(f)
            phi = step_allen_cahn_field(
                phi,
                beta=p.beta,
                kappa=p.kappa,
                mobility=mobility,
                dt=dt_ac,
                u=(ux_now.unsqueeze(0), uy_now.unsqueeze(0), torch.zeros_like(phi)),
                boundary="no_flux",
                solid_mask=wall3,
            )
            C = indicator_from_phase_field(phi.squeeze(0))
            nu = mix_viscosity(C, nu_lo, nu_hi, rule)
            tau = 0.5 + 3.0 * nu
        # LBM operator chain (identical to round 1's ``two_phase_only_step``).
        rho, ux, uy = macroscopic(f)
        rho_s = torch.clamp(rho, min=1e-12)
        ux = ux + 0.5 * force_x / rho_s
        feq = equilibrium(rho, ux, uy)
        f = f - (f - feq) / tau
        f = f + guo_force_term(rho, ux, uy, force_x, torch.zeros_like(force_x), tau)
        f = stream(f)
        f = bounce_back_cells(f, wall)
        steps = step
        if step % check_every == 0:
            rho, ux, uy = macroscopic(f)
            u_peak = float(ux[1:-1, :].abs().max())
            if not torch.isfinite(f).all():
                break
            if prev is not None and abs(u_peak - prev) <= stop_tol * max(abs(prev), 1e-12):
                converged = True
                break
            prev = u_peak
            history.append({"step": int(step), "u_peak": u_peak})
    wall_s = time.time() - t0

    rho, ux, uy = macroscopic(f)
    prof = ux.mean(dim=1).double().cpu().numpy()
    uy_prof = uy.double().cpu().numpy()
    C_prof = indicator_from_phase_field(phi.squeeze(0)).mean(dim=1).double().cpu().numpy()
    diag = interface_diagnostics(C_prof, y_int, first_row=0)
    fluid = slice(1, ny - 1)
    sim, ana = prof[fluid], u_ana[fluid]
    denom = float(np.linalg.norm(ana))
    u_peak_ana = float(np.abs(ana).max()) if np.abs(ana).max() else float("nan")
    budget = float(indicator_from_phase_field(phi.squeeze(0))[1:-1, :].sum())
    width_growth = None
    if diag0["width_10_90"] and diag["width_10_90"]:
        width_growth = diag["width_10_90"] / diag0["width_10_90"] - 1.0
    return {
        "config": {
            "ny": ny, "nx": nx, "G": G, "nu_lo": nu_lo, "nu_hi": nu_hi,
            "nu_ratio": nu_lo / nu_hi, "width": width,
            "width_init": width_init, "rule": rule,
            "sigma": sigma, "mobility": mobility if evolve else 0.0,
            "dt_ac": dt_ac, "evolve": bool(evolve), "max_steps": max_steps,
            "stop_tol": stop_tol, "check_every": check_every,
            "dtype": str(dtype),
        },
        "steps": steps,
        "converged": converged,
        "finite": bool(torch.isfinite(f).all()),
        "wall_s": wall_s,
        "l2_rel": float(np.linalg.norm(sim - ana) / denom) if denom else float("nan"),
        "max_rel_peak": float(np.max(np.abs(sim - ana))) / u_peak_ana,
        "u_peak_sim": float(np.abs(sim).max()),
        "u_peak_ana": u_peak_ana,
        "ma_max": float(np.abs(sim).max()) / math.sqrt(1.0 / 3.0),
        "uy_spurious_rel": float(np.abs(uy_prof[fluid, :]).max()) / max(
            float(np.abs(prof[fluid]).max()), 1e-30
        ),
        "mass_drift_rel": abs(float(rho[1:-1, :].sum()) - mass0) / mass0,
        "phase_field_budget_rel": abs(budget - budget0) / budget0 if budget0 else float("nan"),
        "interface": {
            "initial": diag0,
            "final": diag,
            "width_growth_rel": width_growth,
        },
        "profile": [float(v) for v in prof],
        "profile_ana": [float(v) for v in u_ana],
        "C_profile": [float(v) for v in C_prof],
        "analytic_checks": sol.residuals(),
        "history": history,
    }