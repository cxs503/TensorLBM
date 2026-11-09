"""Cavitation-solver numerics: audit-grade closures and consistency checks.

Mission P40W3B-N02 (sheet / cloud cavitation validation face).  The companion
modules are

* :mod:`tensorlbm.cavitation_airfoil` -- 2-D homogeneous-mixture cavitating
  Navier-Stokes solver (the engine that the P41-N02 validation surface used);
* :mod:`tensorlbm.cavitation_free_streamline` -- panel-method free-streamline
  (constant-pressure cavity) engine.

This module collects the **numerical-consistency layer** that the P41-N02
post-mortem showed was missing: every quantity below is a closed-form (or
one-dimensional quadrature) relation that the solver must satisfy, so the
defects of the mixture engine can be measured instead of guessed.

Four relations are implemented.

1. ``pressure_anchor`` -- the fractional-step projection solves a pure-Neumann
   pressure Poisson problem, so the computed ``p`` is defined only up to a
   constant, while the Schnerr-Sauer mass-transfer closure compares the
   *absolute* pressure with ``p_v``.  The physical level is fixed by the inflow
   (``p = p_inf = 0`` on the inlet plane), which gives the offset reported here.
   Without this offset the evaporation switch ``p < p_v`` is decided by an
   arbitrary constant (measured: offsets of -2.2 ... -5.9 in the P41-N02 runs,
   i.e. several times ``|p_v| = sigma/2``).

2. ``brinkman_leakage`` -- the immersed-body term divides the predictor by
   ``1 + dt/eta`` per step, so the residual slip inside the body decays only to
   ``1/(1 + dt/eta)``.  With the P41-N02 settings (``eta = 1e-3``, ``dt = cfl*h = 0.004``)
   that is a 20 % slip -- the effective body is a leaking staircase and the
   penalisation reaction is not the hydrodynamic force (measured ``Cd = 1.04``
   against ``~0.02`` expected for a NACA0015 at 5 deg / Re = 1e5).

3. ``schnerr_sauer_growth_time`` vs ``rayleigh_plesset_time`` -- the closure
   ``S_alpha = 3 alpha_l alpha_v / R_b sqrt(2|dp|/(3 rho_l))`` is exponential in
   ``alpha`` for fixed ``dp`` when ``R_b`` is held at the sub-grid floor, with
   e-folding time ``R_b / sqrt(2 dp / (3 rho_l))``.  The Rayleigh-Plesset
   linearised bubble has ``omega = sqrt(3 dp / (rho R^2))``.  The ratio is a
   constant ``sqrt(9/2) = 2.12``: the closure is *slower* than the physical
   inertia time, which is why a run of a few convective times cannot grow a
   cavity out of a ``1e-6`` void-fraction seed.

4. ``integrate_closure_growth`` -- integrates the closure ODE at fixed ``dp``
   and returns the measured void-fraction growth exponent, which must approach
   the Rayleigh-Plesset asymptote ``R ~ t sqrt(2 dp / (3 rho))`` (``alpha ~ t^3``
   once the bubble leaves the floor regime).  This is the *dynamic* verification
   of the phase-change source (the static prefactor is anchored in
   ``tests/test_cavitation_schnerr_sauer_reference.py``).

All functions are pure ``torch``/``float`` utilities: no hidden state, no
domain data, no I/O.
"""

from __future__ import annotations

import math

import torch

from .cavitation_airfoil import (
    CavitationFlowConfig,
    cavity_metrics,
    schnerr_sauer_rate,
)

__all__ = [
    "brinkman_leakage",
    "brinkman_eta_for_leakage",
    "cavity_length_isobaric",
    "integrate_closure_growth",
    "pressure_anchor",
    "rayleigh_plesset_time",
    "schnerr_sauer_growth_time",
    "stability_budget",
]


# ---------------------------------------------------------------------------
# 1. pressure level
# ---------------------------------------------------------------------------


def pressure_anchor(
    p: torch.Tensor,
    inlet_columns: int = 3,
    airfoil_mask: torch.Tensor | None = None,
) -> float:
    """Offset that puts ``p = p_inf = 0`` on the inlet plane.

    The fractional-step projection of :class:`~tensorlbm.cavitation_airfoil.
    MixtureCavitationSolver` solves a pure-Neumann problem, so ``p`` is only
    defined up to a constant.  The physical constant is set by the inflow
    condition (the inlet plane is upstream of the body, ``p -> 0`` there).
    Returns the offset ``c0`` such that ``p + c0`` is the physical pressure.

    ``inlet_columns`` columns are averaged (``3`` is the default: one chord
    fraction per column at 300-1200 cells/chord); solid cells are excluded
    when a mask is given.
    """
    cols = p[:, : max(int(inlet_columns), 1)].double()
    if airfoil_mask is not None:
        mask = airfoil_mask[:, : cols.shape[1]].bool()
        if bool((~mask).any()):
            return -float(cols[~mask].mean())
    return -float(cols.mean())


def physical_pressure(p: torch.Tensor, offset: float) -> torch.Tensor:
    """``p + offset`` (see :func:`pressure_anchor`)."""
    return p + offset


# ---------------------------------------------------------------------------
# 2. immersed-body penalisation
# ---------------------------------------------------------------------------


def brinkman_leakage(eta: float, dt: float) -> float:
    """Steady slip fraction left inside a Brinkman-penalised body.

    The solver applies ``u* = (u + dt R) / (1 + dt/eta)`` each step, so a
    constant forcing balances at ``u_inside / u_outside = 1 / (1 + dt/eta)``.
    """
    if eta <= 0.0 or dt <= 0.0:
        msg = "eta and dt must be positive"
        raise ValueError(msg)
    return 1.0 / (1.0 + dt / eta)


def brinkman_eta_for_leakage(dt: float, leakage: float) -> float:
    """``eta`` giving a prescribed steady slip fraction (inverse of the above)."""
    if not 0.0 < leakage < 1.0:
        msg = "leakage must be in (0, 1)"
        raise ValueError(msg)
    return dt * leakage / (1.0 - leakage)


# ---------------------------------------------------------------------------
# 3. phase-change time scales
# ---------------------------------------------------------------------------


def schnerr_sauer_growth_time(
    alpha: float,
    dp: float,
    rho_l: float = 1.0,
    nuclei_density: float = 100.0,
    r_b_floor: float = 0.01,
) -> float:
    """e-folding time of the Schnerr-Sauer void fraction at fixed ``dp``.

    For ``p < p_v`` and a fixed bubble radius the closure is
    ``d alpha / dt = alpha_l (3/R_b) sqrt(2 dp/(3 rho_l)) alpha``, i.e. an
    exponential with rate ``k = (3/R_b) sqrt(2 dp/(3 rho_l))`` (``alpha_l -> 1``
    in the small-void-fraction limit).  Returns ``1/k`` in the solver's
    non-dimensional time unit (``c/U``).
    """
    if dp <= 0.0:
        msg = "dp must be positive (evaporation branch)"
        raise ValueError(msg)
    a = min(max(alpha, 1e-12), 1.0 - 1e-12)
    r_b = (
        (a / (1.0 - a)) * (3.0 / (4.0 * math.pi * nuclei_density))
    ) ** (1.0 / 3.0)
    r_b = max(r_b, r_b_floor)
    rate = (3.0 / r_b) * math.sqrt(2.0 * dp / (3.0 * rho_l))
    return 1.0 / rate


def rayleigh_plesset_time(dp: float, r: float, rho_l: float = 1.0) -> float:
    """Linearised Rayleigh-Plesset time scale ``1/omega``, ``omega^2 = 3 dp/(rho R^2)``."""
    if dp <= 0.0 or r <= 0.0:
        msg = "dp and r must be positive"
        raise ValueError(msg)
    return r * math.sqrt(rho_l / (3.0 * dp))


def integrate_closure_growth(
    dp: float,
    n_steps: int = 4000,
    alpha0: float = 1.0e-5,
    rho_l: float = 1.0,
    nuclei_density: float = 100.0,
    r_b_floor: float = 1.0e-4,
    p_v: float = -0.5,
    dt: float | None = None,
) -> dict:
    """Integrate the closure ODE at fixed pressure and fit the growth exponent.

    The source is the production closure itself
    :func:`tensorlbm.cavitation_airfoil.schnerr_sauer_rate` (no re-derivation
    here), driven by a constant ``p < p_v``.  Returns the measured
    ``d ln(alpha) / d ln(t)`` over the late window and the equivalent bubble
    radius history, so the closure can be compared with the Rayleigh-Plesset
    asymptote ``R = t sqrt(2 dp / (3 rho_l))``.
    """
    if dt is None:
        dt = 0.02 * schnerr_sauer_growth_time(
            alpha0, dp, rho_l, nuclei_density, r_b_floor
        )
    a = torch.tensor([[alpha0]], dtype=torch.float64)
    p = torch.tensor([[p_v - dp]], dtype=torch.float64)
    t = 0.0
    ts: list[float] = []
    als: list[float] = []
    for _ in range(n_steps):
        s = schnerr_sauer_rate(
            a,
            p,
            p_v,
            rho_l=rho_l,
            nuclei_density=nuclei_density,
            r_b_floor=r_b_floor,
            dp_max=None,
        )
        a = (a + dt * s).clamp(0.0, 1.0)
        t += dt
        ts.append(t)
        als.append(float(a))
        if float(a) >= 0.9:
            break
    al = torch.tensor(als)
    tt = torch.tensor(ts)
    lo = tt > 0.25 * ts[-1]
    if int(lo.sum()) < 8:
        lo = torch.ones_like(tt, dtype=torch.bool)
    x = torch.log(tt[lo])
    y = torch.log(al[lo].clamp(min=1e-30))
    xm, ym = x.mean(), y.mean()
    slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    # bubble radius from the void fraction (Schnerr-Sauer inversion)
    r_now = (
        (als[-1] / max(1.0 - als[-1], 1e-12)) * (3.0 / (4.0 * math.pi * nuclei_density))
    ) ** (1.0 / 3.0)
    r_rp = ts[-1] * math.sqrt(2.0 * dp / (3.0 * rho_l))
    return {
        "dp": dp,
        "dt": dt,
        "n_steps": len(ts),
        "t_final": ts[-1],
        "alpha_final": als[-1],
        "growth_exponent": slope,
        "r_bubble_final": r_now,
        "r_rayleigh_plesset": r_rp,
        "ratio_r_rp": r_now / r_rp if r_rp > 0 else float("nan"),
        "time": ts,
        "alpha": als,
    }


# ---------------------------------------------------------------------------
# 4. explicit-scheme budget
# ---------------------------------------------------------------------------


def stability_budget(cfg: CavitationFlowConfig) -> dict:
    """Explicit-scheme safety factors for one :class:`CavitationFlowConfig`.

    Reports the advective CFL, the viscous diffusion number, the Smagorinsky
    diffusion number (at the grid-scale strain ``U/h``) and the Brinkman slip.
    All four must stay below their explicit limits; the P41-N02 runs kept the
    first three inside their limits and still diverged, which is the evidence
    that the blow-up was a boundary/wake instability, not a CFL violation.
    """
    h = cfg.h
    dt = cfg.cfl * h
    nu = cfg.nu + cfg.viscosity_floor
    nu_t = (cfg.smagorinsky * h) ** 2 * (1.0 / h)  # grid-scale strain U/h, U = 1
    return {
        "h": h,
        "dt": dt,
        "advective_cfl": cfg.cfl,
        "viscous_number": nu * dt / h**2,
        "sgs_number": nu_t * dt / h**2,
        "brinkman_slip": brinkman_leakage(1.0e-3, dt),
        "brinkman_dt_over_eta": dt / 1.0e-3,
    }


# ---------------------------------------------------------------------------
# 5. cavity metrics from an (anchored) pressure field
# ---------------------------------------------------------------------------


def cavity_length_isobaric(
    p: torch.Tensor,
    cfg: CavitationFlowConfig,
    geometry=None,
    threshold: float = 0.5,
    anchor_offset: float | None = None,
    sharpness: float = 0.05,
) -> dict:
    """Sheet-cavity length from the isobaric (``p < p_v``) criterion.

    The vapour indicator is ``sigmoid((p_v - p_phys)/(sharpness |p_v|))`` so the
    same :func:`~tensorlbm.cavitation_airfoil.cavity_metrics` sampler can be
    used as for a transported void fraction.  ``anchor_offset`` defaults to the
    inlet-plane anchor; passing ``0.0`` reproduces the *un-anchored* criterion
    of the P41-N02 runs (kept for the audit comparison).
    """
    if anchor_offset is None:
        anchor_offset = pressure_anchor(p)
    p_phys = physical_pressure(p.double(), anchor_offset)
    alpha = torch.sigmoid((cfg.p_v - p_phys) / (sharpness * abs(cfg.p_v)))
    out = cavity_metrics(alpha, cfg, threshold=threshold, geometry=geometry)
    out["anchor_offset"] = anchor_offset
    return out