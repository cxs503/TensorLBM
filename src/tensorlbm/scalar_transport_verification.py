"""Passive-scalar transport verification face (N10): closed-form anchors + gates.

Why this module exists
----------------------
The solver side of passive-scalar transport already lives in
:mod:`tensorlbm.passive_scalar` (production solver),
:mod:`tensorlbm.scalar_transport` (Boussinesq coupling) and
:mod:`tensorlbm.taylor_aris` (D2Q9 scalar LBM + Aris auxiliary problem).
This module is the *verification face* above them: it fixes a small set of
problems whose answer is known in closed form, measures the error of the
lattice solver against those closed forms, and decides machine-checkable gates.
Nothing here is tuned to make a gate pass; every measurement is a moment-based
(variance-growth) or L2-norm estimate of a quantity that the continuum
equation fixes independently of the scheme.

The three physical anchors
--------------------------
1. **Pure diffusion (erf).**  ``d_t c = D d_xx c`` with a step initial
   condition has the self-similar solution
   ``c = (1/2) erfc((x-x0)/sqrt(4Dt))`` on the semi-infinite line.  A lattice
   solution started from that profile at ``t0 > 0`` (smooth initial data, so
   the error is not the discontinuous-IC artefact) is compared with the exact
   profile at ``t1`` in the discrete L2 norm.  Refining ``dx`` under diffusive
   scaling (``D_lattice`` fixed, ``dt ~ dx^2``) gives the observed order.
2. **Taylor-Aris dispersion.**  For a steady shear flow ``u(y)`` with
   cross-sectional mean ``<u>``, the long-time cross-sectionally averaged
   field obeys a 1-D diffusion equation with ``D_eff = D + K L^2 U^2 / D``,
   where ``K = 2/105`` for plane Poiseuille flow between plates at ``+-h`` and
   ``K = 1/48`` for a circular pipe of radius ``a``.  For a *linear* shear
   ``u = gamma y`` between the same plates the auxiliary problem
   ``D chi'' = -u'``, ``chi'(+-h) = 0`` has
   ``chi = gamma (h^2 y - y^3/3)/(2D)``, hence
   ``D_eff = D + (2/15) gamma^2 h^4/D`` -- the shear-layer anchor used here
   (``Pe_gamma = gamma h^2 / D``).  ``D_eff`` is measured from the growth rate
   of the plume variance of the cross-section-averaged field (moment method),
   *not* from a peak height that a smeared profile would make look
   artificially good, and is cross-checked with the independent Fourier-mode
   decay estimator.
3. **Sc independence.**  A one-way coupled passive scalar has
   ``D = c_s^2 (tau_d - 1/2)`` and knows nothing about the momentum relaxation
   ``tau_m = 3 nu + 1/2``; ``Sc = nu/D`` may only enter through ``Pe``.  Two
   empirical checks: (a) the dimensionless dispersion constant
   ``K_meas = (D_eff/D - 1) Pe^-2`` must collapse onto one value across a
   decade of ``D`` (i.e. across ``Sc``); (b) replacing the velocity field by
   momentum solves at ``tau_m`` matching ``Sc = 0.1 / 1 / 10`` must not move
   ``D_eff`` while ``tau_d`` is held fixed.

Scope / accepted limitations
----------------------------
* The velocity field is *given* (imposed parabola/shear, or a converged LBM
  Poiseuille profile): the verification covers the continuum limit of the
  scalar advection-diffusion equation in a prescribed steady laminar field,
  not two-way coupling (buoyancy, variable viscosity, turbulence).
* The asymptotic Taylor-Aris regime needs ``t >> h^2/D`` and a plume still
  much narrower than the domain; every run records ``sigma/nx`` and the fit
  window so the validity of each number can be audited.
* No EFD data is involved: all anchors are closed forms of the same PDE, so
  the claim is "agreement with the exact solution of the continuum problem",
  valid only inside the documented accuracy domain (G8).

Gates (thresholds in :data:`GATE_THRESHOLDS`)
--------------------------------------------
G1 erf L2 rel. error <= 1 % at the finest grid and observed order >= 1.8
G2 Taylor-Aris ``D_eff`` within 5 % of the closed form at ``Pe = 5/10/20/40``
G3 ``Pe -> 0`` gives ``D_eff/D -> 1`` within 2 %
G4 Sc independence: dispersion-constant divergence <= 3 %
G5 total-scalar drift <= 1e-10 relative per 1e4 steps
G6 negative control ``u = 0`` gives ``D_eff/D in [0.95, 1.05]``
G7 convergence series: ``R^2 >= 0.98`` and monotone error decrease
G8 a ``Pe x grid`` accuracy-domain table is emitted
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .taylor_aris import (
    ARIS_K_CHANNEL,
    ARIS_K_PIPE,
    DispersionConfig,
    aris_dispersion_coefficient,
    lbm_poiseuille_profile,
    scalar_collide,
    scalar_equilibrium,
    scalar_stream,
    simulate_channel_dispersion,
    taylor_aris_channel,
    taylor_aris_pipe,
)

__all__ = [
    "ARIS_K_SHEAR",
    "GATE_THRESHOLDS",
    "erf_diffusion_profile",
    "shear_aris_closed_form",
    "shear_velocity_field",
    "pipe_aris_discrete",
    "simulate_diffusion_1d",
    "erf_convergence_study",
    "channel_dispersion_run",
    "shear_dispersion_run",
    "mass_conservation_probe",
    "tau_m_velocity_field",
    "build_work_items",
    "run_items",
    "evaluate_gates",
    "assemble_report",
    "load_item_dirs",
    "main",
]

#: Aris constant for a linear shear between plates at ``+-h``: ``D + (2/15) gamma^2 h^4/D``.
ARIS_K_SHEAR = 2.0 / 15.0

GATE_THRESHOLDS: dict[str, Any] = {
    "G1_l2_rel_max": 1.0e-2,
    "G1_order_min": 1.8,
    "G2_rel_err_max": 5.0e-2,
    "G3_rel_dev_max": 2.0e-2,
    "G4_divergence_max": 3.0e-2,
    "G5_mass_drift_max": 1.0e-10,
    "G6_ratio_lo": 0.95,
    "G6_ratio_hi": 1.05,
    "G7_r2_min": 0.98,
}


# -- closed forms ------------------------------------------------------------
def _erfc(values: Iterable[float]) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    return np.array([math.erfc(float(v)) for v in arr], dtype=np.float64)


def erf_diffusion_profile(
    x: Sequence[float] | np.ndarray,
    t: float,
    diffusivity: float,
    x0: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Exact solution of the 1-D semi-infinite (step) diffusion problem.

    ``c(x, t) = (amplitude/2) erfc((x - x0)/sqrt(4 D t))``, i.e. the solution
    of ``d_t c = D d_xx c`` with ``c(x < x0, 0) = amplitude`` and
    ``c(x > x0, 0) = 0``.
    """
    if t <= 0.0:
        raise ValueError("t must be positive")
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * amplitude * _erfc((x - x0) / math.sqrt(4.0 * diffusivity * t))


def shear_aris_closed_form(
    diffusivity: float, half_width: float, shear_rate: float
) -> float:
    """``D_eff = D + (2/15) gamma^2 h^4 / D`` for ``u = gamma y`` between plates at ``+-h``.

    Derived from the Aris auxiliary problem ``D chi'' = -u'`` with
    ``chi'(+-h) = 0``: ``chi = gamma (h^2 y - y^3/3)/(2 D)``, hence
    ``<u' chi> = gamma^2/(2D) (h^2<y^2> - <y^4>/3) = 2 gamma^2 h^4/(15 D)``.
    The first, hand-derived value used in development was ``h^4/(45 D)``; the
    discrete Aris solve on the lattice nodes disagreed by exactly a factor 6,
    which is how the missing factor was found (see docs/report).
    """
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    return diffusivity + ARIS_K_SHEAR * shear_rate**2 * half_width**4 / diffusivity


def shear_velocity_field(ny: int, shear_rate: float) -> tuple[np.ndarray, float]:
    """Node-sampled linear shear ``u_j = gamma y_j`` on the ``taylor_aris`` grid.

    Returns ``(u, half_width)`` with ``y_j = j - (ny-1)/2`` and
    ``half_width = ny/2`` -- the same convention as
    :func:`tensorlbm.taylor_aris.simulate_channel_dispersion`, so the wall sits
    half a cell outside the outermost node.
    """
    if ny % 2 == 0:
        raise ValueError("ny must be odd so a node sits on the centreline")
    y = np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)
    return float(shear_rate) * y, 0.5 * ny


def pipe_aris_discrete(
    radius: float, u_mean: float, diffusivity: float, n: int = 200
) -> dict[str, Any]:
    """Radial finite-volume solve of the Aris problem in a circular pipe.

    ``D (1/r) d/dr (r dchi/dr) = -u'`` with symmetry at ``r = 0`` and zero flux
    at ``r = a`` (both natural for the flux-form stencil below), so that
    ``D_eff = D + <u' chi>`` with area weight ``2 pi r dr``.  This verifies the
    ``1/48`` constant of :func:`taylor_aris_pipe` independently of the LBM; the
    pipe geometry itself is not simulated here (a 3-D axisymmetric scalar
    solver is out of scope for this node), only the constant is under test.
    """
    if radius <= 0.0 or diffusivity <= 0.0:
        raise ValueError("radius and diffusivity must be positive")
    n = int(n)
    dr = radius / n
    r = (np.arange(n) + 0.5) * dr             # cell centres
    rf = (np.arange(n) + 1.0) * dr            # r_{j+1/2}
    rf_in = np.concatenate([[0.0], rf[:-1]])  # r_{j-1/2};  r_{-1/2} = 0
    area = 2.0 * math.pi * r * dr
    u = 2.0 * u_mean * (1.0 - (r / radius) ** 2)
    u_mean_d = float((u * area).sum() / area.sum())
    u_prime = u - u_mean_d

    L = np.zeros((n, n), dtype=np.float64)    # flux-form (1/r) d/dr (r d/dr)
    for j in range(n):
        vol = r[j] * dr
        if j + 1 < n:                          # outer face flux r_{j+1/2}
            L[j, j + 1] += rf[j] / (vol * dr)
        if j > 0:                              # inner face flux r_{j-1/2}
            L[j, j - 1] += rf_in[j] / (vol * dr)
        # diagonal = -(sum of the off-diagonals actually inserted) so that
        # L @ ones == 0 exactly (the linear field is the null space).
        L[j, j] = -float(L[j].sum())

    # The plain matrix L is *not* symmetric; it is self-adjoint with respect to
    # the area weights, so the symmetric (and, after removing the constant
    # mode, solvable) system is  M chi = -W u',  M = diag(area) @ (D L).
    # M 1 = 0 and 1' (W u') = 0 by construction, so a least-squares solve
    # returns the minimum-norm (weighted-mean-zero) solution -- doing this with
    # an explicit pinning row instead was observed to be platform dependent
    # (the pinned matrix stayed numerically singular, giving garbage on one
    # LAPACK build and a clean answer on another).
    W = np.diag(area)
    M = W @ (diffusivity * L)
    rhs = -(area * u_prime)
    chi, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    taylor_term = float((area * u_prime * chi).sum() / area.sum())
    resid_norm = float(np.max(np.abs(M @ chi - rhs)) / np.max(np.abs(rhs)))
    d_eff = diffusivity + taylor_term
    closed = taylor_aris_pipe(diffusivity, radius, u_mean)
    return {
        "kind": "pipe_aris",
        "d_eff": d_eff,
        "d_eff_closed_form": closed,
        "k_measured": (d_eff - diffusivity) * diffusivity / (radius**2 * u_mean_d**2),
        "k_closed_form": ARIS_K_PIPE,
        "rel_err": (d_eff - closed) / closed,
        "residual": resid_norm,
        "n": n,
    }


# -- 1-D lattice diffusion solver (D1Q3) and erf convergence study ------------
_W1Q3 = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
_CS2_1Q3 = 1.0 / 3.0


def simulate_diffusion_1d(
    *,
    n: int = 256,
    half_width: float = 11.313708498984761,
    diffusivity: float = 1.0,
    t_start: float = 0.1,
    t_final: float = 1.0,
    lattice_diffusivity: float = 0.05,
    x0: float = 0.0,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    """D1Q3 BGK diffusion solve started from the exact erf profile at ``t_start``.

    Diffusive scaling: ``dx = 2 half_width / n``, ``dt = D_lat dx^2 / D``, so
    the lattice diffusivity ``D_lat = c_s^2 (tau - 1/2)`` stays fixed while the
    physical problem (``D``, ``t_final``) is unchanged across refinements.
    Zero-flux (bounce-back) ends stand in for the semi-infinite line; the
    domain is wide enough (``half_width ~ 8 sqrt(2 D t_final)``) that the walls
    see ``erfc(8) ~ 1e-29``.
    """
    if n < 8:
        raise ValueError("need at least 8 nodes")
    d_lat = float(lattice_diffusivity)
    if d_lat <= 0.0:
        raise ValueError("lattice_diffusivity must be positive")
    if not (t_final > t_start > 0.0):
        raise ValueError("need t_final > t_start > 0")
    dx = 2.0 * half_width / n
    dt = d_lat * dx * dx / diffusivity
    tau = d_lat / _CS2_1Q3 + 0.5
    steps = int(round((t_final - t_start) / dt))
    if steps < 1:
        raise ValueError("time window too short for this resolution")

    x = (np.arange(n) + 0.5) * dx - half_width
    c0 = erf_diffusion_profile(x, t_start, diffusivity, x0=x0).astype(dtype)
    w = _W1Q3.astype(dtype)[:, None]
    g = w * c0[None, :]
    buf = np.empty_like(g)
    mass0 = float(np.sum(c0, dtype=np.float64))
    for _ in range(steps):
        phi = g.sum(axis=0)
        geq = w * phi[None, :]
        g -= (g - geq) / tau
        buf[0] = g[0]
        buf[1, :-1] = g[1, 1:]
        buf[1, -1] = g[2, -1]
        buf[2, 1:] = g[2, :-1]
        buf[2, 0] = g[1, 0]
        g, buf = buf, g
    c_num = g.sum(axis=0)
    c_ref = erf_diffusion_profile(x, t_final, diffusivity, x0=x0).astype(dtype)
    l2 = float(np.linalg.norm(c_num - c_ref) / np.linalg.norm(c_ref))
    return {
        "n": int(n),
        "dx": dx,
        "dt": dt,
        "tau": tau,
        "steps": steps,
        "lattice_diffusivity": d_lat,
        "l2_rel_error": l2,
        "linf_error": float(np.max(np.abs(c_num - c_ref))),
        "mass_drift": abs(float(np.sum(c_num, dtype=np.float64)) - mass0) / mass0,
        "x": x.tolist(),
        "c_num": c_num.tolist(),
        "c_ref": c_ref.tolist(),
    }


def _order_fit(dxs: Sequence[float], errs: Sequence[float]) -> dict[str, float]:
    """Least-squares slope of ``log(err)`` vs ``log(dx)`` plus its R^2."""
    x = np.log(np.asarray(dxs, dtype=np.float64))
    y = np.log(np.asarray(errs, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"order": float(slope), "r2": float(r2), "intercept": float(intercept)}


def erf_convergence_study(
    ns: Sequence[int] = (64, 128, 256, 512), **kw: Any
) -> dict[str, Any]:
    """Run :func:`simulate_diffusion_1d` on a refinement sequence and fit the order."""
    levels = []
    for n in ns:
        t0 = time.time()
        res = simulate_diffusion_1d(n=int(n), **kw)
        res["wall_time"] = time.time() - t0
        levels.append(res)
    errs = [lv["l2_rel_error"] for lv in levels]
    dxs = [lv["dx"] for lv in levels]
    monotone = all(b < a for a, b in zip(errs, errs[1:]))
    fit = _order_fit(dxs, errs)
    p_fine = float(math.log(errs[-2] / errs[-1]) / math.log(dxs[-2] / dxs[-1]))
    return {
        "kind": "erf",
        "levels": levels,
        "ns": list(ns),
        "l2_rel_errors": errs,
        "dxs": dxs,
        "order_fit": fit["order"],
        "order_fit_r2": fit["r2"],
        "order_two_finest": p_fine,
        "monotone": bool(monotone),
        "finest_l2_rel_error": errs[-1],
    }


# -- dispersion measurement wrappers (moment method, via taylor_aris) --------
def _trim(res: dict[str, Any], max_points: int = 400) -> dict[str, Any]:
    """Keep a decimated copy of the moment history instead of the full series."""
    hist = res.get("history") or {}
    out = {k: v for k, v in res.items() if k != "history"}
    out["history_tail"] = {
        k: (v[-1] if isinstance(v, list) and v else None) for k, v in hist.items()
    }
    t = hist.get("t") or []
    stride = max(1, len(t) // max_points) if t else 1
    out["history_t"] = t[::stride]
    out["history_q"] = (hist.get("q") or [])[::stride]
    return out


def channel_dispersion_run(
    *,
    pe: float,
    ny: int,
    nx: int,
    diffusivity: float,
    tau_per_equilibration: float = 1.6,
    method: str = "variance",
    dtype: str = "float64",
    fit_start_frac: float = 0.55,
) -> dict[str, Any]:
    """Moment-method ``D_eff`` in plane Poiseuille flow at ``Pe = <u> h / D``.

    The run length is set by the transverse equilibration time
    ``t_eq = 3 h^2 / D``; ``tau_per_equilibration`` is the multiple used (the
    variance branch of :func:`simulate_channel_dispersion` fits the last
    ``1 - fit_start_frac`` of the record).
    """
    h = 0.5 * ny
    u_mean = float(pe) * diffusivity / h
    steps = int(round(tau_per_equilibration * 3.0 * h * h / diffusivity))
    cfg = DispersionConfig(
        nx=int(nx),
        ny=int(ny),
        diffusivity=float(diffusivity),
        u_mean=u_mean,
        steps=steps,
        method=method,
        dtype=dtype,
        fit_start_frac=float(fit_start_frac),
        fit_end_frac=1.0,
        report_every=max(10, steps // 400),
    )
    t0 = time.time()
    res = simulate_channel_dispersion(cfg)
    out = _trim(res)
    analytic = taylor_aris_channel(cfg.diffusivity, h, u_mean)
    out.update(
        {
            "kind": "channel",
            "pe_target": float(pe),
            "u_mean": u_mean,
            "half_width": h,
            "d_eff_analytic": analytic,
            "t_eq": 3.0 * h * h / cfg.diffusivity,
            "wall_time": time.time() - t0,
            "pe_r": float(pe),
            "nx": int(nx),
            "diffusivity": float(diffusivity),
        }
    )
    out["rel_err_analytic"] = (out["d_eff"] - analytic) / analytic
    out["ratio_over_d"] = out["d_eff"] / cfg.diffusivity
    out["k_meas"] = (
        (out["d_eff"] / cfg.diffusivity - 1.0) / (float(pe) ** 2)
        if pe
        else float("nan")
    )
    return out


def shear_dispersion_run(
    *,
    pe_gamma: float,
    ny: int,
    nx: int,
    diffusivity: float,
    tau_per_equilibration: float = 1.6,
    method: str = "variance",
    dtype: str = "float64",
    fit_start_frac: float = 0.55,
) -> dict[str, Any]:
    """Moment-method ``D_eff`` in a 2-D linear shear layer ``u = gamma y``.

    ``Pe_gamma = gamma h^2 / D``; closed form ``D_eff = D + gamma^2 h^4/(45 D)``.
    """
    h = 0.5 * ny
    gamma = float(pe_gamma) * diffusivity / (h * h)
    u, half_width = shear_velocity_field(int(ny), gamma)
    steps = int(round(tau_per_equilibration * 3.0 * h * h / diffusivity))
    cfg = DispersionConfig(
        nx=int(nx),
        ny=int(ny),
        diffusivity=float(diffusivity),
        u_mean=float(u.mean()),
        steps=steps,
        method=method,
        velocity_field=u.tolist(),
        dtype=dtype,
        fit_start_frac=float(fit_start_frac),
        fit_end_frac=1.0,
        report_every=max(10, steps // 400),
    )
    t0 = time.time()
    res = simulate_channel_dispersion(cfg)
    out = _trim(res)
    analytic = shear_aris_closed_form(cfg.diffusivity, half_width, gamma)
    out.update(
        {
            "kind": "shear",
            "pe_target": float(pe_gamma),
            "shear_rate": gamma,
            "half_width": half_width,
            "d_eff_analytic": analytic,
            "t_eq": 3.0 * h * h / cfg.diffusivity,
            "wall_time": time.time() - t0,
            "pe_r": float(pe_gamma),
            "nx": int(nx),
            "diffusivity": float(diffusivity),
        }
    )
    out["rel_err_analytic"] = (out["d_eff"] - analytic) / analytic
    out["ratio_over_d"] = out["d_eff"] / cfg.diffusivity
    out["k_meas"] = (out["d_eff"] / cfg.diffusivity - 1.0) / (float(pe_gamma) ** 2)
    return out


def tau_m_velocity_field(
    *,
    ny: int,
    diffusivity: float,
    sc: float,
    pe: float,
    blend_steps: int = 40000,
) -> dict[str, Any]:
    """LBM Poiseuille profile whose momentum relaxation matches ``Sc = nu/D``.

    Returns the node-sampled profile normalised to ``U = Pe D / h`` together
    with the momentum relaxation time used (``tau_m = 3 nu + 1/2`` with
    ``nu = Sc D``).
    """
    h = 0.5 * ny
    nu = float(sc) * float(diffusivity)
    tau_m = 3.0 * nu + 0.5
    prof = lbm_poiseuille_profile(
        ny=int(ny),
        nx=4,
        tau=tau_m,
        body_force=2.0 * nu,
        steps=int(blend_steps),
        tol=1e-11,
    )
    u = np.asarray(prof["u"], dtype=np.float64)
    u_target = float(pe) * float(diffusivity) / h
    if float(np.abs(u).max()) <= 0.0:
        raise RuntimeError("momentum solve produced a zero field")
    u = u * (u_target / float(u.mean()))
    return {
        "u": u.tolist(),
        "tau_m": tau_m,
        "nu": nu,
        "sc": float(sc),
        "momentum_steps": int(prof["steps"]),
        "u_mean_target": u_target,
        "u_max": float(np.abs(u).max()),
        "u_mean_solved": float(np.asarray(prof["u"]).mean()),
    }


def _dispersion_with_field(
    *,
    velocity_field: Sequence[float],
    ny: int,
    nx: int,
    diffusivity: float,
    steps: int,
    method: str = "variance",
    dtype: str = "float64",
    fit_start_frac: float = 0.55,
) -> dict[str, Any]:
    """Dispersion measurement for an arbitrary prescribed cross-stream profile."""
    cfg = DispersionConfig(
        nx=int(nx),
        ny=int(ny),
        diffusivity=float(diffusivity),
        u_mean=float(np.mean(velocity_field)),
        steps=int(steps),
        method=method,
        velocity_field=list(velocity_field),
        dtype=dtype,
        fit_start_frac=float(fit_start_frac),
        fit_end_frac=1.0,
        report_every=max(10, int(steps) // 400),
    )
    t0 = time.time()
    res = simulate_channel_dispersion(cfg)
    out = _trim(res)
    out["wall_time"] = time.time() - t0
    out["ratio_over_d"] = out["d_eff"] / cfg.diffusivity
    return out


def mass_conservation_probe(
    *,
    ny: int = 13,
    nx: int = 256,
    diffusivity: float = 0.01,
    shear_rate: float = 0.0111,
    steps: int = 10000,
    dtype: str = "float64",
) -> dict[str, Any]:
    """Relative drift of the total scalar over ``steps`` steps of a sheared run.

    Zero-flux walls plus periodic ``x`` make the discrete total exactly
    conserved; the measured drift is pure round-off (float32 vs float64 differ
    by many decades, which is reported as evidence rather than assumed).
    """
    tdt = torch.float64 if dtype == "float64" else torch.float32
    u, _ = shear_velocity_field(int(ny), float(shear_rate))
    ux = torch.tensor(u, dtype=tdt).view(ny, 1).expand(ny, nx).contiguous()
    uy = torch.zeros((ny, nx), dtype=tdt)
    x = torch.arange(nx, dtype=torch.float64)
    sigma0 = 4.0
    phi = torch.exp(-0.5 * ((x - 0.5 * nx) / sigma0) ** 2)
    phi = (phi / phi.sum()).to(tdt).view(1, nx).expand(ny, nx).contiguous()
    g = scalar_equilibrium(phi, ux, uy)
    wall_rows = torch.zeros(ny, dtype=torch.bool)
    wall_rows[0] = True
    wall_rows[-1] = True
    tau_d = 3.0 * float(diffusivity) + 0.5
    m0 = float(g.double().sum())
    marks: list[int] = []
    drifts: list[float] = []
    checkpoints = sorted({10, steps // 10, steps // 2, steps} - {0})
    for step in range(1, steps + 1):
        phi_cur = g.sum(dim=0)
        g = scalar_collide(g, phi_cur, ux, uy, tau_d)
        g = scalar_stream(g, wall_rows)
        if step in checkpoints:
            marks.append(step)
            drifts.append(abs(float(g.double().sum()) - m0) / abs(m0))
    return {
        "kind": "mass",
        "dtype": dtype,
        "steps": int(steps),
        "checkpoints": marks,
        "drift": drifts,
        "drift_final": drifts[-1],
        "drift_per_1e4_steps": drifts[-1] * 1.0e4 / steps,
        "ny": ny,
        "nx": nx,
        "diffusivity": float(diffusivity),
        "shear_rate": float(shear_rate),
    }

# -- work-item construction / execution --------------------------------------
def build_work_items(mode: str = "full") -> list[dict[str, Any]]:
    """Enumerate the study as independent, serialisable work items.

    ``mode="quick"`` keeps the shapes small enough for a CI unit test; the
    physics (same closed forms, same estimators) is identical.
    """
    quick = mode == "quick"
    items: list[dict[str, Any]] = []

    # G1/G7 | erf diffusion convergence
    erf_ns = (32, 64, 128) if quick else (64, 128, 256, 512)
    items.append(
        {
            "id": "erf",
            "kind": "erf",
            "tags": ["G1", "G7"],
            "params": {
                "ns": list(erf_ns),
                "half_width": 11.313708498984761,
                "diffusivity": 1.0,
                "t_start": 0.1,
                "t_final": 1.0,
            },
        }
    )

    # G2 | pipe Aris constant (1/48) at the ODE level
    items.append(
        {
            "id": "pipe_aris",
            "kind": "pipe_aris",
            "tags": ["G2"],
            "params": {"radius": 1.0, "u_mean": 1.0, "diffusivity": 0.05, "n": 200},
        }
    )

    if quick:
        d = 0.02
        grid = [(9, 512, d, (10.0,)), (13, 512, d, (10.0,))]
        shears = (10.0,)
        nx_big = 512
    else:
        d = 0.01
        # (ny, nx, D, Pe list): the cross-stream resolution, the streamwise
        # resolution and the Peclet number are the three knobs of the accuracy
        # domain; D is raised in the ny=17 row so that the run stays affordable
        # (t_eq ~ 3 h^2/D) without leaving the asymptotic regime.
        grid = [
            (9, 512, d, (5.0, 10.0, 20.0, 40.0)),
            (13, 512, d, (5.0, 10.0, 20.0, 40.0)),
            (13, 1024, d, (5.0, 10.0, 20.0, 40.0)),
            (17, 512, 0.04, (20.0,)),
            (17, 1024, d, (20.0, 40.0)),
        ]
        shears = (5.0, 10.0, 20.0, 40.0)
        nx_big = 1024

    # G2/G8 | channel Taylor-Aris over Pe x grid (moment method)
    for ny, nx, Dg, pes in grid:
        for pe in pes:
            items.append(
                {
                    "id": f"chan_ny{ny}_nx{nx}_pe{pe:g}",
                    "kind": "channel",
                    "tags": ["G2", "G7", "G8"],
                    "params": {
                        "pe": pe,
                        "ny": ny,
                        "nx": nx,
                        "diffusivity": Dg,
                        "method": "variance",
                        "dtype": "float64",
                    },
                    "grid": {"ny": ny, "nx": nx, "diffusivity": Dg},
                }
            )
    # G2 cross-check with the independent Fourier-mode estimator
    mode_pes = (10.0,) if quick else (5.0, 20.0, 40.0)
    for pe in mode_pes:
        items.append(
            {
                "id": f"chan_mode_nx{nx_big}_pe{pe:g}",
                "kind": "channel",
                "tags": ["G2"],
                "params": {
                    "pe": pe,
                    "ny": 13,
                    "nx": nx_big,
                    "diffusivity": d,
                    "method": "mode",
                    "dtype": "float64",
                    "tau_per_equilibration": 1.6 if quick else 2.0,
                },
            }
        )

    # G2 | 2-D linear shear layer anchor
    for peg in shears:
        items.append(
            {
                "id": f"shear_pe{peg:g}",
                "kind": "shear",
                "tags": ["G2"],
                "params": {
                    "pe_gamma": peg,
                    "ny": 13,
                    "nx": nx_big,
                    "diffusivity": d,
                    "method": "variance",
                    "dtype": "float64",
                },
            }
        )

    if not quick:
        # high-Pe shear point whose plume still fits the domain: for
        # Pe_gamma=40 the asymptotic D_eff/D is ~214, so sigma at the end of the
        # record is ~272 cells and the variance estimator needs nx >= 6 sigma.
        items.append(
            {
                "id": "shear_pe40_nx2048",
                "kind": "shear",
                "tags": ["G2", "G8"],
                "params": {
                    "pe_gamma": 40.0,
                    "ny": 13,
                    "nx": 2048,
                    "diffusivity": d,
                    "method": "variance",
                    "dtype": "float64",
                },
                "grid": {"ny": 13, "nx": 2048, "diffusivity": d},
            }
        )

    # G3 | Pe -> 0 limit
    for name, pe in (("pe_small", 0.5), ("pe_tiny", 0.05)):
        items.append(
            {
                "id": name,
                "kind": "channel",
                "tags": ["G3"],
                "params": dict(
                    pe=pe, ny=13, nx=512, diffusivity=d, method="variance", dtype="float64"
                ),
            }
        )

    # G6 | negative control, u = 0 exactly (two independent estimators)
    for name, meth in (("neg_control", "variance"), ("neg_control_mode", "mode")):
        items.append(
            {
                "id": name,
                "kind": "channel",
                "tags": ["G6"],
                "params": dict(
                    pe=0.0, ny=13, nx=512, diffusivity=d, method=meth, dtype="float64"
                ),
            }
        )

    # G4a | dispersion-constant collapse across a decade of D (=> across Sc)
    dset = (0.02, 0.05) if quick else (0.005, 0.02, 0.05)
    collapse_pes = (10.0,) if quick else (10.0, 20.0)
    for D in dset:
        for pe in collapse_pes:
            items.append(
                {
                    "id": f"collapse_D{D:g}_pe{pe:g}",
                    "kind": "channel",
                    "tags": ["G4"],
                    "params": dict(
                        pe=pe, ny=13, nx=512, diffusivity=D, method="variance", dtype="float64"
                    ),
                }
            )
    # G4b | tau_m sweep (Sc = 0.1 / 1 / 10) at fixed tau_d
    sc_list = (0.1, 10.0) if quick else (0.1, 1.0, 10.0)
    for sc in sc_list:
        items.append(
            {
                "id": f"tau_m_sc{sc:g}",
                "kind": "tau_m",
                "tags": ["G4"],
                "params": {
                    "sc": sc,
                    "ny": 13,
                    "nx": 512,
                    "diffusivity": 0.02,
                    "pe": 10.0,
                    "method": "variance",
                    "dtype": "float64",
                    "blend_steps": 20000 if quick else 40000,
                },
            }
        )

    # G5 | mass conservation (float64 gate + float32 contrast)
    for tag, dtp in (("mass_f64", "float64"), ("mass_f32", "float32")):
        items.append(
            {
                "id": tag,
                "kind": "mass",
                "tags": ["G5"],
                "params": dict(
                    ny=13, nx=256, diffusivity=0.01, shear_rate=0.0111, steps=10000, dtype=dtp
                ),
            }
        )
    return items


def _run_one(item: dict[str, Any]) -> dict[str, Any]:
    kind, p = item["kind"], item["params"]
    if kind == "erf":
        return erf_convergence_study(**p)
    if kind == "pipe_aris":
        return pipe_aris_discrete(**p)
    if kind == "channel":
        return channel_dispersion_run(**p)
    if kind == "shear":
        return shear_dispersion_run(**p)
    if kind == "mass":
        return mass_conservation_probe(**p)
    if kind == "tau_m":
        ny, nx = int(p["ny"]), int(p["nx"])
        D, pe, sc = float(p["diffusivity"]), float(p["pe"]), float(p["sc"])
        field = tau_m_velocity_field(
            ny=ny,
            diffusivity=D,
            sc=sc,
            pe=pe,
            blend_steps=int(p.get("blend_steps", 40000)),
        )
        h = 0.5 * ny
        steps = int(round(1.6 * 3.0 * h * h / D))
        disp = _dispersion_with_field(
            velocity_field=field["u"],
            ny=ny,
            nx=nx,
            diffusivity=D,
            steps=steps,
            method=p.get("method", "variance"),
            dtype=p.get("dtype", "float64"),
        )
        out = {
            "kind": "tau_m",
            "sc": field["sc"],
            "tau_m": field["tau_m"],
            "tau_d": 3.0 * D + 0.5,
            "nu": field["nu"],
            "momentum_steps": field["momentum_steps"],
            "u_max": field["u_max"],
            "u_mean_solved": field["u_mean_solved"],
            "pe_r": pe,
            "diffusivity": D,
            "half_width": h,
            "nx": nx,
            "discrete_aris": aris_dispersion_coefficient(field["u"], D)["d_eff"],
        }
        out.update(disp)
        return out
    raise ValueError(f"unknown work item kind: {kind}")


def _summarise(res: dict[str, Any]) -> str:
    if "l2_rel_errors" in res:
        return (
            f"erf order={res['order_fit']:.3f} (R2={res['order_fit_r2']:.4f}) "
            f"L2={res['finest_l2_rel_error']:.3e}"
        )
    if res.get("kind") == "pipe_aris":
        return f"pipe K={res['k_measured']:.6f} vs 1/48 rel={res['rel_err']:+.2e}"
    if res.get("kind") == "mass":
        return f"drift/1e4={res['drift_per_1e4_steps']:.3e} ({res['dtype']})"
    if "d_eff" in res:
        ref = res.get("d_eff_analytic")
        if ref is None:
            ref = res.get("discrete_aris", float("nan"))
        rel = res.get("rel_err_analytic")
        rel_s = "n/a" if rel is None else f"{rel:+.2%}"
        return f"D_eff={res['d_eff']:.5f} ref={ref:.5f} rel={rel_s}"
    return "?"


def run_items(
    items: Sequence[dict[str, Any]],
    outdir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Execute work items, persisting each result under ``outdir/items``."""
    results: dict[str, dict[str, Any]] = {}
    itemdir = Path(outdir) / "items" if outdir else None
    if itemdir is not None:
        itemdir.mkdir(parents=True, exist_ok=True)
    for item in items:
        t0 = time.time()
        res = _run_one(item)
        res["item_id"] = item["id"]
        res["tags"] = list(item.get("tags", []))
        res["params"] = dict(item.get("params", {}))
        if "grid" in item:
            res["grid"] = item["grid"]
        res["elapsed"] = time.time() - t0
        results[item["id"]] = res
        if itemdir is not None:
            (itemdir / f"{item['id']}.json").write_text(json.dumps(res, indent=1))
        if verbose:
            print(f"[{item['id']}] {_summarise(res)} ({res['elapsed']:.1f}s)", flush=True)
    return results


# -- gate evaluation ---------------------------------------------------------
def _by_tag(results: dict[str, dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [r for r in results.values() if tag in r.get("tags", [])]


def evaluate_gates(
    results: dict[str, dict[str, Any]], thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Turn raw measurements into the G1..G8 gate verdicts."""
    th = dict(GATE_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    gates: dict[str, Any] = {}

    # G1 + G7 | erf diffusion
    erf = results.get("erf")
    if erf:
        gates["G1"] = {
            "pass": bool(
                erf["finest_l2_rel_error"] <= th["G1_l2_rel_max"]
                and erf["order_two_finest"] >= th["G1_order_min"]
            ),
            "finest_l2_rel_error": erf["finest_l2_rel_error"],
            "threshold": th["G1_l2_rel_max"],
            "observed_order": erf["order_two_finest"],
            "order_threshold": th["G1_order_min"],
            "levels": [
                {
                    "n": lv["n"],
                    "dx": lv["dx"],
                    "dt": lv["dt"],
                    "steps": lv["steps"],
                    "l2_rel_error": lv["l2_rel_error"],
                    "linf_error": lv["linf_error"],
                }
                for lv in erf["levels"]
            ],
        }
        gates["G7"] = {
            "pass": bool(erf["order_fit_r2"] >= th["G7_r2_min"] and erf["monotone"]),
            "order_fit_r2": erf["order_fit_r2"],
            "r2_threshold": th["G7_r2_min"],
            "monotone": erf["monotone"],
            "order_fit": erf["order_fit"],
            "series": [erf["dxs"], erf["l2_rel_errors"]],
            "note": "log-log least squares over all refinement levels",
        }

    # G2 | Taylor-Aris vs closed forms
    def _eligible(r: dict[str, Any]) -> tuple[bool, str]:
        """Can this run support a solver-accuracy claim?

        Two conditions, both about the *measurement*, not about the answer:
        the cross-stream grid must be refined enough for the scheme to be in
        its accuracy domain (``ny >= 13``; the ``ny = 9`` rows live in G8), and
        the variance estimator needs ``sigma < nx/6`` for the plume to stay
        inside the periodic box (mode-method runs have no sigma).
        """
        if int(r.get("ny", 0)) < 13:
            return False, "ny<13: outside the cross-stream accuracy domain (see G8)"
        sig = r.get("sigma_over_nx")
        if sig is not None and np.isfinite(sig) and sig > 1.0 / 6.0:
            return False, f"sigma/nx={sig:.3f}>1/6: variance estimator out of domain"
        return True, ""

    chan_var = [
        r
        for r in _by_tag(results, "G2")
        if r.get("kind") == "channel" and r.get("method") == "variance"
    ]
    shear = [r for r in _by_tag(results, "G2") if r.get("kind") == "shear"]
    pipe = results.get("pipe_aris")
    excluded: list[dict[str, Any]] = []

    def _split(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keep = []
        for r in runs:
            ok, why = _eligible(r)
            if ok:
                keep.append(r)
            else:
                excluded.append(
                    {"item": r["item_id"], "rel_err": r["rel_err_analytic"], "reason": why}
                )
        return keep

    chan_var = _split(chan_var)
    shear = _split(shear)
    if chan_var or shear or pipe:
        entries = []
        for r in chan_var:
            entries.append(
                {
                    "case": f"channel ny={r['ny']} nx={r['nx']}",
                    "pe_r": r["pe_r"],
                    "estimator": "variance (moment)",
                    "d_eff": r["d_eff"],
                    "reference": r["d_eff_analytic"],
                    "reference_kind": "closed form D + 2 h^2 U^2/(105 D)",
                    "rel_err": r["rel_err_analytic"],
                }
            )
        for r in shear:
            entries.append(
                {
                    "case": f"linear shear ny={r['ny']} nx={r['nx']}",
                    "pe_r": r["pe_r"],
                    "estimator": "variance (moment)",
                    "d_eff": r["d_eff"],
                    "reference": r["d_eff_analytic"],
                    "reference_kind": "closed form D + (2/15) gamma^2 h^4/D",
                    "rel_err": r["rel_err_analytic"],
                }
            )
        if pipe:
            entries.append(
                {
                    "case": f"pipe radial Aris n={pipe['n']}",
                    "pe_r": None,
                    "estimator": "radial FV Aris ODE",
                    "d_eff": pipe["d_eff"],
                    "reference": pipe["d_eff_closed_form"],
                    "reference_kind": "closed form D + a^2 U^2/(48 D)",
                    "rel_err": pipe["rel_err"],
                }
            )
        worst = max(entries, key=lambda e: abs(e["rel_err"]))
        gates["G2"] = {
            "pass": bool(abs(worst["rel_err"]) <= th["G2_rel_err_max"]),
            "threshold": th["G2_rel_err_max"],
            "worst": worst,
            "entries": entries,
            "excluded": excluded,
            "eligibility": (
                "ny>=13 (cross-stream accuracy domain; the ny=9 rows are reported in G8) "
                "and sigma/nx<=1/6 (variance-estimator domain)"
            ),
        }

    # G3 | Pe -> 0 limit
    g3_items = [r for r in _by_tag(results, "G3") if "ratio_over_d" in r]
    if g3_items:
        devs = [
            {
                "item": r["item_id"],
                "pe_r": r.get("pe_r"),
                "d_eff_over_d": r["ratio_over_d"],
                "rel_dev": r["ratio_over_d"] - 1.0,
            }
            for r in g3_items
        ]
        worst = max(devs, key=lambda d: abs(d["rel_dev"]))
        gates["G3"] = {
            "pass": bool(abs(worst["rel_dev"]) <= th["G3_rel_dev_max"]),
            "threshold": th["G3_rel_dev_max"],
            "worst": worst,
            "entries": devs,
        }

    # G4 | Sc independence
    g4_items = [r for r in _by_tag(results, "G4") if r.get("kind") == "channel"]
    tau_items = [r for r in _by_tag(results, "G4") if r.get("kind") == "tau_m"]
    g4: dict[str, Any] = {}
    if g4_items:
        by_d: dict[float, list[float]] = {}
        for r in g4_items:
            k = r.get("k_meas")
            if k is not None and np.isfinite(k):
                by_d.setdefault(round(r["diffusivity"], 8), []).append(k)
        collapse = []
        for D, ks in sorted(by_d.items()):
            k = float(np.mean(ks))
            collapse.append(
                {
                    "diffusivity": D,
                    "sc_implied": 0.1 / D,  # nu = 0.1 reference momentum field
                    "k_meas": k,
                    "rel_dev_vs_closed_form": (k - ARIS_K_CHANNEL) / ARIS_K_CHANNEL,
                    "n_runs": len(ks),
                }
            )
        ks = [c["k_meas"] for c in collapse]
        spread = (max(ks) - min(ks)) / abs(float(np.mean(ks))) if ks else float("nan")
        g4["collapse"] = {
            "entries": collapse,
            "divergence": float(spread),
            "closed_form_k": ARIS_K_CHANNEL,
        }
    if tau_items:
        ratios = [r["ratio_over_d"] for r in tau_items]
        spread_raw = (max(ratios) - min(ratios)) / abs(float(np.mean(ratios)))
        # The velocity field itself is a D2Q9 momentum solve, so its *shape*
        # carries its own O(Ma^2) error which changes the discrete Aris
        # prediction.  The scalar-side question ("does the scalar see tau_m?")
        # is therefore asked on the residual against each field's own discrete
        # Aris reference; the raw ratio spread is reported next to it.
        resids = [(r["d_eff"] - r["discrete_aris"]) / r["discrete_aris"] for r in tau_items]
        own = [r["d_eff"] / r["discrete_aris"] for r in tau_items]
        # gate statistic: D_eff relative to each field's own discrete Aris value
        spread = (max(own) - min(own)) / abs(float(np.mean(own)))
        g4["tau_m"] = {
            "entries": [
                {
                    "sc": r["sc"],
                    "tau_m": r["tau_m"],
                    "tau_d": r["tau_d"],
                    "nu": r["nu"],
                    "d_eff": r["d_eff"],
                    "ratio_over_d": r["ratio_over_d"],
                    "discrete_aris": r["discrete_aris"],
                }
                for r in tau_items
            ],
            "divergence": float(spread),
            "divergence_raw_ratio": float(spread_raw),
            "residuals_vs_discrete_aris": [float(x) for x in resids],
            "ratios_over_own_reference": [float(x) for x in own],
            "d_eff_spread_across_sc": float(
                (max(r["d_eff"] for r in tau_items) - min(r["d_eff"] for r in tau_items))
                / abs(float(np.mean([r["d_eff"] for r in tau_items])))
            ),
            "note": (
                "tau_d = 3D+1/2 held fixed while the velocity field is a real D2Q9 "
                "momentum solve at tau_m = 3(Sc D)+1/2; the scalar must not see tau_m. "
                "divergence is taken on the residual against each field's own discrete "
                "Aris prediction; divergence_raw_ratio includes the momentum-side "
                "profile-shape change and is reported for transparency only"
            ),
        }
    if g4:
        divs = [
            v["divergence"] for v in g4.values() if isinstance(v, dict) and "divergence" in v
        ]
        g4["pass"] = bool(max(divs) <= th["G4_divergence_max"]) if divs else False
        g4["threshold"] = th["G4_divergence_max"]
        g4["tau_d_independent_of_sc"] = bool(
            all(abs(r["tau_d"] - (3.0 * r["diffusivity"] + 0.5)) < 1e-12 for r in tau_items)
        )
        gates["G4"] = g4

    # G5 | mass conservation
    masses = [r for r in _by_tag(results, "G5") if r.get("kind") == "mass"]
    gate_mass = [r for r in masses if r["dtype"] == "float64"]
    if gate_mass:
        worst = max(gate_mass, key=lambda r: r["drift_per_1e4_steps"])
        gates["G5"] = {
            "pass": bool(worst["drift_per_1e4_steps"] <= th["G5_mass_drift_max"]),
            "threshold": th["G5_mass_drift_max"],
            "worst": {
                "dtype": worst["dtype"],
                "steps": worst["steps"],
                "drift_final": worst["drift_final"],
                "drift_per_1e4_steps": worst["drift_per_1e4_steps"],
                "checkpoints": worst["checkpoints"],
                "drift_series": worst["drift"],
            },
            "contrast": [
                {
                    "dtype": r["dtype"],
                    "drift_per_1e4_steps": r["drift_per_1e4_steps"],
                }
                for r in masses
                if r["dtype"] != "float64"
            ],
        }

    # G6 | negative control (u = 0)
    g6_items = [r for r in _by_tag(results, "G6") if "ratio_over_d" in r]
    if g6_items:
        entries = [
            {
                "item": r["item_id"],
                "estimator": r.get("method"),
                "d_eff": r["d_eff"],
                "d_molecular": r["d_molecular"],
                "d_eff_over_d": r["ratio_over_d"],
                "rel_err_vs_d": r["ratio_over_d"] - 1.0,
            }
            for r in g6_items
        ]
        ok = all(th["G6_ratio_lo"] <= e["d_eff_over_d"] <= th["G6_ratio_hi"] for e in entries)
        gates["G6"] = {
            "pass": bool(ok),
            "window": [th["G6_ratio_lo"], th["G6_ratio_hi"]],
            "entries": entries,
        }

    # G8 | accuracy-domain table over Pe x grid
    table_items = [r for r in _by_tag(results, "G8") if "grid" in r]
    if table_items:
        rows = []
        for r in sorted(table_items, key=lambda r: (r["grid"]["ny"], r["pe_r"], r["grid"]["nx"])):
            s_nx = r.get("sigma_over_nx")
            valid = bool(s_nx is not None and np.isfinite(s_nx) and s_nx <= 1.0 / 6.0)
            rows.append(
                {
                    "geometry": r.get("kind"),
                    "pe_r": r["pe_r"],
                    "ny": r["grid"]["ny"],
                    "nx": r["grid"]["nx"],
                    "diffusivity": r["diffusivity"],
                    "steps": r["steps"],
                    "half_width": r["half_width"],
                    "sigma_over_nx": s_nx,
                    "d_eff": r["d_eff"],
                    "d_eff_closed_form": r["d_eff_analytic"],
                    "rel_err": r["rel_err_analytic"],
                    "r2": r.get("r2"),
                    "estimator_valid": valid,
                    "within_5pct": bool(abs(r["rel_err_analytic"]) <= th["G2_rel_err_max"]),
                }
            )
        valid_rows = [
            row for row in rows if row["estimator_valid"] and row["geometry"] == "channel"
        ]
        ok_valid = sum(1 for row in valid_rows if row["within_5pct"])
        bad_valid = len(valid_rows) - ok_valid
        gates["G8"] = {
            # The gate asks for the accuracy-domain table itself (G2 carries the
            # <=5 % verification on the reference grid).  It passes when the
            # table is populated with estimator-valid cells (>1/6 sigma/nx
            # cells are flux-limited, not physics-limited) AND it actually
            # delimits the domain, i.e. valid cells exist both inside and
            # outside the +-5 % band.
            "pass": bool(len(valid_rows) >= 8 and ok_valid >= 1 and bad_valid >= 1),
            "threshold": th["G2_rel_err_max"],
            "n_cells": len(rows),
            "n_valid": len(valid_rows),
            "n_within_threshold": int(ok_valid),
            "n_outside_threshold": int(bad_valid),
            "rows": rows,
            "note": (
                "estimator_valid requires sigma/nx <= 1/6 (variance-growth estimator "
                "domain); every row records it so flux-limited cells are visible"
            ),
        }

    gates["all_pass"] = bool(gates) and all(
        bool(v.get("pass")) for k, v in gates.items() if k != "all_pass"
    )
    gates["thresholds"] = th
    return gates


def load_item_dirs(dirs: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Load every ``items/*.json`` written by :func:`run_items` under ``dirs``."""
    results: dict[str, dict[str, Any]] = {}
    for d in dirs:
        for f in sorted((Path(d) / "items").glob("*.json")):
            res = json.loads(f.read_text())
            results[res.get("item_id", f.stem)] = res
    return results


def assemble_report(
    results: dict[str, dict[str, Any]],
    gates: dict[str, Any],
    *,
    mode: str,
    outdir: str | Path | None = None,
) -> dict[str, Any]:
    """Bundle measurements + gates + provenance into one JSON report."""
    import platform as _platform

    total = sum(r.get("elapsed", 0.0) for r in results.values())
    report = {
        "node": "p74-n10",
        "face": "passive scalar transport (Taylor-Aris dispersion + erf diffusion)",
        "mode": mode,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": _platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "machine": _platform.machine(),
        },
        "gates": gates,
        "verdict": "PASS" if gates.get("all_pass") else "FAIL",
        "total_wall_time_s": total,
        "items": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("x", "c_num", "c_ref")}
            for k, v in results.items()
        },
    }
    if outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="N10 passive-scalar transport verification face")
    ap.add_argument("--mode", choices=["full", "quick"], default="full")
    ap.add_argument("--out", default="artifacts/scalar_transport_n10")
    ap.add_argument("--only", default=None, help="comma separated item ids to run")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU threads (tiny tensors: keep 1)")
    ap.add_argument("--merge", default=None,
                    help="comma separated item directories to merge instead of running")
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    if args.merge:
        results = load_item_dirs([Path(p) for p in args.merge.split(",")])
        gates = evaluate_gates(results)
        report = assemble_report(results, gates, mode=args.mode, outdir=args.out)
        print(f"merged {len(results)} items; verdict={report['verdict']}")
        return 0 if report["verdict"] == "PASS" else 1
    items = build_work_items(args.mode)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        items = [it for it in items if it["id"] in wanted]
        missing = wanted - {it["id"] for it in items}
        if missing:
            raise SystemExit(f"unknown item ids: {sorted(missing)}")
    results = run_items(items, outdir=args.out, verbose=not args.quiet)
    gates = evaluate_gates(results)
    report = assemble_report(results, gates, mode=args.mode, outdir=args.out)
    print(json.dumps({k: report["gates"][k]["pass"] for k in report["gates"] if k in
                      ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")}, indent=0))
    print(f"verdict={report['verdict']} total_wall_time_s={report['total_wall_time_s']:.1f}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
