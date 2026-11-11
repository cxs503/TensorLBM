"""Capillary oscillation of a (diffuse-interface) drop: measurement + anchors.

Why this module exists
----------------------
``verification.droplet_oscillation`` measured the oscillation period from the
zero crossings of the ``phi`` second-moment anisotropy and compared it with
:func:`csf.rayleigh_period`, i.e. the **drop-in-vacuum** Rayleigh formula.  A
single-phase free-surface LBM has *one* uniform density on both sides of the
interface (it solves one fluid, there is no density jump), so the physically
matching linear anchor is the **two-fluid** Rayleigh period with
``rho_inner = rho_outer`` (``sqrt(2)`` longer than the vacuum value).  Mixing
the two conventions accounts for a ``+41 %`` period error before any
discretisation error is even considered; see
``docs/P284W3_N02_SURFACE_TENSION_V2.md`` §3.

This module therefore provides

* a phase estimator that is robust on a *damped* signal (least-squares fit of
  ``exp(-gamma t) [a cos(omega t) + b sin(omega t)] + c`` over a
  ``(omega, gamma)`` scan, with the fit residual reported as ``R^2``),
* a purely geometric interface marker (``r`` where ``phi = 0.5`` along the four
  axes) so the second-moment signal can be cross-checked against the actual
  interface shape,
* stability diagnostics (parasitic-current level, growth of ``max|u|``, mass
  drift) because the near-inviscid limit is where a BGK lattice first becomes
  noisy -- a run whose ``max|u|`` grows is *not* evidence,
* the viscous-limit extrapolation used to compare with the inviscid Rayleigh
  anchor: ``T/T_inv - 1 = c sqrt(nu / (omega a^2))``, fitted over a viscosity
  ladder (``nu = cs^2 (tau - 1/2)``).

The reference period for every case is
:func:`tensorlbm.free_surface_tension.csf.rayleigh_period_2fluid` with
``rho_inner = rho_outer = cfg.rho0``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .csf import curvature, gradient, rayleigh_period_2fluid, tanh_ellipse_phi
from .solver import CapillarySolver, SolverConfig

__all__ = [
    "curvature_mode2_amplitude",
    "damped_sinusoid_fit",
    "interface_anisotropy",
    "oscillation_case",
    "inviscid_extrapolation",
    "viscous_parameter",
]


def damped_sinusoid_fit(
    t: Sequence[float],
    y: Sequence[float],
    period_guess: float,
    window: tuple[float, float] = (0.7, 1.4),
    n_omega: int = 601,
    n_gamma: int = 9,
) -> dict:
    """Fit ``y(t) = exp(-g t) (a cos wt + b sin wt) + c`` by linear least squares.

    ``(w, g)`` are scanned on a grid (the amplitude combination is linear for
    fixed ``(w, g)``, so each grid point is one ``lstsq`` solve); the grid point
    with the smallest residual is reported together with its ``R^2``.

    Args:
        t: Sample times (uniformity not required).
        y: Signal.
        period_guess: Initial period; the ``w`` grid spans
            ``window[0] .. window[1]`` times ``2 pi / period_guess``.
        window: Multiplicative bracket for the angular-frequency scan.
        n_omega: Number of ``w`` samples.
        n_gamma: Number of ``g`` samples in ``[0, 4.5 / (t[-1] - t[0])]``.

    Returns:
        ``dict`` with ``period``, ``omega``, ``gamma``, ``amplitude``,
        ``offset``, ``r2`` and ``n_samples``.
    """
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    yy = yy - yy.mean()
    w0 = 2.0 * math.pi / float(period_guess)
    span = float(tt[-1] - tt[0]) or 1.0
    best = None
    for om in np.linspace(window[0] * w0, window[1] * w0, int(n_omega)):
        for g in np.linspace(0.0, 4.5 / span, int(n_gamma)):
            env = np.exp(-g * tt)
            design = np.stack(
                [env * np.cos(om * tt), env * np.sin(om * tt), np.ones_like(tt)], axis=1
            )
            coef, *_ = np.linalg.lstsq(design, yy, rcond=None)
            resid = float(((yy - design @ coef) ** 2).sum())
            if best is None or resid < best[0]:
                best = (resid, om, g, coef)
    resid, om, g, coef = best
    ss_tot = float((yy**2).sum()) or 1.0
    return {
        "period": 2.0 * math.pi / om,
        "omega": float(om),
        "gamma": float(g),
        "amplitude": float(math.hypot(coef[0], coef[1])),
        "offset": float(coef[2]),
        "r2": 1.0 - resid / ss_tot,
        "n_samples": int(len(tt)),
    }


def _crossing_radius(phi: np.ndarray, centre: float, axis: int, sign: int, limit: int) -> float:
    """Distance from ``centre`` to the ``phi = 0.5`` crossing along an axis.

    Cell ``i`` sits at the physical position ``i + 0.5``; the returned value is
    in lattice units and linearly interpolated between the two cells that
    bracket the crossing.  ``nan`` if no crossing is found within ``limit``.
    """
    j = int(centre - 0.5)
    prev = None
    for k in range(1, int(limit)):
        i = j + sign * k
        if i < 0 or i >= phi.shape[axis]:
            return float("nan")
        val = float(phi[j, i]) if axis == 0 else float(phi[i, j])
        if prev is not None and prev[1] >= 0.5 > val:
            frac = (prev[1] - 0.5) / (prev[1] - val)
            return float(k - 1 + frac)
        prev = (k, val)
    return float("nan")


def interface_anisotropy(solver: CapillarySolver, radius: float, n_axes: int = 4) -> float:
    """``(r_x - r_y) / R`` from the ``phi = 0.5`` crossings on the axes.

    The second-moment anisotropy of ``phi`` (``(Q_xx - Q_yy)/(Q_xx + Q_yy)``)
    is a *mass*-weighted shape measure and therefore also senses any
    width modulation of the diffuse interface band; the marker measure is a
    purely geometric cross-check (they agree to ~0.3 % on the cases below).
    """
    phi = solver.phi.detach().to("cpu").double().numpy()
    centre = 0.5 * phi.shape[0]
    limit = int(2.5 * radius + 8 * solver.cfg.width)
    rx = 0.5 * (
        _crossing_radius(phi, centre, 0, +1, limit)
        + _crossing_radius(phi, centre, 0, -1, limit)
    )
    ry = 0.5 * (
        _crossing_radius(phi, centre, 1, +1, limit)
        + _crossing_radius(phi, centre, 1, -1, limit)
    )
    return (rx - ry) / float(radius)


def curvature_mode2_amplitude(
    phi: torch.Tensor,
    semi_axes: tuple[float, float],
    smooth: int = 0,
    n_theta: int = 721,
) -> dict:
    """``cos 2 theta`` component of the discrete curvature, band-weighted.

    The CSF force is ``sigma kappa grad(phi)``, so the mode-2 *stiffness* of a
    capillary oscillation is set by the ``delta_s``-weighted projection of
    ``kappa`` onto the mode-2 displacement ``cos 2 theta``::

        A2 = sum( kappa |grad phi| cos 2 theta ) / sum( |grad phi| )

    which is exactly the generalised force conjugate to a radial displacement
    ``q cos 2 theta`` divided by ``sigma`` and by the interface measure.  The
    projection is taken *over the whole interface band* rather than by
    sampling ``kappa`` at a few points: pointwise values of
    ``-div(n_hat)`` are only meaningful where ``|grad phi|`` is large, and
    sub-cell sampling differences alone move the pointwise mode-2 estimate by
    tens of percent at ``w = 2`` (measured), which makes a pointwise audit
    useless as evidence.

    The analytic reference is the same projection for the exact ellipse
    ``a b / sqrt(b^2 cos^2 theta + a^2 sin^2 theta)`` with its exact curvature,
    integrated over its arc length.

    Args:
        phi: Ellipse indicator, shape ``(L, L)``, centred at ``(L/2, L/2)``.
        semi_axes: ``(a, b)`` semi-axes of the ``phi = 0.5`` ellipse.
        smooth: Binomial passes applied before the curvature.
        n_theta: Samples for the analytic arc-length integral.

    Returns:
        ``dict`` with ``mode2_ratio`` (numeric / analytic), ``a2_numeric``,
        ``a2_exact`` and ``mean_ratio`` (the ``cos 0`` component).
    """
    a, b = float(semi_axes[0]), float(semi_axes[1])
    L = int(phi.shape[0])
    centre = 0.5 * L
    f = phi.detach().to("cpu").double()
    kappa = curvature(f, smooth=int(smooth))
    g = gradient(f)
    delta = torch.sqrt(sum(gi * gi for gi in g) + 1e-30)
    yy, xx = torch.meshgrid(
        torch.arange(L, dtype=torch.float64) + 0.5,
        torch.arange(L, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    theta = torch.atan2(yy - centre, xx - centre)
    w = delta / float(delta.sum())
    a2_num = float((w * kappa * torch.cos(2.0 * theta)).sum())
    a0_num = float((w * kappa).sum())
    th = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    r = a * b / np.sqrt(b * b * np.cos(th) ** 2 + a * a * np.sin(th) ** 2)
    dr = np.gradient(r, th)
    dl = np.sqrt(r * r + dr * dr)
    kap = a * b / (a * a * np.sin(np.arctan2(a * np.sin(th), b * np.cos(th))) ** 2
                   + b * b * np.cos(np.arctan2(a * np.sin(th), b * np.cos(th))) ** 2) ** 1.5
    a2_ex = float((dl * kap * np.cos(2.0 * th)).sum() / dl.sum())
    a0_ex = float((dl * kap).sum() / dl.sum())
    return {
        "mode2_ratio": a2_num / a2_ex,
        "mean_ratio": a0_num / a0_ex,
        "a2_numeric": a2_num,
        "a2_exact": a2_ex,
    }


def viscous_parameter(tau: float, period: float, radius: float, cs2: float = 1.0 / 3.0) -> float:
    """Dimensionless viscous amplitude ``sqrt(nu / (omega a^2))``.

    ``nu = cs^2 (tau - 1/2)`` in lattice units.  This is the small parameter of
    the viscous frequency correction of a capillary oscillator (the ratio of
    the viscous penetration depth ``sqrt(2 nu / omega)`` to the radius), so the
    period is expected to behave as ``T = T_inv (1 + c sqrt(nu/(omega a^2)))``
    for small ``nu``.
    """
    nu = cs2 * (float(tau) - 0.5)
    omega = 2.0 * math.pi / float(period)
    return math.sqrt(nu / (omega * radius * radius))


def oscillation_case(
    radius: float = 16.0,
    L: int | None = None,
    width: float = 2.0,
    tau: float = 0.55,
    sigma: float = 0.04,
    eps: float = 0.02,
    periods: float = 3.0,
    curvature_smooth: int = 1,
    advection: str = "sl",
    dtype: torch.dtype = torch.float64,
    sample: int = 3,
    marker_every: int = 15,
) -> dict:
    """Initial ellipse ``(R(1+eps), R(1-eps))`` relaxing with the mode-2 period.

    The reference period is the **two-fluid** Rayleigh period with equal
    densities (:func:`csf.rayleigh_period_2fluid`), i.e. the limit this
    single-phase solver actually realises.  Both the second-moment signal and
    the interface marker are fitted; the stability diagnostics
    (``umax`` growth, mass drift) are returned so a contaminated run can be
    rejected instead of being reported as evidence.

    Args:
        radius: Equilibrium radius ``R`` (lattice units).
        L: Box size (default ``8 R``: the outer potential-flow tail outside a
            drop in a periodic box already integrates to ``0.998 x`` the
            infinite-domain added mass at ``8 R``).
        width: Diffuse-interface thickness ``w`` (tanh profile).
        tau: BGK relaxation time (viscosity ``nu = (tau - 1/2)/3``).
        sigma: Surface-tension coefficient.
        eps: Initial mode-2 amplitude (``D(0) ~ 2 eps``).
        periods: Integration length in analytic periods.
        curvature_smooth: Binomial passes before the curvature
            (:class:`~.solver.SolverConfig`); ``1`` is the recommended value.
        advection: ``"sl"`` (monotone cubic semi-Lagrangian, default) or
            ``"upwind"`` (conservative flux form).  Exposed so the phase can be
            shown to be insensitive to the interface-transport scheme.
        dtype: Torch dtype.
        sample: Sampling stride for the anisotropy signal.
        marker_every: Sampling stride for the (costlier) geometric marker.

    Returns:
        ``dict`` with ``params``, ``metrics`` and ``series``.
    """
    L = int(8 * radius) if L is None else int(L)
    cfg = SolverConfig(
        sigma=sigma,
        tau_liquid=tau,
        tau_gas=tau,
        width=width,
        advection=advection,
        dtype=dtype,
        curvature_smooth=int(curvature_smooth),
    )
    solver = CapillarySolver((L, L), cfg)
    solver.set_phi(
        tanh_ellipse_phi(
            (L, L), (0.5 * L, 0.5 * L), (radius * (1.0 + eps), radius * (1.0 - eps)), width=width
        )
    )
    t_ref = rayleigh_period_2fluid(sigma, radius, rho_inner=cfg.rho0, rho_outer=cfg.rho0, mode=2, ndim=2)
    t_ref_vacuum = rayleigh_period_2fluid(
        sigma, radius, rho_inner=cfg.rho0, rho_outer=0.0, mode=2, ndim=2
    )
    n_steps = int(periods * t_ref)

    t_hist: list[float] = []
    d_hist: list[float] = []
    a_hist: list[tuple[float, float]] = []
    u_hist: list[float] = []
    m_hist: list[float] = []
    done = 0
    while done < n_steps:
        step = min(sample, n_steps - done)
        solver.step(step, advect=True)
        done += step
        t_hist.append(float(done))
        d_hist.append(solver.second_moment_anisotropy())
        u_hist.append(solver.max_speed())
        m_hist.append(solver.phi_mass())
        if done % (sample * marker_every) < sample:
            a_hist.append((float(done), interface_anisotropy(solver, radius)))

    fit_d = damped_sinusoid_fit(t_hist, d_hist, t_ref)
    if len(a_hist) >= 8:
        fit_a = damped_sinusoid_fit([p[0] for p in a_hist], [p[1] for p in a_hist], t_ref)
    else:
        fit_a = {"period": float("nan"), "gamma": float("nan"), "r2": float("nan"), "amplitude": float("nan")}

    # stability: parasitic-current level and its growth over the run
    u_head = float(np.median(u_hist[: max(3, len(u_hist) // 10)]))
    u_tail = float(np.median(u_hist[-max(3, len(u_hist) // 10) :]))
    amp_head = float(np.max(np.abs(d_hist[: max(3, len(d_hist) // 10)])))
    amp_tail = float(np.max(np.abs(d_hist[-max(3, len(d_hist) // 10) :])))
    mass_drift = abs(m_hist[-1] - m_hist[0]) / m_hist[0]

    period = fit_d["period"]
    return {
        "case": "droplet_oscillation_two_fluid",
        "params": {
            "radius": radius,
            "L": L,
            "width": width,
            "width_over_radius": width / radius,
            "tau": tau,
            "nu": (tau - 0.5) / 3.0,
            "sigma": sigma,
            "eps": eps,
            "periods": periods,
            "steps": n_steps,
            "curvature_smooth": int(curvature_smooth),
            "advection": advection,
            "dtype": str(dtype).replace("torch.", ""),
            "sample": sample,
        },
        "metrics": {
            "period": period,
            "period_marker": fit_a["period"],
            "period_ref_two_fluid": t_ref,
            "period_ref_vacuum": t_ref_vacuum,
            "period_rel_err": period / t_ref - 1.0,
            "period_rel_err_marker": fit_a["period"] / t_ref - 1.0,
            "period_rel_err_vs_vacuum": period / t_ref_vacuum - 1.0,
            "gamma_fit": fit_d["gamma"],
            "r2_fit": fit_d["r2"],
            "r2_marker": fit_a["r2"],
            "viscous_parameter": viscous_parameter(tau, period, radius),
            "umax": max(u_hist),
            "umax_head": u_head,
            "umax_tail": u_tail,
            "umax_growth": u_tail / u_head if u_head else float("inf"),
            "amp_head": amp_head,
            "amp_tail": amp_tail,
            "amp_growth": amp_tail / amp_head if amp_head else float("inf"),
            "mass_drift": mass_drift,
            "stable": bool(u_tail / u_head < 2.0 and mass_drift < 1e-2),
        },
        "series": {
            "t": t_hist,
            "D": d_hist,
            "umax": u_hist,
            "mass": m_hist,
            "marker_t": [p[0] for p in a_hist],
            "marker_a": [p[1] for p in a_hist],
        },
    }


def inviscid_extrapolation(points: Sequence[dict]) -> dict:
    """Fit ``T/T_inv - 1 = c sqrt(nu/(omega a^2))`` and report the ``nu -> 0`` value.

    Args:
        points: Dictionaries with at least ``tau``, ``period``, ``radius``,
            ``period_ref_two_fluid`` and ``stable`` (as produced by
            :func:`oscillation_case` ``["metrics"]`` + ``["params"]``; pass a
            flat dict per point).

    Returns:
        ``dict`` with the fitted slope ``c``, the intercept (the extrapolated
        relative error at ``nu = 0``), ``r2`` and ``n_points``.  Unstable runs
        are excluded (and listed in ``rejected``).
    """
    xs: list[float] = []
    ys: list[float] = []
    used: list[float] = []
    rejected: list[float] = []
    for p in points:
        if not p.get("stable", True):
            rejected.append(float(p["tau"]))
            continue
        x = viscous_parameter(p["tau"], p["period"], p["radius"])
        y = p["period"] / p["period_ref_two_fluid"] - 1.0
        xs.append(x)
        ys.append(y)
        used.append(float(p["tau"]))
    if len(xs) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "n_points": len(xs),
            "taus_used": used,
            "rejected": rejected,
        }
    x = np.asarray(xs)
    y = np.asarray(ys)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return {
        "slope": float(a),
        "intercept": float(b),
        "r2": 1.0 - ss_res / ss_tot,
        "n_points": int(len(x)),
        "taus_used": used,
        "rejected": rejected,
        "x": xs,
        "y": ys,
    }