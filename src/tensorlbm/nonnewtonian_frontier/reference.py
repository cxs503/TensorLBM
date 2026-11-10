"""One-dimensional exact/numerical anchors for generalised-Newtonian channel flow.

Setup
-----
Plane Poiseuille flow between two parallel plates a distance ``2H`` apart
(mid-plane at ``y = 0``), driven by a uniform body acceleration ``a``
(equivalently a pressure gradient ``G = ρ a``), with density ``ρ = 1`` in
lattice units.

For a *fully developed*, streamwise-invariant flow the momentum balance fixes
the shear stress **independently of the rheology**:

.. math::  \\tau(y) = \\rho\\, a\\, |y| , \\qquad 0 \\le |y| \\le H .

The constitutive law then gives the local shear rate by inversion,

.. math::  \\dot\\gamma(y) = \\tau^{-1}\\big(\\rho\\, a\\, |y|\\big) ,

and the velocity profile by one quadrature,

.. math::  u(y) = \\int_{|y|}^{H} \\dot\\gamma(s)\\, \\mathrm{d}s ,

which satisfies ``u(±H) = 0`` (no slip) by construction.  This is an
*independent* anchor: it never sees the lattice, only the closure.

For a power-law fluid the two steps above are analytic and the result reduces to
the classical Ostwald–de Waele profile

.. math::  u(y) = \\frac{n}{n+1}\\Big(\\frac{G}{K}\\Big)^{1/n}
           \\big[H^{(n+1)/n} - |y|^{(n+1)/n}\\big] ,

which this module uses to validate its own integrator (agreement to ~1e-9; see
``tests/test_nonnewtonian_rheology_frontier.py``).

Numerics
--------
The constitutive inversion is a vectorised bisection on ``τ(γ̇)`` (monotone for
every closure in :mod:`tensorlbm.nonnewtonian_frontier.closures`), and the
quadrature is composite 5-point Gauss–Legendre on ``n_panels`` uniform panels of
``[0, H]``, accumulated from the wall inwards.  With the defaults the reference
profile is converged to ``~1e-10`` relative, i.e. three orders of magnitude
below the 3 % engineering gate, so any error reported downstream is the LBM's.
"""

from __future__ import annotations

import math

import torch

from .closures import Closure, GAMMA_FLOOR

__all__ = [
    "channel_reference",
    "power_law_velocity",
    "power_law_wall_shear_rate",
    "plug_half_width_ideal",
    "shortest_distance_from_wall",
]

# 5-point Gauss-Legendre nodes/weights on [-1, 1]
_GL_X = (-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640)
_GL_W = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891)


# ---------------------------------------------------------------------------
# Constitutive inversion
# ---------------------------------------------------------------------------

def gamma_of_tau(
    closure: Closure,
    tau: torch.Tensor,
    n_iter: int = 120,
    gamma_hi: float = 1.0e3,
) -> torch.Tensor:
    """Invert ``τ = γ̇ ν(γ̇)`` by vectorised bisection.

    Args:
        closure: Constitutive closure.
        tau: Shear-stress magnitude(s) (torch, float64 recommended).
        n_iter: Bisection iterations on the bracket ``[0, gamma_hi]``.
        gamma_hi: Upper bracket for the shear rate (lattice units).

    Returns:
        Shear rate with the same shape as ``tau``.
    """
    if closure.tau_y > 0.0:
        # ideal yield stress: below it the material is unyielded (gamma = 0);
        # the regularised closure still returns a (very small) rate, which is
        # the physically meaningful branch for the *regularised* model.
        pass
    tau = torch.as_tensor(tau, dtype=torch.float64)
    lo = torch.zeros_like(tau)
    hi = torch.full_like(tau, float(gamma_hi))
    # guard the bracket: extend hi until tau(hi) > tau
    for _ in range(60):
        need = closure.tau_of_gamma(hi) < tau
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 10.0, hi)
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        below = closure.tau_of_gamma(torch.clamp(mid, min=GAMMA_FLOOR)) < tau
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Reference profile
# ---------------------------------------------------------------------------

def channel_reference(
    closure: Closure,
    accel: float,
    half_width: float,
    y: torch.Tensor | None = None,
    n_panels: int = 4000,
) -> dict:
    """Exact-in-the-limit velocity/shear-rate/stress profile of a plane channel.

    Args:
        closure: Constitutive closure (lattice-unit parameters).
        accel: Body acceleration ``a`` in +x (lattice units).
        half_width: Channel half-width ``H`` (lattice units); walls at ±H.
        y: Sample points, mid-plane origin.  ``None`` returns the fine anchor
            grid itself (``n_panels + 1`` points on ``[0, H]``).
        n_panels: Number of Gauss–Legendre panels on ``[0, H]``.

    Returns:
        dict with ``y`` (mid-plane origin, sign-preserving sample points),
        ``u``, ``gamma_dot``, ``tau`` (all at the sample points), plus
        ``mean_velocity``, ``flow_rate`` (per unit width), ``u_max``,
        ``gamma_dot_wall`` and ``fine`` (the anchor grid and its profile).
    """
    if accel <= 0.0:
        raise ValueError("accel must be > 0")
    if half_width <= 0.0:
        raise ValueError("half_width must be > 0")
    n_panels = int(n_panels)
    if n_panels < 4:
        raise ValueError("n_panels must be >= 4")

    H = float(half_width)
    a = float(accel)
    edges = torch.linspace(0.0, H, n_panels + 1, dtype=torch.float64)
    width = edges[1] - edges[0]
    # 5-point GL panels: s^p_j and the panel integral of gamma_dot
    panel_int = torch.zeros(n_panels, dtype=torch.float64)
    for xj, wj in zip(_GL_X, _GL_W):
        s = edges[:-1] + 0.5 * width * (xj + 1.0)
        panel_int = panel_int + 0.5 * width * wj * gamma_of_tau(closure, a * s)
    # cumulative integral from the wall (s = H) inwards: u(s_i) = sum_{p>=i} int_p
    rev = torch.flip(panel_int, dims=(0,))
    u_rev = torch.cumsum(rev, dim=0)
    u_fine = torch.zeros(n_panels + 1, dtype=torch.float64)
    u_fine[:-1] = torch.flip(u_rev, dims=(0,))  # u at s_0..s_{n-1}; u(s_H) = 0
    gamma_fine = gamma_of_tau(closure, a * edges)

    fine = {"s": edges, "u": u_fine, "gamma_dot": gamma_fine, "tau": a * edges}

    mean_velocity = float(torch.trapz(u_fine, edges) / H)
    out = {
        "u_max": float(u_fine[0]),
        "mean_velocity": mean_velocity,
        "flow_rate": mean_velocity * 2.0 * H,
        "gamma_dot_wall": float(gamma_fine[-1]),
        "tau_wall": a * H,
        "fine": fine,
        "params": {"accel": a, "half_width": H, "closure": closure.name},
    }

    if y is None:
        s = edges
        sign = torch.ones_like(s)
    else:
        y = torch.as_tensor(y, dtype=torch.float64)
        s = torch.clamp(y.abs(), max=H)
        sign = torch.sign(y)
    # linear interpolation of the converged anchor grid (error ~1e-10 here)
    u = torch.zeros_like(s)
    g = torch.zeros_like(s)
    u[:] = _interp(edges, u_fine, s)
    g[:] = _interp(edges, gamma_fine, s)
    out["y"] = s * sign
    out["u"] = u
    out["gamma_dot"] = g
    out["tau"] = a * s
    return out


def _interp(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Monotone linear interpolation of ``y(x)`` at ``xq`` (clamped ends)."""
    n = x.numel()
    idx = torch.searchsorted(x, xq.contiguous(), right=True) - 1
    idx = torch.clamp(idx, 0, n - 2)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# ---------------------------------------------------------------------------
# Closed-form power-law anchors
# ---------------------------------------------------------------------------

def power_law_wall_shear_rate(half_width: float, accel: float, consistency_index: float, flow_index: float) -> float:
    """Wall shear rate ``γ̇_w = (G H / K)^(1/n)`` of a power-law channel."""
    return float((accel * half_width / consistency_index) ** (1.0 / flow_index))


def power_law_velocity(
    y: torch.Tensor,
    half_width: float,
    accel: float,
    consistency_index: float,
    flow_index: float,
) -> torch.Tensor:
    """Closed-form power-law plane-Poiseuille profile (mid-plane origin).

    ``u(y) = n/(n+1) (G/K)^{1/n} [H^{(n+1)/n} - |y|^{(n+1)/n}]``
    """
    n, K, G, H = float(flow_index), float(consistency_index), float(accel), float(half_width)
    if n <= 0.0 or K <= 0.0 or G <= 0.0 or H <= 0.0:
        raise ValueError("n, K, G, H must all be > 0")
    y = torch.as_tensor(y, dtype=torch.float64)
    pref = (n / (n + 1.0)) * (G / K) ** (1.0 / n)
    return pref * (_safe_pow(H, (n + 1.0) / n) - _safe_pow(y.abs(), (n + 1.0) / n))


def _safe_pow(x: torch.Tensor | float, p: float) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.pow(torch.clamp(x, min=0.0), p)


# ---------------------------------------------------------------------------
# Yield-stress geometry
# ---------------------------------------------------------------------------

def plug_half_width_ideal(accel: float, yield_stress: float) -> float:
    """Plug half-width (mid-plane distance) of an ideal yield-stress channel.

    ``τ(y) = G|y|`` and the material yields where ``τ = τ_y``, so the plug
    extends to ``|y| = τ_y / G`` (clipped at the channel half-width).
    """
    if accel <= 0.0:
        raise ValueError("accel must be > 0")
    if yield_stress < 0.0:
        raise ValueError("yield_stress must be >= 0")
    return float(yield_stress / accel)


def shortest_distance_from_wall(y: torch.Tensor, half_width: float) -> torch.Tensor:
    """Distance from the nearest wall for mid-plane-origin coordinates."""
    return float(half_width) - torch.as_tensor(y, dtype=torch.float64).abs()

# ---------------------------------------------------------------------------
# Driver calibration
# ---------------------------------------------------------------------------

def calibrate_accel(
    closure: Closure,
    half_width: float,
    u_max_target: float,
    n_panels: int = 400,
    n_iter: int = 60,
) -> dict:
    """Body acceleration that reproduces a target centreline velocity.

    The bulk velocity of a generalised-Newtonian channel is *not* linear in the
    driving acceleration (the viscosity itself depends on the shear rate), so
    the acceleration is found by a bisection on ``log a`` of the exact 1-D
    solution.  Keeping ``u_max`` fixed across all closures makes the lattice
    Mach number — and therefore the compressibility error — identical in every
    case, which is what makes the cross-model error comparison meaningful.

    Args:
        closure: Constitutive closure.
        half_width: Channel half-width ``H``.
        u_max_target: Target centreline velocity (lattice units).
        n_panels: Quadrature panels for the probe solves.
        n_iter: Bisection iterations.

    Returns:
        dict with ``accel``, ``u_max`` (achieved), ``u_max_target``,
        ``relative_error``, ``iterations``.
    """
    if u_max_target <= 0.0:
        raise ValueError("u_max_target must be > 0")

    def u_max_of(a: float) -> float:
        return channel_reference(closure, a, half_width, y=None, n_panels=n_panels)["u_max"]

    lo, hi = 1.0e-12, 1.0e-1
    best = hi
    for _ in range(int(n_iter)):
        mid = math.sqrt(lo * hi)
        if u_max_of(mid) < u_max_target:
            lo = mid
        else:
            hi = mid
        best = lo
    achieved = u_max_of(best)
    return {
        "accel": best,
        "u_max": achieved,
        "u_max_target": u_max_target,
        "relative_error": achieved / u_max_target - 1.0,
        "iterations": int(n_iter),
    }
