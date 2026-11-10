"""Plug-flow (yield surface) measurement for ice-slurry Herschel–Bulkley flow.

An ideal yield-stress fluid in a plane channel carries the stress
``τ(y) = ρ a |y|`` (momentum balance, rheology-independent) and does not deform
where ``τ < τ_y``, so the plug is the band ``|y| ≤ r_c`` with

.. math::  r_c = \\tau_y / (\\rho a) .

A *regularised* model, however, never has an exactly rigid plug: the
Papanastasiou branch ``ν = τ_y(1-e^{-mγ̇})/γ̇ + Kγ̇^{n-1}`` turns the yield
surface into a transition band of width ``γ̇ ~ 1/m``.  The plug boundary is thus
a *measurement*, and three independent detectors are provided:

``velocity``  outermost mid-plane distance at which the velocity is still within
              ``δ·u_max`` of the centreline value (threshold / knee detector);
``stress``    locus where the stress reconstructed from the *measured* shear
              rate, ``τ = γ̇ν(γ̇)``, crosses ``τ_y`` (constitutive detector);
``fit``       one-parameter least-squares fit of the *ideal* H-B yielded-region
              profile ``u(r) = u_c − A (r − r_c)^{(n+1)/n}`` with ``r_c`` free
              (and ``u_c``, ``A`` implied by ``r_c``).  This one is insensitive
              to the smearing of the regularisation because it uses the whole
              yielded branch, not a single threshold crossing.

Every detector is applied to the LBM profile **and** to the exact reference
profile with the identical code path, so the reported LBM error is a pure
discretisation error while the reference-vs-ideal gap is the regularisation
bias of the chosen ``m``; the two are never mixed up.
"""

from __future__ import annotations

import torch

from .closures import Closure

__all__ = [
    "symmetrise",
    "plug_half_width_velocity",
    "plug_half_width_stress",
    "plug_half_width_fit",
    "plug_half_width_ideal_from_closure",
    "plug_report",
]


def symmetrise(y: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a profile with mid-plane origin onto ``|y|`` (mean of the ±pair).

    Returns ``(r, u_sym)`` with ``r`` ascending on ``[0, H]``.
    """
    y = torch.as_tensor(y, dtype=torch.float64)
    u = torch.as_tensor(u, dtype=torch.float64)
    r = y.abs()
    order = torch.argsort(r)
    r_sorted, u_sorted = r[order], u[order]
    uniq, inverse = torch.unique(r_sorted, return_inverse=True)
    acc = torch.zeros(uniq.numel(), dtype=torch.float64)
    cnt = torch.zeros(uniq.numel(), dtype=torch.float64)
    acc.index_add_(0, inverse, u_sorted)
    cnt.index_add_(0, inverse, torch.ones_like(u_sorted))
    return uniq, acc / cnt


def _crossing_decreasing(r: torch.Tensor, f: torch.Tensor, level: float) -> float:
    """First radius where the decreasing ``f(r)`` falls to ``level`` (linear)."""
    below = f < level
    if not bool(below.any()):
        return float(r[-1])
    i = int(torch.nonzero(below, as_tuple=False)[0, 0])
    if i == 0:
        return float(r[0])
    r0, r1 = float(r[i - 1]), float(r[i])
    f0, f1 = float(f[i - 1]), float(f[i])
    if f1 == f0:
        return r1
    return r0 + (level - f0) * (r1 - r0) / (f1 - f0)


def _crossing_increasing(r: torch.Tensor, f: torch.Tensor, level: float) -> float:
    """First radius where the increasing ``f(r)`` reaches ``level`` (linear)."""
    above = f >= level
    if not bool(above.any()):
        return float(r[-1])
    i = int(torch.nonzero(above, as_tuple=False)[0, 0])
    if i == 0:
        return float(r[0])
    r0, r1 = float(r[i - 1]), float(r[i])
    f0, f1 = float(f[i - 1]), float(f[i])
    if f1 == f0:
        return r1
    return r0 + (level - f0) * (r1 - r0) / (f1 - f0)


def plug_half_width_velocity(r: torch.Tensor, u: torch.Tensor, delta: float = 0.01) -> float:
    """Knee detector: outermost ``r`` with ``u(r) ≥ u_max (1 - δ)``.

    Args:
        r: Mid-plane distance, ascending, ``0 ≤ r ≤ H``.
        u: Symmetrised velocity on ``r``.
        delta: Relative velocity deficit that defines "still plug".

    Returns:
        Plug half-width estimate.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    u = torch.as_tensor(u, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64)
    u_max = float(u.max())
    if u_max <= 0.0:
        raise ValueError("velocity profile must be positive")
    return _crossing_decreasing(r, u, (1.0 - delta) * u_max)


def plug_half_width_stress(
    r: torch.Tensor,
    gamma: torch.Tensor,
    closure: Closure,
    level: float | None = None,
) -> float:
    """Constitutive detector: first ``r`` with ``γ̇(r) ν(γ̇(r)) ≥ τ_y``.

    Args:
        r: Mid-plane distance, ascending.
        gamma: Symmetrised shear-rate profile on ``r``.
        closure: Closure that produced the flow (supplies ``ν(γ̇)``).
        level: Stress level to detect (defaults to the closure's ``τ_y``).

    Returns:
        Plug half-width estimate.
    """
    tau_y = closure.tau_y if level is None else float(level)
    if tau_y <= 0.0:
        raise ValueError("closure has no yield stress; the stress detector needs tau_y > 0")
    g = torch.clamp(torch.as_tensor(gamma, dtype=torch.float64), min=1.0e-300)
    tau = g * closure.nu_of_gamma(g)
    return _crossing_increasing(r, tau, tau_y)


def plug_half_width_fit(
    r: torch.Tensor,
    u: torch.Tensor,
    accel: float,
    closure: Closure,
    r_lo_frac: float = 0.1,
    r_hi_frac: float = 0.85,
    n_scan: int = 400,
    half_width: float | None = None,
) -> float:
    """One-parameter fit of the ideal H-B yielded branch.

    For an ideal Herschel–Bulkley fluid in a plane channel the yielded branch is

    .. math::  u(r) = u_c - A\\,(r - r_c)^{(n+1)/n}, \\quad
               A = \\Big(\\frac{\\rho a}{K}\\Big)^{1/n}\\frac{n}{n+1}, \\quad
               u_c = A\\,(H - r_c)^{(n+1)/n},

    so with ``r_c`` as the only free parameter the whole yielded profile is
    predicted.  The detector scans ``r_c ∈ [r_lo_frac·H, r_hi_frac·H]`` and
    returns the value minimising the least-squares misfit over the yielded
    region ``r ∈ [r_c, H]`` (a fine golden-section refinement follows the scan).

    Args:
        r: Mid-plane distance, ascending, ``0 ≤ r ≤ H``.
        u: Symmetrised velocity on ``r``.
        accel: Body acceleration ``a`` (lattice units).
        closure: H-B closure (supplies ``K``, ``n``; ``τ_y`` only sets the scale).
        r_lo_frac, r_hi_frac: Search bracket as fractions of ``H``.
        n_scan: Number of scan points before the refinement.
        half_width: Channel half-width ``H``.  Defaults to ``r[-1]``; pass it
            explicitly whenever the profile still contains the solid rows
            (whose radius is ``H + 1/2`` and would otherwise inflate ``H``).

    Returns:
        Plug half-width estimate.
    """
    if accel <= 0.0:
        raise ValueError("accel must be > 0")
    params = closure.params
    if "K" not in params or "n" not in params:
        raise ValueError("plug_half_width_fit needs an H-B/power-law closure with K and n")
    K, n = float(params["K"]), float(params["n"])
    r = torch.as_tensor(r, dtype=torch.float64)
    u = torch.as_tensor(u, dtype=torch.float64)
    H = float(r[-1]) if half_width is None else float(half_width)
    if float(r[-1]) > H * (1.0 + 1.0e-12):
        keep = r <= H * (1.0 + 1.0e-12)
        r, u = r[keep], u[keep]
    p = (n + 1.0) / n
    A = (accel / K) ** (1.0 / n) * n / (n + 1.0)

    def misfit(rc: float) -> float:
        if rc >= H:
            return float("inf")
        mask = r >= rc
        rr = r[mask]
        u_pred = A * ((H - rc) ** p - torch.clamp(rr - rc, min=0.0) ** p)
        d = u[mask] - u_pred
        return float((d * d).mean())

    grid = torch.linspace(r_lo_frac * H, r_hi_frac * H, int(n_scan))
    vals = torch.tensor([misfit(float(x)) for x in grid])
    i = int(torch.argmin(vals))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, grid.numel() - 1)])
    # golden-section refinement
    phi = (5 ** 0.5 - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = misfit(c), misfit(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = misfit(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = misfit(d)
    return 0.5 * (a + b)


def plug_half_width_ideal_from_closure(closure: Closure, accel: float) -> float:
    """Ideal (unregularised) plug half-width ``τ_y / (ρ a)``."""
    if accel <= 0.0:
        raise ValueError("accel must be > 0")
    return float(closure.tau_y / accel)


def plug_report(
    closure: Closure,
    accel: float,
    y_lbm: torch.Tensor,
    u_lbm: torch.Tensor,
    gamma_lbm: torch.Tensor,
    y_ref: torch.Tensor | None = None,
    u_ref: torch.Tensor | None = None,
    gamma_ref: torch.Tensor | None = None,
    delta: float = 0.01,
    half_width: float | None = None,
) -> dict:
    """Compare the three plug detectors, LBM vs reference vs ideal.

    ``rel_*`` entries are relative deviations from the ideal ``τ_y/(ρa)`` and
    ``lbm_vs_ref_*`` are like-for-like discretisation errors.
    """
    def _fold(yy, uu):
        rr, uu_s = symmetrise(yy, uu)
        if half_width is not None:
            keep = rr <= float(half_width) * (1.0 + 1.0e-12)
            rr, uu_s = rr[keep], uu_s[keep]
        return rr, uu_s

    r_l, u_l = _fold(y_lbm, u_lbm)
    g_l = torch.clamp(_fold(y_lbm, gamma_lbm)[1], min=0.0)
    r_c_ideal = plug_half_width_ideal_from_closure(closure, accel)
    out = {
        "r_c_ideal": r_c_ideal,
        "delta": delta,
        "r_c_lbm_velocity": plug_half_width_velocity(r_l, u_l, delta=delta),
        "r_c_lbm_stress": plug_half_width_stress(r_l, g_l, closure),
        "r_c_lbm_fit": plug_half_width_fit(r_l, u_l, accel, closure, half_width=half_width),
    }
    if y_ref is not None:
        r_r, u_r = _fold(y_ref, u_ref)
        g_r = torch.clamp(_fold(y_ref, gamma_ref)[1], min=0.0)
        out["r_c_ref_velocity"] = plug_half_width_velocity(r_r, u_r, delta=delta)
        out["r_c_ref_stress"] = plug_half_width_stress(r_r, g_r, closure)
        out["r_c_ref_fit"] = plug_half_width_fit(r_r, u_r, accel, closure, half_width=half_width)
        for key in ("velocity", "stress", "fit"):
            out[f"rel_ref_{key}"] = out[f"r_c_ref_{key}"] / r_c_ideal - 1.0
            out[f"lbm_vs_ref_{key}"] = out[f"r_c_lbm_{key}"] / out[f"r_c_ref_{key}"] - 1.0
    for key in ("velocity", "stress", "fit"):
        out[f"rel_lbm_{key}"] = out[f"r_c_lbm_{key}"] / r_c_ideal - 1.0
    return out