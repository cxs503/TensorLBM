"""Young-Laplace pressure calibration for the Shan-Chen (SCMP) model.

This module implements the *pressure calibration* procedure of the reference
implementation ``lbm-principles-practice/code`` (``chapter9/shanchen.cpp``):

* ``computePressure``       (``:126-139``)  p = sum_s rho_s/3
                                           + sum_s sum_ss g[s][ss] psi_s psi_ss/6
* ``calculateDeltaPressure``(``:326-377``)  4x4 box-averaged bulk pressures,
  radius by linear interpolation of the density crossing ``rho = (rho_l+rho_v)/2``
  on the mid row, surface tension ``gamma = dp * r``.

Two things are *added* here that the reference lacks (both are proven defects of
the reference, see ``xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION``):

1. **Guarded radius estimator.**  The reference declares ``double rad;``
   (``:358``) without initialisation and only assigns it if the scan loop breaks;
   for under-resolved / failed states it prints an undefined ``gamma``.  Here a
   missing crossing raises :class:`RadiusEstimationError`.
2. **Sign of the interaction pressure.**  The library force convention is
   ``F = G * cs^2 * psi * sum_i w_i c_i psi(x+c_i)`` with ``G > 0`` **attractive**
   (see :func:`tensorlbm.multiphase.sc_force`), which is the negative of the
   reference convention (``g < 0`` attractive).  Hence the interaction pressure
   is ``-(G cs^2/2) psi^2``, i.e. attraction *lowers* the pressure:

       p(rho) = cs^2 rho - (G cs^2 / 2) psi(rho)^2 ,     cs^2 = 1/3.

   The ``+G/6 psi^2`` sign found in ``tensorlbm.wetting.measure_young_laplace``
   belongs to the *repulsive* branch and inflates the Laplace pressure by roughly
   two orders of magnitude for the default parameters (see the xval report).

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815; Shan & Chen (1994) Phys. Rev. E 49, 2941.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..d2q9 import CS2
from . import psi_exp

__all__ = [
    "RadiusEstimationError",
    "sc_pressure",
    "interface_radius",
    "laplace_pressure_calibration",
    "fit_laplace",
    "sc_maxwell_coexistence",
]


class RadiusEstimationError(RuntimeError):
    """Raised when the droplet radius cannot be located from a density profile."""


# ---------------------------------------------------------------------------
# Lattice pressure
# ---------------------------------------------------------------------------

def sc_pressure(rho: torch.Tensor, G: float = 5.0, rho0: float = 1.0) -> torch.Tensor:
    """Shan-Chen lattice pressure for the library force convention (G>0 attractive).

    ``p = cs^2 rho - (G cs^2 / 2) psi^2``; equivalently ``rho/3 - G psi^2/6``.

    Derived from the force used by :func:`tensorlbm.multiphase.sc_force`:
    ``F = G cs^2 psi grad(psi) = -grad p_int`` with ``p_int = -(G cs^2/2) psi^2``.
    """
    psi = psi_exp(rho, rho0)
    return CS2 * rho - 0.5 * G * CS2 * psi * psi


# ---------------------------------------------------------------------------
# Radius from a density profile (reference recipe, guarded)
# ---------------------------------------------------------------------------

def interface_radius(
    rho: torch.Tensor,
    rho_av: float,
    center: tuple[float, float] | None = None,
) -> float:
    """Locate the droplet interface on the mid row by linear interpolation.

    Reference: ``chapter9/shanchen.cpp:357-367``.  The scan starts at the domain
    centre and walks towards ``x = 0`` until the density first exceeds
    ``rho_av``; the sub-cell position is linearly interpolated between the two
    bracketing nodes.

    Raises
    ------
    RadiusEstimationError
        If no crossing is found (the reference code silently returns an
        uninitialised radius in this case).
    """
    ny, nx = rho.shape
    if center is None:
        cx, cy = nx / 2.0, ny / 2.0
    else:
        cx, cy = center
    row = int(round(cy)) % ny
    col0 = int(round(cx))
    prof = rho[row].detach().cpu().numpy()

    # Reference scan direction: outside-in on the mid row (chapter9/shanchen.cpp:358-367).
    for x in range(0, col0):
        if prof[x + 1] > rho_av >= prof[x]:
            d_rho = prof[x + 1] - prof[x]
            if d_rho <= 0.0:
                continue
            frac = (rho_av - prof[x]) / d_rho      # from node x towards node x+1
            radius = (col0 - x) - frac
            if not math.isfinite(radius) or radius <= 0.0:
                continue
            return float(radius)
    raise RadiusEstimationError(
        "no density crossing of rho_av={:.6g} on row {} between x=0 and x={}: "
        "state is under-resolved or not phase separated".format(rho_av, row, col0)
    )


# ---------------------------------------------------------------------------
# Full calibration (reference calculateDeltaPressure, guarded)
# ---------------------------------------------------------------------------

def laplace_pressure_calibration(
    rho: torch.Tensor,
    G: float = 5.0,
    box: int = 4,
    center: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Measure ``dp``, radius and ``gamma = dp * r`` exactly as the reference does.

    Gas pressure: ``box x box`` average in the corner ``[0, box)^2``.
    Liquid pressure: ``box x box`` average around the domain centre.
    """
    rho_f = rho.detach().to(torch.float64)
    p = sc_pressure(rho_f, G)
    ny, nx = rho_f.shape
    if box > min(ny, nx) // 4:
        raise ValueError(f"box={box} too large for grid {ny}x{nx}")
    cx, cy = (nx // 2, ny // 2) if center is None else (int(center[0]), int(center[1]))

    gas = rho_f[:box, :box]
    liq = rho_f[cy - box // 2: cy + box // 2, cx - box // 2: cx + box // 2]
    rho_v = float(gas.mean())
    rho_l = float(liq.mean())
    p_v = float(p[:box, :box].mean())
    p_l = float(p[cy - box // 2: cy + box // 2, cx - box // 2: cx + box // 2].mean())
    dp = p_l - p_v

    rho_av = 0.5 * (rho_l + rho_v)
    radius = interface_radius(rho_f, rho_av, center=(cx, cy))
    return {
        "G": float(G),
        "rho_l": rho_l,
        "rho_v": rho_v,
        "rho_av": rho_av,
        "p_l": p_l,
        "p_v": p_v,
        "dp": dp,
        "radius": radius,
        "gamma": dp * radius,
    }


# ---------------------------------------------------------------------------
# Laplace-law fit (dp vs 1/R through the origin)
# ---------------------------------------------------------------------------

def fit_laplace(dps: list[float], radii: list[float]) -> dict[str, float]:
    """Least-squares ``dp = sigma / R`` through the origin; returns sigma and r^2."""
    if len(dps) != len(radii) or len(dps) < 2:
        raise ValueError("need at least two (dp, radius) pairs of equal length")
    inv_r = np.asarray([1.0 / r for r in radii], dtype=float)
    dp = np.asarray(dps, dtype=float)
    sigma = float((inv_r * dp).sum() / (inv_r * inv_r).sum())
    pred = sigma * inv_r
    ss_res = float(((dp - pred) ** 2).sum())
    ss_tot = float(((dp - dp.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    rel = np.abs(dp - pred) / np.maximum(np.abs(pred), 1e-300)
    return {
        "sigma": sigma,
        "r2": r2,
        "max_rel_err": float(rel.max()),
        "n_points": len(dps),
    }


# ---------------------------------------------------------------------------
# Analytic reference: Maxwell construction on the SC equation of state
# ---------------------------------------------------------------------------

def sc_maxwell_coexistence(
    G: float = 5.0,
    rho0: float = 1.0,
    rho_max: float = 8.0,
    n: int = 40001,
) -> dict[str, float]:
    """Maxwell (equal pressure + equal chemical potential) coexistence of the SC EOS.

    ``p(rho) = rho/3 - (G/6) psi(rho)^2``, ``mu(rho) = int (1/rho) dp/drho drho``.
    The pressure is non-monotonic in the two-phase region, so for a trial pressure
    ``p*`` three roots exist; the coexistence pressure is the one where the
    chemical potentials of the outer two roots coincide.

    Returns ``{'rho_l', 'rho_v', 'p': ...}``.
    """
    grid = np.linspace(1e-4, rho_max, n)
    psi = rho0 * (1.0 - np.exp(-grid / rho0))
    p = grid / 3.0 - (G / 6.0) * psi * psi
    dpdrho = 1.0 / 3.0 - (G / 3.0) * psi * np.exp(-grid / rho0)
    # chemical potential up to a constant (mu(1) = 0)
    integrand = dpdrho / grid
    mu_cum = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))))
    mu = mu_cum - np.interp(1.0, grid, mu_cum)

    # spinodal range where p(rho) is non-monotonic
    sign_change = np.where(np.diff(np.sign(dpdrho)) != 0)[0]
    if sign_change.size == 0:
        raise ValueError(f"G={G} has no spinodal: no phase separation")
    p_candidates = p[sign_change]
    p_lo, p_hi = float(min(p_candidates)), float(max(p_candidates))

    def roots(pt):
        f = p - pt
        idx = np.where(np.diff(np.sign(f)) != 0)[0]
        if idx.size < 3:
            return None
        return [float(grid[i] + (grid[i + 1] - grid[i]) * (-f[i]) / (f[i + 1] - f[i])) for i in idx]

    def h(pt):
        r = roots(pt)
        if r is None or len(r) < 3:
            return float("nan")
        return float(np.interp(r[0], grid, mu) - np.interp(r[-1], grid, mu))

    # bracket the coexistence pressure by scanning (the spinodal endpoints
    # themselves are double roots and carry no usable chemical potential)
    scan = np.linspace(p_lo + 1e-9, p_hi - 1e-9, 97)
    hs = [h(float(q)) for q in scan]
    bracket = None
    for i in range(len(scan) - 1):
        if math.isfinite(hs[i]) and math.isfinite(hs[i + 1]) and hs[i] * hs[i + 1] < 0.0:
            bracket = (float(scan[i]), float(hs[i]), float(scan[i + 1]), float(hs[i + 1]))
            break
    if bracket is None:
        raise ValueError("Maxwell bracket failed (no sign change of chemical potential)")
    lo, h_lo, hi, h_hi = bracket
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        h_mid = h(mid)
        if not math.isfinite(h_mid):
            raise ValueError("Maxwell iteration lost the roots")
        if h_lo * h_mid <= 0.0:
            hi, h_hi = mid, h_mid
        else:
            lo, h_lo = mid, h_mid
        if abs(hi - lo) < 1e-14:
            break
    pt = 0.5 * (lo + hi)
    r = roots(pt)
    return {"rho_v": float(r[0]), "rho_l": float(r[-1]), "p": float(pt), "G": float(G)}
