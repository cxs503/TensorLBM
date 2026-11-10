"""Quantitative diagnostics behind the invariance certificate (P493-N02).

Three measurements that the acceptance criteria of :mod:`tensorlbm.invariance`
rely on but that must **not** be conflated with the flow solution itself:

``density_variation``
    ``max|rho - rho0|`` over the fluid.  A channel that is driven by an imposed
    velocity at both ends needs a streamwise pressure gradient; through the
    lattice equation of state that gradient becomes a density gradient
    ``Delta rho = 3 Delta p``, and the weakly compressible continuity constraint
    ``div(rho u) = 0`` then forces a *velocity* deviation of order
    ``Delta rho`` that has nothing to do with the wall treatment under test.
    The gate therefore requires a *zero pressure gradient* test problem (body
    force balancing the wall shear), reported by this number.
``wall_slip``
    The axial velocity extrapolated to the wall from the interior.  For a
    lattice-Boltzmann bounce-back family the no-slip location is only known to
    within a sub-cell offset, which appears as a slip velocity
    ``u_s = u_num(x_wall)`` and is the leading error of the *velocity* field.
``effective_wall_offset``
    The slip converted into a wall-position displacement
    ``delta = -u_s / (du/dn)|_wall`` (in lattice units), i.e. the quantity that
    has to vanish faster than ``O(h)`` for a second-order velocity field.

All functions are pure measurement: they take fields and geometry and return
plain floats, so they can be serialised into the acceptance JSON without any
hand copying.
"""

from __future__ import annotations

import math

import numpy as np


def normal_offset(geom) -> np.ndarray:
    """Signed distance along the channel normal ``n̂·(r−c)`` (0 on the axis)."""
    x = (np.arange(geom.nx) + 0.5)[None, :]
    y = (np.arange(geom.ny) + 0.5)[:, None]
    cx, cy = geom.meta["center"]
    nxn, nyn = geom.normal
    return nxn * (x - cx) + nyn * (y - cy)


def density_variation(rho: np.ndarray, fluid: np.ndarray, rho0: float = 1.0) -> dict:
    """Max/mean deviation of the density from its rest value (compressibility gate)."""
    r = rho[fluid]
    return {
        "max_abs": float(np.max(np.abs(r - rho0))),
        "rms": float(np.sqrt(np.mean((r - rho0) ** 2))),
        "min": float(r.min()),
        "max": float(r.max()),
    }


def wall_slip(
    geom,
    ux: np.ndarray,
    uy: np.ndarray,
    ux_ex: np.ndarray | None = None,
    uy_ex: np.ndarray | None = None,
    *,
    band: float = 1.5,
    trim: float = 0.25,
) -> dict:
    """Axial velocity extrapolated linearly to ``sdf = 0`` for each wall.

    Nodes whose distance to the wall is below ``band`` (and that are not in the
    trimmed inflow/outflow region) are used to fit
    ``u_axial(x) - u_axial^exact(x) = a + b*sdf`` per wall (the two walls are
    separated by the sign of the normal offset); the intercept ``a`` is the
    *wall slip error*, i.e. the velocity that the discrete solution erroneously
    keeps at the wall.  Extrapolating the error instead of the field removes the
    O(sdf^2) curvature of the parabola from the fit.
    """
    axial = ux * geom.axis[0] + uy * geom.axis[1]
    if ux_ex is not None:
        axial = axial - (ux_ex * geom.axis[0] + uy_ex * geom.axis[1])
    d = normal_offset(geom)
    mask = geom.fluid.copy()
    k = int(trim * geom.nx)
    if k > 0:
        mask[:, :k] = False
        mask[:, -k:] = False
    out = {}
    for name, side in (("upper", d < 0.0), ("lower", d > 0.0)):
        m = mask & side & (geom.sdf <= band)
        if m.sum() < 4:
            out[name] = {"slip": float("nan"), "slope": float("nan"), "n": int(m.sum())}
            continue
        A = np.stack([np.ones(m.sum()), geom.sdf[m]], axis=1)
        coef, *_ = np.linalg.lstsq(A, axial[m], rcond=None)
        out[name] = {"slip": float(coef[0]), "slope": float(coef[1]), "n": int(m.sum())}
    out["mean_slip"] = float(np.nanmean([out["upper"]["slip"], out["lower"]["slip"]]))
    return out


def effective_wall_offset(geom, slip: float, u_mean: float, nu: float) -> float:
    """Wall displacement ``delta = -u_s / (du/dn)|_wall`` in lattice units.

    The developed plane-Poiseuille profile ``u = G y'(H-y')/(2 nu)`` driven by
    the body force ``G`` has the wall shear ``du/dn = G H/(2 nu)``; a slip
    ``u_s`` is therefore equivalent to shifting the wall by ``delta``.
    """
    width = float(geom.meta["width"])
    G = 12.0 * nu * u_mean / (width * width)
    dudn = G * width / (2.0 * nu)
    return float(-slip / dudn)


def band_errors(geom, ux, uy, ux_ex, uy_ex, edges=(0.0, 1.0, 2.0, 4.0, 1e9),
                trim: float = 0.25) -> list[dict]:
    """RMS velocity error in wall-distance bands (where does the error live?)."""
    err = np.hypot(ux - ux_ex, uy - uy_ex)
    mask = geom.fluid.copy()
    k = int(trim * geom.nx)
    if k > 0:
        mask[:, :k] = False
        mask[:, -k:] = False
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & (geom.sdf >= lo) & (geom.sdf < hi)
        out.append({
            "sdf_lo": float(lo),
            "sdf_hi": float(hi) if hi < 1e8 else None,
            "n": int(m.sum()),
            "rms": float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan"),
        })
    return out


def exponential_decay_rate(times, values) -> float:
    """``-d ln(values)/dt`` from a least-squares fit (values must be positive)."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    slope, _ = np.polyfit(t, np.log(v), 1)
    return float(-slope)