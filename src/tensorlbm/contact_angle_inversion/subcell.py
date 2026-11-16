"""Sub-cell contact-angle measurement for Shan-Chen sessile drops.

Why this exists
---------------
:func:`tensorlbm.sc_surface_tension.contact_angle_from_mask` measures the cap
from the **binarised** field: per row it takes ``max|x - x_c|`` over the cells
whose density exceeds the mid-density.  That makes the half width an *integer*
multiple of half a cell, so the fitted angle comes out in discrete steps.  On
the 64x32 / R=8 calibration grid of ``contact_angle_inversion`` the steps are
6-12 degrees:

===========================  =====================  ============
``f_w``                      half widths (cells)    measured
===========================  =====================  ============
0.60                         7 7 7 7 6 5 4          95.156 deg
0.62                         7 7 7 7 6 5 4          95.156 deg
0.64                         7 7 7 7 6 5 4          95.156 deg
===========================  =====================  ============

Three different wall potentials produced the *same* mask to within four cells
(only the apex row narrowed, and that row is excluded from the fit).  The
measured curve is therefore a staircase, not a smooth function of ``f_w``: the
"non-monotone calibration curve" that motivated the multi-branch inversion of
m22445 is one of its plateaus, not a physical fold.

This module measures the same interface with **sub-cell resolution**: per row
it locates the outermost ``rho = mid`` crossing by linear interpolation between
the last liquid cell and the first vapour cell, then fits a circle to those
sub-cell half widths.  The angle resolution then follows the interface
steepness instead of the lattice pitch.

The estimator is validated against analytic caps in
``tensorlbm.contact_angle_inversion.synthetic`` (ground truth known exactly),
where the mask estimator carries a +3.0 deg bias at R=8 / theta=90 and this one
does not.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

__all__ = ["contact_angle_subcell"]


def _crossings(row: torch.Tensor, fluid: torch.Tensor, mid: float) -> tuple | None:
    """Sub-cell left/right ``rho = mid`` crossings of one row, or ``None``."""
    liq = (row > mid) & fluid
    if not bool(liq.any()):
        return None
    idx = torch.nonzero(liq).squeeze(1)
    xl, xr = int(idx.min()), int(idx.max())
    nx = row.numel()

    def interp(x_in: int, x_out: int) -> float:
        """crossing position between a liquid cell and its vapour neighbour."""
        r_in, r_out = float(row[x_in]), float(row[x_out])
        if abs(r_in - r_out) < 1e-12:
            return x_in + 0.5
        t = (r_in - mid) / (r_in - r_out)
        return x_in + min(max(t, 0.0), 1.0)

    left = interp(xl, xl - 1) if xl > 0 and bool(fluid[xl - 1]) else xl + 0.5
    right = interp(xr, xr + 1) if xr < nx - 1 and bool(fluid[xr + 1]) else xr + 0.5
    return left, right


def contact_angle_subcell(
    rho: torch.Tensor,
    solid: torch.Tensor,
    mid: float,
    wall_offset: float = 0.5,
    min_rows: int = 6,
    max_half_width_frac: float = 0.40,
) -> Dict[str, object]:
    """Contact angle from a sub-cell circular-cap fit.

    Same geometric model as ``contact_angle_from_mask`` (``w^2 + y^2 = A + B y``
    is linear in ``y`` for a circular cap, fitted by least squares), but with

    * half widths taken from the sub-cell ``rho = mid`` crossings,
    * row coordinates at cell centres (``y + 0.5``),
    * the width measured from the *pair* of crossings instead of ``max|x-x_c|``.

    Args:
        rho: Density field ``(ny, nx)``; row 0 is one wall.
        solid: Boolean solid mask, same shape.
        mid: Density that defines the interface (``0.5 * (rho_l + rho_v)``).
        wall_offset: Distance from the last solid row centre to the wall.
        min_rows: Minimum number of rows for a fit.
        max_half_width_frac: Rows whose half width exceeds this fraction of
            ``nx`` belong to the precursor film, not the cap, and are dropped.

    Returns:
        Dict with ``theta_deg``, ``cos_theta``, ``R_fit``, ``z_c``, ``apex``,
        ``y_wall``, ``n_rows``, ``fit_rms``, ``half_widths`` and ``status``.
    """
    ny, nx = rho.shape
    fluid = ~solid
    nan = float("nan")
    out: Dict[str, object] = {
        "theta_deg": nan, "cos_theta": nan, "R_fit": nan, "z_c": nan,
        "apex": nan, "y_wall": nan, "n_rows": 0, "fit_rms": nan,
        "half_widths": [], "status": "no-liquid",
    }

    ys_liq = torch.nonzero(((rho > mid) & fluid).any(dim=1)).squeeze(1)
    if ys_liq.numel() == 0:
        return out
    apex = int(ys_liq.max())
    y_c0 = float(ys_liq.float().mean())
    solid_rows = torch.arange(ny).view(ny, 1).expand(ny, nx)[solid & (
        torch.arange(ny).view(ny, 1).expand(ny, nx) < y_c0)]
    y_wall = float(solid_rows.max()) if solid_rows.numel() else -0.5
    out["apex"] = float(apex)
    out["y_wall"] = y_wall

    if apex - (y_wall + wall_offset) < 3.0:
        out["status"] = "no-cap-height"
        return out

    ys: List[float] = []
    halves: List[float] = []
    centres: List[float] = []
    for y in range(int(y_wall + wall_offset + 1), apex):
        c = _crossings(rho[y], fluid[y], mid)
        if c is None:
            continue
        left, right = c
        half = 0.5 * (right - left)
        if half > max_half_width_frac * nx:
            continue          # precursor-film row, not part of the cap
        ys.append(y + 0.5)
        halves.append(half)
        centres.append(0.5 * (left + right))

    if len(ys) < min_rows:
        out["status"] = "insufficient-rows"
        out["n_rows"] = len(ys)
        return out

    yy = torch.tensor(ys, dtype=torch.float64)
    ww = torch.tensor(halves, dtype=torch.float64)
    v = ww * ww + yy * yy
    A = torch.stack([torch.ones_like(yy), yy], dim=1)
    sol = torch.linalg.lstsq(A, v.reshape(-1, 1)).solution.squeeze(1)
    a0, b0 = float(sol[0]), float(sol[1])
    yc = 0.5 * b0
    r2 = a0 + yc * yc
    if r2 <= 0:
        out["status"] = "degenerate-fit"
        return out
    R = math.sqrt(r2)
    z_c = yc - (y_wall + wall_offset)
    cos_t = max(-1.0, min(1.0, -z_c / R))
    resid = float(((A @ sol - v) ** 2).mean() ** 0.5)
    out.update({
        "theta_deg": math.degrees(math.acos(cos_t)),
        "cos_theta": cos_t,
        "R_fit": R,
        "z_c": z_c,
        "n_rows": len(ys),
        "fit_rms": resid,
        "x_axis": sum(centres) / len(centres),
        "half_widths": halves,
        "status": "ok",
    })
    return out