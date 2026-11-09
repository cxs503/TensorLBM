"""Verification ledgers: Ghia comparison, conservation budgets, cross-solver error.

Everything operates on *cell-centred* fields with the same layout
(``array[j, i]``, ``j`` = y index, cell centres at ``((i+1/2)/n, (j+1/2)/n)``),
which is what all three reference solvers return, so the three legs of the
benchmark are compared with an identical measurement operator (no interpolation
bias favouring one solver).

Energy ledger
-------------
For a steady, incompressible flow in a closed box with rigid walls the global
kinetic-energy budget is ``P = eps`` where

* ``P  = u_lid * F_lid``  with ``F_lid = nu * int (du/dy)|_lid dx`` (power fed in
  by the moving wall), and
* ``eps = nu * int 2 S_ij S_ij dV = nu * int [2(u_x^2 + v_y^2) + (u_y + v_x)^2] dV``
  (viscous dissipation).

Both integrals are evaluated with second-order one-sided stencils at the walls
(ghost values obtained by linear extrapolation from the wall velocity), so the
two numbers are *independent* measurements of the same physics; their relative
difference is a discretisation-quality metric, not an identity imposed by
construction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GHIA_TABLES",
    "centreline_profiles",
    "compare_ghia",
    "divergence_ledger",
    "energy_ledger",
    "relative_l2",
    "ghia_reference",
]


def ghia_reference(re: int) -> dict | None:
    """Ghia, Ghia & Shin (1982) centreline tables for Re = 100/400/1000.

    Imported from :mod:`tensorlbm.lid_driven_cavity` (single source of truth; the
    tables are not duplicated here).
    """
    from ..lid_driven_cavity import GHIA_RE100, GHIA_RE400, GHIA_RE1000

    return {100: GHIA_RE100, 400: GHIA_RE400, 1000: GHIA_RE1000}.get(int(re))


def _extend(field: np.ndarray, walls: dict) -> np.ndarray:
    """Ghost-extend a cell-centred field using the wall values.

    Ghost cell at ``-h/2`` is ``2*w_wall - w_1`` (linear extrapolation about the
    wall), which makes central differences second order at boundary cells.
    """
    out = np.empty((field.shape[0] + 2, field.shape[1] + 2))
    out[1:-1, 1:-1] = field
    out[0, 1:-1] = 2.0 * walls["bottom"] - field[0, :]
    out[-1, 1:-1] = 2.0 * walls["top"] - field[-1, :]
    out[1:-1, 0] = 2.0 * walls["left"] - field[:, 0]
    out[1:-1, -1] = 2.0 * walls["right"] - field[:, -1]
    out[0, 0] = walls["bottom"][0]
    out[0, -1] = walls["bottom"][-1]
    out[-1, 0] = walls["top"][0]
    out[-1, -1] = walls["top"][-1]
    return out


def _wall_fields(n: int, u_lid: float, u_lid_field=None):
    """Wall velocity (u, v) for each side of the lid-driven cavity."""
    zeros = np.zeros(n)
    top_u = np.full(n, u_lid) if u_lid_field is None else u_lid_field
    return (
        {"bottom": zeros, "top": top_u, "left": zeros, "right": zeros},
        {"bottom": zeros, "top": zeros, "left": zeros, "right": zeros},
    )


def centreline_profiles(u: np.ndarray, v: np.ndarray):
    """Centreline profiles on the cell-centre grid (no interpolation needed)."""
    n = u.shape[0]
    x = (np.arange(n) + 0.5) / n
    jmid = None
    # x = 0.5 line: u along y (grid nodes closest to the centre column)
    col_lo = n // 2 - 1
    col_hi = n // 2
    u_centre = 0.5 * (u[:, col_lo] + u[:, col_hi])
    row_lo = n // 2 - 1
    row_hi = n // 2
    v_centre = 0.5 * (v[row_lo, :] + v[row_hi, :])
    return {"y": x, "u_centre": u_centre, "x": x, "v_centre": v_centre, "jmid": jmid, "j": jmid}


def compare_ghia(u: np.ndarray, v: np.ndarray, u_lid: float, ref: dict) -> dict:
    """Max / RMS deviation of the centreline profiles from Ghia et al. (1982).

    Deviations are reported as percentages of the lid speed, which is how the
    acceptance criterion (``< 3 %``) is defined.
    """
    n = u.shape[0]
    prof = centreline_profiles(u, v)
    y = prof["y"]
    x = prof["x"]
    u_n = prof["u_centre"] / u_lid
    v_n = prof["v_centre"] / u_lid
    gy = np.asarray(ref["y"], dtype=float)
    gu = np.asarray(ref["u"], dtype=float)
    gx = np.asarray(ref["x"], dtype=float)
    gv = np.asarray(ref["v"], dtype=float)
    # interpolate the computed profile onto the reference stations (Ghia's
    # stations lie inside the open cavity; clamp to the available range)
    u_interp = np.interp(gy, y, u_n)
    v_interp = np.interp(gx, x, v_n)
    du = u_interp - gu
    dv = v_interp - gv
    return {
        "n": n,
        "u_max_dev_pct": float(np.abs(du).max() * 100.0),
        "u_rms_dev_pct": float(np.sqrt(np.mean(du**2)) * 100.0),
        "v_max_dev_pct": float(np.abs(dv).max() * 100.0),
        "v_rms_dev_pct": float(np.sqrt(np.mean(dv**2)) * 100.0),
        "y_ref": gy.tolist(),
        "u_lbm": u_interp.tolist(),
        "u_ref": gu.tolist(),
        "x_ref": gx.tolist(),
        "v_lbm": v_interp.tolist(),
        "v_ref": gv.tolist(),
    }


def divergence_ledger(u: np.ndarray, v: np.ndarray, h: float, u_lid: float = 1.0) -> dict:
    """Discrete mass-conservation ledger.

    * ``flux_integral``  -- ``|sum_cells (div u) h^2|``, the signed global mass
      imbalance (equals ``|oint u.n dA|`` for a consistent discrete divergence);
    * ``max_abs_div`` / ``l2_div`` -- local divergence norm scaled by ``u_lid/h``,
      the meaningful metric for a weakly compressible solver (LBM) where the
      *signed* integral is trivially small but the pointwise field is only
      divergence-free to ``O(Ma^2)``.
    """
    n = u.shape[0]
    walls_u, walls_v = _wall_fields(n, u_lid)
    ue = _extend(u, walls_u)
    ve = _extend(v, walls_v)
    dux = (ue[1:-1, 2:] - ue[1:-1, :-2]) / (2.0 * h)
    dvy = (ve[2:, 1:-1] - ve[:-2, 1:-1]) / (2.0 * h)
    div = dux + dvy
    scale = u_lid / h
    return {
        "flux_integral": float(abs(div.sum() * h * h)),
        "max_abs_div": float(np.abs(div).max()),
        "max_abs_div_scaled": float(np.abs(div).max() / scale),
        "l2_div_scaled": float(np.sqrt(np.mean(div**2)) / scale),
    }


def energy_ledger(u: np.ndarray, v: np.ndarray, h: float, nu: float, u_lid: float) -> dict:
    """Steady-state energy budget ``P`` (lid work) vs ``eps`` (dissipation)."""
    n = u.shape[0]
    walls_u, walls_v = _wall_fields(n, u_lid)
    ue = _extend(u, walls_u)
    ve = _extend(v, walls_v)
    dux = (ue[1:-1, 2:] - ue[1:-1, :-2]) / (2.0 * h)
    duy = (ue[2:, 1:-1] - ue[:-2, 1:-1]) / (2.0 * h)
    dvx = (ve[1:-1, 2:] - ve[1:-1, :-2]) / (2.0 * h)
    dvy = (ve[2:, 1:-1] - ve[:-2, 1:-1]) / (2.0 * h)
    two_s2 = 2.0 * (dux**2 + dvy**2) + (duy + dvx) ** 2
    eps = float(nu * two_s2.sum() * h * h)
    # viscous shear on the lid: tau_xy = nu (du/dy + dv/dx), v = 0 on the lid
    # second-order one-sided stencil (the wall value u_lid is known exactly)
    if n >= 2:
        du_dy_lid = (3.0 * u_lid - 4.0 * u[-1, :] + u[-2, :]) / (2.0 * h)
    else:  # pragma: no cover
        du_dy_lid = (u_lid - u[-1, :]) / (0.5 * h)
    force_lid = float(nu * du_dy_lid.sum() * h)
    power = float(u_lid * force_lid)
    denom = max(abs(power), abs(eps), 1e-30)
    return {
        "power_in": power,
        "dissipation": eps,
        "rel_error_pct": float(abs(power - eps) / denom * 100.0),
        "lid_force": force_lid,
        "dissipation_flux_coeff": float(2.0 * nu * force_lid / max(u_lid, 1e-30)) if u_lid else 0.0,
    }


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 difference of two fields, normalised by the ``b`` RMS norm."""
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den > 0 else num