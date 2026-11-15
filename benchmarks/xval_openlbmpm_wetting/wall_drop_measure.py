#!/usr/bin/env python3
"""Wall-attached (cluster-safe) contact-angle measurement for the sessile drop.

Why this module exists
----------------------
The cross-validation in ``docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE.md``
(round 1) declared the solver-level sessile-drop comparison an honest-FAIL
because all three shape estimators of :mod:`tensorlbm.cg_contact_angle`
returned ``180 deg`` / ``nan`` / non-physical values.  The failure is *not* in
the wetting boundary condition: the host solver condenses a small amount of the
wetting phase onto every solid wall, and that trailing blob lies in the same
column as the drop.  A global contour/circle fit therefore mixes the drop with
the blob (apex height jumps to the top wall, Kasa fit degenerates to a straight
line -> ``180 deg``).

The measurement below is therefore **component restricted**:

1. label the ``phi > level`` region of the fluid domain (8-connectivity);
2. keep only the component that touches the bottom wall (the drop);
3. report every other positive-phase component as ``blob_*`` diagnostics -- the
   blob is a genuine solver artefact and must be quantified, not silently
   discarded;
4. footprint ``W`` from the level-set crossing on the first fluid row, apex
   ``H`` from the topmost level-set crossing *of the drop component*, and the
   spherical-cap identity

       theta = 2 atan(2H / W)        (exact for a cap: H/(W/2) = tan(theta/2))

5. an independent Kasa circle fit on the drop contour, with the contact angle
   taken as ``arccos((y_wall - y_c) / R_fit)`` -- kept because it is a
   *different* geometric statement than (4), so the two can falsify each other.

Everything returns ``nan`` instead of raising, so a failed arm stays visible in
the artefacts.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def _crossings(vals: np.ndarray, level: float = 0.0) -> np.ndarray:
    """Linearly interpolated positions where ``vals`` crosses ``level``."""
    d = np.asarray(vals, dtype=np.float64) - level
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if idx.size == 0:
        return np.empty(0)
    x0 = d[idx]
    x1 = d[idx + 1]
    return idx + x0 / (x0 - x1)


def drop_component(phi: np.ndarray, solid: np.ndarray, level: float = 0.0,
                   wall_row: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the wall-attached positive-phase component and the stray blobs."""
    mask = (phi > level) & (~solid)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros_like(mask), []
    wall_labels = set(np.unique(lab[wall_row][lab[wall_row] > 0]).tolist())
    if not wall_labels:
        return np.zeros_like(mask), [lab == k for k in range(1, n + 1)]
    sizes = {k: int((lab == k).sum()) for k in wall_labels}
    main = max(sizes, key=sizes.get)
    blobs = [lab == k for k in range(1, n + 1) if k != main]
    return lab == main, blobs


def measure_wall_drop(phi: np.ndarray, solid: np.ndarray, y_wall: float = 0.5,
                      level: float = 0.0, wall_row: int = 1) -> dict:
    """Contact angle of the wall-attached drop: cap identity + circle fit."""
    drop, blobs = drop_component(phi, solid, level=level, wall_row=wall_row)
    out: dict = {
        "W": float("nan"),
        "H": float("nan"),
        "theta_cap_deg": float("nan"),
        "theta_circle_deg": float("nan"),
        "x_centre": float("nan"),
        "drop_area": float(drop.sum()),
        "blob_area": float(sum(b.sum() for b in blobs)),
        "n_blobs": len(blobs),
        "blob_centroid_y": float("nan"),
        "blob_phi_max": float("nan"),
    }
    if not drop.any():
        return out

    if blobs:
        ys, xs = np.nonzero(np.logical_or.reduce(blobs))
        out["blob_centroid_y"] = float(ys.mean())
        out["blob_phi_max"] = float(max(float(phi[b].max()) for b in blobs))

    # --- footprint on the first fluid row (sign crossing of the drop mask) ----
    row = np.where(drop[wall_row], phi[wall_row], -np.abs(phi[wall_row]) - 1.0)
    xs = _crossings(row, level)
    if xs.size >= 2:
        xl, xr = float(xs.min()), float(xs.max())
        out["W"] = xr - xl
        out["x_centre"] = 0.5 * (xl + xr)

    # --- apex of the drop component --------------------------------------
    cols = np.nonzero(drop.any(axis=0))[0]
    top = -np.inf
    per_col = {}
    for j in cols:
        col = np.where(drop[:, j], phi[:, j], -np.abs(phi[:, j]) - 1.0)
        ys = _crossings(col, level)
        if ys.size:
            per_col[int(j)] = float(ys.max())
            top = max(top, float(ys.max()))
    if math.isfinite(top):
        out["H"] = top - y_wall

    W, H = out["W"], out["H"]
    if math.isfinite(W) and math.isfinite(H) and W > 0 and H > 0:
        out["theta_cap_deg"] = 2.0 * math.degrees(math.atan2(2.0 * H, W))

    # --- independent Kasa circle fit on the drop contour -------------------
    try:
        ys, xs = np.nonzero((np.abs(phi - level) < 0.5) & drop)
        if xs.size >= 8:
            x = xs.astype(np.float64)
            y = ys.astype(np.float64)
            A = np.c_[2.0 * x, 2.0 * y, np.ones_like(x)]
            b = x * x + y * y
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            xc, yc, c0 = sol
            R = math.sqrt(max(c0 + xc * xc + yc * yc, 0.0))
            if R > 0:
                out["theta_circle_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, (y_wall - yc) / R))))
                out["circle"] = {"x_centre": float(xc), "y_centre": float(yc), "radius": float(R)}
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        out["circle_error"] = repr(exc)
    return out