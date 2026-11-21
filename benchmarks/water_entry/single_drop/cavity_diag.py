"""Read-only diagnosis of the crater-depth deficit: drop penetration vs cavity closure.

The probe's ``depth_over_D`` is the depth of the deepest *headspace-connected*
gas-to-liquid floor inside a 1.5 D radius (see ``run.open_depth``).  A deficit
against the Santini case-3 experiment can come from two physically different
causes:

``under-penetration``
    the drop-derived liquid never reaches the depth the experiment shows, so the
    cavity it drags down stops there as well;

``over-closure``
    the cavity is driven deep enough but its gas column is pinched off from the
    headspace, so the deepest part of the cavity leaves the measurement window.

Everything in this module is a *read-only* measurement of the fill field and of
the velocity field.  No solver state is written, so switching the diagnosis on
cannot change the trajectory: the probe's own ``history`` stays bit-identical.

Conventions
-----------
``field``
    liquid volume fraction on the interior grid, shape ``(nz-2, side, side)``;
    a cell counts as liquid when ``field >= 0.5`` (the same rule as the
    benchmark's ``open_depth``), a cell counts as gas when ``field < 0.5``.
``surf``
    sub-cell position of the initial free surface in field coordinates.
``n``
    cells per drop diameter; depths are reported in diameters.

The grid axis 0 points **up** (index 0 is the pool floor, the last index is the
headspace), exactly as in ``run.open_depth``: a depth below the initial surface
is ``surf - index`` and "below the surface" means a *smaller* index.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "DropMaterialMarkers",
    "cavity_diagnostics",
    "closure_point",
    "headspace_mask",
]


def headspace_mask(gas: np.ndarray) -> np.ndarray:
    """Gas cells connected to the top of the domain (the headspace)."""
    seeds = np.zeros_like(gas)
    seeds[-1] = gas[-1]
    return ndimage.binary_propagation(seeds, mask=gas)


def _floor_crossings(field, gas, above, roi, surf):
    """Sub-cell depth of the deepest gas-to-liquid floor under ``above``.

    ``above`` marks the gas cells that may sit on top of the floor; ``roi`` is the
    measurement cylinder.  Returns ``(depth_cells, floor_position)`` with
    ``depth_cells`` ``None`` when no crossing exists.
    """
    k, j, i = np.where(above[1:] & ~gas[:-1] & roi[None])
    if not len(k):
        return None, None
    below = field[k, j, i]
    upper = field[k + 1, j, i]
    span = below - upper
    span = np.where(np.abs(span) < 1e-12, 1e-12, span)
    crossing = k + (below - 0.5) / span
    floor = float(crossing.min())
    return surf - floor, floor


class DropMaterialMarkers:
    """Passive Lagrangian markers that follow the drop material after contact.

    The markers are seeded from the detached drop cells one step before the
    drop joins the pool and are then advected with the local lattice velocity.
    A marker that stays in liquid (``fill >= 0.5``) still carries drop material;
    a marker that ends up in gas has been entrained into the cavity and is
    reported separately.
    """

    def __init__(self, positions: np.ndarray):
        self.pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.alive = np.ones(len(self.pos), dtype=bool)
        self.max_live_depth = 0.0

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "DropMaterialMarkers":
        idx = np.argwhere(mask)
        return cls(idx + 0.5)

    def __len__(self) -> int:
        return len(self.pos)

    def advect(self, ux, uy, uz, field, surf=0.0, n=1.0):
        """Move the markers one lattice step and refresh their phase state."""
        if not len(self.pos):
            return
        cells = np.floor(self.pos).astype(np.int64)
        np.clip(cells[:, 0], 0, field.shape[0] - 1, out=cells[:, 0])
        np.clip(cells[:, 1], 0, field.shape[1] - 1, out=cells[:, 1])
        np.clip(cells[:, 2], 0, field.shape[2] - 1, out=cells[:, 2])
        self.pos[:, 0] += uz[cells[:, 0], cells[:, 1], cells[:, 2]]
        self.pos[:, 1] += uy[cells[:, 0], cells[:, 1], cells[:, 2]]
        self.pos[:, 2] += ux[cells[:, 0], cells[:, 1], cells[:, 2]]
        cells = np.floor(self.pos).astype(np.int64)
        np.clip(cells[:, 0], 0, field.shape[0] - 1, out=cells[:, 0])
        np.clip(cells[:, 1], 0, field.shape[1] - 1, out=cells[:, 1])
        np.clip(cells[:, 2], 0, field.shape[2] - 1, out=cells[:, 2])
        self.alive &= field[cells[:, 0], cells[:, 1], cells[:, 2]] >= 0.5
        if self.alive.any():
            live = (surf - self.pos[self.alive, 0]) / n
            self.max_live_depth = max(self.max_live_depth, float(live.max()))

    def depths(self, surf, n):
        """Depth of every marker below the initial surface, in diameters."""
        return (surf - self.pos[:, 0]) / n


def cavity_diagnostics(
    field: np.ndarray,
    surf: float,
    n: int,
    ux=None,
    uy=None,
    uz=None,
    velocity_scale: float = 1.0,
    markers: "DropMaterialMarkers | None" = None,
):
    """Measure the crater depth the way the benchmark does, plus the closure split.

    Parameters
    ----------
    field
        interior liquid volume fraction, shape ``(nz-2, side, side)``.
    surf
        sub-cell position of the initial free surface in field coordinates.
    n
        cells per drop diameter.
    ux, uy, uz
        optional interior velocity fields (lattice units) used for ``u_back``.
    velocity_scale
        metres per second per lattice velocity unit.
    markers
        optional :class:`DropMaterialMarkers` carrying the tracked drop material.

    Returns a dict with the diagnostic quantities; ``None`` marks a quantity
    that is not defined at this instant.
    """
    side = field.shape[1]
    center = (side - 1) / 2
    yy, xx = np.indices(field.shape[1:])
    roi = (yy - center) ** 2 + (xx - center) ** 2 <= (1.5 * n) ** 2

    gas = field < 0.5
    liquid = ~gas
    headspace = headspace_mask(gas)

    # --- cavity depth, benchmark definition (headspace connected) -------------
    d_open, floor_open = _floor_crossings(field, gas, headspace, roi, surf)
    if d_open is not None:
        d_open = max(0.0, d_open)
    # --- cavity depth ignoring headspace connectivity (trapped pockets count) -
    d_all, floor_all = _floor_crossings(field, gas, gas, roi, surf)
    if d_all is not None:
        d_all = max(0.0, d_all)

    below = np.arange(field.shape[0])[:, None, None] < surf
    cavity_gas = gas & below & roi[None]
    # the interior grid starts on the solid pool floor, which is not cavity gas
    cavity_gas[0] = False
    vent_ratio = None
    pockets = None
    body = None
    body_cells = 0
    if cavity_gas.any():
        labels, count = ndimage.label(cavity_gas)
        pockets = int(count)
        # the cavity body is the connected pocket that reaches deepest
        depths = np.where(cavity_gas, np.arange(cavity_gas.shape[0])[:, None, None], 10**6)
        deepest = int(depths.min())
        body = labels == labels[np.unravel_index(int(depths.argmin()), labels.shape)]
        body_cells = int(body.sum())
        vent_ratio = float((headspace & body).sum() / max(body_cells, 1))

    # --- geometric underside of the drop-derived liquid ----------------------
    # A liquid cell whose neighbour below (smaller index) belongs to the cavity
    # body is the underside of a liquid body reaching into the void; the crater
    # floor instead carries gas above it and is therefore not selected here.
    d_ceil = None
    if body is not None and body.any():
        ceiling = liquid & roi[None]
        ceiling[1:] &= body[:-1]
        ceiling[0] = False
        if ceiling.any():
            zz = np.nonzero(ceiling)[0]
            d_ceil = float((surf - zz.min()) / n)

    # --- rebound / collapse jet speed ----------------------------------------
    u_back = None
    if uz is not None:
        sub = liquid & below & roi[None]
        if sub.any():
            u_back = float(uz[sub].max() * velocity_scale)

    out = {
        "d_cav_open": None if d_open is None else d_open / n,
        "d_cav_all": None if d_all is None else d_all / n,
        "d_cav_gap": None if (d_open is None or d_all is None) else (d_all - d_open) / n,
        "d_ceil": d_ceil,
        "p_cav": None if (d_open is None or not d_all) else d_open / max(d_all, 1e-9),
        "cav_vent_ratio": vent_ratio,
        "cav_pockets": pockets,
        "cav_body_cells": body_cells,
        "u_back": u_back,
        "floor_open_z": floor_open,
        "floor_all_z": floor_all,
    }
    if markers is not None and len(markers):
        depths = markers.depths(surf, n)
        alive = markers.alive
        out["d_drop"] = float(depths[alive].max()) if alive.any() else None
        out["d_drop_any"] = float(depths.max())
        out["d_drop_max_live"] = float(markers.max_live_depth)
        out["markers"] = int(len(markers))
        out["markers_alive"] = int(alive.sum())
    else:
        out["d_drop"] = None
        out["d_drop_any"] = None
        out["d_drop_max_live"] = None
        out["markers"] = 0
        out["markers_alive"] = 0
    return out


def closure_point(field, gas, headspace, roi, surf, n):
    """Track the deepest open gas-liquid floor (closure point) and its position.

    The closure point is the deepest headspace-connected gas-to-liquid floor
    crossing inside the measurement ROI.  When the cavity is open (p_cav ~ 1)
    this is the bottom of the open cavity; when the neck starts to pinch off
    the deepest open point moves up toward the neck.

    Returns a dict with the sub-cell (x, z) position, the depth in diameters,
    or ``None`` when no open floor crossing exists.
    """
    k, j, i = np.where(headspace[1:] & ~gas[:-1] & roi[None])
    if not len(k):
        return None
    below = field[k, j, i]
    upper = field[k + 1, j, i]
    span = below - upper
    span = np.where(np.abs(span) < 1e-12, 1e-12, span)
    crossing = k + (below - 0.5) / span
    idx = np.argmin(crossing)
    z = float(crossing[idx])
    x = float(i[idx])
    y = float(j[idx])
    return {
        "z": z,
        "x": x,
        "y": y,
        "depth": (surf - z) / n,
    }
