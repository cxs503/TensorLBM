"""KVLCC2 three-dimensional hull surface + signed-distance machinery.

The full-ship ice-resistance chain needs three things from the hull:

1. a **verified** geometry (block coefficient gate ``Cb``), because every force
   integral is scaled by the wetted area and the displaced volume;
2. a **smooth** (non-staircase) description of the surface so that the lattice
   Boltzmann force integrals are not dominated by staircase noise -- the root
   cause of the 123 % GCI reported by the previous campaign node P66-W2-N04;
3. an **ice-drift-level outline** ``b(x)`` (half-breadth at the floe draft) used
   by the DEM branch: the bow shape enters the ice resistance through this
   outline (entrance angle, shoulder curvature), which is exactly the 3D effect
   that a 2D cross-section certification (IN37-N05) cannot capture.

Geometry model
--------------
The hull is described by a double-body analytic offset surface ``|y| = b(x, z)``
with ``x`` from the aft perpendicular (AP) and ``z`` from the baseline.  It
occupies ``{y < b(x, z), keel(x) < z < 0}`` with

    b(x, z) = B/2 * w(xi) * v(ze),      xi = x / L_pp,  ze = z / T

``w`` is the waterline factor (elliptical forebody entrance, fuller aftbody run,
parallel midbody), ``v`` the vertical section factor (U-shaped KVLCC2 sections:

``v(ze) = (1 - (1-ze)**p)**q``).  The two factors are separable, so

    Cb = mean(w) * mean(v)

which makes the block-coefficient gate transparent.  The **aftbody fullness
exponent is calibrated once** to the published ``Cb = 0.8098`` (standard
hull-form reconstruction); the waterplane coefficient ``C_wp`` is then an
*independent* check that the reconstruction is not degenerate (published
KVLCC2: C_wp = 0.8168).

References
----------
- SIMMAN 2008 workshop, KVLCC2 particulars (L_pp = 230 m, B = 32.29 m,
  T = 14.5 m, Cb = 0.8098, C_wp = 0.8168).
- ``tensorlbm.ship_hull_geometry``: the platform's offset-table variant used by
  the hydrodynamics modules (kept as an independent cross-check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# Physical constants of the full-scale ship (SIMMAN 2008).
L_PP = 230.0
BEAM = 32.29
DRAFT = 14.5
CB_TARGET = 0.8098
CWP_TARGET = 0.8168
CM_TARGET = 0.9878

#: ice-drift vertical level used for the floe outline: floes float with draft
#: ``rho_ice/rho_w * h_ice`` so their mid-plane sits just below the waterline.
#: Vertical coordinate convention: z = 0 at the baseline (keel), z = T at the
#: waterline (identical to ``kvlcc2_halfbreadth`` in ``ship_hull_geometry``).
ICE_LEVEL_Z = DRAFT - 0.5 * (917.0 / 1025.0) * 0.15  # ~ 14.43 m elevation


def _keel_z(x) -> np.ndarray:
    """Keel line elevation ``keel(x)`` from the baseline (>= 0).

    Nearly flat over the midbody; rises rapidly at the forefoot (sonar dome) and
    at the aft overhang.  The forefoot rise is what makes the bow a genuinely
    three-dimensional object at the ice level.
    """
    xi = np.clip(np.asarray(x, dtype=np.float64) / L_PP, 0.0, 1.0)
    aft = 0.30 * DRAFT * np.clip((0.06 - xi) / 0.06, 0.0, 1.0) ** 2
    fore = 1.00 * DRAFT * np.clip((xi - 0.90) / 0.10, 0.0, 1.0) ** 2.2
    return aft + fore


@dataclass
class KVLCC2Hull:
    """Analytic KVLCC2 double-body hull with a block-coefficient gate."""

    L_pp: float = L_PP
    B: float = BEAM
    T: float = DRAFT
    cb_target: float = CB_TARGET
    cwp_target: float = CWP_TARGET
    cm_target: float = CM_TARGET
    x_fore_start: float = 0.52      # start of the forebody (fraction of L_pp)
    x_aft_end: float = 0.28         # end of the aftbody (fraction of L_pp)
    n_fore: float = 2.0             # elliptical waterline entrance
    p_vert: float = 8.0             # vertical section exponent (calibrated to Cm)
    q_vert: float = 0.50            # vertical section exponent
    #: calibrated below so that the integrated Cb matches the published value
    n_aft: float = 2.5
    cb_tolerance: float = 0.01
    cwp_tolerance: float = 0.05

    # -- waterline factor --------------------------------------------------
    def _w(self, xi) -> np.ndarray:
        xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
        w = np.ones_like(xi)
        # forebody: s = 0 at the shoulder, 1 at the FP
        s = np.clip((xi - self.x_fore_start) / (1.0 - self.x_fore_start), 0.0, 1.0)
        w_fore = (1.0 - s ** self.n_fore) ** (1.0 / self.n_fore)
        # aftbody: a = 0 at the AP, 1 at the aft shoulder
        a = np.clip(xi / self.x_aft_end, 0.0, 1.0)
        w_aft = 1.0 - (1.0 - a) ** self.n_aft
        w = np.where(xi >= self.x_fore_start, w_fore, np.where(xi <= self.x_aft_end, w_aft, 1.0))
        return w

    def _v(self, ze) -> np.ndarray:
        ze = np.clip(np.asarray(ze, dtype=np.float64), 0.0, 1.0)
        return (1.0 - (1.0 - ze) ** self.p_vert) ** self.q_vert

    # -- offsets -----------------------------------------------------------
    def half_breadth(self, x, z):
        """Half-breadth ``b(x, z)`` in metres (``x`` from AP, ``z`` from baseline)."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        b = 0.5 * self.B * self._w(np.asarray(x / self.L_pp)) * self._v(np.asarray(z / self.T))
        outside = (np.asarray(z) < _keel_z(x)) | (np.asarray(z) > self.T)
        return np.where(outside, 0.0, b)

    def outline(self, x, z: float = ICE_LEVEL_Z):
        """Half-breadth at a fixed level (the DEM hull obstacle)."""
        x = np.asarray(x, dtype=np.float64)
        return self.half_breadth(x, np.full(x.shape, z))

    def signed_distance(self, x, y, z):
        """Approximate signed distance to the hull surface (positive = water)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        hx = 1e-3 * self.L_pp
        hz = 1e-3 * self.T
        b = self.half_breadth(x, z)
        bx = (self.half_breadth(x + hx, z) - self.half_breadth(x - hx, z)) / (2 * hx)
        bz = (self.half_breadth(x, z + hz) - self.half_breadth(x, z - hz)) / (2 * hz)
        d_lat = (b - np.abs(y)) / np.sqrt(1.0 + bx ** 2 + bz ** 2)
        d_keel = z - _keel_z(x)
        d_top = self.T - z
        return -np.minimum(np.minimum(d_lat, d_keel), d_top)

    # -- integrals / gates -------------------------------------------------
    def volume(self, n_x: int = 2400, n_z: int = 480) -> float:
        x = (np.arange(n_x) + 0.5) * self.L_pp / n_x
        z = (np.arange(n_z) + 0.5) * self.T / n_z
        XX, ZZ = np.meshgrid(x, z, indexing="ij")
        integrand = 2.0 * self.half_breadth(XX, ZZ)
        return float(integrand.sum() * (self.L_pp / n_x) * (self.T / n_z))

    def waterplane_area(self, n_x: int = 2400) -> float:
        x = (np.arange(n_x) + 0.5) * self.L_pp / n_x
        return float(2.0 * self.half_breadth(x, np.full_like(x, 0.999 * self.T)).sum()
                     * self.L_pp / n_x)

    def wetted_area(self, n_x: int = 1200, n_z: int = 240) -> float:
        """Wetted surface by quadrature of the offset surface (both sides)."""
        x = (np.arange(n_x) + 0.5) * self.L_pp / n_x
        z = (np.arange(n_z) + 0.5) * self.T / n_z
        XX, ZZ = np.meshgrid(x, z, indexing="ij")
        b = self.half_breadth(XX, ZZ)
        hx = np.gradient(x)
        hz = np.gradient(z)
        bx = np.gradient(b, axis=0) / hx[:, None]
        bz = np.gradient(b, axis=1) / hz[None, :]
        dA = np.sqrt(1.0 + bx ** 2 + bz ** 2)
        return float(2.0 * dA.sum() * hx.mean() * hz.mean())

    def section_coefficient(self, n_z: int = 2000) -> float:
        """Midship section coefficient ``Cm = mean_z v(ze)``."""
        ze = (np.arange(n_z) + 0.5) / n_z
        return float(self._v(ze).mean())

    def waterline_coefficient(self, n_x: int = 4000) -> float:
        """Waterplane coefficient ``Cwp = mean_x w(xi)``."""
        xi = (np.arange(n_x) + 0.5) / n_x
        return float(self._w(xi).mean())

    def calibrate(self, verbose: bool = False) -> Dict[str, float]:
        """Fit two shape exponents to two published particulars.

        * ``p_vert`` is solved so that the midship section coefficient matches the
          published ``Cm = 0.9878`` (near-rectangular tanker section);
        * ``n_aft`` is solved so that ``mean(w) = Cb / Cm``, i.e. the integrated
          block coefficient matches the published ``Cb = 0.8098``.

        ``C_wp`` is deliberately *not* used in the fit: it is the independent
        check in :meth:`verify`.
        """
        lo, hi = 2.0, 40.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            self.p_vert = mid
            if self.section_coefficient() < self.cm_target:
                lo = mid
            else:
                hi = mid
        self.p_vert = 0.5 * (lo + hi)

        # n_aft: match the integrated (keel-corrected) block coefficient
        lo, hi = 1.0, 40.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            self.n_aft = mid
            cb = self.volume(400, 120) / (self.L_pp * self.B * self.T)
            if cb < self.cb_target:
                lo = mid
            else:
                hi = mid
        self.n_aft = 0.5 * (lo + hi)
        out = {"p_vert": self.p_vert, "n_aft": self.n_aft,
               "cm": self.section_coefficient(), "cwp_pred": self.waterline_coefficient()}
        if verbose:
            print(out)
        return out

    def verify(self, n_x: int = 2400, n_z: int = 480, raise_on_fail: bool = False) -> Dict[str, float]:
        """Geometry gate: ``Cb`` (hard) and ``C_wp`` (independent sanity check)."""
        vol = self.volume(n_x, n_z)
        cb = vol / (self.L_pp * self.B * self.T)
        cb_err = abs(cb - self.cb_target) / self.cb_target
        awp = self.waterplane_area()
        cwp = awp / (self.L_pp * self.B)
        cm = self.section_coefficient()
        cwp_err = abs(cwp - self.cwp_target) / self.cwp_target
        out = {
            "volume_m3": vol,
            "cb": cb,
            "cb_target": self.cb_target,
            "cb_rel_error": cb_err,
            "cwp": cwp,
            "cwp_target": self.cwp_target,
            "cm": cm,
            "cm_target": self.cm_target,
            "cm_rel_error": abs(cm - self.cm_target) / self.cm_target,
            "cwp_rel_error": cwp_err,
            "pass_cb": bool(cb_err <= self.cb_tolerance),
            "pass_cwp": bool(cwp_err <= self.cwp_tolerance),
            "L_pp": self.L_pp, "B": self.B, "T": self.T, "n_aft": self.n_aft,
        }
        out["pass"] = bool(out["pass_cb"] and out["pass_cwp"])
        if raise_on_fail and not out["pass_cb"]:
            raise ValueError(f"KVLCC2 Cb gate failed: {cb:.4f} vs {self.cb_target:.4f}")
        return out


def make_hull(calibrate: bool = True, **kw) -> KVLCC2Hull:
    hull = KVLCC2Hull(**kw)
    if calibrate:
        hull.calibrate()
    return hull


def ice_drift_outline(hull: KVLCC2Hull, x, z: float = ICE_LEVEL_Z):
    return hull.outline(x, z)


def outline_entrance_angle(hull: KVLCC2Hull, z: float = ICE_LEVEL_Z,
                           d: float = 1.0) -> float:
    """Entrance half-angle at the ice level, in degrees (measured ``d`` m aft of FP)."""
    x0 = hull.L_pp - 1e-3
    b0 = float(hull.outline(np.array([x0]), z)[0])
    b1 = float(hull.outline(np.array([x0 - d]), z)[0])
    dbdx = (b0 - b1) / d
    return float(math.degrees(math.atan(max(abs(dbdx), 1e-9))))


def rasterise_mask(hull: KVLCC2Hull, x_lo: float, z_lo: float, d: float,
                   n_x: int, n_y: int, n_z: int, dz: Optional[float] = None) -> np.ndarray:
    """Voxelise the hull on a uniform lattice (cell-centred, ``y >= 0`` half).

    Returns a boolean array ``(n_z, n_y, n_x)`` = True where the hull occupies
    the cell centre.  ``d`` is the (isotropic) cell size in metres.
    """
    dz = d if dz is None else dz
    xc = x_lo + (np.arange(n_x) + 0.5) * d
    yc = (np.arange(n_y) + 0.5) * d
    zc = z_lo + (np.arange(n_z) + 0.5) * dz
    XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
    b = hull.half_breadth(XX, ZZ)              # (n_x, n_z)
    keel = _keel_z(xc)
    solid = np.zeros((n_z, n_y, n_x), dtype=bool)
    for k in range(n_z):
        z = zc[k]
        inside_z = (z > keel) & (z <= hull.T)
        bj = np.where(inside_z, b[:, k], 0.0)
        solid[k] = (yc[None, :] < bj[:, None]).T & inside_z[None, :]
    return solid