"""Analytical geometry for the invariance harness.

Every wall is described by an *analytical signed distance function* (SDF) that
can be evaluated at an arbitrary point of the plane.  Two consequences matter
for invariance work:

1. the geometry that is discretised is **identical** for every orientation - only
   the sampling lattice changes, which is precisely the transformation whose
   effect we want to measure;
2. the sub-grid wall distance ``q`` of the interpolated bounce-back scheme can be
   taken from the SDF (via a bisection along the link) instead of from a
   staircase count, so the wall location error is ``O(h^2)`` and, to leading
   order, orientation independent.

Coordinates
-----------
``x_lat`` are lattice coordinates.  A :class:`Frame` maps them to *physical*
coordinates ``x_phys = R(theta) @ (x_lat - origin)``.  Physical geometries
(circles, ...) are therefore sampled on a lattice that is rotated by ``theta``
with respect to the body, which is exactly the "grid orientation" experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --- frame -------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """Rotation (+ translation) between lattice and physical coordinates."""

    theta_deg: float = 0.0
    origin: tuple[float, float] = (0.0, 0.0)

    @property
    def theta(self) -> float:
        return math.radians(self.theta_deg)

    def to_physical(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        xl = x - self.origin[0]
        yl = y - self.origin[1]
        return c * xl - s * yl, s * xl + c * yl

    def to_lattice_velocity(self, vx: np.ndarray | float, vy: np.ndarray | float):
        """Physical velocity components -> lattice-frame components."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * vx + s * vy, -s * vx + c * vy

    def to_physical_velocity(self, vx, vy):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * vx - s * vy, s * vx + c * vy


# --- geometries --------------------------------------------------------------
class Circle:
    """Circle of radius ``radius`` centred at the physical origin."""

    def __init__(self, radius: float, frame: Frame | None = None) -> None:
        self.radius = float(radius)
        self.frame = frame or Frame()

    def sdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp, yp = self.frame.to_physical(x, y)
        return np.hypot(xp, yp) - self.radius

    def __call__(self, x, y):
        return self.sdf(x, y)


class TiltedChannelArray:
    """Periodic array of plane channels tilted by ``theta_deg`` w.r.t. the lattice.

    The fluid occupies the slabs ``|n . x - k d| < width / 2`` where
    ``n = (-sin t, cos t)`` and ``d`` is the perpendicular spacing between channel
    walls of neighbouring channels.  For a doubly periodic box of side ``L`` the
    array is periodic iff ``tan(t) = k/m`` is rational, ``d = L / sqrt(m^2+k^2)``.
    """

    def __init__(self, theta_deg: float, width: float, spacing: float) -> None:
        self.theta_deg = float(theta_deg)
        self.width = float(width)
        self.spacing = float(spacing)
        t = math.radians(self.theta_deg)
        self.normal = np.array([-math.sin(t), math.cos(t)])
        self.axis = np.array([math.cos(t), math.sin(t)])

    def sdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u = self.normal[0] * x + self.normal[1] * y
        d = self.spacing
        s = np.mod(u + 0.5 * d, d) - 0.5 * d
        return np.abs(s) - 0.5 * self.width

    def __call__(self, x, y):
        return self.sdf(x, y)


# --- sampling helpers --------------------------------------------------------
def grid(nx: int, ny: int):
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    return np.meshgrid(x, y)


def sample(sdf, nx: int, ny: int) -> np.ndarray:
    x, y = grid(nx, ny)
    return sdf(x, y)


def ray_distance(sdf, x: np.ndarray, y: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                 iters: int = 52) -> np.ndarray:
    """Distance (in units of the link vector) from ``(x, y)`` along ``(cx, cy)`` to ``sdf=0``.

    Vectorised bisection; the caller guarantees ``sdf(x, y) < 0 < sdf(x+c, y+c)``.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    cx = cx.astype(np.float64)
    cy = cy.astype(np.float64)
    lo = np.zeros_like(x)
    hi = np.ones_like(x)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = sdf(x + mid * cx, y + mid * cy)
        positive = val > 0.0
        hi = np.where(positive, mid, hi)
        lo = np.where(positive, lo, mid)
    return 0.5 * (lo + hi)


def make_ray_distance(sdf):
    """Bind :func:`ray_distance` to ``sdf`` and offset-index arrays in a closure."""

    def _ray(ix, iy, cx, cy):
        return ray_distance(sdf, np.asarray(ix), np.asarray(iy), cx, cy)

    return _ray


def fluid_area_fraction(mask_fluid: np.ndarray) -> float:
    return float(mask_fluid.sum()) / float(mask_fluid.size)


def effective_width(mask_fluid: np.ndarray) -> float:
    """Perpendicular channel width implied by the resolved fluid area fraction.

    For the tilted channel array the exact statement is ``H / d = A_fluid / A_box``;
    the returned value is the width actually seen by the discretisation and is the
    geometry sanity check used before any physics is trusted.
    """
    frac = fluid_area_fraction(mask_fluid)
    return frac  # dimensionless; callers multiply by spacing


def obstacle_perimeter_estimate(phi: np.ndarray, h: float = 1.0) -> float:
    """Length of the ``phi = 0`` contour, from the resolved solid mask."""
    solid = (phi > 0).astype(np.float64)
    grad_x = np.abs(np.diff(solid, axis=1)).sum()
    grad_y = np.abs(np.diff(solid, axis=0)).sum()
    return float((grad_x + grad_y) * h)