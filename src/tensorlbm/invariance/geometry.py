"""Analytic geometry (SDF) rotated arbitrarily relative to the lattice.

The invariance test-suite needs the *same physical problem* discretised with
the computational lattice at an arbitrary orientation ``theta`` relative to the
body.  Instead of rotating the lattice (impossible on a fixed D2Q9 stencil) we
keep the lattice fixed and rotate the analytic geometry / driving direction by
``theta``.  Because the physical problem is rotation invariant, every
``theta`` describes the same flow; only the lattice/body orientation differs.

Conventions
-----------
* Lattice spacings are ``dx = dy = 1``; cell ``(i, j)`` has centre
  ``(x, y) = (j + 0.5, i + 0.5)`` (``i`` = row, ``j`` = column).
* Every geometry exposes a **true signed distance function** ``sdf``
  (positive in the fluid) and its exact gradient ``grad``.  Both are used by
  the interpolated bounce-back in :mod:`tensorlbm.invariance.solver` to get the
  wall-distance fraction ``q`` along each lattice direction without any
  staircase approximation of the distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry2D:
    """Container for a 2-D body description on a lattice-aligned grid."""

    solid: np.ndarray          # bool[ny, nx], True = solid
    sdf: np.ndarray            # float[ny, nx], >0 in fluid
    gx: np.ndarray             # ∂sdf/∂x
    gy: np.ndarray             # ∂sdf/∂y
    nx: int
    ny: int
    theta_rad: float
    axis: tuple[float, float]  # flow / channel axis unit vector (t̂)
    normal: tuple[float, float]
    meta: dict

    @property
    def fluid(self) -> np.ndarray:
        return ~self.solid


def _grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=np.float64) + 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    return np.meshgrid(x, y, indexing="xy")  # X, Y both (ny, nx)


def tilted_channel(
    nx: int,
    ny: int,
    theta_deg: float,
    width: float,
    center: tuple[float, float] | None = None,
) -> Geometry2D:
    """Infinite strip (plane channel) of width ``width`` inclined by ``theta``.

    The strip runs through the box from the left face to the right face; the
    part of the box outside the strip is solid.  The two walls are at
    ``n̂·(r - c) = ±width/2`` with ``n̂ = (-sinθ, cosθ)`` (a unit normal) and
    ``t̂ = (cosθ, sinθ)`` the channel axis.
    """
    theta = math.radians(theta_deg)
    nx_hat = (-math.sin(theta), math.cos(theta))
    t_hat = (math.cos(theta), math.sin(theta))
    X, Y = _grid(nx, ny)
    if center is None:
        center = (0.5 * nx, 0.5 * ny)
    cx, cy = center
    d = nx_hat[0] * (X - cx) + nx_hat[1] * (Y - cy)      # signed normal offset
    sdf = 0.5 * width - np.abs(d)
    sgn = -np.sign(d)
    gx = sgn * nx_hat[0]
    gy = sgn * nx_hat[1]
    solid = sdf <= 0.0
    return Geometry2D(
        solid=solid,
        sdf=sdf,
        gx=gx,
        gy=gy,
        nx=nx,
        ny=ny,
        theta_rad=theta,
        axis=t_hat,
        normal=nx_hat,
        meta={"kind": "tilted_channel", "width": width, "theta_deg": theta_deg,
              "center": (cx, cy)},
    )


def circle(
    nx: int,
    ny: int,
    radius: float,
    center: tuple[float, float] | None = None,
    theta_deg: float = 0.0,
) -> Geometry2D:
    """Solid circular cylinder in a surrounding fluid (external-flow convention).

    ``sdf = |r - c| - R`` (positive in the fluid outside the disk) and
    ``grad = (r - c)/|r - c|``.  The body is rotationally symmetric, so
    rasterising it is independent of ``theta``; what changes with ``theta`` is
    the orientation of the driving force with respect to the lattice axes,
    which ``axis``/``normal`` record.
    """
    theta = math.radians(theta_deg)
    t_hat = (math.cos(theta), math.sin(theta))
    n_hat = (-math.sin(theta), math.cos(theta))
    X, Y = _grid(nx, ny)
    if center is None:
        center = (0.5 * nx, 0.5 * ny)
    cx, cy = center
    rx = X - cx
    ry = Y - cy
    dist = np.sqrt(rx * rx + ry * ry)
    dist_safe = np.maximum(dist, 1e-12)
    sdf = dist - radius
    gx = rx / dist_safe
    gy = ry / dist_safe
    solid = sdf <= 0.0
    return Geometry2D(
        solid=solid, sdf=sdf, gx=gx, gy=gy, nx=nx, ny=ny, theta_rad=theta,
        axis=t_hat, normal=n_hat,
        meta={"kind": "circle", "radius": radius, "theta_deg": theta_deg,
              "center": (cx, cy)},
    )


def channel_boundary_cells(geom: Geometry2D) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks ``(inlet, outlet)`` for the tilted channel.

    The inlet is the fluid part of the first column (``j = 0``) and the outlet
    the fluid part of the last column (``j = nx-1``).  Both are lattice-aligned
    faces, so velocity Dirichlet data can be applied column-wise.
    """
    fluid = ~geom.solid
    inlet = np.zeros_like(fluid)
    outlet = np.zeros_like(fluid)
    inlet[:, 0] = fluid[:, 0]
    outlet[:, -1] = fluid[:, -1]
    return inlet, outlet


def poiseuille_profile(geom: Geometry2D, u_mean: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact developed plane-Poiseuille velocity field for the tilted channel.

    ``u(y') = G y'(H - y') / (2 ν)`` in wall-normal coordinates, i.e. a parabola
    with maximum ``u_max = 3/2 u_mean`` on the channel axis.
    """
    width = float(geom.meta["width"])
    sdf = geom.sdf
    # |d| = H/2 - sdf  (inside the strip); eta = 2|d|/H in [0, 1]
    eta = np.clip(1.0 - 2.0 * sdf / width, 0.0, 1.0)
    profile = 1.5 * u_mean * (1.0 - eta * eta)
    tx, ty = geom.axis
    ux = profile * tx
    uy = profile * ty
    return ux, uy


def rotated_unit(theta_deg: float) -> tuple[float, float, float, float]:
    """Return ``(tx, ty, nx_hat_x, nx_hat_y)`` for a rotation angle."""
    th = math.radians(theta_deg)
    return math.cos(th), math.sin(th), -math.sin(th), math.cos(th)