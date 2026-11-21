"""Minimal, self-contained D2Q9 core used *only* by the invariance harness.

Why a dedicated kernel instead of the production solver
-------------------------------------------------------
The invariance face must be able to answer "does a *change of frame* change the
answer?"  That question is only meaningful if the discretisation is held fixed
while the frame is rotated / boosted.  The production solvers in
:mod:`tensorlbm.solver` / :mod:`tensorlbm.cylinder_flow` bundle many modelling
choices (wall functions, LES closures, force schemes) that would move with the
frame and blur the measurement.  This module therefore keeps the reference
numeric scheme as small and as explicit as possible:

* D2Q9, BGK collision, Guo forcing (second-order accurate body force),
* streaming by exact lattice shifts (``numpy.roll``),
* walls: Bouzidi-Firdaouss-Lallemand (BFL) interpolated bounce-back with a
  *sub-grid distance taken from the analytical signed-distance function*
  (see :mod:`tensorlbm.invariance.geometry`), so that the wall location error
  does not depend on the lattice orientation,
* straight boundaries: Zou-He velocity inlet / pressure outlet, specular
  (free-slip) walls,
* force on an immersed body: Ladd momentum exchange on the wall links.

All arrays are ``float64``; the field layout is ``f[9, ny, nx]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- lattice constants (d2q9) ------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
CW = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0
CS = np.sqrt(CS2)

_W = CW.reshape(9, 1, 1)
_CX = CX.reshape(9, 1, 1).astype(np.float64)
_CY = CY.reshape(9, 1, 1).astype(np.float64)


# --- basic operators ---------------------------------------------------------
def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 second-order equilibrium, ``rho``/``ux``/``uy`` shape ``(ny, nx)``."""
    cu = _CX * ux[None] + _CY * uy[None]
    usq = ux * ux + uy * uy
    return _W * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    ux = np.tensordot(CX.astype(np.float64), f, axes=(0, 0)) / rho
    uy = np.tensordot(CY.astype(np.float64), f, axes=(0, 0)) / rho
    return rho, ux, uy


def stream(f_post: np.ndarray) -> np.ndarray:
    """Exact lattice streaming ``f_i(x, t+1) = f_i^*(x - c_i, t)``.

    ``numpy.roll`` is used for *all* directions.  For a non-periodic boundary the
    wrap-around only ever contaminates the populations whose source node lies
    outside the box, and those are exactly the unknown populations that the
    boundary condition then overwrites - so mixing periodic and non-periodic
    sides in one array is safe.
    """
    out = np.empty_like(f_post)
    for i in range(9):
        out[i] = np.roll(np.roll(f_post[i], CX[i], axis=1), CY[i], axis=0)
    return out


# --- wall links --------------------------------------------------------------
@dataclass
class WallLinks:
    """Fluid->solid links of one immersed object.

    ``ix``/``iy`` are the fluid node indices, ``direction`` the lattice
    direction from the fluid node to the solid neighbour, ``q`` the sub-grid
    distance to the wall in units of the link length (0 < q <= 1).
    """

    ix: np.ndarray
    iy: np.ndarray
    direction: np.ndarray
    q: np.ndarray
    behind_fluid: np.ndarray
    object_id: int = 0
    force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_links: int = 0

    @property
    def size(self) -> int:
        return int(self.ix.size)


def build_wall_links(
    phi: np.ndarray,
    phi_ray,
    object_id: int = 0,
    radius_of_object: int | None = None,
) -> WallLinks:
    """Enumerate the fluid->solid links of the region ``phi > 0``.

    ``phi`` is the analytical signed distance sampled on the lattice (negative
    inside the fluid).  ``phi_ray(x, y, cx, cy)`` returns the distance, measured
    in units of the link vector, from ``(x, y)`` along ``(cx, cy)`` to the
    boundary; it is used to obtain the sub-grid distance ``q`` without any
    staircase approximation.
    """
    # nodes lying exactly on the wall (phi == 0) are treated as solid: keeping them
    # fluid produces links with a sub-grid distance of ~0, whose interpolated
    # bounce-back weights (2q -> 0) are the stiffest possible and were the source
    # of the startup instability observed on coarse cylinders.
    solid = phi >= 0.0
    fluid = ~solid
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ds: list[np.ndarray] = []
    for i in range(1, 9):
        # neighbour of (x, y) in direction i  ->  shifted solid mask
        nb_solid = np.roll(np.roll(solid, -CX[i], axis=1), -CY[i], axis=0)
        hit = fluid & nb_solid
        jy, kx = np.nonzero(hit)
        if jy.size == 0:
            continue
        xs.append(kx.astype(np.int64))
        ys.append(jy.astype(np.int64))
        ds.append(np.full(jy.size, i, dtype=np.int64))
    if not xs:
        empty = np.zeros(0, dtype=np.int64)
        return WallLinks(empty, empty, empty, np.zeros(0), np.zeros(0, bool), object_id)
    ix = np.concatenate(xs)
    iy = np.concatenate(ys)
    direction = np.concatenate(ds)
    q_raw = phi_ray(ix, iy, CX[direction].astype(np.float64), CY[direction].astype(np.float64))
    q = np.clip(q_raw, 0.05, 1.0)
    by = iy - CY[direction]
    bx = ix - CX[direction]
    ny, nx = phi.shape
    inside = (by >= 0) & (by < ny) & (bx >= 0) & (bx < nx)
    behind_fluid = np.zeros(ix.size, dtype=bool)
    behind_fluid[inside] = fluid[by[inside], bx[inside]]
    # where the node behind the link is solid as well (a one-cell liquid sliver)
    # the interpolation stencil does not exist: fall back to half-way bounce-back,
    # which is the standard robust choice and keeps the reflection well-posed.
    q = np.where(behind_fluid, q, 0.5)
    return WallLinks(ix, iy, direction, q, behind_fluid, object_id)


def apply_wall_links(f_new: np.ndarray, f_post: np.ndarray, links: WallLinks) -> np.ndarray:
    """BFL interpolated bounce-back + Ladd momentum exchange (returns force)."""
    if links.size == 0:
        return np.zeros(2)
    i = links.direction
    cy = CY[i]
    cx = CX[i]
    jy = links.iy
    kx = links.ix
    y2 = np.where(links.behind_fluid, jy - cy, jy)
    x2 = np.where(links.behind_fluid, kx - cx, kx)
    fi = f_post[i, jy, kx]
    fi2 = f_post[i, y2, x2]
    q = links.q
    near = q < 0.5
    a = np.where(near, 2.0 * q, 0.5 / q)
    b = np.where(near, 1.0 - 2.0 * q, 1.0 - 0.5 / q)
    fb = a * fi + b * fi2
    f_new[OPP[i], jy, kx] = fb
    # momentum exchanged with the solid along the link (Ladd)
    dp = fb + fi
    force = np.array([np.sum(cx * dp), np.sum(cy * dp)])
    links.force = force
    links.n_links = int(ix_size(links))
    return force


def ix_size(links: WallLinks) -> int:
    return int(links.ix.size)


# --- simulation --------------------------------------------------------------
class Simulation:
    """D2Q9 BGK solver with SDF walls and the boundary set used by the harness."""

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float,
        force: tuple[float, float] = (0.0, 0.0),
        bc: dict[str, str] | None = None,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        if self.tau <= 0.5:
            raise ValueError(f"tau must exceed 0.5, got {self.tau}")
        zeros = np.zeros((self.ny, self.nx))
        self.f = equilibrium(np.ones((self.ny, self.nx)), zeros, zeros)
        self.force = np.asarray(force, dtype=np.float64)
        self.bc = {"w": "periodic", "e": "periodic", "s": "periodic", "n": "periodic"}
        if bc:
            self.bc.update(bc)
        self.links: list[WallLinks] = []
        self.inlet_velocity = np.zeros(2)
        self.outlet_density = 1.0
        self.t = 0
        self._f_post = self.f.copy()

    # -- configuration --
    def add_wall_links(self, links: WallLinks) -> WallLinks:
        self.links.append(links)
        return links

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0

    # -- state --
    def fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return macroscopic(self.f)

    def initialise_equilibrium(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> None:
        self.f = equilibrium(rho, ux, uy)

    # -- one time step --
    def _guo_source(self, rho, ux, uy) -> np.ndarray:
        fx, fy = self.force
        cuf = _CX * fx + _CY * fy
        cu = _CX * ux[None] + _CY * uy[None]
        src = 3.0 * (cuf - (ux[None] * fx + uy[None] * fy)) + 9.0 * cu * cuf
        return (1.0 - 0.5 / self.tau) * _W * src

    def step(self) -> None:
        rho, ux, uy = macroscopic(self.f)
        if self.force.any():
            fx, fy = self.force
            uxg = ux + 0.5 * fx / rho
            uyg = uy + 0.5 * fy / rho
        else:
            uxg, uyg = ux, uy
        feq = equilibrium(rho, uxg, uyg)
        f_post = self.f + (feq - self.f) / self.tau
        if self.force.any():
            f_post += self._guo_source(rho, uxg, uyg)
        self._f_post = f_post
        f_new = stream(f_post)
        for links in self.links:
            apply_wall_links(f_new, f_post, links)
        self._apply_side_bc(f_new)
        self.f = f_new
        self.t += 1

    # -- boundary conditions --
    def _apply_side_bc(self, f: np.ndarray) -> None:
        bc = self.bc
        if bc["w"] == "velocity":
            # Zou-He velocity inlet, non-equilibrium bounce-back (NBB) closure.
            # The previous version mixed the NBB unknowns with a transverse
            # velocity built from the *unknown* populations; that inconsistency
            # is unconditionally unstable (verified: a uniform flow diverged to
            # NaN within 200 steps).  With the NBB-consistent set below a uniform
            # flow is reproduced exactly and the transverse velocity is exact at
            # equilibrium.
            ux, uy = self.inlet_velocity
            f0, f2, f4 = f[0, :, 0], f[2, :, 0], f[4, :, 0]
            f3, f6, f7 = f[3, :, 0], f[6, :, 0], f[7, :, 0]
            rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)
            f[1, 1:-1, 0] = f3[1:-1] + (2.0 / 3.0) * rho[1:-1] * ux
            f[5, 1:-1, 0] = f7[1:-1] + rho[1:-1] * (ux + uy) / 6.0
            f[8, 1:-1, 0] = f6[1:-1] + rho[1:-1] * (ux - uy) / 6.0
        if bc["e"] == "neumann":
            # zero-gradient outflow with the density anchored to rho_out.
            # Eine Zou-He pressure outlet reflects the vortices that cross it and
            # was measured to blow up between step 150 and 200 on the cylinder
            # case (the max |u| first appeared in the outlet column).  The Neumann
            # outlet is the standard robust choice for vortex convection.
            f[:, :, -1] = f[:, :, -2]
            rho_col = f[:, :, -1].sum(axis=0)
            f[:, :, -1] *= (self.outlet_density / rho_col)[None, :]
        if bc["e"] == "pressure":
            # Zou-He pressure outlet, NBB closure
            rho_w = self.outlet_density
            f0, f2, f4 = f[0, :, -1], f[2, :, -1], f[4, :, -1]
            f1, f5, f8 = f[1, :, -1], f[5, :, -1], f[8, :, -1]
            ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_w
            uy = 1.5 * (f2 - f4) / rho_w
            f[3, 1:-1, -1] = f1[1:-1] - (2.0 / 3.0) * rho_w * ux[1:-1]
            f[6, 1:-1, -1] = f8[1:-1] + rho_w * (uy[1:-1] - ux[1:-1]) / 6.0
            f[7, 1:-1, -1] = f5[1:-1] - rho_w * (ux[1:-1] + uy[1:-1]) / 6.0
        if bc["s"] == "velocity":
            # far-field velocity Dirichlet on the south row (unknowns: 2, 5, 6)
            ux, uy = self.inlet_velocity
            f0, f1, f3 = f[0, 0, :], f[1, 0, :], f[3, 0, :]
            f4, f7, f8 = f[4, 0, :], f[7, 0, :], f[8, 0, :]
            rho = (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / (1.0 - uy)
            f[2, 0, 1:-1] = f4[1:-1] + (2.0 / 3.0) * rho[1:-1] * uy
            f[5, 0, 1:-1] = f7[1:-1] + rho[1:-1] * (ux + uy) / 6.0
            f[6, 0, 1:-1] = f8[1:-1] + rho[1:-1] * (uy - ux) / 6.0
        if bc["n"] == "velocity":
            ux, uy = self.inlet_velocity
            f0, f1, f3 = f[0, -1, :], f[1, -1, :], f[3, -1, :]
            f2, f5, f6 = f[2, -1, :], f[5, -1, :], f[6, -1, :]
            rho = (f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)) / (1.0 + uy)
            f[4, -1, 1:-1] = f2[1:-1] - (2.0 / 3.0) * rho[1:-1] * uy
            f[8, -1, 1:-1] = f6[1:-1] + rho[1:-1] * (ux - uy) / 6.0
            f[7, -1, 1:-1] = f5[1:-1] - rho[1:-1] * (ux + uy) / 6.0
        if bc["s"] == "free_slip":
            # specular wall: populations entering from row -1 have cy = +1
            # (i = 2, 5, 6) and mirror the outgoing (cy = -1) ones (i = 4, 7, 8)
            f[2, 0, :] = f[4, 0, :]
            f[6, 0, :] = f[7, 0, :]
            f[5, 0, :] = f[8, 0, :]
        if bc["n"] == "free_slip":
            f[4, -1, :] = f[2, -1, :]
            f[7, -1, :] = f[6, -1, :]
            f[8, -1, :] = f[5, -1, :]

    # -- diagnostics --
    def total_momentum(self) -> np.ndarray:
        rho, ux, uy = macroscopic(self.f)
        return np.array([np.sum(rho * ux), np.sum(rho * uy)])

    def max_velocity_change(self, ux_old: np.ndarray, uy_old: np.ndarray) -> float:
        _, ux, uy = macroscopic(self.f)
        return float(max(np.abs(ux - ux_old).max(), np.abs(uy - uy_old).max()))