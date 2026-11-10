"""Spherical (facet) force method: surface-element force integration for curved LBM walls.

Motivation
----------
The force routes shipped so far in this package are *link based*: momentum
exchange (ME) sums the population pairs crossing every cut link, and the control
volume (CV) route integrates the stress tensor over a closed box
(``docs/FORCE_METHODS_SURVEY.md``).  Neither returns the quantity a hydrodynamics
load study actually needs: the **pressure and friction distribution on the
surface itself**.  The spherical force method (SFM) closes that gap.  The wetted
surface is discretised into *facets* (segments in 2D, triangles in 3D); on every
facet the pressure and the viscous traction are evaluated at quadrature points,
multiplied by the facet measure and summed:

    F = sum_f A_f * [ -p(x_f) n_f + mu (grad u + grad u^T)(x_f) . n_f ]

Because the sum runs over the *geometry* instead of over lattice links, one
integrator serves a 2D cylinder, a 3D sphere and (in principle) any faceted hull,
provided the flow is available on a lattice.  It also produces per-facet Cp, the
input of every load study, which ME/CV cannot give.

Wall quantities are reconstructed from the fluid side only: at each quadrature
point the field is evaluated at ``x_f + d n_f`` for d = 1.5, 2.5, 3.5 lattice
spacings and extrapolated back to the wall with a quadratic (three-point)
formula.  That is second order accurate and never touches solid nodes, whose
populations are meaningless under bounce-back.

Contents
--------
* :class:`FacetMesh`, :func:`circle_facet_mesh`, :func:`sphere_facet_mesh` -
  faceted geometry with normals, measures, quadrature and point-in-body tests.
* :func:`integrate_surface_force` - the SFM integrator (2D and 3D).
* :func:`potential_flow_grid`, :func:`potential_flow_cp_theory` - analytic
  potential-flow reference fields used to verify the integrator on a
  lattice-sampled field.
* :func:`run_cylinder_case` - self-contained numpy D2Q9 driver (TRT collision,
  Bouzidi interpolated bounce-back evaluated on the *facet* geometry) which
  produces the field the integrator consumes, plus momentum-exchange
  cross-checks and the domain mass balance.
* :func:`mass_flow_profile` - inflow/outflow mass balance.

References
----------
* Ladd (1994) J. Fluid Mech. 271, 285 - momentum exchange.
* Bouzidi, Firdaouss & Lallemand (2001) Phys. Fluids 13, 3452 - interpolated
  bounce-back; the reflection weights used here are the ones already verified in
  :mod:`tensorlbm.curved_wall_force_scaling`.
* Zdravkovich (1997) *Flow Around Circular Cylinders*, Sucker & Brauer (1975) -
  cylindrical Cd reference curve in :data:`CYLINDER_CD_REFERENCE`.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .curved_wall_force_scaling import CX, CY, OPP, W

#: lattice sound speed squared
CS2 = 1.0 / 3.0

CX_ARR = np.asarray(CX, dtype=np.float64)
CY_ARR = np.asarray(CY, dtype=np.float64)
W_ARR = np.asarray(W, dtype=np.float64)
OPP_ARR = np.asarray(OPP, dtype=np.int64)

#: drag coefficient of the *unbounded* circular cylinder (Sucker & Brauer 1975 /
#: Zdravkovich 1997 tabulations, interpolated on Re).
CYLINDER_CD_REFERENCE: dict[float, float] = {
    10.0: 2.85,
    20.0: 2.05,
    40.0: 1.54,
    60.0: 1.42,
    100.0: 1.34,
    150.0: 1.34,
    200.0: 1.34,
}

#: polar angles (deg) at which the sphere Cp distribution is reported
SPHERE_CP_ANGLES_DEG: tuple[float, ...] = (
    0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0,
    105.0, 120.0, 135.0, 150.0, 165.0, 180.0,
)


# ---------------------------------------------------------------------------
# faceted surface geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FacetMesh:
    """Closed faceted surface: segments in 2D, triangles in 3D."""

    vertices: np.ndarray
    faces: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    center: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def total_area(self) -> float:
        """Perimeter in 2D, surface area in 3D."""
        return float(self.areas.sum())

    def quadrature(self, kind: str = "auto"):
        """Quadrature ``(points, weights, normals)``; weights sum to ``total_area``."""
        if kind == "auto":
            kind = "gauss2" if self.dim == 2 else "bary3"
        if self.dim == 2:
            a = self.vertices[self.faces[:, 0]]
            b = self.vertices[self.faces[:, 1]]
            if kind in {"midpoint", "order1"}:
                return 0.5 * (a + b), self.areas.copy(), self.normals.copy()
            if kind in {"gauss2", "order3"}:
                t = 0.5 + np.array([-1.0, 1.0]) / (2.0 * math.sqrt(3.0))
                # facet-major ordering: quadrature points of one facet are adjacent
                pts = (a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]).reshape(-1, 2)
                w = np.repeat(0.5 * self.areas, 2)
                return pts, w, np.repeat(self.normals, 2, axis=0)
            raise ValueError(f"unknown quadrature {kind!r}")
        tris = self.vertices[self.faces]
        if kind in {"centroid", "order1"}:
            return self.centroids.copy(), self.areas.copy(), self.normals.copy()
        if kind in {"bary3", "order3"}:
            b = np.array([[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]]) / 6.0
            # barycentric combination of the three facet vertices, facet-major order
            pts = np.stack([b[k] @ tris for k in range(3)], axis=1).reshape(-1, 3)
            w = np.repeat(self.areas / 3.0, 3)
            return pts, w, np.repeat(self.normals, 3, axis=0)
        raise ValueError(f"unknown quadrature {kind!r}")

    def inside(self, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Point-in-body test (convex bodies only; both builders here are convex)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        inside = np.ones(pts.shape[0], dtype=bool)
        if self.dim == 2:
            for k in range(self.n_faces):
                a = self.vertices[self.faces[k, 0]]
                b = self.vertices[self.faces[k, 1]]
                e = b - a
                cross = e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])
                inside &= cross >= -tol
            return inside
        for k in range(self.n_faces):
            d = pts - self.centroids[k]
            inside &= d @ self.normals[k] <= tol
        return inside

    def polar_angles(self, axis: int = 0) -> np.ndarray:
        """Angle (rad) of every facet centroid measured from ``+axis`` at the centre."""
        d = self.centroids - self.center
        r = np.linalg.norm(d, axis=1)
        return np.arccos(np.clip(d[:, axis] / np.maximum(r, 1e-30), -1.0, 1.0))

    def closure_residual(self) -> float:
        """``|sum A n|`` relative to the total measure; zero for a closed surface."""
        proj = self.areas @ self.normals
        return float(np.linalg.norm(proj) / max(self.total_area, 1e-30))

    def summary(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "n_faces": self.n_faces,
            "area": self.total_area,
            "closure_relative": self.closure_residual(),
        }


def circle_facet_mesh(
    center: Sequence[float] = (0.0, 0.0),
    radius: float = 1.0,
    n_faces: int = 64,
    phase: float = 0.0,
) -> FacetMesh:
    """Regular polygon inscribed in a circle, oriented counter-clockwise."""
    if n_faces < 3:
        raise ValueError("a 2D facet mesh needs at least 3 faces")
    c = np.asarray(center, dtype=np.float64).reshape(2)
    ang = phase + 2.0 * math.pi * np.arange(n_faces) / n_faces
    verts = np.stack([c[0] + radius * np.cos(ang), c[1] + radius * np.sin(ang)], axis=1)
    faces = np.stack([np.arange(n_faces), (np.arange(n_faces) + 1) % n_faces], axis=1)
    a, b = verts[faces[:, 0]], verts[faces[:, 1]]
    centroids = 0.5 * (a + b)
    t = b - a
    length = np.linalg.norm(t, axis=1)
    normals = np.stack([t[:, 1], -t[:, 0]], axis=1) / length[:, None]
    flip = ((centroids - c) * normals).sum(axis=1) < 0.0
    normals[flip] *= -1.0
    return FacetMesh(verts, faces, centroids, normals, length, c)


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Unit icosahedron: vertices on the unit sphere, 20 triangular faces."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return verts, faces


def sphere_facet_mesh(
    center: Sequence[float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    subdivisions: int = 1,
) -> FacetMesh:
    """Icosphere: ``20 * 4**subdivisions`` triangles projected onto the sphere."""
    verts, faces = _icosahedron()
    for _ in range(int(subdivisions)):
        edge_mid: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []
        verts = list(verts)
        for tri in faces:
            mids = []
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                if key not in edge_mid:
                    m = 0.5 * (verts[a] + verts[b])
                    verts.append(m / np.linalg.norm(m))
                    edge_mid[key] = len(verts) - 1
                mids.append(edge_mid[key])
            m01, m12, m20 = mids
            new_faces.extend(
                [[tri[0], m01, m20], [tri[1], m12, m01], [tri[2], m20, m12], [m01, m12, m20]]
            )
        verts = np.asarray(verts, dtype=np.float64)
        verts /= np.linalg.norm(verts, axis=1)[:, None]
        faces = np.asarray(new_faces, dtype=np.int64)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    verts = verts * radius + c
    tris = verts[faces]
    centroids = tris.mean(axis=1)
    cr = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    two_area = np.linalg.norm(cr, axis=1)
    normals = cr / np.maximum(two_area, 1e-30)[:, None]
    flip = ((centroids - c) * normals).sum(axis=1) < 0.0
    normals = normals.copy()
    normals[flip] *= -1.0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return FacetMesh(verts, faces, centroids, normals, 0.5 * two_area, c)


# ---------------------------------------------------------------------------
# lattice field interpolation / wall reconstruction
# ---------------------------------------------------------------------------


def interp_lattice(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear / trilinear interpolation of ``field`` at Cartesian ``points``.

    ``field`` axes are ``(z, y, x)`` for 3D and ``(y, x)`` for 2D, while the
    columns of ``points`` are ``(x, y[, z])``.  Points outside the array are
    clamped to the boundary (the bodies under study sit far from the edges).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ndim = pts.shape[1]
    shape = field.shape[-ndim:]
    coords = []
    for d in range(ndim):
        ax = ndim - 1 - d
        coords.append(np.clip(pts[:, d], 0.0, float(shape[ax] - 1)))
    base = [np.floor(c).astype(np.int64) for c in coords]
    out = np.zeros(pts.shape[0], dtype=np.float64)
    for corner in np.ndindex(*([2] * ndim)):
        w = np.ones(pts.shape[0], dtype=np.float64)
        idx = []
        for d, c in enumerate(coords):
            ax = ndim - 1 - d
            i = np.clip(base[d] + corner[d], 0, shape[ax] - 1)
            frac = c - base[d]
            w = w * (frac if corner[d] else (1.0 - frac))
            idx.append(i)
        out += w * np.asarray(field[tuple(idx[::-1])], dtype=np.float64)
    return out


def _extrapolate_to_wall(values: np.ndarray, offsets: Sequence[float]) -> np.ndarray:
    """Lagrange extrapolation of ``values`` (N, K) sampled at ``offsets`` to d = 0."""
    d = np.asarray(offsets, dtype=np.float64)
    k = len(d)
    if values.shape[1] != k:
        raise ValueError("offset count mismatch")
    out = np.zeros(values.shape[0], dtype=np.float64)
    for i in range(k):
        w = 1.0
        for j in range(k):
            if i != j:
                w *= (0.0 - d[j]) / (d[i] - d[j])
        out += w * values[:, i]
    return out


def velocity_gradient_fields(velocity: Sequence[np.ndarray]):
    """Central-difference velocity gradients ``d u_i / d x_j`` on the lattice.

    Returns ``grad[i][j]`` with ``i`` the velocity component and ``j`` the
    coordinate direction (one-sided differences on the outermost layer).
    """
    grads: list[list[np.ndarray]] = []
    ndim = len(velocity)
    for comp in velocity:
        row = []
        for d in range(ndim):
            ax = ndim - 1 - d
            row.append(np.gradient(np.asarray(comp, dtype=np.float64), 1.0, axis=ax))
        grads.append(row)
    return grads


@dataclass
class SurfaceForceResult:
    """Per-facet and total surface loads produced by :func:`integrate_surface_force`."""

    mesh: FacetMesh
    points: np.ndarray
    weights: np.ndarray
    cp: np.ndarray
    pressure: np.ndarray
    force_facet: np.ndarray
    pressure_force_facet: np.ndarray
    viscous_force_facet: np.ndarray
    rho_ref: float
    u_ref: float
    ref_length: float
    ref_area: float
    facet_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def dim(self) -> int:
        return int(self.force_facet.shape[1])

    @property
    def total_force(self) -> np.ndarray:
        return self.force_facet.sum(axis=0)

    @property
    def pressure_force(self) -> np.ndarray:
        return self.pressure_force_facet.sum(axis=0)

    @property
    def viscous_force(self) -> np.ndarray:
        return self.viscous_force_facet.sum(axis=0)

    @property
    def drag(self) -> float:
        return float(self.total_force[0])

    @property
    def lift(self) -> float:
        return float(self.total_force[1]) if self.dim > 1 else 0.0

    def coefficients(self) -> dict:
        """Non-dimensionalise: ``Cd = Fx / (0.5 rho U^2 A_ref)``."""
        dyn = 0.5 * self.rho_ref * self.u_ref ** 2 * self.ref_area
        return {
            "cd": float(self.total_force[0] / dyn),
            "cl": float(self.total_force[1] / dyn) if self.dim > 1 else 0.0,
            "cd_pressure": float(self.pressure_force[0] / dyn),
            "cd_friction": float(self.viscous_force[0] / dyn),
            "force": [float(v) for v in self.total_force],
            "force_per_length": [
                float(v / max(self.ref_length, 1e-30)) for v in self.total_force
            ],
            "dynamic_reference": float(dyn),
            "cp_min": float(np.min(self.cp)),
            "cp_max": float(np.max(self.cp)),
        }


def integrate_surface_force(
    mesh: FacetMesh,
    *,
    rho: np.ndarray,
    velocity: Sequence[np.ndarray],
    nu: float,
    rho_ref: float = 1.0,
    u_ref: float = 1.0,
    ref_length: float | None = None,
    ref_area: float | None = None,
    quadrature: str = "auto",
    wall_offsets: Sequence[float] = (1.5, 2.5, 3.5),
    pressure_mode: str = "bernoulli",
) -> SurfaceForceResult:
    """Spherical/facet force method: integrate pressure and viscous traction.

    Parameters
    ----------
    rho, velocity
        Field on the lattice; ``velocity`` holds components in ``x, y[, z]``.
    nu
        Kinematic viscosity in lattice units for the viscous traction
        (``mu = nu rho_ref``).  Pass ``0.0`` for the inviscid (potential-flow)
        check, which then isolates the pressure integral.
    pressure_mode
        ``"bernoulli"`` (default) reconstructs ``p = rho cs^2`` from the density
        field, which is what an LBM solution carries.  ``"incompressible"`` uses
        ``p = p_inf + 0.5 rho_ref (u_ref^2 - |u|^2)``, for velocity-only inputs.
    """
    pts, w, normals = mesh.quadrature(quadrature)
    nq = pts.shape[0]
    ndim = mesh.dim
    qfacet = np.repeat(np.arange(mesh.n_faces), nq // mesh.n_faces)
    offsets = np.asarray(wall_offsets, dtype=np.float64)

    # pressure at the wall: sample along the outward normal, extrapolate to d = 0
    rho_s = np.stack(
        [interp_lattice(np.asarray(rho, dtype=np.float64), pts + d * normals) for d in offsets],
        axis=1,
    )
    rho_wall = _extrapolate_to_wall(rho_s, offsets)
    p_inf = CS2 * rho_ref
    if pressure_mode == "bernoulli":
        p_wall = CS2 * rho_wall
    elif pressure_mode == "incompressible":
        usq = np.zeros(nq)
        for comp in velocity:
            usq += interp_lattice(np.asarray(comp, dtype=np.float64), pts) ** 2
        p_wall = p_inf + 0.5 * rho_ref * (u_ref ** 2 - usq)
        rho_wall = p_wall / CS2
    else:
        raise ValueError(f"unknown pressure_mode {pressure_mode!r}")

    # viscous traction: mu (grad u + grad u^T) . n at the wall
    mu = float(nu) * float(rho_ref)
    visc = np.zeros((nq, ndim), dtype=np.float64)
    if mu != 0.0:
        grads = velocity_gradient_fields(velocity)
        wall_grad = [[None] * ndim for _ in range(ndim)]
        for i in range(ndim):
            for j in range(ndim):
                samples = np.stack(
                    [interp_lattice(grads[i][j], pts + d * normals) for d in offsets], axis=1
                )
                wall_grad[i][j] = _extrapolate_to_wall(samples, offsets)
        for i in range(ndim):
            t_i = np.zeros(nq)
            for j in range(ndim):
                t_i = t_i + (wall_grad[i][j] + wall_grad[j][i]) * normals[:, j]
            visc[:, i] = mu * t_i

    traction = visc - p_wall[:, None] * normals
    force_pts = traction * w[:, None]
    dyn = 0.5 * rho_ref * u_ref ** 2
    cp = (p_wall - p_inf) / max(dyn, 1e-300)

    force_facet = np.zeros((mesh.n_faces, ndim))
    pfacet = np.zeros((mesh.n_faces, ndim))
    vfacet = np.zeros((mesh.n_faces, ndim))
    np.add.at(force_facet, qfacet, force_pts)
    np.add.at(pfacet, qfacet, (-p_wall[:, None] * normals) * w[:, None])
    np.add.at(vfacet, qfacet, visc * w[:, None])

    length = float(ref_length if ref_length is not None else mesh.total_area)
    area = float(ref_area if ref_area is not None else mesh.total_area)
    return SurfaceForceResult(
        mesh=mesh,
        points=pts,
        weights=w,
        cp=cp,
        pressure=p_wall,
        force_facet=force_facet,
        pressure_force_facet=pfacet,
        viscous_force_facet=vfacet,
        rho_ref=float(rho_ref),
        u_ref=float(u_ref),
        ref_length=length,
        ref_area=area,
        facet_index=qfacet,
    )


def cp_at_angles(
    result: SurfaceForceResult,
    angles_deg: Sequence[float],
    *,
    axis: int = 0,
    mode: str = "quadrature",
):
    """Interpolate the surface Cp distribution onto requested polar angles.

    ``mode="quadrature"`` (default) uses every quadrature point with its own
    polar angle, which is the point-wise pressure distribution the integrator
    evaluates; ``mode="facet"`` averages each facet and bins it at the facet
    centroid, i.e. the coarse-graining a load table would use.  For coarse
    facets the two differ by O(dtheta^2) purely because a facet mean is
    compared against a point value, not because of an integration error.
    """
    mesh = result.mesh
    if mode == "quadrature":
        d = result.points - mesh.center
        r = np.linalg.norm(d, axis=1)
        theta = np.degrees(np.arccos(np.clip(d[:, axis] / np.maximum(r, 1e-30), -1.0, 1.0)))
        cp_vals = np.asarray(result.cp, dtype=np.float64)
    elif mode == "facet":
        theta = np.degrees(mesh.polar_angles(axis=axis))
        cp_vals = result.cp.reshape(mesh.n_faces, -1).mean(axis=1)
    else:
        raise ValueError(f"unknown cp mode {mode!r}")
    order = np.argsort(theta)
    th_sorted = theta[order]
    cp_sorted = cp_vals[order]
    th_ext = np.concatenate([th_sorted - 360.0, th_sorted, th_sorted + 360.0])
    cp_ext = np.concatenate([cp_sorted, cp_sorted, cp_sorted])
    out = np.interp(np.asarray(angles_deg, dtype=np.float64), th_ext, cp_ext)
    for k, a in enumerate(angles_deg):
        j = int(np.argmin(np.abs(th_sorted - a)))
        if abs(th_sorted[j] - a) < 1e-6:
            out[k] = cp_sorted[j]
    return {"theta_deg": np.asarray(angles_deg, dtype=np.float64), "cp": out}


# ---------------------------------------------------------------------------
# analytic potential-flow reference fields
# ---------------------------------------------------------------------------


def potential_flow_grid(
    shape: Sequence[int],
    *,
    center: Sequence[float],
    radius: float,
    u_inf: float,
    rho_inf: float = 1.0,
    body: str = "cylinder",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Lattice-sampled potential flow past a cylinder (2D) or a sphere (3D).

    The velocity potential is evaluated on the same lattice layout an LBM solver
    uses (``(y, x)`` / ``(z, y, x)``) and the density follows from Bernoulli in
    lattice units, ``rho = rho_inf + 1.5 (u_inf^2 - |u|^2)``, so that
    ``p = rho cs^2`` matches the LBM pressure the integrator reads.
    """
    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError("only 2D and 3D potential flow are implemented")
    grids = np.meshgrid(*[np.arange(float(n)) for n in shape], indexing="ij")[::-1]
    d = [g - c0 for g, c0 in zip(grids, center)]
    body = body.lower()
    if ndim == 2:
        if body != "cylinder":
            raise ValueError("2D potential flow is the cylinder solution")
        dx, dy = d
        r2 = np.maximum(dx * dx + dy * dy, 1e-12)
        a2 = float(radius) ** 2
        ux = u_inf * (1.0 - a2 * (dx * dx - dy * dy) / (r2 * r2))
        uy = -2.0 * u_inf * a2 * dx * dy / (r2 * r2)
        vel = [ux, uy]
    else:
        if body != "sphere":
            raise ValueError("3D potential flow is the sphere solution")
        dx, dy, dz = d
        r2 = np.maximum(dx * dx + dy * dy + dz * dz, 1e-12)
        r = np.sqrt(r2)
        a3 = float(radius) ** 3
        rho_perp = np.sqrt(dy * dy + dz * dz)
        cos_t = dx / r
        sin_t = rho_perp / r
        u_r = u_inf * (1.0 - a3 / r2 ** 1.5) * cos_t
        u_th = -u_inf * (1.0 + 0.5 * a3 / r2 ** 1.5) * sin_t
        ux = u_r * cos_t - u_th * sin_t
        u_perp = u_r * sin_t + u_th * cos_t
        safe = np.maximum(rho_perp, 1e-30)
        vel = [ux, u_perp * dy / safe, u_perp * dz / safe]
    usq = np.zeros(shape, dtype=np.float64)
    for v in vel:
        usq += np.asarray(v, dtype=np.float64) ** 2
    rho = rho_inf + 0.5 * (u_inf ** 2 - usq) / CS2
    return rho, [np.asarray(v, dtype=np.float64) for v in vel]


def potential_flow_cp_theory(theta_deg: np.ndarray, body: str = "sphere") -> np.ndarray:
    """Analytic surface Cp of the potential-flow limit.

    Cylinder: ``Cp = 1 - 4 sin^2(th)``; sphere: ``Cp = 1 - (9/4) sin^2(th)``.
    """
    s2 = np.sin(np.radians(np.asarray(theta_deg, dtype=np.float64))) ** 2
    coeff = 4.0 if body.lower() == "cylinder" else 2.25
    return 1.0 - coeff * s2


def sphere_cp_study(
    *,
    radius: float = 20.0,
    subdivisions: int = 2,
    u_inf: float = 0.05,
    pad: float = 4.0,
    angles_deg: Sequence[float] = SPHERE_CP_ANGLES_DEG,
    quadrature: str = "auto",
) -> dict[str, Any]:
    """Verify the SFM against potential-flow Cp(theta) on a lattice-sampled field."""
    n = int(round(2.0 * (radius + pad))) + 1
    center = (radius + pad, radius + pad, radius + pad)
    rho, vel = potential_flow_grid(
        (n, n, n), center=center, radius=radius, u_inf=u_inf, body="sphere"
    )
    mesh = sphere_facet_mesh(center=center, radius=radius, subdivisions=subdivisions)
    res = integrate_surface_force(
        mesh, rho=rho, velocity=vel, nu=0.0, rho_ref=1.0, u_ref=u_inf,
        ref_area=math.pi * radius ** 2, ref_length=radius, quadrature=quadrature,
    )
    prof = cp_at_angles(res, angles_deg, axis=0)
    theory = potential_flow_cp_theory(prof["theta_deg"], "sphere")
    err = prof["cp"] - theory
    denom = np.maximum(np.abs(theory), 1e-12)
    return {
        "radius": radius,
        "grid": n,
        "n_faces": mesh.n_faces,
        "quadrature": quadrature,
        "u_inf": u_inf,
        "theta_deg": [float(v) for v in prof["theta_deg"]],
        "cp": [float(v) for v in prof["cp"]],
        "cp_theory": [float(v) for v in theory],
        "abs_error": [float(v) for v in err],
        "rel_error": [float(v) for v in err / denom],
        "max_abs_error": float(np.max(np.abs(err))),
        "max_rel_error": float(np.max(np.abs(err / denom))),
        "cd_potential": float(res.coefficients()["cd"]),
        "closure_relative": mesh.closure_residual(),
    }


# ---------------------------------------------------------------------------
# mass balance
# ---------------------------------------------------------------------------


def mass_flow_profile(rho: np.ndarray, ux: np.ndarray) -> np.ndarray:
    """Streamwise mass flux ``sum_y rho ux`` per column."""
    return (np.asarray(rho, dtype=np.float64) * np.asarray(ux, dtype=np.float64)).sum(axis=0)


def mass_balance(rho: np.ndarray, ux: np.ndarray, *, inlet_col: int = 3, outlet_col: int = -4,
                 skip: int = 8) -> dict[str, Any]:
    """Inflow/outflow mass balance of the channel (relative imbalance)."""
    prof = mass_flow_profile(rho, ux)
    half = max(1, skip // 2)

    def _window(col: int) -> float:
        c = col if col >= 0 else prof.size + col
        return float(np.mean(prof[max(0, c - half): c + half + 1]))

    inlet = _window(inlet_col)
    outlet = _window(outlet_col)
    return {
        "inlet_flux": inlet,
        "outlet_flux": outlet,
        "abs_imbalance": abs(inlet - outlet),
        "rel_imbalance": abs(inlet - outlet) / max(abs(inlet), 1e-30),
        "profile": prof,
    }

# ---------------------------------------------------------------------------
# D2Q9 driver with Bouzidi bounce-back on the facet geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CylinderCaseConfig:
    """Resolution and flow settings of one facet-cylinder solve."""

    diameter: int = 32
    reynolds: float = 100.0
    lattice_speed: float = 0.1
    n_faces: int = 64
    upstream_diameters: float = 6.0
    downstream_diameters: float = 14.0
    half_height_diameters: float = 8.0
    steps_per_diameter: float = 400.0
    transient_diameters: float = 20.0
    steady_fraction: float = 0.4
    ramp_advective_times: float = 2.0
    sample_every: int = 20
    collision: str = "trt"
    geometry: str = "facet"
    sponge_width: int = 16
    sponge_strength: float = 0.2
    outlet_density_mode: str = "unit"

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def nx(self) -> int:
        return int(round(self.diameter * (self.upstream_diameters + self.downstream_diameters))) + 1

    @property
    def ny(self) -> int:
        return int(round(2.0 * self.half_height_diameters * self.diameter)) + 2

    @property
    def cx(self) -> float:
        return self.upstream_diameters * self.diameter

    @property
    def cy(self) -> float:
        return 0.5 * (self.ny - 1)

    @property
    def nu(self) -> float:
        return self.lattice_speed * self.diameter / self.reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def omega_minus(self) -> float:
        """TRT counter-relaxation rate with the magic parameter Lambda = 3/16."""
        return 8.0 * (2.0 - self.omega) / (8.0 - self.omega)

    @property
    def steps(self) -> int:
        return int(round(self.steps_per_diameter * self.diameter))

    @property
    def transient_step(self) -> int:
        return int(round(self.transient_diameters * self.diameter / self.lattice_speed))

    @property
    def ramp_steps(self) -> int:
        return max(1, int(round(self.ramp_advective_times * self.diameter / self.lattice_speed)))

    @property
    def steady_window(self) -> int:
        return max(1, int(round(self.steady_fraction * self.steps)))

    @property
    def dynamic_pressure_force(self) -> float:
        """``0.5 rho U^2 D`` in lattice units (rho = 1)."""
        return 0.5 * self.lattice_speed ** 2 * self.diameter

    def mesh(self) -> FacetMesh:
        """Facet surface of the obstacle (the authoritative geometry)."""
        if self.geometry == "facet":
            return circle_facet_mesh((self.cx, self.cy), self.radius, self.n_faces)
        if self.geometry == "facet_phase":
            return circle_facet_mesh(
                (self.cx, self.cy), self.radius, self.n_faces, phase=math.pi / self.n_faces
            )
        raise ValueError(f"unknown geometry {self.geometry!r}")

    def describe(self) -> dict[str, Any]:
        return {
            "diameter": self.diameter,
            "reynolds": self.reynolds,
            "lattice_speed": self.lattice_speed,
            "n_faces": self.n_faces,
            "geometry": self.geometry,
            "collision": self.collision,
            "nx": self.nx,
            "ny": self.ny,
            "nu": self.nu,
            "tau": self.tau,
            "upstream_diameters": self.upstream_diameters,
            "downstream_diameters": self.downstream_diameters,
            "half_height_diameters": self.half_height_diameters,
            "blockage": 1.0 / (2.0 * self.half_height_diameters),
            "outlet_density_mode": self.outlet_density_mode,
            "sponge_width": self.sponge_width,
            "sponge_strength": self.sponge_strength,
        }

    def validate(self) -> None:
        if self.diameter < 8:
            raise ValueError("diameter must be >= 8 lattice nodes")
        if not 0.0 < self.lattice_speed <= 0.2:
            raise ValueError("lattice_speed must lie in (0, 0.2]")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        if self.collision not in {"bgk", "trt"}:
            raise ValueError("collision must be 'bgk' or 'trt'")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")
        if self.n_faces < 8:
            raise ValueError("n_faces must be >= 8")


@dataclass
class CutLinks:
    """Cut links of the facet surface, in flat-index form."""

    node: np.ndarray
    direction: np.ndarray
    q: np.ndarray
    st1: np.ndarray
    st1_ok: np.ndarray
    st2: np.ndarray
    st2_ok: np.ndarray
    solid: np.ndarray
    solid_2d: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cx(self) -> np.ndarray:
        return CX_ARR[self.direction]

    @property
    def cy(self) -> np.ndarray:
        return CY_ARR[self.direction]

    @property
    def opp(self) -> np.ndarray:
        return OPP_ARR[self.direction]


def _ray_distance_to_facets(mesh: FacetMesh, origins: np.ndarray, direction: np.ndarray):
    """Distance (in units of ``direction``) from ``origins`` to the facet surface."""
    verts = mesh.vertices
    edges = np.roll(verts, -1, axis=0) - verts
    best = np.full(origins.shape[0], np.inf)
    face_of = np.full(origins.shape[0], -1, dtype=np.int64)
    d = np.asarray(direction, dtype=np.float64)
    for k in range(mesh.n_faces):
        e = edges[k]
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(denom) < 1e-14:
            continue
        ao = verts[k] - origins
        t = (ao[:, 0] * e[1] - ao[:, 1] * e[0]) / denom
        s = (ao[:, 0] * d[1] - ao[:, 1] * d[0]) / denom
        hit = (t > 1e-9) & (t <= 1.0) & (s >= -1e-12) & (s <= 1.0 + 1e-12) & (t < best)
        best[hit] = t[hit]
        face_of[hit] = k
    return best, face_of


def build_cut_links(mesh: FacetMesh, ny: int, nx: int) -> CutLinks:
    """Enumerate cut links between fluid nodes and the faceted body.

    ``q`` is the fraction of the link at which the facet surface sits, measured
    from the fluid node (``q = 1/2`` is the half-way case).  The solid *mask* is
    centre based on the same facet mesh, so the geometry the flow sees and the
    geometry the force integrator sums over are identical.
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)], axis=1)
    solid_2d = mesh.inside(pts).reshape(ny, nx)
    fluid_2d = ~solid_2d

    pad = 4
    # the padding is *not* solid: only nodes of the faceted body may create cut
    # links, otherwise the domain border would masquerade as a wall
    sp = np.zeros((ny + 2 * pad, nx + 2 * pad), dtype=bool)
    sp[pad:pad + ny, pad:pad + nx] = solid_2d
    fp = np.zeros((ny + 2 * pad, nx + 2 * pad), dtype=bool)
    fp[pad:pad + ny, pad:pad + nx] = fluid_2d

    def shifted(a, ex, ey):
        return a[pad + ey:pad + ey + ny, pad + ex:pad + ex + nx]

    nodes, dirs, qs, s1ok, s1i, s2ok, s2i = [], [], [], [], [], [], []
    stats: dict[str, Any] = {"fallbacks": 0, "links_by_direction": {}}
    for i in range(1, 9):
        ex, ey = int(CX[i]), int(CY[i])
        cut = fluid_2d & shifted(sp, ex, ey)
        if not cut.any():
            continue
        iy, ix = np.nonzero(cut)
        origins = np.stack([ix.astype(np.float64), iy.astype(np.float64)], axis=1)
        t, _ = _ray_distance_to_facets(mesh, origins, np.array([ex, ey], dtype=np.float64))
        bad = ~np.isfinite(t) | (t <= 1e-9)
        stats["fallbacks"] += int(bad.sum())
        t = np.where(bad, 0.5, np.clip(t, 1e-6, 1.0))
        ok1 = shifted(fp, -ex, -ey)[iy, ix]
        ok2 = shifted(fp, -2 * ex, -2 * ey)[iy, ix]
        flat = iy * nx + ix
        s1 = (iy - ey) * nx + (ix - ex)
        s2 = (iy - 2 * ey) * nx + (ix - 2 * ex)
        nodes.append(flat)
        dirs.append(np.full(flat.shape, i, dtype=np.int64))
        qs.append(t)
        s1ok.append(ok1)
        s1i.append(np.where(ok1, s1, flat))
        s2ok.append(ok2)
        s2i.append(np.where(ok2, s2, np.where(ok1, s1, flat)))
        stats["links_by_direction"][f"dir{i}"] = int(flat.size)

    out = CutLinks(
        node=np.concatenate(nodes),
        direction=np.concatenate(dirs),
        q=np.concatenate(qs),
        st1=np.concatenate(s1i),
        st1_ok=np.concatenate(s1ok),
        st2=np.concatenate(s2i),
        st2_ok=np.concatenate(s2ok),
        solid=np.nonzero(solid_2d.ravel())[0],
        solid_2d=solid_2d,
    )
    q = out.q
    out.meta = {
        "links": int(out.node.size),
        "q_min": float(q.min()) if q.size else 0.0,
        "q_max": float(q.max()) if q.size else 0.0,
        "q_mean": float(q.mean()) if q.size else 0.0,
        "low_branch_fraction": float((q < 0.5).mean()) if q.size else 0.0,
        "solid_nodes": int(solid_2d.sum()),
        "solid_area_over_facet_area": float(solid_2d.sum()) / max(mesh.total_area, 1e-30),
        **stats,
    }
    return out


def reflected_populations(fpost: np.ndarray, links: CutLinks, *, quadratic: bool = True):
    """Bouzidi reflected population for every cut link.

    The linear/quadratic weights are the ones verified against the Couette
    solution in :mod:`tensorlbm.curved_wall_force_scaling`.
    """
    direction = links.direction
    opp = links.opp
    node = links.node
    q = links.q
    o = fpost[direction, node]
    fi_m1 = np.where(links.st1_ok, fpost[direction, links.st1], o)
    fi_m2 = np.where(links.st2_ok, fpost[direction, links.st2], fi_m1)
    foi_x = fpost[opp, node]
    foi_m1 = np.where(links.st1_ok, fpost[opp, links.st1], foi_x)
    low = q < 0.5
    lin = np.where(
        low,
        2.0 * q * o + (1.0 - 2.0 * q) * fi_m1,
        o / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * foi_x,
    )
    if not quadratic:
        return lin
    quad_low = (
        q * (2.0 * q + 1.0) * o
        + (1.0 + 2.0 * q) * (1.0 - 2.0 * q) * fi_m1
        - q * (1.0 - 2.0 * q) * fi_m2
    )
    quad_high = (
        o / (q * (2.0 * q + 1.0))
        + (2.0 * q - 1.0) / q * foi_x
        - (2.0 * q - 1.0) / (2.0 * q + 1.0) * foi_m1
    )
    quad = np.where(low, quad_low, quad_high)
    quad_ok = links.st1_ok & np.where(low, links.st2_ok, True)
    return np.where(quad_ok, quad, lin)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium populations, shape ``(9, ...)``."""
    usq = ux * ux + uy * uy
    cu = CX_ARR[:, None, None] * ux[None] + CY_ARR[:, None, None] * uy[None]
    return W_ARR[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def macroscopic_into(f, rho, ux, uy, scratch) -> None:
    """In-place version of :func:`macroscopic` (buffers supplied by the caller)."""
    np.add.reduce(f, axis=0, out=rho)
    np.maximum(rho, 1e-12, out=scratch["safe"])
    np.multiply(f, CX_ARR[:, None, None], out=scratch["tmp9"])
    np.add.reduce(scratch["tmp9"], axis=0, out=ux)
    ux /= scratch["safe"]
    np.multiply(f, CY_ARR[:, None, None], out=scratch["tmp9"])
    np.add.reduce(scratch["tmp9"], axis=0, out=uy)
    uy /= scratch["safe"]


def _equilibrium_into(out, rho, ux, uy, scratch) -> None:
    """In-place D2Q9 equilibrium, ``out = W rho (1 + 3 cu + 4.5 cu^2 - 1.5 u^2)``."""
    cu, sq, usq, tmp = scratch["cu"], scratch["sq"], scratch["usq"], scratch["tmp"]
    np.multiply(ux, ux, out=usq)
    np.multiply(uy, uy, out=tmp)
    usq += tmp
    usq *= -1.5
    np.multiply(CX_ARR[:, None, None], ux[None], out=cu)
    np.multiply(CY_ARR[:, None, None], uy[None], out=sq)
    cu += sq
    np.multiply(cu, cu, out=out)
    out *= 4.5
    cu *= 3.0
    out += cu
    out += 1.0
    out += usq
    out *= W_ARR[:, None, None]
    out *= rho[None]


def macroscopic(f: np.ndarray):
    """Density and velocity from D2Q9 populations."""
    rho = f.sum(axis=0)
    rho_safe = np.maximum(rho, 1e-12)
    ux = (f * CX_ARR[:, None, None]).sum(axis=0) / rho_safe
    uy = (f * CY_ARR[:, None, None]).sum(axis=0) / rho_safe
    return rho, ux, uy


def _stream(fnext: np.ndarray, fpost: np.ndarray) -> None:
    """Pull-scheme streaming; outer rows/columns are overwritten by the BCs."""
    fnext[0] = fpost[0]
    fnext[1, :, 1:] = fpost[1, :, :-1]
    fnext[2, 1:, :] = fpost[2, :-1, :]
    fnext[3, :, :-1] = fpost[3, :, 1:]
    fnext[4, :-1, :] = fpost[4, 1:, :]
    fnext[5, 1:, 1:] = fpost[5, :-1, :-1]
    fnext[6, 1:, :-1] = fpost[6, :-1, 1:]
    fnext[7, :-1, :-1] = fpost[7, 1:, 1:]
    fnext[8, :-1, 1:] = fpost[8, 1:, :-1]


def run_cylinder_case(
    config: CylinderCaseConfig,
    *,
    max_steps: int | None = None,
    store_fields: bool = False,
    label: str = "",
) -> dict[str, Any]:
    """Run one facet-cylinder solve and record facet/MEM force histories.

    Returns a JSON-friendly dict with the steady-window averages of the facet
    (SFM) drag and of the momentum-exchange reference, the pressure/friction
    split, the Cp distribution, the domain mass balance and run statistics.
    """
    config.validate()
    nx, ny = config.nx, config.ny
    mesh = config.mesh()
    links = build_cut_links(mesh, ny, nx)
    solid = links.solid
    opp = links.opp
    node = links.node

    shape = (ny, nx)
    # start from the free stream (not from rest): the wake then develops within
    # about one flow-through time instead of filling the whole channel first
    ones = np.ones(shape)
    f = equilibrium(ones, np.full(shape, float(config.lattice_speed)), np.zeros(shape))
    fpost = np.empty_like(f)
    fnext = np.empty_like(f)
    feq = np.empty_like(f)
    fopp = np.empty_like(f)
    foppeq = np.empty_like(f)
    tmp1 = np.empty_like(f)
    tmp2 = np.empty_like(f)
    rho = np.empty(shape)
    ux = np.empty(shape)
    uy = np.empty(shape)
    rho_n = np.empty(shape)
    ux_n = np.empty(shape)
    uy_n = np.empty(shape)
    feq_n = np.empty_like(f)
    scratch = {
        "cu": np.empty_like(f),
        "sq": np.empty_like(f),
        "usq": np.empty(shape),
        "tmp": np.empty(shape),
        "tmp9": np.empty_like(f),
        "safe": np.empty(shape),
    }

    n_steps = int(config.steps if max_steps is None else max_steps)
    transient = min(config.transient_step, max(1, n_steps - 1))
    steady_window = min(config.steady_window, n_steps - transient)
    steady_window = max(1, steady_window)
    wp, wm = config.omega, config.omega_minus
    a_trt = 0.5 * (wp + wm)
    b_trt = 0.5 * (wp - wm)
    u_max = float(config.lattice_speed)
    ramp = max(1, config.ramp_steps)
    dyn = config.dynamic_pressure_force

    sponge = None
    if config.sponge_width > 0 and config.sponge_strength > 0.0:
        width = min(int(config.sponge_width), max(1, nx // 8))
        sigma_x = np.zeros(nx)
        for offset in range(width):
            value = config.sponge_strength * (1.0 - offset / width) ** 2
            sigma_x[offset] = value
            sigma_x[nx - 1 - offset] = value
        sponge = sigma_x.reshape(1, 1, nx)

    cd_facet: list[float] = []
    cd_pressure: list[float] = []
    cd_friction: list[float] = []
    cd_mem: list[float] = []
    cl_facet: list[float] = []
    sample_steps: list[int] = []
    cp_snapshot: dict[str, Any] | None = None
    rho_mean: list[float] = []
    mass_rel: list[float] = []
    started = _time.perf_counter()
    sample_every = max(1, int(config.sample_every))

    for step in range(1, n_steps + 1):
        # ramp the inlet over the first few advective times
        ramp_factor = 0.5 * (1.0 - math.cos(math.pi * min(1.0, step / ramp)))
        u_in = u_max * ramp_factor

        macroscopic_into(f, rho, ux, uy, scratch)
        _equilibrium_into(feq, rho, ux, uy, scratch)
        if config.collision == "trt":
            # TRT as f - a (f - feq) - b (f_opp - feq_opp); the symmetric /
            # antisymmetric split is never materialised.
            np.subtract(f, feq, out=tmp1)
            np.take(f, OPP_ARR, axis=0, out=fopp)
            np.take(feq, OPP_ARR, axis=0, out=foppeq)
            np.subtract(fopp, foppeq, out=tmp2)
            tmp1 *= a_trt
            tmp2 *= b_trt
            tmp1 += tmp2
            np.subtract(f, tmp1, out=fpost)
        else:
            np.subtract(f, feq, out=tmp1)
            tmp1 *= wp
            np.subtract(f, tmp1, out=fpost)
        if sponge is not None:
            # absorbing layers at both ends: relax the non-equilibrium part so the
            # inflow/outflow do not pile up pressure in a confined channel
            np.subtract(fpost, feq, out=tmp1)
            tmp1 *= sponge
            np.subtract(fpost, tmp1, out=fpost)
        fpost_v = fpost.reshape(9, -1)
        fpost_v[:, solid] = 0.0

        _stream(fnext, fpost)

        r = reflected_populations(fpost_v, links)
        o = fpost_v[links.direction, node]
        fnext.reshape(9, -1)[opp, node] = r

        # free-slip lateral walls (specular about y = 1 / y = ny - 2)
        fnext[2, 1, :] = fnext[4, 1, :]
        fnext[5, 1, :] = fnext[8, 1, :]
        fnext[6, 1, :] = fnext[7, 1, :]
        fnext[4, ny - 2, :] = fnext[2, ny - 2, :]
        fnext[7, ny - 2, :] = fnext[5, ny - 2, :]
        fnext[8, ny - 2, :] = fnext[6, ny - 2, :]

        # inlet: velocity Dirichlet by non-equilibrium extrapolation
        macroscopic_into(fnext, rho_n, ux_n, uy_n, scratch)
        _equilibrium_into(feq_n, rho_n, ux_n, uy_n, scratch)
        rho_in = rho_n[:, 1]
        feq_in = equilibrium(rho_in, np.full_like(rho_in, u_in), np.zeros_like(rho_in))
        for d in (1, 5, 8):
            fnext[d, :, 0] = feq_in[d] + (fnext[d, :, 1] - feq_n[d, :, 1])

        # outlet: zero-gradient velocity; the density pins the pressure level
        if config.outlet_density_mode == "unit":
            rho_out = np.ones_like(rho_n[:, -2])
        else:
            rho_out = rho_n[:, -2]
        feq_out = equilibrium(rho_out, ux_n[:, -2], uy_n[:, -2])
        for d in (3, 6, 7):
            fnext[d, :, -1] = feq_out[d] + (fnext[d, :, -2] - feq_n[d, :, -2])

        fnext[:, 0, :] = 0.0
        fnext[:, -1, :] = 0.0

        f, fnext = fnext, f
        mem_force = float(np.sum(((o + r) * links.cx))) / dyn

        if step % sample_every == 0 or step == n_steps:
            rho_s, ux_s, uy_s = macroscopic(f)
            res = integrate_surface_force(
                mesh,
                rho=rho_s,
                velocity=[ux_s, uy_s],
                nu=config.nu,
                rho_ref=1.0,
                u_ref=u_max,
                ref_length=config.diameter,
                ref_area=config.diameter,
            )
            coeff = res.coefficients()
            cd_facet.append(coeff["cd"])
            cd_pressure.append(coeff["cd_pressure"])
            cd_friction.append(coeff["cd_friction"])
            cl_facet.append(coeff["cl"])
            cd_mem.append(mem_force)
            sample_steps.append(step)
            rho_mean.append(float(rho_s[~links.solid_2d].mean()))
            mass_rel.append(
                mass_balance(
                    rho_s, ux_s,
                    inlet_col=config.sponge_width + 8,
                    outlet_col=nx - 1 - config.sponge_width - 8,
                )["rel_imbalance"]
            )
            if step == n_steps or step >= n_steps - sample_every:
                prof = cp_at_angles(res, np.arange(0.0, 181.0, 15.0), axis=0)
                cp_snapshot = {
                    "theta_deg": [float(v) for v in prof["theta_deg"]],
                    "cp": [float(v) for v in prof["cp"]],
                    "cd": coeff["cd"],
                    "cd_pressure": coeff["cd_pressure"],
                    "cd_friction": coeff["cd_friction"],
                    "cl": coeff["cl"],
                }

    elapsed = _time.perf_counter() - started
    window = min(steady_window // sample_every + 1, len(cd_facet))
    tail = slice(len(cd_facet) - window, len(cd_facet))

    def _mean(a):
        return float(np.mean(np.asarray(a)[tail]))

    def _std(a):
        return float(np.std(np.asarray(a)[tail]))

    rho_final, ux_final, uy_final = macroscopic(f)
    # measure the balance outside the absorbing layers
    col_in = config.sponge_width + 8
    col_out = nx - 1 - col_in
    bal = mass_balance(rho_final, ux_final, inlet_col=col_in, outlet_col=col_out)
    mean_rho = _mean(rho_mean)

    out: dict[str, Any] = {
        "label": label or f"N{config.diameter}-Re{config.reynolds:g}-F{config.n_faces}",
        "config": config.describe(),
        "mesh": mesh.summary(),
        "boundary": links.meta,
        "steps": n_steps,
        "steady_window_samples": window,
        "sample_steps": sample_steps,
        "cd_sf": _mean(cd_facet),
        "cd_sf_std": _std(cd_facet),
        "cd_sf_pressure": _mean(cd_pressure),
        "cd_sf_friction": _mean(cd_friction),
        "cl_sf": _mean(cl_facet),
        "cl_sf_std": _std(cl_facet),
        "cd_mem": _mean(cd_mem),
        "cd_mem_std": _std(cd_mem),
        "cd_sf_history": {"step": list(sample_steps), "value": list(cd_facet)},
        "cd_mem_history": {"step": list(sample_steps), "value": list(cd_mem)},
        "mean_density": mean_rho,
        "mass_imbalance_steady": float(np.mean(np.asarray(mass_rel)[tail])),
        "mass_imbalance_max": float(np.max(np.asarray(mass_rel)[tail])),
        "cd_sf_density_norm": _mean(cd_facet) / mean_rho if mean_rho else float("nan"),
        "mass_balance": {k: v for k, v in bal.items() if k != "profile"},
        "cp": cp_snapshot,
        "dynamic_pressure_force": dyn,
        "wall_seconds": elapsed,
        "cell_steps_per_second": nx * ny * n_steps / max(elapsed, 1e-9),
    }
    if store_fields:
        out["fields"] = {"rho": rho_final, "ux": ux_final, "uy": uy_final}
    return out


# ---------------------------------------------------------------------------
# periodic body-force driver (robust configuration for the drag benchmark)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodicCylinderConfig:
    """Periodic channel driven by a uniform body force (constant-Reinolds setup).

    The channel is periodic in x, free-slip in y and the flow is driven by a Guo
    body force whose magnitude is adapted until the mean streamwise velocity hits
    ``target_speed``.  There is no inlet/outlet pair, so no pressure feedback loop
    can build up: the mass flux adjusts itself and the momentum balance of the
    whole domain provides an independent drag estimate (``cd_global``).
    """

    diameter: int = 16
    reynolds: float = 100.0
    n_faces: int = 48
    target_speed: float = 0.08
    length_diameters: float = 12.0
    half_height_diameters: float = 8.0
    steps_per_diameter: float = 500.0
    transient_fraction: float = 0.55
    steady_fraction: float = 0.45
    sample_every: int = 20
    control_every: int = 50
    collision: str = "trt"
    body_force: float = 1.0e-6

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def nx(self) -> int:
        return int(round(self.diameter * self.length_diameters))

    @property
    def ny(self) -> int:
        return int(round(2.0 * self.half_height_diameters * self.diameter)) + 2

    @property
    def cx(self) -> float:
        return 0.5 * self.diameter * self.length_diameters

    @property
    def cy(self) -> float:
        return 0.5 * (self.ny - 1)

    @property
    def nu(self) -> float:
        return self.target_speed * self.diameter / self.reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def omega_minus(self) -> float:
        return 8.0 * (2.0 - self.omega) / (8.0 - self.omega)

    @property
    def steps(self) -> int:
        return int(round(self.steps_per_diameter * self.diameter))

    def mesh(self) -> FacetMesh:
        return circle_facet_mesh((self.cx, self.cy), self.radius, self.n_faces)

    def describe(self) -> dict[str, Any]:
        return {
            "driver": "periodic body force",
            "diameter": self.diameter,
            "reynolds": self.reynolds,
            "n_faces": self.n_faces,
            "target_speed": self.target_speed,
            "collision": self.collision,
            "nx": self.nx,
            "ny": self.ny,
            "nu": self.nu,
            "tau": self.tau,
            "length_diameters": self.length_diameters,
            "half_height_diameters": self.half_height_diameters,
            "blockage": 1.0 / (2.0 * self.half_height_diameters),
        }

    def validate(self) -> None:
        if self.diameter < 8:
            raise ValueError("diameter must be >= 8 lattice nodes")
        if not 0.0 < self.target_speed <= 0.15:
            raise ValueError("target_speed must lie in (0, 0.15]")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        if self.n_faces < 8:
            raise ValueError("n_faces must be >= 8")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")


def _stream_periodic(fnext: np.ndarray, fpost: np.ndarray) -> None:
    """Streaming with periodic x and open y (wall rows are refilled by the BCs)."""
    fnext[0] = fpost[0]
    fnext[1] = np.roll(fpost[1], 1, axis=1)
    fnext[2, 1:, :] = fpost[2, :-1, :]
    fnext[3] = np.roll(fpost[3], -1, axis=1)
    fnext[4, :-1, :] = fpost[4, 1:, :]
    fnext[5, 1:, :] = np.roll(fpost[5, :-1, :], 1, axis=1)
    fnext[6, 1:, :] = np.roll(fpost[6, :-1, :], -1, axis=1)
    fnext[7, :-1, :] = np.roll(fpost[7, 1:, :], -1, axis=1)
    fnext[8, :-1, :] = np.roll(fpost[8, 1:, :], 1, axis=1)


def run_cylinder_periodic_case(
    config: PeriodicCylinderConfig,
    *,
    max_steps: int | None = None,
    store_fields: bool = False,
    label: str = "",
) -> dict[str, Any]:
    """Run one periodic, body-force driven cylinder solve and report the SFM drag."""
    config.validate()
    nx, ny = config.nx, config.ny
    mesh = config.mesh()
    links = build_cut_links(mesh, ny, nx)
    solid = links.solid
    opp = links.opp
    node = links.node

    shape = (ny, nx)
    f = equilibrium(np.full(shape, 1.0), np.full(shape, config.target_speed), np.zeros(shape))
    fpost = np.empty_like(f)
    fnext = np.empty_like(f)
    feq = np.empty_like(f)
    fopp = np.empty_like(f)
    foppeq = np.empty_like(f)
    tmp1 = np.empty_like(f)
    tmp2 = np.empty_like(f)
    force_term = np.empty_like(f)
    rho = np.empty(shape)
    ux = np.empty(shape)
    uy = np.empty(shape)
    scratch = {
        "cu": np.empty_like(f),
        "sq": np.empty_like(f),
        "usq": np.empty(shape),
        "tmp": np.empty(shape),
        "tmp9": np.empty_like(f),
        "safe": np.empty(shape),
    }

    n_steps = int(config.steps if max_steps is None else max_steps)
    transient = int(config.transient_fraction * n_steps)
    wp, wm = config.omega, config.omega_minus
    a_trt = 0.5 * (wp + wm)
    b_trt = 0.5 * (wp - wm)
    force_gain = 1.0 - 0.5 * wp
    fluid_mask = ~links.solid_2d
    n_fluid = int(fluid_mask.sum())
    body_force = float(config.body_force)
    cx_col = CX_ARR[:, None, None]
    cy_col = CY_ARR[:, None, None]
    w_col = W_ARR[:, None, None]

    samples: list[dict[str, float]] = []
    started = _time.perf_counter()

    for step in range(1, n_steps + 1):
        macroscopic_into(f, rho, ux, uy, scratch)
        _equilibrium_into(feq, rho, ux, uy, scratch)
        # ---- TRT collision
        np.subtract(f, feq, out=tmp1)
        np.take(f, OPP_ARR, axis=0, out=fopp)
        np.take(feq, OPP_ARR, axis=0, out=foppeq)
        np.subtract(fopp, foppeq, out=tmp2)
        tmp1 *= a_trt
        tmp2 *= b_trt
        tmp1 += tmp2
        np.subtract(f, tmp1, out=fpost)
        # ---- Guo body force (uniform, +x)
        cu = cx_col * ux[None] + cy_col * uy[None]
        force_term = w_col * (3.0 * (cx_col - ux[None]) + 9.0 * cu * cx_col) * body_force
        fpost += force_gain * force_term
        fpost.reshape(9, -1)[:, solid] = 0.0

        _stream_periodic(fnext, fpost)
        r = reflected_populations(fpost.reshape(9, -1), links)
        fnext.reshape(9, -1)[opp, node] = r

        # free-slip lateral walls
        fnext[2, 1, :] = fnext[4, 1, :]
        fnext[5, 1, :] = fnext[8, 1, :]
        fnext[6, 1, :] = fnext[7, 1, :]
        fnext[4, ny - 2, :] = fnext[2, ny - 2, :]
        fnext[7, ny - 2, :] = fnext[5, ny - 2, :]
        fnext[8, ny - 2, :] = fnext[6, ny - 2, :]
        fnext[:, 0, :] = 0.0
        fnext[:, -1, :] = 0.0
        f, fnext = fnext, f

        if step % config.control_every == 0 or step == 1:
            rho_s, ux_s, uy_s = macroscopic(f)
            u_mean = float(ux_s[fluid_mask].mean())
            if u_mean > 1e-9:
                body_force *= float(config.target_speed) / u_mean
            else:  # pragma: no cover - defensive
                body_force *= 1.5

        if step % config.sample_every == 0 or step == n_steps:
            rho_s, ux_s, uy_s = macroscopic(f)
            u_mean = float(ux_s[fluid_mask].mean())
            res = integrate_surface_force(
                mesh,
                rho=rho_s,
                velocity=[ux_s, uy_s],
                nu=config.nu,
                rho_ref=1.0,
                u_ref=max(u_mean, 1e-12),
                ref_length=config.diameter,
                ref_area=config.diameter,
            )
            coeff = res.coefficients()
            drag_global = body_force * n_fluid
            dyn = 0.5 * 1.0 * max(u_mean, 1e-12) ** 2 * config.diameter
            samples.append(
                {
                    "step": step,
                    "cd_sf": coeff["cd"],
                    "cd_pressure": coeff["cd_pressure"],
                    "cd_friction": coeff["cd_friction"],
                    "cl_sf": coeff["cl"],
                    "cd_global": drag_global / dyn,
                    "u_mean": u_mean,
                    "re_eff": u_mean * config.diameter / config.nu,
                    "rho_mean": float(rho_s[fluid_mask].mean()),
                    "body_force": body_force,
                }
            )
            if step >= n_steps - config.sample_every:
                prof = cp_at_angles(res, np.arange(0.0, 181.0, 15.0), axis=0)
                cp_snapshot = {
                    "theta_deg": [float(v) for v in prof["theta_deg"]],
                    "cp": [float(v) for v in prof["cp"]],
                    "cd": coeff["cd"],
                    "cd_pressure": coeff["cd_pressure"],
                    "cd_friction": coeff["cd_friction"],
                    "cl": coeff["cl"],
                }
            else:
                cp_snapshot = None

    elapsed = _time.perf_counter() - started
    tail = [s for s in samples if s["step"] > transient]
    if not tail:  # pragma: no cover - defensive
        tail = samples[-1:] or [{"cd_sf": float("nan")}]

    def _stat(key: str) -> tuple[float, float]:
        vals = np.asarray([s.get(key, np.nan) for s in tail], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    cd_sf, cd_sf_std = _stat("cd_sf")
    cd_glob, cd_glob_std = _stat("cd_global")
    u_mean, u_std = _stat("u_mean")
    re_eff, _ = _stat("re_eff")
    rho_mean, _ = _stat("rho_mean")
    cp_mean, cp_std = _stat("cd_pressure")
    cf_mean, cf_std = _stat("cd_friction")
    cl_mean, cl_std = _stat("cl_sf")

    out: dict[str, Any] = {
        "label": label or f"P{config.diameter}-Re{config.reynolds:g}-F{config.n_faces}",
        "config": config.describe(),
        "mesh": mesh.summary(),
        "boundary": links.meta,
        "steps": n_steps,
        "transient_step": transient,
        "steady_samples": len(tail),
        "cd_sf": cd_sf,
        "cd_sf_std": cd_sf_std,
        "cd_sf_pressure": cp_mean,
        "cd_sf_friction": cf_mean,
        "cd_global": cd_glob,
        "cd_global_std": cd_glob_std,
        "cl_sf": cl_mean,
        "cl_sf_std": cl_std,
        "re_effective": re_eff,
        "u_mean": u_mean,
        "u_mean_std": u_std,
        "mean_density": rho_mean,
        "body_force_final": float(body_force),
        "cd_sf_history": {
            "step": [int(s["step"]) for s in samples],
            "value": [float(s["cd_sf"]) for s in samples],
        },
        "cd_global_history": {
            "step": [int(s["step"]) for s in samples],
            "value": [float(s["cd_global"]) for s in samples],
        },
        "re_history": {
            "step": [int(s["step"]) for s in samples],
            "value": [float(s["re_eff"]) for s in samples],
        },
        "cp": cp_snapshot,
        "wall_seconds": elapsed,
        "cell_steps_per_second": nx * ny * n_steps / max(elapsed, 1e-9),
    }
    if store_fields:
        out["fields"] = {"rho": rho_s, "ux": ux_s, "uy": uy_s}
    return out
