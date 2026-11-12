"""Three-dimensional LBM–FEM–DEM load-exchange interface (round 1 of GAP-LBM-T5-ICE-COUPLING).

The 2-D chain in :mod:`tensorlbm.three_way` (``core2d``/``ibm2d``/``floe``/
``beam``/``scenario``) answers the physics question "does the three-way loop
close in 2-D".  The flagship T5 node asks for a *three-dimensional* LBM–FEM–DEM
chain (hull water hydrodynamics + hull-girder FEM + DEM ice failure).  The part
that does not exist yet — and that everything else in the 3-D chain depends on —
is the **load-exchange interface**: how a traction living on a 3-D Lagrangian
surface, or a point force living on a DEM particle, is mapped onto the DOFs of a
structural discretisation without losing momentum, moment or interface work.

This module is the reference implementation of that interface together with the
machine-checkable invariants the design claims.  The invariants are stated as
identities that hold *exactly in exact arithmetic* and are gated to machine
precision in ``scripts/three_way_interface3d_gates.py``:

====================  =================================================  ==============
id                     invariant                                          exact value
====================  =================================================  ==============
I1 partition of unity  ``sum_i N_i(x) = 1``                            -> resultant
                                                                        conservation
I2 linear completeness ``sum_i N_i(x) r_i = x``                         -> moment
                                                                        conservation
I3 work conjugacy      ``sum_i f_i·u_i = ∫ t·u dA`` for affine ``u``    -> no spurious
                                                                        interface work
I4 clock consistency   lag ``m`` in ``[0, n_sub]`` plus sub-cycling      -> no silent
                       must transport a known force history              time offset
I5 added-mass guard    explicit Dirichlet–Neumann staggered coupling is   -> μ < 1
                       unstable once ``μ = m_a/m_s >= 1``
I6 ledger closure      ``W_interface = ΔE_struct`` (trapezoid rule)      -> no energy
                                                                        created at the
                                                                        interface
====================  =================================================  ==============

Why shape-function transfer rather than a spread/nearest-node kernel
---------------------------------------------------------------------
Isoparametric Q1 hexahedra interpolate the position field exactly
(``sum_i N_i(ξ) r_i = x(ξ)`` is the map itself), therefore the **consistent**
transfer ``f_i += N_i(x_p) F`` conserves the resultant *and* the moment of the
load exactly, and its transpose ``u_p = sum_i N_i u_i`` is work-conjugate for
affine velocity fields.  The naive alternatives are measurably worse and are
kept in the module (:meth:`HexGrid3D.distribute_point_lumped`,
:func:`nearest_node_weights`) purely as the *control* that the gate compares
against, so the gate is falsifiable rather than tautological.

References
----------
- Peskin, C. S. (2002), *The immersed boundary method*, Acta Numerica 11, 479
  (Lagrangian surface tractions used here).
- Causin, P., Gerbeau, J. F., Nobile, F. (2005), *Added-mass effect in the
  design of partitioned algorithms for fluid-structure problems*,
  Comput. Methods Appl. Mech. Engrg. 194, 4506 (I5 criterion μ ≥ 1).
- Förster, C., Wall, W. A., Ramm, E. (2007), *Artificial added mass
  instabilities in sequential staggered coupling*, Comput. Methods Appl. Mech.
  Engrg. 196, 1278 (relaxation bound ``ω < 2/(1+μ)``).
- Bathe, K.-J. (2006), *Finite Element Procedures* (consistent load vectors,
  partition of unity / linear completeness of isoparametric elements).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Lagrangian surfaces (3-D)
# ---------------------------------------------------------------------------


@dataclass
class Surface3D:
    """A 3-D Lagrangian surface quadrature (markers carrying area weights/normals).

    Every field is a ``(M,)`` float64 tensor, so the object can also represent a
    triangulated hull patch: markers are triangle centroids, ``nx/ny/nz`` the
    geometric (area-weighted) normal and ``dA`` the triangle area.  With that
    convention ``sum_m n_m dA_m = 0`` holds exactly for a closed surface, which
    is the analytic anchor used by gate I1.
    """

    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    nx: torch.Tensor
    ny: torch.Tensor
    nz: torch.Tensor
    dA: torch.Tensor

    def __len__(self) -> int:
        return int(self.x.numel())

    # -- geometry ---------------------------------------------------------
    @property
    def positions(self) -> torch.Tensor:
        return torch.stack([self.x, self.y, self.z], dim=1)

    @property
    def normals(self) -> torch.Tensor:
        return torch.stack([self.nx, self.ny, self.nz], dim=1)

    @property
    def area(self) -> float:
        return float(self.dA.sum())

    def centroid(self) -> torch.Tensor:
        return (self.positions * self.dA.view(-1, 1)).sum(0) / self.dA.sum()

    # -- load integrals ---------------------------------------------------
    def traction_force(self, traction: torch.Tensor) -> torch.Tensor:
        """``F = ∫ t dA`` (traction normalised per unit area, ``(M,3)``)."""
        return (traction * self.dA.view(-1, 1)).sum(0)

    def pressure_force(self, pressure: float) -> torch.Tensor:
        """``F = ∫ -p n dA`` for a uniform pressure ``p``."""
        return -pressure * (self.normals * self.dA.view(-1, 1)).sum(0)

    def moment(self, traction: torch.Tensor, origin: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``M = ∫ (r - origin) × t dA``."""
        r = self.positions
        if origin is not None:
            r = r - origin
        return torch.cross(r, traction * self.dA.view(-1, 1), dim=1).sum(0)

    def work(self, traction: torch.Tensor, velocity: torch.Tensor) -> float:
        """``W = ∫ t·u dA`` with an *pointwise* velocity field ``(M,3)``."""
        return float(((traction * velocity).sum(1) * self.dA).sum())


def _triangulated_surface(
    verts: torch.Tensor, faces: torch.Tensor, flip: bool = False
) -> Surface3D:
    """Build a quadrature from a triangle soup (centroid rule, geometric normals)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cr = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = 0.5 * torch.linalg.norm(cr, dim=1)
    normal = cr / (2.0 * area).view(-1, 1)
    if flip:
        normal = -normal
    centroid = (v0 + v1 + v2) / 3.0
    return Surface3D(
        x=centroid[:, 0], y=centroid[:, 1], z=centroid[:, 2],
        nx=normal[:, 0], ny=normal[:, 1], nz=normal[:, 2],
        dA=area,
    )


def sphere_surface(
    centre: Sequence[float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    n_theta: int = 12,
    n_phi: int = 24,
    hemisphere: bool = False,
    cap: bool = False,
) -> Surface3D:
    """UV-sphere (or open hemisphere, optionally closed by a disc cap).

    Parameters
    ----------
    n_theta, n_phi:
        polar / azimuthal divisions.  The number of triangles is
        ``2*n_theta*n_phi`` for a full sphere.
    hemisphere:
        keep ``z >= 0`` only; the ``z = const`` rim is then an open boundary
        unless ``cap=True`` adds the flat disc (closed surface).
    cap:
        close a hemisphere with the equatorial disc.
    """
    c = torch.tensor(centre, dtype=DTYPE)
    n_th = n_theta if not hemisphere else n_theta // 2
    th = torch.linspace(0.0, math.pi / 2 if hemisphere else math.pi, n_theta + 1, dtype=DTYPE)
    ph = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=DTYPE)
    th_grid, ph_grid = torch.meshgrid(th, ph[:-1], indexing="ij")
    xs = torch.sin(th_grid) * torch.cos(ph_grid)
    ys = torch.sin(th_grid) * torch.sin(ph_grid)
    zs = torch.cos(th_grid)
    verts = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3) * radius + c

    faces = []
    n_ph = n_phi
    rows = n_theta  # latitude bands; a band touching a pole contributes one triangle
    sin_th = torch.sin(th).abs()

    def vid(i: int, j: int) -> int:
        return i * n_ph + (j % n_ph)

    for i in range(rows):
        north_pole = bool(sin_th[i] < 1e-12)
        south_pole = bool(sin_th[i + 1] < 1e-12)
        for j in range(n_ph):
            a, b = vid(i, j), vid(i, j + 1)
            cc, d = vid(i + 1, j), vid(i + 1, j + 1)
            if north_pole:
                faces.append((a, cc, d))
            elif south_pole:
                faces.append((a, cc, b))
            else:
                faces.append((a, cc, d))
                faces.append((a, d, b))
    faces = torch.tensor(sorted(set(faces)), dtype=torch.long)
    # drop degenerate triangles (zero area)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    faces = faces[area > 1e-14]
    surf = _triangulated_surface(verts, faces)

    if hemisphere and cap:
        ring = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=DTYPE)[:-1]
        rim = torch.stack([torch.cos(ring), torch.sin(ring), torch.zeros_like(ring)], 1) * radius + c
        hub = c.view(1, 3)
        cap_faces, cap_verts = [], [hub]
        for j in range(n_phi):
            cap_faces.append((0, 1 + j, 1 + (j + 1) % n_phi))
        cap_verts = torch.cat([hub, rim], 0)
        cv = torch.tensor(cap_faces, dtype=torch.long)
        cap_surf = _triangulated_surface(cap_verts, cv, flip=True)
        return Surface3D(
            x=torch.cat([surf.x, cap_surf.x]), y=torch.cat([surf.y, cap_surf.y]),
            z=torch.cat([surf.z, cap_surf.z]), nx=torch.cat([surf.nx, cap_surf.nx]),
            ny=torch.cat([surf.ny, cap_surf.ny]), nz=torch.cat([surf.nz, cap_surf.nz]),
            dA=torch.cat([surf.dA, cap_surf.dA]),
        )
    return surf


def rectangle_surface(
    origin: Sequence[float],
    e1: Sequence[float],
    e2: Sequence[float],
    n1: int = 4,
    n2: int = 4,
    normal: Optional[Sequence[float]] = None,
) -> Surface3D:
    """Planar rectangle split into ``n1×n2`` quads (2 triangles each).

    The discretisation is geometrically exact, so the reference integrals for
    the patch tests are exact analytic values of the *continuous* rectangle.
    """
    o = torch.tensor(origin, dtype=DTYPE)
    a1 = torch.tensor(e1, dtype=DTYPE)
    a2 = torch.tensor(e2, dtype=DTYPE)
    if normal is None:
        nvec = torch.cross(a1, a2, dim=0)
        nvec = nvec / torch.linalg.norm(nvec)
    else:
        nvec = torch.tensor(normal, dtype=DTYPE)
        nvec = nvec / torch.linalg.norm(nvec)

    verts = []
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            verts.append(o + a1 * (i / n1) + a2 * (j / n2))
    verts = torch.stack(verts, 0)

    def vid(i: int, j: int) -> int:
        return i * (n2 + 1) + j

    faces = []
    for i in range(n1):
        for j in range(n2):
            faces.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            faces.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    faces = torch.tensor(faces, dtype=torch.long)
    surf = _triangulated_surface(verts, faces)
    # orient all normals along ``nvec``
    if float((surf.normals * nvec).sum()) < 0.0:
        surf = _triangulated_surface(verts, faces, flip=True)
    return surf


def gauss_reference(fn: Callable[[torch.Tensor], torch.Tensor],
                    origin: Sequence[float], e1: Sequence[float], e2: Sequence[float],
                    n_gauss: int = 6) -> torch.Tensor:
    """Reference ``∫ fn(x) dA`` over the rectangle, Gauss-Legendre ``n_gauss²``.

    ``6×6`` points integrate polynomials of degree ≤ 11 exactly, i.e. the
    product of two affine fields used by the patch tests, so this is an
    *analytic* reference and not a discretisation of the same kind as the
    transfer being tested.
    """
    o = torch.tensor(origin, dtype=DTYPE)
    a1 = torch.tensor(e1, dtype=DTYPE)
    a2 = torch.tensor(e2, dtype=DTYPE)
    import numpy as _np

    gp = torch.tensor(_np.polynomial.legendre.leggauss(n_gauss)[0], dtype=DTYPE)
    gw = torch.tensor(_np.polynomial.legendre.leggauss(n_gauss)[1], dtype=DTYPE)
    acc = torch.zeros(3, dtype=DTYPE)
    for xi, wi in zip(gp, gw):
        for eta, wj in zip(gp, gw):
            p = o + a1 * (0.5 * (xi + 1.0)) + a2 * (0.5 * (eta + 1.0))
            acc = acc + wi * wj * fn(p.view(1, 3)).view(3)
    jac = 0.25 * torch.linalg.norm(torch.cross(a1, a2, dim=0))
    return acc * jac


# ---------------------------------------------------------------------------
# Q1 hexahedral structural discretisation (FEM side of the interface)
# ---------------------------------------------------------------------------


@dataclass
class HexGrid3D:
    """Structured 8-node trilinear (Q1) hexahedral grid.

    Only the *mapping* is implemented here (shape functions, consistent load
    projection, velocity gathering).  The structural solve itself is supplied by
    the FEM backend (``tensorlbm.beam_fem`` for the hull girder, a shell/solid
    solver for the round-2 chain); the interface contract requires the backend
    to expose nodal forces / nodal velocities in this node ordering.
    """

    origin: Sequence[float]
    extent: Sequence[float]           # physical size of the block (3,)
    n_cells: Sequence[int]            # cells per direction (3,)

    def __post_init__(self) -> None:
        self._o = torch.tensor(self.origin, dtype=DTYPE)
        self._L = torch.tensor(self.extent, dtype=DTYPE)
        self._nc = tuple(int(v) for v in self.n_cells)
        self.h = self._L / torch.tensor(self._nc, dtype=DTYPE)
        self.n_nodes = tuple(n + 1 for n in self._nc)
        ix = torch.arange(self.n_nodes[0], dtype=DTYPE)
        iy = torch.arange(self.n_nodes[1], dtype=DTYPE)
        iz = torch.arange(self.n_nodes[2], dtype=DTYPE)
        X, Y, Z = torch.meshgrid(ix, iy, iz, indexing="ij")
        self.nodes = torch.stack(
            [self._o[0] + X * self.h[0], self._o[1] + Y * self.h[1], self._o[2] + Z * self.h[2]], -1
        ).reshape(-1, 3)

    # -- helpers ----------------------------------------------------------
    def _node_id(self, i: int, j: int, k: int) -> int:
        return (i * self.n_nodes[1] + j) * self.n_nodes[2] + k

    def _cell_of(self, p: torch.Tensor) -> Tuple[int, int, int]:
        rel = (p - self._o) / self.h
        idx = []
        for d in range(3):
            v = int(torch.floor(rel[d]).item())
            v = min(max(v, 0), self._nc[d] - 1)
            idx.append(v)
        return tuple(idx)  # type: ignore[return-value]

    def local_coords(self, p: torch.Tensor) -> Tuple[Tuple[int, int, int], torch.Tensor]:
        """Cell index and local coordinates ``ξ ∈ [0,1]³`` (clipped at the boundary)."""
        (i, j, k) = self._cell_of(p)
        xi = (p - (self._o + torch.tensor([i, j, k], dtype=DTYPE) * self.h)) / self.h
        return (i, j, k), xi.clamp(0.0, 1.0)

    @staticmethod
    def shape8(xi: torch.Tensor) -> torch.Tensor:
        """Trilinear shape functions at ``ξ`` (ordering ``i + 2j + 4k``)."""
        x, y, z = xi[0], xi[1], xi[2]
        return torch.stack([
            (1 - x) * (1 - y) * (1 - z),
            x * (1 - y) * (1 - z),
            (1 - x) * y * (1 - z),
            x * y * (1 - z),
            (1 - x) * (1 - y) * z,
            x * (1 - y) * z,
            (1 - x) * y * z,
            x * y * z,
        ])

    def element_nodes(self, cell: Tuple[int, int, int]) -> torch.Tensor:
        i, j, k = cell
        ids = []
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    ids.append(self._node_id(i + di, j + dj, k + dk))
        return torch.tensor(ids, dtype=torch.long)

    def containing_element(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cell, xi = self.local_coords(p)
        return self.element_nodes(cell), self.shape8(xi)

    # -- load transfer ----------------------------------------------------
    def distribute_point_consistent(self, p: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """Consistent Q1 transfer of a point force: ``f_i += N_i(x_p) F``."""
        ids, w = self.containing_element(p)
        out = torch.zeros(self.nodes.shape[0], 3, dtype=DTYPE)
        out.index_add_(0, ids, w.view(-1, 1) * force.view(1, 3))
        return out

    def distribute_point_lumped(self, p: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """Control (deliberately worse): everything on the nearest node."""
        d = torch.linalg.norm(self.nodes - p.view(1, 3), dim=1)
        out = torch.zeros_like(self.nodes)
        out[int(torch.argmin(d))] = force
        return out

    def distribute_surface(self, surface: Surface3D, traction: torch.Tensor) -> torch.Tensor:
        """Consistent surface load: ``f_i += Σ_m N_i(x_m) t_m dA_m``."""
        out = torch.zeros(self.nodes.shape[0], 3, dtype=DTYPE)
        for m in range(len(surface)):
            ids, w = self.containing_element(surface.positions[m])
            out.index_add_(0, ids, (w * surface.dA[m]).view(-1, 1) * traction[m].view(1, 3))
        return out

    # -- kinematics -------------------------------------------------------
    def gather_point_velocity(self, p: torch.Tensor, u_nodes: torch.Tensor) -> torch.Tensor:
        """Transpose operator: ``u_p = Σ_i N_i(x_p) u_i`` (work-conjugate to the load)."""
        ids, w = self.containing_element(p)
        return (w.view(-1, 1) * u_nodes[ids]).sum(0)

    def gather_surface_velocity(self, surface: Surface3D, u_nodes: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.gather_point_velocity(surface.positions[m], u_nodes)
                            for m in range(len(surface))], 0)


def nearest_node_weights(nodes: torch.Tensor, p: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Control kernel: unit weight on the nearest node (non work-conjugate)."""
    d = torch.linalg.norm(nodes - p.view(1, 3), dim=1)
    idx = int(torch.argmin(d))
    w = torch.zeros(nodes.shape[0], dtype=DTYPE)
    w[idx] = 1.0
    return idx, w


# ---------------------------------------------------------------------------
# Hull-girder Hermite mapping (3-D surface tractions -> 1-D beam DOFs)
# ---------------------------------------------------------------------------


def hermite_shape(xi: torch.Tensor, le: float) -> torch.Tensor:
    """Hermite cubic shape functions ``[N1, N2, N3, N4]`` on ``ξ ∈ [0,1]``."""
    return torch.stack([
        1 - 3 * xi ** 2 + 2 * xi ** 3,
        le * (xi - 2 * xi ** 2 + xi ** 3),
        3 * xi ** 2 - 2 * xi ** 3,
        le * (-xi ** 2 + xi ** 3),
    ])


def project_to_beam(
    x_loads: torch.Tensor,
    loads: torch.Tensor,
    length: float,
    n_elements: int,
) -> torch.Tensor:
    """Map transverse loads ``loads`` at ``x_loads`` onto beam DOFs ``[w, θ]``.

    Uses the same Hermite shape functions as :class:`tensorlbm.beam_fem.EulerBernoulliBeamFEM`,
    so the transfer is the transpose of the beam's own interpolation and therefore
    conserves the resultant transverse force exactly and its first moment
    (``∫ x q dx``) to machine precision.
    """
    le = length / n_elements
    dof = torch.zeros(2 * (n_elements + 1), dtype=DTYPE)
    for xv, fv in zip(x_loads, loads):
        x = float(xv)
        e = min(int(x / le), n_elements - 1)
        xi = torch.tensor((x - e * le) / le, dtype=DTYPE)
        w = hermite_shape(xi, le)
        dof[2 * e: 2 * e + 4] += w * fv
    return dof


# ---------------------------------------------------------------------------
# Time integration: sub-cycling, lag, added-mass guard
# ---------------------------------------------------------------------------


@dataclass
class CouplingClock:
    """Partitioned explicit coupling clock (fluid dt, structural dt, lag).

    The fluid advances ``n_sub`` steps per structural step; the structural force
    is evaluated from the fluid state lagged by ``lag`` structural steps.  Both
    are *declared* here so that the lag used by a run can be read back from the
    artefact instead of being implicit in the driver.
    """

    dt_fluid: float
    dt_struct: float
    n_sub: int = 1
    lag: int = 0

    def __post_init__(self) -> None:
        if self.dt_fluid <= 0 or self.dt_struct <= 0:
            raise ValueError("time steps must be positive")
        if self.n_sub < 1:
            raise ValueError("n_sub must be >= 1")
        if self.lag < 0:
            raise ValueError("lag must be >= 0")

    @property
    def mismatch(self) -> float:
        """|n_sub·dt_fluid - dt_struct| / dt_struct — must be 0 for a consistent clock."""
        return abs(self.n_sub * self.dt_fluid - self.dt_struct) / self.dt_struct

    def check(self, tol: float = 1e-12) -> None:
        if self.mismatch > tol:
            raise ValueError(
                f"inconsistent clock: n_sub*dt_fluid={self.n_sub * self.dt_fluid:.6e} "
                f"!= dt_struct={self.dt_struct:.6e} (rel {self.mismatch:.2e})"
            )


def added_mass_ratio(m_s: float, rho_f: float, displaced_volume: float,
                     added_mass_coefficient: float = 1.0) -> float:
    """``μ = m_a / m_s`` with ``m_a = C_a ρ_f V`` (``C_a = 1`` for a sphere)."""
    if m_s <= 0:
        raise ValueError("structural mass must be positive")
    return added_mass_coefficient * rho_f * displaced_volume / m_s


def explicit_staggered_stable(mu: float) -> bool:
    """Explicit Dirichlet–Neumann staggering is stable iff ``μ < 1`` (Causin 2005)."""
    return mu < 1.0


def relaxation_bound(mu: float) -> float:
    """Interface fixed-point iteration converges iff ``0 < ω < 2/(1+μ)`` (Förster 2007)."""
    return 2.0 / (1.0 + mu)


def _staggered_matrix(mu: float, omega_n_dt: float) -> torch.Tensor:
    """State matrix of the 1-DOF added-mass scheme (state ``[x, v, a]``).

    Structural part: velocity-Verlet (symplectic, ``|λ| = 1`` for ``μ = 0``);
    fluid part: the added-mass reaction is taken from the *previous* acceleration,
    ``a_{n+1} = -ω_n² x_{n+1} - μ a_n`` — the reduced form of the explicit
    Dirichlet–Neumann staggering (Causin et al. 2005).
    """
    w2 = omega_n_dt ** 2
    dt = 1.0  # time unit = dt
    A = torch.zeros(3, 3, dtype=DTYPE)
    A[0, 0], A[0, 1], A[0, 2] = 1.0, dt, 0.5 * dt * dt       # x_{n+1}
    for c in range(3):
        A[2, c] = -w2 * A[0, c]                               # a_{n+1} = -w2 x_{n+1} - mu a_n
    A[2, 2] += -mu
    for c in range(3):
        A[1, c] = 0.5 * dt * A[2, c]                          # v_{n+1} = v_n + dt/2 (a_n + a_{n+1})
    A[1, 1] += 1.0
    A[1, 2] += 0.5 * dt
    return A


def staggered_spectral_radius(mu: float, omega_n_dt: float = 1e-2) -> float:
    """Closed-form spectral radius of :func:`_staggered_matrix`.

    ``|λ| = 1`` for ``μ < 1`` (marginally stable, no growth) and ``|λ| → μ`` for
    ``μ > 1``, so the sign of ``|λ| - 1`` is exactly the criterion implemented by
    :func:`explicit_staggered_stable`.  The number returned here is the *exact*
    linear-algebra result for the discrete recursion, independent of the time
    marching in :func:`march_one_dof`.
    """
    return float(torch.linalg.eigvals(_staggered_matrix(mu, omega_n_dt)).abs().max())


def march_one_dof(mu: float, omega_n_dt: float = 1e-2, n_steps: int = 200,
                  x0: float = 1.0) -> torch.Tensor:
    """Time-march the reduced added-mass model; returns the ``[x, v, a]`` history.

    Used by the acceptance gate to compare the *measured* growth rate against
    :func:`staggered_spectral_radius` — the gate is falsifiable because the two
    numbers come from independent code paths (march vs eigendecomposition).
    """
    A = _staggered_matrix(mu, omega_n_dt)
    s = torch.zeros(3, n_steps + 1, dtype=DTYPE)
    s[:, 0] = torch.tensor([x0, 0.0, -omega_n_dt ** 2 * x0], dtype=DTYPE)
    for n in range(n_steps):
        s[:, n + 1] = A @ s[:, n]
    return s


@dataclass
class InterfaceLedger:
    """Interface work bookkeeping (trapezoid rule, no energy created).

    The work an interface force performs on the structure is accumulated as
    ``W += ½ (f_{n-1} + f_n)·(x_n - x_{n-1})``; because the structural update uses
    the same trapezoid rule for the velocity, ``W`` equals the change in
    structural kinetic + strain energy to machine precision when the interface is
    the only channel (gate I6).
    """

    work: float = 0.0
    _prev_force: Optional[torch.Tensor] = None
    _prev_x: Optional[torch.Tensor] = None

    def add(self, force: torch.Tensor, displacement: torch.Tensor) -> float:
        f = force.view(-1).to(DTYPE)
        x = displacement.view(-1).to(DTYPE)
        if self._prev_force is not None:
            self.work += float(0.5 * ((self._prev_force + f) * (x - self._prev_x)).sum())
        self._prev_force, self._prev_x = f.clone(), x.clone()
        return self.work

    def closure(self, delta_energy: float) -> float:
        """Relative residual ``|W - ΔE| / max(|ΔE|, eps)``."""
        denom = max(abs(delta_energy), 1e-300)
        return abs(self.work - delta_energy) / denom


# ---------------------------------------------------------------------------
# Convenience: one consistent exchange cycle
# ---------------------------------------------------------------------------


def exchange_cycle(
    grid: HexGrid3D,
    surface: Surface3D,
    traction: torch.Tensor,
    *,
    particle_positions: Optional[torch.Tensor] = None,
    particle_forces: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """One fluid→structure hand-off: surface tractions + DEM point forces.

    Returns the nodal force vector together with the two diagnostics the gates
    consume (the resultant and the moment of the *transferred* load), so that no
    driver has to recompute them differently from the gate.
    """
    nodal = grid.distribute_surface(surface, traction)
    if particle_positions is not None and particle_forces is not None:
        for p, f in zip(particle_positions, particle_forces):
            nodal = nodal + grid.distribute_point_consistent(p, f)
    resultant = nodal.sum(0)
    moment = torch.cross(grid.nodes, nodal, dim=1).sum(0)
    return {"nodal": nodal, "resultant": resultant, "moment": moment}