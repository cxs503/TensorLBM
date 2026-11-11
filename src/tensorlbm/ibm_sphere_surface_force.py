"""Immersed-boundary *surface-force* verification surface for a sphere (3-D).

Why this module exists
----------------------
The repository already ships two families of immersed-boundary (IBM) kernels
(:mod:`tensorlbm.ibm`, :mod:`tensorlbm.ibm_force_transfer`) and a 2-D
curved-surface quadrature (:mod:`tensorlbm.ibm_curved_force`), but there was
no way to *certify* what the force they return means for a three-dimensional
body.  Two gaps were open (gap ledger, deep-water army):

* **G12** -- a *pressure-only* wall integral (`extrap='none'`) systematically
  under-predicts the drag of a curved body, because the near-wall cell-centred
  pressure misses the stagnation over-shoot; and it carries **no tangential
  traction at all**, so it cannot certify a torque.
* **G15** -- the Ladd-style *momentum exchange* sum pairs populations from two
  different time levels across a curved wall, which leaves an **additive
  free-stream background**: on a staircase sphere it reported
  ``Cd = 3.98`` against a reference ``1.09``.

Both defects have the same root cause: the returned number is a *sum of
populations*, not a **quadrature of the surface traction**.  This module adds
the missing structure and then measures it on the three layers the node asks
for:

1. **Drag over three decades of Re** (``Re in [1, 1000]``) against two
   independent empirical anchors -- Schiller-Naumann (1933, ``±5 %`` for
   ``Re < 800``) and Clift-Gauvin (1971, valid to ``Re ~ 3e5``) -- which are
   themselves *mutually adjudicated* (:func:`adjudicate_empirical`: the two
   fits differ by a few percent, so the acceptance band is the *intersection
   of the two bands*, not one fit's point value).
2. **Observed spatial order** from a fixed-``Re`` refinement study
   (:func:`observed_order`, Richardson + GCI).
3. **Torque coefficient of a steadily rotating sphere**, the layer that
   *discriminates* a normal-only estimator from a full traction: for a purely
   azimuthal Stokes flow the pressure is constant on the sphere, so a
   pressure-only estimator returns ``C_M = 0`` where the exact result is
   ``C_M = 8/Re_omega`` (:func:`stokes_torque_coefficient`).

Design contract
---------------
* **Exact surface measure.**  Markers sit in Gauss-Legendre cells of the polar
  angle and uniform cells of the azimuth; marker ``k`` owns the *exact*
  spherical area of its own cell, ``A_k = R^2 (integral sin(theta) dtheta)
  dphi``, so ``sum_k A_k = 4 pi R^2`` to round-off **independently of the
  marker count**.  Since every force statement is an integral, the measure
  never inherits the ``O(ds^2)`` error of the marker positions.
* **Traction, not populations.**  ``tau_k = rho_k (u_b - u_k^*)`` is a
  *traction* (momentum per unit area per step); ``F_k = tau_k A_k`` is the
  marker force; ``f(x) = sum_k F_k delta_h(x - X_k)`` is what the fluid
  receives.  By partition of unity ``sum_x f(x) = sum_k F_k``, which is what
  makes the recoil identity machine-checkable rather than "by construction".
* **Closed-surface null space.**  A *uniform* flow that satisfies the boundary
  condition (rigid translation) gives ``F_k = 0`` exactly, and the quadrature
  of a constant traction over a closed surface is exactly zero
  (`sum_k n_k A_k = 0`): the additive free-stream background of G15 has no
  spectral component in this operator.  This is the negative control that the
  old estimators fail (:func:`legacy_momentum_exchange_force` reproduces the
  failure mode for the record).
* **Guo half-force corrector + normal projection.**  The physical velocity at
  a node is ``u^* = u_mom + F/(2 rho)``; extra interpolation passes remove
  that bias, and a normal-projection pass (Breugem 2012) cancels the residual
  wall-normal slip without touching the tangential (friction) traction.

Conventions
-----------
Lattice units throughout: ``dt = dx = rho_0 = 1``.  A marker force ``F_k`` is
a momentum change per time step; the force **on the body** is ``-sum_k F_k``,
the same sign convention as :func:`tensorlbm.ibm_force_transfer.direct_forcing`.

References
----------
Peskin, C. S. (2002) *Acta Numerica* 11, 479.
Uhlmann, M. (2005) *J. Comput. Phys.* 209, 448.
Guo, Z., Zheng, C. & Shi, B. (2002) *Phys. Rev. E* 65, 046308.
Breugem, W.-P. (2012) *J. Comput. Phys.* 231, 4469.
Schiller, L. & Naumann, A. (1933) *VDI-Zeitung* 77, 318.
Clift, R. & Gauvin, W. H. (1971) *Proc. Chemeca '70* 14, 14.
Landau, L. D. & Lifshitz, E. M. (1987) *Fluid Mechanics*, 2nd ed., Sec. 20
    (torque on a steadily rotating sphere, ``T = 8 pi mu Omega a^3``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .d3q19 import _c_on, _w_on, equilibrium3d, macroscopic3d
from .drag_closure import cd_clift_gauvin, cd_schiller_naumann
from .ibm_force_transfer import IBMStencil, build_stencil

__all__ = [
    "SphereSurface",
    "TractionResult",
    "adjudicate_empirical",
    "build_sphere_surface",
    "drag_coefficient",
    "interior_rigid_force",
    "legacy_unweighted_marker_force",
    "observed_order",
    "pressure_only_torque",
    "run_sphere_drag",
    "run_sphere_rotation",
    "sphere_quadrature",
    "stokes_drag_coefficient",
    "stokes_torque_coefficient",
    "surface_traction",
    "torque_coefficient",
]


# --------------------------------------------------------------------------- #
# Exact spherical measure
# --------------------------------------------------------------------------- #


def _gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """Nodes/weights of the ``n``-point Gauss-Legendre rule on ``[-1, 1]``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x: list[float] = []
    w: list[float] = []
    for i in range(1, n + 1):
        # Newton iteration from the Chebyshev-like initial guess.
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, z
            for k in range(2, n + 1):
                p0, p1 = p1, ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k
            dp = n * (z * p1 - p0) / (z * z - 1.0)
            dz = p1 / dp
            z -= dz
            if abs(dz) < 1e-16:
                break
        x.append(z)
        w.append(2.0 / ((1.0 - z * z) * dp * dp))
    return x, w


def sphere_quadrature(
    radius: float,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    n_theta: int = 24,
    n_phi: int = 48,
    *,
    gl_order: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Markers on a sphere with their **exact** cell areas.

    The polar angle is split into ``n_theta`` cells and the azimuth into
    ``n_phi`` cells.  Each cell's area weight is evaluated with an
    ``gl_order``-point Gauss-Legendre rule for ``integral sin(theta) dtheta``,
    and the marker is placed at the **area centroid** of its cell (not at the
    cell midpoint), so the first moment of the measure is also exact.

    Args:
        radius: sphere radius in lattice units.
        center: ``(x, y, z)`` centre in lattice coordinates.
        n_theta: number of polar cells (``theta in (0, pi)``).
        n_phi: number of azimuthal cells (``phi in [0, 2 pi)``).
        gl_order: Gauss-Legendre order per polar cell.

    Returns:
        ``(points (M, 3) float64, areas (M,) float64)`` with
        ``sum(areas) == 4 pi R^2`` to round-off.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if n_theta < 2 or n_phi < 3:
        raise ValueError("need n_theta >= 2 and n_phi >= 3")
    gl_x, gl_w = _gauss_legendre(gl_order)
    edges = [math.pi * i / n_theta for i in range(n_theta + 1)]
    dphi = 2.0 * math.pi / n_phi

    sin_integral: list[float] = []
    cos_integral: list[float] = []
    for i in range(n_theta):
        a, b = edges[i], edges[i + 1]
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        s = 0.0
        c = 0.0
        for xi, wi in zip(gl_x, gl_w):
            th = mid + half * xi
            s += wi * math.sin(th)
            c += wi * th * math.sin(th)
        sin_integral.append(half * s)
        cos_integral.append(half * c)

    pts: list[list[float]] = []
    areas: list[float] = []
    cx, cy, cz = (float(v) for v in center)
    for i in range(n_theta):
        theta_c = cos_integral[i] / sin_integral[i]  # area centroid angle
        st, ct = math.sin(theta_c), math.cos(theta_c)
        for j in range(n_phi):
            phi = (j + 0.5) * dphi
            pts.append(
                [
                    cx + radius * st * math.cos(phi),
                    cy + radius * st * math.sin(phi),
                    cz + radius * ct,
                ]
            )
            areas.append(radius * radius * sin_integral[i] * dphi)
    return (
        torch.tensor(pts, dtype=torch.float64),
        torch.tensor(areas, dtype=torch.float64),
    )


@dataclass(frozen=True)
class SphereSurface:
    """Sphere marker set + interior mask + IBM stencil on a periodic grid."""

    center: torch.Tensor
    radius: float
    points: torch.Tensor
    areas: torch.Tensor
    normals: torch.Tensor
    mask: torch.Tensor
    shape: tuple[int, int, int]
    stencil: IBMStencil

    @property
    def n_markers(self) -> int:
        return int(self.points.shape[0])

    @property
    def area_total(self) -> float:
        return float(self.areas.sum().item())


def build_sphere_surface(
    shape: Sequence[int],
    center: Sequence[float],
    radius: float,
    *,
    n_theta: int = 24,
    n_phi: int = 48,
    kernel: str = "4pt",
) -> SphereSurface:
    """Build the marker set, interior mask and IBM stencil for a sphere.

    Args:
        shape: grid shape ``(nz, ny, nx)`` (the solver layout).
        center: centre in lattice coordinates ``(x, y, z)``.
        radius: radius in lattice units.
        n_theta / n_phi: quadrature resolution of the surface.
        kernel: IBM delta kernel (``"4pt"`` or ``"hat"``).
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError("shape must be (nz, ny, nx)")
    ctr = torch.tensor([float(v) for v in center], dtype=torch.float64)
    pts, areas = sphere_quadrature(radius, tuple(ctr.tolist()), n_theta, n_phi)
    normals = (pts - ctr.unsqueeze(0)) / radius

    nz, ny, nx = shape
    zz = torch.arange(nz, dtype=torch.float64).view(nz, 1, 1)
    yy = torch.arange(ny, dtype=torch.float64).view(1, ny, 1)
    xx = torch.arange(nx, dtype=torch.float64).view(1, 1, nx)
    d2 = (xx - ctr[0]) ** 2 + (yy - ctr[1]) ** 2 + (zz - ctr[2]) ** 2
    mask = d2 <= radius * radius

    stencil = build_stencil(pts, shape, kernel, axis_order="xyz")
    return SphereSurface(
        center=ctr,
        radius=float(radius),
        points=pts,
        areas=areas,
        normals=normals,
        mask=mask,
        shape=shape,
        stencil=stencil,
    )


# --------------------------------------------------------------------------- #
# Surface traction -> force / torque
# --------------------------------------------------------------------------- #


@dataclass
class TractionResult:
    """Output of :func:`surface_traction`."""

    marker_force: torch.Tensor       # (M, 3) momentum per step
    marker_traction: torch.Tensor    # (M, 3) momentum per step per unit area
    force_grid: torch.Tensor         # (3, nz, ny, nx) Eulerian forcing
    force: torch.Tensor              # (3,) force **on the body**
    torque: torch.Tensor             # (3,) torque **on the body** about the centre
    slip_before: float = 0.0
    slip_after: float = 0.0
    closure_residual: float = 0.0    # |sum_grid - sum_marker| / |sum_marker|


def surface_traction(
    surface: SphereSurface,
    u_grid: torch.Tensor,
    u_target: torch.Tensor,
    rho_grid: torch.Tensor,
    *,
    n_correctors: int = 2,
    n_normal_passes: int = 1,
) -> TractionResult:
    """Traction-based immersed-boundary force for a closed sphere surface.

    ``tau_k = rho_k (u_b,k - u_k^*)`` with the Guo half-force velocity
    ``u^* = u_mom + F/(2 rho)``; ``F_k = tau_k A_k``; the marker forces are
    spread with the 4-point Peskin kernel.  ``n_normal_passes`` Breugem passes
    remove the residual wall-normal slip (the *mass* closure) without touching
    the tangential traction (which carries the torque).

    Args:
        surface: geometry from :func:`build_sphere_surface`.
        u_grid: ``(3, nz, ny, nx)`` momentum velocity ``u_mom``.
        u_target: ``(M, 3)`` boundary velocity at the markers.
        rho_grid: ``(nz, ny, nx)`` density.
        n_correctors: extra half-force correction passes.
        n_normal_passes: Breugem normal-projection passes.

    Returns:
        :class:`TractionResult` (force/torque **on the body**).
    """
    st = surface.stencil
    dtype = u_grid.dtype
    rho_m = st.interpolate(rho_grid.to(torch.float64)).clamp(min=1e-12)  # (M,)
    u_int = st.interpolate(u_grid.to(torch.float64))                    # (3, M)
    target = torch.as_tensor(u_target, dtype=torch.float64).T           # (3, M)

    traction = rho_m.unsqueeze(0) * (target - u_int)
    slip_before = float(traction.abs().sum().item())

    for _ in range(max(0, int(n_correctors))):
        u_phys = u_int + 0.5 * traction / rho_m.unsqueeze(0)
        traction = traction + rho_m.unsqueeze(0) * (target - u_phys)

    normal = surface.normals.to(traction.dtype)        # (M, 3)
    for _ in range(max(0, int(n_normal_passes))):
        u_phys = u_int + 0.5 * traction / rho_m.unsqueeze(0)
        slip = (target - u_phys).T                     # (M, 3)
        slip_n = (slip * normal).sum(dim=1)            # (M,)
        traction = traction + rho_m.unsqueeze(0) * (slip_n.unsqueeze(1) * normal).T

    u_phys = u_int + 0.5 * traction / rho_m.unsqueeze(0)
    slip_after = float((rho_m.unsqueeze(0) * (target - u_phys)).abs().sum().item())

    f_marker = (traction * surface.areas.unsqueeze(0)).T.contiguous()  # (M, 3)
    f_grid = st.spread(f_marker).to(dtype)                            # (3, nz, ny, nx)

    s_marker = f_marker.sum(dim=0)
    s_grid = f_grid.reshape(3, -1).sum(dim=1)
    scale = float(s_marker.abs().max().item()) or 1.0
    closure = float((s_grid - s_marker).abs().max().item()) / scale

    rel = surface.points - surface.center.unsqueeze(0)
    torque = -torch.cross(rel, f_marker, dim=1).sum(dim=0)
    return TractionResult(
        marker_force=f_marker,
        marker_traction=traction.T.contiguous(),
        force_grid=f_grid,
        force=-s_marker.to(dtype),
        torque=torque.to(dtype),
        slip_before=slip_before,
        slip_after=slip_after,
        closure_residual=closure,
    )


def interior_rigid_force(
    surface: SphereSurface,
    u_grid: torch.Tensor,
    rho_grid: torch.Tensor,
    u_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eulerian direct forcing that holds the sphere interior as a rigid body.

    A closed Lagrangian shell alone leaves the trapped interior fluid free to
    be accelerated by the drive; the interior must carry the rigid-body
    velocity too.  The recoil of this interior force belongs to the body, so
    it is returned alongside the field.

    Returns:
        ``(force_grid (3, nz, ny, nx), body_force (3,))`` with the body force
        being the **recoil on the sphere** from the interior.
    """
    mask = surface.mask
    if mask.sum() == 0:
        zeros = torch.zeros(3, dtype=u_grid.dtype)
        return torch.zeros_like(u_grid), zeros
    tar = torch.as_tensor(u_target, dtype=u_grid.dtype)
    if tar.dim() == 1:  # uniform rigid-body velocity
        tar = tar.view(3, 1, 1, 1)
    elif tar.shape != u_grid.shape:
        raise ValueError(
            f"u_target must be (3,) or {tuple(u_grid.shape)}, got {tuple(tar.shape)}"
        )
    f_int = torch.where(
        mask.unsqueeze(0),
        (tar - u_grid) * rho_grid.unsqueeze(0),
        torch.zeros_like(u_grid),
    )
    body_force = -f_int.reshape(3, -1).sum(dim=1)
    return f_int, body_force


def pressure_only_torque(
    surface: SphereSurface,
    rho_grid: torch.Tensor,
    u_grid: torch.Tensor,
    *,
    dt: float = 1.0,
) -> torch.Tensor:
    """Torque produced by a **pressure-only** (normal) surface integral.

    This is the G12 estimator: ``p = rho/3`` in lattice units, ``tau = -p n``,
    so the traction is purely normal.  On a purely azimuthal flow the pressure
    is constant on the sphere and the result is zero, which is exactly why a
    pressure-only route cannot certify a torque coefficient.  Returned for the
    negative-control record, never for certification.
    """
    st = surface.stencil
    rho_m = st.interpolate(rho_grid.to(torch.float64))
    p = rho_m / 3.0
    f_marker = -p.unsqueeze(1) * surface.normals * surface.areas.unsqueeze(1) / dt
    rel = surface.points - surface.center.unsqueeze(0)
    return -torch.cross(rel, f_marker, dim=1).sum(dim=0)


def legacy_unweighted_marker_force(
    surface: SphereSurface,
    u_grid: torch.Tensor,
    u_target: torch.Tensor,
    rho_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy marker *sum* (no surface measure) -- the G12/G15 failure mode.

    The IBM kernels shipped in :mod:`tensorlbm.ibm` and
    :mod:`tensorlbm.ibm_force_transfer` return ``sum_k tau_k``, i.e. a sum over
    markers with unit weights.  That is not a quadrature of the traction: for a
    **uniform** traction field on a closed surface the exact integral is zero
    (:func:`sphere_quadrature` gives ``sum_k A_k n_k = 0``) while the marker
    sum returns ``M * tau`` -- an additive background proportional to the
    marker count, which is exactly the O(1) free-stream offset of G15 and the
    origin of the marker-count dependence of the drag.  Returned for the
    negative-control record only; never used for certification.
    """
    st = surface.stencil
    rho_m = st.interpolate(rho_grid.to(torch.float64)).clamp(min=1e-12)
    u_int = st.interpolate(u_grid.to(torch.float64))
    target = torch.as_tensor(u_target, dtype=torch.float64).T
    f_marker = rho_m.unsqueeze(0) * (target - u_int)          # (3, M) no measure
    force = -f_marker.sum(dim=1)
    rel = surface.points - surface.center.unsqueeze(0)
    torque = -torch.cross(rel, f_marker.T, dim=1).sum(dim=0)
    return force, torque


# --------------------------------------------------------------------------- #
# Reference coefficients
# --------------------------------------------------------------------------- #


def stokes_drag_coefficient(re: float) -> float:
    """``C_D = 24/Re`` -- exact ``Re -> 0`` limit (unbounded flow)."""
    return 24.0 / float(re)


def stokes_torque_coefficient(re_omega: float) -> float:
    """``C_M = 8/Re_omega`` for a steadily rotating sphere in a quiescent fluid.

    Exact solution of the Stokes equations (Landau & Lifshitz, *Fluid
    Mechanics*, Sec. 20): ``T = 8 pi mu Omega a^3``.  With
    ``C_M = T / (0.5 rho U_tip^2 A a)``, ``U_tip = Omega a``, ``A = 4 pi a^2``
    and ``Re_omega = U_tip (2a) / nu`` this is ``C_M = 8/Re_omega`` exactly.
    Deviations at finite ``Re_omega`` are ``O(Re_omega)`` (the flow is purely
    azimuthal only in the Stokes limit).
    """
    return 8.0 / float(re_omega)


def adjudicate_empirical(re: float) -> dict[str, float]:
    """Mutual adjudication of the two empirical drag anchors.

    Schiller-Naumann (1933) and Clift-Gauvin (1971) are *independent fits*;
    over ``Re in [1, 1000]`` they disagree by a few percent.  Quoting one of
    them as "the" reference hides that model spread, so the acceptance band is
    the interval spanned by the two fits (widened by the fit's own stated
    uncertainty only through the reported ``half_width``).
    """
    sn = cd_schiller_naumann(re)
    cg = cd_clift_gauvin(re)
    lo, hi = (min(sn, cg), max(sn, cg))
    mid = 0.5 * (lo + hi)
    return {
        "re": float(re),
        "schiller_naumann": sn,
        "clift_gauvin": cg,
        "band_low": lo,
        "band_high": hi,
        "band_mid": mid,
        "spread_pct": 100.0 * (hi - lo) / mid if mid else 0.0,
    }


def drag_coefficient(force: torch.Tensor | Sequence[float], rho: float, speed: float, diameter: float) -> float:
    """``C_D = |F_x| / (0.5 rho U^2 D^2 pi/4)`` -- frontal-area convention."""
    fx = float(torch.as_tensor(force).reshape(-1)[0].item())
    area = 0.25 * math.pi * diameter * diameter
    return abs(fx) / (0.5 * rho * speed * speed * area)


def torque_coefficient(torque: torch.Tensor | Sequence[float], rho: float, tip_speed: float, diameter: float) -> float:
    """``C_M = |T| / (0.5 rho U_tip^2 (4 pi a^2) a)`` -- sphere-surface area.

    The normalisation matters: with the sphere **surface** area ``4 pi a^2``
    the Stokes rotating sphere gives exactly ``C_M = 8/Re_omega``
    (:func:`stokes_torque_coefficient`); with the frontal area ``pi a^2`` the
    same physics reads ``C_M = 32/Re_omega``.  This module uses the
    surface-area convention so that the analytic anchor is the compact
    ``8/Re_omega`` and the number can be checked against literature tables
    that use it.  Drag keeps the standard frontal-area convention.
    """
    t = float(torch.as_tensor(torque).reshape(-1).norm().item())
    a = 0.5 * diameter
    area = 4.0 * math.pi * a * a
    return t / (0.5 * rho * tip_speed * tip_speed * area * a)


def observed_order(
    spacings: Sequence[float],
    values: Sequence[float],
) -> dict[str, float]:
    """Observed order of accuracy from a refinement study.

    Fits ``value(h) = value_0 + C h^p`` to at least three grids.  For every
    trial order on a scan grid the pair ``(value_0, C)`` is obtained by linear
    least squares and the residual is recorded; ``p`` is the minimiser, refined
    by a golden-section search.  A single three-grid Richardson estimate is
    *not* used, because the measured values here are not guaranteed monotone;
    the fit reports its own RMS so an ill-conditioned study is visible.

    Args:
        spacings: characteristic cell size per grid (``1/D``).
        values: measured quantity per grid (``C_D``).
    """
    if len(spacings) < 3 or len(values) != len(spacings):
        raise ValueError("need at least three (spacing, value) pairs")
    h = torch.tensor([float(v) for v in spacings], dtype=torch.float64)
    q = torch.tensor([float(v) for v in values], dtype=torch.float64)
    order = torch.argsort(h)
    h, q = h[order], q[order]
    if not bool((h > 0).all()):
        raise ValueError("spacings must be positive")

    def residual(p: float) -> tuple[float, float, float]:
        if p <= 0.0:
            return float("inf"), 0.0, 0.0
        basis = h**p
        # least squares on [1, basis]
        a_mat = torch.stack([torch.ones_like(basis), basis], dim=1)
        sol = torch.linalg.lstsq(a_mat, q.unsqueeze(1)).solution.reshape(-1)
        q0, c = float(sol[0]), float(sol[1])
        pred = q0 + c * basis
        rms = float(torch.sqrt(((q - pred) ** 2).mean()).item())
        return rms, q0, c

    # coarse scan then refine the minimiser
    grid = [0.05 * k for k in range(2, 141)]  # p in [0.1, 7.0]
    scored = [(residual(p)[0], p) for p in grid]
    _, p_best = min(scored)
    lo, hi = max(p_best - 0.05, 1e-3), p_best + 0.05
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c1 = hi - phi * (hi - lo)
        c2 = lo + phi * (hi - lo)
        if residual(c1)[0] < residual(c2)[0]:
            hi = c2
        else:
            lo = c1
    p_best = 0.5 * (lo + hi)
    rms, q0, _ = residual(p_best)

    # Grid convergence index of the finest grid (Roache), safety factor 1.25.
    h1, h2 = float(h[-1]), float(h[-2])
    q1, q2 = float(q[-1]), float(q[-2])
    r = h1 / h2
    denom = abs(q1) if abs(q1) > 0 else 1.0
    gci = 1.25 * abs(q1 - q2) / denom / max(r**p_best - 1.0, 1e-12) if p_best > 0 else float("nan")
    rel_rms = 100.0 * rms / max(abs(q0), 1e-12)
    return {
        "order": p_best,
        "extrapolated": q0,
        "fit_rms": rms,
        "fit_rms_pct": rel_rms,
        "gci_finest": 100.0 * gci,
        "n_grids": len(spacings),
    }


# --------------------------------------------------------------------------- #
# Solver: D3Q19 BGK/TRT + immersed-boundary surface force (periodic, body-force drive)
# --------------------------------------------------------------------------- #


def _default_dtype() -> torch.dtype:
    return torch.float64


_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[str(name).lower()]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; use 'float32' or 'float64'") from None


def _apply_guo_source(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    tau: float,
    *,
    guo: bool,
) -> torch.Tensor:
    """Add the momentum source of a body-force field to a post-collision ``f``.

    With ``guo=True`` the (1 - 1/(2 tau)) Guo factor is used together with the
    half-force equilibrium velocity, so the momentum injection is exactly
    ``F`` per node (the collision shift contributes ``F/(2 tau)``).
    """
    c = _c_on(f.device).to(f.dtype)
    w = _w_on(f.device, f.dtype).view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * fx.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * fy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * fz.unsqueeze(0)
    )
    g = (1.0 - 0.5 / tau) if guo else 1.0
    return f + w * 3.0 * g * cu


def _lbm_step(
    f: torch.Tensor,
    force: torch.Tensor,
    tau: float,
    *,
    stream,
    collision: str = "bgk",
    guo: bool = True,
    moments: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """One D3Q19 step with a body-force field ``force`` of shape (3, nz, ny, nx).

    ``moments`` may carry the ``(rho, ux, uy, uz)`` already computed by the
    caller (the immersed-boundary operator needs them anyway), which saves one
    full field sweep per step.
    """
    if moments is None:
        rho, ux, uy, uz = macroscopic3d(f)
    else:
        rho, ux, uy, uz = moments
    rho_safe = rho.clamp(min=1e-12)
    if guo:
        us = (
            ux + 0.5 * force[0] / rho_safe,
            uy + 0.5 * force[1] / rho_safe,
            uz + 0.5 * force[2] / rho_safe,
        )
    else:
        us = (ux, uy, uz)
    feq = equilibrium3d(rho, us[0], us[1], us[2])
    if collision == "bgk":
        f = f - (f - feq) / tau
    else:
        # TRT relaxes the populations recomputed from ``f``; the Guo factor and
        # the half-force equilibrium velocity must then be switched off, or the
        # momentum injected per step is (1 - 1/(2 tau)) F instead of F and the
        # recoil identity stops being exact.
        from .solver3d import collide_trt3d

        if guo:
            raise ValueError("collision='trt' requires guo=False (exact momentum source)")
        f = collide_trt3d(f, tau)
    f = _apply_guo_source(f, force[0], force[1], force[2], tau, guo=guo)
    return stream(f)


@dataclass
class SphereDragConfig:
    """Configuration of one flow-past-a-sphere drag case (lattice units)."""

    diameter: float = 12.0
    reynolds: float = 100.0
    lateral_diameters: float = 8.0
    streamwise_diameters: float = 8.0
    target_speed: float = 0.04
    n_theta: int = 24
    n_phi: int = 48
    kernel: str = "4pt"
    n_correctors: int = 2
    n_normal_passes: int = 1
    collision: str = "bgk"
    guo: bool = True
    dtype: str = "float64"
    max_steps: int = 40000
    min_steps: int = 4000
    report_interval: int = 200
    steady_tol: float = 2.0e-3
    steady_windows: int = 6
    drive_gain: float = 0.7
    axial_initial_flow: bool = True
    interior_rigid: bool = True
    snapshot_path: str | None = None

    def grid(self) -> tuple[int, int, int]:
        nz = max(8, int(round(self.lateral_diameters * self.diameter)))
        ny = nz
        nx = max(8, int(round(self.streamwise_diameters * self.diameter)))
        return nz, ny, nx

    @property
    def nu(self) -> float:
        return self.target_speed * self.diameter / self.reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu


def run_sphere_drag(config: SphereDragConfig, *, device: str = "cpu", verbose: bool = False) -> dict:
    """Steady/unsteady flow past a sphere: drag from the surface quadrature.

    The flow is driven by a uniform body force on the *fluid* nodes only (a
    Galilean frame with a periodic box), controlled proportionally to the
    measured mean fluid velocity.  In the statistically steady state the drive
    force is an IB-independent statement of the drag,

        ``rho g V_fluid = |F_body|``,

    which is reported next to the marker force as a cross-check that no
    momentum is created or lost by the immersed-boundary operator.
    """
    from .solver3d import stream3d

    torch.set_grad_enabled(False)
    dtype = _resolve_dtype(config.dtype)
    nz, ny, nx = config.grid()
    shape = (nz, ny, nx)
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    surface = build_sphere_surface(
        shape,
        center,
        config.diameter / 2.0,
        n_theta=config.n_theta,
        n_phi=config.n_phi,
        kernel=config.kernel,
    )

    # Start from the uniform approach flow: the sphere is impulsively started,
    # which removes the O(V/(U A)) momentum spin-up of a quiescent start.
    rho = torch.ones(shape, dtype=dtype)
    zeros = torch.zeros(shape, dtype=dtype)
    ux0 = torch.zeros(shape, dtype=dtype)
    f = equilibrium3d(rho, ux0, zeros, zeros)

    if config.axial_initial_flow:
        ux0 = config.target_speed * (~surface.mask).to(dtype)
        f = equilibrium3d(rho, ux0, zeros, zeros)
    n_fluid = int((~surface.mask).sum().item())
    fluid = (~surface.mask).to(dtype)
    u_target_markers = torch.zeros((surface.n_markers, 3), dtype=dtype)
    u_target_interior = torch.zeros(3, dtype=dtype)

    tau = config.tau
    nu = config.nu
    # Initial drive: match the adjudicated empirical drag at the target Re.
    area = 0.25 * math.pi * config.diameter * config.diameter
    cd_guess = cd_schiller_naumann(config.reynolds)
    g = 0.5 * config.target_speed**2 * area * cd_guess / max(n_fluid, 1)
    mass0 = float(f.sum().item())
    mass_prev = mass0
    history: list[dict] = []
    cd_window: list[float] = []
    steady = False
    reason = "max_steps"

    for step in range(1, config.max_steps + 1):
        rho_f, ux, uy, uz = macroscopic3d(f)
        u_grid = torch.stack([ux, uy, uz])
        tr = surface_traction(
            surface,
            u_grid,
            u_target_markers,
            rho_f,
            n_correctors=config.n_correctors,
            n_normal_passes=config.n_normal_passes,
        )
        if config.interior_rigid:
            f_int, bf_int = interior_rigid_force(surface, u_grid, rho_f, u_target_interior)
        else:
            f_int = torch.zeros_like(u_grid)
            bf_int = torch.zeros(3, dtype=dtype)

        drive = torch.zeros_like(f_int)
        drive[0] = g * rho_f * fluid

        force = tr.force_grid + f_int + drive
        f = _lbm_step(
            f,
            force,
            tau,
            stream=stream3d,
            collision=config.collision,
            guo=config.guo and config.collision == "bgk",
            moments=(rho_f, ux, uy, uz),
        )

        if step % config.report_interval == 0:
            rho_f, ux, uy, uz = macroscopic3d(f)
            u_fluid = float((ux * fluid).sum().item()) / max(n_fluid, 1)
            force_body = tr.force + bf_int
            cd = drag_coefficient(force_body, 1.0, abs(u_fluid), config.diameter)
            drive_force = g * float((rho_f * fluid).sum().item())
            cd_drive = drag_coefficient(
                torch.tensor([drive_force, 0.0, 0.0], dtype=dtype), 1.0, abs(u_fluid), config.diameter
            )
            mass = float(f.sum().item())
            mass_step = abs(mass - mass_prev) / abs(mass_prev) / config.report_interval
            mass_prev = mass
            history.append(
                {
                    "step": step,
                    "cd": cd,
                    "mass_step_residual": mass_step,
                    "cd_drive": cd_drive,
                    "ux_mean": u_fluid,
                    "g": g,
                    "reynolds": abs(u_fluid) * config.diameter / nu,
                    "mass_residual": abs(mass - mass0) / abs(mass0),
                    "force_body_x": float(force_body[0].item()),
                    "force_surface_x": float(tr.force[0].item()),
                    "force_interior_x": float(bf_int[0].item()),
                    "drive_force_x": drive_force,
                    "closure_residual": tr.closure_residual,
                    "slip_before": tr.slip_before,
                    "slip_after": tr.slip_after,
                    "max_mach": float((torch.sqrt(ux * ux + uy * uy + uz * uz) * fluid).max().item() / math.sqrt(1.0 / 3.0)),
                }
            )
            cd_window.append(cd)
            if verbose:
                print(
                    f"step {step:6d} Re={history[-1]['reynolds']:8.3f} "
                    f"Cd={cd:7.4f} Cd_drive={cd_drive:7.4f} g={g:.3e} "
                    f"mass={history[-1]['mass_residual']:.2e}",
                    flush=True,
                )
            # relative proportional controller on the measured fluid velocity
            if u_fluid > 0.0:
                ratio = config.target_speed / u_fluid
                g *= min(max(1.0 + config.drive_gain * (ratio * ratio - 1.0), 0.5), 1.5)
            if step >= config.min_steps and len(cd_window) >= 2 * config.steady_windows:
                recent = cd_window[-config.steady_windows :]
                prior = cd_window[-2 * config.steady_windows : -config.steady_windows]
                m_recent = sum(recent) / len(recent)
                m_prior = sum(prior) / len(prior)
                if abs(m_recent - m_prior) / max(abs(m_recent), 1e-12) < config.steady_tol:
                    steady = True
                    reason = "steady"
                    break

    if config.snapshot_path:
        rho_f, ux, uy, uz = macroscopic3d(f)
        j = nz // 2
        # vorticity of the (x, z) mid-plane: omega_y = du_x/dz - du_z/dx
        dux_dz = torch.gradient(ux, dim=0)[0]
        duz_dx = torch.gradient(uz, dim=2)[0]
        snapshot = {
            "ux": ux[j].cpu(),
            "uy": uy[j].cpu(),
            "uz": uz[j].cpu(),
            "rho": rho_f[j].cpu(),
            "omega_y": (dux_dz - duz_dx)[j].cpu(),
            "mask": surface.mask[j].cpu(),
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "diameter": config.diameter,
            "reynolds_measured": float(history[-1]["reynolds"]) if history else float("nan"),
            "steps": history[-1]["step"] if history else 0,
        }
        torch.save(snapshot, config.snapshot_path)

    # statistics on the second half of the recorded history
    half = history[len(history) // 2 :] or history
    cds = [h["cd"] for h in half]
    cd_mean = sum(cds) / len(cds)
    cd_std = (sum((c - cd_mean) ** 2 for c in cds) / max(len(cds) - 1, 1)) ** 0.5
    re_mean = sum(h["reynolds"] for h in half) / len(half)
    ux_mean = sum(h["ux_mean"] for h in half) / len(half)
    cd_drive_mean = sum(h["cd_drive"] for h in half) / len(half)
    mass_res = max(h["mass_residual"] for h in history)
    closure = max(h["closure_residual"] for h in history)
    anchor = adjudicate_empirical(re_mean)
    return {
        "case": "drag",
        "config": {
            "diameter": config.diameter,
            "reynolds_target": config.reynolds,
            "lateral_diameters": config.lateral_diameters,
            "streamwise_diameters": config.streamwise_diameters,
            "target_speed": config.target_speed,
            "n_markers": surface.n_markers,
            "n_theta": config.n_theta,
            "n_phi": config.n_phi,
            "kernel": config.kernel,
            "n_correctors": config.n_correctors,
            "n_normal_passes": config.n_normal_passes,
            "collision": config.collision,
            "guo": config.guo and config.collision == "bgk",
            "dtype": str(dtype).replace("torch.", ""),
            "tau": tau,
            "nu": nu,
            "grid": [nz, ny, nx],
            "n_fluid": n_fluid,
            "steps": history[-1]["step"] if history else 0,
            "max_steps": config.max_steps,
        },
        "steady": steady,
        "stop_reason": reason,
        "cd_mean": cd_mean,
        "cd_std": cd_std,
        "cd_drive_mean": cd_drive_mean,
        "cd_vs_schiller_naumann_pct": 100.0 * (cd_mean - anchor["schiller_naumann"]) / anchor["schiller_naumann"],
        "cd_vs_clift_gauvin_pct": 100.0 * (cd_mean - anchor["clift_gauvin"]) / anchor["clift_gauvin"],
        "reynolds_measured": re_mean,
        "ux_measured": ux_mean,
        "max_mass_residual": mass_res,
        "max_closure_residual": closure,
        "anchor": anchor,
        "history": history,
    }


@dataclass
class SphereRotationConfig:
    """Configuration of one steadily-rotating-sphere torque case (lattice units)."""

    diameter: float = 12.0
    re_omega: float = 1.0
    box_diameters: float = 8.0
    tip_speed: float = 0.02
    n_theta: int = 24
    n_phi: int = 48
    kernel: str = "4pt"
    n_correctors: int = 2
    n_normal_passes: int = 1
    collision: str = "bgk"
    guo: bool = True
    dtype: str = "float64"
    max_steps: int = 40000
    min_steps: int = 4000
    report_interval: int = 200
    steady_tol: float = 2.0e-3
    steady_windows: int = 6

    def grid(self) -> tuple[int, int, int]:
        n = max(8, int(round(self.box_diameters * self.diameter)))
        return n, n, n

    @property
    def omega(self) -> float:
        return self.tip_speed / (0.5 * self.diameter)

    @property
    def nu(self) -> float:
        # Re_omega = U_tip D / nu
        return self.tip_speed * self.diameter / self.re_omega

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu


def run_sphere_rotation(
    config: SphereRotationConfig, *, device: str = "cpu", verbose: bool = False
) -> dict:
    """Torque on a steadily rotating sphere in a quiescent periodic box.

    The sphere rotates about ``+z`` through its centre.  Two estimators are
    recorded per report interval: the **traction quadrature** of
    :func:`surface_traction` (plus the interior recoil) and the
    **pressure-only** integral of :func:`pressure_only_torque` (the G12 route),
    which on a purely azimuthal flow returns ~0 and therefore cannot certify a
    torque.  ``C_M`` is normalised with the tip speed and the frontal area.
    """
    from .solver3d import stream3d

    torch.set_grad_enabled(False)
    dtype = _resolve_dtype(config.dtype)
    nz, ny, nx = config.grid()
    shape = (nz, ny, nx)
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    surface = build_sphere_surface(
        shape,
        center,
        config.diameter / 2.0,
        n_theta=config.n_theta,
        n_phi=config.n_phi,
        kernel=config.kernel,
    )

    rho = torch.ones(shape, dtype=dtype)
    zeros = torch.zeros(shape, dtype=dtype)
    f = equilibrium3d(rho, zeros, zeros, zeros)

    rel = (surface.points - surface.center.unsqueeze(0)).to(dtype)
    omega = torch.tensor([0.0, 0.0, config.omega], dtype=dtype)
    u_target_markers = torch.cross(omega.unsqueeze(0).expand(rel.shape[0], 3), rel, dim=1)
    u_target_markers = u_target_markers.contiguous()

    yy = torch.arange(ny, dtype=dtype).view(1, ny, 1)
    xx = torch.arange(nx, dtype=dtype).view(1, 1, nx)
    rx = xx - surface.center[0]
    ry = yy - surface.center[1]
    bx, by = torch.broadcast_tensors(rx, ry)
    bx = bx.expand(nz, ny, nx)
    by = by.expand(nz, ny, nx)
    u_target_interior = torch.stack(
        [-config.omega * by, config.omega * bx, torch.zeros((nz, ny, nx), dtype=dtype)]
    ).contiguous()

    tau = config.tau
    nu = config.nu
    mass0 = float(f.sum().item())
    history: list[dict] = []
    window: list[float] = []
    steady = False
    reason = "max_steps"

    for step in range(1, config.max_steps + 1):
        rho_f, ux, uy, uz = macroscopic3d(f)
        u_grid = torch.stack([ux, uy, uz])
        tr = surface_traction(
            surface,
            u_grid,
            u_target_markers,
            rho_f,
            n_correctors=config.n_correctors,
            n_normal_passes=config.n_normal_passes,
        )
        f_int, _ = interior_rigid_force(surface, u_grid, rho_f, u_target_interior)
        # torque recoil of the interior (rigid-body rotation about +z)
        f_int_z = f_int * surface.mask.unsqueeze(0)
        mz_int = -(rx * f_int_z[1] - ry * f_int_z[0]).sum()
        force = tr.force_grid + f_int
        f = _lbm_step(
            f,
            force,
            tau,
            stream=stream3d,
            collision=config.collision,
            guo=config.guo and config.collision == "bgk",
            moments=(rho_f, ux, uy, uz),
        )

        if step % config.report_interval == 0:
            rho_f, ux, uy, uz = macroscopic3d(f)
            torque_total = tr.torque + torch.tensor([0.0, 0.0, float(mz_int.item())], dtype=dtype)
            cm = torque_coefficient(torque_total, 1.0, config.tip_speed, config.diameter)
            cm_pressure_only = torque_coefficient(
                pressure_only_torque(surface, rho_f, torch.stack([ux, uy, uz])),
                1.0,
                config.tip_speed,
                config.diameter,
            )
            mass = float(f.sum().item())
            history.append(
                {
                    "step": step,
                    "cm": cm,
                    "cm_stokes": stokes_torque_coefficient(config.re_omega),
                    "cm_pressure_only": cm_pressure_only,
                    "torque_z": float(torque_total[2].item()),
                    "torque_z_marker": float(tr.torque[2].item()),
                    "torque_z_interior": float(mz_int.item()),
                    "mass_residual": abs(mass - mass0) / abs(mass0),
                    "closure_residual": tr.closure_residual,
                }
            )
            window.append(cm)
            if verbose:
                print(
                    f"step {step:6d} C_M={cm:.5f} (Stokes {history[-1]['cm_stokes']:.5f}) "
                    f"C_M[pressure-only]={cm_pressure_only:.5f} mass={history[-1]['mass_residual']:.2e}",
                    flush=True,
                )
            if step >= config.min_steps and len(window) >= 2 * config.steady_windows:
                recent = window[-config.steady_windows :]
                prior = window[-2 * config.steady_windows : -config.steady_windows]
                m_recent = sum(recent) / len(recent)
                m_prior = sum(prior) / len(prior)
                if abs(m_recent - m_prior) / max(abs(m_recent), 1e-12) < config.steady_tol:
                    steady = True
                    reason = "steady"
                    break

    half = history[len(history) // 2 :] or history
    cms = [h["cm"] for h in half]
    cm_mean = sum(cms) / len(cms)
    cm_std = (sum((c - cm_mean) ** 2 for c in cms) / max(len(cms) - 1, 1)) ** 0.5
    cm_po = sum(h["cm_pressure_only"] for h in half) / len(half)
    reference = stokes_torque_coefficient(config.re_omega)
    return {
        "case": "rotation",
        "config": {
            "diameter": config.diameter,
            "re_omega_target": config.re_omega,
            "tip_speed": config.tip_speed,
            "omega": config.omega,
            "box_diameters": config.box_diameters,
            "n_markers": surface.n_markers,
            "n_theta": config.n_theta,
            "n_phi": config.n_phi,
            "kernel": config.kernel,
            "n_correctors": config.n_correctors,
            "n_normal_passes": config.n_normal_passes,
            "collision": config.collision,
            "guo": config.guo and config.collision == "bgk",
            "dtype": str(dtype).replace("torch.", ""),
            "tau": tau,
            "nu": nu,
            "grid": [nz, ny, nx],
            "steps": history[-1]["step"] if history else 0,
            "max_steps": config.max_steps,
        },
        "steady": steady,
        "stop_reason": reason,
        "cm_mean": cm_mean,
        "cm_std": cm_std,
        "cm_pressure_only_mean": cm_po,
        "cm_stokes_reference": reference,
        "cm_vs_stokes_pct": 100.0 * (cm_mean - reference) / reference,
        "max_mass_residual": max(h["mass_residual"] for h in history),
        "max_closure_residual": max(h["closure_residual"] for h in history),
        "history": history,
    }
