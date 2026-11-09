"""Kinetic-flux compressible solver (lattice-Boltzmann equilibrium flux, KFVS).

Motivation
----------
The DDF LBM path in :mod:`tensorlbm.compressible.compressible_lbm` is limited to
lattice Mach numbers of a few tenths and cannot represent transonic shocks.
This module supplies the *shock-capturing* compressible face of TensorLBM with a
kinetic scheme whose numerical flux is the **half-space moment of the
lattice-Boltzmann (Maxwellian) equilibrium** -- the Boltzmann equilibrium flux
used as the flux of a finite-volume discretisation (Kinetic Flux Vector
Splitting: Pullin 1980, Mandal & Deshpande 1994, Xu 1998 gas-kinetic lineage).
It is therefore an LBM-family method: the same equilibrium distribution that
propagates populations on a lattice is here integrated analytically over half
the velocity space to build an upwind flux.

Implemented
-----------
* :func:`kfvs_flux` -- analytic half-space Maxwellian flux for arbitrary
  ``gamma`` (monatomic gamma=5/3 form generalised with
  ``beta = gamma/(gamma-1) - 3/2``).
* :class:`AnnulusMesh` -- body-fitted O-grid from radial extrusion of a closed
  airfoil contour (periodic in ``i``, wall at ``j=0``, far field at ``j=nj-1``).
* :class:`KineticFluxSolver` -- MUSCL-Hancock second order + minmod limiter,
  slip-wall pressure flux, characteristic far field, local time stepping,
  entropy shock sensor + *localised* artificial viscosity.
* :class:`NormalShock1D` -- steady normal shock / shock tube driver used to
  verify the scheme against Rankine-Hugoniot (machine-checkable anchor).
* Diagnostics: wall ``Cp``, force integration, shock location, entropy
  production monitor, total-enthalpy monitor.

Evidence discipline
-------------------
Verification anchors are analytic (Rankine-Hugoniot, Ackeret linear supersonic
theory, Karman-Tsien compressibility correction).  Experimental (AGARD)
surface-pressure data are *not* shipped in this repository;
:func:`experimental_anchor_status` reports ``WITHHELD`` unless the caller
supplies a digitised dataset with provenance.  No experimental value is
fabricated anywhere in this module.

References
----------
* Pullin, D.I., J. Comput. Phys. 34, 231-244, 1980.
* Mandal, J.C., Deshpande, S.M., Computers & Fluids 23(2), 447-478, 1994.
* Xu, K., "Gas-kinetic schemes ...", VKI Lecture Series, 1998.
* Toro, E.F., "Riemann Solvers and Numerical Methods for Fluid Dynamics", 2009.
* Ackeret, J., Z. Flugtech. Motorluftsch. 16, 129-136, 1925.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is already a test dependency of this repository
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - defensive
    _erf = np.vectorize(math.erf, otypes=[float])


GAMMA = 1.4
_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Kinetic (KFVS) flux
# ---------------------------------------------------------------------------
def _half_space_moments(q: np.ndarray, sigma: np.ndarray):
    """Half-space moments of the 1D Maxwellian with mean ``q``, ``sigma=sqrt(RT)``.

    Returns ``(g1p, g2p, g3p, g1m, g2m, g3m)`` where
    ``gk± = int_{xi><0} xi^k f_M dxi / rho`` (k = 1, 2, 3).
    Closed form (erf/exp), no quadrature error.
    """
    s = q / sigma
    phi = np.exp(-0.5 * s * s) / _SQRT_2PI
    Phi = 0.5 * (1.0 + _erf(s / _SQRT2))
    Psi = 1.0 - Phi
    q2 = q * q
    s2 = sigma * sigma
    g1p = q * Phi + sigma * phi
    g1m = q * Psi - sigma * phi
    g2p = (q2 + s2) * Phi + q * sigma * phi
    g2m = (q2 + s2) * Psi - q * sigma * phi
    common = q * q2 + 3.0 * q * s2
    g3p = common * Phi + (2.0 * sigma * s2 + q2 * sigma) * phi
    g3m = common * Psi - (2.0 * sigma * s2 + q2 * sigma) * phi
    return g1p, g2p, g3p, g1m, g2m, g3m


def internal_energy_correction(gamma: float = GAMMA) -> float:
    """``beta = gamma/(gamma-1) - 3/2``: generalises the monatomic energy flux."""
    return gamma / (gamma - 1.0) - 1.5


def kfvs_flux(rhoL, uL, vL, pL, rhoR, uR, vR, pR, nx, ny, gamma: float = GAMMA):
    """Kinetic (equilibrium) flux through a face with unit normal ``(nx, ny)``.

    Positive velocity half space taken from the left state, negative from the
    right state -- exactly the LBM equilibrium populations split by propagation
    direction.  Returns ``(F_rho, F_rho_u, F_rho_v, F_E)`` per unit area.
    """
    beta = internal_energy_correction(gamma)
    qL = uL * nx + vL * ny
    tL = -uL * ny + vL * nx
    qR = uR * nx + vR * ny
    tR = -uR * ny + vR * nx
    sigL = np.sqrt(pL / rhoL)
    sigR = np.sqrt(pR / rhoR)

    g1p, g2p, g3p, _, _, _ = _half_space_moments(qL, sigL)
    _, _, _, g1m, g2m, g3m = _half_space_moments(qR, sigR)

    F_rho = rhoL * g1p + rhoR * g1m
    F_mn = rhoL * g2p + rhoR * g2m
    F_mt = rhoL * tL * g1p + rhoR * tR * g1m
    F_E = (0.5 * (rhoL * (g3p + g1p * tL * tL) + rhoR * (g3m + g1m * tR * tR))
           + beta * (rhoL * g1p * sigL * sigL + rhoR * g1m * sigR * sigR))

    F_mx = F_mn * nx - F_mt * ny
    F_my = F_mn * ny + F_mt * nx
    return F_rho, F_mx, F_my, F_E


def rusanov_dissipation(rhoL, uL, vL, pL, rhoR, uR, vR, pR, nx, ny,
                        coeff: float, gamma: float = GAMMA):
    """Upwind dissipation floor (entropy fix) added to the kinetic flux.

    The pure equilibrium splitting of the momentum flux uses the *average* of
    the left/right pressures, so a grid-aligned stationary shock can support an
    odd-even (sawtooth) mode.  A Rusanov term scaled by ``coeff`` removes that
    mode; it vanishes for equal states and is therefore second-order neutral in
    smooth flow.  ``coeff = 0`` recovers the plain KFVS flux.
    """
    if coeff <= 0.0:
        return tuple(np.zeros_like(F) for F in (rhoL, rhoL, rhoL, rhoL))
    UL = prim_to_cons(rhoL, uL, vL, pL, gamma)
    UR = prim_to_cons(rhoR, uR, vR, pR, gamma)
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    qL = np.abs(uL * nx + vL * ny)
    qR = np.abs(uR * nx + vR * ny)
    sp = 0.5 * (qL + aL + qR + aR)
    return tuple(coeff * sp * (UL[k] - UR[k]) for k in range(4))



def numerical_flux_x(rhoL, uL, pL, rhoR, uR, pR, mode: str = "kfvs_energy_upwind",
                     gamma: float = GAMMA):
    """1D numerical flux (mass, momentum, energy) with selectable closure.

    ``mode``:
      * ``"kfvs"`` -- pure equilibrium (Maxwellian) splitting.
      * ``"kfvs_energy_upwind"`` -- kinetic mass/momentum flux (gamma-independent,
        exact Euler limit) plus a Rusanov-upwinded total-energy flux.  The
        pseudo-Maxwellian internal-energy closure required for gamma != 5/3 is
        not guaranteed dissipative, so the energy flux is taken upwind.
      * ``"rusanov"`` -- local Lax-Friedrichs (robust reference flux).
    """
    if mode == "kfvs":
        z = np.zeros_like(rhoL)
        o = np.ones_like(rhoL)
        F = kfvs_flux(rhoL, uL, z, pL, rhoR, uR, z, pR, o, z, gamma)
        return F[0], F[1], F[3]
    EL = pL / (gamma - 1.0) + 0.5 * rhoL * uL * uL
    ER = pR / (gamma - 1.0) + 0.5 * rhoR * uR * uR
    FL = np.stack([rhoL * uL, rhoL * uL * uL + pL, (EL + pL) * uL])
    FR = np.stack([rhoR * uR, rhoR * uR * uR + pR, (ER + pR) * uR])
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    lam = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
    UL = np.stack([rhoL, rhoL * uL, EL])
    UR = np.stack([rhoR, rhoR * uR, ER])
    F = 0.5 * (FL + FR) - 0.5 * lam * (UR - UL)
    if mode == "rusanov":
        return F[0], F[1], F[2]
    z = np.zeros_like(rhoL)
    o = np.ones_like(rhoL)
    Fk = kfvs_flux(rhoL, uL, z, pL, rhoR, uR, z, pR, o, z, gamma)
    return Fk[0], Fk[1], F[2]


def prim_to_cons(rho, u, v, p, gamma: float = GAMMA) -> np.ndarray:
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E])


def cons_to_prim(U: np.ndarray, gamma: float = GAMMA):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


# ---------------------------------------------------------------------------
# Profiles and body-fitted mesh
# ---------------------------------------------------------------------------
def naca0012_half_thickness(x: np.ndarray, t: float = 0.12,
                            closed_te: bool = True) -> np.ndarray:
    """NACA 4-digit half-thickness distribution.

    ``closed_te=True`` uses the ``-0.1036 x^4`` coefficient (zero thickness at
    ``x=1``); ``closed_te=False`` uses ``-0.1015 x^4``, i.e. the 0.21 % c thick
    open trailing edge of Abbott & von Doenhoff (1959) and of the AGARD
    transonic test cases.  The open TE is required for a body-fitted O-grid: a
    zero-thickness TE makes the first TE cell a sliver of area ~1e-10 c^2 that
    destabilises the explicit update.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    c4 = 0.1036 if closed_te else 0.1015
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                      + 0.2843 * x**3 - c4 * x**4)


def biconvex_half_thickness(x: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """Bi-convex profile ``y = +-2 tau x (1-x)`` (linear slope -> N-wave)."""
    x = np.asarray(x, dtype=float)
    return 2.0 * tau * x * (1.0 - x)


def diamond_half_thickness(x: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """Diamond (double-wedge) profile, ``tau`` = half thickness ratio."""
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.5, 2.0 * tau * x, 2.0 * tau * (1.0 - x))


def contour_loop(half_thickness: Callable[[np.ndarray], np.ndarray],
                 n_points: int = 200, chord: float = 1.0):
    """Closed body contour (cosine clustered at LE/TE), points periodic."""
    m = n_points // 2 + 1
    beta = np.linspace(0.0, math.pi, m)
    xs = 0.5 * (1.0 + np.cos(beta))            # 1 -> 0 (TE -> LE)
    yt = np.asarray(half_thickness(xs), dtype=float)
    xu = chord * xs
    yu = chord * yt
    x_loop = np.concatenate([xu[:-1], xu[::-1][1:]])
    y_loop = np.concatenate([yu[:-1], -yu[::-1][1:]])
    return x_loop, y_loop


def radial_levels(nj: int, r0: float, r_out: float, stretch: float = 2.2) -> np.ndarray:
    """Radial node distribution clustered at the wall.

    .. warning::
       The first level is ``r0``, i.e. the *inner boundary of the O-grid is an
       offset curve* ``r0`` away from the body, not the body surface.  For a
       NACA0012 leading edge (nose radius ``1.58e-2 c``) any ``r0`` comparable
       to the nose radius makes that offset curve fold onto itself, which
       corrupts the wall pressure and creates spurious entropy drag.
       Use :func:`radial_levels_from_surface` (body surface = grid line 0) for
       production runs; this helper is kept for backward compatibility only.
    """
    eta = np.linspace(0.0, 1.0, nj)
    if stretch <= 0.0:
        f = eta
    else:
        f = 0.5 * (1.0 + np.tanh(stretch * (2.0 * eta - 1.0)) / math.tanh(stretch))
        f = f / f[-1]
    return r0 + (r_out - r0) * f


def radial_levels_from_surface(nj: int, r_out: float,
                               first_cell: float = 0.002,
                               growth: Optional[float] = None) -> np.ndarray:
    """Radial levels that **start on the body surface** (``r = 0``).

    ``nj`` nodes, geometric growth from a first cell height ``first_cell``
    (in chords) up to the far field ``r_out``: the growth ratio is *solved* so
    that the ladder spans ``r_out`` exactly, which keeps ``first_cell`` intact
    for every ``nj``.  Using this distribution the
    first grid line *is* the airfoil surface, so the wall pressure/force
    integration is evaluated on the true geometry.
    """
    n = int(nj) - 1
    if n < 1:
        raise ValueError("nj must be >= 2")
    first_cell = float(first_cell)
    r_out = float(r_out)
    if first_cell * n >= r_out or first_cell <= 0.0:
        raise ValueError("first_cell incompatible with nj / r_out")
    if n == 1:
        return np.array([0.0, r_out])
    # growth ratio solved so that the geometric ladder spans exactly r_out;
    # never rescale (rescaling a fixed ratio crushes the wall cells to nothing
    # as soon as nj grows -- that used to make the wall cell height collapse
    # from 2e-3 c to 3e-5 c between nj=64 and nj=96 and blew the solver up).
    lo, hi = 1.0 + 1e-12, 4.0
    for _ in range(200):
        q = 0.5 * (lo + hi)
        if first_cell * (q ** n - 1.0) / (q - 1.0) < r_out:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    if growth is not None:
        q = min(q, max(float(growth), 1.0 + 1e-12))
    dr = first_cell * q ** np.arange(n)
    if dr.sum() > r_out:            # ratio capped by ``growth``: rescale the tail
        dr = dr / dr.sum() * r_out
    return np.concatenate([[0.0], np.cumsum(dr)])


def mesh_quality(mesh) -> dict:
    """Signed-area / fold diagnostics of a structured O-grid.

    ``build_annulus_mesh`` stores ``|area|``; a self-intersecting (folded) quad
    still has a small positive ``area`` but its four corner cross products do
    not share a sign.  Folded cells break the face-normal bookkeeping of the
    finite-volume update, so refinement studies must check this first.
    """
    X, Y = mesh.X, mesh.Y
    ip = np.roll(np.arange(mesh.nc), -1)
    p0x, p0y = X[:, :-1], Y[:, :-1]
    p1x, p1y = X[ip, :-1], Y[ip, :-1]
    p2x, p2y = X[ip, 1:], Y[ip, 1:]
    p3x, p3y = X[:, 1:], Y[:, 1:]
    c0 = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
    c1 = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
    c2 = (p3x - p2x) * (p0y - p2y) - (p3y - p2y) * (p0x - p2x)
    c3 = (p0x - p3x) * (p1y - p3y) - (p0y - p3y) * (p1x - p3x)
    cr = np.stack([c0, c1, c2, c3])
    signed = 0.5 * (c0 + c2)
    folded = ~((cr > 0).all(axis=0) | (cr < 0).all(axis=0))
    return {"n_cells": int(mesh.nc * (mesh.nj - 1)),
            "n_folded": int(folded.sum()),
            "folded_cells": np.argwhere(folded).tolist()[:10],
            "min_abs_area": float(np.abs(signed).min()),
            "min_signed_area": float(signed.min()),
            "max_area": float(np.abs(signed).max())}


@dataclass
class AnnulusMesh:
    """Body-fitted O-grid: ``nc`` x ``(nj-1)`` cells, periodic in ``i``."""

    X: np.ndarray
    Y: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    area: np.ndarray
    iface_nx: np.ndarray
    iface_ny: np.ndarray
    iface_len: np.ndarray
    jface_nx: np.ndarray
    jface_ny: np.ndarray
    jface_len: np.ndarray
    wall_x: np.ndarray
    wall_y: np.ndarray
    nc: int = 0
    nj: int = 0

    @property
    def n_cells(self) -> int:
        return int(self.nc * (self.nj - 1))


def build_annulus_mesh(x_loop, y_loop, r_levels, centroid=(0.4, 0.0),
                       blend: bool = True) -> AnnulusMesh:
    """Radially extrude a closed contour into a body-fitted O-grid.

    ``r_levels`` is either a 1-D ladder of ``nj`` offsets shared by every
    contour point, or a ``(nc, nj)`` array of per-point offsets (per-point
    ladders let the near-wall spacing follow the local surface spacing).  If
    ``r_levels[..., 0] == 0`` the first grid
    line coincides with the body surface, so ``wall_x/wall_y`` are the true
    surface points (this is what force integration and wall ``Cp`` need).
    """
    x_loop = np.asarray(x_loop, dtype=float)
    y_loop = np.asarray(y_loop, dtype=float)
    nc = x_loop.size
    r_levels = np.asarray(r_levels, dtype=float)
    if r_levels.ndim == 1:
        r_levels = np.broadcast_to(r_levels[None, :], (nc, r_levels.size)).copy()
    if r_levels.shape[0] != nc:
        raise ValueError("r_levels must have one row per contour point")
    nj = r_levels.shape[1]

    # contour tangent -> outward normal
    tx = np.roll(x_loop, -1) - np.roll(x_loop, 1)
    ty = np.roll(y_loop, -1) - np.roll(y_loop, 1)
    nx, ny = ty, -tx
    sgn = np.sign(nx * (x_loop - centroid[0]) + ny * (y_loop - centroid[1]))
    sgn[sgn == 0.0] = 1.0
    nx *= sgn
    ny *= sgn
    nn = np.hypot(nx, ny)
    nx /= nn
    ny /= nn

    X = x_loop[:, None] + r_levels * nx[:, None]
    Y = y_loop[:, None] + r_levels * ny[:, None]
    if blend:
        # Blend the radial extrusion towards a polar ring around the body
        # centroid: the extrusion alone self-intersects where the surface is
        # concave (trailing-edge closure), the polar ring never does.
        f = np.linspace(0.0, 1.0, nj)
        w = 0.5 * (1.0 - np.cos(np.pi * f))
        th = np.arctan2(y_loop - centroid[1], x_loop - centroid[0])
        R_out = r_levels[:, -1][:, None]
        Xp = centroid[0] + R_out * np.cos(th)[:, None]
        Yp = centroid[1] + R_out * np.sin(th)[:, None]
        # keep the inner rings close to the extrusion, the outer ring circular
        wj = w[None, :] ** 2
        X = (1.0 - wj) * X + wj * Xp
        Y = (1.0 - wj) * Y + wj * Yp

    ip = np.roll(np.arange(nc), -1)
    p0x, p0y = X, Y
    p1x, p1y = X[ip, :], Y[ip, :]
    p2x, p2y = X[ip, 1:], Y[ip, 1:]
    p3x, p3y = X[:, 1:], Y[:, 1:]
    xc = 0.25 * (p0x[:, :-1] + p1x[:, :-1] + p2x + p3x)
    yc = 0.25 * (p0y[:, :-1] + p1y[:, :-1] + p2y + p3y)
    cross = (p0x[:, :-1] * p1y[:, :-1] - p1x[:, :-1] * p0y[:, :-1]
             + p1x[:, :-1] * p2y - p2x * p1y[:, :-1]
             + p2x * p3y - p3x * p2y
             + p3x * p0y[:, :-1] - p0x[:, :-1] * p3y)
    area = 0.5 * np.abs(cross)

    # i-faces: nodes (f, j)-(f, j+1); normal -> +i
    fx0, fy0 = X[:, :-1], Y[:, :-1]
    fx1, fy1 = X[:, 1:], Y[:, 1:]
    sx, sy = fx1 - fx0, fy1 - fy0
    ilen = np.hypot(sx, sy)
    inx, iny = sy / ilen, -sx / ilen
    fmx = 0.5 * (fx0 + fx1) - centroid[0]
    fmy = 0.5 * (fy0 + fy1) - centroid[1]
    # one *global* orientation (never per-face sign tests: they flip where the
    # radius vector is parallel to the face and destroy the metric closure)
    if float(np.sum(inx * fmx + iny * fmy)) < 0.0:
        inx = -inx
        iny = -iny

    # j-faces: nodes (i, j)-(i+1, j); normal -> +j
    jx0, jy0 = X, Y
    jx1, jy1 = X[ip, :], Y[ip, :]
    gx, gy = jx1 - jx0, jy1 - jy0
    jlen = np.hypot(gx, gy)
    jnx, jny = gy / jlen, -gx / jlen
    gmx = 0.5 * (jx0 + jx1) - centroid[0]
    gmy = 0.5 * (jy0 + jy1) - centroid[1]
    if float(np.sum(jnx * gmx + jny * gmy)) < 0.0:
        jnx = -jnx
        jny = -jny

    return AnnulusMesh(
        X=X, Y=Y, xc=xc, yc=yc, area=area,
        iface_nx=inx, iface_ny=iny, iface_len=ilen,
        jface_nx=jnx, jface_ny=jny, jface_len=jlen,
        wall_x=0.5 * (jx0[:, 0] + jx1[:, 0]), wall_y=0.5 * (jy0[:, 0] + jy1[:, 0]),
        nc=nc, nj=nj,
    )


# ---------------------------------------------------------------------------
# 2D solver
# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    gamma: float = GAMMA
    cfl: float = 0.45
    order: int = 2
    av_coeff: float = 0.0            # localised artificial viscosity
    av_sensor_floor: float = 0.01
    rho_inf: float = 1.0
    p_inf: float = 1.0
    alpha_deg: float = 0.0
    mach: float = 0.0

    def freestream(self) -> Tuple[float, float, float, float]:
        a = math.sqrt(self.gamma * self.p_inf / self.rho_inf)
        u = self.mach * a * math.cos(math.radians(self.alpha_deg))
        v = self.mach * a * math.sin(math.radians(self.alpha_deg))
        return self.rho_inf, u, v, self.p_inf


class KineticFluxSolver:
    """Explicit finite-volume KFVS solver on a body-fitted O-grid."""

    def __init__(self, mesh: AnnulusMesh, cfg: SolverConfig):
        self.mesh = mesh
        self.cfg = cfg
        self.gamma = cfg.gamma
        rho, u, v, p = cfg.freestream()
        shape = (mesh.nc, mesh.nj - 1)
        self.W = [np.full(shape, rho), np.full(shape, u),
                  np.full(shape, v), np.full(shape, p)]
        self.history: List[dict] = []
        self.calls = 0

    # -- reconstruction ---------------------------------------------------
    def _slopes(self):
        cells = list(self.W)
        di, dj = [], []
        for Wv in cells:
            di.append(_minmod(Wv - np.roll(Wv, 1, axis=0),
                              np.roll(Wv, -1, axis=0) - Wv))
            df = np.empty_like(Wv)
            db = np.empty_like(Wv)
            df[:, :-1] = Wv[:, 1:] - Wv[:, :-1]
            df[:, -1] = df[:, -2]
            db[:, 1:] = Wv[:, 1:] - Wv[:, :-1]
            db[:, 0] = db[:, 1]
            dj.append(_minmod(db, df))
        return di, dj

    def _face_states(self):
        m = self.mesh
        cells = list(self.W)
        if self.cfg.order >= 2:
            di, dj = self._slopes()
        else:
            di = [np.zeros_like(c) for c in cells]
            dj = [np.zeros_like(c) for c in cells]
        Wi_L, Wi_R = [], []
        for c, d in zip(cells, di):
            Wi_L.append(np.roll(c + 0.5 * d, 1, axis=0))
            Wi_R.append(c - 0.5 * d)
        Wj_L, Wj_R = [], []
        for c, d in zip(cells, dj):
            aL = np.empty((m.nc, m.nj))
            aR = np.empty((m.nc, m.nj))
            aL[:, 1:-1] = c[:, :-1] + 0.5 * d[:, :-1]
            aR[:, 1:-1] = c[:, 1:] - 0.5 * d[:, 1:]
            aL[:, 0] = c[:, 0]
            aR[:, 0] = c[:, 0]
            aL[:, -1] = c[:, -1]
            aR[:, -1] = c[:, -1]
            Wj_L.append(aL)
            Wj_R.append(aR)
        # positivity guard on the reconstructed states (rho, p >= floor)
        for group in (Wi_L, Wi_R, Wj_L, Wj_R):
            group[0] = np.maximum(group[0], 1e-8)
            group[3] = np.maximum(group[3], 1e-8)
        return Wi_L, Wi_R, Wj_L, Wj_R

    # -- residual ---------------------------------------------------------
    def residual(self):
        m, g = self.mesh, self.gamma
        rho, u, v, p = self.W
        Wi_L, Wi_R, Wj_L, Wj_R = self._face_states()

        Fi = list(kfvs_flux(Wi_L[0], Wi_L[1], Wi_L[2], Wi_L[3],
                            Wi_R[0], Wi_R[1], Wi_R[2], Wi_R[3],
                            m.iface_nx, m.iface_ny, g))
        Fi = [f * m.iface_len for f in Fi]

        Fj = [np.zeros((m.nc, m.nj)) for _ in range(4)]
        jf = kfvs_flux(Wj_L[0][:, 1:-1], Wj_L[1][:, 1:-1], Wj_L[2][:, 1:-1],
                       Wj_L[3][:, 1:-1],
                       Wj_R[0][:, 1:-1], Wj_R[1][:, 1:-1], Wj_R[2][:, 1:-1],
                       Wj_R[3][:, 1:-1],
                       m.jface_nx[:, 1:-1], m.jface_ny[:, 1:-1], g)
        for k in range(4):
            Fj[k][:, 1:-1] = jf[k] * m.jface_len[:, 1:-1]

        # slip wall: pure pressure flux (adiabatic)
        pw = p[:, 0]
        Fj[1][:, 0] = pw * m.jface_nx[:, 0] * m.jface_len[:, 0]
        Fj[2][:, 0] = pw * m.jface_ny[:, 0] * m.jface_len[:, 0]

        # far field: KFVS with freestream outside (characteristic upwinding)
        rinf, uinf, vinf, pinf = self.cfg.freestream()
        n_out = m.nj - 1
        sh = (m.nc,)
        fo = kfvs_flux(Wj_L[0][:, n_out], Wj_L[1][:, n_out], Wj_L[2][:, n_out],
                       Wj_L[3][:, n_out],
                       np.full(sh, rinf), np.full(sh, uinf),
                       np.full(sh, vinf), np.full(sh, pinf),
                       m.jface_nx[:, n_out], m.jface_ny[:, n_out], g)
        for k in range(4):
            Fj[k][:, n_out] = fo[k] * m.jface_len[:, n_out]

        if self.cfg.av_coeff > 0.0:
            U_L = prim_to_cons(Wi_L[0], Wi_L[1], Wi_L[2], Wi_L[3], g)
            U_R = prim_to_cons(Wi_R[0], Wi_R[1], Wi_R[2], Wi_R[3], g)
            pL = np.maximum(Wi_L[3], 1e-12)
            pR = np.maximum(Wi_R[3], 1e-12)
            sensor = np.clip(np.abs(pL - pR) / (pL + pR)
                             - self.cfg.av_sensor_floor, 0.0, None)
            aL = np.sqrt(g * pL / Wi_L[0])
            aR = np.sqrt(g * pR / Wi_R[0])
            speed = 0.5 * (np.abs(Wi_L[1] * m.iface_nx + Wi_L[2] * m.iface_ny) + aL
                           + np.abs(Wi_R[1] * m.iface_nx + Wi_R[2] * m.iface_ny) + aR)
            nu = self.cfg.av_coeff * sensor * speed
            for k in range(4):
                Fi[k] = Fi[k] + nu * (U_L[k] - U_R[k]) * m.iface_len

            Uj_L = prim_to_cons(Wj_L[0][:, 1:-1], Wj_L[1][:, 1:-1],
                                Wj_L[2][:, 1:-1], Wj_L[3][:, 1:-1], g)
            Uj_R = prim_to_cons(Wj_R[0][:, 1:-1], Wj_R[1][:, 1:-1],
                                Wj_R[2][:, 1:-1], Wj_R[3][:, 1:-1], g)
            pLj = np.maximum(Wj_L[3][:, 1:-1], 1e-12)
            pRj = np.maximum(Wj_R[3][:, 1:-1], 1e-12)
            sensj = np.clip(np.abs(pLj - pRj) / (pLj + pRj)
                            - self.cfg.av_sensor_floor, 0.0, None)
            aLj = np.sqrt(g * pLj / Wj_L[0][:, 1:-1])
            aRj = np.sqrt(g * pRj / Wj_R[0][:, 1:-1])
            spj = 0.5 * (np.abs(Wj_L[1][:, 1:-1] * m.jface_nx[:, 1:-1]
                                + Wj_L[2][:, 1:-1] * m.jface_ny[:, 1:-1]) + aLj
                         + np.abs(Wj_R[1][:, 1:-1] * m.jface_nx[:, 1:-1]
                                  + Wj_R[2][:, 1:-1] * m.jface_ny[:, 1:-1]) + aRj)
            nuj = self.cfg.av_coeff * sensj * spj
            for k in range(4):
                Fj[k][:, 1:-1] += nuj * (Uj_L[k] - Uj_R[k]) * m.jface_len[:, 1:-1]

        res = []
        for k in range(4):
            tot = (np.roll(Fi[k], -1, axis=0) - Fi[k]
                   + Fj[k][:, 1:] - Fj[k][:, :-1])
            res.append(tot / m.area)
        return res

    # -- time integration --------------------------------------------------
    def local_dt(self, cfl: Optional[float] = None) -> np.ndarray:
        m, g = self.mesh, self.gamma
        cfl = self.cfg.cfl if cfl is None else cfl
        rho, u, v, p = self.W
        a = np.sqrt(g * p / rho)
        qn_i = u * m.iface_nx + v * m.iface_ny
        qn_j = u * m.jface_nx[:, :-1] + v * m.jface_ny[:, :-1]
        lam_i = (0.5 * (np.abs(qn_i) + np.abs(np.roll(qn_i, -1, axis=0)))
                 + 0.5 * (a + np.roll(a, -1, axis=0))) * m.iface_len
        lam_j = (np.abs(qn_j) + a) * 0.5 * (m.jface_len[:, :-1] + m.jface_len[:, 1:])
        spec = lam_i + np.roll(lam_i, 1, axis=0) + lam_j
        return cfl * m.area / np.maximum(spec, 1e-30)

    def step(self):
        res = self.residual()
        dt = self.local_dt()
        U = prim_to_cons(*self.W, self.gamma)
        for k in range(4):
            U[k] = U[k] - dt * res[k]
        U[0] = np.maximum(U[0], 1e-10)
        rho, u, v, p = cons_to_prim(U, self.gamma)
        p = np.maximum(p, 1e-10)
        self.W = [rho, u, v, p]
        self.calls += 1
        return res

    def density_residual_norm(self) -> float:
        m = self.mesh
        res = self.residual()
        return float(np.sqrt(np.sum((res[0] * m.area) ** 2) / np.sum(m.area ** 2)))

    def run(self, n_iter: int, tol: float = 1e-7, verbose: bool = False,
            log_every: int = 500):
        for it in range(n_iter):
            self.step()
            if (it + 1) % log_every == 0 or it == 0:
                rn = self.density_residual_norm()
                self.history.append({"iter": it + 1, "residual": rn})
                if verbose:
                    print(f"  iter {it + 1:6d}  |Res_rho|={rn:.3e}", flush=True)
                if rn < tol:
                    break
        return self.history

    # -- diagnostics -------------------------------------------------------
    def wall_pressure(self) -> np.ndarray:
        return self.W[3][:, 0]

    def wall_cp(self):
        rinf, uinf, vinf, pinf = self.cfg.freestream()
        q = 0.5 * rinf * (uinf * uinf + vinf * vinf)
        cp = (self.wall_pressure() - pinf) / q
        return self.mesh.wall_x, cp

    def mach_field(self) -> np.ndarray:
        rho, u, v, p = self.W
        a = np.sqrt(self.gamma * p / rho)
        return np.sqrt(u * u + v * v) / a

    def entropy_field(self) -> np.ndarray:
        rho, u, v, p = self.W
        return np.log(p) - self.gamma * np.log(rho)

    def entropy_production_field(self) -> np.ndarray:
        """Steady entropy transport residual ``rho * u . grad(s)``.

        Positive where irreversible (shock) dissipation occurs, ~0 in smooth
        regions.  Gradients use the curvilinear metric of the O-grid.
        """
        m = self.mesh
        s = self.entropy_field()
        dsdi = 0.5 * (np.roll(s, -1, axis=0) - np.roll(s, 1, axis=0))
        dxdi = 0.5 * (np.roll(m.xc, -1, axis=0) - np.roll(m.xc, 1, axis=0))
        dydi = 0.5 * (np.roll(m.yc, -1, axis=0) - np.roll(m.yc, 1, axis=0))
        dsj = np.empty_like(s)
        dsj[:, :-1] = s[:, 1:] - s[:, :-1]
        dsj[:, -1] = dsj[:, -2]
        dxj = np.empty_like(s)
        dxj[:, :-1] = m.xc[:, 1:] - m.xc[:, :-1]
        dxj[:, -1] = dxj[:, -2]
        dyj = np.empty_like(s)
        dyj[:, :-1] = m.yc[:, 1:] - m.yc[:, :-1]
        dyj[:, -1] = dyj[:, -2]
        det = dxdi * dyj - dxj * dydi
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        dsdx = (dyj * dsdi - dydi * dsj) / det
        dsdy = (-dxj * dsdi + dxdi * dsj) / det
        rho, u, v, p = self.W
        return rho * (u * dsdx + v * dsdy)

    def total_enthalpy_field(self) -> np.ndarray:
        rho, u, v, p = self.W
        return self.gamma / (self.gamma - 1.0) * p / rho + 0.5 * (u * u + v * v)

    def total_enthalpy_deviation(self, skip_rows: int = 0) -> dict:
        H = self.total_enthalpy_field()
        rinf, uinf, vinf, pinf = self.cfg.freestream()
        H0 = self.gamma / (self.gamma - 1.0) * pinf / rinf + 0.5 * (uinf**2 + vinf**2)
        core = H[:, skip_rows:] if skip_rows else H
        dev = np.abs(core - H0) / H0
        return {"H_inf": float(H0), "max_rel_dev": float(dev.max()),
                "mean_rel_dev": float(dev.mean()),
                "p99_rel_dev": float(np.percentile(dev, 99.0))}

    def field_arrays(self) -> Dict[str, np.ndarray]:
        rho, u, v, p = self.W
        rinf, uinf, vinf, pinf = self.cfg.freestream()
        q = 0.5 * rinf * (uinf * uinf + vinf * vinf)
        return {"rho": rho, "u": u, "v": v, "p": p, "mach": self.mach_field(),
                "cp": (p - pinf) / q,
                "entropy_prod": self.entropy_production_field()}


# ---------------------------------------------------------------------------
# 1D steady normal shock (Rankine-Hugoniot anchor)
# ---------------------------------------------------------------------------
class NormalShock1D:
    """1D KFVS solver: supersonic inflow (fixed) + subsonic outflow.

    Converges to a steady normal shock; the uniform states on either side are
    compared with the Rankine-Hugoniot relations (conservation is exact for the
    finite-volume flux balance, so the observed error is the discretisation
    error of the state extraction, not a modelling error).
    """

    def __init__(self, nx: int = 400, length: float = 1.0, mach: float = 2.0,
                 gamma: float = GAMMA, cfl: float = 0.5, order: int = 2,
                 av_coeff: float = 0.0, rusanov: float = 0.0,
                 flux_mode: str = "kfvs_energy_upwind",
                 p_out: Optional[float] = None, x_shock0: float = 0.5):
        self.nx = nx
        self.dx = length / (nx - 1)
        self.gamma = gamma
        self.cfl = cfl
        self.order = order
        self.av_coeff = av_coeff
        self.rusanov = rusanov
        self.flux_mode = flux_mode
        self.p_out = p_out
        self.mach = mach
        self.rho_inf, self.p_inf = 1.0, 1.0
        a = math.sqrt(gamma * self.p_inf / self.rho_inf)
        self.u_inf = mach * a
        self.W = [np.full(nx, self.rho_inf), np.full(nx, self.u_inf),
                  np.full(nx, self.p_inf)]
        # initial guess: jump at mid-length (exact stationary shock is the
        # converged state of this initial-value problem)
        x = np.linspace(0.0, length, nx)
        if p_out is None:
            # start from the exact RH jump: a stationary-shock consistency check
            rh = normal_shock_ratios(mach, gamma)
            jump = x > x_shock0 * length
            self.W[0][jump] = rh["rho_ratio"]
            self.W[2][jump] = rh["p_ratio"]
            self.W[1][jump] = self.u_inf * rh["u_ratio"]

    # -- residual ---------------------------------------------------------
    def _face_states(self):
        rho, u, p = self.W
        nx = self.nx
        outL = [np.empty(nx + 1) for _ in range(3)]
        outR = [np.empty(nx + 1) for _ in range(3)]
        for k, Wv in enumerate(self.W):
            if self.order >= 2:
                d = _minmod(Wv - np.roll(Wv, 1), np.roll(Wv, -1) - Wv)
                d[0] = Wv[1] - Wv[0]
                d[-1] = Wv[-1] - Wv[-2]
            else:
                d = np.zeros_like(Wv)
            left = Wv + 0.5 * d
            right = Wv - 0.5 * d
            outL[k][1:] = left           # face f: left state from cell f-1
            outR[k][:-1] = right         # face f: right state from cell f
            outR[k][-1] = right[-1]      # outflow face: extrapolated
            # supersonic inflow: positive half space from the fixed ghost state
            outL[k][0] = self.rho_inf if k == 0 else (
                self.u_inf if k == 1 else self.p_inf)
            outR[k][0] = Wv[0]
        if self.p_out is not None:
            # subsonic outflow: impose back pressure, extrapolate rho and u
            outL[0][-1] = self.W[0][-1]
            outL[1][-1] = self.W[1][-1]
            outL[2][-1] = self.p_out
            outR[0][-1] = self.W[0][-1]
            outR[1][-1] = self.W[1][-1]
            outR[2][-1] = self.p_out
        return outL, outR

    def residual(self):
        g = self.gamma
        L, R = self._face_states()
        ones = np.ones_like(L[0])
        zeros = np.zeros_like(L[0])
        F = list(numerical_flux_x(L[0], L[1], L[2], R[0], R[1], R[2],
                                  self.flux_mode, g))
        if self.av_coeff > 0.0:
            rL, uL, pL = L[0], L[1], L[2]
            rR, uR, pR = R[0], R[1], R[2]
            sensor = np.clip(np.abs(pL - pR) / (pL + pR) - 0.01, 0.0, None)
            aL = np.sqrt(g * pL / np.maximum(rL, 1e-12))
            aR = np.sqrt(g * pR / np.maximum(rR, 1e-12))
            sp = 0.5 * (np.abs(uL) + aL + np.abs(uR) + aR)
            nu = self.av_coeff * sensor * sp
            UL = prim_to_cons(rL, uL, zeros, pL, g)
            UR = prim_to_cons(rR, uR, zeros, pR, g)
            for k in range(3):
                F[k] = F[k] + nu * (UL[k] - UR[k])
        return [(F[k][1:] - F[k][:-1]) / self.dx for k in range(3)]

    def run(self, n_iter: int = 20000, tol: float = 1e-9, verbose: bool = False,
            log_every: int = 2000):
        hist = []
        for it in range(n_iter):
            res = self.residual()
            rho, u, p = self.W
            a = np.sqrt(self.gamma * p / rho)
            dtg = self.cfl * self.dx / float(np.max(np.abs(u) + a))
            U = prim_to_cons(rho, u, np.zeros_like(rho), p, self.gamma)
            U[0] = U[0] - dtg * res[0]
            U[1] = U[1] - dtg * res[1]
            U[3] = U[3] - dtg * res[2]
            U[0] = np.maximum(U[0], 1e-10)
            rho, u, _, p = cons_to_prim(U, self.gamma)
            self.W = [rho, u, np.maximum(p, 1e-10)]
            if (it + 1) % log_every == 0:
                rn = float(np.sqrt(np.mean(res[0] ** 2)))
                hist.append({"iter": it + 1, "residual": rn})
                if verbose:
                    print(f"  1D iter {it + 1:6d} |Res|={rn:.3e}", flush=True)
                if rn < tol:
                    break
        return hist

    def profiles(self):
        rho, u, p = self.W
        return {"x": np.linspace(0.0, 1.0, self.nx), "rho": rho, "u": u, "p": p,
                "T": p / rho, "s": np.log(p) - self.gamma * np.log(rho),
                "H": self.gamma / (self.gamma - 1.0) * p / rho + 0.5 * u * u}

    def entropy_production(self):
        """``rho * u * ds/dx``: positive entropy generation inside the shock."""
        pr = self.profiles()
        s = pr["s"]
        dsdx = np.gradient(s, self.dx)
        return pr["rho"] * pr["u"] * dsdx

    def plateau_ratios(self, lo: int = 40, hi: int = 40):
        """Ratios between the upstream (first ``lo`` cells) and downstream plateaus."""
        rho, u, p = self.W
        r1 = float(np.mean(rho[:lo]))
        p1 = float(np.mean(p[:lo]))
        r2 = float(np.mean(rho[-hi:]))
        p2 = float(np.mean(p[-hi:]))
        return {"rho_ratio": r2 / r1, "p_ratio": p2 / p1,
                "T_ratio": (p2 / r2) / (p1 / r1),
                "u_ratio": float(np.mean(u[-hi:]) / np.mean(u[:lo])),
                "rho_std_down": float(np.std(rho[-hi:]) / r2),
                "p_std_down": float(np.std(p[-hi:]) / p2)}



class ShockTube1D:
    """1D shock tube (transmissive ends) used to verify shock capturing.

    The shock forms from an initial discontinuity, so its strength and speed are
    determined by the initial data (Riemann problem), not by the boundary
    conditions.  This makes it a genuine test of the flux: the post-shock
    plateau is compared with the Rankine-Hugoniot relations evaluated at the
    *measured* shock Mach number.
    """

    def __init__(self, nx: int = 800, length: float = 1.0,
                 left: Sequence[float] = (1.0, 0.0, 1.0),
                 right: Sequence[float] = (0.125, 0.0, 0.1),
                 gamma: float = GAMMA, cfl: float = 0.4, order: int = 2,
                 flux_mode: str = "kfvs_energy_upwind", av_coeff: float = 0.0):
        self.nx = int(nx)
        self.length = float(length)
        self.dx = self.length / (self.nx - 1)
        self.gamma = gamma
        self.cfl = cfl
        self.order = order
        self.flux_mode = flux_mode
        self.av_coeff = av_coeff
        x = np.linspace(0.0, self.length, self.nx)
        mid = 0.5 * self.length
        self.W = [np.where(x < mid, left[0], right[0]),
                  np.where(x < mid, left[1], right[1]),
                  np.where(x < mid, left[2], right[2])]
        self.time = 0.0

    def _face_states(self):
        outL = [np.empty(self.nx + 1) for _ in range(3)]
        outR = [np.empty(self.nx + 1) for _ in range(3)]
        for k, Wv in enumerate(self.W):
            if self.order >= 2:
                d = _minmod(Wv - np.roll(Wv, 1), np.roll(Wv, -1) - Wv)
                d[0] = Wv[1] - Wv[0]
                d[-1] = Wv[-1] - Wv[-2]
            else:
                d = np.zeros_like(Wv)
            outL[k][1:] = Wv + 0.5 * d            # face f: left = cell f-1
            outR[k][:-1] = Wv - 0.5 * d           # face f: right = cell f
            outR[k][-1] = Wv[-1] - 0.5 * d[-1]
            # transmissive (zero-gradient) ghosts at both ends
            outL[k][0] = Wv[0] + 0.5 * d[0]
            outR[k][0] = Wv[0] - 0.5 * d[0]
            outL[k][-1] = Wv[-1] + 0.5 * d[-1]
        return outL, outR

    def residual(self):
        g = self.gamma
        L, R = self._face_states()
        zeros = np.zeros_like(L[0])
        ones = np.ones_like(L[0])
        F = list(numerical_flux_x(L[0], L[1], L[2], R[0], R[1], R[2],
                                  self.flux_mode, g))
        if self.av_coeff > 0.0:
            sensor = np.clip(np.abs(L[2] - R[2]) / (L[2] + R[2]) - 0.01, 0.0, None)
            aL = np.sqrt(g * L[2] / L[0])
            aR = np.sqrt(g * R[2] / R[0])
            nu = self.av_coeff * sensor * 0.5 * (np.abs(L[1]) + aL + np.abs(R[1]) + aR)
            UL = prim_to_cons(L[0], L[1], zeros, L[2], g)
            UR = prim_to_cons(R[0], R[1], zeros, R[2], g)
            F = [F[0] + nu * (UL[0] - UR[0]), F[1] + nu * (UL[1] - UR[1]),
                 F[2] + nu * (UL[3] - UR[3])]
        return [(F[k][1:] - F[k][:-1]) / self.dx for k in range(3)]

    def run_to(self, t_end: float, max_steps: int = 200000):
        steps = 0
        while self.time < t_end and steps < max_steps:
            rho, u, p = self.W
            a = np.sqrt(self.gamma * p / rho)
            dt = self.cfl * self.dx / float(np.max(np.abs(u) + a))
            dt = min(dt, t_end - self.time)
            res = self.residual()
            U = prim_to_cons(rho, u, np.zeros_like(rho), p, self.gamma)
            U[0] -= dt * res[0]
            U[1] -= dt * res[1]
            U[3] -= dt * res[2]
            U[0] = np.maximum(U[0], 1e-10)
            rho, u, _, p = cons_to_prim(U, self.gamma)
            self.W = [rho, u, np.maximum(p, 1e-10)]
            self.time += dt
            steps += 1
        return steps

    def profiles(self):
        rho, u, p = self.W
        return {"x": np.linspace(0.0, self.length, self.nx),
                "rho": rho.copy(), "u": u.copy(), "p": p.copy(),
                "T": (p / rho).copy(),
                "s": np.log(p) - self.gamma * np.log(rho),
                "H": self.gamma / (self.gamma - 1.0) * p / rho + 0.5 * u * u}

    def shock_position(self) -> float:
        """Location of the steepest pressure gradient (the captured shock)."""
        pr = self.profiles()
        gp = np.abs(np.gradient(pr["p"], self.dx))
        gp[:10] = 0.0
        gp[-10:] = 0.0
        return float(pr["x"][int(np.argmax(gp))])

    def entropy_production(self):
        pr = self.profiles()
        return pr["rho"] * pr["u"] * np.gradient(pr["s"], self.dx)


def moving_shock_ratios(Ms: float, gamma: float = GAMMA) -> Dict[str, float]:
    """Rankine-Hugoniot ratios for a shock moving at Mach ``Ms`` (lab frame).

    ``Ms = (S - u1)/a1``; the post-shock state follows from the stationary
    relations written in the shock frame (Toro, eq. 3.42).
    """
    if Ms <= 1.0:
        raise ValueError("Ms must exceed 1")
    Ms2 = Ms * Ms
    rho_ratio = (gamma + 1.0) * Ms2 / (2.0 + (gamma - 1.0) * Ms2)
    p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (Ms2 - 1.0)
    u_jump = 2.0 / (gamma + 1.0) * (Ms - 1.0 / Ms)     # (u2-u1)/a1
    return {"Ms": Ms, "rho_ratio": rho_ratio, "p_ratio": p_ratio,
            "T_ratio": p_ratio / rho_ratio, "u_jump_over_a1": u_jump}


def normal_shock_ratios(M1: float, gamma: float = GAMMA) -> Dict[str, float]:
    """Rankine-Hugoniot ratios across a stationary normal shock."""
    if M1 <= 1.0:
        raise ValueError("M1 must exceed 1")
    M1s = M1 * M1
    M2s = (1.0 + 0.5 * (gamma - 1.0) * M1s) / (gamma * M1s - 0.5 * (gamma - 1.0))
    rho_ratio = (gamma + 1.0) * M1s / (2.0 + (gamma - 1.0) * M1s)
    p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1s - 1.0)
    T_ratio = p_ratio / rho_ratio
    u_ratio = 1.0 / rho_ratio          # continuity with rho*u = const
    return {"M1": M1, "M2": math.sqrt(M2s), "rho_ratio": rho_ratio,
            "p_ratio": p_ratio, "T_ratio": T_ratio, "u_ratio": u_ratio}


# ---------------------------------------------------------------------------
# Analytic anchors
# ---------------------------------------------------------------------------
def ackeret_cp(theta: np.ndarray, mach: float) -> np.ndarray:
    """Ackeret (linear supersonic) pressure coefficient ``Cp = 2 theta/sqrt(M^2-1)``."""
    if mach <= 1.0:
        raise ValueError("Ackeret theory requires Mach > 1")
    return 2.0 * np.asarray(theta, dtype=float) / math.sqrt(mach * mach - 1.0)


def karman_tsien_cp(cp_incompressible, mach: float):
    """Karman-Tsien compressibility correction of an incompressible ``Cp``."""
    beta = math.sqrt(max(1.0 - mach * mach, 1e-12))
    cp0 = np.asarray(cp_incompressible, dtype=float)
    return cp0 / (beta + mach * mach / (1.0 + beta) * cp0 / 2.0)


def prandtl_glauert_delta_cp(alpha_deg: float, mach: float) -> float:
    """Flat-plate ``dCp = Cp_lower - Cp_upper`` with Prandtl-Glauert correction."""
    beta = math.sqrt(max(1.0 - mach * mach, 1e-12))
    return 4.0 * math.radians(alpha_deg) / beta


def shock_location(x: np.ndarray, cp: np.ndarray, window=(0.05, 0.98)):
    """Locate a shock on a surface distribution: station of steepest Cp rise.

    Returns ``(x_shock, cp_upstream, cp_downstream, dp)`` or ``None``.
    """
    x = np.asarray(x, float)
    cp = np.asarray(cp, float)
    order = np.argsort(x)
    xs, cs = x[order], cp[order]
    mask = (xs >= window[0]) & (xs <= window[1])
    if mask.sum() < 8:
        return None
    grad = np.gradient(cs, xs)
    idx = int(np.argmax(np.where(mask, grad, -np.inf)))
    # upstream (min Cp) / downstream plateau amplitude around the steepest rise
    lo = max(idx - 12, 0)
    hi = min(idx + 12, len(xs) - 1)
    cp_up = float(np.min(cs[lo:idx + 1]))
    cp_dn = float(np.max(cs[idx:hi + 1]))
    return {"x_shock": float(xs[idx]), "cp_upstream": cp_up,
            "cp_downstream": cp_dn, "dp": cp_dn - cp_up,
            "grad_max": float(grad[idx])}


def integrate_forces(x_surf, y_surf, cp, alpha_deg: float, chord: float = 1.0):
    """Integrate surface ``Cp`` into ``C_L``, ``C_D`` (inviscid, closed contour)."""
    x = np.asarray(x_surf, float)
    y = np.asarray(y_surf, float)
    cp = np.asarray(cp, float)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    nx = dy
    ny = -dx
    cn = -np.sum(cp * ny) / chord
    ca = -np.sum(cp * nx) / chord
    a = math.radians(alpha_deg)
    cl = cn * math.cos(a) - ca * math.sin(a)
    cd = cn * math.sin(a) + ca * math.cos(a)
    return float(cl), float(cd)


def experimental_anchor_status(path: Optional[str] = None) -> Dict[str, object]:
    """Status of the AGARD experimental ``Cp`` anchor.

    The repository does not ship digitised AGARD surface-pressure data.  This
    helper makes that explicit and machine checkable: without a provenance-
    carrying JSON dataset the status is ``WITHHELD`` (never silently "passed").
    """
    import json
    import os
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        prov = data.get("provenance", {})
        required = ("source", "digitised_by", "date")
        missing = [k for k in required if not prov.get(k)]
        if missing:
            return {"status": "WITHHELD",
                    "reason": f"missing provenance fields: {missing}", "path": path}
        return {"status": "AVAILABLE", "path": path,
                "cases": list(data.get("cases", {}).keys())}
    return {"status": "WITHHELD",
            "reason": "no digitised AGARD Cp dataset supplied (provenance required)",
            "path": path}


def active_viscosity_sensitivity(mesh: AnnulusMesh, cfg: SolverConfig,
                                 n_iter: int, av_values: Sequence[float]):
    """Run the same case with/without the shock-sensor artificial viscosity.

    Returns per-run oscillation amplitude of the wall ``Cp`` (total variation) so
    that the effect of the localised dissipation is a measured number.
    """
    out = []
    for av in av_values:
        c = SolverConfig(**{**cfg.__dict__, "av_coeff": float(av)})
        sol = KineticFluxSolver(mesh, c)
        sol.run(n_iter, log_every=max(n_iter, 1))
        _, cp = sol.wall_cp()
        out.append({"av_coeff": float(av),
                    "cp_total_variation": float(np.sum(np.abs(np.diff(cp)))),
                    "cl_cd": integrate_forces(mesh.wall_x, mesh.wall_y, cp,
                                              c.alpha_deg)})
    return out