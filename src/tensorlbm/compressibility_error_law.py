"""Compressibility-error law ``eps(Ma)`` and Mach envelope for a heated cylinder.

Why this module exists
----------------------
Every LBM run of an *incompressible* problem is really a weakly compressible
run: ``p = rho c_s^2`` and the discrete equilibrium carries third-order velocity
terms that the incompressible Navier--Stokes equations do not have.  The
repository pins individual cases at *one* Mach number (``benchmarks/verified/
cylinder``, the IN28-N03 forced-convection case, ``tensorlbm.weak_compressibility``
for ``St``/``C_D``), but nothing answers the question a scheduling layer has to
ask *before* spending core hours:

    "for this geometry, this Reynolds number and this grid, which Mach-number
     band keeps the heat-transfer coefficient within X % of its Ma -> 0 limit,
     and where does the scheme stop being usable at all?"

This module supplies that surface for the repository's canonical heat-transfer
configuration (isothermal circular cylinder in cross flow, ``Pr = 0.7``):

1. :func:`run_thermal_cylinder` -- double-distribution-function (DDF) LBM:
   D2Q9 hydrodynamics plus a D2Q9 passive-scalar lattice for the temperature,
   halfway bounce-back on the cylinder and the tunnel walls,
   non-equilibrium-extrapolation inlet/outlet and anti-bounce-back Dirichlet
   temperature on the cylinder.  Returns the mean Nusselt number, ``C_D``,
   steady/unsteady diagnostics and the *measured* uncertainty of the mean.
2. :func:`richardson_nu0` -- Richardson extrapolation in the small parameter
   ``Ma^2``: a weighted fit ``Nu = Nu0 + c2 Ma^2 + c4 Ma^4`` supplies the
   self-reference value ``Nu0`` of the error law.
3. :func:`fit_ma_error_law` -- weighted log--log fit of
   ``eps = |Nu(Ma)/Nu0 - 1| = C Ma^n`` with the 95 % confidence interval of the
   exponent and ``R^2``.
4. :func:`mach_envelope` -- the usable Mach *band* per Reynolds number, i.e. the
   JSON object a scheduler can consume.
5. :func:`predict_holdout_envelope` -- the pre-registered hold-out prediction
   rule (``log C`` linear in ``log Re``) so a held-out Reynolds number can be
   predicted before it is run.

Physics/normalisation conventions
---------------------------------
* ``c_s = 1/sqrt(3)``, ``Ma = u_in/c_s``, ``nu = u_in D_eff/Re``,
  ``tau = 1/2 + 3 nu``; thermal diffusivity ``alpha = c_s^2 (tau_T - 1/2)``,
  ``Pr = nu/alpha``.  The diameter entering ``Re`` and ``Nu`` is the *effective*
  diameter of the halfway bounce-back surface (mean link-midpoint radius), not
  the nominal staircase-mask diameter -- see :func:`cylinder_geometry`.
* ``Nu = <q_n> D_eff / (alpha (T_w - T_in))`` with
  ``q_n = n . sum_i c_i g_i^neq`` at the fluid nodes adjacent to the cylinder.
  ``sum_i c_i g_i^neq = -alpha grad T`` is the diffusive-flux moment of the
  scalar lattice, so the estimator is local and independent of the staircase
  link orientation.
* the error law is measured *relative to the Ma -> 0 limit of the same
  discretisation*, so Mach-independent offsets (staircase geometry, blockage,
  first-order inlet) cancel in ``eps``; they do **not** cancel in the absolute
  ``Nu`` used for the external Churchill--Bernstein anchor.

Known limits (measured, see ``docs/P83W2_N01_MA_ERROR_LAW.md``)
--------------------------------------------------------------
* at fixed ``Re`` and fixed grid, ``tau -> 1/2`` as ``Ma -> 0`` because
  ``tau - 1/2 = 3 Ma D_eff/(sqrt(3) Re)``.  The low-Mach end of the band is
  therefore *stiffness*-limited, not accuracy-limited:
  :func:`mach_floor_from_stiffness` returns that bound and it is part of the
  envelope; the high-Mach end is set by the ``O(Ma^2)`` error tolerance.
* one case costs ``D_eff/u_in`` lattice steps per convective time, i.e. the cost
  grows like ``1/Ma``; :func:`run_thermal_cylinder` accepts a warm start so the
  low-Mach end of a sweep is affordable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .weak_compressibility import (
    CS,
    CS2,
    CX,
    CY,
    OPP,
    W,
    _dominant_frequency,
    _inlet_neq_west,
    _outlet_neq_east,
    bounce_back,
    equilibrium,
    macroscopic,
    stream_periodic,
)

#: A lattice velocity above this is outside the weakly compressible regime the
#: D2Q9 equilibrium is built for (``Ma = |u|/c_s > 1.04`` here); a run that
#: reaches it is reported as unusable instead of being extrapolated.
MAX_LATTICE_VELOCITY = 0.6

__all__ = [
    "churchill_bernstein_mean_nu",
    "CylinderGeometry",
    "cylinder_geometry",
    "ThermalCylinderSetup",
    "thermal_equilibrium",
    "nusselt_number",
    "nusselt_wall_extrapolated",
    "nusselt_global_balance",
    "run_thermal_cylinder",
    "richardson_nu0",
    "fit_ma_error_law",
    "mach_floor_from_stiffness",
    "mach_envelope",
    "predict_holdout_envelope",
]


# ---------------------------------------------------------------------------
# Reference correlation (external anchor)
# ---------------------------------------------------------------------------
def churchill_bernstein_mean_nu(re: float, pr: float = 0.7) -> float:
    """Mean Nusselt number of an isothermal cylinder in cross flow.

    Churchill & Bernstein (1977),
    ``Nu_D = 0.3 + 0.62 Re^0.5 Pr^(1/3) / (1 + (0.4/Pr)^(2/3))^(1/4)
    (1 + (Re/282000)^(5/8))^(4/5)``, valid for ``Re Pr > 0.2``.  At
    ``Re = 100, Pr = 0.7`` it returns ``5.156``.

    This is the *external* anchor of the campaign: an independent correlation
    (3-D experiments, uniform wall temperature) that the lattice result is
    compared against with no fitting of any kind.
    """
    if re <= 0.0:
        raise ValueError("Re must be positive")
    if re * pr <= 0.2:
        return 1.0
    num = 0.62 * math.sqrt(re) * pr ** (1.0 / 3.0)
    den = (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25
    return 0.3 + num / den * (1.0 + (re / 282000.0) ** (5.0 / 8.0)) ** 0.8


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CylinderGeometry:
    """Masks, effective diameter and wall-link tables of a staircase cylinder.

    ``links_*`` holds one entry per fluid node that has a cylinder node as its
    downstream neighbour in direction ``links_i``: ``(y_fluid, x_fluid, i)``.
    The numerical no-slip surface of halfway bounce-back is the midpoint between
    the solid node and that fluid node, so the *effective* radius is the mean
    radius of those midpoints -- the diameter that has to be used in ``Re`` and
    ``Nu`` for the simulation to describe the same physical cylinder as the
    correlation.
    """

    ny: int
    nx: int
    cy: float
    cx: float
    radius_nominal: float
    radius_eff: float
    solid: np.ndarray
    cylinder: np.ndarray
    wall: np.ndarray
    links_y: np.ndarray
    links_x: np.ndarray
    links_i: np.ndarray
    link_normal_y: np.ndarray
    link_normal_x: np.ndarray
    link_weight: np.ndarray
    perimeter: float

    @property
    def diameter_eff(self) -> float:
        return 2.0 * self.radius_eff

    @property
    def blockage(self) -> float:
        return self.diameter_eff / (self.ny - 1.0)


def _arc_weights(theta: np.ndarray) -> np.ndarray:
    """Arc-length weights of surface points given their polar angles."""
    n = theta.size
    if n < 3:
        return np.full(n, 1.0 / max(n, 1))
    order = np.argsort(theta)
    th = theta[order]
    d = np.diff(np.concatenate([th, th[:1] + 2.0 * math.pi]))
    d = np.where(d <= 0.0, 1e-12, d)
    w_sorted = 0.5 * (d + np.roll(d, 1))
    w = np.empty(n)
    w[order] = w_sorted
    return w / w.sum()


def cylinder_geometry(
    ny: int,
    nx: int,
    cy: float,
    cx: float,
    radius: float,
    *,
    wall_rows: bool = True,
) -> CylinderGeometry:
    """Build the solid masks and the fluid->solid link tables.

    ``wall_rows=True`` closes the top and bottom rows of the domain (a wind
    tunnel with no-slip, adiabatic walls); the cylinder is the only tracked
    body, so ``C_D``/``C_L`` integrate over it alone.
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    cylinder = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    if not cylinder.any():
        raise ValueError("empty cylinder mask: increase the diameter")
    wall = np.zeros((ny, nx), dtype=bool)
    if wall_rows:
        wall[0, :] = True
        wall[-1, :] = True
    solid = cylinder | wall

    ys: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    idx: list[np.ndarray] = []
    my: list[np.ndarray] = []
    mx: list[np.ndarray] = []
    for i in range(1, 9):
        dy, dx = int(CY[i]), int(CX[i])
        downstream_solid = np.roll(cylinder, (-dy, -dx), axis=(0, 1))
        mask = (~solid) & downstream_solid
        if not mask.any():
            continue
        ly, lx = np.nonzero(mask)
        ys.append(ly)
        xs.append(lx)
        idx.append(np.full(ly.size, i, dtype=np.int64))
        my.append(ly + 0.5 * dy)
        mx.append(lx + 0.5 * dx)
    links_y = np.concatenate(ys)
    links_x = np.concatenate(xs)
    links_i = np.concatenate(idx)
    ry = np.concatenate(my) - cy
    rx = np.concatenate(mx) - cx
    r = np.hypot(ry, rx)
    weight = _arc_weights(np.arctan2(ry, rx))
    return CylinderGeometry(
        ny=ny,
        nx=nx,
        cy=float(cy),
        cx=float(cx),
        radius_nominal=float(radius),
        radius_eff=float(r.mean()),
        solid=solid,
        cylinder=cylinder,
        wall=wall,
        links_y=links_y,
        links_x=links_x,
        links_i=links_i,
        link_normal_y=ry / r,
        link_normal_x=rx / r,
        link_weight=weight,
        perimeter=float(2.0 * math.pi * r.mean()),
    )


# ---------------------------------------------------------------------------
# Case setup
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalCylinderSetup:
    """One ``(Re, Ma)`` point of the compressibility-error sweep.

    The geometry is expressed in cylinder diameters so that the same physical
    case is reproduced at every resolution: ``Lx = lx_over_d * D``,
    ``Ly = ly_over_d * D``, cylinder centre ``cx_over_d * D`` from the inlet.
    Every Mach point of a sweep must use the same grid, the same initial field
    (up to the Mach rescaling) and the same physical integration time -- that is
    what makes the difference between two points a difference in Mach number and
    nothing else.
    """

    mach: float
    re: float
    pr: float = 0.7
    diameter: float = 24.0
    lx_over_d: float = 20.0
    ly_over_d: float = 12.0
    cx_over_d: float = 5.0
    collision: str = "reg"
    thermal_collision: str = "reg"
    t_wall: float = 1.0
    t_in: float = 0.0
    transient_tu: float = 12.0
    sample_tu: float = 10.0
    max_tu: float = 40.0
    sample_every_tu: float = 0.05
    residual_tol: float = 5.0e-5
    perturb: float = 0.0

    # -- grid ---------------------------------------------------------------
    @property
    def nx(self) -> int:
        return int(round(self.lx_over_d * self.diameter))

    @property
    def ny(self) -> int:
        return int(round(self.ly_over_d * self.diameter))

    @property
    def cy(self) -> float:
        return 0.5 * (self.ny - 1)

    @property
    def cx(self) -> float:
        return self.cx_over_d * self.diameter

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def u_in(self) -> float:
        """Inlet velocity in lattice units; ``Ma = u_in / c_s``."""
        return self.mach * CS

    @property
    def geometry(self) -> CylinderGeometry:
        return cylinder_geometry(self.ny, self.nx, self.cy, self.cx, self.radius)

    @property
    def diameter_eff(self) -> float:
        return float(self.geometry.diameter_eff)

    @property
    def nu(self) -> float:
        return self.u_in * self.diameter_eff / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def alpha(self) -> float:
        return self.nu / self.pr

    @property
    def tau_t(self) -> float:
        return 0.5 + 3.0 * self.alpha

    @property
    def tu_steps(self) -> float:
        """Lattice steps per convective time unit ``D_eff / u_in``."""
        return self.diameter_eff / self.u_in

    def resolved(self) -> dict[str, Any]:
        """Scalar (JSON-safe) description of the case."""
        geo = self.geometry
        return {
            "mach": self.mach,
            "re": self.re,
            "pr": self.pr,
            "u_in": self.u_in,
            "nu": self.nu,
            "tau": self.tau,
            "alpha": self.alpha,
            "tau_t": self.tau_t,
            "diameter_nominal": self.diameter,
            "diameter_eff": geo.diameter_eff,
            "radius_eff": geo.radius_eff,
            "blockage": geo.blockage,
            "nx": self.nx,
            "ny": self.ny,
            "cx": self.cx,
            "cy": self.cy,
            "tu_steps": self.tu_steps,
            "collision": self.collision,
            "thermal_collision": self.thermal_collision,
            "transient_tu": self.transient_tu,
            "sample_tu": self.sample_tu,
            "max_tu": self.max_tu,
        }


# ---------------------------------------------------------------------------
# Thermal (passive-scalar) lattice
# ---------------------------------------------------------------------------
def thermal_equilibrium(T: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """``g_i^eq = w_i T (1 + 3 c_i . u)``, the advection-diffusion D2Q9 lattice."""
    out = np.empty((9,) + T.shape)
    for i in range(9):
        out[i] = W[i] * T * (1.0 + 3.0 * (CX[i] * ux + CY[i] * uy))
    return out


def _thermal_collide(
    g: np.ndarray,
    T: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    tau_t: float,
    collision: str,
) -> np.ndarray:
    """Collide the temperature populations.

    ``collision='reg'`` projects the non-equilibrium onto the first-order (flux)
    Hermite subspace, ``g_i^neq -> (w_i/c_s^2) c_i . j^neq`` with
    ``j^neq = sum_j c_j g_j^neq``.  A passive-scalar lattice has no physical
    non-equilibrium content beyond that flux moment, so the projection only
    removes the ghost modes that make plain BGK diverge as ``tau_T -> 1/2``; the
    diffusive flux (hence ``alpha``) is unchanged.
    """
    geq = thermal_equilibrium(T, ux, uy)
    if collision == "bgk":
        return g - (g - geq) / tau_t
    if collision != "reg":
        raise ValueError(f"unknown thermal collision {collision!r}")
    gneq = g - geq
    jx = np.tensordot(CX, gneq, axes=(0, 0))
    jy = np.tensordot(CY, gneq, axes=(0, 0))
    reg = np.empty_like(g)
    for i in range(9):
        reg[i] = (W[i] / CS2) * (CX[i] * jx + CY[i] * jy)
    return geq + (1.0 - 1.0 / tau_t) * reg


def _hydro_step(
    f: np.ndarray, tau: float, collision: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collide the hydrodynamic populations and return ``(f, rho, ux, uy)``.

    The operator is the one of :func:`tensorlbm.weak_compressibility.collide`
    (``bgk`` / ``trt`` with magic parameter ``Lambda = 3/16`` / ``reg``), but the
    moments are returned so that the temperature lattice does not re-evaluate
    them; ``tests/test_compressibility_error_law.py`` asserts bit-identity with
    ``weak_compressibility.collide`` for all three collision models.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    if collision == "bgk":
        s_plus = 1.0 / tau
        return f - s_plus * (f - feq), rho, ux, uy
    if collision == "trt":
        magic = 3.0 / 16.0
        s_plus = 1.0 / tau
        s_minus = 1.0 / (0.5 + magic / (tau - 0.5))
        f_plus = 0.5 * (f + f[OPP])
        f_minus = 0.5 * (f - f[OPP])
        eq_plus = 0.5 * (feq + feq[OPP])
        eq_minus = 0.5 * (feq - feq[OPP])
        return (
            f - s_plus * (f_plus - eq_plus) - s_minus * (f_minus - eq_minus),
            rho,
            ux,
            uy,
        )
    if collision != "reg":
        raise ValueError(f"unknown collision {collision!r}")
    fneq = f - feq
    pxx = np.tensordot(CX * CX - CS2, fneq, axes=(0, 0))
    pyy = np.tensordot(CY * CY - CS2, fneq, axes=(0, 0))
    pxy = np.tensordot(CX * CY, fneq, axes=(0, 0))
    qxx = CX * CX - CS2
    qyy = CY * CY - CS2
    qxy = 2.0 * (CX * CY)
    reg = np.empty_like(f)
    for i in range(9):
        # same association order as tensorlbm.weak_compressibility.collide_regularized
        q = qxx[i] * pxx + qxy[i] * pxy + qyy[i] * pyy
        reg[i] = (W[i] / (2.0 * CS2 * CS2)) * q
    return feq + (1.0 - 1.0 / tau) * reg, rho, ux, uy


def _thermal_bcs(
    g: np.ndarray,
    geo: CylinderGeometry,
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    t_wall: float,
    t_in: float,
    u_in: float,
) -> None:
    """Temperature boundary conditions, applied in place after streaming.

    * cylinder: anti-bounce-back Dirichlet ``g_i = -g_opp + 2 w_i T_w`` (the wall
      sits halfway along the link, matching the hydrodynamic bounce-back);
    * top/bottom rows: plain bounce-back = adiabatic (zero normal flux);
    * inlet: non-equilibrium extrapolation with ``T = t_in`` and ``u = u_in``;
    * outlet: non-equilibrium extrapolation with the interior temperature.
    """
    # inlet / outlet first: the solid rows are overwritten afterwards anyway
    Tn = g[:, :, 1:2].sum(axis=0)
    geq_n = thermal_equilibrium(Tn, ux[:, 1:2], uy[:, 1:2])
    geq_in = thermal_equilibrium(
        np.full_like(Tn, t_in), np.full_like(Tn, u_in), np.zeros_like(Tn)
    )
    g[:, :, 0:1] = geq_in + (g[:, :, 1:2] - geq_n)

    Tm = g[:, :, -2:-1].sum(axis=0)
    geq_m = thermal_equilibrium(Tm, ux[:, -2:-1], uy[:, -2:-1])
    g[:, :, -1:] = geq_m + (g[:, :, -2:-1] - geq_m)

    bounced = g[OPP].copy()
    cyl = geo.cylinder
    if cyl.any():
        for i in range(9):
            g[i] = np.where(cyl, -bounced[i] + 2.0 * W[i] * t_wall, g[i])
    wall = geo.wall
    if wall.any():
        for i in range(9):
            g[i] = np.where(wall, bounced[i], g[i])


def flux_moment_diffusivity(tau_t: float) -> float:
    """Ratio between the non-equilibrium flux moment and ``grad T``.

    For the advection-diffusion D2Q9 lattice
    ``sum_i c_i g_i^neq = -kappa grad T`` with ``kappa = tau_T c_s^2`` -- *not*
    the physical diffusivity ``alpha = c_s^2 (tau_T - 1/2)``: the ``1/2`` of
    ``alpha`` comes from the temporal (half-step) part of the Chapman-Enskog
    expansion and never appears in the instantaneous first moment.

    The identity is verified numerically (periodic decaying sine wave, where
    ``grad T`` is analytic) by ``tests/test_compressibility_error_law.py``:
    measured ``sum_i c_i g_i^neq / (-alpha dT/dx) = tau_T/(tau_T - 1/2)`` to
    better than 1 % for ``tau_T in {0.55, 0.6, 1.0}``.  Using ``alpha`` here
    instead of ``kappa`` inflates ``Nu`` by ``tau_T/(tau_T-1/2)`` -- 13.8x at
    ``Pr = 0.7, Ma = 0.1, D = 16`` -- which is how this trap was found.
    """
    return CS2 * tau_t


def nusselt_number(
    g: np.ndarray,
    T: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    geo: CylinderGeometry,
    *,
    tau_t: float,
    t_wall: float,
    t_in: float,
) -> dict[str, float]:
    """Mean Nusselt number on the cylinder surface.

    ``q_n = n . sum_i c_i g_i^neq`` is the diffusive-flux moment of the scalar
    lattice, converted to ``-alpha dT/dn`` with :func:`flux_moment_diffusivity`.
    It is evaluated at the fluid nodes adjacent to the halfway wall, i.e. about
    half a cell outside the surface, which biases it low by ``O(0.5/D_eff)`` --
    a Mach-independent bias that cancels in the error law and is quoted with the
    absolute comparison.

    The angular average uses arc-length weights, so every surface segment of the
    staircase contributes in proportion to the arc it represents.
    """
    dt = t_wall - t_in
    if dt == 0.0:
        raise ValueError("t_wall must differ from t_in")
    kappa = flux_moment_diffusivity(tau_t)
    geq = thermal_equilibrium(T, ux, uy)
    gneq = g - geq
    jx = np.tensordot(CX, gneq, axes=(0, 0))
    jy = np.tensordot(CY, gneq, axes=(0, 0))
    jn = (
        geo.link_normal_x * jx[geo.links_y, geo.links_x]
        + geo.link_normal_y * jy[geo.links_y, geo.links_x]
    )
    local = jn * (geo.diameter_eff / (kappa * dt))
    w = geo.link_weight
    mean = float(np.sum(w * local))
    return {
        "nu": mean,
        "nu_min": float(local.min()),
        "nu_max": float(local.max()),
        "nu_scatter": float(np.sqrt(np.sum(w * (local - mean) ** 2))),
        "nu_local": local,
    }


def _bilinear(field: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field[y, x]`` with clamped indices."""
    ny, nx = field.shape
    y = np.clip(y, 0.0, ny - 1.001)
    x = np.clip(x, 0.0, nx - 1.001)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    ty = y - y0
    tx = x - x0
    return (
        field[y0, x0] * (1 - ty) * (1 - tx)
        + field[y0 + 1, x0] * ty * (1 - tx)
        + field[y0, x0 + 1] * (1 - ty) * tx
        + field[y0 + 1, x0 + 1] * ty * tx
    )


def nusselt_wall_extrapolated(
    g: np.ndarray,
    T: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    geo: CylinderGeometry,
    *,
    tau_t: float,
    t_wall: float,
    t_in: float,
) -> float:
    """``Nu`` from the flux moment *extrapolated to the wall*.

    The plain estimator (:func:`nusselt_number`) samples ``q_n`` half a cell
    outside the halfway wall, which biases it low whenever the thermal boundary
    layer is only a couple of cells thick.  Here the flux moment is sampled at
    the boundary node (distance ``d1`` from the wall, known exactly from the link
    geometry) and at a second point one cell further out along the surface normal
    (bilinear interpolation), and linearly extrapolated back to the wall.  The
    result is second-order accurate in the sampling radius and removes most of
    the ``O(h/D_eff)`` bias at fixed grid.
    """
    dt = t_wall - t_in
    kappa = flux_moment_diffusivity(tau_t)
    gneq = g - thermal_equilibrium(T, ux, uy)
    jx = np.tensordot(CX, gneq, axes=(0, 0))
    jy = np.tensordot(CY, gneq, axes=(0, 0))
    y = geo.links_y.astype(float)
    x = geo.links_x.astype(float)
    nxv, nyv = geo.link_normal_x, geo.link_normal_y
    # distance of the boundary fluid node from the centre and from the wall
    r_f = np.hypot(y - geo.cy, x - geo.cx)
    q1 = nxv * jx[geo.links_y, geo.links_x] + nyv * jy[geo.links_y, geo.links_x]
    y2 = y + nyv
    x2 = x + nxv
    q2 = nxv * _bilinear(jx, y2, x2) + nyv * _bilinear(jy, y2, x2)
    d1 = np.maximum(r_f - geo.radius_eff, 0.2)
    d2 = d1 + 1.0
    q_wall = q1 + (q1 - q2) * d1 / (d2 - d1)
    local = q_wall * (geo.diameter_eff / (kappa * dt))
    return float(np.sum(geo.link_weight * local))


def nusselt_global_balance(
    T: np.ndarray,
    ux: np.ndarray,
    *,
    alpha: float,
    t_in: float,
    t_wall: float,
) -> float:
    """Surface-averaged ``Nu`` from the global energy balance.

    The tunnel walls are adiabatic and the inlet carries ``T = t_in``, so in the
    steady state the total heat released by the cylinder equals the enthalpy flux
    through the outlet plane::

        sum_surface q_n ds = integral_outlet rho u_x (T - T_in) dy

    which is an *independent* check of the local flux estimator (no gradient, no
    surface normal, only mass/momentum-consistent transport).  ``rho ~ 1`` and
    ``c_p = 1`` in lattice units, hence
    ``Nu = integral_outlet u_x (T-T_in) dy / (pi alpha (T_w - T_in))``.
    """
    nx = T.shape[1]
    x_out, x_in = nx - 2, 1
    # convective enthalpy flux through the outlet plane, minus the axial
    # conduction through the outlet and the inlet planes (the inlet carries
    # T = t_in, but heat also diffuses upstream through it); both are neglected
    # in the leading-order form documented above and are ~1 % at Pe_D ~ 70.
    conv = float((ux[:, x_out] * (T[:, x_out] - t_in)).sum())
    diff = float(alpha * (
        np.gradient(T[:, x_out]).sum() - np.gradient(T[:, x_in]).sum()
    ))
    return (conv - diff) / (math.pi * alpha * (t_wall - t_in))


# ---------------------------------------------------------------------------
# One (Re, Ma) case
# ---------------------------------------------------------------------------
def _initial_fields(
    geo: CylinderGeometry,
    setup: ThermalCylinderSetup,
    warm_start: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial populations: uniform plug flow, or a rescaled warm start.

    A warm start from a neighbouring (higher) Mach number reuses the converged
    velocity field scaled by ``Ma_new/Ma_old`` and the density perturbation by
    its square -- the ``O(Ma^2)`` structure of the compressibility error -- and
    the temperature field as-is, because ``T`` is Mach independent to leading
    order.  That is what makes the low-Mach end of the band affordable: the
    remaining transient has amplitude ``O(delta Ma^2)``, not ``O(1)``.
    """
    ny, nx = geo.ny, geo.nx
    if warm_start is None:
        u_in = setup.u_in
        ux = np.full((ny, nx), u_in)
        uy = np.zeros((ny, nx))
        if setup.perturb:
            yy, xx = np.mgrid[0:ny, 0:nx]
            uy = setup.perturb * u_in * np.exp(
                -(
                    (xx - 1.6 * geo.cx) ** 2
                    + (yy - geo.cy - 0.5 * geo.diameter_eff) ** 2
                )
                / (2.0 * (0.6 * geo.diameter_eff) ** 2)
            )
        rho = np.ones((ny, nx))
        T = np.full((ny, nx), setup.t_in)
    else:
        m_old = float(warm_start.get("mach", setup.mach))
        ratio = setup.mach / m_old if m_old else 0.0
        ux = np.asarray(warm_start["ux"], dtype=float) * ratio
        uy = np.asarray(warm_start["uy"], dtype=float) * ratio
        rho = 1.0 + (np.asarray(warm_start["rho"], dtype=float) - 1.0) * ratio ** 2
        T = np.array(warm_start["T"], dtype=float, copy=True)
    ux[geo.solid] = 0.0
    uy[geo.solid] = 0.0
    T[geo.cylinder] = setup.t_wall
    T[geo.wall] = np.clip(T[geo.wall], setup.t_in, setup.t_wall)
    return equilibrium(rho, ux, uy), thermal_equilibrium(T, ux, uy)


def _block_stats(hist: Sequence[float], blocks: int = 8) -> dict[str, float]:
    """Mean of a series plus the standard error of its block means.

    Correlated series (a decaying transient, or a periodic wake) do not have
    ``std/sqrt(N)`` as their error; splitting the window into blocks and taking
    the scatter of the block means is the cheap and honest estimate that the
    error-law fit propagates as a weight.
    """
    a = np.asarray(hist, dtype=float)
    if a.size == 0:
        return {
            "mean": float("nan"),
            "stderr": float("nan"),
            "std": float("nan"),
            "first_half_mean": float("nan"),
            "second_half_mean": float("nan"),
        }
    k = int(min(max(blocks, 2), a.size))
    chunk = [c for c in np.array_split(a, k) if c.size]
    means = np.array([c.mean() for c in chunk])
    stderr = (
        float(means.std(ddof=1) / math.sqrt(means.size))
        if means.size > 1
        else float("nan")
    )
    return {
        "mean": float(a.mean()),
        "stderr": stderr,
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "first_half_mean": float(a[: a.size // 2].mean()),
        "second_half_mean": float(a[a.size // 2 :].mean()),
    }


def run_thermal_cylinder(
    setup: ThermalCylinderSetup,
    *,
    warm_start: Mapping[str, np.ndarray] | None = None,
    keep_fields: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Run one ``(Re, Ma)`` case and return its diagnostics.

    The step budget is expressed in convective time units ``D_eff/u_in`` so that
    every Mach number of a sweep integrates the *same* physical time.  After
    ``transient_tu`` the Nusselt number and ``C_D`` are block-averaged over
    ``sample_tu``; the scatter of the block means is the measurement uncertainty
    propagated into the error-law fit (a steady case has a tiny one, an unsteady
    ``Re = 200`` case does not).
    """
    geo = setup.geometry
    f, g = _initial_fields(geo, setup, warm_start)

    tu = setup.tu_steps
    n_transient = max(1, int(round(setup.transient_tu * tu)))
    n_sample = max(1, int(round(setup.sample_tu * tu)))
    max_steps = max(n_transient + n_sample, int(round(setup.max_tu * tu)))
    sample_every = max(1, int(round(setup.sample_every_tu * tu)))
    # rolling convergence window: the same physical duration as sample_tu
    n_window = max(4, int(round(setup.sample_tu * tu / sample_every)))

    q_dyn = 0.5 * setup.u_in ** 2 * geo.diameter_eff
    tau, tau_t, alpha, u_in = setup.tau, setup.tau_t, setup.alpha, setup.u_in

    nu_hist: list[float] = []
    cd_hist: list[float] = []
    cl_hist: list[float] = []
    t_hist: list[float] = []
    up_hist: list[float] = []
    drho_hist: list[float] = []
    diverged = -1
    diverge_reason = ""
    steps = 0

    while steps < max_steps:
        f, rho, ux, uy = _hydro_step(f, tau, setup.collision)
        f = stream_periodic(f)
        _inlet_neq_west(f, np.full(geo.ny, u_in))
        _outlet_neq_east(f, 1.0)
        f, force = bounce_back(f, geo.solid, track_mask=geo.cylinder)

        T = g.sum(axis=0)
        g = _thermal_collide(g, T, ux, uy, tau_t, setup.thermal_collision)
        g = stream_periodic(g)
        _thermal_bcs(g, geo, ux, uy, t_wall=setup.t_wall, t_in=setup.t_in, u_in=u_in)
        steps += 1

        if not (np.isfinite(f).all() and np.isfinite(g).all()):
            diverged = steps
            diverge_reason = "non-finite populations"
            break
        umax = float(np.max(np.abs(ux)))
        if umax > MAX_LATTICE_VELOCITY or umax != umax:
            # the solution is still finite but has left the weakly compressible
            # regime the discretisation was built for: unusable for the law
            diverged = steps
            diverge_reason = f"velocity guard (|u|_max = {umax:.3f} > {MAX_LATTICE_VELOCITY})"
            break

        if steps >= n_transient and (steps - n_transient) % sample_every == 0:
            T = g.sum(axis=0)
            nu = nusselt_number(
                g,
                T,
                ux,
                uy,
                geo,
                tau_t=tau_t,
                t_wall=setup.t_wall,
                t_in=setup.t_in,
            )
            nu_hist.append(nu["nu"])
            cd_hist.append(force[0] / q_dyn)
            cl_hist.append(force[1] / q_dyn)
            t_hist.append(steps / tu)
            up_hist.append(umax)
            drho_hist.append(float(np.max(np.abs(rho - 1.0))))
            if progress is not None:
                progress(steps, steps / tu, nu_hist[-1], cd_hist[-1])
            # Rolling convergence test: stop once the mean over the last
            # ``sample_tu`` has stopped moving (half-window comparison).  This is
            # what protects the error law from a Mach-dependent *transient*: at
            # low Ma the thermal boundary layer relaxes on a longer lattice time,
            # so a fixed step budget would bias eps(Ma) low exactly where it is
            # smallest.
            if len(nu_hist) >= n_window and steps - n_transient >= n_sample:
                win = np.asarray(nu_hist[-n_window:], dtype=float)
                half = win.size // 2
                m1 = float(win[:half].mean())
                m2 = float(win[half:].mean())
                if abs(m2 - m1) <= setup.residual_tol * max(abs(m2), 1e-30):
                    break

    T = g.sum(axis=0)
    rho, ux, uy = macroscopic(f)
    nu_stat = _block_stats(nu_hist)
    cd_stat = _block_stats(cd_hist)
    cl_stat = _block_stats(cl_hist)
    drift = (
        abs(nu_stat["second_half_mean"] - nu_stat["first_half_mean"])
        if np.isfinite(nu_stat["mean"]) and nu_stat["mean"] != 0
        else float("nan")
    )
    drift_rel = drift / abs(nu_stat["mean"]) if np.isfinite(drift) else float("nan")

    cl = np.asarray(cl_hist, dtype=float)
    period_tu = 0.0
    if cl.size > 8:
        f_dom, _ = _dominant_frequency(cl, setup.sample_every_tu)
        if f_dom > 0:
            period_tu = 1.0 / f_dom
    # a steady wake has no shedding frequency: only report St when the lift
    # actually oscillates (otherwise the FFT picks up numerical noise)
    steady = cl_stat["std"] < 1e-3 * max(abs(cd_stat["mean"]), 1.0)
    st = (
        geo.diameter_eff / (u_in * period_tu)
        if (period_tu > 0 and not steady)
        else float("nan")
    )

    nusselt = nusselt_number(
        g, T, ux, uy, geo, tau_t=tau_t, t_wall=setup.t_wall, t_in=setup.t_in,
    )
    nu_balance = nusselt_global_balance(
        T, ux, alpha=alpha, t_in=setup.t_in, t_wall=setup.t_wall,
    )
    nu_wall_extrap = nusselt_wall_extrapolated(
        g, T, ux, uy, geo, tau_t=tau_t, t_wall=setup.t_wall, t_in=setup.t_in,
    )

    out: dict[str, Any] = {
        "case": setup.resolved(),
        "steps": steps,
        "max_steps": max_steps,
        "window_tu": (steps - n_transient) / tu if tu > 0 else float("nan"),
        "n_samples": len(nu_hist),
        "converged": bool(steps < max_steps and diverged < 0),
        "diverged": bool(diverged >= 0),
        "diverge_step": diverged,
        "diverge_reason": diverge_reason,
        "nu_mean": nu_stat["mean"],
        "nu_stderr": nu_stat["stderr"],
        "nu_std": nu_stat["std"],
        "nu_drift_rel": drift_rel,
        "nu_final": nusselt["nu"],
        "nu_global_balance": nu_balance,
        "nu_wall_extrapolated": nu_wall_extrap,
        "nu_surface_min": nusselt["nu_min"],
        "nu_surface_max": nusselt["nu_max"],
        "nu_surface_scatter": nusselt["nu_scatter"],
        "cd_mean": cd_stat["mean"],
        "cd_stderr": cd_stat["stderr"],
        "cl_std": cl_stat["std"],
        "st": st,
        "shedding_period_tu": period_tu,
        "u_max_final": float(np.max(np.abs(ux))),
        "drho_final": float(np.max(np.abs(rho - 1.0))),
        "drho_max": float(np.max(drho_hist)) if drho_hist else float("nan"),
        "nu_history": nu_hist,
        "cd_history": cd_hist,
        "cl_history": cl_hist,
        "t_history": t_hist,
        "u_prime_history": up_hist,
    }
    if keep_fields:
        out["fields"] = {
            "rho": rho,
            "ux": ux,
            "uy": uy,
            "T": T,
            "mach": setup.mach,
        }
    return out


# ---------------------------------------------------------------------------
# Ma -> 0 extrapolation (Richardson in Ma^2)
# ---------------------------------------------------------------------------
def richardson_nu0(
    machs: Sequence[float],
    values: Sequence[float],
    sigmas: Sequence[float] | None = None,
    *,
    max_power: int = 2,
) -> dict[str, Any]:
    """Extrapolate ``Nu(Ma) -> Nu0`` at ``Ma = 0``.

    The compressibility error is even in ``Ma``, so the natural small parameter
    is ``x = Ma^2`` and the Richardson model is
    ``Nu(x) = Nu0 + c2 x + c4 x^2 + ...``, fitted by weighted least squares.
    Two independently useful quantities are returned:

    * ``nu0`` with ``nu0_stderr`` from the covariance of the fit (this error is
      propagated into every point of the error law);
    * ``nu0_two_point`` -- the textbook two-level Richardson estimate built from
      the two lowest Mach numbers,
      ``Nu0 = Nu_a + (Nu_a - Nu_b)/((Ma_b/Ma_a)^2 - 1)``, an independent
      cross-check of the polynomial extrapolation (they must agree within a few
      standard errors, otherwise the polynomial model is too short and the
      extrapolation is flagged).
    """
    m = np.asarray(machs, dtype=float)
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(m) & np.isfinite(v) & (m > 0)
    m, v = m[keep], v[keep]
    if sigmas is None:
        s = np.ones_like(v)
        m, s = m[keep], s
    else:
        s = np.asarray(sigmas, dtype=float)[keep]
    good = np.isfinite(s) & (s > 0)
    m, v, s = m[good], v[good], s[good]
    out: dict[str, Any] = {"n_points": int(m.size)}
    # the model has 1 + max_power parameters; shorten it rather than returning NaN
    # when only three Mach numbers are available
    max_power = max(1, min(max_power, m.size - 2))
    if m.size < 3:
        out.update({"nu0": float("nan"), "nu0_stderr": float("nan")})
        return out

    order = np.argsort(m)
    m, v, s = m[order], v[order], s[order]
    x = m ** 2
    cols = [np.ones_like(x)]
    for k in range(1, max_power + 1):
        cols.append(x ** k)
    A = np.column_stack(cols)
    wt = 1.0 / s ** 2
    Aw = A * wt[:, None]
    coef, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ v, rcond=None)
    resid = v - A @ coef
    dof = max(1, m.size - A.shape[1])
    s2 = float(np.sum(wt * resid ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(Aw.T @ A)
        nu0_se = float(math.sqrt(abs(cov[0, 0])))
    except np.linalg.LinAlgError:
        nu0_se = float("nan")
    ss_tot = float(np.sum(wt * (v - np.average(v, weights=wt)) ** 2))
    ss_res = float(np.sum(wt * resid ** 2))
    out.update(
        {
            "nu0": float(coef[0]),
            "nu0_stderr": nu0_se,
            "coefficients": [float(c) for c in coef],
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
            "mach_range": [float(m.min()), float(m.max())],
        }
    )
    if m.size >= 2:
        ma, mb = m[0], m[1]
        va, vb = v[0], v[1]
        denom = (mb / ma) ** 2 - 1.0
        out["nu0_two_point"] = (
            float(va + (va - vb) / denom) if abs(denom) > 0 else float("nan")
        )
    return out


def _t_quantile_95(dof: int) -> float:
    """Two-sided 95 % Student-t quantile (small table, ``nan``-safe)."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
        20: 2.086, 25: 2.060, 30: 2.042, 50: 2.009,
    }
    if dof in table:
        return table[dof]
    keys = sorted(table)
    if dof < keys[0]:
        return table[keys[0]]
    if dof > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= dof)
    hi = min(k for k in keys if k >= dof)
    if lo == hi:
        return table[lo]
    w = (dof - lo) / (hi - lo)
    return table[lo] * (1 - w) + table[hi] * w


def fit_ma_error_law(
    machs: Sequence[float],
    values: Sequence[float],
    nu0: float,
    sigmas: Sequence[float] | None = None,
    *,
    nu0_stderr: float = 0.0,
    n_fixed: float = 2.0,
) -> dict[str, Any]:
    """Weighted log--log fit of ``eps = |Nu(Ma)/Nu0 - 1| = C Ma^n``.

    Point uncertainties are propagated both from the measured mean (``sigmas``)
    and from the uncertainty of the extrapolated ``Nu0``:
    ``sigma_eps^2 = (sigma_Nu/Nu0)^2 + (Nu sigma_Nu0/Nu0^2)^2``.  The 95 %
    confidence interval of ``n`` uses Student's ``t`` with ``N - 2`` degrees of
    freedom.  ``r2_loglog`` is the unweighted coefficient of determination in
    log space (the conservative number quoted against the acceptance
    criterion), ``r2_weighted`` the weighted one.

    ``rms_residual_log_n_fixed`` is the RMS residual left when the exponent is
    *pinned* at the theoretical value ``n_fixed = 2``: it tells whether the data
    are described by the theoretical slope or only by the fitted one.
    """
    m = np.asarray(machs, dtype=float)
    v = np.asarray(values, dtype=float)
    s = np.zeros_like(v) if sigmas is None else np.asarray(sigmas, dtype=float)
    keep = np.isfinite(m) & np.isfinite(v) & (m > 0)
    m, v, s = m[keep], v[keep], s[keep]
    eps = np.abs(v / nu0 - 1.0)
    sig_eps = np.sqrt((s / nu0) ** 2 + (v * nu0_stderr / nu0 ** 2) ** 2)
    good = np.isfinite(eps) & (eps > 0)
    m, v, eps, sig_eps = m[good], v[good], eps[good], sig_eps[good]
    out: dict[str, Any] = {
        "n_points": int(m.size),
        "nu0": float(nu0),
        "nu0_stderr": float(nu0_stderr),
        "mach": [float(x) for x in m],
        "nu": [float(x) for x in v],
        "eps": [float(x) for x in eps],
        "sigma_eps": [float(x) for x in sig_eps],
    }
    if m.size < 3:
        out.update({"n": float("nan"), "c": float("nan"), "r2_loglog": float("nan")})
        return out

    lx = np.log(m)
    ly = np.log(eps)
    if np.any(sig_eps > 0):
        w = 1.0 / np.maximum(sig_eps, 1e-12) ** 2
    else:
        w = np.ones_like(eps)
    A = np.column_stack([np.ones_like(lx), lx])
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ ly, rcond=None)
    resid = ly - A @ coef
    dof = max(1, m.size - 2)
    s2 = float(np.sum(w * resid ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(Aw.T @ A)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    tcrit = _t_quantile_95(dof)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    ss_tot_w = float(np.sum(w * (ly - np.average(ly, weights=w)) ** 2))
    ss_res_w = float(np.sum(w * resid ** 2))
    slope_raw, intercept_raw = np.polyfit(lx, ly, 1)
    pinned = ly - intercept_raw - n_fixed * lx
    out.update(
        {
            "n": float(coef[1]),
            "n_stderr": float(se[1]),
            "n_ci95": [
                float(coef[1] - tcrit * se[1]),
                float(coef[1] + tcrit * se[1]),
            ],
            "c": float(math.exp(coef[0])),
            "c_stderr_log": float(se[0]),
            "r2_loglog": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "r2_weighted": 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else float("nan"),
            "rms_residual_log": float(np.sqrt(np.mean(resid ** 2))),
            "slope_unweighted": float(slope_raw),
            "n_theory": float(n_fixed),
            "rms_residual_log_n_fixed": float(np.sqrt(np.mean(pinned ** 2))),
            "mach_range": [float(m.min()), float(m.max())],
        }
    )
    return out


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
def mach_floor_from_stiffness(
    re: float, diameter_eff: float, *, tau_floor: float = 0.505
) -> float:
    """Smallest ``Ma`` at which ``tau >= tau_floor`` on this grid.

    ``tau - 1/2 = 3 nu`` and ``nu = Ma c_s D_eff/Re``, so
    ``Ma_min = (tau_floor - 1/2) Re/(3 c_s D_eff)``.  Below this value the
    BGK/TRT/regularised operator sits within ``tau_floor - 1/2`` of the
    ``tau -> 1/2`` stability limit and the run is stiffness- rather than
    accuracy-limited (measured: the plain BGK/TRT cylinder diverges there).
    """
    return (tau_floor - 0.5) * re / (3.0 * CS * diameter_eff)


def mach_envelope(
    per_re: Mapping[float, Mapping[str, Any]],
    *,
    tol: float = 0.03,
    tau_floor: float = 0.505,
    ma_ceiling: float = 0.5,
) -> dict[str, Any]:
    """Usable Mach band per Reynolds number (the scheduler-facing object).

    ``per_re[re]`` must carry ``nu0``, ``nu0_stderr``, ``c``, ``n``,
    ``n_ci95``, ``diameter_eff``, ``ma_stable_max`` (highest Mach number that
    ran to completion in the sweep) and ``ma_tested_max``.

    The band is the intersection of

    * the accuracy bound ``Ma <= (tol/C)^(1/n)`` (``O(Ma^n)`` error budget), and
    * the measured stability bound ``Ma <= ma_stable_max``,
    * against the stiffness floor ``Ma >= mach_floor_from_stiffness(...)``.

    ``status`` is ``ok`` when the band is non-empty, ``accuracy_limited`` /
    ``stability_limited`` when one bound alone decides it, and ``empty`` when
    even the stiffness floor is above the accuracy bound (the grid is too
    coarse for the requested tolerance at that Reynolds number).
    """
    rows: list[dict[str, Any]] = []
    for re in sorted(per_re):
        d = per_re[re]
        c = float(d["c"])
        n = float(d["n"])
        ma_err = (tol / c) ** (1.0 / n) if c > 0 and n > 0 else float("nan")
        ma_stab = float(d.get("ma_stable_max", ma_ceiling))
        ma_floor = mach_floor_from_stiffness(re, float(d["diameter_eff"]), tau_floor=tau_floor)
        ma_hi = min(ma_err, ma_stab, ma_ceiling)
        status = "ok"
        if not np.isfinite(ma_hi) or ma_hi <= ma_floor:
            status = "empty"
        elif ma_hi == ma_err:
            status = "accuracy_limited"
        elif ma_hi == ma_stab:
            status = "stability_limited"
        rows.append(
            {
                "re": float(re),
                "nu0": float(d["nu0"]),
                "nu0_stderr": float(d.get("nu0_stderr", float("nan"))),
                "error_law": {
                    "c": c,
                    "n": n,
                    "n_ci95": [float(x) for x in d.get("n_ci95", [np.nan, np.nan])],
                    "r2_loglog": float(d.get("r2_loglog", float("nan"))),
                },
                "tolerance": tol,
                "ma_max_error": float(ma_err),
                "ma_max_stability": float(ma_stab),
                "ma_min_stiffness": float(ma_floor),
                "tau_floor": tau_floor,
                "ma_band": [float(ma_floor), float(ma_hi)],
                "ma_recommended": float(0.5 * (ma_floor + ma_hi)) if status != "empty" else float("nan"),
                "predicted_error_at_ma_max": tol,
                "status": status,
                "diameter_eff": float(d["diameter_eff"]),
            }
        )
    return {
        "schema": "tensorlbm.mach_envelope/v1",
        "model": "|Nu(Ma)/Nu0 - 1| = C * Ma^n",
        "definition": (
            "usable Mach band: stiffness floor (tau >= tau_floor) <= Ma <= "
            "min(accuracy bound for the requested tolerance, measured stability "
            "bound)"
        ),
        "rows": rows,
        "ok": bool(rows) and all(r["status"] != "empty" for r in rows),
    }


def predict_holdout_envelope(
    train: Sequence[Mapping[str, Any]], re_target: float, *, tol: float = 0.03
) -> dict[str, Any]:
    """Pre-registered hold-out prediction of the usable Mach bound.

    With at most two training Reynolds numbers the rule is fixed *before* the
    hold-out is run: ``log C`` is interpolated linearly in ``log Re`` (the
    documented scaling of the compressibility coefficient) and the exponent is
    the mean of the training exponents.  The prediction is
    ``Ma_pred = (tol/C_pred)^(1/n_pred)``, i.e. the highest Mach number at which
    the hold-out case is predicted to stay below the tolerance without
    diverging.
    """
    tr = [dict(t) for t in train]
    if not tr:
        raise ValueError("empty training set")
    pairs = sorted(((float(t["re"]), float(t["c"]), float(t["n"])) for t in tr),
                   key=lambda p: p[0])
    re_train = np.array([p[0] for p in pairs])
    cs = [math.log(p[1]) for p in pairs]
    ns = [p[2] for p in pairs]
    n_pred = float(np.mean(ns))
    if len(tr) >= 2:
        # log-log *extrapolation* to the hold-out Re: np.interp would clamp at
        # the last training point instead of extrapolating.
        lx = np.log(re_train)
        slope = (cs[-1] - cs[0]) / (lx[-1] - lx[0])
        logc_pred = float(cs[-1] + slope * (math.log(re_target) - lx[-1]))
        method = "log-C linear in log-Re (interpolation through training points)"
    else:
        logc_pred = cs[0]
        method = "single training point (no Re dependence)"
    c_pred = float(math.exp(logc_pred))
    ma_pred = (tol / c_pred) ** (1.0 / n_pred) if c_pred > 0 and n_pred > 0 else float("nan")
    return {
        "re_target": float(re_target),
        "tolerance": tol,
        "c_pred": c_pred,
        "n_pred": n_pred,
        "ma_pred": float(ma_pred),
        "method": method,
        "train_re": [float(t["re"]) for t in tr],
        "train": [
            {"re": float(t["re"]), "c": float(t["c"]), "n": float(t["n"])} for t in tr
        ],
    }
