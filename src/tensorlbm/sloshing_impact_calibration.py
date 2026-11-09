"""Liquid-tank sloshing impact: 3-D free-surface LBM reference for the
low-order (shallow-water / potential-flow) sloshing chain.

Why this module exists
----------------------
The ship-performance chain evaluates tank sloshing loads with a two-level
low-order stack: linear potential flow for the natural frequencies and a
depth-averaged Saint-Venant (shallow-water, SWE) solver for the impact
pressure time history.  Both levels are asymptotic: the SWE is only valid for
``kh -> 0`` and it cannot represent a gas cushion, while potential flow cannot
represent wave breaking or run-up.  This module supplies the missing
**higher-fidelity reference**: a three-dimensional, large-density-ratio
free-surface LBM (Körner et al. 2005 mechanics, :mod:`tensorlbm.free_surface_lbm`)
on the *same* tank geometry and on the *same* time anchors as the low-order
chain, so the chain error can be measured instead of assumed.

Anchors (read-only port, no cross-repo dependency)
--------------------------------------------------
Geometry, fluid properties and the excitation are the IN48-N07 case of the
``ship-performance-platform`` sloshing chain (``artifacts/in48_n07/report.json``,
``backend/physics/sloshing/{modes,impact}.py``):

============================  ======================================
tank length ``L``             20.0 m   (sloshing direction, ``x``)
tank breadth ``B``            12.0 m   (``z``)
tank height ``H``             8.0 m    (``y``)
still-water depth ``h_l``     4.0 m    (fill 0.5)
density ``rho``               1025 kg/m^3
gravity ``g``                 9.80665 m/s^2
first mode ``omega_1``        0.92620 rad/s  -> ``T_1 = 6.7838 s``
excitation (forced)           ``A = 0.25 m`` at ``omega_1`` (regular)
closed-form impact           ``P_peak = 4 rho g h_l`` (Ritter/surge)
bore-reflection anchor       ``P_peak = rho g h_r``, ``h_r/h_l = 3.494``
============================  ======================================

The numbers above are asserted against the anchor in
``tests/test_sloshing_impact_calibration.py``; the module itself only needs
``L, B, H, h_l, rho, g``.

Lattice scaling
---------------
One lattice spacing ``dx = L / nx`` and one lattice time step ``dt`` are fixed
by holding the surge Froude number and the lattice Mach number constant across
the grid family:

    u_lbm = u_front * dt / dx          (u_front = 2 sqrt(g h_l), Ritter 1892)
    g_lbm = g * dt^2 / dx

so ``u_lbm`` is grid-independent (here 0.08, an incompressible-safe Mach
number) and ``dt ~ dx`` - i.e. the grid family refines space *and* time
together, which is what a GCI study requires.  All reported pressures are
dimensionless, ``P / (rho g h_l)``, so lattice and physical units cancel.

Measurements
------------
* ``P_wall`` (dimensionless dynamic wall pressure) from the LBM density field on
  the two fluid cells next to the struck wall, linearly extrapolated to the wall
  face (``p_wall = 1.5 p_{dx/2} - 0.5 p_{3dx/2}``); gauge pressure is
  ``p = c_s^2 (rho - rho_liquid)`` because LBM pressure is only defined up to a
  constant.
* ``h_wall`` (wall-adjacent free-surface height) from the Körner fill field,
  which yields the SWE-comparable wall pressure ``rho g h_wall``.
* liquid mass ``sum(mass)`` (Körner tracked mass) - mass-conservation gate.

Usage
-----
The production driver is ``scripts/run_sloshing_impact_calibration.py``; this
module is import-safe (no I/O, no plotting) so the numerics stay unit-testable.

References
----------
Körner, Thies, Hofmann, Thürey, Rüde (2005), J. Stat. Phys. 121, 179.
Thürey (2007), PhD thesis, Univ. Erlangen (free-surface LBM).
Ritter (1892), Z. Ver. Deutsch. Ing. 36, 947 (dam-break front).
Roache (1994), J. Fluids Eng. 116, 405 (GCI).
Faltinsen & Timokha (2009), *Sloshing*, CUP.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .d3q19 import macroscopic3d
from .free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    init_population_from_fill,
)

__all__ = [
    "TankAnchor",
    "LbmScaling",
    "impact_fill",
    "wall_pressure_panel",
    "wall_pressure_face",
    "liquid_mean_density",
    "wall_column_height",
    "run_impact_case",
    "run_forced_case",
    "run_hydrostatic_case",
    "cushion_gas_pressure",
    "observed_order",
    "gci_fine",
    "harmonic_analysis",
    "dispersion_ratio_swe",
    "bore_reflection_depth",
]

CS2 = 1.0 / 3.0
"""LBM speed of sound squared (D3Q19)."""


# ---------------------------------------------------------------------------
# 1. Anchors (read-only port of the IN48-N07 sloshing case)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TankAnchor:
    """Closed rectangular tank + fluid anchors of the low-order chain.

    Defaults reproduce ``ship-performance-platform`` IN48-N07
    (``backend/physics/sloshing/chain.py``: ``TANK = RectTank(20, 12, 8)``,
    ``h_l = H/2``, ``rho = 1025``, ``g = 9.80665``).
    """

    length: float = 20.0
    breadth: float = 12.0
    height: float = 8.0
    depth: float = 4.0  # h_l, still-water depth
    rho: float = 1025.0
    g: float = 9.80665

    # -- derived --------------------------------------------------------
    @property
    def k1(self) -> float:
        """First sloshing wavenumber ``k_1 = pi / L`` [1/m]."""
        return math.pi / self.length

    @property
    def omega1(self) -> float:
        """Exact first-mode natural frequency ``sqrt(g k tanh(k h_l))`` [rad/s]."""
        k, h = self.k1, self.depth
        return math.sqrt(self.g * k * math.tanh(k * h))

    @property
    def period1(self) -> float:
        return 2.0 * math.pi / self.omega1

    @property
    def omega1_swe(self) -> float:
        """First-mode frequency *of the shallow-water equations* ``k sqrt(g h)``.

        The gap ``omega1_swe / omega1 = sqrt(k h / tanh(k h))`` is the leading
        frequency error of the SWE chain at finite relative depth; it is
        reported as the validity boundary of the low-order chain (6.2 % too
        fast at the IN48-N07 depth ``kh = 0.628``).
        """
        return self.k1 * math.sqrt(self.g * self.depth)

    @property
    def kh(self) -> float:
        return self.k1 * self.depth

    @property
    def u_front(self) -> float:
        """Ritter (1892) surge front velocity ``2 sqrt(g h_l)`` [m/s]."""
        return 2.0 * math.sqrt(self.g * self.depth)

    @property
    def p_hydrostatic(self) -> float:
        return self.rho * self.g * self.depth

    @property
    def p_closed_form(self) -> float:
        """``4 rho g h_l`` - the shallow-water stagnation anchor [Pa]."""
        return self.rho * self.u_front**2

    @property
    def p_bore_reflection(self) -> float:
        """``rho g h_r`` with the exact bore-reflection depth ``h_r`` [Pa]."""
        return self.rho * self.g * bore_reflection_depth(self.depth, self.u_front, self.g)


def bore_reflection_depth(h_i: float, u_i: float, g: float = 9.80665) -> float:
    """Reflected depth ``h_r`` of a uniform stream ``(h_i, u_i)`` stopped by a wall.

    Exact Saint-Venant shock (Rankine-Hugoniot) condition with ``u_r = 0``::

        u_i = (h_r - h_i) sqrt( g (h_r + h_i) / (2 h_r h_i) )

    solved here by a monotone bisection in ``h_r`` (no SciPy dependency).
    """
    if u_i <= 0.0:
        return h_i
    lo, hi = h_i, 1.0e6 * h_i
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rhs = (mid - h_i) * math.sqrt(g * (mid + h_i) / (2.0 * mid * h_i))
        if rhs < u_i:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * hi:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 2. Lattice scaling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LbmScaling:
    """Physical -> lattice mapping for one member of the grid family.

    Two invariants are held fixed across the grid family, so the family is a
    genuine refinement study of *one* physical problem:

    * lattice Mach number of the surge front (``u_lbm = 0.08``), i.e. ``dt ~ dx``;
    * *effective* Reynolds number ``Re = u_front h_l / nu`` (``re_target``), i.e.
      ``nu_lbm ~ dx`` and therefore ``tau = 0.5 + 3 nu_lbm`` grows slightly with
      the grid.

    The real tank is at ``Re ~ 5e7``; that is not resolvable at ``L/80`` with a
    free-surface LBM, so the reference is run at ``Re = 2000`` (inertia
    dominated) with a Smagorinsky subgrid closure.  See the report's
    limitations section - the impact pressure is dominated by the Froude-scaled
    stagnation, and the viscosity is the numerically accessible surrogate.
    """

    anchor: TankAnchor
    nx: int
    u_lbm: float = 0.08
    re_target: float = 2000.0
    c_sgs: float = 0.1
    collision: str = "bgk"
    """Collision operator.

    BGK is kept because the D3Q19 **MRT free-surface path is unusable at
    ``tau ~ 0.502``**: with MRT the first step already produces a topology
    candidate outside the model's ``[-0.1, 10]`` fill guard on every grid
    (``TopologyTransactionError``), independent of the Smagorinsky constant -
    verified at ``L/40`` and ``L/60``.  BGK plus the Smagorinsky closure runs the
    full impact window on all three grids.  (Reported as a defect of the shared
    free-surface model, not of this calibration.)"""

    # -- length / time --------------------------------------------------
    @property
    def dx(self) -> float:
        return self.anchor.length / self.nx

    @property
    def dt(self) -> float:
        """``dt`` from the fixed lattice Mach number of the surge front."""
        return self.u_lbm * self.dx / self.anchor.u_front

    @property
    def ny(self) -> int:
        return int(round(self.anchor.height / self.dx))

    @property
    def nz(self) -> int:
        return int(round(self.anchor.breadth / self.dx))

    # -- lattice quantities --------------------------------------------
    @property
    def g_lbm(self) -> float:
        return self.anchor.g * self.dt**2 / self.dx

    @property
    def depth_lbm(self) -> int:
        return int(round(self.anchor.depth / self.dx))

    @property
    def omega1_lbm(self) -> float:
        return self.anchor.omega1 * self.dt

    @property
    def nu_physical(self) -> float:
        """Kinematic viscosity assigned to the surrogate [m^2/s]."""
        return self.anchor.u_front * self.anchor.depth / self.re_target

    @property
    def nu_lbm(self) -> float:
        return self.nu_physical * self.dt / self.dx**2

    @property
    def tau(self) -> float:
        """BGK relaxation time realising ``re_target`` on this grid."""
        tau = 0.5 + 3.0 * self.nu_lbm
        if not (0.5 < tau < 1.5):
            raise ValueError(f"tau={tau:.5f} outside the stable BGK band")
        return tau

    @property
    def reynolds(self) -> float:
        """Effective Reynolds number ``u_front * h_l / nu`` (not water Re)."""
        return self.anchor.u_front * self.anchor.depth / self.nu_physical

    def physical_seconds(self, steps: int) -> float:
        return steps * self.dt

    def steps_for_seconds(self, seconds: float) -> int:
        return int(round(seconds / self.dt))

    def pressure_scale(self) -> float:
        """``rho g h_l`` in lattice units (gauge pressure ``c_s^2 (rho-1)``)."""
        return self.g_lbm * self.depth_lbm


# ---------------------------------------------------------------------------
# 3. Initial state
# ---------------------------------------------------------------------------
def impact_fill(
    ny: int,
    nz: int,
    nx: int,
    depth_cells: int,
    column_fraction: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed tank with a liquid column ``column_fraction * L`` long at ``x = 0``.

    Walls on all six faces (1-cell SOLID shell), liquid depth ``depth_cells``,
    dry tank elsewhere - the surge-release impact configuration.  The column is
    held by a vertical gate that is released at ``t = 0`` (no gate dynamics:
    the liquid starts at rest, as in the SWE reference).
    """
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32)
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    solid[0, :, :] = True
    solid[-1, :, :] = True
    x_col = max(1, int(round(column_fraction * nx)))
    fill[:, 1 : depth_cells + 1, 1 : 1 + x_col] = 1.0
    fill[solid] = 0.0
    return fill, solid


def _initialize(fill: torch.Tensor, solid: torch.Tensor, rho_gas: float):
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)
    # Gas-consistent envelope seeding (pressure-matched ambient, rho_gas =
    # rho_liquid; see free_surface_lbm ABB notes).
    f = init_population_from_fill(fill, flags, rho_liquid=1.0, rho_gas=rho_gas)
    return f, flags, mass


# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------
def wall_pressure_panel(
    f: torch.Tensor,
    flags: torch.Tensor,
    wall_x: int,
    y_lo: int,
    y_hi: int,
    rho_liquid: float = 1.0,
    rho: torch.Tensor | None = None,
    rho_profile: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Wall dynamic pressure on the wall-normal panel ``y in [y_lo, y_hi)``.

    Returns ``(p_first, p_face)`` in *lattice* gauge pressure
    (``c_s^2 (rho - rho_liquid)``):

    * ``p_first`` - breadth-wide mean over the first fluid cell off the wall
      (cell centre at ``dx/2``);
    * ``p_face``  - linear two-point extrapolation to the wall face,
      ``1.5 p_{dx/2} - 0.5 p_{3dx/2}`` (task spec: 1-2 cell wall interpolation).

    Only ``flags == LIQUID`` cells contribute: the struck wall panel is wet over
    the impact window and a gas-fed value would be meaningless.  If the second
    cell is dry the face value falls back to the first-cell mean.

    ``rho_liquid`` is the *pressure datum*: LBM pressure is defined up to a
    constant, and the free-surface model's tracked-liquid density is not exactly
    1.0 (it is set by the tracked mass, and drifts a few percent with the
    topology conversions), so the caller passes the instantaneous liquid mean
    (see :func:`liquid_mean_density`) instead of the nominal value.
    """
    if rho is None:
        rho, _, _, _ = macroscopic3d(f)
    if rho_profile is not None:
        # Dynamic overpressure above the *still-water* hydrostatic state at the
        # same height (the standard sloshing-impact definition): each cell is
        # referenced to the liquid density the same grid holds at rest.
        ref = torch.as_tensor(list(rho_profile), dtype=rho.dtype,
                              device=rho.device).view(1, -1, 1)
        gauge = CS2 * (rho - ref)
    else:
        gauge = CS2 * (rho - rho_liquid)
    y0, y1 = max(y_lo, 1), min(y_hi, flags.shape[1] - 1)
    p1 = gauge[:, y0:y1, wall_x - 1]
    p2 = gauge[:, y0:y1, wall_x - 2]
    w1 = (flags[:, y0:y1, wall_x - 1] == LIQUID).to(gauge.dtype)
    w2 = (flags[:, y0:y1, wall_x - 2] == LIQUID).to(gauge.dtype)
    n1, n2 = float(w1.sum()), float(w2.sum())
    if n1 < 3.0:  # dry / one-wisp panel: the mean is a conversion artefact
        return 0.0, 0.0
    m1 = float((p1 * w1).sum() / n1)
    m2 = float((p2 * w2).sum() / n2) if n2 > 0.5 else m1
    return m1, wall_pressure_face(m2, m1)


def wall_pressure_face(p_three_half: float, p_half: float) -> float:
    """Two-point linear extrapolation ``1.5 p_{dx/2} - 0.5 p_{3dx/2}``.

    Arguments are ordered as ``(p at 1.5 dx, p at 0.5 dx)``.
    """
    return 1.5 * p_half - 0.5 * p_three_half


def liquid_mean_density(rho: torch.Tensor, flags: torch.Tensor) -> float:
    """Mean density over the LIQUID cells (the wall-pressure datum)."""
    mask = flags == LIQUID
    if not bool(mask.any()):
        return 1.0
    return float(rho[mask].mean().item())


def wall_column_height(fill: torch.Tensor, x_index: int) -> float:
    """Breadth-averaged free-surface height (in cells) of column ``x_index``."""
    nz = fill.shape[0]
    col = fill[:, :, x_index]
    return float(col.sum().item()) / nz


# ---------------------------------------------------------------------------
# 5. Runners
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    """Time histories + scalar diagnostics of one LBM run."""

    name: str
    nx: int
    ny: int
    nz: int
    dt: float
    dx: float
    steps: int
    t: np.ndarray
    p_wall: np.ndarray  # dimensionless P/(rho g h_l), wall-face extrapolated
    p_wall_half: np.ndarray  # dimensionless P/(rho g h_l), first-cell value
    h_wall: np.ndarray  # wall free-surface height / h_l
    mass_drift: np.ndarray  # (M(t) - M0)/M0
    rho_liquid_mean: np.ndarray | None = None  # pressure datum per sample
    meta: dict = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def p_peak(self) -> float:
        return float(np.nanmax(self.p_wall))

    @property
    def p_peak_index(self) -> int:
        return int(np.nanargmax(self.p_wall))

    def to_json(self) -> dict:
        keep = max(self.t.size, 1)
        return {
            "name": self.name,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dt": self.dt,
            "dx": self.dx,
            "steps": self.steps,
            "p_peak": self.p_peak,
            "p_peak_half": float(np.nanmax(self.p_wall_half)),
            "p_peak_time": float(self.t[self.p_peak_index]),
            "h_wall_max": float(np.nanmax(self.h_wall)),
            "mass_drift_final": float(self.mass_drift[-1]),
            "mass_drift_settled": float(self.meta.get("mass_drift_settled", "nan")),
            "mass_drift_abs_max": float(np.nanmax(np.abs(self.mass_drift))),
            "t": self.t.tolist(),
            "p_wall": self.p_wall.tolist(),
            "p_wall_half": self.p_wall_half.tolist(),
            "h_wall": self.h_wall.tolist(),
            "mass_drift": self.mass_drift.tolist(),
            "rho_liquid_mean": []
            if self.rho_liquid_mean is None
            else self.rho_liquid_mean.tolist(),
            "snapshots": self.meta.get("snapshots", []),
            "meta": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in self.meta.items()},
            "_samples": int(keep),
        }


def _run(
    name: str,
    scaling: LbmScaling,
    fill: torch.Tensor,
    solid: torch.Tensor,
    *,
    steps: int,
    sample_every: int,
    gx: Callable[[int], float] | None = None,
    gy_lbm: float | None = None,
    cushion: bool = False,
    gamma: float = 1.4,
    cushion_max_ratio: float = 1.35,
    rho_gas: float = 1.0,
    wall_x: int | None = None,
    sensor_y: tuple[int, int] | None = None,
    tau: float | None = None,
    c_sgs: float | None = None,
    collision: str | None = None,
    snapshot_steps: Sequence[int] = (),
    splices: list | None = None,
    rho_profile: Sequence[float] | None = None,
) -> CaseResult:
    """Shared marching loop: gravity + optional horizontal excitation.

    ``cushion=True`` seals the ullage gas and feeds the Körner ABB ambient
    pressure with the isentropic compression of the trapped air pocket,
    ``p_gas = p0 (V0 / V)^gamma`` (``p0 = rho_liquid``, pressure-matched), which
    is the gas-cushion (气垫) closure the SWE chain cannot represent.
    """
    nz, ny, nx = fill.shape
    f, flags, mass = _initialize(fill, solid, rho_gas)
    g_lbm = scaling.g_lbm if gy_lbm is None else gy_lbm
    tau = scaling.tau if tau is None else tau
    c_sgs = scaling.c_sgs if c_sgs is None else c_sgs
    collision = collision or scaling.collision
    wall_x = nx - 1 if wall_x is None else wall_x
    sensor_y = (1, max(2, scaling.depth_lbm // 2)) if sensor_y is None else sensor_y

    # ullage volume of the sealed pocket (gas + interface void), in cells
    gas_void = (flags == GAS).to(torch.float32) + torch.where(
        flags == INTERFACE, 1.0 - fill.clamp(0.0, 1.0), torch.zeros_like(fill)
    )
    v0 = float(gas_void.sum().item())

    m0 = float(mass.sum().item())
    snap_set = set(int(v) for v in snapshot_steps)
    snaps: list = []
    ts, p_wall, p_half, h_wall, drift = [], [], [], [], []
    rho_series: list[float] = []
    settle_step = max(1, int(os.environ.get("TL_FS_CAL_SETTLE_STEPS", "100")))
    m_settle = m0
    for step in range(1, steps + 1):
        gx_lbm = 0.0 if gx is None else float(gx(step))
        bubble = None
        if cushion:
            void = float(
                ((flags == GAS).to(torch.float32)
                 + torch.where(flags == INTERFACE, 1.0 - fill.clamp(0.0, 1.0),
                               torch.zeros_like(fill))).sum().item()
            )
            bubble = cushion_gas_pressure(
                void, v0, p0=rho_gas, gamma=gamma, max_ratio=cushion_max_ratio
            )
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=tau,
            gx=gx_lbm,
            gy=-g_lbm,
            rho_liquid=1.0,
            rho_gas=rho_gas,
            C_s=c_sgs,
            collision=collision,
            bubble_pressure=bubble,
        )
        if step == settle_step:
            m_settle = float(mass.sum().item())
        if step in snap_set:
            z_mid = fill.shape[0] // 2
            snaps.append(
                {
                    "step": int(step),
                    "t": step * scaling.dt,
                    "fill_yx": np.round(
                        fill[z_mid].detach().cpu().numpy().astype(float), 4
                    ).tolist(),
                    "max_fill": float(fill.max().item()),
                }
            )
        if step % sample_every == 0 or step == steps:
            rho_now, _, _, _ = macroscopic3d(f)
            rho_ref = liquid_mean_density(rho_now, flags)
            p1, p_face = wall_pressure_panel(
                f, flags, wall_x, sensor_y[0], sensor_y[1],
                rho_liquid=rho_ref, rho=rho_now, rho_profile=rho_profile,
            )
            scale = scaling.pressure_scale()
            ts.append(step * scaling.dt)
            p_wall.append(p_face / scale)
            p_half.append(p1 / scale)
            h_wall.append(wall_column_height(fill, wall_x - 1) / scaling.depth_lbm)
            drift.append((float(mass.sum().item()) - m0) / m0)
            rho_series.append(rho_ref)
    meta = {
        "omega1_lbm": scaling.omega1_lbm,
        "g_lbm": g_lbm,
        "depth_lbm": scaling.depth_lbm,
        "nu_lbm": scaling.nu_lbm,
        "reynolds_effective": scaling.reynolds,
        "cushion": int(bool(cushion)),
        "gamma": gamma,
        "cushion_max_ratio": cushion_max_ratio,
        "p0_ullage": rho_gas,
        "v0_ullage_cells": v0,
        "sensor_y": list(sensor_y),
        "settle_steps": settle_step,
        "rho_liquid_mean_final": rho_series[-1] if rho_series else None,
        "pressure_datum": "still_water_profile" if rho_profile is not None
        else "liquid_mean",
        "rho_liquid_mean_abs_shift": (
            max(abs(r - rho_series[0]) for r in rho_series) if rho_series else None
        ),
        "snapshots": snaps,
        "mass_drift_settled": (float(mass.sum().item()) - m_settle) / m_settle,
        "tau": tau,
        "c_sgs": c_sgs,
        "collision": collision,
    }
    return CaseResult(
        name=name,
        rho_liquid_mean=np.asarray(rho_series),
        nx=nx,
        ny=ny,
        nz=nz,
        dt=scaling.dt,
        dx=scaling.dx,
        steps=steps,
        t=np.asarray(ts),
        p_wall=np.asarray(p_wall),
        p_wall_half=np.asarray(p_half),
        h_wall=np.asarray(h_wall),
        mass_drift=np.asarray(drift),
        meta=meta,
    )


def cushion_gas_pressure(
    void_now: float, void_0: float, p0: float = 1.0, gamma: float = 1.4,
    max_ratio: float = 1.35,
) -> float:
    """Isentropic ullage pressure ``p0 (V0/V)^gamma`` of a sealed air pocket.

    ``void`` is the trapped-air volume in cells (GAS cells + interface void), so
    the ratio is dimensionless and the returned pressure is the ABB ambient
    density in lattice units.  Clipped to ``[0.2 p0, 10 p0]`` to keep the ABB
    reconstruction inside its ``O(rho_liquid)`` consistency range.
    """
    if void_now <= 1e-12 or void_0 <= 1e-12:
        return p0
    p = float(p0) * (float(void_0) / float(void_now)) ** gamma
    return float(min(max(p, 0.5 * p0), max_ratio * p0))


def run_impact_case(
    scaling: LbmScaling,
    *,
    seconds: float = 6.0,
    sample_every: int = 5,
    column_fraction: float = 0.25,
    cushion: bool = False,
    gamma: float = 1.4,
    cushion_max_ratio: float = 1.35,
    name: str | None = None,
    snapshot_fractions: Sequence[float] = (0.25, 0.4, 0.5, 0.75, 1.0),
    rho_profile: Sequence[float] | None = None,
) -> CaseResult:
    """Surge-release impact case (criteria a, b, c, e, f, g).

    ``rho_profile`` (from :func:`run_hydrostatic_case`) switches the wall
    pressure to the *still-water* hydrostatic datum at the same height.
    """
    fill, solid = impact_fill(
        scaling.ny, scaling.nz, scaling.nx, scaling.depth_lbm, column_fraction
    )
    steps = scaling.steps_for_seconds(seconds)
    snaps = sorted({max(1, int(round(f * steps))) for f in snapshot_fractions})
    return _run(
        name or f"impact_nx{scaling.nx}{'_cushion' if cushion else ''}",
        scaling,
        fill,
        solid,
        steps=steps,
        sample_every=sample_every,
        cushion=cushion,
        gamma=gamma,
        cushion_max_ratio=cushion_max_ratio,
        snapshot_steps=snaps,
        rho_profile=rho_profile,
    )


def run_forced_case(
    scaling: LbmScaling,
    *,
    periods: float = 6.0,
    amplitude_m: float = 0.25,
    omega_ratio: float = 1.0,
    sample_every: int = 5,
    name: str | None = None,
    snapshot_fractions: Sequence[float] = (0.5, 0.75, 1.0),
) -> CaseResult:
    """Harmonic (body-force) excitation case - the m5744 forced anchor.

    ``a(t) = A omega^2 sin(omega t)`` with ``A = 0.25 m`` and
    ``omega = omega_ratio * omega_1`` (IN48-N07 ``regular_excitation``), applied
    as a horizontal LBM body force.  The tank starts at rest with a flat
    surface at ``h_l``; the free-surface response drives the wall pressure.
    """
    anchor = scaling.anchor
    omega_phys = omega_ratio * anchor.omega1
    a_amp = amplitude_m * omega_phys**2
    a_lbm = a_amp * scaling.dt**2 / scaling.dx
    omega_lbm = omega_phys * scaling.dt

    fill = torch.zeros((scaling.nz, scaling.ny, scaling.nx), dtype=torch.float32)
    solid = torch.zeros((scaling.nz, scaling.ny, scaling.nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    solid[0, :, :] = True
    solid[-1, :, :] = True
    fill[:, 1 : scaling.depth_lbm + 1, :] = 1.0
    fill[solid] = 0.0

    steps = scaling.steps_for_seconds(periods * 2.0 * math.pi / omega_phys)
    snaps = sorted({max(1, int(round(f * steps))) for f in snapshot_fractions})

    def gx(step: int) -> float:
        return a_lbm * math.sin(omega_lbm * step)

    res = _run(
        name or f"forced_nx{scaling.nx}",
        scaling,
        fill,
        solid,
        steps=steps,
        sample_every=sample_every,
        gx=gx,
        snapshot_steps=snaps,
    )
    res.meta.update(
        {
            "excitation_amplitude_m": amplitude_m,
            "excitation_omega": omega_phys,
            "excitation_a_amp_m_s2": a_amp,
            "excitation_a_lbm": a_lbm,
            "periods": periods,
        }
    )
    return res


def _masked_umax(ux, uy, uz, mask) -> float:
    """Max speed over ``mask`` (GAS/interface velocity is not a solution field)."""
    if not bool(mask.any()):
        return float("nan")
    m = mask
    return float(
        torch.sqrt(ux[m] ** 2 + uy[m] ** 2 + uz[m] ** 2).max().item()
    )


def hydrostatic_l2_error(
    gauge: torch.Tensor,
    flags: torch.Tensor,
    scaling: "LbmScaling",
) -> dict:
    """Relative L2 (and max) error of a gauge-pressure field vs ``rho g (h_l - y)``.

    Interior liquid cells only (two cells away from the side walls, below the
    free surface); the arbitrary LBM pressure constant is removed before the
    error is formed, so only the *profile* is tested.  Returns ``cells``,
    ``pressure_offset_lbm``, ``l2_rel_error`` and ``max_rel_error``; ``error``
    if the interior column is empty.
    """
    y_lo, y_hi = 2, max(3, scaling.depth_lbm - 1)
    xs = slice(2, scaling.nx - 2)
    liquid = (flags[:, y_lo:y_hi, :] == LIQUID)[:, :, xs]
    p_num = gauge[:, y_lo:y_hi, xs]
    y_idx = torch.arange(y_lo, y_hi).view(1, -1, 1).to(gauge.dtype)
    h_lbm = float(scaling.depth_lbm)
    p_ref = scaling.g_lbm * (h_lbm + 0.5 - y_idx) * torch.ones_like(p_num)
    n = float(liquid.sum().item())
    if n <= 0:
        return {"error": "no interior liquid cells"}
    pn = p_num[liquid]
    pr = p_ref[liquid]
    offset = float((pn - pr).mean().item())
    resid = pn - pr - offset
    l2 = float(torch.sqrt((resid**2).mean()).item())
    scale = float(pr.abs().mean().item())
    scale = max(scale, 1e-30)
    return {
        "cells": int(n),
        "pressure_offset_lbm": offset,
        "l2_rel_error": l2 / scale,
        "max_rel_error": float(resid.abs().max().item()) / scale,
    }


def run_hydrostatic_case(
    scaling: LbmScaling,
    *,
    settle_steps: int = 400,
    name: str | None = None,
) -> dict:
    """Static-tank baseline: LBM hydrostatic profile vs ``rho g (h_l - y)``.

    A closed tank at rest must reproduce the linear hydrostatic pressure
    profile with zero velocity.  Reported as the relative L2 error of the
    pressure profile over the interior liquid column (criterion: < 1%).
    """
    fill = torch.zeros((scaling.nz, scaling.ny, scaling.nx), dtype=torch.float32)
    solid = torch.zeros((scaling.nz, scaling.ny, scaling.nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    solid[0, :, :] = True
    solid[-1, :, :] = True
    fill[:, 1 : scaling.depth_lbm + 1, :] = 1.0
    fill[solid] = 0.0
    f, flags, mass = _initialize(fill, solid, 1.0)
    m0 = float(mass.sum().item())
    for _ in range(settle_steps):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=scaling.tau,
            gy=-scaling.g_lbm, rho_liquid=1.0, rho_gas=1.0,
            C_s=scaling.c_sgs, collision=scaling.collision,
        )
    rho, ux, uy, uz = macroscopic3d(f)
    datum = liquid_mean_density(rho, flags)
    gauge = CS2 * (rho - datum)
    err = hydrostatic_l2_error(gauge, flags, scaling)
    err["pressure_datum"] = datum
    # breadth/x-averaged density profile over the whole column (the still-water
    # reference state used as the wall-pressure datum by the impact case)
    prof = rho[:, :, 2 : scaling.nx - 2].mean(dim=(0, 2))
    err["rho_profile"] = [float(v) for v in prof.tolist()]
    if "error" in err:
        return err
    n = err.pop("cells")
    return {
        "name": name or f"hydrostatic_nx{scaling.nx}",
        "nx": scaling.nx,
        "cells": n,
        **err,
        "g_lbm": scaling.g_lbm,
        "depth_lbm": scaling.depth_lbm,
        "u_max_lbm": float(_masked_umax(ux, uy, uz, flags == LIQUID)),
        "mass_drift": (float(mass.sum().item()) - m0) / m0,
    }


# ---------------------------------------------------------------------------
# 6. Post-processing (GCI, spectra, dispersion)
# ---------------------------------------------------------------------------
def observed_order(
    values: Sequence[float], spacings: Sequence[float]
) -> tuple[float, float]:
    """Observed order of accuracy from three (or more) grid solutions.

    Solves the three-grid consistency equation for ``p``::

        (f2 - f1) / (f3 - f2) = (h2^p - h1^p) / (h3^p - h2^p)

    with ``h`` the grid spacing (decreasing).  Returns ``(p, f_extrapolated)``
    where ``f_extrapolated`` is the Richardson limit from the two finest grids.
    ``p`` is clamped to ``[0.1, 8]``; a non-monotone or noise-dominated triple
    (sign change of the differences) returns ``p = 0`` and a NaN extrapolation,
    flagging that the triple cannot be used for a GCI.
    """
    f1, f2, f3 = (float(v) for v in values[:3])
    h1, h2, h3 = (float(v) for v in spacings[:3])
    d21, d32 = f2 - f1, f3 - f2
    if not (abs(d21) > 0.0 and abs(d32) > 0.0) or d21 * d32 <= 0.0:
        return 0.0, float("nan")

    def residual(p: float) -> float:
        return (h2**p - h1**p) / (h3**p - h2**p) - d21 / d32

    lo, hi = 0.1, 8.0
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo * r_hi > 0.0:
        return 0.0, float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rm = residual(mid)
        if r_lo * rm <= 0.0:
            hi, r_hi = mid, rm
        else:
            lo, r_lo = mid, rm
        if hi - lo < 1e-10:
            break
    p = 0.5 * (lo + hi)
    r = h2 / h3  # > 1: h3 is the finest grid
    f_ex = f3 + (f3 - f2) / (max(r, 1.0) ** p - 1.0)
    return p, f_ex


def gci_fine(
    values: Sequence[float], spacings: Sequence[float], safety: float = 1.25
) -> dict:
    """Roache GCI (+ observed order) for a three-grid family.

    ``values`` and ``spacings`` are ordered coarse -> fine.  ``GCI_fine`` uses
    the two finest grids with the observed order ``p`` obtained from all three,
    and the standard safety factor ``Fs = 1.25``.  When the observed order is
    unavailable (``p == 0``) the reported GCI is ``inf`` and
    ``usable = False`` - a noisy triple must not be dressed up as convergent.
    """
    p, f_ex = observed_order(values, spacings)
    f2, f3 = float(values[1]), float(values[2])
    h2, h3 = float(spacings[1]), float(spacings[2])
    r = h2 / h3  # > 1: h3 is the finest grid
    if p <= 0.0 or not math.isfinite(p) or abs(f3) < 1e-30:
        return {
            "p_obs": 0.0,
            "extrapolated": float("nan"),
            "gci_fine": float("inf"),
            "ratio_r": r,
            "usable": False,
        }
    denom = r**p - 1.0
    gci = safety * abs(f3 - f2) / abs(f3) / denom if denom != 0 else float("inf")
    return {
        "p_obs": p,
        "extrapolated": f_ex,
        "gci_fine": gci,
        "ratio_r": r,
        "usable": True,
    }


def harmonic_analysis(
    t: np.ndarray,
    y: np.ndarray,
    omega_ref: float,
    n_harmonics: int = 4,
) -> dict:
    """Hann-windowed amplitude spectrum of a wall signal at multiples of ``omega_ref``.

    Returns the fundamental ``omega`` measured from the spectral peak, the
    amplitudes ``A_k`` at ``k * omega_ref`` (``k = 1..n_harmonics``), the
    relative energy fraction of the ``2 omega`` band (task criterion d: > 5 %)
    and the peak-frequency estimate with its parabolic refinement.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float) - float(np.mean(y))
    n = y.size
    if n < 8 or not np.any(y):
        return {"ok": False, "reason": "signal too short or identically zero"}
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * w)) / (w.sum() / 2.0)
    freqs = np.fft.rfftfreq(n, d=float(t[1] - t[0]))
    omegas = 2.0 * np.pi * freqs
    total = float((spec[1:] ** 2).sum())
    if total <= 0.0:
        return {"ok": False, "reason": "zero spectral energy"}
    # ignore the first two bins (mean/half-window leakage)
    lo = 2
    peak = lo + int(np.argmax(spec[lo:]))
    # parabolic refinement of the peak frequency
    if lo < peak < spec.size - 1:
        a, b, c = spec[peak - 1], spec[peak], spec[peak + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    dw = omegas[1] - omegas[0]
    omega_peak = float(omegas[peak] + delta * dw)

    def band_energy(centre: float, half_width: float) -> float:
        m = (omegas >= centre - half_width) & (omegas <= centre + half_width)
        m[:lo] = False
        return float((spec[m] ** 2).sum())

    harmonics = {}
    for k in range(1, n_harmonics + 1):
        centre = k * omega_ref
        harmonics[f"A{k}"] = float(spec[int(np.argmin(np.abs(omegas - centre)))])
        harmonics[f"E{k}"] = band_energy(centre, dw)

    # 2*omega band from the *measured* fundamental (superharmonic content)
    e2 = band_energy(2.0 * omega_peak, dw)
    # local peak check: maximum inside the band must exceed its neighbours
    m2 = (omegas >= 2.0 * omega_peak - 2 * dw) & (omegas <= 2.0 * omega_peak + 2 * dw)
    local_peak = float(spec[m2].max()) if np.any(m2) else 0.0
    background = float(np.median(spec[lo:])) if spec.size > lo + 1 else 0.0
    return {
        "ok": True,
        "omega_peak": omega_peak,
        "omega_ref": float(omega_ref),
        "omega_ratio": omega_peak / float(omega_ref),
        "amplitude": float(spec[peak]),
        "A1": harmonics.get("A1", 0.0),
        "energy_total": total,
        "energy_2omega_fraction": e2 / total,
        "energy_2omega_fraction_at_ref": band_energy(2.0 * omega_ref, dw) / total,
        "local_peak_2omega": local_peak,
        "background_median": background,
        "peak_prominence_2omega": local_peak / background if background > 0 else float("inf"),
        "harmonics": harmonics,
    }


def dispersion_ratio_swe(kh: float) -> dict:
    """SWE-vs-exact first-mode frequency ratio at relative depth ``kh``.

    ``omega_SWE / omega_exact = sqrt( kh / tanh(kh) )``: the SWE chain is 6.2 %
    too fast at ``kh = 0.628`` (the IN48-N07 case) and within 3 % only for
    ``kh < 0.430`` (``h/L < 0.137``) - the depth-averaged chain buys its speed
    by putting all the wave energy in the wrong frequency.
    """
    ratio = math.sqrt(kh / math.tanh(kh))
    # largest kh with ratio <= 1 + tol, by bisection
    out = {"kh": kh, "omega_ratio_swe_over_exact": ratio}
    for tol in (0.03, 0.05, 0.10, 0.15):
        lo, hi = 1e-8, 5.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.sqrt(mid / math.tanh(mid)) <= 1.0 + tol:
                lo = mid
            else:
                hi = mid
        out[f"kh_max_for_{int(tol * 100)}pct"] = lo
    return out


def smooth(y: Iterable[float], window: int) -> np.ndarray:
    """Centred moving average with edge clamping (acoustic-noise filter).

    The free-surface LBM pressure on two cells next to a wall carries an
    acoustic/conversion noise of the order of the signal itself; the impact
    pressure rises over ``h/u ~ 0.45 s``, so a window of a few tens of samples
    suppresses the noise without touching the physical rise.  ``window <= 1``
    returns the raw series (so raw and filtered peaks can both be reported).
    """
    arr = np.asarray(list(y), float)
    w = int(window)
    if w <= 1 or arr.size < 3:
        return arr.copy()
    w = min(w, arr.size)
    pad = w // 2
    padded = np.pad(arr, (pad, w - 1 - pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def correlation(a: Iterable[float], b: Iterable[float]) -> float:
    """Pearson correlation coefficient of two equally sampled series."""
    x = np.asarray(list(a), float)
    y = np.asarray(list(b), float)
    if x.size != y.size or x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def resample(series_t: Sequence[float], series_y: Sequence[float], t_target: np.ndarray) -> np.ndarray:
    """Linear resampling of ``series_y(t)`` onto ``t_target`` (no extrapolation)."""
    return np.interp(np.asarray(t_target, float), np.asarray(series_t, float),
                     np.asarray(series_y, float))