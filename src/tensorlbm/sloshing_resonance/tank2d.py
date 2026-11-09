"""Free-surface LBM driver for a two-dimensional rectangular sloshing tank.

The lattice solver is the audited Körner free-surface model
(:func:`tensorlbm.free_surface_lbm_2d.koerner_step_2d`, independent mass ledger,
anti-bounce-back gas pressure, fluid-side half-way walls).  This module adds only
the *tank + scaling + measurement* layer needed by the P87W2-N01 verification
node:

* geometry ``L x H = 1.0 m x 0.5 m`` (``x`` = sloshing direction, ``y`` up from
  the tank bottom) with solid bottom / side walls and a constant-pressure gas
  head-space on top,
* a **lattice scaling** that ties ``dx`` and ``dt`` through a fixed lattice
  velocity (``u_lat = u_char dt / dx``), so grid refinement preserves the Froude
  number and the Mach number — the standard requirement for a GCI study,
* the *free-decay* initial condition of theory eq. (2):
  ``eta(x,0) = A cos(k_n x)`` with the fluid at rest, i.e. the tank is released
  from a pure standing-wave deformation (no forcing),
* a hydrostatic density pre-conditioning (``p = c_s^2 rho`` and
  ``d p / d y = -rho g``) so the first acoustic transient does not contaminate
  the frequency measurement,
* measurements: sub-cell free-surface elevation per column from the Körner mass
  ledger, modal amplitudes ``a_n(t)``, wall dynamic-pressure gauges, and the
  liquid inventory ledger (mass-conservation gate).

Coordinate conventions
----------------------
Field arrays are ``(ny, nx)``.  Row ``y = 0`` is the solid bottom, rows
``y = 1 .. ny-2`` are fluid/gas cells, row ``ny-1`` is a solid lid; columns
``x = 0`` and ``x = nx-1`` are solid side walls.  The still-water depth is
``h`` cells measured from row 1, so cell row ``j`` covers the vertical interval
``[j-1, j]`` cells above the bottom.

Damping caveat (honest, measured — see the campaign report)
-----------------------------------------------------------
The Körner model's numerical viscosity is orders of magnitude above water's, so
a free-decay tank run is always *over-damped* in lattice units.  This module
therefore reports both the measured (damped) frequency ``f_d`` and the
undamped-frequency estimate ``f_n = f_d / sqrt(1 - zeta^2)`` obtained from the
standard linear damped-oscillator relation, with the fitted damping ratio
``zeta``.  Which of the two is used is a *criterion choice* recorded in the
campaign JSON; both are always reported so nothing is hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from ..d2q9 import equilibrium
from ..free_surface_lbm_2d import (
    GAS,
    INTERFACE,
    LIQUID,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)

__all__ = [
    "CS2",
    "SloshingTank",
    "LatticeScaling",
    "TankField",
    "build_tank_field",
    "column_liquid_height",
    "surface_elevation",
    "modal_amplitude",
    "wall_pressure_profile",
    "run_free_decay",
    "run_sway",
]

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# geometry + scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SloshingTank:
    """Rectangular tank (SI units) and its fluid."""

    length: float = 1.0
    height: float = 0.5
    g: float = 9.81
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0

    def check(self) -> None:
        if self.length <= 0.0 or self.height <= 0.0:
            raise ValueError("tank dimensions must be positive")
        if self.g <= 0.0:
            raise ValueError("g must be positive")


@dataclass(frozen=True)
class LatticeScaling:
    """``dx``/``dt`` pair fixing the lattice velocity (Froude + Mach consistent).

    ``u_char`` is the characteristic *physical* sloshing velocity used to define
    the lattice Mach number (``u_lat = u_char dt / dx``).  Refining ``dx`` at
    fixed ``u_lat`` refines space and time together, which is what the grid
    convergence study of criterion (e) requires.
    """

    dx: float
    dt: float
    g_lattice: float
    tau: float
    u_lat: float
    u_char: float

    @classmethod
    def from_resolution(
        cls,
        tank: SloshingTank,
        cells_per_height: int,
        *,
        u_lat: float = 0.002,
        u_char: float = 0.074,
        tau: float = 0.502,
    ) -> "LatticeScaling":
        if cells_per_height < 8:
            raise ValueError("cells_per_height must be >= 8")
        if not (0.0 < u_lat < 0.1):
            raise ValueError("u_lat must be small (lattice Mach number)")
        if u_char <= 0.0:
            raise ValueError("u_char must be positive")
        dx = tank.height / float(cells_per_height)
        dt = u_lat * dx / u_char
        g_lat = tank.g * dt * dt / dx
        return cls(dx=dx, dt=dt, g_lattice=g_lat, tau=tau, u_lat=u_lat, u_char=u_char)

    def check(self, depth_cells: float) -> None:
        if not (0.5 < self.tau <= 3.0):
            raise ValueError("tau must be in (0.5, 3] for the BGK free-surface model")
        ratio = 3.0 * self.g_lattice * depth_cells
        if ratio > 0.25:
            raise ValueError(
                f"weak-compressibility ratio 3 g h / c_s^2 = {ratio:.3f} too large; reduce u_lat"
            )

    @property
    def nu_lattice(self) -> float:
        return (self.tau - 0.5) / 3.0

    def nu_physical(self, tank: SloshingTank) -> float:
        """Physical kinematic viscosity corresponding to ``tau`` (m^2/s)."""
        return self.nu_lattice * self.dx * self.dx / self.dt

    def pressure_scale(self, tank: SloshingTank) -> float:
        """Lattice -> Pa factor for ``p = c_s^2 rho`` with ``rho`` in lattice units."""
        return tank.rho_liquid * self.dx * self.dx / (self.dt * self.dt)

    def time_scale(self) -> float:
        return self.dt


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


@dataclass
class TankField:
    """Mutable solver state of one 2D tank."""

    tank: SloshingTank
    scaling: LatticeScaling
    depth_cells: float
    mode: int
    amplitude: float
    f: torch.Tensor
    mass: torch.Tensor
    flags: torch.Tensor
    solid: torch.Tensor
    rho_ref: torch.Tensor
    step: int = 0
    inventory0: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def rho(self) -> torch.Tensor:
        return self.f.sum(dim=0)

    @property
    def nx(self) -> int:
        return int(self.flags.shape[1])

    @property
    def ny(self) -> int:
        return int(self.flags.shape[0])

    @property
    def inventory(self) -> float:
        return total_liquid_inventory_2d(self.f, self.mass, self.flags, 1.0)


def _solid_mask(nx: int, ny: int, device) -> torch.Tensor:
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    return solid


def build_tank_field(
    tank: SloshingTank,
    scaling: LatticeScaling,
    *,
    depth: float,
    mode: int = 1,
    amplitude: float = 0.01,
    headroom: float = 0.25,
    surface: Callable[[float], float] | None = None,
    device: str | torch.device = "cpu",
) -> TankField:
    """Build the free-surface state for ``eta(x,0) = A cos(k_n x)`` (SI inputs).

    ``depth`` and ``amplitude`` are in metres, ``headroom`` is the gas head-space
    height in metres.  A custom ``surface(x_m) -> elevation_m`` overrides the
    single-mode initial condition (used by the second-mode and negative-control
    cases).
    """
    tank.check()
    dx = scaling.dx
    nx_int = int(round(tank.length / dx))
    if nx_int < 8:
        raise ValueError("too few columns")
    h_cells = depth / dx
    if h_cells < 3.0:
        raise ValueError("still-water depth must be >= 3 lattice cells")
    head_cells = max(int(math.ceil(headroom / dx)), 3)
    nx = nx_int + 2
    ny = 1 + int(math.ceil(h_cells)) + head_cells + 1
    scaling.check(h_cells)

    solid = _solid_mask(nx, ny, device)
    xs = (torch.arange(1, nx - 1, dtype=torch.float64) - 0.5) * dx  # column centres (m)
    k = mode * math.pi / tank.length
    if surface is None:
        eta0 = amplitude * torch.cos(k * xs)
    else:
        eta0 = torch.tensor([surface(float(x)) for x in xs], dtype=torch.float64)

    # column liquid height in cells, measured from row 1 (bottom of the fluid)
    height = h_cells + eta0 / dx
    rows = torch.arange(1, ny - 1, dtype=torch.float64).view(-1, 1) - 1.0  # cells above bottom
    fill = (height.view(1, -1) - rows).clamp(0.0, 1.0)
    full = torch.zeros((ny, nx), dtype=torch.float32)
    full[1:-1, 1:-1] = fill.to(torch.float32)
    flags = init_flags_from_fill_2d(full, solid)
    mass = init_mass_from_fill_2d(full, flags)

    # Hydrostatic pre-conditioning.  ``rho_ref`` is the *still-water* profile
    # (the gauge reference); the initial field uses the *local* column height, so
    # a released deformation is in exact hydrostatic balance and the dynamic
    # gauge reads ``rho g eta_wall`` at t = 0.  With p = c_s^2 rho and
    # dp/dy = -rho g, the lattice profile is rho = 1 + (1/c_s^2) g (H_col - y)
    # (1/c_s^2 = 3).
    y_cells = (torch.arange(1, ny - 1, dtype=torch.float32).view(-1, 1) - 1.0).clamp(min=0.0)
    h_col = height.to(torch.float32).view(1, -1)  # local column height (cells)
    rho_ref = torch.zeros((ny, nx), dtype=torch.float32)
    rho_ref[1:-1, 1:-1] = (1.0 + 3.0 * scaling.g_lattice * (h_cells - y_cells)).clamp(
        min=tank.rho_gas, max=1.5
    )
    rho_ref[solid] = 0.0

    rho0 = torch.zeros((ny, nx), dtype=torch.float32)
    rho0[1:-1, 1:-1] = (1.0 + 3.0 * scaling.g_lattice * (h_col - y_cells)).clamp(
        min=tank.rho_gas, max=1.5
    )
    rho0[solid] = 0.0
    rho0[flags == GAS] = 0.0
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = torch.where((flags == GAS).unsqueeze(0), torch.zeros_like(f), f)

    state = TankField(
        tank=tank,
        scaling=scaling,
        depth_cells=h_cells,
        mode=mode,
        amplitude=amplitude,
        f=f,
        mass=mass,
        flags=flags,
        solid=solid,
        rho_ref=rho_ref,
    )
    state.inventory0 = state.inventory
    return state


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _fluid_slice(state: TankField):
    return slice(1, state.ny - 1), slice(1, state.nx - 1)


def column_liquid_height(state: TankField) -> torch.Tensor:
    """Liquid height per interior column in lattice cells (mass-ledger based).

    The Körner ledger stores mass in *lattice* density units (``rho_l = 1``), so
    the column sum is already a height in cells and must not be divided by the
    physical density.
    """
    ys, xs = _fluid_slice(state)
    m = state.mass[ys, xs].to(torch.float64)
    return m.sum(dim=0)


def surface_elevation(state: TankField) -> torch.Tensor:
    """``eta(x)`` in metres, sub-cell accurate, zero-mean referenced to ``h``."""
    return (column_liquid_height(state) - state.depth_cells) * state.scaling.dx


def modal_amplitude(state: TankField, mode: int, *, trim: int = 2) -> float:
    """Modal coefficient ``a_m`` of ``eta(x) = sum_m a_m cos(k_m x)`` in metres.

    ``trim`` interior columns are dropped at each wall: the Körner model keeps a
    thin liquid film against the wall (documented artefact) and ``cos(k x)`` is
    extremal there, so an untrimmed projection is contaminated.
    """
    eta = surface_elevation(state)
    n = eta.shape[0]
    lo, hi = int(trim), n - int(trim)
    if hi - lo < 4:
        raise ValueError("trim leaves too few columns")
    idx = torch.arange(lo, hi, dtype=torch.float64) + 0.5  # column centre in cells
    k = mode * math.pi / state.tank.length
    basis = torch.cos(k * idx * state.scaling.dx)
    return float((eta[lo:hi] * basis).sum() / (basis * basis).sum())


def wall_pressure_profile(state: TankField) -> torch.Tensor:
    """Dynamic wall pressure (Pa) at the left wall for every row.

    ``p_dyn = c_s^2 (rho - rho_ref)`` on the first two fluid columns, linearly
    extrapolated to the wall face (``1.5 p_1 - 0.5 p_2``); rows that are not
    liquid/interface are NaN so they can be masked.  ``rho_ref`` is the
    still-water hydrostatic profile, so the result is the pressure in excess of
    the rest state (what theory eq. (3) predicts).
    """
    rho = state.rho
    dp = CS2 * (rho - state.rho_ref) * state.scaling.pressure_scale(state.tank)
    p1, p2 = dp[:, 1], dp[:, 2]
    ok = ((state.flags[:, 1] == LIQUID) | (state.flags[:, 1] == INTERFACE)) & (
        (state.flags[:, 2] == LIQUID) | (state.flags[:, 2] == INTERFACE)
    )
    p_wall = 1.5 * p1 - 0.5 * p2
    return torch.where(ok, p_wall, torch.full_like(p_wall, float("nan")))


# ---------------------------------------------------------------------------
# marching
# ---------------------------------------------------------------------------


def _record(state: TankField, modes: Sequence[int], gauge_rows: Sequence[int], out: dict) -> None:
    eta = surface_elevation(state)
    out["t"].append(state.step)
    out["eta"].append(eta.clone())
    out["modal"].append([modal_amplitude(state, m) for m in modes])
    prof = wall_pressure_profile(state)
    out["pressure"].append(prof[list(gauge_rows)].clone())
    inv = state.inventory
    out["inventory"].append(inv)
    rho = state.rho
    act = (state.flags == LIQUID) | (state.flags == INTERFACE)
    cx = torch.tensor([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=rho.dtype).view(9, 1, 1)
    mom = (state.f * cx).sum(dim=0)
    ux = torch.where(act, mom / rho.clamp(min=1e-9), torch.zeros_like(rho))
    out["umax"].append(float(ux.abs().max()))


def _finish(state: TankField, out: dict, modes: Sequence[int], gauge_rows: Sequence[int]) -> dict:
    dt = state.scaling.dt
    t = torch.tensor(out["t"], dtype=torch.float64) * dt
    eta = torch.stack(out["eta"]).to(torch.float64) if out["eta"] else torch.zeros((0, 0))
    modal = torch.tensor(out["modal"], dtype=torch.float64).view(len(out["t"]), len(modes))
    press = (
        torch.stack(out["pressure"]).to(torch.float64)
        if out["pressure"]
        else torch.zeros((0, len(gauge_rows)))
    )
    inv = torch.tensor(out["inventory"], dtype=torch.float64)
    res = {
        "t": t,
        "t_lattice": torch.tensor(out["t"], dtype=torch.float64),
        "eta": eta,
        "eta_modes": {m: modal[:, i] for i, m in enumerate(modes)},
        "pressure": press,
        "gauge_rows": list(gauge_rows),
        "inventory": inv,
        "inventory0": state.inventory0,
        "umax": torch.tensor(out["umax"], dtype=torch.float64),
        "dt": dt,
        "dx": state.scaling.dx,
        "depth_cells": state.depth_cells,
        "mode": state.mode,
        "amplitude": state.amplitude,
        "mass_drift": float((inv[-1] - state.inventory0) / state.inventory0),
        "max_mass_drift": float((inv - state.inventory0).abs().max() / state.inventory0),
        "tau": state.scaling.tau,
        "g_lattice": state.scaling.g_lattice,
        "nu_physical": state.scaling.nu_physical(state.tank),
        "u_lat": state.scaling.u_lat,
    }
    return res


def run_free_decay(
    state: TankField,
    n_periods: float,
    *,
    record_every: int = 4,
    modes: Sequence[int] = (1, 2, 3),
    gauge_rows: Sequence[int] | None = None,
    g_scale: float = 1.0,
    slip_walls: bool = False,
) -> dict:
    """March the free-decay case for ``n_periods`` of the *mode-n* linear period.

    ``g_scale`` multiplies the lattice gravity (``0`` gives the negative control
    of criterion (g): no restoring force, hence no oscillation).
    """
    from .theory import abramson_period

    T = abramson_period(
        state.mode, state.depth_cells * state.scaling.dx, state.tank.length, state.tank.g
    )
    n_steps = max(int(round(n_periods * T / state.scaling.dt)), 4)
    if gauge_rows is None:
        hc = int(round(state.depth_cells))
        gauge_rows = [max(1, hc - int(0.5 * hc)), max(1, int(round(0.5 * hc)))]
    gauge_rows = [min(max(int(r), 1), state.ny - 2) for r in gauge_rows]

    out = {"t": [], "eta": [], "modal": [], "pressure": [], "inventory": [], "umax": []}
    _record(state, modes, gauge_rows, out)
    gy = -state.scaling.g_lattice * float(g_scale)
    slip = {
        "free_slip_mask": state.solid if slip_walls else None,
        "free_slip_all": False,
    }
    for n in range(1, n_steps + 1):
        kwargs = dict(
            tau=state.scaling.tau,
            gx=0.0,
            gy=gy,
            rho_liquid=1.0,
            rho_gas=1.0,
        )
        if slip_walls:
            kwargs.update(slip)
        state.f, state.mass, state.flags = koerner_step_2d(
            state.f, state.mass, state.flags, state.solid, **kwargs
        )
        state.step += 1
        if n % record_every == 0 or n == n_steps:
            _record(state, modes, gauge_rows, out)
    res = _finish(state, out, modes, gauge_rows)
    res["n_steps"] = n_steps
    res["period_theory"] = T
    return res


def run_sway(
    state: TankField,
    n_periods: float,
    *,
    sway_amplitude: float,
    omega: float,
    record_every: int = 4,
    modes: Sequence[int] = (1, 2, 3),
    gauge_rows: Sequence[int] | None = None,
    ramp_periods: float = 1.0,
    slip_walls: bool = False,
) -> dict:
    """Harmonic sway excitation ``gx(t) = X0 omega^2 sin(omega t)`` (tank frame).

    ``sway_amplitude`` is in metres, ``omega`` in rad/s (physical).  A linear
    ramp over ``ramp_periods`` avoids the impulsive start transient.
    """
    from .theory import abramson_period

    T = abramson_period(
        state.mode, state.depth_cells * state.scaling.dx, state.tank.length, state.tank.g
    )
    n_steps = max(int(round(n_periods * T / state.scaling.dt)), 4)
    n_ramp = max(int(round(ramp_periods * T / state.scaling.dt)), 1)
    if gauge_rows is None:
        hc = int(round(state.depth_cells))
        gauge_rows = [max(1, hc - int(0.5 * hc)), max(1, int(round(0.5 * hc)))]
    gauge_rows = [min(max(int(r), 1), state.ny - 2) for r in gauge_rows]

    out = {"t": [], "eta": [], "modal": [], "pressure": [], "inventory": [], "umax": []}
    _record(state, modes, gauge_rows, out)
    amp_lat = sway_amplitude / state.scaling.dx  # lattice displacement amplitude
    w_lat = omega * state.scaling.dt  # radians per lattice step
    for n in range(1, n_steps + 1):
        ramp = min(1.0, n / float(n_ramp))
        gx = ramp * amp_lat * w_lat * w_lat * math.sin(w_lat * n)
        kwargs = dict(
            tau=state.scaling.tau,
            gx=gx,
            gy=-state.scaling.g_lattice,
            rho_liquid=1.0,
            rho_gas=1.0,
        )
        if slip_walls:
            kwargs.update({"free_slip_mask": state.solid, "free_slip_all": False})
        state.f, state.mass, state.flags = koerner_step_2d(
            state.f, state.mass, state.flags, state.solid, **kwargs
        )
        state.step += 1
        if n % record_every == 0 or n == n_steps:
            _record(state, modes, gauge_rows, out)
    res = _finish(state, out, modes, gauge_rows)
    res["n_steps"] = n_steps
    res["period_theory"] = T
    res["sway_amplitude"] = sway_amplitude
    res["omega"] = omega
    return res
