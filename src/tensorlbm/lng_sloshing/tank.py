"""Two-dimensional free-surface LBM tank driver for LNG cargo sloshing.

The lattice solver is the audited Körner free-surface model already shipped in
:mod:`tensorlbm.free_surface_lbm_2d` (``koerner_step_2d``: independent mass
ledger, anti-bounce-back gas pressure, fluid-side half-way walls, Thürey table
conversion).  This module only adds the *tank* layer on top:

* a rectangular tank geometry with solid bottom + two solid side walls and an
  open (constant-pressure gas) top,
* a mass-consistent liquid surface definition on the *fill* field, so the free
  surface elevation ``eta(x,t)`` is the column liquid inventory (sub-cell
  accurate through the interface fill fraction),
* harmonic sway excitation applied as an inertial body force in the tank frame,
  ``gx(t) = X0 omega^2 sin(omega t)`` (equivalent to prescribed wall motion for
  a rigid tank, and the standard LBM sloshing excitation),
* hydrostatic pre-conditioning of the density field so that the first acoustic
  transient does not contaminate the frequency measurement,
* wall dynamic pressure gauges (``p = c_s^2 * (rho - rho_ref)`` extrapolated to
  the wall from the first two fluid columns) plus the liquid inventory ledger.

Coordinate conventions
----------------------
Field arrays are ``(ny, nx)``.  Solid cells: row ``y = 0`` (bottom), columns
``x = 0`` and ``x = nx-1`` (side walls).  The fluid interior spans
``x = 1 .. length`` and ``y = 1 ..``; the still-water surface sits at
``y = depth`` counted from the tank bottom, so cell ``y = j`` covers the
vertical interval ``[j-1, j]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..d2q9 import equilibrium
from ..free_surface_lbm_2d import (
    INTERFACE,
    LIQUID,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)

__all__ = [
    "TankConfig",
    "TankState",
    "build_tank",
    "column_liquid_height",
    "modal_projection",
    "wall_pressure_profile",
    "simulate_tank",
    "fill_ratio_to_depth",
]

CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class TankConfig:
    """Rectangular tank + free-surface LBM parameters (all in lattice units)."""

    length: int = 80
    depth: float = 40.0
    headspace: float = 0.30
    g: float = 1.25e-3
    tau: float = 0.8
    rho_liquid: float = 1.0
    rho_gas: float = 1.0
    slip_side_walls: bool = False

    @property
    def nx(self) -> int:
        return self.length + 2

    @property
    def ny(self) -> int:
        return int(math.ceil(self.depth + self.headspace * self.length)) + 2

    @property
    def fill_ratio(self) -> float:
        return self.depth / self.length

    def check(self) -> None:
        if self.length < 16:
            raise ValueError("length must be >= 16 lattice cells")
        if not (0.0 < self.depth < 0.8 * self.length + self.headspace * self.length):
            raise ValueError("depth out of range for the configured tank")
        if not (0.0 < self.g < 0.01):
            raise ValueError("g must be small (lattice units) to stay incompressible")
        if 3.0 * self.g * self.depth > 0.6:
            raise ValueError(
                "g*depth too large: 3*g*h = %.3f exceeds the 0.6 weak-"
                "compressibility budget" % (3.0 * self.g * self.depth)
            )
        if not (0.55 <= self.tau <= 3.0):
            raise ValueError("tau outside the free-surface model's stable range")


def fill_ratio_to_depth(fill_ratio: float, length: int) -> float:
    """Abramson's ``h/L`` fill ratio -> still-water depth in lattice cells."""
    if not (0.0 < fill_ratio <= 1.5):
        raise ValueError(f"fill_ratio must be in (0, 1.5], got {fill_ratio}")
    return float(fill_ratio) * float(length)


@dataclass
class TankState:
    """Mutable solver state of one tank."""

    cfg: TankConfig
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
    def liquid_inventory(self) -> float:
        return total_liquid_inventory_2d(
            self.f, self.mass, self.flags, self.cfg.rho_liquid
        )


def _flat_surface(cfg: TankConfig):
    return lambda x: cfg.depth


def _solid_mask(cfg: TankConfig, device) -> torch.Tensor:
    solid = torch.zeros((cfg.ny, cfg.nx), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    return solid


def _initial_fill(cfg: TankConfig, solid: torch.Tensor, surface) -> torch.Tensor:
    """Fill fraction field with a sub-cell accurate liquid surface."""
    fill = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float32, device=solid.device)
    for i in range(1, cfg.nx - 1):
        h = float(surface(i - 0.5))
        if h <= 0.0:
            continue
        lo = 1
        hi = min(cfg.ny - 1, int(math.ceil(h)) + 1)
        idx = torch.arange(lo, hi + 1, device=solid.device, dtype=torch.float32)
        cell_fill = (h - (idx - 1.0)).clamp(0.0, 1.0)
        fill[lo : hi + 1, i] = cell_fill
    fill = torch.clamp(fill, 0.0, 1.0)
    fill[solid] = 0.0
    return fill


def _hydrostatic_density(cfg: TankConfig, solid: torch.Tensor, surface) -> torch.Tensor:
    """Hydrostatic density field ``rho(y) = rho_gas * exp(3 g (y_s - y))``.

    ``rho' = -3 g rho`` is the LBM hydrostatic balance for ``p = c_s^2 rho`` with
    body force ``-g`` in ``y``; the exponential is expanded linearly because
    ``3 g h`` is bounded by the config check.
    """
    rho = torch.full((cfg.ny, cfg.nx), cfg.rho_gas, dtype=torch.float32, device=solid.device)
    for i in range(1, cfg.nx - 1):
        h = float(surface(i - 0.5))
        hi = min(cfg.ny - 1, int(math.ceil(h)))
        if hi < 1:
            continue
        idx = torch.arange(1, hi + 1, device=solid.device, dtype=torch.float32)
        yc = idx - 0.5                     # cell-centre height above the bottom
        rho[1 : hi + 1, i] = cfg.rho_gas * (1.0 + 3.0 * cfg.g * (h - yc)).clamp(min=0.2)
    return rho


def build_tank(
    cfg: TankConfig,
    surface=None,
    device: str | torch.device = "cpu",
) -> TankState:
    """Create a resting tank; ``surface(x)`` gives the initial free-surface height."""
    cfg.check()
    device = torch.device(device)
    if surface is None:
        surface = _flat_surface(cfg)
    solid = _solid_mask(cfg, device)
    fill = _initial_fill(cfg, solid, surface)
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags, cfg.rho_liquid)
    rho = _hydrostatic_density(cfg, solid, surface)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)
    active = ((flags == LIQUID) | (flags == INTERFACE)).unsqueeze(0)
    f = torch.where(active, f, torch.zeros_like(f))
    state = TankState(cfg=cfg, f=f, mass=mass, flags=flags, solid=solid, rho_ref=rho.clone())
    state.inventory0 = state.liquid_inventory
    return state


def liquid_content(state: TankState) -> torch.Tensor:
    """Per-cell liquid content (fill) from the conserved mass ledger.

    LIQUID cells hold exactly one cell of liquid; INTERFACE cells hold
    ``mass/rho_liquid`` (kept *unclamped above 1* because a temporarily
    overfilled interface cell must not lose its excess in the measurement).
    """
    cfg = state.cfg
    content = torch.zeros_like(state.mass)
    content = torch.where(state.flags == LIQUID, torch.ones_like(content), content)
    content = torch.where(
        state.flags == INTERFACE,
        (state.mass / cfg.rho_liquid).clamp(min=0.0),
        content,
    )
    return content


def column_liquid_height(state: TankState) -> torch.Tensor:
    """Per-column liquid thickness (cells) from the conserved mass ledger."""
    return liquid_content(state).sum(dim=0)


def x_centroid(state: TankState) -> float:
    """Liquid x-centroid (lattice cells, cell centres at ``i - 0.5``).

    An integral (mass-weighted) measure: it oscillates at the sloshing
    frequency with far better signal-to-noise than a single surface column
    because it is insensitive to the interface conversion jitter.
    """
    cfg = state.cfg
    w = liquid_content(state)
    xs = torch.arange(cfg.nx, dtype=w.dtype, device=w.device) - 0.5
    tot = w.sum().clamp(min=1e-12)
    return float((w.sum(dim=0) * xs).sum() / tot)


def surface_elevation(state: TankState) -> torch.Tensor:
    """``eta(x) = thickness(x) - h`` on the interior columns."""
    h = column_liquid_height(state)
    eta = h - state.cfg.depth
    return eta[1:-1]


def modal_projection(eta, mode: int, trim: int = 0) -> float:
    """Least-squares cosine-mode amplitude of a free-surface elevation profile.

    ``A = sum eta cos(k x) / sum cos^2(k x)`` over the interior columns
    (``x_i = i - 0.5``), optionally trimming ``trim`` columns at each wall.  The
    trimming matters: the free-surface model keeps a thin liquid film in the
    cell adjacent to each wall (documented artefact), and ``cos(k x)`` is largest
    there, so an untrimmed projection is dominated by that film.
    """
    n = eta.shape[0]
    lo, hi = int(trim), n - int(trim)
    if hi - lo < 4:
        raise ValueError("trim leaves too few columns")
    sub = eta[lo:hi]
    idx = torch.arange(lo + 1, hi + 1, dtype=eta.dtype, device=eta.device) - 0.5
    k = mode * math.pi / float(n)
    basis = torch.cos(k * idx)
    return float((sub * basis).sum() / (basis * basis).sum())


def wall_pressure_profile(state: TankState) -> torch.Tensor:
    """Dynamic wall pressure (left wall) per vertical cell, extrapolated to the wall.

    ``p(y) = c_s^2 (rho - rho_ref)`` at the first two fluid columns, linearly
    extrapolated to ``x = 0`` (second-order, wall value).  Cells that are not
    fully liquid on both columns are returned as NaN so they can be masked.
    """
    rho = state.rho
    dp = CS2 * (rho - state.rho_ref)
    p1 = dp[:, 1]
    p2 = dp[:, 2]
    ok = ((state.flags[:, 1] == LIQUID) | (state.flags[:, 1] == INTERFACE)) & (
        (state.flags[:, 2] == LIQUID) | (state.flags[:, 2] == INTERFACE)
    )
    p_wall = 1.5 * p1 - 0.5 * p2
    return torch.where(ok, p_wall, torch.full_like(p_wall, float("nan")))


def _advance(
    state: TankState,
    n_steps: int,
    *,
    gx_of_t=None,
    record_every: int = 5,
    gauge_y=None,
    n_modes: int = 3,
    on_record=None,
) -> dict:
    """Internal marching loop; collects modal elevation, gauges and the ledger."""
    cfg = state.cfg
    if gauge_y is None:
        gauge_y = [
            int(round(cfg.depth + d))
            for d in (0.0, -0.1 * cfg.depth, -0.3 * cfg.depth, -0.6 * cfg.depth)
        ]
    gauge_y = [min(max(int(y), 1), cfg.ny - 2) for y in gauge_y]

    times: list[float] = []
    eta_hist: list[torch.Tensor] = []
    press_hist: list[torch.Tensor] = []
    inv_hist: list[float] = []
    umax_hist: list[float] = []
    xcm_hist: list[float] = []
    content_hist: list[float] = []

    def record(t: int) -> None:
        eta = surface_elevation(state)
        eta_hist.append(eta.clone())
        prof = wall_pressure_profile(state)
        press_hist.append(prof[gauge_y].clone())
        inv_hist.append(state.liquid_inventory)
        rho = state.rho
        act = (state.flags == LIQUID) | (state.flags == INTERFACE)
        ux = torch.zeros_like(rho)
        # horizontal momentum per cell / rho, only where active
        cx = torch.tensor([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=rho.dtype).view(9, 1, 1)
        mom = (state.f * cx).sum(dim=0)
        ux = torch.where(act, mom / rho.clamp(min=1e-9), torch.zeros_like(rho))
        umax_hist.append(float(ux.abs().max()))
        xcm_hist.append(x_centroid(state))
        content_hist.append(float(liquid_content(state).sum()))
        times.append(float(t))

    record(0)
    for n in range(1, n_steps + 1):
        gx = 0.0 if gx_of_t is None else float(gx_of_t(n))
        state.f, state.mass, state.flags = koerner_step_2d(
            state.f,
            state.mass,
            state.flags,
            state.solid,
            tau=cfg.tau,
            gx=gx,
            gy=-cfg.g,
            rho_liquid=cfg.rho_liquid,
            rho_gas=cfg.rho_gas,
        )
        state.step += 1
        if n % record_every == 0:
            record(n)
            if on_record is not None:
                on_record(state, float(n))

    eta_arr = (
        torch.stack(eta_hist) if eta_hist else torch.zeros((0, cfg.length))
    ).to(torch.float64)
    press_arr = (
        torch.stack(press_hist) if press_hist else torch.zeros((0, len(gauge_y)))
    ).to(torch.float64)
    n = cfg.length
    idx = torch.arange(1, n + 1, dtype=torch.float64) - 0.5
    modes = {}
    for m in range(1, n_modes + 1):
        k = m * math.pi / float(n)
        modes[m] = (2.0 / n) * (eta_arr * torch.cos(k * idx)).sum(dim=1)
    return {
        "t": torch.tensor(times, dtype=torch.float64),
        "eta": eta_arr,
        "eta_modes": modes,
        "pressure_gauges": press_arr,
        "gauge_y": gauge_y,
        "inventory": torch.tensor(inv_hist, dtype=torch.float64),
        "x_centroid": torch.tensor(xcm_hist, dtype=torch.float64),
        "content_total": torch.tensor(content_hist, dtype=torch.float64),
        "ux_max": torch.tensor(umax_hist, dtype=torch.float64),
        "inventory0": state.inventory0,
    }


def simulate_tank(
    cfg: TankConfig,
    n_steps: int,
    *,
    sway_amplitude: float = 0.0,
    omega: float = 0.0,
    tilt_amplitude: float = 0.0,
    record_every: int = 4,
    gauge_y=None,
    n_modes: int = 3,
    device: str | torch.device = "cpu",
    on_record=None,
) -> dict:
    """Run one sloshing case.

    Either a free-decay run (``tilt_amplitude > 0``: initial elevation
    ``eta = A cos(k_1 x)``, zero velocity) or a forced harmonic-sway run
    (``sway_amplitude X0 > 0`` at angular frequency ``omega``), or both.
    """
    def surface(x: float) -> float:
        if tilt_amplitude <= 0.0:
            return cfg.depth
        k = math.pi / cfg.length
        return cfg.depth + tilt_amplitude * math.cos(k * x)

    state = build_tank(cfg, surface=surface if tilt_amplitude > 0 else None, device=device)

    gx_of_t = None
    if sway_amplitude > 0.0 and omega > 0.0:
        gx_of_t = lambda t: sway_amplitude * omega * omega * math.sin(omega * t)  # noqa: E731

    out = _advance(
        state,
        n_steps,
        gx_of_t=gx_of_t,
        record_every=record_every,
        gauge_y=gauge_y,
        n_modes=n_modes,
        on_record=on_record,
    )
    out["cfg"] = cfg
    out["sway_amplitude"] = sway_amplitude
    out["omega"] = omega
    out["tilt_amplitude"] = tilt_amplitude
    out["dt"] = 1.0
    out["mass_drift"] = float(
        (out["inventory"][-1] - out["inventory0"]) / out["inventory0"]
    )
    return out