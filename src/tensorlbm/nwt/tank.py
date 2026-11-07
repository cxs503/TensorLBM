"""Two-dimensional numerical wave tank (NWT) on the Körner free-surface LBM.

Layout of the flume (one row of cells per ``y`` index, ``y`` increasing upward,
``y = 0`` is the bed)::

    x = 0                                                             x = nx-1
    |wall| <--- generation zone (>= 1 lambda) ---> <--- probes ---> <-- absorber --> |wall|

* The **generation zone** is a relaxation zone (Mayer/Jacobsen): the analytic
  second-order Stokes solution of :mod:`tensorlbm.nwt.stokes2` is blended into
  the numerical state with a cosine weight that is 1 on the wall side and 0 at
  the end of the zone.  Both the liquid column inventory (free-surface
  elevation) and the populations (velocity) are relaxed.
* The **absorber** at the leeward end is the same relaxation machinery with the
  still-water target (``eta = 0``, ``u = 0``); ``sponge_enabled=False`` gives the
  negative control required to prove the absorber is doing the work.
* **Probes** record ``eta(x_j, t) = V(x_j, t) - h`` from the conserved liquid
  inventory of the free-surface model — no geometric reconstruction and no
  threshold parameter.

The free-surface physics is the existing Körner model
(:func:`tensorlbm.free_surface_lbm_2d.koerner_step_2d`), unchanged; this module
only drives it and records diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import torch

from ..free_surface_lbm_2d import (
    GAS,
    SOLID,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
)
from ..d2q9 import equilibrium
from .stokes2 import Stokes2Wave
from .wavemaker import (
    column_free_surface,
    damp_velocity,
    ramp_factor,
    relaxation_weight,
    relax_column_inventory,
    relax_velocity_to_target,
)

__all__ = ["NWTConfig", "NWTResult", "build_geometry", "run_nwt"]

_C4 = 1.0 / 3.0  # lattice sound speed squared (D2Q9)


@dataclass(frozen=True)
class NWTConfig:
    """Geometry, wave and discretisation parameters of one NWT run.

    The lattice time step is 1.  Given ``wavelength`` and ``steps_per_period``
    the angular frequency follows, and gravity is *derived* from the linear
    dispersion relation, ``g = omega^2 / (k tanh(k h))``, so the pair
    ``(omega, k)`` is exactly on the closed-form dispersion curve.  The
    resulting hydrostatic compressibility ratio ``g h / c_s^2`` is reported in
    :meth:`diagnostics` because it is the main LBM-specific error driver.
    """

    wavelength: float = 69.0
    depth: float = 33.0
    steepness: float = 0.02
    steps_per_period: int = 1200
    n_periods: float = 6.0
    ramp_periods: float = 2.0
    analysis_periods: float = 2.0
    generator_wavelengths: float = 1.0
    sponge_wavelengths: float = 2.5
    sponge_enabled: bool = True
    sponge_strength: float = 1.0
    #: absorber construction: "relax" (Dirichlet-blend of eta and u),
    #: "damping" (viscous momentum sink) or "relax+damping" (both)
    sponge_mode: str = "damping"
    sponge_damping: float = 0.15
    probe_spacing_wavelengths: float = 0.15
    n_probes: int = 6
    probe_start_wavelengths: float = 0.5
    tau: float = 0.505
    rho_gas: float = 1.0
    vertical_margin: float = 0.35
    free_slip: bool = True
    record_every: int = 1
    n_snapshots: int = 8
    generator_enabled: bool = True
    initial_condition: str = "still"  # "still" | "stokes"
    device: str = "cpu"

    # -- derived -----------------------------------------------------------
    @property
    def wavenumber(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.steps_per_period

    @property
    def gravity(self) -> float:
        k = self.wavenumber
        return self.omega**2 / (k * math.tanh(k * self.depth))

    @property
    def wave(self) -> Stokes2Wave:
        return Stokes2Wave.from_steepness(self.steepness, self.wavenumber, self.depth, self.gravity)

    @property
    def total_steps(self) -> int:
        return int(round(self.n_periods * self.steps_per_period))

    def diagnostics(self) -> dict:
        w = self.wave
        return {
            "wavelength": self.wavelength,
            "depth": self.depth,
            "steepness": self.steepness,
            "kh": w.kh,
            "omega": self.omega,
            "period": float(self.steps_per_period),
            "gravity": self.gravity,
            "viscosity": (self.tau - 0.5) / 3.0,
            "hydrostatic_compressibility_gh_cs2": self.gravity * self.depth / _C4,
            "max_surface_velocity_ma": w.horizontal_velocity_amplitude_at_swl(),
            "second_harmonic_ratio": w.skewness_bound,
            "total_steps": self.total_steps,
            "tau": self.tau,
        }

    def variant(self, **kwargs) -> "NWTConfig":
        return replace(self, **kwargs)


@dataclass
class NWTResult:
    config: NWTConfig
    geometry: dict
    probe_x: list[float]
    eta_snapshots: torch.Tensor  # (n_snapshots, nx) full-surface snapshots
    snapshot_steps: list[int]
    eta_probes: torch.Tensor  # (n_probes, n_samples)
    inventory: torch.Tensor  # (n_samples_full,)
    time: torch.Tensor  # (n_samples,)
    wave: Stokes2Wave
    wall_seconds: float
    steps: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def analysis_start_step(self) -> int:
        return self.geometry["analysis_start_step"]

    def to_dict(self) -> dict:
        return {
            "config": {
                k: getattr(self.config, k) for k in self.config.__dataclass_fields__
            },
            "diagnostics": self.diagnostics,
            "geometry": self.geometry,
            "probe_x": self.probe_x,
            "wave": self.wave.as_dict(),
            "wall_seconds": self.wall_seconds,
            "steps": self.steps,
        }


def build_geometry(cfg: NWTConfig) -> dict:
    """Resolve the flume layout in integer cells (all positions are cell centres)."""
    lam = cfg.wavelength
    gen_len = int(round(cfg.generator_wavelengths * lam))
    probe_start = int(round(cfg.probe_start_wavelengths * lam))
    probe_dx = cfg.probe_spacing_wavelengths * lam
    probe_x = [1 + gen_len + probe_start + int(round(j * probe_dx)) for j in range(cfg.n_probes)]
    probe_end = probe_x[-1]
    gap = int(round(0.4 * lam))
    sponge_len = int(round(cfg.sponge_wavelengths * lam)) if cfg.sponge_enabled else 0
    nx = probe_end + gap + sponge_len + 2
    ny = 1 + int(math.ceil(cfg.depth)) + int(math.ceil(cfg.depth * cfg.vertical_margin)) + 1
    # the analysis window is an exact integer number of wave periods (no spectral
    # leakage in the harmonic fit) and sits at the end of the run
    analysis_start_step = max(
        0, cfg.total_steps - int(round(cfg.analysis_periods * cfg.steps_per_period))
    )
    return {
        "nx": nx,
        "ny": ny,
        "x_generator_end": 1 + gen_len,
        "x_sponge_start": probe_end + gap,
        "x_sponge_end": nx - 2,
        "sponge_len": sponge_len,
        "generator_len": gen_len,
        "analysis_start_step": analysis_start_step,
        "n_analysis_samples": cfg.total_steps - analysis_start_step,
    }


def _initial_fill_and_flags(cfg: NWTConfig, geom: dict, device: torch.device):
    """Build the initial fill / flags / mass / populations of the flume.

    Two start-up states are supported:

    * ``"still"`` (default, used for the acceptance runs): flat free surface plus
      the hydrostatic density profile.  Every source of nonlinear content in the
      records then comes from the solver and the wavemaker, not from the IC.
    * ``"stokes"`` (diagnostic): the full second-order Stokes IC (elevation,
      orbital velocity and hydrostatic pressure) at ``t = 0``.  Combined with
      ``generator_enabled=False`` this isolates numerical damping and dispersion
      of the propagation region from any wavemaker artefact.
    """

    ny, nx = geom["ny"], geom["nx"]
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[0, :] = True  # bed
    solid[:, 0] = True  # offshore wall
    solid[:, -1] = True  # leeward wall
    yy = torch.arange(ny, device=device, dtype=torch.float32).view(ny, 1)
    # Cell y = 1 sits immediately above the bed wall face (y = 0.5), so a column
    # whose fills sum to ``depth`` has its free surface at y = depth + 0.5.
    # ``fill(y) = clamp(depth - (y - 1), 0, 1)`` gives sum_y fill = depth exactly,
    # i.e. eta = 0 at start-up (no spurious 1-cell level jump into the tank).
    wave = cfg.wave
    if cfg.initial_condition == "stokes":
        xx = torch.arange(nx, device=device, dtype=torch.float32).view(1, nx)
        eta0 = wave.elevation(xx, 0.0).view(1, nx)
    else:
        eta0 = torch.zeros((1, nx), device=device)
    fill = (cfg.depth + eta0 - (yy - 1.0)).clamp(0.0, 1.0).expand(ny, nx).contiguous()
    fill = torch.where(solid, torch.zeros_like(fill), fill)
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags, rho_liquid=1.0)
    # Hydrostatic start-up: with gravity applied as a body force, a uniform-density
    # column is *not* in equilibrium (it would launch a compression transient of
    # relative size g*h/c_s^2).  Initialize rho(y) = rho_surf + g*z/c_s^2 below the
    # still-water level, with rho_surf = rho_gas (the interface ABB pressure).
    z_below_swl = (cfg.depth + 0.5 - yy).clamp(min=0.0)
    rho_h = (cfg.rho_gas + cfg.gravity * z_below_swl / _C4).expand(ny, nx).contiguous()
    if cfg.initial_condition == "stokes":
        xx = torch.arange(nx, device=device, dtype=torch.float32).view(1, nx)
        u0, w0 = wave.kinematics(xx.expand(ny, nx), yy.expand(ny, nx), 0.0)
    else:
        u0 = torch.zeros_like(fill)
        w0 = torch.zeros_like(fill)
    f = equilibrium(rho_h, u0, w0)
    gas_or_solid = (flags == GAS) | (flags == SOLID)
    f = torch.where(gas_or_solid.unsqueeze(0), torch.zeros_like(f), f)
    return f, mass, flags, solid


def run_nwt(cfg: NWTConfig | None = None, **overrides) -> NWTResult:
    """Run one NWT case and return the recorded probe / budget diagnostics."""
    cfg = (cfg or NWTConfig()).variant(**overrides) if overrides else (cfg or NWTConfig())
    device = torch.device(cfg.device)
    geom = build_geometry(cfg)
    ny, nx = geom["ny"], geom["nx"]
    wave_cfg = cfg.wave
    g = cfg.gravity
    k = cfg.wavenumber

    f, mass, flags, solid = _initial_fill_and_flags(cfg, geom, device)

    xx = torch.arange(nx, device=device, dtype=torch.float32).view(1, nx)
    yy = torch.arange(ny, device=device, dtype=torch.float32).view(ny, 1)

    # --- relaxation weights ------------------------------------------------
    gen_start, gen_end = 1.0, float(geom["x_generator_end"])
    gen_w = relaxation_weight(xx, gen_start, gen_end).expand(ny, nx).contiguous()
    if cfg.sponge_enabled:
        sp_start, sp_end = float(geom["x_sponge_start"]), float(geom["x_sponge_end"])
        sp_w = cfg.sponge_strength * relaxation_weight(xx, sp_end, sp_start)
        sp_w = sp_w.expand(ny, nx).contiguous()
    else:
        sp_w = torch.zeros((ny, nx), device=device)

    # generator column mask (for the inventory target we only need columns)
    gen_col_w = gen_w[0, :]
    sp_col_w = sp_w[0, :]

    total_steps = cfg.total_steps
    n_ana = max(1, geom["n_analysis_samples"])
    probe_x = [float(x) for x in _probe_positions(cfg, geom)]

    eta_rec = torch.zeros((len(probe_x), n_ana), dtype=torch.float64)
    snap_every = max(1, n_ana // max(1, cfg.n_snapshots))
    eta_snap = torch.zeros((cfg.n_snapshots, nx), dtype=torch.float64)
    snap_steps: list[int] = []
    n_snap = 0
    inv_all = torch.zeros(total_steps, dtype=torch.float64)
    rec_idx = 0
    t_start = time.perf_counter()

    for step in range(total_steps):
        t = float(step)
        amp = ramp_factor(t, cfg.ramp_periods * cfg.steps_per_period)
        wave = Stokes2Wave.from_steepness(cfg.steepness * amp, k, cfg.depth, g) if amp > 0 else None

        V = column_free_surface(f, mass, flags, 1.0)

        if wave is not None and cfg.generator_enabled and float(gen_col_w.max()) > 0:
            eta_t = wave.elevation(xx, t).view(nx)
            target_v = cfg.depth + eta_t
            mass = relax_column_inventory(
                mass,
                flags,
                target_v,
                gen_col_w,
                current_inventory=V,
                rho_liquid=1.0,
                f=f,
            )
            u_t, w_t = wave.kinematics(xx.expand(ny, nx), yy.expand(ny, nx), t)
            f = relax_velocity_to_target(f, flags, u_t, w_t, gen_w, rho_liquid=1.0)

        if cfg.sponge_enabled:
            if "relax" in cfg.sponge_mode:
                target_v = torch.full_like(V, cfg.depth)
                mass = relax_column_inventory(
                    mass,
                    flags,
                    target_v,
                    sp_col_w,
                    current_inventory=V,
                    rho_liquid=1.0,
                    f=f,
                )
                zeros = torch.zeros_like(f[0])
                f = relax_velocity_to_target(f, flags, zeros, zeros, sp_w, rho_liquid=1.0)
            if "damping" in cfg.sponge_mode:
                f = damp_velocity(f, flags, sp_w, cfg.sponge_damping)

        f, mass, flags = koerner_step_2d(
            f,
            mass,
            flags,
            solid,
            tau=cfg.tau,
            gx=0.0,
            gy=-g,
            rho_liquid=1.0,
            rho_gas=cfg.rho_gas,
            free_slip_all=cfg.free_slip,
        )

        V = column_free_surface(f, mass, flags, 1.0)
        inv_all[step] = float(V.sum().item())
        if step >= geom["analysis_start_step"]:
            if len(probe_x):
                eta_rec[:, rec_idx] = (V[probe_x] - cfg.depth).to(torch.float64)
            rec_idx += 1
            if n_snap < cfg.n_snapshots and (rec_idx - 1) % snap_every == 0:
                eta_snap[n_snap] = (V - cfg.depth).to(torch.float64)
                snap_steps.append(step)
                n_snap += 1

    wall = time.perf_counter() - t_start
    n_rec = min(rec_idx, n_ana)
    eta_rec = eta_rec[:, :n_rec]
    eta_snap = eta_snap[:n_snap]
    time_axis = torch.arange(n_rec, dtype=torch.float64) + geom["analysis_start_step"]

    return NWTResult(
        config=cfg,
        geometry=geom,
        probe_x=probe_x,
        eta_snapshots=eta_snap,
        snapshot_steps=snap_steps,
        eta_probes=eta_rec,
        inventory=inv_all,
        time=time_axis,
        wave=cfg.wave,
        wall_seconds=wall,
        steps=total_steps,
        diagnostics={
            **cfg.diagnostics(),
            "nx": nx,
            "ny": ny,
            "cells": nx * ny,
            "cells_evaluated": nx * ny * total_steps,
            "analysis_start_step": geom["analysis_start_step"],
            "n_analysis_samples": n_rec,
            "probe_x": probe_x,
            "n_snapshots": n_snap,
        },
    )


def _probe_positions(cfg: NWTConfig, geom: dict) -> list[int]:
    lam = cfg.wavelength
    gen_len = int(round(cfg.generator_wavelengths * lam))
    probe_start = int(round(cfg.probe_start_wavelengths * lam))
    probe_dx = cfg.probe_spacing_wavelengths * lam
    return [1 + gen_len + probe_start + int(round(j * probe_dx)) for j in range(cfg.n_probes)]