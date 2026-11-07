"""2-D numerical wave tank: relaxation-zone wavemaker + sponge on free-surface LBM.

Layout (``x`` = streamwise, ``y`` = vertical, one solid row at ``y = 0``)::

    wall |<-- wavemaker zone -->|<---- propagation ---->|<-- sponge -->| wall
    x=0  0 ................. Lw   Lw .............. Lw+Lp      ...     Lx-1

The wavemaker is *not* a moving boundary: inside the relaxation zone the
solution is blended toward the analytical Stokes second-order field with the
classical relaxation function of waves2Foam (Jacobsen et al. 2012),

.. math:: w(\\chi) = 1 - \\exp(-\\chi^{3.5}),

where ``chi`` runs from 0 at the interface with the propagation region to 1 at
the wall.  The sponge layer uses the same weight with the still-water state as
target, which absorbs the outgoing wave.  Relaxation is applied to *both* the
free-surface position (the Körner mass ledger, hence the liquid volume of the
column) and the velocity field (populations blended toward the local
equilibrium of the analytical kinematics), so the generated wave leaves the
zone with the prescribed surface elevation, kinematics and zero net volume flux.

Because the relaxation zone is a source/sink of liquid, the mass injected by it
is accumulated and reported (``mass_injected``) so the total-inventory audit of
the tank stays interpretable: outside the zones the Körner ledger is exactly
conservative.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
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
from . import goda
from .periodic import _analytic_fields
from .stokes2 import Stokes2Wave


def _jsonable(obj):
    """Recursively make a result dict JSON safe (complex phasors -> [re, im]).

    The reflection analysis returns complex phasors on purpose (they carry the
    phase information); the run summary is serialised with this adapter so a
    finished 15-minute run can never be lost at the last statement.
    """
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, (np.complex128, np.complex64)):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


__all__ = ["TankConfig", "TankGeometry", "build_geometry", "build_initial_state",
           "relax_zones", "run_tank", "relaxation_weight"]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class TankConfig:
    """Numerical wave tank parameters (all lengths in lattice units)."""

    h: float = 20.0                     # still-water depth
    kh: float = 1.5                     # relative depth (k = kh/h)
    eps: float = 0.02                   # wave steepness k*a
    c_target: float = 0.04              # target phase speed -> lattice gravity
    tau: float = 0.5025
    n_wavemaker: float = 1.0            # wavemaker relaxation length [wavelengths]
    n_propagation: float = 2.0          # propagation length [wavelengths]
    n_sponge: float = 1.5               # sponge length [wavelengths]
    headroom: int = 12                  # gas cells above still-water level
    rho_gas: float = 1.0
    second_order: bool = True
    sponge: bool = True                 # False -> negative control (solid end wall)
    sponge_strength: float = 1.0        # multiplier on the relaxation weight
    wavemaker_strength: float = 1.0
    ramp_periods: float = 1.0           # soft start of the wavemaker
    #: coupling mode of the relaxation zones.  ``"mass"`` also resets the Körner
    #: mass ledger / cell flags to the analytical surface (measured to grow the
    #: interface layer and the liquid ledger in this solver, see the campaign
    #: report); ``"velocity"`` only relaxes the populations toward the analytical
    #: kinematics and leaves the free-surface bookkeeping untouched.
    coupling: str = "mass"
    #: probe distances from the reflecting end wall, in wavelengths; they must
    #: sit seaward of the sponge (``> n_sponge``) and inside the propagation
    #: stretch, with 0.25 wavelength spacing for an optimally conditioned
    #: two-point separation (``|sin k ds| = 1``).
    probe_offsets: tuple = (1.75, 2.00, 2.25, 2.50)
    record_every: int = 5
    n_periods: float = 4.0
    dtype: str = "float32"

    def wave_and_g(self) -> tuple:
        wave, g = _wave_from_params(self.h, self.kh, self.eps, self.c_target)
        return wave, g

    def lengths(self) -> dict:
        wave, _ = self.wave_and_g()
        lam = wave.wavelength
        lw = max(int(round(self.n_wavemaker * lam)), 4)
        lp = max(int(round(self.n_propagation * lam)), 8)
        ls = max(int(round(self.n_sponge * lam)), 8)
        nx = lw + lp + ls + 1
        ny = 1 + int(math.ceil(self.h + self.headroom))
        return {"lw": lw, "lp": lp, "ls": ls, "nx": nx, "ny": ny, "lambda": lam}


def _wave_from_params(h: float, kh: float, eps: float, c_target: float):
    """Build a :class:`Stokes2Wave` from ``(h, kh, eps)`` and pick the lattice gravity."""
    k = kh / h
    sig = math.tanh(kh)
    g = c_target ** 2 * k / sig
    a = eps / k
    return Stokes2Wave(a=a, k=k, h=h, g=g), g


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@dataclass
class TankGeometry:
    nx: int
    ny: int
    lw: int
    lp: int
    ls: int
    x_wavemaker: slice
    x_sponge: slice
    probe_x: list = field(default_factory=list)
    wall_x: int = 0

    def probe_distances(self) -> np.ndarray:
        """Distance of each probe from the reflecting end wall."""
        return np.array([float(self.wall_x - x) for x in self.probe_x])


def relaxation_weight(chi: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """waves2Foam relaxation function ``1 - exp(-chi^3.5)`` scaled by ``strength``."""
    chi = np.clip(np.asarray(chi, dtype=float), 0.0, 1.0)
    return np.clip(strength * (1.0 - np.exp(-chi ** 3.5)), 0.0, 1.0)


def build_geometry(cfg: TankConfig) -> TankGeometry:
    L = cfg.lengths()
    lw, lp, ls, nx, ny = L["lw"], L["lp"], L["ls"], L["nx"], L["ny"]
    lo_x, hi_x = lw + 3, nx - 1 - ls - 3
    probe_x = sorted({int(round(nx - 1 - off * L["lambda"])) for off in cfg.probe_offsets})
    probe_x = [min(max(x, lo_x), hi_x) for x in probe_x]
    if len(set(probe_x)) != len(cfg.probe_offsets):
        raise ValueError(
            "probe offsets collapse to the same column: propagation region "
            f"[{lo_x}, {hi_x}] is too short for offsets {cfg.probe_offsets}")
    return TankGeometry(
        nx=nx, ny=ny, lw=lw, lp=lp, ls=ls,
        x_wavemaker=slice(0, lw),
        x_sponge=slice(nx - ls, nx),
        probe_x=probe_x,
        wall_x=nx - 1,
    )


# ---------------------------------------------------------------------------
# state / relaxation
# ---------------------------------------------------------------------------
def _fill_from_state(flags: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    fill = torch.zeros_like(mass)
    fill[flags == LIQUID] = 1.0
    fill[flags == INTERFACE] = mass[flags == INTERFACE]
    return fill


def _target_fill(h_target, y: torch.Tensor, second_order=True) -> torch.Tensor:
    """Fill field whose column sum equals ``h_target`` (same convention as the rig)."""
    return torch.clamp(h_target - y + 1.0, 0.0, 1.0)


def build_initial_state(cfg: TankConfig, wave: Stokes2Wave, geom: TankGeometry,
                        g: float):
    """Still water everywhere; the wavemaker zone ramps the wave in from rest."""
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    ny, nx = geom.ny, geom.nx
    y = torch.arange(ny, dtype=dtype).view(ny, 1)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    fill = torch.clamp(torch.tensor(cfg.h, dtype=dtype) - y + 1.0, 0.0, 1.0)
    fill = fill.expand(ny, nx).clone()
    fill[solid] = 0.0
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags)
    active = flags != GAS
    rho = torch.where(active, torch.ones_like(fill), torch.full_like(fill, cfg.rho_gas))
    zero = torch.zeros_like(fill)
    f = equilibrium(rho, zero, zero)
    return f, mass, flags, solid


def relax_zones(f, mass, flags, solid, wave: Stokes2Wave, g: float, step: int,
                cfg: TankConfig, geom: TankGeometry,
                weight_wm: torch.Tensor, weight_sp: torch.Tensor,
                ramp: float) -> tuple:
    """Blend the solution toward the analytical state inside both relaxation zones.

    ``weight_wm`` / ``weight_sp`` are ``(1, nx)`` weight fields; only their zone
    slices are used.  The returned ``injected`` value is the liquid mass that the
    relaxation added to the domain on this step (audit quantity).
    """
    ny = geom.ny
    dtype = mass.dtype
    y = torch.arange(ny, dtype=dtype).view(ny, 1)
    fill = _fill_from_state(flags, mass)

    eta_a, u_a, w_a = _analytic_fields(wave, torch.arange(geom.nx, dtype=dtype).view(1, -1),
                                       y, float(step), cfg.second_order)
    # target surface: wavemaker -> analytical wave (ramped in), sponge -> still water
    eta_wm = ramp * eta_a
    h_wm = torch.tensor(cfg.h, dtype=dtype) + eta_wm
    h_sp = torch.full_like(eta_wm, float(cfg.h))

    inj = 0.0
    for sl, weight, h_target, vel_scale in (
        (geom.x_wavemaker, weight_wm, h_wm, ramp),
        (geom.x_sponge, weight_sp, h_sp, 1.0),
    ):
        w = weight[:, sl]
        if float(w.max()) == 0.0:
            continue
        if cfg.coupling == "velocity":
            x0v = (sl.start or 0)
            x1v = sl.stop if sl.stop is not None else geom.nx
            f = f.clone()
            u_t = vel_scale * u_a[:, sl]
            w_t = vel_scale * w_a[:, sl]
            active = (flags[:, x0v:x1v] != GAS).to(dtype)
            feq = equilibrium(torch.ones((ny, x1v - x0v), dtype=dtype),
                              torch.where(active > 0, u_t, torch.zeros_like(u_t)),
                              torch.where(active > 0, w_t, torch.zeros_like(w_t)))
            f[:, :, x0v:x1v] = ((1.0 - w).unsqueeze(0) * f[:, :, x0v:x1v]
                                + w.unsqueeze(0) * feq)
            continue
        x0 = (sl.start or 0)
        x1 = sl.stop if sl.stop is not None else geom.nx
        ft = _target_fill(h_target[:, sl], y, cfg.second_order)
        fill_t = (1.0 - w) * fill[:, sl] + w * ft
        m_before = float(mass[:, x0:x1].sum().item())
        # flags are rebuilt from the blended fill inside the zone (+/- one column
        # so that the mandatory liquid/gas interface envelope stays attached)
        lo = max(x0 - 1, 0)
        hi = min(x1 + 1, geom.nx)
        fill_glob = fill.clone()
        fill_glob[:, lo:hi] = fill[:, lo:hi]
        fill_glob[:, x0:x1] = fill_t
        flags_glob = init_flags_from_fill_2d(fill_glob, solid)
        mass_glob = init_mass_from_fill_2d(fill_glob, flags_glob)
        flags[:, lo:hi] = flags_glob[:, lo:hi]
        mass[:, x0:x1] = mass_glob[:, x0:x1]
        f = f.clone()
        # velocities: blend populations toward the analytical local equilibrium
        u_t = vel_scale * u_a[:, sl]
        w_t = vel_scale * w_a[:, sl]
        active = (flags[:, x0:x1] != GAS).to(dtype)
        feq = equilibrium(torch.ones((ny, x1 - x0), dtype=dtype),
                          torch.where(active > 0, u_t, torch.zeros_like(u_t)),
                          torch.where(active > 0, w_t, torch.zeros_like(w_t)))
        f[:, :, x0:x1] = (1.0 - w).unsqueeze(0) * f[:, :, x0:x1] + w.unsqueeze(0) * feq
        inj += float(mass[:, x0:x1].sum().item()) - m_before
    return f, mass, flags, inj


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_tank(cfg: TankConfig, verbose: bool = True) -> dict:
    """Integrate the wave tank and return probe records plus reflection metrics."""
    torch.set_default_dtype(torch.float32 if cfg.dtype == "float32" else torch.float64)
    torch.set_num_threads(int(os.environ.get("TL_WT_THREADS", "1")))
    wave, g = cfg.wave_and_g()
    geom = build_geometry(cfg)
    f, mass, flags, solid = build_initial_state(cfg, wave, geom, g)
    omega, period = wave.omega, wave.period

    # relaxation weights (chi = 0 at the inner edge, 1 at the wall)
    xw = np.arange(geom.nx, dtype=float)
    chi_wm = np.clip(1.0 - xw / max(geom.lw, 1), 0.0, 1.0)
    chi_sp = np.clip((xw - (geom.nx - geom.ls)) / max(geom.ls, 1), 0.0, 1.0)
    w_wm = np.zeros(geom.nx)
    w_sp = np.zeros(geom.nx)
    w_wm[: geom.lw] = relaxation_weight(chi_wm[: geom.lw], cfg.wavemaker_strength)
    if cfg.sponge:
        w_sp[geom.nx - geom.ls:] = relaxation_weight(
            chi_sp[geom.nx - geom.ls:], cfg.sponge_strength)
    weight_wm = torch.tensor(w_wm, dtype=mass.dtype).view(1, -1)
    weight_sp = torch.tensor(w_sp, dtype=mass.dtype).view(1, -1)

    n_steps = int(cfg.n_periods * period)
    ramp_steps = max(int(cfg.ramp_periods * period), 1)
    m0 = total_liquid_inventory_2d(f, mass, flags)
    injected = 0.0
    probe_series: dict = {int(x): [] for x in geom.probe_x}
    times: list = []
    snapshots: list = []
    t_start = time.time()
    for step in range(1, n_steps + 1):
        f, mass, flags = koerner_step_2d(f, mass, flags, solid, tau=cfg.tau, gy=-g,
                                         rho_gas=cfg.rho_gas)
        ramp = min(1.0, step / ramp_steps)
        f, mass, flags, inj = relax_zones(f, mass, flags, solid, wave, g, step, cfg,
                                          geom, weight_wm, weight_sp, ramp)
        injected += inj
        if step % cfg.record_every == 0:
            times.append(step)
            fill = _fill_from_state(flags, mass).sum(dim=0)
            for x in geom.probe_x:
                probe_series[int(x)].append(float(fill[x].item()) - cfg.h)
            if step % (cfg.record_every * 40) == 0:
                snapshots.append((step, (fill - cfg.h).numpy().copy()))
        if verbose and step % 1000 == 0:
            print(f"    step {step}/{n_steps} t/T={step / period:.2f} "
                  f"inj={injected:.3e} ({time.time() - t_start:.0f}s)", flush=True)

    t = np.array(times, dtype=float) - 1.0
    series = {x: np.array(v) for x, v in probe_series.items()}
    m_final = total_liquid_inventory_2d(f, mass, flags)

    # ---- reflection analysis over the last two periods ---------------------
    mask = t > t[-1] - 2.0 * period
    tm = t[mask]
    x_probes = np.array(geom.probe_x, dtype=float)
    s_probes = (geom.nx - 1) - x_probes          # distance from the far end wall
    z = np.array([goda.phasor_from_series(tm, series[int(x)][mask], omega)
                  for x in geom.probe_x])
    two_point = {}
    for i in range(len(x_probes) - 1):
        try:
            res = goda.two_point_reflection(z[i], z[i + 1], wave.k, s_probes[i], s_probes[i + 1])
        except ValueError as exc:
            res = {"error": str(exc)}
        two_point[f"{int(x_probes[i])}_{int(x_probes[i + 1])}"] = res
    ls_fit = goda.least_squares_reflection(z, s_probes, wave.k)
    incident_amp = abs(ls_fit["I"])
    # incident wave travelling toward the far wall: +x propagation, reflected -x
    return _jsonable({
        "config": {"h": cfg.h, "kh": cfg.kh, "eps": cfg.eps, "tau": cfg.tau,
                   "sponge": cfg.sponge, "sponge_strength": cfg.sponge_strength,
                   "n_periods": cfg.n_periods, "n_steps": n_steps,
                   "runtime_s": time.time() - t_start},
        "geometry": {"nx": geom.nx, "ny": geom.ny, "lw": geom.lw, "lp": geom.lp,
                     "ls": geom.ls, "probe_x": geom.probe_x,
                     "probe_s": s_probes.tolist(), "wall_x": geom.wall_x},
        "wave": wave.summary(),
        "g": g,
        "reflection": {
            "K_R_least_squares": ls_fit["K_R"],
            "phase_least_squares": ls_fit["phase"],
            "fit_rms_residual": ls_fit["rms_residual"],
            "incident_amplitude": incident_amp,
            "incident_amplitude_error": abs(incident_amp - wave.a) / wave.a,
            "two_point": two_point,
        },
        "mass": {
            "relative_drift": float((m_final - m0) / m0),
            "injected_volume_cells": float(injected),
            "injected_per_period": float(injected / max(t[-1] / period, 1e-9)),
        },
        "probes": {str(x): series[x].tolist() for x in geom.probe_x},
        "t": t.tolist(),
        "snapshots": {str(int(s)): a.tolist() for s, a in snapshots[-3:]},
    })