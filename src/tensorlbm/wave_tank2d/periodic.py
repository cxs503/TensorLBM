"""Periodic-domain Stokes-wave propagation: the free-surface LBM verification rig.

The periodic rig is the cleanest possible numerical experiment for the *wave*
part of a wave tank: a horizontally periodic free-surface LBM box is initialised
with the exact Stokes second-order field (surface elevation + kinematics), so
the only thing being measured is how well the free-surface lattice-Boltzmann
model transports a gravity wave.  No wavemaker or absorber can contaminate the
phase, amplitude or mass diagnostics.

Diagnostics produced by :func:`run_periodic`:

* ``c_meas`` — phase speed from the slope of the fitted phasor phase over a row
  of columns, compared with the closed form ``c^2 = (g/k) tanh(k h)``;
* ``a1_meas`` / ``a2_meas`` — first and bound-second harmonic amplitudes at a
  probe, compared with ``a`` and ``(k a^2/4)(3-sigma^2)/sigma^3``;
* liquid-mass drift of the Körner ledger normalised per wave period.

All lengths are lattice units; ``g`` is the lattice gravity acceleration
(``gy = -g``), chosen from a target phase speed so every case runs at the same
Mach number and the cost stays bounded.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

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
from .stokes2 import Stokes2Wave

__all__ = ["PeriodicWaveConfig", "surface_elevation", "run_periodic"]


@dataclass
class PeriodicWaveConfig:
    """Discretisation of a periodic Stokes-wave box (all lengths in lattice units)."""

    h: float = 20.0                 # still-water depth
    kh: float = 1.5                 # relative depth -> k = kh / h
    eps: float = 0.02               # steepness k*a
    c_target: float = 0.04          # target phase speed (sets the lattice gravity)
    tau: float = 0.5025             # BGK relaxation time
    n_wavelengths: int = 3          # wavelengths resolved in the (periodic) box
    headroom: int = 12              # gas cells above the still-water level
    second_order: bool = True       # drive with Stokes 2nd order (else linear)
    rho_gas: float = 1.0
    record_every: int = 3
    n_periods: float = 2.0
    hydrostatic_init: bool = False   # start from the hydrostatic density profile
    dtype: str = "float32"

    # -- derived ------------------------------------------------------------
    @property
    def nx(self) -> int:
        return int(round(self.n_wavelengths * 2.0 * math.pi * self.h / self.kh))

    @property
    def ny(self) -> int:
        return 1 + int(math.ceil(self.h + self.headroom))

    def wave(self) -> tuple[Stokes2Wave, float]:
        """Return ``(wave, g)`` with the wave number snapped to the discrete box."""
        if self.h <= 0.0:
            raise ValueError(f"water depth must be positive, got h={self.h}")
        if self.kh <= 0.0:
            raise ValueError(f"relative depth must be positive, got kh={self.kh}")
        if self.eps <= 0.0:
            raise ValueError(f"steepness must be positive, got eps={self.eps}")
        if self.c_target <= 0.0:
            raise ValueError(f"target phase speed must be positive, got {self.c_target}")
        nx = self.nx
        k = 2.0 * math.pi * self.n_wavelengths / nx
        sig = math.tanh(k * self.h)
        g = self.c_target ** 2 * k / sig
        a = self.eps / k
        return Stokes2Wave(a=a, k=k, h=self.h, g=g), g


def surface_elevation(flags: torch.Tensor, mass: torch.Tensor, h: float) -> torch.Tensor:
    """Column liquid height above the still-water level.

    ``sum_y fill(y, x)`` is the liquid column height, where a LIQUID cell counts
    one full cell and an INTERFACE cell contributes its tracked fill level
    (``mass``).  Subtracting the still-water depth gives ``eta(x)`` with the
    same convention as the analytical surface (vertical cell centres at
    ``y = 0, 1, ...`` above the bed).
    """
    filled = (flags == LIQUID).to(mass.dtype) + torch.where(
        flags == INTERFACE, mass, torch.zeros_like(mass)
    )
    return filled.sum(dim=0) - float(h)


def _initial_state(cfg: PeriodicWaveConfig, wave: Stokes2Wave, g: float):
    nx, ny = cfg.nx, cfg.ny
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    y = torch.arange(ny, dtype=dtype).view(ny, 1)
    x = torch.arange(nx, dtype=dtype).view(1, nx)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True

    eta, u, w = _analytic_fields(wave, x, y, 0.0, cfg.second_order)
    # ``wave.eta`` is relative to the still-water level; add the depth here
    fill = torch.clamp(eta + wave.h - y + 1.0, 0.0, 1.0)
    fill[solid] = 0.0
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags)
    active = flags != GAS
    rho = torch.where(active, torch.ones_like(fill), torch.full_like(fill, cfg.rho_gas))
    if cfg.hydrostatic_init:
        # p = cs^2 rho  =>  d(rho)/dy = -g/cs^2 = -3g below the free surface
        rho = torch.where(active,
                          (1.0 + 3.0 * g * (wave.h - y)).clamp(min=cfg.rho_gas),
                          rho)
    zero = torch.zeros_like(fill)
    f = equilibrium(rho, torch.where(active, u, zero), torch.where(active, w, zero))
    return f, mass, flags, solid


def _analytic_fields(wave: Stokes2Wave, x, y, t: float, second_order: bool):
    """Analytic ``eta, u, w`` sampled on the lattice (``y`` = height above bed).

    ``x``/``y`` may be torch tensors (lattice coordinate views); the analytic
    theory is evaluated in numpy and the result is returned on the same dtype
    and device as ``x`` so it can drive the torch solver directly.
    """
    ref = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    xn = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=float)
    yn = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y, dtype=float)
    eta = wave.eta(xn, t, second_order)
    u, w = wave.kinematics(xn, yn - wave.h, t, second_order)
    out = tuple(torch.as_tensor(v, dtype=ref.dtype, device=ref.device) for v in (eta, u, w))
    return out


def run_periodic(cfg: PeriodicWaveConfig, verbose: bool = True) -> dict:
    """Integrate a periodic Stokes wave and return the verification metrics."""
    torch.set_default_dtype(torch.float32 if cfg.dtype == "float32" else torch.float64)
    torch.set_num_threads(int(os.environ.get("TL_WT_THREADS", "1")))
    wave, g = cfg.wave()
    f, mass, flags, solid = _initial_state(cfg, wave, g)
    omega, period = wave.omega, wave.period
    # Discrete measure calibration: the lattice column sum equals the liquid
    # height above the bed row, so a constant half-cell offset vs the analytic
    # still-water level is removed here (it is a discretisation constant, not a
    # physical mean set-up).
    eta_offset = float(surface_elevation(flags, mass, cfg.h).mean().item())
    n_steps = int(cfg.n_periods * period)
    m0 = total_liquid_inventory_2d(f, mass, flags)

    times: list[float] = []
    records: list[np.ndarray] = []
    mass_samples: list[tuple[float, float]] = []
    t_start = time.time()
    for step in range(1, n_steps + 1):
        f, mass, flags = koerner_step_2d(f, mass, flags, solid, tau=cfg.tau, gy=-g,
                                         rho_gas=cfg.rho_gas)
        if step % cfg.record_every == 0:
            times.append(step)
            records.append(
                (surface_elevation(flags, mass, cfg.h) - eta_offset).numpy().copy())
        if step % max(int(period / 4.0), 1) == 0:
            mass_samples.append((step / period,
                                 (total_liquid_inventory_2d(f, mass, flags) - m0) / m0))
        if verbose and step % 1000 == 0:
            print(f"    step {step}/{n_steps} t/T={step / period:.2f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    S = np.array(records)            # (nt, nx) surface elevation
    true_t = np.array(times, dtype=float) - 1.0
    t = true_t
    m_drift = (total_liquid_inventory_2d(f, mass, flags) - m0) / m0

    # ---- analysis over the last period (steady, transient-free window) -----
    mask = t > t[-1] - period
    tm = t[mask]
    Sm = S[mask]
    x = np.arange(cfg.nx, dtype=float)

    phasors = np.array([goda.phasor_from_series(tm, Sm[:, j], omega) for j in range(cfg.nx)])
    k_fit = goda.wave_number_from_phasors(phasors, x)
    k_meas = abs(k_fit["k"])
    c_meas = omega / k_meas

    harm = goda.fit_harmonics(tm, Sm[:, 0], omega, n_harm=2)
    a1_meas = harm["amplitudes"][0]
    a2_meas = harm["amplitudes"][1]
    amp_profile = np.abs(phasors)

    drift_per_period = max(
        (abs(v) for _, v in mass_samples), default=abs(m_drift))

    return {
        "config": {
            "h": cfg.h, "kh": cfg.kh, "eps": cfg.eps, "tau": cfg.tau,
            "c_target": cfg.c_target, "nx": cfg.nx, "ny": cfg.ny,
            "n_wavelengths": cfg.n_wavelengths, "second_order": cfg.second_order,
            "n_steps": n_steps, "n_periods": cfg.n_periods,
            "hydrostatic_init": cfg.hydrostatic_init,
            "runtime_s": time.time() - t_start,
        },
        "wave": wave.summary(),
        "g": g,
        "eta_offset": eta_offset,
        "dispersion": {
            "k_closed_form": wave.k,
            "k_measured": k_meas,
            "k_rel_error": abs(k_meas - wave.k) / wave.k,
            "c_closed_form": wave.c,
            "c_measured": c_meas,
            "c_rel_error": abs(c_meas - wave.c) / wave.c,
            "phase_scatter": k_fit["phase_scatter"],
        },
        "harmonics": {
            "a1_analytic": wave.a,
            "a1_measured": a1_meas,
            "a1_rel_error": abs(a1_meas - wave.a) / wave.a,
            "a2_analytic": wave.a2,
            "a2_measured": a2_meas,
            "a2_rel_error": (abs(a2_meas - wave.a2) / wave.a2) if wave.a2 > 0 else float("nan"),
            "fit_rms_residual": harm["rms_residual"],
            "amplitude_decay_over_box": float(
                (amp_profile[: cfg.nx // 4].mean() - amp_profile[-cfg.nx // 4:].mean())
                / amp_profile.mean()) if cfg.nx >= 8 else 0.0,
        },
        "mass": {
            "relative_drift_total": float(m_drift),
            "max_abs_drift_over_samples": float(drift_per_period),
            "samples_period_drift": [[float(a), float(b)] for a, b in mass_samples],
        },
        "series": {
            "t": t.tolist(),
            "eta_probe0": S[:, 0].tolist(),
            "eta_snapshot_final": S[-1].tolist(),
            "amplitude_profile": amp_profile.tolist(),
        },
        "field_matrix": {"t": true_t.tolist(), "eta": S.tolist()},
        "eta_analytic_final": wave.eta(x, t[-1]).tolist(),
    }