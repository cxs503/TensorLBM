"""Two-dimensional phase-field LBM wave tank (Stokes-2nd generation/absorption).

Physics setup
-------------
* The free surface is the **existing phase-field model** of this repository,
  :class:`tensorlbm.cac_lbm.CACSim` (conservative Allen-Cahn interface + HCH
  pressure LBE on D3Q19).  No new interface scheme is introduced: the tank is a
  quasi-2D slab (``nz = 3``, periodic in ``z``) so the D3Q19 stencil collapses
  to an isotropic in-plane 2-D stencil.
* The **wavemaker** and the **absorber** are relaxation zones (waves2Foam
  style): inside a zone the distributions are relaxed towards a target state
  with weight ``w(x)`` peaking at the wall,

      ``f <- (1 - w) f + w f_eq(phi_t, u_t)``,
      ``g <- (1 - w) g + w g_eq(p_t, rho_t, u_t)``,

  with ``(phi_t, u_t)`` from the analytic second-order Stokes solution
  (:mod:`tensorlbm.wave_tank.stokes2`) and a hydrostatic target pressure
  ``p_t = p0 + rho_t g max(eta - y, 0)``.  The target amplitude is ramped in
  over ``ramp_periods`` to avoid a start-up shock.
* The **gauge** is the column integral of the liquid fraction (see
  :func:`tensorlbm.wave_tank.gauges.surface_elevation`), sub-cell accurate and
  independent of the interface width.

Everything is in lattice units (``dx = dt = 1``, ``cs^2 = 1/3``).
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from tensorlbm.cac_lbm import (
    CACSim,
    f_equilibrium,
    g_equilibrium,
    init_phi_layers,
    rho_of_phi,
)

from .gauges import liquid_fraction, surface_elevation
from .stokes2 import Stokes2Wave

__all__ = ["WaveTankConfig", "WaveTank", "WaveTankResult", "run_case"]


@dataclass
class WaveTankConfig:
    """Geometry, physics and schedule of one wave-tank run (lattice units)."""

    # --- wave ---
    kh: float = 1.5
    eps: float = 0.08
    h: float = 24.0  # still-water depth in cells
    g: float = 6.5e-3  # gravity acceleration (lattice), c ~ 0.26-0.31

    # --- phase-field fluid ---
    nu: float = 0.008  # liquid kinematic viscosity (= mu_l / rho_l)
    rho_l: float = 1.0
    rho_g: float = 0.1
    sigma: float = 1.0e-4  # surface tension (capillary speed << wave speed)
    width: float = 2.0  # interface width (cells)
    mobility: float = 0.05
    p0: float = 0.01  # air reference pressure
    tau_floor: float = 0.0  # optional added viscosity for stability studies

    # --- tank geometry ---
    nz: int = 3
    air_cells: int = 12
    lam_in: float = 0.7  # inlet zone length in wavelengths
    lam_mid: float = 1.0  # propagation stretch between the two zones
    lam_out: float = 0.7  # outlet (absorbing) zone length in wavelengths
    probe_frac: tuple[float, ...] = (0.20, 0.37)  # from the inlet-zone end

    # --- schedule ---
    ramp_periods: float = 1.5
    warm_periods: float = 4.0
    meas_periods: float = 4.0
    sample_every: int = 2

    # --- controls ---
    absorb: bool = True  # False = reflective wall (negative control)
    dtype: str = "float32"

    # -- derived -----------------------------------------------------------

    @property
    def g_wave(self) -> float:
        """Effective gravity of the free surface, ``g (1 - rho_g/rho_l)``.

        The phase-field model applies ``rho g`` to both phases, so a surface
        displacement pushes liquid down against gas buoyancy; the restoring
        acceleration seen by the interface is therefore reduced by the density
        ratio.  With ``rho_g/rho_l = 0.1`` using ``g`` directly biases the
        wavenumber by ~12 % (measured: lambda = 89 cells against 100.5
        predicted) -- see the benchmark report.
        """
        return self.g * (1.0 - self.rho_g / self.rho_l)

    @property
    def wave(self) -> Stokes2Wave:
        return Stokes2Wave.from_kh(self.kh, self.eps, self.h, self.g_wave)

    @property
    def lambda_cells(self) -> float:
        return self.wave.wavelength

    def geometry(self) -> dict:
        w = self.wave
        lam = w.wavelength
        ny = int(round(self.h)) + int(self.air_cells) + 2  # + bottom/top wall rows
        n_in = int(round(self.lam_in * lam))
        n_out = int(round(self.lam_out * lam))
        nx = 2 + n_in + int(round(self.lam_mid * lam)) + n_out
        probes = [n_in + f * lam for f in self.probe_frac]
        return {
            "ny": ny,
            "nx": nx,
            "n_in": n_in,
            "n_out": n_out,
            "probes": [float(p) for p in probes],
            "y_s": float(self.h),
        }

    def schedule(self) -> dict:
        w = self.wave
        return {
            "period": w.period,
            "wavelength": w.wavelength,
            "celerity": w.celerity,
            "omega": w.omega,
            "k": w.k,
            "eta2": w.eta2,
            "ramp_steps": int(round(self.ramp_periods * w.period)),
            "warm_steps": int(round(self.warm_periods * w.period)),
            "meas_steps": int(round(self.meas_periods * w.period)),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["probe_frac"] = list(self.probe_frac)
        return d


@dataclass
class WaveTankResult:
    """Scalars and time series of one wave-tank run."""

    config: dict
    metrics: dict
    series: dict = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"config": self.config, "metrics": self.metrics, "series": self.series}


class WaveTank(CACSim):
    """Phase-field LBM wave tank with relaxation-zone wavemaker/absorber."""

    def __init__(self, cfg: WaveTankConfig):
        geo = cfg.geometry()
        ny, nx = geo["ny"], geo["nx"]
        shape = (cfg.nz, ny, nx)
        wall = torch.zeros(shape, dtype=torch.bool)
        wall[:, 0, :] = True  # bottom wall
        wall[:, -1, :] = True  # top wall (kept far above the free surface)
        wall[:, :, 0] = True  # left wall (behind the wavemaker zone)
        wall[:, :, -1] = True  # right wall (behind the absorber)
        super().__init__(
            shape,
            rho_l=cfg.rho_l,
            rho_g=cfg.rho_g,
            mu_l=cfg.nu * cfg.rho_l,
            mu_g=cfg.nu * cfg.rho_l,
            sigma=cfg.sigma,
            width=cfg.width,
            mobility=cfg.mobility,
            mu_law="harmonic",
            p0=cfg.p0,
            gravity=(0.0, -cfg.g, 0.0),
            gravity_mode="rho",
            axes="pww",
            wall_mask=wall,
            wall_phi_policy="neumann",
            device=torch.device("cpu"),
            dtype=getattr(torch, cfg.dtype),
        )
        self.cfg = cfg
        self.geo = geo
        self.sched = cfg.schedule()
        self.wave = cfg.wave
        phi0 = init_phi_layers(shape, cfg.width, cfg.h, dtype=self.dtype)
        self.initialize(phi0)
        # hydrostatic start: g = g_eq(p_hydro) instead of a uniform p0 field,
        # otherwise the whole column free-falls for the first few hundred steps.
        self._set_hydrostatic_pressure()
        # coordinate views (y = row index, 0 = bottom wall row)
        self._y = torch.arange(ny, dtype=self.dtype).view(1, ny, 1)
        self._x = torch.arange(nx, dtype=self.dtype).view(1, 1, nx)
        self._zero = torch.zeros((cfg.nz, ny, nx), dtype=self.dtype)
        # still-water reference column mass (gauge datum)
        self.y_ref = float(
            liquid_fraction(self.phi_np()[1:-1, :]).sum(axis=0).mean()
        )
        self.mass0 = self.liquid_mass()

    def _set_hydrostatic_pressure(self) -> None:
        """Initialise ``g`` in hydrostatic equilibrium with gravity."""
        cfg = self.cfg
        ny = self.shape[1]
        y = torch.arange(ny, dtype=self.dtype).view(1, ny, 1)
        phi = self.f.sum(dim=0)
        rho = rho_of_phi(phi, cfg.rho_l, cfg.rho_g)
        p = cfg.p0 + rho * cfg.g * torch.clamp(cfg.h - y, min=0.0)
        p = p.expand(self.shape[0], ny, self.shape[2]).contiguous()
        zero = torch.zeros_like(phi)
        self.g = g_equilibrium(p, rho, (zero, zero, zero), self.wv, self.cf, self.w0)

    # -- diagnostics -------------------------------------------------------

    def phi_np(self) -> np.ndarray:
        """Phase field as a numpy array (z-averaged)."""
        with torch.no_grad():
            phi = self.f.sum(dim=0).mean(dim=0)
        return phi.detach().cpu().numpy().astype(np.float64)

    def liquid_mass(self) -> float:
        """Total liquid-fraction mass over the fluid nodes (z-integrated)."""
        with torch.no_grad():
            phi = self.f.sum(dim=0)
            fl = torch.clamp(phi, 0.0, 1.0) * self.fluid_mask.to(phi.dtype)
            return float(fl.mean(dim=0).sum().item())

    def elevation(self, phi: np.ndarray | None = None) -> np.ndarray:
        """Free-surface elevation along x (cells, relative to the SWL)."""
        if phi is None:
            phi = self.phi_np()
        return surface_elevation(phi, y_start=1, y_stop=phi.shape[0] - 1, y_ref=self.y_ref)

    # -- relaxation zones --------------------------------------------------

    def _zone_target(self, i0: int, i1: int, t: float, *, generating: bool):
        """Target phase field / velocity / pressure on columns ``[i0, i1)``."""
        cfg = self.cfg
        ny = self.shape[1]
        x = self._x[:, :, i0:i1]
        y = self._y
        nzi = cfg.nz
        width = cfg.width
        ramp = 1.0
        if generating and self.sched["ramp_steps"] > 0:
            n = t / self.sched["ramp_steps"]
            ramp = 0.5 * (1.0 - math.cos(math.pi * min(n, 1.0)))
        if generating:
            w = self.wave
            th = w.k * x - w.omega * t
            eta = ramp * (w.a * torch.cos(th) + w.eta2 * torch.cos(2.0 * th))
        else:
            eta = torch.zeros_like(x)  # absorber target: flat, at rest
            th = torch.zeros_like(x)
        # target surface row and smooth signed distance (positive below it);
        # row 0 is the bottom wall, so the still-water surface sits at row h.
        ys = cfg.h + eta  # broadcast (1, 1, nxi)
        z = ys - y  # broadcast (1, ny, nxi)
        prof = torch.tanh(2.0 * z / width)
        fl = 0.5 * (1.0 + prof)
        phi_t = 0.5 + 0.5 * prof

        if generating:
            w = self.wave
            kh = w.kh
            sh = math.sinh(kh)
            yr = y - cfg.h  # height above SWL (negative below)
            ky1 = w.k * (yr + cfg.h)
            amp1 = ramp * w.a * w.omega / sh
            u1 = amp1 * torch.cosh(ky1) * torch.cos(th)
            w1 = amp1 * torch.sinh(ky1) * torch.sin(th)
            C = w.second_order_potential()
            ky2 = 2.0 * w.k * (yr + cfg.h)
            u2 = 2.0 * w.k * C * torch.cosh(ky2) * torch.cos(2.0 * th)
            w2 = 2.0 * w.k * C * torch.sinh(ky2) * torch.sin(2.0 * th)
            u_t = fl * (u1 + u2)
            w_t = fl * (w1 + w2)
        else:
            u_t = torch.zeros_like(z)
            w_t = torch.zeros_like(z)

        rho_t = rho_of_phi(phi_t, cfg.rho_l, cfg.rho_g)
        p_t = cfg.p0 + rho_t * cfg.g * torch.clamp(ys - y, min=0.0)

        def _bc(a: torch.Tensor) -> torch.Tensor:
            return a.expand(nzi, ny, a.shape[-1]).contiguous()

        e = torch.zeros_like(z)
        return _bc(phi_t), (_bc(u_t), _bc(w_t), _bc(e)), _bc(p_t), _bc(rho_t)

    def _force_zone(self, i0: int, i1: int, t: float, *, generating: bool) -> None:
        cfg = self.cfg
        phi_t, (u_t, w_t, v_t), p_t, rho_t = self._zone_target(i0, i1, t, generating=generating)
        nxi = i1 - i0
        feq = f_equilibrium(phi_t, (u_t, w_t, v_t), self.wv, self.cf)
        geq = g_equilibrium(p_t, rho_t, (u_t, w_t, v_t), self.wv, self.cf, self.w0)
        # blending weight: 1 at the wall side, 0 at the zone exit
        idx = torch.arange(nxi, dtype=self.dtype)
        span = max(nxi - 1, 1)
        if generating:  # wall is on the left of the zone
            wt = torch.cos(0.5 * math.pi * idx / span) ** 2
        else:  # wall is on the right of the zone
            wt = torch.cos(0.5 * math.pi * (span - idx) / span) ** 2
        wt = wt.view(1, 1, nxi).expand(cfg.nz, self.shape[1], nxi).to(self.dtype)
        fm = self.fluid_mask[:, :, i0:i1].unsqueeze(0).to(self.dtype)
        w3 = wt.unsqueeze(0)
        f_slice = self.f[:, :, :, i0:i1]
        g_slice = self.g[:, :, :, i0:i1]
        f_new = (1.0 - w3) * f_slice + w3 * feq
        g_new = (1.0 - w3) * g_slice + w3 * geq
        self.f[:, :, :, i0:i1] = torch.where(fm > 0, f_new, f_slice)
        self.g[:, :, :, i0:i1] = torch.where(fm > 0, g_new, g_slice)

    # -- one tank step -----------------------------------------------------

    def step_tank(self, t: float) -> None:
        self.step()
        self._force_zone(1, 1 + self.geo["n_in"], t, generating=True)
        if self.cfg.absorb:
            nx = self.shape[2]
            n_out = self.geo["n_out"]
            self._force_zone(nx - 1 - n_out, nx - 1, t, generating=False)

    # -- driver ------------------------------------------------------------

    def run(self, *, quiet: bool = True, progress_every: int = 2000) -> WaveTankResult:
        cfg = self.cfg
        sched = self.sched
        n_total = sched["warm_steps"] + sched["meas_steps"]
        probe_cols = [int(round(p)) for p in self.geo["probes"]]
        ts, etas, masses = [], [], []
        t0 = _time.time()
        for it in range(n_total):
            t = float(it)
            self.step_tank(t)
            if it % cfg.sample_every == 0:
                phi = self.phi_np()
                eta = self.elevation(phi)
                ts.append(t)
                etas.append(eta[probe_cols])
                masses.append(self.liquid_mass())
            if not quiet and progress_every and (it + 1) % progress_every == 0:
                dt = (_time.time() - t0) / (it + 1)
                print(f"[wave_tank] {it + 1}/{n_total} steps, {dt * 1e3:.1f} ms/step", flush=True)
        wall = _time.time() - t0
        phi_final = self.phi_np()
        eta_final = self.elevation(phi_final)
        series = {
            "t": np.asarray(ts, dtype=np.float64),
            "probe_eta": np.asarray(etas, dtype=np.float64),  # (n_samples, n_probes)
            "mass": np.asarray(masses, dtype=np.float64),
            "probe_cols": np.asarray(probe_cols, dtype=np.int64),
        }
        snapshot = {
            "phi": phi_final,
            "eta": eta_final,
            "x": np.arange(self.shape[2], dtype=np.float64),
        }
        metrics = {
            "mass0": self.mass0,
            "y_ref": self.y_ref,
            "wall_time_s": wall,
            "steps": n_total,
            "ms_per_step": 1000.0 * wall / max(n_total, 1),
            "ny": self.shape[1],
            "nx": self.shape[2],
            "nz": self.shape[0],
        }
        return WaveTankResult(config=cfg.to_dict(), metrics=metrics, series=series, snapshot=snapshot)


def run_case(cfg: WaveTankConfig, *, quiet: bool = True) -> WaveTankResult:
    """Build and run one tank configuration."""
    tank = WaveTank(cfg)
    return tank.run(quiet=quiet)