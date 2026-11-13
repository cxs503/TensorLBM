"""Rayleigh–Schlichting acoustic streaming in a 2D channel via LBM.

Implements second-order acoustic streaming (Rayleigh streaming) between two
parallel plates driven by a standing acoustic wave.  The simulation extracts
the time-averaged (DC) streaming velocity field and validates it against
Rayleigh's (1884) analytical solution.

Theory
------
A standing wave of angular frequency ω and velocity amplitude U₀ in a
viscous channel of height H generates a steady secondary flow (acoustic
streaming) due to Reynolds stress divergence in the Stokes boundary layer.

Key quantities:
  - Stokes layer thickness:  δ = √(2ν/ω)
  - Streaming Reynolds number:  Re_s = U₀²/(νω)
  - Rayleigh streaming velocity scale:  U_s ~ U₀²/(ωH)

Rayleigh's analytical solution for the outer streaming velocity profile
(inviscid core, y ∈ [0, H]):

    u_outer(y) = (3 U₀²)/(4 ω H) · sin(2kx) · [cos(2πy/H) − 1/3]

where k = π/H for the fundamental mode (half-wavelength = H).

The inner (Stokes) layer near the wall has thickness δ = √(2ν/ω) and
exhibits a Schlichting streaming pattern with reversed flow direction.

LBM Setup
---------
  - D2Q9 lattice, BGK collision
  - Channel: periodic in x, no-slip (bounce-back) at y=0 and y=H
  - Standing wave sustained by oscillating body force
  - Time averaging over many periods to extract DC streaming component

Validation criteria (per campaign P87W2-N07):
  (a) Outer velocity profile vs Rayleigh solution: error < 5%
  (b) Stokes layer thickness δ vs √(2ν/ω): error < 5%
  (c) Scaling U_max ∝ U₀²/ω: exponent error < 3%
  (d) Time-average convergence: 200 vs 400 periods, change < 1%
  (e) Wall shear τ_w ∝ I (linear at low amplitude): residual < 5%
  (f) Negative control: ν→0 ⇒ streaming → 0
  (g) Energy balance: acoustic dissipation = viscous dissipation integral (<3%)
  (h) pytest ≥ 8 items PASS

References
----------
Rayleigh, Lord (1884). "On the circulation of air observed in Kundt's tubes."
    *Phil. Trans. R. Soc. Lond.* 175, 1–21.
Lighthill, J. (1978). "Acoustic streaming." *J. Sound Vib.* 61, 391–418.
Nyborg, W.L. (1953). "Acoustic streaming near a boundary." *J. Acoust. Soc. Am.* 25.
Hamilton, M.F. & Blackstock, D.T. (eds.) (1998). *Nonlinear Acoustics*, Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .d2q9 import equilibrium as d2q9_equilibrium
from .d2q9 import macroscopic as d2q9_macroscopic
from .d2q9 import C as D2Q9_C, W as D2Q9_W, OPPOSITE as D2Q9_OPPOSITE

__all__ = [
    "AcousticStreamingConfig",
    "AcousticStreamingResult",
    "AcousticStreamingSimulator",
    "rayleigh_analytical_outer",
    "stokes_layer_thickness",
    "compute_streaming_reynolds",
    "compute_wall_shear_rate",
    "compute_acoustic_intensity",
    "compute_viscous_dissipation",
    "time_average_field",
    "extract_streaming_velocity",
    "fit_scaling_exponent",
    "run_acoustic_streaming",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AcousticStreamingConfig:
    """Configuration for acoustic streaming simulation."""
    nx: int = 128
    ny: int = 64
    U0: float = 0.01
    omega: float = 0.05
    tau: float = 1.0
    n_warmup: int = 200
    n_avg_periods: int = 200
    period_steps: int = 126
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    @property
    def cs2(self) -> float:
        return 1.0 / 3.0

    @property
    def cs(self) -> float:
        return math.sqrt(1.0 / 3.0)

    @property
    def nu(self) -> float:
        return self.cs2 * (self.tau - 0.5)

    @property
    def mach(self) -> float:
        return self.U0 / self.cs

    @property
    def stokes_thickness(self) -> float:
        return math.sqrt(2.0 * self.nu / self.omega)

    @property
    def streaming_reynolds(self) -> float:
        return self.U0**2 / (self.nu * self.omega)

    @property
    def total_steps(self) -> int:
        return self.n_warmup + self.n_avg_periods * self.period_steps

    def validate(self) -> list[str]:
        errors = []
        if self.nx < 32:
            errors.append(f"nx={self.nx} too small (min 32)")
        if self.ny < 16:
            errors.append(f"ny={self.ny} too small (min 16)")
        if self.tau <= 0.5:
            errors.append(f"tau={self.tau} must be > 0.5 for stability")
        if self.mach > 0.1:
            errors.append(f"Ma={self.mach:.4f} too large (should be ≪ 0.1)")
        if self.nu <= 0:
            errors.append(f"nu={self.nu} must be > 0 (tau must be > 0.5)")
            cells_in_stokes = 0
        else:
            cells_in_stokes = self.stokes_thickness
        if cells_in_stokes < 3 and self.nu > 0:
            errors.append(
                f"Stokes layer δ={cells_in_stokes:.2f} cells < 3; "
                f"increase ny or reduce ω"
            )
        return errors


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------


def rayleigh_analytical_outer(
    y: np.ndarray,
    H: float,
    U0: float,
    omega: float,
    x_phase: float = None,
) -> np.ndarray:
    """Rayleigh's analytical outer streaming velocity profile.

    u_s(y) = (3 U₀²)/(4 ω H) · sin(2kx) · [cos(2πy/H) − 1/3]

    At x = H/4 (quarter wavelength), sin(2kx) = sin(π/2) = 1.
    """
    k = math.pi / H
    if x_phase is None:
        x_phase = H / 4.0
    amplitude = (3.0 * U0**2) / (4.0 * omega * H)
    return amplitude * math.sin(2.0 * k * x_phase) * (
        np.cos(2.0 * math.pi * y / H) - 1.0 / 3.0
    )


def stokes_layer_thickness(nu: float, omega: float) -> float:
    return math.sqrt(2.0 * nu / omega)


def compute_streaming_reynolds(U0: float, nu: float, omega: float) -> float:
    return U0**2 / (nu * omega)


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------


def compute_wall_shear_rate(
    u: torch.Tensor,
    dy: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau_bottom = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * dy)
    ny = u.shape[0]
    tau_top = (3.0 * u[ny - 1] - 4.0 * u[ny - 2] + u[ny - 3]) / (2.0 * dy)
    return tau_bottom, tau_top


def compute_acoustic_intensity(U0: float, rho0: float = 1.0, c0: float = None) -> float:
    if c0 is None:
        c0 = math.sqrt(1.0 / 3.0)
    return 0.5 * rho0 * c0 * U0**2


def compute_viscous_dissipation(
    u: torch.Tensor,
    v: torch.Tensor,
    nu: float,
    dx: float = 1.0,
    dy: float = 1.0,
) -> torch.Tensor:
    dudx = (u[:, 1:] - u[:, :-1]) / dx
    dudy = (u[1:, :] - u[:-1, :]) / dy
    dvdx = (v[:, 1:] - v[:, :-1]) / dx
    dvdy = (v[1:, :] - v[:-1, :]) / dy
    dudx = torch.cat([dudx, dudx[:, -1:]], dim=1)
    dudy = torch.cat([dudy, dudy[-1:, :]], dim=0)
    dvdx = torch.cat([dvdx, dvdx[:, -1:]], dim=1)
    dvdy = torch.cat([dvdy, dvdy[-1:, :]], dim=0)
    Phi = nu * (2.0 * dudx**2 + 2.0 * dvdy**2 + (dudy + dvdx)**2)
    return Phi


def time_average_field(
    field_history: list[torch.Tensor],
    start: int = 0,
) -> torch.Tensor:
    if not field_history or start >= len(field_history):
        raise ValueError("No data to average")
    subset = field_history[start:]
    return torch.stack(subset).mean(dim=0)


def extract_streaming_velocity(
    u_history: list[torch.Tensor],
    v_history: list[torch.Tensor],
    n_warmup_steps: int,
    period_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_avg = time_average_field(u_history, start=n_warmup_steps)
    v_avg = time_average_field(v_history, start=n_warmup_steps)
    return u_avg, v_avg


def fit_scaling_exponent(
    x_vals: list[float],
    y_vals: list[float],
) -> tuple[float, float, float]:
    x = np.array(x_vals)
    y = np.array(y_vals)
    log_x = np.log(x)
    log_y = np.log(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    log_a = coeffs[1]
    a = math.exp(log_a)
    y_pred = a * x**b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, a, r_squared


# ---------------------------------------------------------------------------
# Optimized LBM Acoustic Streaming Simulator
# ---------------------------------------------------------------------------


class AcousticStreamingSimulator:
    """Optimized LBM simulator for Rayleigh acoustic streaming.

    Uses vectorized streaming and bounce-back.  Sustains the standing wave
    via an oscillating body force.
    """

    _SHIFTS = [
        (0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 0, -1), (4, -1, 0),
        (5, 1, 1), (6, 1, -1), (7, -1, -1), (8, -1, 1),
    ]

    def __init__(self, config: AcousticStreamingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.nx, self.ny = config.nx, config.ny
        self.cs2 = config.cs2
        self.cs = config.cs
        self.tau = config.tau
        self.omega = config.omega
        self.U0 = config.U0
        self.k = math.pi / config.ny

        self.C = D2Q9_C.to(self.device)
        self.W = D2Q9_W.to(self.device).to(self.dtype)
        self.OPPOSITE = D2Q9_OPPOSITE.to(self.device)

        # Pre-compute spatial force profile
        self._precompute_force_profile()

        self._init_distributions()
        self.step_count = 0
        self.u_history: list[torch.Tensor] = []
        self.v_history: list[torch.Tensor] = []
        self.rho_history: list[torch.Tensor] = []

    def _precompute_force_profile(self):
        """Pre-compute the spatial profile of the body force."""
        ny, nx = self.ny, self.nx
        device = self.device
        dtype = self.dtype

        y = torch.arange(ny, dtype=dtype, device=device)
        x = torch.arange(nx, dtype=dtype, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')

        # Force profile: F_x ∝ sin(kx) to drive standing wave
        # For a standing wave: u = U0 sin(kx) cos(ωt)
        # The force needed: F = ρ ∂u/∂t = -ρ U0 ω sin(kx) sin(ωt)
        # We use: F_x = F0 sin(kx) sin(ωt) where F0 = U0 * ω
        self._force_spatial = torch.sin(self.k * gx)

    def _init_distributions(self):
        ny, nx = self.ny, self.nx
        device = self.device
        dtype = self.dtype

        y = torch.arange(ny, dtype=dtype, device=device)
        x = torch.arange(nx, dtype=dtype, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')

        # Start from rest - the body force will build up the standing wave
        ux = torch.zeros((ny, nx), dtype=dtype, device=device)
        uy = torch.zeros((ny, nx), dtype=dtype, device=device)
        rho = torch.ones((ny, nx), dtype=dtype, device=device)

        self.f = d2q9_equilibrium(rho, ux, uy)
        self.rho = rho
        self.ux = ux
        self.uy = uy

    def _stream_vectorized(self):
        f_new = torch.empty_like(self.f)
        for q in range(9):
            sy = int(self._SHIFTS[q][1])
            sx = int(self._SHIFTS[q][2])
            if sy == 0 and sx == 0:
                f_new[q] = self.f[q]
            else:
                f_new[q] = torch.roll(self.f[q], shifts=(sy, sx), dims=(0, 1))
        self.f = f_new

    def _bounce_back_vectorized(self):
        ny = self.ny
        # Bottom wall (y=0)
        self.f[4, 0, :], self.f[2, 1, :] = self.f[2, 1, :].clone(), self.f[4, 0, :].clone()
        self.f[8, 0, :], self.f[6, 1, :] = self.f[6, 1, :].clone(), self.f[8, 0, :].clone()
        self.f[7, 0, :], self.f[5, 1, :] = self.f[5, 1, :].clone(), self.f[7, 0, :].clone()
        # Top wall (y=ny-1)
        self.f[2, ny-1, :], self.f[4, ny-2, :] = self.f[4, ny-2, :].clone(), self.f[2, ny-1, :].clone()
        self.f[6, ny-1, :], self.f[8, ny-2, :] = self.f[8, ny-2, :].clone(), self.f[6, ny-1, :].clone()
        self.f[5, ny-1, :], self.f[7, ny-2, :] = self.f[7, ny-2, :].clone(), self.f[5, ny-1, :].clone()

    def _collide_with_force(self):
        """BGK collision with oscillating body force (Guo forcing scheme)."""
        rho, ux, uy = d2q9_macroscopic(self.f)
        feq = d2q9_equilibrium(rho, ux, uy)

        # Oscillating body force: F_x = F0 · cos(kx) · sin(ωt)
        t = self.step_count
        force_amp = self.U0 * self.omega * math.sin(self.omega * t)
        fx = force_amp * self._force_spatial

        # Guo forcing term
        # F_i = w_i · (1 - 1/(2τ)) · (c_i · F) / c_s²
        force_term = torch.zeros_like(self.f)
        for q in range(9):
            cx_q = float(self.C[q, 0])
            force_term[q] = self.W[q] * (1.0 - 1.0 / (2.0 * self.tau)) * cx_q * fx / self.cs2

        self.f = self.f - (self.f - feq) / self.tau + force_term

    def step(self):
        self._collide_with_force()
        self._stream_vectorized()
        self._bounce_back_vectorized()
        self.step_count += 1

    def get_macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return d2q9_macroscopic(self.f)

    def record_snapshot(self):
        rho, ux, uy = self.get_macroscopic()
        self.u_history.append(ux.cpu())
        self.v_history.append(uy.cpu())
        self.rho_history.append(rho.cpu())

    def run(self, n_steps: int, record_interval: int = 1) -> dict:
        for step in range(n_steps):
            self.step()
            if step % record_interval == 0:
                self.record_snapshot()
        return {"total_steps": self.step_count, "n_snapshots": len(self.u_history)}

    def get_streaming_field(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.u_history:
            raise ValueError("No history recorded")
        u_stream = torch.stack(self.u_history).mean(dim=0)
        v_stream = torch.stack(self.v_history).mean(dim=0)
        return u_stream, v_stream


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


@dataclass
class AcousticStreamingResult:
    config: AcousticStreamingConfig
    u_stream: np.ndarray
    v_stream: np.ndarray
    u_rms: np.ndarray
    v_rms: np.ndarray
    rho_mean: np.ndarray
    wall_shear_bottom: np.ndarray
    wall_shear_top: np.ndarray
    rayleigh_analytical: np.ndarray
    stokes_thickness: float
    streaming_reynolds: float
    max_streaming_velocity: float
    n_steps: int
    convergence_metric: float = 0.0
    energy_dissipation: float = 0.0


def run_acoustic_streaming(config: AcousticStreamingConfig) -> AcousticStreamingResult:
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid config: {'; '.join(errors)}")

    sim = AcousticStreamingSimulator(config)
    warmup_record = max(1, config.period_steps // 4)
    sim.run(config.n_warmup, record_interval=warmup_record)
    sim.run(config.n_avg_periods * config.period_steps, record_interval=1)

    u_stream, v_stream = sim.get_streaming_field()

    u_stack = torch.stack(sim.u_history)
    v_stack = torch.stack(sim.v_history)
    u_mean = u_stack.mean(dim=0)
    v_mean = v_stack.mean(dim=0)
    u_rms = ((u_stack - u_mean)**2).mean(dim=0).sqrt()
    v_rms = ((v_stack - v_mean)**2).mean(dim=0).sqrt()

    rho_stack = torch.stack(sim.rho_history)
    rho_mean = rho_stack.mean(dim=0)

    tau_bottom, tau_top = compute_wall_shear_rate(u_stream)

    y_vals = np.arange(config.ny)
    rayleigh_sol = rayleigh_analytical_outer(y_vals, config.ny, config.U0, config.omega)

    max_streaming = float(u_stream.abs().max())

    n_snap = len(sim.u_history)
    half = n_snap // 2
    u_first = torch.stack(sim.u_history[:half]).mean(dim=0)
    u_second = torch.stack(sim.u_history[half:]).mean(dim=0)
    convergence = float((u_second - u_first).abs().max() / (u_stream.abs().max() + 1e-15))

    nu = config.nu
    dissipation = compute_viscous_dissipation(u_stream, v_stream, nu)
    total_dissipation = float(dissipation.sum())

    return AcousticStreamingResult(
        config=config,
        u_stream=u_stream.numpy(),
        v_stream=v_stream.numpy(),
        u_rms=u_rms.numpy(),
        v_rms=v_rms.numpy(),
        rho_mean=rho_mean.numpy(),
        wall_shear_bottom=tau_bottom.numpy(),
        wall_shear_top=tau_top.numpy(),
        rayleigh_analytical=rayleigh_sol,
        stokes_thickness=config.stokes_thickness,
        streaming_reynolds=config.streaming_reynolds,
        max_streaming_velocity=max_streaming,
        n_steps=sim.step_count,
        convergence_metric=convergence,
        energy_dissipation=total_dissipation,
    )
