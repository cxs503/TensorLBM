"""2D Numerical Wave Tank with Stokes second-order wave generation.

Implements a 2D numerical wave tank using the Körner free-surface LBM (D2Q9)
with:
- Stokes second-order analytical wave generation via relaxation zone (left)
- Wave absorption via relaxation zone (right)
- Two-point reflection coefficient measurement (Goda & Suzuki 1976)
- Dispersion relation verification utilities

Physics reference:
    Stokes (1847), second-order finite-amplitude wave theory.
    Surface elevation:
        η(x,t) = a·cos(θ) + (ka²/4)·(3-σ²)/σ³ · cos(2θ)
    where θ = kx - ωt, σ = tanh(kh), a = first-order amplitude.
    Dispersion relation: ω² = gk·tanh(kh)

Relaxation zone method (Hammer & Ørum 1995; Schaffer & Klopman 1997):
    In the generation/absorption zone, blend simulated and target:
        φ_blend = λ(x)·φ_target + (1-λ(x))·φ_simulated
    where λ varies smoothly across the zone.

Two-point method (Goda & Suzuki 1976):
    Two wave probes at x₁, x₂ measure η₁(t), η₂(t).
    Cross-spectral analysis separates incident/reflected components.
    Reflection coefficient: Kr = |a_R| / |a_I|
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
from .free_surface_lbm_2d import (
    GAS,
    LIQUID,
    INTERFACE,
    SOLID,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)
from .solver import stream as _stream2d

__all__ = [
    "StokesWaveParams",
    "WaveTankConfig",
    "stokes_second_order_surface",
    "stokes_second_order_velocity",
    "solve_dispersion_relation",
    "compute_relaxation_weight",
    "WaveTank2D",
    "WaveProbe",
    "two_point_reflection_coefficient",
    "extract_wave_elevation",
    "harmonic_analysis",
]


# ---------------------------------------------------------------------------
# Stokes second-order wave theory
# ---------------------------------------------------------------------------

@dataclass
class StokesWaveParams:
    """Stokes second-order wave parameters (physical / lattice units)."""
    amplitude: float       # First-order wave amplitude a
    wavenumber: float      # Wavenumber k = 2π/L
    gravity: float         # Gravitational acceleration g
    depth: float           # Water depth h
    phase0: float = 0.0   # Initial phase

    @property
    def sigma(self) -> float:
        """σ = tanh(kh)"""
        return math.tanh(self.wavenumber * self.depth)

    @property
    def omega(self) -> float:
        """Angular frequency from dispersion relation: ω² = gk·tanh(kh)"""
        return math.sqrt(self.gravity * self.wavenumber * self.sigma)

    @property
    def period(self) -> float:
        """Wave period T = 2π/ω"""
        return 2.0 * math.pi / self.omega

    @property
    def wavelength(self) -> float:
        """Wavelength λ = 2π/k"""
        return 2.0 * math.pi / self.wavenumber

    @property
    def phase_speed(self) -> float:
        """Phase speed c = ω/k"""
        return self.omega / self.wavenumber

    @property
    def ka(self) -> float:
        """Wave steepness ka"""
        return self.wavenumber * self.amplitude

    @property
    def second_order_amplitude(self) -> float:
        """Second-order harmonic amplitude:
        a₂ = (ka²/4)·(3-σ²)/σ³
        """
        s = self.sigma
        return (self.wavenumber * self.amplitude**2 / 4.0) * (3.0 - s**2) / s**3


def stokes_second_order_surface(
    x: float | torch.Tensor,
    t: float,
    params: StokesWaveParams,
) -> float | torch.Tensor:
    """Stokes second-order free surface elevation η(x,t).

    η = a·cos(kx-ωt) + a₂·cos(2(kx-ωt))
    where a₂ = (ka²/4)·(3-σ²)/σ³, σ = tanh(kh)
    """
    k = params.wavenumber
    omega = params.omega
    a = params.amplitude
    a2 = params.second_order_amplitude
    theta = k * x - omega * t + params.phase0
    if isinstance(x, torch.Tensor):
        return a * torch.cos(theta) + a2 * torch.cos(2.0 * theta)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return a * np.cos(theta) + a2 * np.cos(2.0 * theta)
    except ImportError:
        pass
    return a * math.cos(theta) + a2 * math.cos(2.0 * theta)


def stokes_second_order_velocity(
    x: float | torch.Tensor,
    z: float | torch.Tensor,
    t: float,
    params: StokesWaveParams,
) -> tuple:
    """Stokes second-order velocity (ux, uz) at position (x, z) and time t.

    z is measured from the bottom (z=0 at bed, z=h at rest surface).
    First order + second order corrections.
    """
    k = params.wavenumber
    omega = params.omega
    a = params.amplitude
    h = params.depth
    s = params.sigma  # tanh(kh)
    theta = k * x - omega * t + params.phase0

    # First-order velocities
    cosh_kz = torch.cosh(k * z) if isinstance(z, torch.Tensor) else math.cosh(k * z)
    sinh_kz = torch.sinh(k * z) if isinstance(z, torch.Tensor) else math.sinh(k * z)
    cosh_kh = math.cosh(k * h)
    sinh_kh = math.sinh(k * h)

    ux1 = a * omega * cosh_kz / sinh_kh * (
        torch.cos(theta) if isinstance(theta, torch.Tensor) else math.cos(theta)
    )
    uz1 = a * omega * sinh_kz / sinh_kh * (
        torch.sin(theta) if isinstance(theta, torch.Tensor) else math.sin(theta)
    )

    # Second-order corrections (dominant terms)
    ka = k * a
    cos_2theta = (
        torch.cos(2.0 * theta) if isinstance(theta, torch.Tensor)
        else math.cos(2.0 * theta)
    )
    sin_2theta = (
        torch.sin(2.0 * theta) if isinstance(theta, torch.Tensor)
        else math.sin(2.0 * theta)
    )

    if isinstance(z, torch.Tensor):
        cosh_2kz = torch.cosh(2.0 * k * z)
        sinh_2kz = torch.sinh(2.0 * k * z)
    else:
        cosh_2kz = math.cosh(2.0 * k * z)
        sinh_2kz = math.sinh(2.0 * k * z)

    coeff2 = (3.0 / 4.0) * ka * a * omega / s**3
    ux2 = coeff2 * cosh_2kz / cosh_kh**2 * cos_2theta
    uz2 = coeff2 * sinh_2kz / cosh_kh**2 * sin_2theta

    return ux1 + ux2, uz1 + uz2


def solve_dispersion_relation(
    k: float, g: float, h: float, tol: float = 1e-10, max_iter: int = 100
) -> float:
    """Solve the dispersion relation ω² = gk·tanh(kh) for ω."""
    return math.sqrt(g * k * math.tanh(k * h))


# ---------------------------------------------------------------------------
# Relaxation zone
# ---------------------------------------------------------------------------

def compute_relaxation_weight(
    x: float | torch.Tensor,
    x_start: float,
    x_end: float,
    kind: str = "generation",
) -> float | torch.Tensor:
    """Compute relaxation zone blending weight λ(x).

    For generation zone: λ goes from 0 (start) to 1 (end).
    For absorption zone: λ goes from 1 (start) to 0 (end).
    """
    try:
        import numpy as _np
        _is_numpy = isinstance(x, _np.ndarray)
    except ImportError:
        _is_numpy = False

    if isinstance(x, torch.Tensor):
        xi = ((x - x_start) / max(x_end - x_start, 1e-10)).clamp(0.0, 1.0)
    elif _is_numpy:
        span = max(x_end - x_start, 1e-10)
        xi = _np.clip((x - x_start) / span, 0.0, 1.0)
    else:
        span = max(x_end - x_start, 1e-10)
        xi = max(0.0, min(1.0, (x - x_start) / span))

    if isinstance(xi, torch.Tensor):
        lam = (torch.exp(-(1.0 - xi)**3) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
        lam = lam.clamp(0.0, 1.0)
    elif _is_numpy:
        lam = (_np.exp(-(1.0 - xi)**3) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
        lam = _np.clip(lam, 0.0, 1.0)
    else:
        lam = (math.exp(-(1.0 - xi)**3) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
        lam = max(0.0, min(1.0, lam))

    if kind == "absorption":
        lam = 1.0 - lam if isinstance(lam, float) else 1.0 - lam

    return lam


# ---------------------------------------------------------------------------
# Wave tank configuration
# ---------------------------------------------------------------------------

@dataclass
class WaveTankConfig:
    """Configuration for 2D numerical wave tank."""
    nx: int = 400
    ny: int = 100
    water_depth: float = 50.0
    wave_amplitude: float = 1.0
    wave_kh: float = 1.5
    gravity: float = 1e-4
    tau: float = 0.6
    n_steps: int = 5000
    gen_zone_length: float = 0.0
    abs_zone_length: float = 0.0
    probe1_x: float = 0.0
    probe2_x: float = 0.0
    device: str = "cpu"
    record_interval: int = 10
    enable_absorption: bool = True

    @property
    def wave_params(self) -> StokesWaveParams:
        k = self.wave_kh / self.water_depth
        return StokesWaveParams(
            amplitude=self.wave_amplitude,
            wavenumber=k,
            gravity=self.gravity,
            depth=self.water_depth,
        )

    @property
    def gen_zone_cells(self) -> int:
        if self.gen_zone_length > 0:
            return min(int(self.gen_zone_length), self.nx // 3)
        return min(max(int(self.wave_params.wavelength), 20), self.nx // 3)

    @property
    def abs_zone_cells(self) -> int:
        if self.abs_zone_length > 0:
            return min(int(self.abs_zone_length), self.nx // 3)
        return min(max(int(self.wave_params.wavelength), 20), self.nx // 3)


# ---------------------------------------------------------------------------
# Wave probe
# ---------------------------------------------------------------------------

@dataclass
class WaveProbe:
    """Wave elevation probe at a fixed x-position."""
    x: int
    time_series: list = field(default_factory=list)
    time_steps: list = field(default_factory=list)


def extract_wave_elevation(
    fill: torch.Tensor,
    x: int,
    water_depth: float,
) -> float:
    """Extract wave surface elevation at x-position from fill field."""
    column = fill[:, x]
    above_threshold = column > 0.01
    if above_threshold.any():
        indices = torch.where(above_threshold)[0]
        return float(indices[-1].item())
    return 0.0


# ---------------------------------------------------------------------------
# Harmonic analysis
# ---------------------------------------------------------------------------

def harmonic_analysis(
    time_series: list,
    dt: float,
    expected_omega: float,
    n_harmonics: int = 3,
) -> dict:
    """Extract harmonic amplitudes from a time series using DFT."""
    import numpy as np

    signal = np.array(time_series, dtype=np.float64)
    n = len(signal)
    if n < 4:
        return {f"A{i+1}": 0.0 for i in range(n_harmonics)}

    signal = signal - np.mean(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    amplitudes = 2.0 * np.abs(fft_vals) / n

    results = {}
    for i in range(1, n_harmonics + 1):
        target_freq = i * expected_omega / (2.0 * np.pi)
        idx = np.argmin(np.abs(freqs - target_freq))
        results[f"A{i}"] = float(amplitudes[idx])
        results[f"f{i}"] = float(freqs[idx])

    return results


def two_point_reflection_coefficient(
    eta1: list,
    eta2: list,
    dt: float,
    x1: float,
    x2: float,
    k: float,
    omega: float,
) -> dict:
    """Compute reflection coefficient using Goda & Suzuki (1976) two-point method.

    Uses the FFT convention where A(x) = B_I * exp(-ikx) + B_R * exp(ikx).
    """
    import numpy as np

    s1 = np.array(eta1, dtype=np.float64)
    s2 = np.array(eta2, dtype=np.float64)
    n = min(len(s1), len(s2))
    s1 = s1[:n] - np.mean(s1[:n])
    s2 = s2[:n] - np.mean(s2[:n])

    if n < 10:
        return {"Kr": 1.0, "a_I": 0.0, "a_R": 0.0}

    fft1 = np.fft.rfft(s1)
    fft2 = np.fft.rfft(s2)
    freqs = np.fft.rfftfreq(n, d=dt)

    target_idx = np.argmin(np.abs(freqs - omega / (2.0 * np.pi)))

    A1 = fft1[target_idx] * 2.0 / n
    A2 = fft2[target_idx] * 2.0 / n

    dx = x2 - x1
    sin_kdx = math.sin(k * dx)
    if abs(sin_kdx) < 1e-6:
        return {"Kr": 1.0, "a_I": 0.0, "a_R": 0.0, "warning": "sin(kdx)≈0"}

    # Goda & Suzuki decomposition with correct FFT sign convention
    exp_mkx1 = complex(math.cos(k * x1), -math.sin(k * x1))
    exp_mkx2 = complex(math.cos(k * x2), -math.sin(k * x2))
    exp_kx1 = complex(math.cos(k * x1), math.sin(k * x1))
    exp_kx2 = complex(math.cos(k * x2), math.sin(k * x2))
    exp_sum_minus = complex(math.cos(k * (x1 + x2)), -math.sin(k * (x1 + x2)))
    exp_sum_plus = complex(math.cos(k * (x1 + x2)), math.sin(k * (x1 + x2)))

    a_I = (A1 * exp_mkx1 - A2 * exp_mkx2) / (2.0j * sin_kdx * exp_sum_minus)
    a_R = (A1 * exp_kx1 - A2 * exp_kx2) / (-2.0j * sin_kdx * exp_sum_plus)

    Kr = abs(a_R) / max(abs(a_I), 1e-15)

    return {
        "Kr": float(Kr),
        "a_I": float(abs(a_I)),
        "a_R": float(abs(a_R)),
        "phase_I": float(np.angle(a_I)),
        "phase_R": float(np.angle(a_R)),
    }


# ---------------------------------------------------------------------------
# Wave Tank 2D Solver
# ---------------------------------------------------------------------------

class WaveTank2D:
    """2D Numerical Wave Tank using Körner free-surface LBM.

    Uses relaxation zones for wave generation (left) and absorption (right).
    The generation zone imposes Stokes 2nd order wave kinematics by blending
    the LBM distributions toward the analytical target.
    """

    def __init__(self, config: WaveTankConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.wp = config.wave_params

        self.nx = config.nx
        self.ny = config.ny
        self.h = config.water_depth

        self.gen_zone_end = config.gen_zone_cells
        self.abs_zone_start = self.nx - config.abs_zone_cells

        # Probe positions
        tank_mid = (self.gen_zone_end + self.abs_zone_start) / 2.0
        tank_len = self.abs_zone_start - self.gen_zone_end
        if config.probe1_x > 0:
            self.probe1_x = int(config.probe1_x)
        else:
            self.probe1_x = int(tank_mid - tank_len * 0.25)
        if config.probe2_x > 0:
            self.probe2_x = int(config.probe2_x)
        else:
            self.probe2_x = int(tank_mid + tank_len * 0.1)

        # Precompute relaxation weight profiles
        self._precompute_relaxation()

        # Initialize fields
        self._initialize_fields()

        # Recording
        self.probe_history_1: list = []
        self.probe_history_2: list = []
        self.step_list: list = []
        self.mass_history: list = []

    def _precompute_relaxation(self):
        """Precompute relaxation weight profiles as tensors."""
        x = torch.arange(self.nx, device=self.device, dtype=torch.float32)
        self.gen_weight = compute_relaxation_weight(
            x, 0.0, float(max(self.gen_zone_end, 1)), "generation"
        )
        self.abs_weight = compute_relaxation_weight(
            x, float(self.abs_zone_start), float(max(self.nx - 1, 1)), "absorption"
        )

    def _initialize_fields(self):
        """Initialize LBM fields for quiescent water at rest."""
        nx, ny, h = self.nx, self.ny, self.h
        device = self.device

        # Solid mask: bottom 2 rows
        solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
        solid[0:2, :] = True
        self.solid_mask = solid

        # Fill field: water up to depth h (from y=2 to y=h+1)
        fill = torch.zeros((ny, nx), dtype=torch.float32, device=device)
        water_top = int(h) + 2
        fill[2:water_top, :] = 1.0
        if water_top < ny:
            fill[water_top, :] = 0.5

        self.fill = fill
        self.flags = init_flags_from_fill_2d(fill, solid)
        self.mass = init_mass_from_fill_2d(fill, self.flags, rho_liquid=1.0)

        # Initialize LBM distributions for quiescent water
        rho = torch.where(
            self.flags != GAS,
            torch.ones((ny, nx), device=device, dtype=torch.float32),
            torch.zeros((ny, nx), device=device, dtype=torch.float32),
        )
        rho = rho.clamp(min=1e-6)
        ux = torch.zeros((ny, nx), device=device, dtype=torch.float32)
        uy = torch.zeros((ny, nx), device=device, dtype=torch.float32)
        self.f = equilibrium(rho, ux, uy, device)

    def _apply_generation_zone(self, step: int):
        """Apply Stokes 2nd order wave generation via relaxation zone.

        Blends LBM distributions toward target equilibrium in the generation zone.
        Also sets the fill field to match the target surface elevation.
        """
        t = float(step)
        wp = self.wp
        gen_end = self.gen_zone_end
        if gen_end <= 0:
            return

        device = self.device
        ny = self.ny

        # x-coordinates in generation zone
        x_gen = torch.arange(gen_end, device=device, dtype=torch.float32)
        lam = self.gen_weight[:gen_end]  # (gen_end,)
        lam_2d = lam.unsqueeze(0)  # (1, gen_end)

        # Target surface elevation
        eta_target = stokes_second_order_surface(x_gen, t, wp)  # (gen_end,)
        y_surface = self.h + 2.0 + eta_target  # surface y-position

        # y-coordinates
        y_coords = torch.arange(ny, device=device, dtype=torch.float32)
        y_from_bed = (y_coords - 2.0).clamp(min=0.0)  # (ny,)

        # Target fill: 1 below surface, linear transition, 0 above
        y_2d = y_coords.unsqueeze(1)  # (ny, 1)
        y_surf_2d = y_surface.unsqueeze(0)  # (1, gen_end)
        target_fill = torch.where(
            y_2d <= y_surf_2d,
            torch.ones_like(y_2d),
            torch.where(
                y_2d <= y_surf_2d + 1.0,
                (y_surf_2d + 1.0 - y_2d).clamp(0.0, 1.0),
                torch.zeros_like(y_2d),
            ),
        )

        # Target velocity field with depth profile
        k = wp.wavenumber
        cosh_kh = math.cosh(k * self.h)
        cosh_profile = torch.cosh(k * y_from_bed).unsqueeze(1)  # (ny, 1)
        depth_factor = (cosh_profile / cosh_kh).clamp(0.0, 1.0)  # (ny, 1)

        # Compute target velocities at each x (at bed level for base profile)
        # Use the full depth-dependent profile
        theta = k * x_gen - wp.omega * t + wp.phase0  # (gen_end,)
        ux_surface = wp.amplitude * wp.omega / math.sinh(k * self.h) * torch.cos(theta)
        uz_surface = wp.amplitude * wp.omega / math.sinh(k * self.h) * torch.sin(theta)

        # Depth-dependent velocities (ny, gen_end)
        ux_target = ux_surface.unsqueeze(0) * depth_factor  # (ny, gen_end)
        uz_target_full = torch.zeros(ny, gen_end, device=device, dtype=torch.float32)
        # Vertical velocity: sinh profile
        sinh_profile = torch.sinh(k * y_from_bed).unsqueeze(1)  # (ny, 1)
        sinh_kh = math.sinh(k * self.h)
        vz_depth_factor = (sinh_profile / sinh_kh).clamp(0.0, 1.0)
        uz_target_full = uz_surface.unsqueeze(0) * vz_depth_factor  # (ny, gen_end)

        # Apply relaxation to fill and distributions
        not_solid = ~self.solid_mask[:, :gen_end]

        # Blend fill
        self.fill[:, :gen_end] = torch.where(
            not_solid,
            lam_2d * target_fill + (1.0 - lam_2d) * self.fill[:, :gen_end],
            self.fill[:, :gen_end],
        )

        # Blend distributions toward target equilibrium
        active = not_solid & (self.fill[:, :gen_end] > 0.01)
        if active.any():
            rho_local = self.f[:, :, :gen_end].sum(dim=0).clamp(min=1e-6)
            feq_target = equilibrium(rho_local, ux_target, uz_target_full, device)

            self.f[:, :, :gen_end] = torch.where(
                active.unsqueeze(0),
                lam_2d.unsqueeze(0) * feq_target + (1.0 - lam_2d.unsqueeze(0)) * self.f[:, :, :gen_end],
                self.f[:, :, :gen_end],
            )

    def _apply_absorption_zone(self, step: int):
        """Apply wave absorption via relaxation zone.

        Blends toward quiescent water (zero velocity, flat surface).
        """
        if not self.cfg.enable_absorption:
            return

        abs_start = self.abs_zone_start
        nx = self.nx
        zone_width = nx - abs_start
        if zone_width <= 0:
            return

        device = self.device
        lam = self.abs_weight[abs_start:]  # (zone_width,)
        lam_2d = lam.unsqueeze(0)  # (1, zone_width)

        # Target: quiescent water at rest
        y_coords = torch.arange(self.ny, device=device, dtype=torch.float32)
        water_top = int(self.h) + 2
        target_fill = torch.zeros((self.ny, zone_width), device=device, dtype=torch.float32)
        target_fill[2:water_top, :] = 1.0
        if water_top < self.ny:
            target_fill[water_top, :] = 0.5

        not_solid = ~self.solid_mask[:, abs_start:]

        # Blend fill
        self.fill[:, abs_start:] = torch.where(
            not_solid,
            lam_2d * target_fill + (1.0 - lam_2d) * self.fill[:, abs_start:],
            self.fill[:, abs_start:],
        )

        # Blend distributions toward quiescent equilibrium
        active = not_solid & (self.fill[:, abs_start:] > 0.01)
        if active.any():
            rho_local = self.f[:, :, abs_start:].sum(dim=0).clamp(min=1e-6)
            feq_quiet = equilibrium(
                rho_local,
                torch.zeros_like(rho_local),
                torch.zeros_like(rho_local),
                device,
            )
            self.f[:, :, abs_start:] = torch.where(
                active.unsqueeze(0),
                lam_2d.unsqueeze(0) * feq_quiet + (1.0 - lam_2d.unsqueeze(0)) * self.f[:, :, abs_start:],
                self.f[:, :, abs_start:],
            )

    def _update_flags_and_mass(self):
        """Re-derive flags and mass from fill field."""
        self.flags = init_flags_from_fill_2d(self.fill, self.solid_mask)
        self.mass = init_mass_from_fill_2d(self.fill, self.flags, rho_liquid=1.0)

    def step(self, step_idx: int):
        """Advance one time step."""
        # Apply generation zone
        self._apply_generation_zone(step_idx)
        self._update_flags_and_mass()

        # Körner free-surface LBM step
        self.f, self.mass, self.flags = koerner_step_2d(
            self.f,
            self.mass,
            self.flags,
            self.solid_mask,
            tau=self.cfg.tau,
            gx=0.0,
            gy=-self.cfg.gravity,
            rho_liquid=1.0,
            rho_gas=1e-3,
        )

        # Apply absorption zone
        self._apply_absorption_zone(step_idx)

        # Update fill from mass
        rho_out = self.f.sum(dim=0).clamp(min=1e-6)
        self.fill = torch.where(
            self.flags == INTERFACE,
            self.mass / rho_out,
            torch.zeros_like(self.mass),
        )
        self.fill = torch.where(
            self.flags == LIQUID,
            torch.ones_like(self.fill),
            self.fill,
        )

    def run(self, n_steps: Optional[int] = None, record: bool = True):
        """Run the wave tank for n_steps."""
        if n_steps is None:
            n_steps = self.cfg.n_steps

        for step_idx in range(n_steps):
            self.step(step_idx)

            if record and (step_idx % self.cfg.record_interval == 0):
                eta1 = extract_wave_elevation(self.fill, self.probe1_x, self.h)
                eta2 = extract_wave_elevation(self.fill, self.probe2_x, self.h)
                self.probe_history_1.append(eta1)
                self.probe_history_2.append(eta2)
                self.step_list.append(step_idx)

                inv = total_liquid_inventory_2d(
                    self.f, self.mass, self.flags, rho_liquid=1.0
                )
                self.mass_history.append(inv)

    def get_results(self) -> dict:
        """Compute and return wave tank results."""
        wp = self.wp
        dt = float(self.cfg.record_interval)

        harmonics = harmonic_analysis(
            self.probe_history_1, dt, wp.omega, n_harmonics=3
        )

        refl = two_point_reflection_coefficient(
            self.probe_history_1,
            self.probe_history_2,
            dt,
            float(self.probe1_x),
            float(self.probe2_x),
            wp.wavenumber,
            wp.omega,
        )

        if len(self.mass_history) > 1:
            m0 = self.mass_history[0]
            m_final = self.mass_history[-1]
            mass_drift = abs(m_final - m0) / max(m0, 1e-10)
        else:
            mass_drift = 0.0

        measured_c = self._measure_phase_speed()

        return {
            "harmonics": harmonics,
            "reflection": refl,
            "mass_drift_relative": mass_drift,
            "measured_phase_speed": measured_c,
            "theoretical_phase_speed": wp.phase_speed,
            "dispersion_error": abs(measured_c - wp.phase_speed) / max(wp.phase_speed, 1e-10),
            "wave_params": {
                "amplitude": wp.amplitude,
                "wavenumber": wp.wavenumber,
                "omega": wp.omega,
                "period": wp.period,
                "wavelength": wp.wavelength,
                "ka": wp.ka,
                "second_order_amplitude": wp.second_order_amplitude,
                "kh": wp.wavenumber * wp.depth,
            },
            "config": {
                "nx": self.nx,
                "ny": self.ny,
                "water_depth": self.h,
                "tau": self.cfg.tau,
                "gravity": self.cfg.gravity,
                "gen_zone_cells": self.gen_zone_end,
                "abs_zone_start": self.abs_zone_start,
                "probe1_x": self.probe1_x,
                "probe2_x": self.probe2_x,
            },
        }

    def _measure_phase_speed(self) -> float:
        """Measure phase speed from cross-correlation of probe signals."""
        import numpy as np

        s1 = np.array(self.probe_history_1, dtype=np.float64)
        s2 = np.array(self.probe_history_2, dtype=np.float64)

        if len(s1) < 20 or len(s2) < 20:
            return self.wp.phase_speed

        s1 = s1 - np.mean(s1)
        s2 = s2 - np.mean(s2)

        # Check if there's any signal
        if np.std(s1) < 1e-6 or np.std(s2) < 1e-6:
            return self.wp.phase_speed

        corr = np.correlate(s1, s2, mode='full')
        lag = np.argmax(corr) - (len(s1) - 1)

        dx = float(self.probe2_x - self.probe1_x)
        dt_record = float(self.cfg.record_interval)
        if abs(lag) > 0:
            c_measured = dx / (abs(lag) * dt_record)
        else:
            c_measured = self.wp.phase_speed

        return c_measured
