"""Acoustic wave propagation verification for TensorLBM.

Simulates 1D and 2D acoustic wave propagation in a uniform medium using
the D2Q9 lattice Boltzmann method, and verifies:

1. **Lattice sound speed consistency**: c_s = 1/√3 (relative error < 1%)
2. **Waveform preservation**: L2 error < 2%
3. **Numerical dissipation scaling**: Im(ω) ∝ k² (second-order accuracy)

Theory
------
In the weakly compressible limit (Ma < 0.1), the LBM recovers the linear
acoustic wave equation:

    ∂²p'/∂t² = c_s² ∇²p'

where c_s = 1/√3 in lattice units (Δx = Δt = 1).

For a sinusoidal wave p'(x, 0) = A sin(kx), u'(x, 0) = 0, the analytical
solution (including numerical dissipation) is:

    p'(x, t) = A sin(kx) cos(c_s k t) exp(-ν_num k² t)

where ν_num = c_s²(τ - 0.5) is the numerical kinematic viscosity.

Numerical dissipation causes amplitude decay:

    A(t) = A₀ exp(-Γ k² t)

where Γ = ν_num.

References
----------
- Lallemand & Luo (2000), JCP 159, 1-37
- Succi (2001), "The Lattice Boltzmann Equation", Oxford Univ. Press
- Krüger et al. (2017), "The Lattice Boltzmann Method", Springer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .d2q9 import C, W, equilibrium, macroscopic

__all__ = [
    "GaussianPulseConfig",
    "SinusoidalWaveConfig",
    "WaveVerificationResult",
    "compute_lattice_sound_speed",
    "run_1d_sinusoidal_wave",
    "run_2d_gaussian_pulse",
    "run_dissipation_scan",
]

# Lattice sound speed squared (exact)
CS2_LATTICE = 1.0 / 3.0
CS_LATTICE = math.sqrt(CS2_LATTICE)  # 1/√3 ≈ 0.57735


@dataclass
class GaussianPulseConfig:
    """Configuration for a Gaussian acoustic pulse."""
    amplitude: float = 1e-4
    center: float | tuple[float, float] = 0.0
    sigma: float = 10.0


@dataclass
class SinusoidalWaveConfig:
    """Configuration for a sinusoidal acoustic wave."""
    amplitude: float = 1e-4
    wavelength: float = 64.0
    direction: tuple[float, float] = (1.0, 0.0)


@dataclass
class WaveVerificationResult:
    """Result of an acoustic wave verification run."""
    case_name: str
    measured_cs: float = 0.0
    theoretical_cs: float = CS_LATTICE
    cs_relative_error: float = 0.0
    l2_error: float = 0.0
    dissipation_rate: float = 0.0
    theoretical_dissipation: float = 0.0
    k_values: Optional[list[float]] = None
    measured_omega_im: Optional[list[float]] = None
    time_history: Optional[list[float]] = None
    pressure_history: Optional[list[np.ndarray]] = None
    pressure_2d_history: Optional[list[np.ndarray]] = None
    passed_cs: bool = False
    passed_l2: bool = False
    passed_dissipation: bool = False


# ---------------------------------------------------------------------------
# Core LBM step
# ---------------------------------------------------------------------------


def _lbm_step_1d(f: torch.Tensor, tau: float, ny: int = 1) -> torch.Tensor:
    """Single LBM collision + streaming step for 1D (embedded in 2D)."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f = f - (f - feq) / tau
    f_new = torch.zeros_like(f)
    for i in range(9):
        cx = int(C[i, 0])
        cy = int(C[i, 1])
        if cx == 0 and cy == 0:
            f_new[i] = f[i]
        else:
            f_new[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return f_new


def _lbm_step_2d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single LBM collision + streaming step for 2D."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f = f - (f - feq) / tau
    f_new = torch.zeros_like(f)
    for i in range(9):
        cx = int(C[i, 0])
        cy = int(C[i, 1])
        if cx == 0 and cy == 0:
            f_new[i] = f[i]
        else:
            f_new[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return f_new


# ---------------------------------------------------------------------------
# 1D sinusoidal wave simulation
# ---------------------------------------------------------------------------


def run_1d_sinusoidal_wave(
    nx: int = 512,
    tau: float = 1.0,
    n_steps: int = 400,
    wave: Optional[SinusoidalWaveConfig] = None,
    device: torch.device | None = None,
) -> dict:
    """Simulate 1D sinusoidal acoustic wave propagation.

    Uses a sinusoidal density perturbation with zero initial velocity.
    The analytical solution (including numerical dissipation) is:

        ρ(x,t) = ρ₀ + A sin(kx) cos(c_s k t) exp(-ν_num k² t)

    Args:
        nx: Number of grid points.
        tau: BGK relaxation time.
        n_steps: Number of time steps.
        wave: Sinusoidal wave configuration.
        device: Torch device.

    Returns:
        Dictionary with simulation results.
    """
    if wave is None:
        wave = SinusoidalWaveConfig(amplitude=1e-4, wavelength=128.0)

    if device is None:
        device = torch.device("cpu")

    ny = 1
    rho0 = 1.0
    k = 2 * math.pi / wave.wavelength

    # Initial condition: sinusoidal density perturbation, zero velocity
    x = torch.arange(nx, dtype=torch.float64, device=device)
    rho_init = rho0 + wave.amplitude * torch.sin(k * x)
    ux_init = torch.zeros(nx, dtype=torch.float64, device=device)
    uy_init = torch.zeros(nx, dtype=torch.float64, device=device)

    f = equilibrium(rho_init, ux_init, uy_init, device=device)

    # Record history
    rho_history = []
    time_history = []

    for t in range(n_steps + 1):
        if t % max(1, n_steps // 20) == 0 or t == 0:
            rho, _, _ = macroscopic(f)
            rho_history.append(rho[0].cpu().numpy())
            time_history.append(float(t))
        if t < n_steps:
            f = _lbm_step_1d(f, tau, ny=ny)

    t_final = float(n_steps)
    cs = CS_LATTICE
    nu_num = CS2_LATTICE * (tau - 0.5)

    # Analytical solution (with dissipation)
    x_np = np.arange(nx, dtype=np.float64)
    rho_analytical = rho0 + wave.amplitude * np.sin(k * x_np) * np.cos(
        cs * k * t_final
    ) * np.exp(-nu_num * k ** 2 * t_final)

    rho_final = rho_history[-1]

    # L2 error (relative to perturbation amplitude)
    l2_num = np.sqrt(np.sum((rho_final - rho_analytical) ** 2))
    l2_den = np.sqrt(np.sum((rho_analytical - rho0) ** 2))
    l2_error = l2_num / l2_den if l2_den > 0 else 0.0

    # Sound speed from von Neumann (phase speed at this k)
    # For the sinusoidal wave, the phase speed is Re(ω)/k
    # We can estimate it from the zero-crossing or peak timing
    # For simplicity, use the von Neumann result
    phase_speed = cs  # Will be refined by von Neumann analysis

    return {
        "rho_history": rho_history,
        "time_history": time_history,
        "rho_final": rho_final,
        "rho_analytical": rho_analytical,
        "l2_error": l2_error,
        "k": k,
        "wavelength": wave.wavelength,
        "tau": tau,
        "n_steps": n_steps,
        "nu_num": nu_num,
        "wave": wave,
    }


# ---------------------------------------------------------------------------
# 2D Gaussian pulse simulation
# ---------------------------------------------------------------------------


def run_2d_gaussian_pulse(
    nx: int = 256,
    ny: int = 256,
    tau: float = 1.0,
    n_steps: int = 200,
    pulse: Optional[GaussianPulseConfig] = None,
    device: torch.device | None = None,
) -> dict:
    """Simulate 2D Gaussian acoustic pulse propagation."""
    if pulse is None:
        pulse = GaussianPulseConfig(
            amplitude=1e-4,
            center=(nx / 2, ny / 2),
            sigma=12.0,
        )

    if device is None:
        device = torch.device("cpu")

    rho0 = 1.0
    cx, cy = pulse.center

    y, x = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64, device=device),
        torch.arange(nx, dtype=torch.float64, device=device),
        indexing="ij",
    )
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    rho_init = rho0 + pulse.amplitude * torch.exp(-r2 / (2 * pulse.sigma ** 2))
    ux_init = torch.zeros_like(rho_init)
    uy_init = torch.zeros_like(rho_init)

    f = equilibrium(rho_init, ux_init, uy_init, device=device)

    rho_history = []
    time_history = []

    for t in range(n_steps + 1):
        if t % max(1, n_steps // 10) == 0 or t == 0:
            rho, _, _ = macroscopic(f)
            rho_history.append(rho.cpu().numpy())
            time_history.append(float(t))
        if t < n_steps:
            f = _lbm_step_2d(f, tau)

    return {
        "rho_history": rho_history,
        "time_history": time_history,
        "rho_final": rho_history[-1],
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "n_steps": n_steps,
        "pulse": pulse,
    }


# ---------------------------------------------------------------------------
# Dissipation rate scan (von Neumann based)
# ---------------------------------------------------------------------------


def compute_lattice_sound_speed() -> float:
    """Return the theoretical lattice sound speed c_s = 1/√3."""
    return CS_LATTICE


def run_dissipation_scan(
    tau: float = 1.0,
    k_min: float = 0.02,
    k_max: float = 0.5,
    n_k: int = 20,
    direction: tuple[float, float] = (1.0, 0.0),
) -> dict:
    """Scan numerical dissipation rate vs wavenumber via von Neumann analysis."""
    from .von_neumann.lattice import D2Q9Lattice
    from .von_neumann.collision import BGKCollision
    from .von_neumann.analysis import AmplificationMatrix, eigenvalues_to_omega, identify_physical_modes

    lattice = D2Q9Lattice()
    collision = BGKCollision(lattice, tau)
    amp = AmplificationMatrix(lattice, collision)

    d = np.array(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)

    k_values = np.linspace(k_min, k_max, n_k)
    omega_re_acoustic = []
    omega_im_acoustic = []

    for k_mag in k_values:
        k = k_mag * d
        M = amp(k)
        eigs = np.linalg.eigvals(M)
        omega = eigenvalues_to_omega(eigs)
        modes = identify_physical_modes(eigs, lattice, k)

        if len(modes["acoustic"]) > 0:
            acoustic_idx = modes["acoustic"]
            forward_idx = acoustic_idx[np.argmax(omega[acoustic_idx].real)]
            omega_re_acoustic.append(omega[forward_idx].real)
            omega_im_acoustic.append(omega[forward_idx].imag)
        else:
            omega_re_acoustic.append(np.nan)
            omega_im_acoustic.append(np.nan)

    omega_re_acoustic = np.array(omega_re_acoustic)
    omega_im_acoustic = np.array(omega_im_acoustic)

    # Fit: Im(ω) = -Γ k²
    valid = ~np.isnan(omega_im_acoustic)
    k_valid = k_values[valid]
    omega_im_valid = omega_im_acoustic[valid]

    n_fit = min(len(k_valid) // 2, 10)
    if n_fit >= 3:
        k2_fit = k_valid[:n_fit] ** 2
        omega_fit = omega_im_valid[:n_fit]
        coeffs = np.polyfit(k2_fit, omega_fit, 1)
        gamma_measured = -coeffs[0]
        gamma_theory = CS2_LATTICE * (tau - 0.5)
        omega_pred = coeffs[0] * k2_fit + coeffs[1]
        ss_res = np.sum((omega_fit - omega_pred) ** 2)
        ss_tot = np.sum((omega_fit - np.mean(omega_fit)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        gamma_measured = np.nan
        gamma_theory = CS2_LATTICE * (tau - 0.5)
        r_squared = 0.0

    # Phase speed from smallest k
    valid_re = ~np.isnan(omega_re_acoustic) & (k_values > 0)
    if np.any(valid_re):
        k_re = k_values[valid_re]
        omega_re = omega_re_acoustic[valid_re]
        phase_speed = omega_re[0] / k_re[0]
        cs_error = abs(phase_speed - CS_LATTICE) / CS_LATTICE
    else:
        phase_speed = np.nan
        cs_error = np.nan

    return {
        "k_values": k_values.tolist(),
        "omega_re_acoustic": omega_re_acoustic.tolist(),
        "omega_im_acoustic": omega_im_acoustic.tolist(),
        "gamma_measured": gamma_measured,
        "gamma_theory": gamma_theory,
        "gamma_relative_error": abs(gamma_measured - gamma_theory) / gamma_theory if gamma_theory > 0 else np.nan,
        "phase_speed": phase_speed,
        "phase_speed_error": cs_error,
        "r_squared": r_squared,
        "tau": tau,
    }


# ---------------------------------------------------------------------------
# Full verification pipeline
# ---------------------------------------------------------------------------


def run_full_verification(
    nx_1d: int = 512,
    nx_2d: int = 256,
    ny_2d: int = 256,
    tau: float = 1.0,
    n_steps_1d: int = 200,
    n_steps_2d: int = 200,
) -> WaveVerificationResult:
    """Run the complete acoustic wave verification suite."""
    result = WaveVerificationResult(case_name="acoustic_wave_verification")

    # 1D sinusoidal wave (eigenmode for clean L2 error)
    res_1d = run_1d_sinusoidal_wave(
        nx=nx_1d, tau=tau, n_steps=n_steps_1d,
        wave=SinusoidalWaveConfig(amplitude=1e-4, wavelength=128.0),
    )
    result.l2_error = res_1d["l2_error"]
    result.time_history = res_1d["time_history"]
    result.pressure_history = res_1d["rho_history"]
    result.passed_l2 = result.l2_error < 0.02

    # 2D Gaussian pulse (visualization)
    res_2d = run_2d_gaussian_pulse(
        nx=nx_2d, ny=ny_2d, tau=tau, n_steps=n_steps_2d,
        pulse=GaussianPulseConfig(amplitude=1e-4, center=(nx_2d / 2, ny_2d / 2), sigma=12.0),
    )
    result.pressure_2d_history = res_2d["rho_history"]

    # Dissipation scan (sound speed + dissipation rate)
    res_diss = run_dissipation_scan(tau=tau)
    result.measured_cs = res_diss["phase_speed"]
    result.cs_relative_error = res_diss["phase_speed_error"]
    result.k_values = res_diss["k_values"]
    result.measured_omega_im = res_diss["omega_im_acoustic"]
    result.dissipation_rate = res_diss["gamma_measured"]
    result.theoretical_dissipation = res_diss["gamma_theory"]
    result.passed_cs = result.cs_relative_error < 0.01
    result.passed_dissipation = res_diss["r_squared"] > 0.99

    return result
