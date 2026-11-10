"""Experimental measurement of LBM modal growth rates.

Runs LBM simulations in a periodic box with single Fourier mode
initial conditions and measures:
- Modal amplitude growth rate (exponential fit)
- Phase velocity

These measurements are compared against von Neumann predictions
to validate the analytical stability analysis.

APPLICABILITY: Linear regime only (small perturbation amplitude).

For stability boundary detection (τ < 0.5), we also track the
non-equilibrium norm which captures kinetic mode instabilities.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# D2Q9 lattice
_C2 = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=np.float64)

_W2 = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)


@dataclass
class ExperimentalResult:
    """Result of experimental growth rate measurement.

    Attributes:
        kx: x wave number
        ky: y wave number
        k_mag: wave number magnitude
        tau: relaxation time
        ma: Mach number
        measured_growth_rate: measured exponential growth rate per time step
        measured_phase_speed: measured phase speed
        amplitude_history: array of modal amplitudes over time
        neq_history: array of non-equilibrium norms over time
        fit_residual: residual of exponential fit
        n_steps: number of time steps simulated
        is_divergent: True if amplitude grew without bound
        is_neq_divergent: True if non-equilibrium norm grew without bound
    """
    kx: float
    ky: float
    k_mag: float
    tau: float
    ma: float
    measured_growth_rate: float
    measured_phase_speed: float
    amplitude_history: np.ndarray
    neq_history: np.ndarray
    fit_residual: float
    n_steps: int
    is_divergent: bool
    is_neq_divergent: bool


def _equilibrium_d2q9(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium distribution."""
    Q = 9
    ny, nx = rho.shape
    feq = np.zeros((Q, ny, nx), dtype=np.float64)
    u_sq = ux**2 + uy**2
    for i in range(Q):
        cu = _C2[i, 0] * ux + _C2[i, 1] * uy
        feq[i] = _W2[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
    return feq


def _collide_bgk(f: np.ndarray, tau: float) -> np.ndarray:
    """BGK collision step."""
    # Compute macroscopic
    rho = f.sum(axis=0)
    rho_safe = np.maximum(rho, 1e-12)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    for i in range(9):
        ux += _C2[i, 0] * f[i]
        uy += _C2[i, 1] * f[i]
    ux /= rho_safe
    uy /= rho_safe

    feq = _equilibrium_d2q9(rho, ux, uy)
    return f - (f - feq) / tau


def _stream_periodic(f: np.ndarray) -> np.ndarray:
    """Streaming step with periodic boundary conditions."""
    Q, ny, nx = f.shape
    f_new = np.zeros_like(f)
    for i in range(Q):
        cx, cy = int(_C2[i, 0]), int(_C2[i, 1])
        f_new[i] = np.roll(f[i], shift=(-cy, -cx), axis=(0, 1))
    return f_new


def _compute_neq_norm(f: np.ndarray) -> float:
    """Compute the L2 norm of the non-equilibrium part."""
    rho = f.sum(axis=0)
    rho_safe = np.maximum(rho, 1e-12)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    for i in range(9):
        ux += _C2[i, 0] * f[i]
        uy += _C2[i, 1] * f[i]
    ux /= rho_safe
    uy /= rho_safe

    feq = _equilibrium_d2q9(rho, ux, uy)
    neq = f - feq
    return np.sqrt(np.sum(neq**2))


def _init_fourier_mode(
    nx: int, ny: int, kx: float, ky: float,
    rho0: float, amplitude: float,
    ux0: float = 0.0, uy0: float = 0.0,
    add_neq_perturbation: bool = False,
) -> np.ndarray:
    """Initialize LBM with a single Fourier mode perturbation.

    ρ(x,y) = ρ₀ + A cos(kx x + ky y)
    u(x,y) = u₀ (uniform background flow)

    Args:
        nx, ny: grid dimensions
        kx, ky: wave numbers (in units of 2π/L)
        rho0: background density
        amplitude: perturbation amplitude (must be small for linear regime)
        ux0, uy0: background flow velocity
        add_neq_perturbation: if True, add non-equilibrium perturbation
                              (needed to trigger kinetic mode instabilities)

    Returns:
        Initial distribution function f of shape (9, ny, nx)
    """
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Fourier mode perturbation
    phase = kx * xx + ky * yy
    rho = rho0 + amplitude * np.cos(phase)

    # Uniform velocity (background flow)
    ux = np.full((ny, nx), ux0, dtype=np.float64)
    uy = np.full((ny, nx), uy0, dtype=np.float64)

    f = _equilibrium_d2q9(rho, ux, uy)

    # Add non-equilibrium perturbation if requested
    if add_neq_perturbation:
        # Perturb the stress mode (pxx-like)
        for i in range(9):
            f[i] += amplitude * np.cos(phase) * (_C2[i, 0]**2 - _C2[i, 1]**2)

    return f


def _measure_fourier_amplitude(
    f: np.ndarray, kx: float, ky: float
) -> tuple[float, float]:
    """Measure the amplitude and phase of a Fourier mode in the density field.

    Computes: ρ̃(k) = Σ ρ(x) exp(-i k·x) / N
    Returns: (amplitude, phase)

    Args:
        f: distribution function (9, ny, nx)
        kx, ky: wave numbers

    Returns:
        (amplitude, phase) of the Fourier mode
    """
    rho = f.sum(axis=0)
    ny, nx = rho.shape
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    phase_grid = kx * xx + ky * yy
    rho_k = np.sum(rho * np.exp(-1j * phase_grid)) / (nx * ny)

    amplitude = np.abs(rho_k)
    phase = np.angle(rho_k)
    return amplitude, phase


class ExperimentalGrowthRate:
    """Measure growth rates from LBM simulations.

    Runs periodic-box LBM with single Fourier mode initial conditions
    and fits exponential growth/decay to measure the modal growth rate.

    APPLICABILITY: Linear regime only. Amplitude must be small enough
    that nonlinear effects are negligible.
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        rho0: float = 1.0,
        amplitude: float = 1e-4,
    ):
        """Initialize experimental setup.

        Args:
            nx, ny: grid size (must be large enough for the wave)
            rho0: background density
            amplitude: perturbation amplitude (must be << rho0 for linearity)
        """
        self.nx = nx
        self.ny = ny
        self.rho0 = rho0
        self.amplitude = amplitude

    def measure(
        self,
        kx: float,
        ky: float,
        tau: float,
        ma: float = 0.0,
        flow_angle: float = 0.0,
        n_steps: int = 2000,
        sample_interval: int = 10,
        check_neq: bool = False,
    ) -> ExperimentalResult:
        """Run LBM and measure growth rate of a Fourier mode.

        Args:
            kx, ky: wave numbers (in units of 2π/L, so kx=1 means λ=L)
            tau: relaxation time
            ma: Mach number (background flow speed / cs)
            flow_angle: direction of background flow (rad)
            n_steps: total simulation steps
            sample_interval: steps between amplitude measurements
            check_neq: if True, also track non-equilibrium norm
                      (needed for detecting kinetic mode instabilities)

        Returns:
            ExperimentalResult with measured growth rate and phase speed
        """
        cs = np.sqrt(1.0 / 3.0)
        ux0 = ma * cs * np.cos(flow_angle)
        uy0 = ma * cs * np.sin(flow_angle)

        # Initialize - add NEQ perturbation if checking for kinetic instabilities
        f = _init_fourier_mode(
            self.nx, self.ny, kx, ky,
            self.rho0, self.amplitude,
            ux0, uy0,
            add_neq_perturbation=check_neq,
        )

        # Time stepping
        n_samples = n_steps // sample_interval + 1
        amplitudes = np.zeros(n_samples)
        phases = np.zeros(n_samples)
        neq_norms = np.zeros(n_samples)
        times = np.zeros(n_samples)

        amp, phase = _measure_fourier_amplitude(f, kx, ky)
        amplitudes[0] = amp
        phases[0] = phase
        neq_norms[0] = _compute_neq_norm(f)
        times[0] = 0

        is_divergent = False
        is_neq_divergent = False
        for step in range(1, n_steps + 1):
            f = _collide_bgk(f, tau)
            f = _stream_periodic(f)

            if step % sample_interval == 0:
                idx = step // sample_interval
                if idx < n_samples:
                    amp, phase = _measure_fourier_amplitude(f, kx, ky)
                    amplitudes[idx] = amp
                    phases[idx] = phase
                    neq_norms[idx] = _compute_neq_norm(f)
                    times[idx] = step

                    # Check for divergence (density mode)
                    if amp > 100 * self.amplitude:
                        is_divergent = True
                        for j in range(idx + 1, n_samples):
                            amplitudes[j] = amp * 2.0
                            neq_norms[j] = neq_norms[idx] * 2.0
                            times[j] = step + (j - idx) * sample_interval
                        break

                    # Check for NEQ divergence (kinetic mode)
                    if check_neq:
                        if not np.isfinite(neq_norms[idx]) or neq_norms[idx] > 100 * neq_norms[0]:
                            is_neq_divergent = True

        # Fit exponential growth for density mode: A(t) = A0 exp(σ t)
        valid = amplitudes > 1e-15
        if np.sum(valid) < 3:
            growth_rate = 0.0
            fit_residual = float('inf')
        else:
            log_amp = np.log(amplitudes[valid])
            t_valid = times[valid]
            coeffs = np.polyfit(t_valid, log_amp, 1)
            growth_rate = coeffs[0]
            fit_pred = np.polyval(coeffs, t_valid)
            fit_residual = np.std(log_amp - fit_pred)

        # Phase speed from phase evolution
        k_mag = np.sqrt(kx**2 + ky**2)
        if k_mag > 0 and len(phases[valid]) > 2:
            phase_unwrapped = np.unwrap(phases[valid])
            coeffs_phase = np.polyfit(times[valid], phase_unwrapped, 1)
            omega = -coeffs_phase[0]
            phase_speed = omega / k_mag
        else:
            phase_speed = 0.0

        return ExperimentalResult(
            kx=kx,
            ky=ky,
            k_mag=k_mag,
            tau=tau,
            ma=ma,
            measured_growth_rate=growth_rate,
            measured_phase_speed=phase_speed,
            amplitude_history=amplitudes,
            neq_history=neq_norms,
            fit_residual=fit_residual,
            n_steps=n_steps,
            is_divergent=is_divergent,
            is_neq_divergent=is_neq_divergent,
        )

    def check_stability(
        self,
        kx: float,
        ky: float,
        tau: float,
        ma: float = 0.0,
        n_steps: int = 5000,
    ) -> bool:
        """Quick stability check: does the mode diverge?

        Checks both density mode and non-equilibrium (kinetic) modes.

        Args:
            kx, ky: wave numbers
            tau: relaxation time
            ma: Mach number
            n_steps: simulation steps

        Returns:
            True if stable (neither density nor NEQ mode diverges)
        """
        result = self.measure(
            kx, ky, tau, ma, n_steps=n_steps,
            check_neq=True, sample_interval=10
        )
        return not (result.is_divergent or result.is_neq_divergent)
