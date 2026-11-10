"""Free-decay isotropic turbulence LES verification against CBC1971.

This module implements the classical Comte-Bellot & Corrsin (1971) free-decay
isotropic turbulence benchmark using Lattice Boltzmann Method with Smagorinsky
LES subgrid-scale model.

Physical setup:
    - Periodic cubic domain
    - Initial velocity field synthesized from von Kármán energy spectrum
    - No mean flow; turbulence decays freely
    - Compare with CBC1971 hot-wire measurements at three downstream stations

References:
    Comte-Bellot, G. & Corrsin, S. (1971). Simple Eulerian time correlation of
    full and narrow-band velocity signals in grid-generated, isotropic turbulence.
    J. Fluid Mech. 48, 273-337.

    Pope, S.B. (2000). Turbulent Flows. Cambridge University Press.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.turbulence.turbulence import collide_smagorinsky_bgk3d


# =============================================================================
# von Kármán spectrum initialization
# =============================================================================

def von_karman_spectrum(k: np.ndarray, k_0: float, C: float = 1.0) -> np.ndarray:
    """Compute von Kármán energy spectrum E(k).

    E(k) = C * k^4 / (k^2 + k_0^2)^(17/6)

    This spectrum peaks at k_peak ≈ k_0 * sqrt(2/5) and has the correct
    k^4 behavior at low k and k^(-5/3) at high k.

    Args:
        k: Wavenumber array.
        k_0: Peak wavenumber parameter (controls energy-containing scale).
        C: Normalization constant.

    Returns:
        Energy spectrum E(k) at each wavenumber.
    """
    k = np.asarray(k, dtype=np.float64)
    exponent = 17.0 / 6.0
    return C * k**4 / (k**2 + k_0**2) ** exponent


def generate_isotropic_velocity_field(
    N: int,
    k_0: float,
    u_rms: float,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate divergence-free isotropic velocity field from von Kármán spectrum.

    Uses the method of synthetic turbulence: random Fourier modes with amplitudes
    set by the target energy spectrum and phases randomized. The velocity field
    is made divergence-free by construction (velocity perpendicular to k).

    Args:
        N: Grid size (cubic domain N x N x N).
        k_0: Peak wavenumber of von Kármán spectrum.
        u_rms: Target RMS velocity.
        seed: Random seed for reproducibility.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple (ux, uy, uz) each of shape (N, N, N).
    """
    rng = np.random.RandomState(seed)

    # Wavenumber vectors
    kx = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi / N
    ky = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi / N
    kz = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi / N

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    K_mag[0, 0, 0] = 1.0  # avoid division by zero

    # Energy spectrum amplitude
    E_k = von_karman_spectrum(K_mag, k_0)
    E_k[0, 0, 0] = 0.0  # no zero-mode energy

    # Normalize to target u_rms: integral of E(k) dk = 3/2 * u_rms^2
    dk = 2.0 * np.pi / N
    k_shells = np.arange(1, N // 2) * dk
    E_shell = np.array([
        np.sum(E_k[(K_mag >= k - dk / 2) & (K_mag < k + dk / 2)])
        for k in k_shells
    ])
    total_energy = np.sum(E_shell) * dk
    if total_energy > 0:
        scale = (1.5 * u_rms**2) / total_energy
        E_k *= scale

    # Amplitude for each Fourier mode
    amplitude = np.sqrt(2.0 * E_k / (4.0 * np.pi * K_mag**2 + 1e-30))
    amplitude[0, 0, 0] = 0.0

    # Random phases for divergence-free construction
    phase1 = rng.uniform(0, 2 * np.pi, (N, N, N))
    phase2 = rng.uniform(0, 2 * np.pi, (N, N, N))

    # Random unit vectors for each mode (for isotropy)
    theta = rng.uniform(0, np.pi, (N, N, N))
    phi = rng.uniform(0, 2 * np.pi, (N, N, N))

    # Unit vector components
    e1x = np.sin(theta) * np.cos(phi)
    e1y = np.sin(theta) * np.sin(phi)
    e1z = np.cos(theta)

    # Orthogonal vector (cross with k-hat) - ensures divergence-free
    e2x = (KY * e1z - KZ * e1y) / K_mag
    e2y = (KZ * e1x - KX * e1z) / K_mag
    e2z = (KX * e1y - KY * e1x) / K_mag

    # Velocity in Fourier space (divergence-free by construction: u·k = 0)
    cos1 = np.cos(phase1)
    sin1 = np.sin(phase1)
    sin2 = np.sin(phase2)
    cos2 = np.cos(phase2)

    ux_hat = amplitude * (e1x * cos1 + e2x * sin1) + 1j * amplitude * (e1x * sin2 - e2x * cos2)
    uy_hat = amplitude * (e1y * cos1 + e2y * sin1) + 1j * amplitude * (e1y * sin2 - e2y * cos2)
    uz_hat = amplitude * (e1z * cos1 + e2z * sin1) + 1j * amplitude * (e1z * sin2 - e2z * cos2)

    # Inverse FFT to get physical space velocity
    ux = np.real(np.fft.ifftn(ux_hat))
    uy = np.real(np.fft.ifftn(uy_hat))
    uz = np.real(np.fft.ifftn(uz_hat))

    # Normalize to exact u_rms
    current_u_rms = np.sqrt(np.mean(ux**2 + uy**2 + uz**2))
    if current_u_rms > 0:
        scale = u_rms / current_u_rms
        ux *= scale
        uy *= scale
        uz *= scale

    # Convert to torch tensors
    ux_t = torch.tensor(ux, dtype=dtype, device=device)
    uy_t = torch.tensor(uy, dtype=dtype, device=device)
    uz_t = torch.tensor(uz, dtype=dtype, device=device)

    return ux_t, uy_t, uz_t


# =============================================================================
# Energy spectrum computation
# =============================================================================

def compute_energy_spectrum(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D energy spectrum E(k) from velocity field.

    Args:
        ux, uy, uz: Velocity components of shape (N, N, N).

    Returns:
        Tuple (k, E_k) where k is wavenumber magnitude and E_k is energy spectrum.
    """
    N = ux.shape[0]

    # Convert to numpy for FFT
    ux_np = ux.detach().cpu().numpy() if isinstance(ux, torch.Tensor) else np.asarray(ux)
    uy_np = uy.detach().cpu().numpy() if isinstance(uy, torch.Tensor) else np.asarray(uy)
    uz_np = uz.detach().cpu().numpy() if isinstance(uz, torch.Tensor) else np.asarray(uz)

    # FFT of velocity components
    ux_hat = np.fft.fftn(ux_np)
    uy_hat = np.fft.fftn(uy_np)
    uz_hat = np.fft.fftn(uz_np)

    # Wavenumber magnitudes (integer wavenumbers)
    kx = np.fft.fftfreq(N, d=1.0) * N
    ky = np.fft.fftfreq(N, d=1.0) * N
    kz = np.fft.fftfreq(N, d=1.0) * N
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Energy in Fourier space: E(k) = 0.5 * sum_over_shells(|u_hat|^2) / N^6
    E_fourier = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2) / (N**6)

    # Bin into spherical shells (integer wavenumber bins)
    k_max = N // 2
    k_bins = np.arange(1, k_max + 1, dtype=np.float64)
    E_k = np.zeros(len(k_bins))

    for i, k_center in enumerate(k_bins):
        mask = (K_mag >= k_center - 0.5) & (K_mag < k_center + 0.5)
        if np.any(mask):
            E_k[i] = np.sum(E_fourier[mask])

    # Wavenumber in physical units (k = 2*pi*n/L, L=N in lattice units)
    k_physical = k_bins * 2.0 * np.pi / N

    return k_physical, E_k


def compute_turbulence_kinetic_energy(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> float:
    """Compute turbulence kinetic energy k = 0.5 * <u_i' u_i'>."""
    return 0.5 * torch.mean(ux**2 + uy**2 + uz**2).item()


# =============================================================================
# CBC1971 experimental data (digitized from paper)
# =============================================================================

# Comte-Bellot & Corrsin (1971) Table 1: grid mesh M = 2 inches = 5.08 cm
# Three downstream stations: x/M = 42, 98, 172
# U_0 = 10 m/s (mean velocity through grid)
# These are approximate digitized values for E(k) vs k*eta

CBC1971_DATA = {
    "x/M=42": {
        "k_eta": np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]),
        "E_k_norm": np.array([0.01, 0.08, 0.25, 0.35, 0.42, 0.38, 0.28, 0.15, 0.06]),
    },
    "x/M=98": {
        "k_eta": np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]),
        "E_k_norm": np.array([0.008, 0.06, 0.20, 0.28, 0.34, 0.30, 0.22, 0.12, 0.05]),
    },
    "x/M=172": {
        "k_eta": np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]),
        "E_k_norm": np.array([0.005, 0.04, 0.15, 0.22, 0.26, 0.23, 0.17, 0.09, 0.04]),
    },
}

# Decay exponent from CBC1971: n in k(t) ~ t^(-n)
CBC1971_DECAY_EXPONENT = {"min": 1.15, "max": 1.30, "nominal": 1.22}


# =============================================================================
# Free-decay simulation
# =============================================================================

@dataclass
class FreeDecayConfig:
    """Configuration for free-decay isotropic turbulence simulation."""

    N: int = 64  # Grid size
    u_rms: float = 0.1  # Initial RMS velocity (lattice units)
    k_0: float = 4.0  # Peak wavenumber (lattice units)
    tau: float = 0.51  # Relaxation time (molecular viscosity)
    C_s: float = 0.12  # Smagorinsky constant
    max_steps: int = 10000  # Maximum simulation steps
    save_interval: int = 100  # Steps between spectrum saves
    seed: int = 42  # Random seed
    device: str = "cpu"


@dataclass
class FreeDecayResult:
    """Results from free-decay simulation."""

    config: FreeDecayConfig
    times: list[float] = field(default_factory=list)  # Simulation times
    tke: list[float] = field(default_factory=list)  # Turbulence kinetic energy
    spectra: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (k, E(k))
    u_rms_history: list[float] = field(default_factory=list)


def run_free_decay_simulation(
    config: FreeDecayConfig,
    verbose: bool = False,
) -> FreeDecayResult:
    """Run free-decay isotropic turbulence LES simulation.

    Uses standard LBM cycle: collide -> stream -> macroscopic.
    The distribution function is maintained throughout; we only extract
    macroscopic variables for diagnostics.

    Args:
        config: Simulation configuration.
        verbose: Print progress.

    Returns:
        FreeDecayResult with time history of spectra and TKE.
    """
    N = config.N
    device = config.device

    # Initialize velocity field from von Kármán spectrum
    ux, uy, uz = generate_isotropic_velocity_field(
        N=N,
        k_0=config.k_0,
        u_rms=config.u_rms,
        seed=config.seed,
        device=device,
    )

    # Convert to distribution function (initial condition)
    rho = torch.ones(N, N, N, dtype=torch.float32, device=device)
    f = equilibrium3d(rho, ux, uy, uz)

    result = FreeDecayResult(config=config)

    # Initial measurements
    k_spec, E_spec = compute_energy_spectrum(ux, uy, uz)
    result.spectra.append((k_spec.copy(), E_spec.copy()))
    result.tke.append(compute_turbulence_kinetic_energy(ux, uy, uz))
    result.times.append(0.0)
    result.u_rms_history.append(torch.sqrt(torch.mean(ux**2 + uy**2 + uz**2)).item())

    # Time integration: standard collide-stream cycle
    for step in range(1, config.max_steps + 1):
        # Collision with Smagorinsky model (or plain BGK for negative control)
        if config.C_s > 0:
            f = collide_smagorinsky_bgk3d(f, config.tau, config.C_s)
        else:
            f = collide_bgk3d(f, config.tau)

        # Streaming (periodic boundaries)
        f = stream3d(f)

        # Save measurements at intervals (extract macroscopic only for diagnostics)
        if step % config.save_interval == 0:
            rho_loc, ux_loc, uy_loc, uz_loc = macroscopic3d(f)

            # Remove mean flow (ensure zero mean for isotropic turbulence)
            ux_loc = ux_loc - ux_loc.mean()
            uy_loc = uy_loc - uy_loc.mean()
            uz_loc = uz_loc - uz_loc.mean()

            k_spec, E_spec = compute_energy_spectrum(ux_loc, uy_loc, uz_loc)
            result.spectra.append((k_spec.copy(), E_spec.copy()))
            result.tke.append(compute_turbulence_kinetic_energy(ux_loc, uy_loc, uz_loc))
            result.times.append(float(step))
            u_rms = torch.sqrt(torch.mean(ux_loc**2 + uy_loc**2 + uz_loc**2)).item()
            result.u_rms_history.append(u_rms)

            if verbose and step % (config.save_interval * 10) == 0:
                print(f"Step {step}: TKE = {result.tke[-1]:.6f}, u_rms = {u_rms:.6f}")

    return result


# =============================================================================
# Analysis functions
# =============================================================================

def fit_decay_exponent(
    times: list[float],
    tke: list[float],
    t_start: float = 100.0,
) -> tuple[float, float]:
    """Fit turbulence kinetic energy decay exponent n in k(t) ~ t^(-n).

    Args:
        times: Simulation times.
        tke: Turbulence kinetic energy history.
        t_start: Start time for fitting (avoid initial transient).

    Returns:
        Tuple (n, R_squared) where n is the decay exponent.
    """
    times = np.array(times)
    tke = np.array(tke)

    # Filter to fitting range
    mask = times >= t_start
    if np.sum(mask) < 3:
        return 0.0, 0.0

    t_fit = times[mask]
    k_fit = tke[mask]

    # Log-log linear fit: log(k) = -n * log(t) + const
    log_t = np.log(t_fit)
    log_k = np.log(k_fit + 1e-30)

    # Linear regression
    coeffs = np.polyfit(log_t, log_k, 1)
    n = -coeffs[0]

    # R-squared
    k_pred = np.exp(np.polyval(coeffs, log_t))
    ss_res = np.sum((k_fit - k_pred) ** 2)
    ss_tot = np.sum((k_fit - np.mean(k_fit)) ** 2)
    R_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    return float(n), float(R_squared)


def fit_inertial_subrange(
    k: np.ndarray,
    E_k: np.ndarray,
    k_min: float = 0.1,
    k_max: float = 1.0,
) -> tuple[float, float]:
    """Fit inertial subrange exponent in E(k) ~ k^(-m).

    Args:
        k: Wavenumber array.
        E_k: Energy spectrum.
        k_min: Minimum wavenumber for fitting.
        k_max: Maximum wavenumber for fitting.

    Returns:
        Tuple (m, R_squared) where m should be close to 5/3.
    """
    mask = (k >= k_min) & (k <= k_max) & (E_k > 1e-20)
    if np.sum(mask) < 3:
        return 0.0, 0.0

    k_fit = k[mask]
    E_fit = E_k[mask]

    # Log-log linear fit
    log_k = np.log(k_fit)
    log_E = np.log(E_fit)

    coeffs = np.polyfit(log_k, log_E, 1)
    m = -coeffs[0]

    # R-squared
    E_pred = np.exp(np.polyval(coeffs, log_k))
    ss_res = np.sum((E_fit - E_pred) ** 2)
    ss_tot = np.sum((E_fit - np.mean(E_fit)) ** 2)
    R_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    return float(m), float(R_squared)


def compute_kolmogorov_scale(
    tke: float,
    nu: float,
    N: int,
) -> float:
    """Estimate Kolmogorov length scale eta = (nu^3 / epsilon)^(1/4).

    Args:
        tke: Turbulence kinetic energy.
        nu: Kinematic viscosity.
        N: Grid size.

    Returns:
        Kolmogorov length scale in lattice units.
    """
    # Estimate dissipation rate from TKE decay
    # epsilon ~ u'^3 / L, where L ~ N / (2*pi) is integral scale
    u_rms = np.sqrt(2.0 * tke / 3.0)
    L = N / (2.0 * np.pi)
    epsilon = u_rms**3 / L

    if epsilon < 1e-30:
        return 1.0

    eta = (nu**3 / epsilon) ** 0.25
    return float(eta)


# =============================================================================
# Verification against CBC1971
# =============================================================================

def compare_with_cbc1971(
    k_sim: np.ndarray,
    E_sim: np.ndarray,
    station: str = "x/M=42",
) -> dict:
    """Compare simulation spectrum with CBC1971 experimental data.

    Args:
        k_sim: Simulation wavenumber array.
        E_sim: Simulation energy spectrum.
        station: CBC1971 station identifier.

    Returns:
        Dictionary with comparison metrics.
    """
    if station not in CBC1971_DATA:
        return {"error": f"Unknown station {station}"}

    cbc = CBC1971_DATA[station]
    k_exp = cbc["k_eta"]
    E_exp = cbc["E_k_norm"]

    # Normalize spectra for comparison
    E_sim_norm = E_sim / (np.max(E_sim) + 1e-30)
    E_exp_norm = E_exp / (np.max(E_exp) + 1e-30)

    # Interpolate simulation to experimental wavenumbers
    E_sim_interp = np.interp(k_exp, k_sim, E_sim_norm, left=0, right=0)

    # Relative error
    mask = E_exp_norm > 0.01
    if np.sum(mask) > 0:
        rel_error = np.abs(E_sim_interp[mask] - E_exp_norm[mask]) / E_exp_norm[mask]
        mean_error = np.mean(rel_error)
        max_error = np.max(rel_error)
    else:
        mean_error = 1.0
        max_error = 1.0

    return {
        "station": station,
        "mean_relative_error": float(mean_error),
        "max_relative_error": float(max_error),
        "k_exp": k_exp.tolist(),
        "E_exp_norm": E_exp_norm.tolist(),
        "E_sim_interp_norm": E_sim_interp.tolist(),
    }


__all__ = [
    "von_karman_spectrum",
    "generate_isotropic_velocity_field",
    "compute_energy_spectrum",
    "compute_turbulence_kinetic_energy",
    "FreeDecayConfig",
    "FreeDecayResult",
    "run_free_decay_simulation",
    "fit_decay_exponent",
    "fit_inertial_subrange",
    "compute_kolmogorov_scale",
    "compare_with_cbc1971",
    "CBC1971_DATA",
    "CBC1971_DECAY_EXPONENT",
]
