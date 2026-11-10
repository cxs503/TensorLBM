"""Ffowcs Williams–Hawkings (FW-H) acoustic analogy for rotating sources.

Implements the FW-H surface integral formulation for rotating noise sources
(e.g. marine propellers, rotors, fans) with:

1. **Retarded time windowing**: iterative Newton–Raphson solution of
   t − τ = |x − y(τ)| / c₀ for each source–observer pair.

2. **Doppler correction**: factor 1 / (1 − Mᵣ) where Mᵣ is the radial
   Mach number between source and observer.

3. **FW-H decomposition**:
   - Thickness (monopole) term:  ∂/∂t [ρ₀ vₙ δ(f)]
   - Loading (dipole) term:     −∂/∂xᵢ [Lᵢⱼ nⱼ δ(f)]
   - Quadrupole term:            ∂²/∂xᵢ∂xⱼ [Tᵢⱼ H(f)]  (neglected at low M)

4. **BPF harmonic analysis**: blade-passing frequency f_BPF = N_blades × RPM / 60,
   with harmonic amplitude decay law.

Analytical validation anchors
-----------------------------
- **Gardner rotating monopole**: exact solution for a point monopole on a
  circular trajectory, used to validate thickness term + Doppler.
- **Curle dipole**: cos θ directivity pattern for a stationary dipole,
  used to validate loading term directivity.
- **Blake BPF harmonics**: 1/n amplitude decay for harmonic series,
  used to validate spectral content.

Limitations
-----------
- Subsonic rotation only (M_tip < 0.7).
- Linear acoustics (no shock waves).
- Quadrupole term neglected (valid for low Mach number flows).
- Compact source approximation for analytical cases.

References
----------
Ffowcs Williams, J.E. & Hawkings, D.L. (1969). "Sound generated
    aerodynamically." *Phil. Trans. R. Soc. A* 264, 321–342.
Gardner, S. (1970s). Rotating monopole analytical solution.
Curle, N. (1955). "The influence of solid boundaries upon aerodynamic sound."
    *Proc. R. Soc. Lond. A* 231, 505–514.
Blake, W.K. (1986). *Mechanics of Flow-Induced Sound and Vibration*.
    Academic Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

__all__ = [
    "solve_retarded_time",
    "compute_doppler_factor",
    "RotatingSourceConfig",
    "RotatingSourceResult",
    "compute_rotating_source_fwh",
    "gardner_monopole_analytical",
    "curle_dipole_analytical",
    "blake_harmonic_spectrum",
    "compute_spl_spectrum",
    "compute_directivity",
    "compute_acoustic_power",
]

# ---------------------------------------------------------------------------
# Retarded time solver
# ---------------------------------------------------------------------------


def solve_retarded_time(
    observer_pos: np.ndarray,
    source_trajectory,
    t_obs: float,
    c0: float,
    tau_init: Optional[float] = None,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> tuple[float, int, bool]:
    """Solve the retarded time equation iteratively via Newton–Raphson.

    The retarded time τ satisfies:
        t_obs − τ = |x_obs − y(τ)| / c₀

    where y(τ) is the source position at emission time τ.

    Define F(τ) = τ + |x_obs − y(τ)| / c₀ − t_obs = 0.
    Newton iteration: τ_{k+1} = τ_k − F(τ_k) / F'(τ_k)

    F'(τ) = 1 − (x_obs − y(τ)) · v(τ) / (c₀ |x_obs − y(τ)|)
          = 1 − Mᵣ

    Args:
        observer_pos: Observer position array (3,).
        source_trajectory: Callable τ → (position(3,), velocity(3,)).
        t_obs: Observer time.
        c0: Speed of sound.
        tau_init: Initial guess (default: t_obs).
        tol: Convergence tolerance on |Δτ|.
        max_iter: Maximum Newton iterations.

    Returns:
        (tau, n_iter, converged) tuple.
    """
    if tau_init is None:
        tau_init = t_obs

    tau = tau_init
    converged = False
    n_iter = 0

    for k in range(max_iter):
        pos, vel = source_trajectory(tau)
        r_vec = observer_pos - pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-15:
            # Observer at source position — degenerate case
            return tau, k + 1, True

        F = tau + r_mag / c0 - t_obs
        Mr = np.dot(r_vec, vel) / (c0 * r_mag)  # radial Mach number
        dF = 1.0 - Mr

        if abs(dF) < 1e-12:
            # Singular — Mᵣ ≈ 1 (sonic), bail out
            return tau, k + 1, False

        delta_tau = -F / dF
        tau += delta_tau
        n_iter = k + 1

        if abs(delta_tau) < tol:
            converged = True
            break

    return tau, n_iter, converged


# ---------------------------------------------------------------------------
# Doppler factor
# ---------------------------------------------------------------------------


def compute_doppler_factor(
    observer_pos: np.ndarray,
    source_pos: np.ndarray,
    source_vel: np.ndarray,
    c0: float,
) -> float:
    """Compute the Doppler amplification factor 1 / (1 − Mᵣ).

    Mᵣ = (r · v) / (c₀ |r|) is the radial Mach number.

    Args:
        observer_pos: Observer position (3,).
        source_pos: Source position (3,).
        source_vel: Source velocity (3,).
        c0: Speed of sound.

    Returns:
        Doppler factor 1 / (1 − Mᵣ).
    """
    r_vec = observer_pos - source_pos
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-15:
        return 1.0
    Mr = np.dot(r_vec, source_vel) / (c0 * r_mag)
    return 1.0 / (1.0 - Mr)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RotatingSourceConfig:
    """Configuration for a rotating source FW-H computation.

    Attributes:
        n_blades: Number of blades.
        rpm: Rotational speed (revolutions per minute).
        radius: Propeller radius (m).
        hub_radius: Hub radius (m), default 0.
        c0: Speed of sound (m/s), default 340.
        rho0: Reference density (kg/m³), default 1000 (water).
        thrust: Total thrust force (N), for loading term.
        torque: Total torque (N·m), for loading term.
        n_segments: Number of blade segments per blade (discretization).
        observer_positions: Array of observer positions (n_obs, 3).
        t_start: Start time (s).
        t_end: End time (s).
        dt: Time step (s).
        include_thickness: Include thickness (monopole) term.
        include_loading: Include loading (dipole) term.
    """
    n_blades: int
    rpm: float
    radius: float
    hub_radius: float = 0.0
    c0: float = 340.0
    rho0: float = 1000.0
    thrust: float = 1000.0
    torque: float = 200.0
    n_segments: int = 20
    observer_positions: Optional[np.ndarray] = None
    t_start: float = 0.0
    t_end: float = 0.1
    dt: float = 1e-5
    include_thickness: bool = True
    include_loading: bool = True

    @property
    def omega(self) -> float:
        """Angular velocity (rad/s)."""
        return 2.0 * math.pi * self.rpm / 60.0

    @property
    def tip_speed(self) -> float:
        """Tip speed (m/s)."""
        return self.omega * self.radius

    @property
    def tip_mach(self) -> float:
        """Tip Mach number."""
        return self.tip_speed / self.c0

    @property
    def bpf(self) -> float:
        """Blade passing frequency (Hz)."""
        return self.n_blades * self.rpm / 60.0


@dataclass
class RotatingSourceResult:
    """Result of a rotating source FW-H computation.

    Attributes:
        time: Time array (s).
        pressure: Acoustic pressure at each observer (n_obs, n_time).
        observer_positions: Observer positions (n_obs, 3).
        spl: Sound pressure level spectrum (n_obs, n_freq), dB re 1 μPa.
        frequencies: Frequency array (Hz).
        directivity: Directivity index DI(θ) if computed.
        bpf_harmonics: BPF harmonic amplitudes.
        retarded_time_stats: Statistics on retarded time convergence.
        config: Original configuration.
    """
    time: np.ndarray
    pressure: np.ndarray
    observer_positions: np.ndarray
    spl: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None
    directivity: Optional[dict] = None
    bpf_harmonics: Optional[dict] = None
    retarded_time_stats: Optional[dict] = None
    config: Optional[RotatingSourceConfig] = None


# ---------------------------------------------------------------------------
# Source trajectory for rotating blade element
# ---------------------------------------------------------------------------


def _blade_element_trajectory(
    blade_idx: int,
    n_blades: int,
    omega: float,
    radius: float,
    radial_pos: float,
    phase_offset: float = 0.0,
) -> callable:
    """Return a callable τ → (position, velocity) for a blade element.

    The blade element is at radial position `radial_pos` on blade `blade_idx`.

    Args:
        blade_idx: Blade index (0 to n_blades-1).
        n_blades: Total number of blades.
        omega: Angular velocity (rad/s).
        radius: Propeller radius (m).
        radial_pos: Radial position of the element (m).
        phase_offset: Additional phase offset (rad).

    Returns:
        Callable τ → (pos(3,), vel(3,)).
    """
    theta_0 = 2.0 * math.pi * blade_idx / n_blades + phase_offset

    def trajectory(tau: float) -> tuple[np.ndarray, np.ndarray]:
        theta = theta_0 + omega * tau
        # Position in x-y plane (rotation about z-axis)
        pos = np.array([
            radial_pos * math.cos(theta),
            radial_pos * math.sin(theta),
            0.0,
        ])
        # Velocity (tangential)
        vel = np.array([
            -radial_pos * omega * math.sin(theta),
            radial_pos * omega * math.cos(theta),
            0.0,
        ])
        return pos, vel

    return trajectory


# ---------------------------------------------------------------------------
# FW-H surface integral for rotating source
# ---------------------------------------------------------------------------


def compute_rotating_source_fwh(
    config: RotatingSourceConfig,
) -> RotatingSourceResult:
    """Compute FW-H acoustic pressure for a rotating source.

    Implements the thickness (monopole) and loading (dipole) terms
    of the FW-H equation with retarded time and Doppler correction.

    Thickness term (Farassat Formulation 1A):
        p_T(x, t) = ρ₀ / (4π) ∫_S [ (L̇_n) / (r (1-Mᵣ)²) + ... ] dS

    Loading term:
        p_L(x, t) = 1 / (4π) ∫_S [ (Ṫ_n) / (r (1-Mᵣ)²) + ... ] dS

    For a simplified rotating source model, we use blade element theory
    with distributed monopole (thickness) and dipole (loading) sources.

    Args:
        config: Rotating source configuration.

    Returns:
        RotatingSourceResult with pressure time series and spectra.
    """
    n_obs = len(config.observer_positions)
    n_time = int(round((config.t_end - config.t_start) / config.dt)) + 1
    time_arr = np.linspace(config.t_start, config.t_end, n_time)

    pressure = np.zeros((n_obs, n_time))

    # Blade element discretization
    radial_positions = np.linspace(config.hub_radius, config.radius, config.n_segments)
    dr = (config.radius - config.hub_radius) / config.n_segments

    # Distributed source strength per blade element
    # Thickness: proportional to blade volume displacement rate
    # Loading: proportional to blade force (thrust + torque)
    total_blade_area = math.pi * (config.radius**2 - config.hub_radius**2)
    area_per_segment = total_blade_area / (config.n_blades * config.n_segments)

    # Simplified source strength distribution
    # Thickness source strength (volume flow rate per element)
    q_thickness = config.rho0 * config.omega * config.radius * area_per_segment / config.n_segments

    # Loading source strength (force per element)
    # Distribute thrust and torque along blade
    f_loading = config.thrust / (config.n_blades * config.n_segments)
    t_loading = config.torque / (config.n_blades * config.n_segments)

    # Retarded time convergence tracking
    rt_iters = []
    rt_converged_count = 0
    rt_total = 0

    for obs_idx in range(n_obs):
        obs_pos = config.observer_positions[obs_idx]

        for t_idx, t_obs in enumerate(time_arr):
            p_total = 0.0

            for blade_idx in range(config.n_blades):
                for seg_idx, r_pos in enumerate(radial_positions):
                    traj = _blade_element_trajectory(
                        blade_idx, config.n_blades, config.omega,
                        config.radius, r_pos,
                    )

                    # Solve retarded time
                    tau, n_iter, converged = solve_retarded_time(
                        obs_pos, traj, t_obs, config.c0,
                        tol=1e-8, max_iter=20,
                    )
                    rt_iters.append(n_iter)
                    rt_total += 1
                    if converged:
                        rt_converged_count += 1

                    pos, vel = traj(tau)
                    r_vec = obs_pos - pos
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag < 1e-10:
                        continue

                    # Doppler factor
                    doppler = compute_doppler_factor(obs_pos, pos, vel, config.c0)
                    doppler_sq = doppler ** 2

                    # Unit vector
                    r_hat = r_vec / r_mag

                    # Thickness term (monopole) — simplified
                    if config.include_thickness:
                        # Monopole strength proportional to normal velocity
                        v_n = np.dot(vel, r_hat)  # radial component
                        # Time derivative of thickness source
                        p_thick = (config.rho0 * q_thickness * config.omega *
                                   doppler_sq / (4.0 * math.pi * r_mag))
                        # Phase modulation from rotation
                        theta = math.atan2(pos[1], pos[0])
                        p_thick *= math.sin(config.omega * tau + theta)
                        p_total += p_thick

                    # Loading term (dipole) — simplified
                    if config.include_loading:
                        # Dipole strength from blade forces
                        # Force direction: axial (thrust) + tangential (torque)
                        f_axial = f_loading  # along z
                        f_tangential = t_loading / r_pos if r_pos > 1e-10 else 0.0

                        # Dipole pressure: p ~ (F · r̂) / (r (1-Mᵣ)²)
                        f_vec = np.array([
                            -f_tangential * math.sin(math.atan2(pos[1], pos[0])),
                            f_tangential * math.cos(math.atan2(pos[1], pos[0])),
                            f_axial,
                        ])
                        f_dot_r = np.dot(f_vec, r_hat)
                        p_load = f_dot_r * doppler_sq / (4.0 * math.pi * r_mag)
                        p_total += p_load

            pressure[obs_idx, t_idx] = p_total

    # Compute spectra
    spl, freqs = compute_spl_spectrum(pressure, config.dt, ref_pressure=1e-6)

    # BPF harmonic analysis
    bpf_harmonics = _extract_bpf_harmonics(
        pressure, freqs, config.bpf, n_harmonics=10,
    )

    # Retarded time statistics
    rt_stats = {
        "mean_iters": float(np.mean(rt_iters)) if rt_iters else 0.0,
        "max_iters": int(np.max(rt_iters)) if rt_iters else 0,
        "convergence_rate": rt_converged_count / max(rt_total, 1),
    }

    return RotatingSourceResult(
        time=time_arr,
        pressure=pressure,
        observer_positions=config.observer_positions,
        spl=spl,
        frequencies=freqs,
        bpf_harmonics=bpf_harmonics,
        retarded_time_stats=rt_stats,
        config=config,
    )


# ---------------------------------------------------------------------------
# SPL spectrum computation
# ---------------------------------------------------------------------------


def compute_spl_spectrum(
    pressure: np.ndarray,
    dt: float,
    ref_pressure: float = 1e-6,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sound pressure level spectrum from pressure time series.

    SPL(f) = 20 log₁₀(|P(f)| / p_ref)  [dB]

    Args:
        pressure: Pressure signal (n_obs, n_time) or (n_time,).
        dt: Time step (s).
        ref_pressure: Reference pressure (Pa), default 1 μPa (underwater).
        window: Window function name.

    Returns:
        (spl, frequencies) arrays.
    """
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)

    n_obs, n_time = pressure.shape
    n_freq = n_time // 2 + 1
    fs = 1.0 / dt

    # Apply window
    if window == "hann":
        w = np.hanning(n_time)
    elif window == "hamming":
        w = np.hamming(n_time)
    else:
        w = np.ones(n_time)

    spl_all = np.zeros((n_obs, n_freq))
    freqs = np.fft.rfftfreq(n_time, dt)

    for i in range(n_obs):
        # FFT
        p_windowed = pressure[i] * w
        P = np.fft.rfft(p_windowed)
        # One-sided spectrum amplitude
        amp = np.abs(P) * 2.0 / n_time
        amp[0] /= 2.0  # DC component
        if n_freq > 1:
            amp[-1] /= 2.0  # Nyquist

        # SPL
        with np.errstate(divide="ignore", invalid="ignore"):
            spl = 20.0 * np.log10(amp / ref_pressure)
        spl = np.where(np.isfinite(spl), spl, -200.0)
        spl_all[i] = spl

    return spl_all, freqs


# ---------------------------------------------------------------------------
# BPF harmonic extraction
# ---------------------------------------------------------------------------


def _extract_bpf_harmonics(
    pressure: np.ndarray,
    freqs: np.ndarray,
    bpf: float,
    n_harmonics: int = 10,
    bandwidth: float = 0.5,
) -> dict:
    """Extract BPF harmonic amplitudes from spectrum.

    Args:
        pressure: Pressure signal (n_obs, n_time).
        freqs: Frequency array.
        bpf: Blade passing frequency (Hz).
        n_harmonics: Number of harmonics to extract.
        bandwidth: Search bandwidth around each harmonic (Hz).

    Returns:
        Dictionary with harmonic frequencies and amplitudes.
    """
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)

    n_obs = pressure.shape[0]
    harmonics = {
        "frequencies": [],
        "amplitudes": [],
        "amplitudes_db": [],
    }

    for n in range(1, n_harmonics + 1):
        f_target = n * bpf
        mask = np.abs(freqs - f_target) < bandwidth
        if not np.any(mask):
            harmonics["frequencies"].append(f_target)
            harmonics["amplitudes"].append(0.0)
            harmonics["amplitudes_db"].append(-200.0)
            continue

        # Find peak in band
        idx = np.argmax(np.abs(freqs[mask] - f_target) < bandwidth)
        actual_freq = freqs[mask][idx]

        # Average amplitude across observers
        amps = []
        for i in range(n_obs):
            # Re-compute from pressure for accuracy
            n_time = pressure.shape[1]
            P = np.fft.rfft(pressure[i] * np.hanning(n_time))
            amp = np.abs(P) * 2.0 / n_time
            amps.append(amp[mask][idx])

        mean_amp = float(np.mean(amps))
        with np.errstate(divide="ignore", invalid="ignore"):
            amp_db = 20.0 * np.log10(mean_amp / 1e-6)
        if not np.isfinite(amp_db):
            amp_db = -200.0

        harmonics["frequencies"].append(float(actual_freq))
        harmonics["amplitudes"].append(mean_amp)
        harmonics["amplitudes_db"].append(float(amp_db))

    return harmonics


# ---------------------------------------------------------------------------
# Directivity computation
# ---------------------------------------------------------------------------


def compute_directivity(
    pressure: np.ndarray,
    observer_positions: np.ndarray,
    source_center: Optional[np.ndarray] = None,
    ref_pressure: float = 1e-6,
) -> dict:
    """Compute directivity index DI(θ, φ) from pressure field.

    DI(θ, φ) = SPL(θ, φ) − SPL_omni

    where SPL_omni is the omnidirectional average.

    Args:
        pressure: Pressure RMS or peak at each observer (n_obs,).
        observer_positions: Observer positions (n_obs, 3).
        source_center: Source center position (3,). Default: origin.
        ref_pressure: Reference pressure.

    Returns:
        Dictionary with theta, phi, DI arrays.
    """
    if source_center is None:
        source_center = np.zeros(3)

    n_obs = len(observer_positions)
    if pressure.ndim > 1:
        pressure = pressure.reshape(-1)

    # Compute angles
    theta = np.zeros(n_obs)  # polar angle from z-axis
    phi = np.zeros(n_obs)    # azimuthal angle in x-y plane

    for i in range(n_obs):
        r_vec = observer_positions[i] - source_center
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-10:
            continue
        theta[i] = math.acos(np.clip(r_vec[2] / r_mag, -1.0, 1.0))
        phi[i] = math.atan2(r_vec[1], r_vec[0])

    # SPL at each observer
    with np.errstate(divide="ignore", invalid="ignore"):
        spl = 20.0 * np.log10(np.abs(pressure) / ref_pressure)
    spl = np.where(np.isfinite(spl), spl, -200.0)

    # Omnidirectional average
    spl_omni = np.mean(spl[spl > -100]) if np.any(spl > -100) else 0.0

    di = spl - spl_omni

    return {
        "theta": theta,
        "phi": phi,
        "spl": spl,
        "di": di,
        "spl_omni": float(spl_omni),
    }


# ---------------------------------------------------------------------------
# Acoustic power computation
# ---------------------------------------------------------------------------


def compute_acoustic_power(
    pressure: np.ndarray,
    observer_positions: np.ndarray,
    c0: float,
    rho0: float,
    source_center: Optional[np.ndarray] = None,
) -> float:
    """Compute total acoustic power by integrating intensity over a sphere.

    W = ∫ I · dA = ∫ (p_rms² / (ρ₀ c₀)) dA

    Uses trapezoidal integration over the observer sphere.

    Args:
        pressure: RMS pressure at each observer (n_obs,).
        observer_positions: Observer positions (n_obs, 3).
        c0: Speed of sound.
        rho0: Reference density.
        source_center: Source center. Default: origin.

    Returns:
        Total acoustic power (W).
    """
    if source_center is None:
        source_center = np.zeros(3)

    if pressure.ndim > 1:
        pressure = pressure.reshape(-1)

    n_obs = len(observer_positions)
    r_vecs = observer_positions - source_center
    r_mags = np.linalg.norm(r_vecs, axis=1)

    # Intensity at each observer: I = p_rms² / (ρ₀ c₀)
    intensity = pressure**2 / (rho0 * c0)

    # Approximate surface area per observer (Voronoi-like)
    # For uniformly distributed observers on a sphere:
    mean_r = np.mean(r_mags)
    area_per_obs = 4.0 * math.pi * mean_r**2 / n_obs

    # Total power
    power = np.sum(intensity * area_per_obs)
    return float(power)


# ---------------------------------------------------------------------------
# Analytical validation cases
# ---------------------------------------------------------------------------


def gardner_monopole_analytical(
    observer_pos: np.ndarray,
    source_radius: float,
    omega: float,
    source_strength: float,
    c0: float,
    rho0: float,
    t: np.ndarray,
) -> np.ndarray:
    """Gardner rotating monopole analytical solution.

    A point monopole of strength Q rotating on a circle of radius R
    at angular velocity ω produces pressure:

    p'(t) = ρ₀ Q ω² R / (4π c₀ r) · cos(ωτ − φ) / (1 − Mᵣ)²

    where τ is the retarded time and φ is the observer azimuth.

    This is the exact solution for a rotating compact monopole source.

    Args:
        observer_pos: Observer position (3,).
        source_radius: Rotation radius (m).
        omega: Angular velocity (rad/s).
        source_strength: Monopole strength (m³/s).
        c0: Speed of sound (m/s).
        rho0: Reference density (kg/m³).
        t: Time array (s).

    Returns:
        Pressure time series.
    """
    n_time = len(t)
    pressure = np.zeros(n_time)

    obs_r = np.linalg.norm(observer_pos)
    obs_phi = math.atan2(observer_pos[1], observer_pos[0])

    for i, t_obs in enumerate(t):
        # Iterative retarded time
        tau = t_obs
        for _ in range(10):
            src_x = source_radius * math.cos(omega * tau)
            src_y = source_radius * math.sin(omega * tau)
            r_vec = observer_pos - np.array([src_x, src_y, 0.0])
            r_mag = np.linalg.norm(r_vec)
            tau_new = t_obs - r_mag / c0
            if abs(tau_new - tau) < 1e-10:
                break
            tau = tau_new

        # Source position and velocity at retarded time
        src_x = source_radius * math.cos(omega * tau)
        src_y = source_radius * math.sin(omega * tau)
        src_pos = np.array([src_x, src_y, 0.0])
        src_vel = np.array([
            -source_radius * omega * math.sin(omega * tau),
            source_radius * omega * math.cos(omega * tau),
            0.0,
        ])

        r_vec = observer_pos - src_pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-10:
            continue

        Mr = np.dot(r_vec, src_vel) / (c0 * r_mag)
        doppler = 1.0 / (1.0 - Mr)

        # Gardner formula
        pressure[i] = (rho0 * source_strength * omega**2 * source_radius *
                       doppler**2 * math.cos(omega * tau - obs_phi) /
                       (4.0 * math.pi * c0 * r_mag))

    return pressure


def curle_dipole_analytical(
    observer_pos: np.ndarray,
    force_amplitude: float,
    omega: float,
    c0: float,
    rho0: float,
    t: np.ndarray,
    force_direction: np.ndarray = None,
) -> np.ndarray:
    """Curle dipole analytical solution.

    A compact dipole source with oscillating force F(t) = F₀ sin(ωt)
    produces pressure:

    p'(t) = (F₀ · r̂) / (4π r) · ω / c₀ · sin(ω(t − r/c₀))

    The directivity follows cos θ where θ is the angle between the
    force direction and the observer direction.

    Args:
        observer_pos: Observer position (3,).
        force_amplitude: Force amplitude (N).
        omega: Angular frequency (rad/s).
        c0: Speed of sound (m/s).
        rho0: Reference density (kg/m³).
        t: Time array (s).
        force_direction: Force direction unit vector (3,). Default: z-axis.

    Returns:
        Pressure time series.
    """
    if force_direction is None:
        force_direction = np.array([0.0, 0.0, 1.0])

    r_vec = observer_pos
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-10:
        return np.zeros_like(t)

    r_hat = r_vec / r_mag
    cos_theta = np.dot(force_direction, r_hat)

    # Retarded time
    t_ret = t - r_mag / c0

    # Curle dipole formula
    pressure = (force_amplitude * cos_theta / (4.0 * math.pi * r_mag) *
                omega / c0 * np.sin(omega * t_ret))

    return pressure


def blake_harmonic_spectrum(
    n_harmonics: int,
    fundamental_amplitude: float,
    decay_exponent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blake harmonic spectrum for BPF series.

    The harmonic amplitudes follow a power-law decay:
        A_n = A₁ / n^α

    where α ≈ 1 for compact sources.

    Args:
        n_harmonics: Number of harmonics.
        fundamental_amplitude: Amplitude of the fundamental (n=1).
        decay_exponent: Decay exponent α (default 1.0).

    Returns:
        (harmonic_numbers, amplitudes) arrays.
    """
    n = np.arange(1, n_harmonics + 1)
    amplitudes = fundamental_amplitude / n**decay_exponent
    return n, amplitudes
