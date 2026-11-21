"""Rotating-source aeroacoustics: retarded-time FW-H + Doppler/BPF analysis.

This module extends the stationary FWH analogy in ``acoustics.py`` to handle
**moving / rotating sources** (e.g. propeller blades).  The key additions are:

1. **Retarded-time Newton solver** – solves the implicit equation
   ``t_r = t - |x - y(t_r)| / c`` for each observer-source pair at every
   observation time, using Newton iteration with warm-start from the previous
   time step.

2. **Sliding time-window buffer** – a circular buffer that retains only the
   source history within ``[t - Δt_w, t]``, reducing memory from O(N·T) to
   O(N·N_window).

3. **Analytical source models** (no flow field required) for verification:
   - Stationary monopole / dipole
   - Rotating monopole / dipole
   - Pulsating sphere

4. **Doppler frequency-shift calculator** – ``f_obs = f_src * c / (c - v·cosθ)``.

5. **Blade-Passing-Frequency (BPF) analyzer** – identifies BPF = n·RPM/60
   and its harmonics in the acoustic spectrum.

References
----------
Ffowcs Williams, J.E. & Hawkings, D.L. (1969).
    "Sound generation by turbulence and surfaces in arbitrary motion."
    *Phil. Trans. R. Soc. Lond. A* 264, 321–342.
Farassat, F. (1996).
    "Analytical solutions for the Ffowcs Williams–Hawkings equation."
    *NASA TM-110626*.
Brentner, K.S. & Farassat, F. (1998).
    "Modeling aerodynamically generated sound of helicopter rotors."
    *Prog. Aerospace Sci.* 34, 67–120.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "RetardedTimeSolver",
    "SlidingWindowBuffer",
    "StationaryMonopole",
    "RotatingMonopole",
    "StationaryDipole",
    "RotatingDipole",
    "PulsatingSphere",
    "compute_doppler_shift",
    "compute_bpf_peaks",
    "compute_rotating_fwh",
    "AnalyticalSource",
]

_TWO_PI = 2.0 * math.pi
_REF_PRESSURE = 20.0e-6  # 20 µPa


# ---------------------------------------------------------------------------
# Retarded-time Newton solver
# ---------------------------------------------------------------------------


class RetardedTimeSolver:
    """Solve the retarded-time equation t_r = t - |x - y(t_r)| / c.

    For a moving source with trajectory ``y(t)``, the retarded time ``t_r``
    at observation time ``t`` satisfies the implicit equation above.  This
    solver uses Newton iteration with warm-start (previous ``t_r`` as initial
    guess) for rapid convergence.

    Parameters
    ----------
    max_iter : int
        Maximum Newton iterations (default 10).
    tol : float
        Convergence tolerance on |t_r_new - t_r_old| (default 1e-10).
    """

    def __init__(self, max_iter: int = 10, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        obs_pos: np.ndarray,
        source_trajectory,
        t_obs: float,
        t_r_guess: float | None = None,
        c0: float = 343.0,
    ) -> tuple[float, float]:
        """Solve for retarded time.

        Parameters
        ----------
        obs_pos : np.ndarray, shape (3,)
            Observer position.
        source_trajectory : callable
            Function ``y(t) -> np.ndarray`` of shape (3,).
        t_obs : float
            Observation time.
        t_r_guess : float, optional
            Initial guess (warm-start from previous step).
        c0 : float
            Speed of sound.

        Returns
        -------
        t_r : float
            Converged retarded time.
        residual : float
            Final residual |f(t_r)|.
        """
        if t_r_guess is None:
            t_r_guess = t_obs  # initial guess: no delay

        t_r = t_r_guess
        for _ in range(self.max_iter):
            y_tr = source_trajectory(t_r)
            r_vec = obs_pos - y_tr
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1e-15:
                return t_obs, 0.0

            # f(t_r) = t_r - t_obs + r(t_r)/c0
            f_val = t_r - t_obs + r_mag / c0

            # Numerical derivative for Newton step
            dt = 1e-12
            y_tr_p = source_trajectory(t_r + dt)
            r_vec_p = obs_pos - y_tr_p
            r_mag_p = np.linalg.norm(r_vec_p)
            f_val_p = (t_r + dt) - t_obs + r_mag_p / c0

            df_dt = (f_val_p - f_val) / dt
            if abs(df_dt) < 1e-20:
                break

            t_r_new = t_r - f_val / df_dt
            residual = abs(t_r_new - t_r)
            t_r = t_r_new

            if residual < self.tol:
                return t_r, abs(f_val)

        return t_r, abs(f_val)


# ---------------------------------------------------------------------------
# Sliding time-window buffer (circular buffer)
# ---------------------------------------------------------------------------


class SlidingWindowBuffer:
    """Circular buffer retaining source history within [t - Δt_w, t].

    Memory complexity: O(N_sources × N_window) instead of O(N × T).

    Parameters
    ----------
    n_sources : int
        Number of source points.
    window_size : int
        Number of time steps to retain.
    dt : float
        Physical time step.
    """

    def __init__(self, n_sources: int, window_size: int, dt: float):
        self.n_sources = n_sources
        self.window_size = window_size
        self.dt = dt
        # Circular buffer: (n_sources, window_size)
        self.buffer = np.zeros((n_sources, window_size), dtype=np.float64)
        self.write_idx = 0  # next write position
        self.total_steps = 0  # total steps written

    def push(self, values: np.ndarray) -> None:
        """Push a new time slice into the buffer.

        Parameters
        ----------
        values : np.ndarray, shape (n_sources,)
            Source values at current time step.
        """
        self.buffer[:, self.write_idx] = values
        self.write_idx = (self.write_idx + 1) % self.window_size
        self.total_steps += 1

    def get_history(self, n_steps: int) -> np.ndarray:
        """Retrieve the last n_steps of history.

        Parameters
        ----------
        n_steps : int
            Number of steps to retrieve (≤ window_size).

        Returns
        -------
        history : np.ndarray, shape (n_sources, n_steps)
            Time history, oldest first.
        """
        n_steps = min(n_steps, self.window_size, self.total_steps)
        if n_steps == 0:
            return np.zeros((self.n_sources, 0))

        # Indices in circular buffer (oldest first)
        start = (self.write_idx - n_steps) % self.window_size
        if start + n_steps <= self.window_size:
            return self.buffer[:, start:start + n_steps].copy()
        else:
            return np.concatenate([
                self.buffer[:, start:],
                self.buffer[:, :start + n_steps - self.window_size],
            ], axis=1)

    @property
    def current_time(self) -> float:
        """Current physical time based on total steps written."""
        return self.total_steps * self.dt


# ---------------------------------------------------------------------------
# Analytical source models
# ---------------------------------------------------------------------------


class AnalyticalSource:
    """Base class for analytical acoustic sources."""

    def pressure_at(self, obs_pos: np.ndarray, t: float, c0: float = 343.0) -> float:
        """Compute acoustic pressure at observer position and time."""
        raise NotImplementedError


class StationaryMonopole(AnalyticalSource):
    """Stationary monopole (pulsating point source).

    p'(x, t) = (ρ₀ / 4πr) · dQ/dt(t - r/c₀)

    For a harmonic source Q(t) = Q₀ sin(ωt):
    p'(x, t) = (ρ₀ Q₀ ω / 4πr) · cos(ω(t - r/c₀))
    """

    def __init__(
        self,
        position: np.ndarray,
        source_strength: float,
        frequency: float,
        rho0: float = 1.225,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.source_strength = source_strength  # Q₀ (volume flux amplitude)
        self.frequency = frequency  # Hz
        self.rho0 = rho0
        self.omega = _TWO_PI * frequency

    def pressure_at(self, obs_pos: np.ndarray, t: float, c0: float = 343.0) -> float:
        r_vec = np.asarray(obs_pos) - self.position
        r = np.linalg.norm(r_vec)
        if r < 1e-15:
            return 0.0
        tau = t - r / c0
        # p' = (ρ₀ Q₀ ω / 4πr) cos(ωτ)
        return (self.rho0 * self.source_strength * self.omega / (4 * math.pi * r)) * math.cos(
            self.omega * tau
        )

    def trajectory(self, t: float) -> np.ndarray:
        return self.position.copy()


class RotatingMonopole(AnalyticalSource):
    """Monopole rotating on a circle of radius R in the xy-plane.

    y(t) = (R cos(Ωt), R sin(Ωt), 0)
    """

    def __init__(
        self,
        radius: float,
        omega_rot: float,
        source_strength: float,
        frequency: float,
        rho0: float = 1.225,
        phase: float = 0.0,
    ):
        self.radius = radius
        self.omega_rot = omega_rot  # rotational speed (rad/s)
        self.source_strength = source_strength
        self.frequency = frequency
        self.rho0 = rho0
        self.omega = _TWO_PI * frequency
        self.phase = phase

    def trajectory(self, t: float) -> np.ndarray:
        theta = self.omega_rot * t + self.phase
        return np.array([
            self.radius * math.cos(theta),
            self.radius * math.sin(theta),
            0.0,
        ])

    def velocity(self, t: float) -> np.ndarray:
        theta = self.omega_rot * t + self.phase
        return np.array([
            -self.radius * self.omega_rot * math.sin(theta),
            self.radius * self.omega_rot * math.cos(theta),
            0.0,
        ])

    def pressure_at(self, obs_pos: np.ndarray, t: float, c0: float = 343.0) -> float:
        # Solve retarded time
        solver = RetardedTimeSolver()
        t_r, _ = solver.solve(obs_pos, self.trajectory, t, c0=c0)
        y_tr = self.trajectory(t_r)
        r_vec = np.asarray(obs_pos) - y_tr
        r = np.linalg.norm(r_vec)
        if r < 1e-15:
            return 0.0
        tau = t_r  # retarded time IS the source time
        # Doppler factor: 1 / (1 - M_r) where M_r = (v·r̂)/c
        v = self.velocity(t_r)
        r_hat = r_vec / r
        M_r = np.dot(v, r_hat) / c0
        doppler = 1.0 / (1.0 - M_r) if abs(1.0 - M_r) > 1e-10 else 1.0

        return (
            self.rho0
            * self.source_strength
            * self.omega
            / (4 * math.pi * r)
            * math.cos(self.omega * tau)
            * doppler
        )


class StationaryDipole(AnalyticalSource):
    """Stationary dipole (oscillating force in a given direction).

    p'(x, t) = (1 / 4πc₀) · (r̂·F̂) · dF/dt(t - r/c₀) / r
             + (1 / 4π) · (r̂·F̂) · F(t - r/c₀) / r²

    For harmonic force F(t) = F₀ sin(ωt) in direction d̂:
    """

    def __init__(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        force_amplitude: float,
        frequency: float,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        self.direction = d / np.linalg.norm(d)
        self.force_amplitude = force_amplitude
        self.frequency = frequency
        self.omega = _TWO_PI * frequency

    def pressure_at(self, obs_pos: np.ndarray, t: float, c0: float = 343.0) -> float:
        r_vec = np.asarray(obs_pos) - self.position
        r = np.linalg.norm(r_vec)
        if r < 1e-15:
            return 0.0
        r_hat = r_vec / r
        cos_theta = np.dot(r_hat, self.direction)
        tau = t - r / c0

        # Far-field term: (cosθ / 4πc₀r) · dF/dt
        df_dt = self.force_amplitude * self.omega * math.cos(self.omega * tau)
        far = cos_theta * df_dt / (4 * math.pi * c0 * r)

        # Near-field term: (cosθ / 4πr²) · F
        f_val = self.force_amplitude * math.sin(self.omega * tau)
        near = cos_theta * f_val / (4 * math.pi * r * r)

        return far + near

    def trajectory(self, t: float) -> np.ndarray:
        return self.position.copy()


class RotatingDipole(AnalyticalSource):
    """Dipole rotating on a circle (e.g. blade loading noise).

    The dipole direction rotates with the source.
    """

    def __init__(
        self,
        radius: float,
        omega_rot: float,
        force_amplitude: float,
        frequency: float,
        phase: float = 0.0,
    ):
        self.radius = radius
        self.omega_rot = omega_rot
        self.force_amplitude = force_amplitude
        self.frequency = frequency
        self.omega = _TWO_PI * frequency
        self.phase = phase

    def trajectory(self, t: float) -> np.ndarray:
        theta = self.omega_rot * t + self.phase
        return np.array([
            self.radius * math.cos(theta),
            self.radius * math.sin(theta),
            0.0,
        ])

    def velocity(self, t: float) -> np.ndarray:
        theta = self.omega_rot * t + self.phase
        return np.array([
            -self.radius * self.omega_rot * math.sin(theta),
            self.radius * self.omega_rot * math.cos(theta),
            0.0,
        ])

    def dipole_direction(self, t: float) -> np.ndarray:
        """Dipole direction rotates with the source (radial)."""
        theta = self.omega_rot * t + self.phase
        return np.array([math.cos(theta), math.sin(theta), 0.0])

    def pressure_at(self, obs_pos: np.ndarray, t: float, c0: float = 343.0) -> float:
        solver = RetardedTimeSolver()
        t_r, _ = solver.solve(obs_pos, self.trajectory, t, c0=c0)
        y_tr = self.trajectory(t_r)
        r_vec = np.asarray(obs_pos) - y_tr
        r = np.linalg.norm(r_vec)
        if r < 1e-15:
            return 0.0
        r_hat = r_vec / r
        d_dir = self.dipole_direction(t_r)
        cos_theta = np.dot(r_hat, d_dir)
        tau = t_r

        v = self.velocity(t_r)
        M_r = np.dot(v, r_hat) / c0
        doppler = 1.0 / (1.0 - M_r) if abs(1.0 - M_r) > 1e-10 else 1.0

        df_dt = self.force_amplitude * self.omega * math.cos(self.omega * tau)
        far = cos_theta * df_dt / (4 * math.pi * c0 * r) * doppler

        f_val = self.force_amplitude * math.sin(self.omega * tau)
        near = cos_theta * f_val / (4 * math.pi * r * r)

        return far + near


class PulsatingSphere(AnalyticalSource):
    """Pulsating sphere – exact analytical solution.

    For a sphere of radius a pulsating with radial velocity
    u_r(a, t) = U₀ sin(ωt), the far-field pressure is:

    p'(r, t) = (ρ₀ c₀ k a² U₀ / r) · cos(ω(t - (r-a)/c₀)) / sqrt(1 + (ka)²)

    where k = ω/c₀ is the wavenumber.
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        velocity_amplitude: float,
        frequency: float,
        rho0: float = 1.225,
        c0: float = 343.0,
    ):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.velocity_amplitude = velocity_amplitude
        self.frequency = frequency
        self.rho0 = rho0
        self.c0 = c0
        self.omega = _TWO_PI * frequency
        self.k = self.omega / c0

    def pressure_at(self, obs_pos: np.ndarray, t: float, c0: float = 343.0) -> float:
        r_vec = np.asarray(obs_pos) - self.center
        r = np.linalg.norm(r_vec)
        if r <= self.radius:
            return 0.0
        # Effective distance from sphere surface
        r_eff = r - self.radius
        tau = t - r_eff / c0
        ka = self.k * self.radius
        amplitude = (
            self.rho0 * c0 * self.k * self.radius * self.radius * self.velocity_amplitude
            / r
            / math.sqrt(1 + (ka * ka))
        )
        return amplitude * math.cos(self.omega * tau)

    def trajectory(self, t: float) -> np.ndarray:
        return self.center.copy()


# ---------------------------------------------------------------------------
# Doppler frequency shift
# ---------------------------------------------------------------------------


def compute_doppler_shift(
    f_src: float,
    v_source: np.ndarray,
    obs_pos: np.ndarray,
    source_pos: np.ndarray,
    c0: float = 343.0,
) -> float:
    """Compute observed frequency with Doppler shift.

    f_obs = f_src · c₀ / (c₀ - v·cosθ)

    where cosθ = (v̂ · r̂) and r̂ points from source to observer.

    Parameters
    ----------
    f_src : float
        Source frequency (Hz).
    v_source : np.ndarray, shape (3,)
        Source velocity vector.
    obs_pos : np.ndarray, shape (3,)
        Observer position.
    source_pos : np.ndarray, shape (3,)
        Source position.
    c0 : float
        Speed of sound.

    Returns
    -------
    f_obs : float
        Observed frequency (Hz).
    """
    r_vec = np.asarray(obs_pos) - np.asarray(source_pos)
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-15:
        return f_src

    r_hat = r_vec / r_mag
    v_mag = np.linalg.norm(v_source)
    if v_mag < 1e-15:
        return f_src

    v_hat = v_source / v_mag
    cos_theta = np.dot(v_hat, r_hat)

    denom = c0 - v_mag * cos_theta
    if abs(denom) < 1e-10:
        return f_src  # singularity protection

    return f_src * c0 / denom


# ---------------------------------------------------------------------------
# BPF (Blade Passing Frequency) analysis
# ---------------------------------------------------------------------------


@dataclass
class BPFPeak:
    """A detected BPF harmonic peak."""

    harmonic: int  # 1 = fundamental BPF, 2 = 2×BPF, etc.
    frequency: float  # Hz
    amplitude: float  # Pa (RMS)
    spl_db: float  # dB re 20 µPa


def compute_bpf_peaks(
    p_signal: np.ndarray,
    dt: float,
    rpm: float,
    n_blades: int,
    max_harmonic: int = 8,
) -> list[BPFPeak]:
    """Identify BPF and its harmonics in an acoustic pressure signal.

    Parameters
    ----------
    p_signal : np.ndarray, shape (T,)
        Acoustic pressure time series.
    dt : float
        Time step (seconds).
    rpm : float
        Rotational speed (revolutions per minute).
    n_blades : int
        Number of blades.
    max_harmonic : int
        Maximum harmonic number to search.

    Returns
    -------
    peaks : list[BPFPeak]
        Detected BPF peaks.
    """
    bpf = n_blades * rpm / 60.0  # fundamental BPF in Hz
    T = len(p_signal)
    fs = 1.0 / dt

    # FFT
    window = np.hanning(T)
    fft_vals = np.fft.rfft(p_signal * window)
    freqs = np.fft.rfftfreq(T, d=dt)
    psd = np.abs(fft_vals) ** 2 / (T * T)
    psd[1:-1] *= 2.0  # one-sided

    peaks = []
    for h in range(1, max_harmonic + 1):
        target_freq = h * bpf
        # Find nearest frequency bin
        idx = np.argmin(np.abs(freqs - target_freq))
        # Search in a narrow window around target (±1 frequency bin)
        search_range = max(1, int(0.5 * bpf / (fs / T)))
        lo = max(0, idx - search_range)
        hi = min(len(freqs), idx + search_range + 1)
        local_max_idx = lo + np.argmax(psd[lo:hi])

        # RMS amplitude from PSD
        df = fs / T
        amp_rms = math.sqrt(psd[local_max_idx] * df)
        spl = 20.0 * math.log10(max(amp_rms, 1e-30) / _REF_PRESSURE)

        peaks.append(BPFPeak(
            harmonic=h,
            frequency=float(freqs[local_max_idx]),
            amplitude=float(amp_rms),
            spl_db=float(spl),
        ))

    return peaks


# ---------------------------------------------------------------------------
# Rotating FWH integration (multi-source, retarded time)
# ---------------------------------------------------------------------------


@dataclass
class RotatingFWHConfig:
    """Configuration for rotating FWH computation."""

    n_sources: int
    dt: float
    c0: float = 343.0
    rho0: float = 1.225
    window_size: int = 1024
    newton_max_iter: int = 10
    newton_tol: float = 1e-10


def compute_rotating_fwh(
    source_trajectories: list,  # list of callable y(t) -> np.ndarray (3,)
    source_pressures: list,  # list of callable p(t) -> float
    source_velocities: list | None,  # list of callable v(t) -> np.ndarray (3,)
    obs_positions: list[np.ndarray],
    t_start: float,
    t_end: float,
    dt: float,
    c0: float = 343.0,
    rho0: float = 1.225,
    window_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute far-field pressure for rotating sources with retarded time.

    Parameters
    ----------
    source_trajectories : list of callable
        Each callable returns source position y(t) as np.ndarray (3,).
    source_pressures : list of callable
        Each callable returns source pressure p(t) as float.
    source_velocities : list of callable or None
        Each callable returns source velocity v(t) as np.ndarray (3,).
    obs_positions : list of np.ndarray
        Observer positions, each shape (3,).
    t_start : float
        Start time.
    t_end : float
        End time.
    dt : float
        Time step.
    c0 : float
        Speed of sound.
    rho0 : float
        Ambient density.
    window_size : int
        Sliding window size.

    Returns
    -------
    p_prime : np.ndarray, shape (n_obs, n_steps)
        Far-field pressure time series.
    times : np.ndarray, shape (n_steps,)
        Observation times.
    """
    n_src = len(source_trajectories)
    n_obs = len(obs_positions)
    n_steps = int(round((t_end - t_start) / dt)) + 1
    times = np.linspace(t_start, t_end, n_steps)

    p_prime = np.zeros((n_obs, n_steps))
    solver = RetardedTimeSolver(max_iter=10, tol=1e-10)

    # Warm-start: previous retarded times
    t_r_prev = np.full((n_obs, n_src), np.nan)

    for step in range(n_steps):
        t_obs = t_start + step * dt

        for oi, obs_pos in enumerate(obs_positions):
            for si in range(n_src):
                traj = source_trajectories[si]
                p_src = source_pressures[si]

                # Guess from previous step
                guess = None
                if not np.isnan(t_r_prev[oi, si]):
                    guess = t_r_prev[oi, si]

                t_r, residual = solver.solve(obs_pos, traj, t_obs, t_r_guess=guess, c0=c0)
                t_r_prev[oi, si] = t_r

                y_tr = traj(t_r)
                r_vec = obs_pos - y_tr
                r = np.linalg.norm(r_vec)
                if r < 1e-15:
                    continue

                r_hat = r_vec / r

                # Source pressure at retarded time
                p_val = p_src(t_r)

                # Time derivative via finite difference (using previous step)
                if step > 0:
                    t_r_prev_step = t_r_prev[oi, si] if step > 1 else t_r - dt
                    p_prev = p_src(t_r_prev_step)
                    dp_dt = (p_val - p_prev) / max(dt, 1e-15)
                else:
                    dp_dt = 0.0

                # Doppler factor
                if source_velocities is not None and source_velocities[si] is not None:
                    v = source_velocities[si](t_r)
                    M_r = np.dot(v, r_hat) / c0
                    doppler = 1.0 / (1.0 - M_r) if abs(1.0 - M_r) > 1e-10 else 1.0
                else:
                    doppler = 1.0

                # FWH loading term (dipole)
                # p' = (1/4πc₀) · (dp/dt)/r · doppler + (1/4π) · p/r²
                p_prime[oi, step] += (
                    dp_dt / (4 * math.pi * c0 * r) * doppler
                    + p_val / (4 * math.pi * r * r)
                )

    return p_prime, times


# ---------------------------------------------------------------------------
# Multi-source analytical pressure computation (vectorised)
# ---------------------------------------------------------------------------


def compute_analytical_pressure(
    sources: list[AnalyticalSource],
    obs_pos: np.ndarray,
    times: np.ndarray,
    c0: float = 343.0,
) -> np.ndarray:
    """Compute total pressure from multiple analytical sources.

    Parameters
    ----------
    sources : list of AnalyticalSource
        Analytical source models.
    obs_pos : np.ndarray, shape (3,)
        Observer position.
    times : np.ndarray, shape (T,)
        Observation times.
    c0 : float
        Speed of sound.

    Returns
    -------
    p_total : np.ndarray, shape (T,)
        Total acoustic pressure time series.
    """
    p_total = np.zeros(len(times))
    for src in sources:
        for i, t in enumerate(times):
            p_total[i] += src.pressure_at(obs_pos, t, c0=c0)
    return p_total
