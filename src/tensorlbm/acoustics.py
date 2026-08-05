"""Aeroacoustic post-processing using the Ffowcs Williams–Hawkings (FWH) analogy.

Computes far-field acoustic pressure from near-field surface data extracted
from an LBM simulation.  This implements the *porous FWH* formulation
(Farassat Formulation 1A), which evaluates monopole (thickness) and dipole
(loading) source contributions from a closed control surface surrounding the
aerodynamic body.

Background
----------
The FWH acoustic analogy (Ffowcs Williams & Hawkings 1969) decomposes the
far-field sound into:

*  **Thickness noise** (monopole) – due to fluid displaced by a moving body.
*  **Loading noise** (dipole) – due to surface pressure and viscous stress.
*  **Quadrupole noise** – volume sources (neglected for subsonic flows).

For *stationary* surfaces (typical in ground-based CFD), the porous FWH
formula reduces to an integration of surface pressure fluctuations p′ and
normal velocity fluctuations u_n over the control surface:

.. math::

    p'(x, t) ≈ \\frac{1}{4π} ∫_S \\left[
        \\frac{ṗ'}{c_0 r} + \\frac{p'}{r^2}
    \\right]_{τ} \\hat{r} · \\hat{n} \\, dS

where r = |x − y| is the emission distance, τ is retarded time, c_0 is the
speed of sound, and n̂ is the outward surface normal.

This module provides a *simplified far-field* approximation suitable for
engineering noise estimates from LBM simulations:

1. Probe pressure time-series on a set of surface points (or use existing
   `forces.csv` data) to form the monopole source signal p′(y, t).
2. Apply the FWH integration for a stationary compact or distributed source.
3. Return the far-field Sound Pressure Level (SPL) spectrum.

References
----------
Ffowcs Williams, J.E. & Hawkings, D.L. (1969).
    "Sound generation by turbulence and surfaces in arbitrary motion."
    *Phil. Trans. R. Soc. Lond. A* 264, 321–342.
Farassat, F. & Succi, G.P. (1980).
    "The prediction of helicopter rotor discrete frequency noise."
    *Vertica* 4, 305–308.
Brentner, K.S. & Farassat, F. (1998).
    "Modeling aerodynamically generated sound of helicopter rotors."
    *Prog. Aerospace Sci.* 34, 67–120.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "AcousticObserver",
    "FWHSurface",
    "FWHResult",
    "compute_fwh_far_field",
    "compute_spl_spectrum",
    "extract_surface_pressure",
]

_TWO_PI = 2.0 * math.pi
_REF_PRESSURE = 20.0e-6  # 20 µPa – acoustic reference pressure in Pa


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AcousticObserver:
    """Far-field observer position.

    Args:
        x: Observer x-coordinate (physical units, e.g. metres).
        y: Observer y-coordinate.
        z: Observer z-coordinate (use 0 for 2-D problems).
        label: Optional human-readable label.
    """
    x: float
    y: float
    z: float = 0.0
    label: str = ""


@dataclass
class FWHSurface:
    """Porous FWH control surface defined by discrete source points.

    Args:
        positions: Source-point coordinates, shape ``(N, 3)``.  For 2-D LBM
            the z-component should be zero.
        normals: Outward unit normals at each source point, shape ``(N, 3)``.
        areas: Area (or line length for 2-D) associated with each source
            point, shape ``(N,)``.
        pressure: Pressure time-series at each point, shape ``(N, T)`` where
            T is the number of time steps.
        dt: Physical time step (seconds).
        c0: Speed of sound (m/s; default 343 for air at 20 °C).
    """
    positions: torch.Tensor
    normals: torch.Tensor
    areas: torch.Tensor
    pressure: torch.Tensor
    dt: float
    c0: float = 343.0


@dataclass
class FWHResult:
    """Container for FWH far-field acoustic results.

    Attributes:
        time: Physical time array (length T), seconds.
        p_prime: Far-field acoustic pressure time series at each observer
            (shape ``(n_observers, T)``), in Pascals.
        frequencies: Frequency bins from the FFT spectrum (Hz).
        spl: Sound Pressure Level spectrum at each observer
            (shape ``(n_observers, n_freq)``), in dB re 20 µPa.
        oaspl: Overall A-weighted SPL for each observer (dB).
        observers: List of :class:`AcousticObserver` objects.
    """
    time: list[float]

    p_prime: torch.Tensor
    frequencies: list[float]
    spl: torch.Tensor
    oaspl: list[float]
    observers: list[AcousticObserver]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _retarded_time_index(
    distance: float,
    c0: float,
    dt: float,
    t_idx: int,
    n_steps: int,
) -> int:
    """Compute the retarded-time sample index for a given emission distance.

    Returns the index clamped to ``[0, n_steps - 1]``.
    """
    delay_samples = int(round(distance / (c0 * dt)))
    return max(0, min(t_idx - delay_samples, n_steps - 1))


def compute_fwh_far_field(
    surface: FWHSurface,
    observers: list[AcousticObserver],
) -> tuple[torch.Tensor, list[float]]:
    """Compute far-field acoustic pressure time series for each observer.

    Uses the compact-source, far-field FWH approximation:

    .. math::

        p'(x, t) = \\frac{1}{4π c_0}
            \\sum_{n} \\frac{\\dot{p}'_n(τ_n)}{r_n} (\\hat{r}_n · \\hat{n}_n) A_n

    where the sum is over surface source points *n*, τ_n = t − r_n/c_0 is
    the retarded time, r_n = |x − y_n| the emission distance, and Ȧ denotes
    the time derivative.

    For the near-field (r ≪ acoustic wavelength) monopole term the 1/r² term
    is included:

    .. math::

        p'_{near}(x, t) = \\frac{1}{4π}
            \\sum_{n} \\frac{p'_n(τ_n)}{r_n^2} (\\hat{r}_n · \\hat{n}_n) A_n

    The total pressure is the sum of both contributions.

    Args:
        surface: :class:`FWHSurface` containing source geometry and pressure.
        observers: List of far-field observers.

    Returns:
        Tuple ``(p_prime, time)`` where *p_prime* has shape
        ``(n_observers, T)`` and *time* is a list of physical times.
    """
    n_src = surface.positions.shape[0]
    if surface.pressure.ndim != 2 or surface.pressure.shape[0] != n_src:
        raise ValueError("pressure must have shape (N, T) matching positions")
    if surface.positions.shape != (n_src, 3) or surface.normals.shape != (n_src, 3):
        raise ValueError("positions and normals must each have shape (N, 3)")
    if surface.areas.shape != (n_src,):
        raise ValueError("areas must have shape (N,)")
    _, T = surface.pressure.shape
    if T < 2:
        raise ValueError("at least two pressure samples are required")
    dt = surface.dt
    c0 = surface.c0
    if dt <= 0.0 or c0 <= 0.0:
        raise ValueError("dt and c0 must both be positive")

    # Time derivative of surface pressure via central differences
    p = surface.pressure  # (N, T)
    dp_dt = torch.zeros_like(p)
    dp_dt[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dt)
    dp_dt[:, 0] = (p[:, 1] - p[:, 0]) / dt
    dp_dt[:, -1] = (p[:, -1] - p[:, -2]) / dt

    pos = surface.positions  # (N, 3)
    nrm = surface.normals    # (N, 3)
    area = surface.areas     # (N,)

    p_prime_all = p.new_zeros((len(observers), T))
    src_idx = torch.arange(n_src, device=p.device)

    for obs_idx, obs in enumerate(observers):
        obs_pos = torch.tensor([obs.x, obs.y, obs.z], dtype=pos.dtype, device=pos.device)

        # Vector from source points to observer
        r_vec = obs_pos.unsqueeze(0) - pos          # (N, 3)
        r_dist = torch.norm(r_vec, dim=-1).clamp(min=1e-10)  # (N,)
        r_hat = r_vec / r_dist.unsqueeze(-1)         # (N, 3)

        # Directivity: r̂ · n̂
        cos_theta = (r_hat * nrm).sum(dim=-1)        # (N,)

        p_obs = p.new_zeros(T)
        delay_samples = (r_dist / (c0 * dt)).round().long()

        for t_idx in range(T):
            # No contribution exists before the propagation delay.  Clamping
            # a negative retarded index to zero copied p(t=0) into the
            # pre-arrival signal and violated causality for a nonzero source.
            valid = t_idx >= delay_samples
            if not bool(valid.any()):
                continue
            tau_indices = t_idx - delay_samples[valid]

            # Far-field (1/r) term
            dp_tau = dp_dt[src_idx[valid], tau_indices]
            far_field = (dp_tau / r_dist[valid]) * cos_theta[valid] * area[valid]

            # Near-field (1/r²) term
            p_tau = p[src_idx[valid], tau_indices]
            near_field = (p_tau / (r_dist[valid] * r_dist[valid])) * cos_theta[valid] * area[valid]

            p_obs[t_idx] = (far_field.sum() / (_TWO_PI * 2.0 * c0)
                            + near_field.sum() / (_TWO_PI * 2.0))

        p_prime_all[obs_idx] = p_obs

    time_list = [i * dt for i in range(T)]
    return p_prime_all, time_list


def compute_spl_spectrum(
    p_prime: torch.Tensor,
    dt: float,
    n_fft: int | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Compute Sound Pressure Level (SPL) spectra via Welch's method.

    For each observer, computes the single-sided power spectral density and
    converts to SPL in dB re 20 µPa.

    Args:
        p_prime: Acoustic pressure, shape ``(n_observers, T)``.
        dt: Physical time step (seconds).
        n_fft: FFT length.  Defaults to the full signal length T.

    Returns:
        Tuple ``(spl, frequencies)`` where *spl* has shape
        ``(n_observers, n_freq)`` (dB) and *frequencies* is a list of Hz
        values.
    """
    n_obs, T = p_prime.shape
    if n_fft is None:
        n_fft = T

    # Pad or truncate to n_fft
    sig = p_prime[:, :n_fft]
    if sig.shape[1] < n_fft:
        pad = torch.zeros(n_obs, n_fft - sig.shape[1])
        sig = torch.cat([sig, pad], dim=1)
    # Apply Hann window to reduce spectral leakage (window length must match sig)
    window = torch.hann_window(n_fft, periodic=False)
    sig = sig * window.unsqueeze(0)

    # Real FFT → one-sided PSD
    fft_vals = torch.fft.rfft(sig, n=n_fft, dim=-1)
    psd = (fft_vals.abs() ** 2) / (n_fft * n_fft)
    # Double non-DC / Nyquist bins to get one-sided
    psd[:, 1:-1] *= 2.0

    # Frequency axis
    freqs = torch.fft.rfftfreq(n_fft, d=dt).tolist()

    # Convert PSD to SPL: SPL = 10 log10(p_rms² / p_ref²)
    # p_rms² ≈ PSD * Δf  (but we keep as spectral density in dB/Hz here)
    eps = torch.finfo(torch.float32).eps
    spl = 10.0 * torch.log10(psd.clamp(min=eps) / (_REF_PRESSURE**2))

    return spl, freqs


def extract_surface_pressure(
    rho_history: torch.Tensor,
    surface_indices: torch.Tensor,
    surface_normals: torch.Tensor,
    surface_areas: torch.Tensor,
    dt: float = 1.0,
    c0: float = 343.0,
    physical_dx: float = 1.0,
) -> FWHSurface:
    """Build a :class:`FWHSurface` from a lattice density time-history.

    Converts lattice-unit density to pressure fluctuations via
    p′ = (ρ − ρ̄) c_s² where c_s² = 1/3 and ρ̄ is the time-averaged density.

    Args:
        rho_history: Density history, shape ``(T, ny, nx)`` (2-D) or
            ``(T, nz, ny, nx)`` (3-D).
        surface_indices: Integer indices of the surface nodes.  For 2-D:
            shape ``(N, 2)`` with columns [iy, ix].
        surface_normals: Outward unit normals at each surface node,
            shape ``(N, 3)``.  For 2-D, set z to zero.
        surface_areas: Area/length associated with each node, shape ``(N,)``.
        dt: Physical time step (seconds = lattice steps × conversion factor).
        c0: Speed of sound in the physical medium (m/s).
        physical_dx: Physical grid spacing (m/lattice unit) for 3-D position
            conversion.

    Returns:
        :class:`FWHSurface` ready for :func:`compute_fwh_far_field`.
    """
    T = rho_history.shape[0]
    N = surface_indices.shape[0]

    cs2 = 1.0 / 3.0  # lattice cs²

    # Extract density at surface nodes across all time steps
    p_history = torch.zeros(N, T)
    for n_idx in range(N):
        if surface_indices.shape[1] == 2:
            iy, ix = int(surface_indices[n_idx, 0]), int(surface_indices[n_idx, 1])
            p_history[n_idx] = rho_history[:, iy, ix] * cs2
        else:
            iz, iy, ix = (int(surface_indices[n_idx, 0]),
                          int(surface_indices[n_idx, 1]),
                          int(surface_indices[n_idx, 2]))
            p_history[n_idx] = rho_history[:, iz, iy, ix] * cs2

    # Remove mean (fluctuation only)
    p_mean = p_history.mean(dim=-1, keepdim=True)
    p_history = p_history - p_mean

    # Build physical positions from lattice indices
    if surface_indices.shape[1] == 2:
        positions = torch.zeros(N, 3)
        positions[:, 0] = surface_indices[:, 1].float() * physical_dx  # x
        positions[:, 1] = surface_indices[:, 0].float() * physical_dx  # y
    else:
        positions = torch.zeros(N, 3)
        positions[:, 0] = surface_indices[:, 2].float() * physical_dx
        positions[:, 1] = surface_indices[:, 1].float() * physical_dx
        positions[:, 2] = surface_indices[:, 0].float() * physical_dx

    return FWHSurface(
        positions=positions,
        normals=surface_normals,
        areas=surface_areas,
        pressure=p_history,
        dt=dt,
        c0=c0,
    )


# ---------------------------------------------------------------------------
# Overall A-weighted SPL
# ---------------------------------------------------------------------------

def oaspl(
    p_prime: torch.Tensor,
    dt: float,
) -> list[float]:
    """Compute Overall Sound Pressure Level (OASPL) in dB for each observer.

    OASPL = 20 log10(p_rms / p_ref)

    Args:
        p_prime: Acoustic pressure, shape ``(n_observers, T)``.
        dt: Physical time step (used for RMS normalisation; here only T
            matters since RMS is computed directly).

    Returns:
        List of OASPL values in dB re 20 µPa, one per observer.
    """
    p_rms = torch.sqrt((p_prime ** 2).mean(dim=-1))  # (n_observers,)
    eps = torch.finfo(torch.float32).eps
    oaspl_vals = 20.0 * torch.log10(p_rms.clamp(min=eps) / _REF_PRESSURE)
    return oaspl_vals.tolist()


def compute_fwh_result(
    surface: FWHSurface,
    observers: list[AcousticObserver],
    n_fft: int | None = None,
) -> FWHResult:
    """Convenience wrapper: run FWH integration and spectral analysis.

    Args:
        surface: :class:`FWHSurface` with pressure history.
        observers: Far-field observer positions.
        n_fft: FFT length for spectral analysis.

    Returns:
        :class:`FWHResult` with time-domain and frequency-domain results.
    """
    p_prime, time_list = compute_fwh_far_field(surface, observers)
    spl, freqs = compute_spl_spectrum(p_prime, surface.dt, n_fft=n_fft)
    oaspl_vals = oaspl(p_prime, surface.dt)

    return FWHResult(
        time=time_list,
        p_prime=p_prime,
        frequencies=freqs,
        spl=spl,
        oaspl=oaspl_vals,
        observers=observers,
    )
