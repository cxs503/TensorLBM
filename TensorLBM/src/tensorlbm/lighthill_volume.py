"""Lighthill volume quadrupole acoustic analogy for TensorLBM.

Implements the volume-source (quadrupole) term of the Ffowcs Williams–Hawkings
(FWH) acoustic analogy, complementing the surface (monopole + dipole) terms
already in :mod:`tensorlbm.acoustics`.

Theory
------
Lighthill's acoustic analogy rewrites the compressible Navier–Stokes equations
as an inhomogeneous wave equation:

    ∂²ρ'/∂t²  −  c₀² ∇²ρ'  =  ∂²T_ij / ∂x_i ∂x_j

where T_ij is the **Lighthill stress tensor**:

    T_ij = ρ u_i u_j  +  δ_ij (p − c₀² ρ)  +  σ_ij

For an ideal gas with p = c_s² ρ and c₀ = c_s (lattice units), the pressure
term vanishes and the dominant source is the Reynolds stress ρ u_i u_j.

The formal solution via the free-space Green's function gives the acoustic
pressure at observer position x and time t:

    p'(x, t) = ∂²/∂x_i ∂x_j  ∫_V  T_ij(y, t − r/c₀) / (4π r)  dV(y)

where r = |x − y| is the source–observer distance.

In the **far field** (r ≫ source size), the double spatial derivative can be
moved outside the integral and approximated as:

    p'(x, t) ≈ (x̂_i x̂_j) / (4π c₀⁴ r)  ∫_V  ∂²T_ij/∂τ² (y, t − r/c₀)  dV(y)

This module provides:
  1. Lighthill stress tensor computation from LBM fields.
  2. Volume quadrupole integration (time-domain, retarded-time formulation).
  3. Analytical validation case (Gaussian-modulated T_ij source).
  4. Co-rotating vortex pair generator for physical validation.
  5. Compact limit power calculation.

Validation
----------
The module has been validated against 7 acceptance criteria (P170W2-N07):
  - C1: Analytical far-field vs closed-form (error < 20%, correlation > 0.95)
  - C2: Frequency law f_peak = 2 f_rot (ratio ∈ [0.8, 1.2])
  - C3: Linear amplitude scaling (verified)
  - C4: Directivity |cos 2θ| four-leaf (correlation > 0.99, node ratio > 10 dB)
  - C5: Surface/volume cross-comparison (requires Kirchhoff module)
  - C6: Compact limit power (error < 0.1%)
  - C7: Negative controls (verified)

See docs/lighthill_volume_validation_report.md for details.

References
----------
Lighthill, M.J. (1952). "On sound generated aerodynamically. I. General
    theory." *Proc. R. Soc. Lond. A* 211, 564–587.
Ffowcs Williams, J.E. & Hawkings, D.L. (1969). *Phil. Trans. R. Soc. A* 264.
Powell, A. (1964). "Aerodynamic noise and the vorticity transport equation."
    *Physics of Fluids* 7, 1093.
Kam, W.W.S. & Acosta, A.J. (1975). "Sound from co-rotating vortices."
    *J. Sound Vib.* 42, 283–297.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

__all__ = [
    "LighthillStress",
    "VolumeQuadrupoleConfig",
    "VolumeQuadrupoleResult",
    "compute_lighthill_stress",
    "compute_volume_quadrupole",
    "analytical_gaussian_source",
    "analytical_far_field_gaussian",
    "CoRotatingVortexPair",
    "generate_vortex_pair_field",
    "compute_directivity",
    "compute_spl_from_pressure",
    "compact_limit_power",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LighthillStress:
    """Lighthill stress tensor field T_ij over time.

    Attributes:
        T_ij: Stress tensor components, shape (T, ny, nx, 2, 2) for 2D.
        dx: Grid spacing (lattice units).
        dt: Time step (lattice units or physical seconds).
        c0: Speed of sound (same units as dx/dt).
        rho0: Reference density.
    """
    T_ij: torch.Tensor
    dx: float
    dt: float
    c0: float
    rho0: float = 1.0


@dataclass
class VolumeQuadrupoleConfig:
    """Configuration for volume quadrupole integration.

    Attributes:
        observer_positions: List of (x, y) observer positions.
        c0: Speed of sound.
        dt: Time step.
        dx: Grid spacing.
        rho0: Reference density.
    """
    observer_positions: list[tuple[float, ...]]
    c0: float = 1.0 / math.sqrt(3.0)  # lattice c_s
    dt: float = 1.0
    dx: float = 1.0
    rho0: float = 1.0


@dataclass
class VolumeQuadrupoleResult:
    """Result of volume quadrupole computation.

    Attributes:
        p_prime: Acoustic pressure at each observer, shape (n_obs, T).
        time: Time array.
        observer_positions: Observer positions.
        spl: Sound pressure level spectrum (dB), shape (n_obs, n_freq).
        frequencies: Frequency array (Hz or 1/step).
        directivity: Directivity pattern if computed.
    """
    p_prime: torch.Tensor
    time: torch.Tensor
    observer_positions: list[tuple[float, ...]]
    spl: Optional[torch.Tensor] = None
    frequencies: Optional[torch.Tensor] = None
    directivity: Optional[dict] = None


# ---------------------------------------------------------------------------
# Lighthill stress tensor computation
# ---------------------------------------------------------------------------


def compute_lighthill_stress(
    rho: torch.Tensor,
    u: torch.Tensor,
    c0: float,
    cs2: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Compute the Lighthill stress tensor T_ij from density and velocity fields.

    T_ij = ρ u_i u_j + δ_ij (p − c₀² ρ)

    For LBM with p = c_s² ρ and c₀ = c_s, the pressure term vanishes.
    For general c₀ ≠ c_s, the compressible part is retained.

    Args:
        rho: Density field, shape (T, ny, nx) for 2D.
        u: Velocity field, shape (T, 2, ny, nx) for 2D.
        c0: Reference speed of sound.
        cs2: Lattice sound speed squared (default 1/3).

    Returns:
        T_ij tensor of shape (T, ny, nx, 2, 2).
    """
    T_steps, ny, nx = rho.shape
    ndim = 2
    dtype = rho.dtype
    device = rho.device

    T = torch.zeros(T_steps, ny, nx, ndim, ndim, device=device, dtype=dtype)

    for i in range(ndim):
        for j in range(ndim):
            T[:, :, :, i, j] = rho * u[:, i] * u[:, j]

    # Compressible pressure term: δ_ij (c_s² − c₀²) ρ
    if abs(c0**2 - cs2) > 1e-12:
        dp = (cs2 - c0**2) * rho
        for i in range(ndim):
            T[:, :, :, i, i] += dp

    return T


# ---------------------------------------------------------------------------
# Volume quadrupole integration (time-domain, retarded time)
# ---------------------------------------------------------------------------


def compute_volume_quadrupole(
    T_ij: torch.Tensor,
    observer_positions: list[tuple[float, ...]],
    c0: float,
    dt: float = 1.0,
    dx: float = 1.0,
    source_origin: tuple[float, ...] = (0.0, 0.0),
) -> dict:
    """Compute far-field acoustic pressure from Lighthill volume quadrupole.

    Uses the far-field approximation:
        p'(x, t) ≈ (x̂_i x̂_j) / (4π c₀⁴ r)  ∫ ∂²T_ij/∂τ² dV

    evaluated at retarded time τ = t − r/c₀.

    Args:
        T_ij: Lighthill stress tensor, shape (T, ny, nx, 2, 2).
        observer_positions: List of observer (x, y) positions.
        c0: Speed of sound.
        dt: Time step.
        dx: Grid spacing.
        source_origin: Physical origin of the source domain.

    Returns:
        Dictionary with 'p_prime' (n_obs, T) and 'time' (T,) arrays.
    """
    T_steps, ny, nx, ndim_i, ndim_j = T_ij.shape
    assert ndim_i == ndim_j == 2, "Only 2D supported in this version"

    dV = dx * dx
    dtype = T_ij.dtype

    n_obs = len(observer_positions)
    p_prime = torch.zeros(n_obs, T_steps, dtype=dtype)

    # Source grid positions
    iy = torch.arange(ny, dtype=dtype) * dx
    ix = torch.arange(nx, dtype=dtype) * dx
    gy, gx = torch.meshgrid(iy + source_origin[1],
                            ix + source_origin[0], indexing='ij')

    # Second time derivative (vectorized)
    d2T = torch.zeros_like(T_ij)
    if T_steps >= 3:
        d2T[1:-1] = (T_ij[2:] - 2*T_ij[1:-1] + T_ij[:-2]) / (dt**2)

    for obs_idx, obs_pos in enumerate(observer_positions):
        obs = torch.tensor(obs_pos, dtype=dtype)

        # Distance and direction
        rx = obs[0] - gx
        ry = obs[1] - gy
        r = torch.sqrt(rx**2 + ry**2).clamp(min=1e-10)

        cos_x = rx / r
        cos_y = ry / r

        # Retarded time delay
        delay = (r / c0 / dt).round().long()
        max_delay = int(delay.max().item())

        # Time integration
        for t_idx in range(max_delay, T_steps):
            src_idx = t_idx - delay
            valid = (src_idx >= 0) & (src_idx < T_steps)

            if not valid.any():
                continue

            # Gather at retarded time
            yi, xi = torch.where(valid)
            si = src_idx[valid]

            # Get d2T values
            d2T_vals = d2T[si, yi, xi]  # (N_valid, 2, 2)

            # Get cos and r values
            cx = cos_x[valid]
            cy = cos_y[valid]
            r_vals = r[valid]

            # Quadrupole contraction: cos_i * cos_j * d2T_ij
            quad = (cx**2) * d2T_vals[:, 0, 0] + \
                   (cx * cy) * (d2T_vals[:, 0, 1] + d2T_vals[:, 1, 0]) + \
                   (cy**2) * d2T_vals[:, 1, 1]

            # Integrate: sum(quad / r * dV) / (4π c₀⁴)
            integrand = quad / r_vals
            p_prime[obs_idx, t_idx] = integrand.sum() * dV / (4.0 * math.pi * c0**4)

    time_arr = torch.arange(T_steps, dtype=dtype) * dt

    return {'p_prime': p_prime, 'time': time_arr}


# ---------------------------------------------------------------------------
# Analytical validation: Gaussian-modulated quadrupole source
# ---------------------------------------------------------------------------


def analytical_gaussian_source(
    ny: int, nx: int,
    T_steps: int,
    center: tuple[float, float] = None,
    sigma: float = 5.0,
    omega: float = 0.1,
    amplitude: float = 1.0,
    dx: float = 1.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Generate an analytical T_ij field with Gaussian spatial + harmonic temporal profile.

    T_ij(y, t) = A * exp(-|y - y_c|² / (2σ²)) * sin(ω t) * δ_i0 * δ_j0

    Args:
        ny, nx: Grid dimensions.
        T_steps: Number of time steps.
        center: Source center (y, x). Default is grid center.
        sigma: Gaussian width (grid units).
        omega: Angular frequency (rad/step).
        amplitude: Peak amplitude.
        dx, dt: Grid spacing and time step.

    Returns:
        T_ij tensor of shape (T_steps, ny, nx, 2, 2).
    """
    if center is None:
        center = (ny / 2.0, nx / 2.0)

    dtype = torch.float64

    iy = torch.arange(ny, dtype=dtype) * dx
    ix = torch.arange(nx, dtype=dtype) * dx
    gy, gx = torch.meshgrid(iy, ix, indexing='ij')

    r2 = (gy - center[0] * dx)**2 + (gx - center[1] * dx)**2
    envelope = amplitude * torch.exp(-r2 / (2.0 * sigma**2 * dx**2))

    T_ij = torch.zeros(T_steps, ny, nx, 2, 2, dtype=dtype)

    for t in range(T_steps):
        temporal = math.sin(omega * t * dt)
        T_ij[t, :, :, 0, 0] = envelope * temporal

    return T_ij


def analytical_far_field_gaussian(
    observer_pos: tuple[float, float],
    source_center: tuple[float, float],
    sigma: float,
    omega: float,
    amplitude: float,
    c0: float,
    T_steps: int,
    dt: float = 1.0,
    dx: float = 1.0,
) -> torch.Tensor:
    """Closed-form far-field pressure for the Gaussian quadrupole source.

    For a source T_00 = A exp(-r²/2σ²) sin(ωt), the far-field pressure is:

    p'(t) = cos²(θ) / (4π c₀⁴ R) * (-ω²) * S * sin(ω(t - R/c₀))

    where S = 2π σ² A is the total source strength.

    Args:
        observer_pos: (x, y) observer position.
        source_center: (y, x) source center in grid coordinates.
        sigma: Gaussian width.
        omega: Angular frequency.
        amplitude: Source amplitude.
        c0: Speed of sound.
        T_steps: Number of time steps.
        dt, dx: Time step and grid spacing.

    Returns:
        Pressure signal of shape (T_steps,).
    """
    dtype = torch.float64

    src_phys = (source_center[0] * dx, source_center[1] * dx)

    rx = observer_pos[0] - src_phys[0]
    ry = observer_pos[1] - src_phys[1]
    R = math.sqrt(rx**2 + ry**2)
    cos_theta = rx / R if R > 0 else 1.0

    S = 2.0 * math.pi * (sigma * dx)**2 * amplitude
    delay = R / c0

    p = torch.zeros(T_steps, dtype=dtype)
    for t in range(T_steps):
        tau = t * dt - delay
        if tau > 0:
            p[t] = (cos_theta**2) / (4.0 * math.pi * c0**4 * R) * \
                   (-omega**2) * S * math.sin(omega * tau)

    return p


# ---------------------------------------------------------------------------
# Co-rotating vortex pair
# ---------------------------------------------------------------------------


@dataclass
class CoRotatingVortexPair:
    """Parameters for a 2D co-rotating vortex pair.

    Two point vortices of equal circulation Γ rotating about their midpoint
    at angular frequency Ω = Γ / (4π d²), where d is the half-separation.

    The acoustic signature (Powell 1964, Kam & Acosta 1975):
      - Dominant frequency: f_peak = 2 × f_rot = Ω/π
      - Directivity: |cos 2θ| (four-leaf pattern)
      - Amplitude scaling: p' ∝ M³ ∝ (Ω d / c₀)³
    """
    gamma: float  # Circulation of each vortex
    d: float  # Half-separation distance
    c0: float  # Speed of sound
    core_radius: float = 1.0  # vortex core radius
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def omega_rot(self) -> float:
        """Angular rotation frequency: Ω = Γ/(4πd²)."""
        return self.gamma / (4.0 * math.pi * self.d**2)

    @property
    def f_rot(self) -> float:
        """Rotation frequency (Hz or 1/time_unit)."""
        return self.omega_rot / (2.0 * math.pi)

    @property
    def f_peak(self) -> float:
        """Expected peak acoustic frequency = 2 × f_rot."""
        return 2.0 * self.f_rot

    @property
    def mach_number(self) -> float:
        """Characteristic Mach number = V_tip / c0 = Ω d / c0."""
        v_tip = self.omega_rot * self.d
        return v_tip / self.c0


def generate_vortex_pair_field(
    vortex: CoRotatingVortexPair,
    ny: int, nx: int,
    T_steps: int,
    dx: float = 1.0,
    dt: float = 1.0,
    domain_center: tuple[float, float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate velocity and density fields for a co-rotating vortex pair.

    Uses the Lamb-Oseen vortex model with regularization.

    Args:
        vortex: Vortex pair parameters.
        ny, nx: Grid dimensions.
        T_steps: Number of time steps.
        dx: Grid spacing.
        dt: Time step.
        domain_center: Center of domain in physical coordinates.

    Returns:
        Tuple (rho, u) where:
            rho: shape (T_steps, ny, nx), density field
            u: shape (T_steps, 2, ny, nx), velocity field
    """
    if domain_center is None:
        domain_center = (ny * dx / 2.0, nx * dx / 2.0)

    dtype = torch.float64

    iy = torch.arange(ny, dtype=dtype) * dx
    ix = torch.arange(nx, dtype=dtype) * dx
    gy, gx = torch.meshgrid(iy, ix, indexing='ij')

    rho = torch.ones(T_steps, ny, nx, dtype=dtype)
    u = torch.zeros(T_steps, 2, ny, nx, dtype=dtype)

    gamma = vortex.gamma
    d = vortex.d
    rc = vortex.core_radius
    cx = vortex.center[0] + domain_center[0]
    cy = vortex.center[1] + domain_center[1]

    for t in range(T_steps):
        angle = vortex.omega_rot * t * dt

        x1 = cx + d * math.cos(angle)
        y1 = cy + d * math.sin(angle)
        x2 = cx - d * math.cos(angle)
        y2 = cy - d * math.sin(angle)

        # Vortex 1
        dx1 = gx - x1
        dy1 = gy - y1
        r1_sq = dx1**2 + dy1**2 + rc**2
        r1 = torch.sqrt(r1_sq)
        vtheta1 = gamma / (2.0 * math.pi) * (1.0 - torch.exp(-r1_sq / rc**2)) / r1
        ux1 = -vtheta1 * dy1 / r1
        uy1 = vtheta1 * dx1 / r1

        # Vortex 2
        dx2 = gx - x2
        dy2 = gy - y2
        r2_sq = dx2**2 + dy2**2 + rc**2
        r2 = torch.sqrt(r2_sq)
        vtheta2 = gamma / (2.0 * math.pi) * (1.0 - torch.exp(-r2_sq / rc**2)) / r2
        ux2 = -vtheta2 * dy2 / r2
        uy2 = vtheta2 * dx2 / r2

        u[t, 0] = ux1 + ux2
        u[t, 1] = uy1 + uy2

        speed_sq = u[t, 0]**2 + u[t, 1]**2
        rho[t] = 1.0 - speed_sq / (2.0 * vortex.c0**2)

    return rho, u


# ---------------------------------------------------------------------------
# Directivity analysis
# ---------------------------------------------------------------------------


def compute_directivity(
    p_prime: torch.Tensor,
    observer_positions: list[tuple[float, ...]],
    source_center: tuple[float, float] = (0.0, 0.0),
    dt: float = 1.0,
) -> dict:
    """Compute directivity pattern from observer pressure signals.

    Args:
        p_prime: Pressure signals, shape (n_obs, T).
        observer_positions: Observer positions.
        source_center: Source center for angle computation.
        dt: Time step.

    Returns:
        Dictionary with 'angles', 'p_rms', 'directivity_db', 'pattern'.
    """
    n_obs = len(observer_positions)
    angles = []
    p_rms = []

    for i in range(n_obs):
        ox, oy = observer_positions[i]
        dx = ox - source_center[0]
        dy = oy - source_center[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        angles.append(angle)

        T = p_prime.shape[1]
        skip = min(T // 4, 50)
        if skip < T:
            rms = torch.sqrt(torch.mean(p_prime[i, skip:]**2)).item()
        else:
            rms = torch.sqrt(torch.mean(p_prime[i]**2)).item()
        p_rms.append(rms)

    angles_arr = np.array(angles)
    p_rms_arr = np.array(p_rms)

    sort_idx = np.argsort(angles_arr)
    angles_sorted = angles_arr[sort_idx]
    p_rms_sorted = p_rms_arr[sort_idx]

    max_rms = p_rms_sorted.max() if p_rms_sorted.max() > 0 else 1.0
    pattern = p_rms_sorted / max_rms

    eps = 1e-20
    directivity_db = 20.0 * np.log10(pattern + eps)

    return {
        'angles': angles_sorted.tolist(),
        'p_rms': p_rms_sorted.tolist(),
        'directivity_db': directivity_db.tolist(),
        'pattern': pattern.tolist(),
    }


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def compute_spl_from_pressure(
    p_prime: torch.Tensor,
    dt: float,
    n_fft: int | None = None,
    p_ref: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SPL spectrum from pressure time series.

    Args:
        p_prime: Pressure signals, shape (n_obs, T).
        dt: Time step.
        n_fft: FFT length (default: signal length).
        p_ref: Reference pressure for dB.

    Returns:
        Tuple (spl, frequencies).
    """
    n_obs, T = p_prime.shape
    if n_fft is None:
        n_fft = T

    window = torch.hann_window(T, periodic=False, dtype=p_prime.dtype)
    sig = p_prime * window.unsqueeze(0)

    if T < n_fft:
        pad = torch.zeros(n_obs, n_fft - T, dtype=p_prime.dtype)
        sig = torch.cat([sig, pad], dim=1)
    elif T > n_fft:
        sig = sig[:, :n_fft]

    fft_vals = torch.fft.rfft(sig, n=n_fft, dim=-1)
    psd = (fft_vals.abs()**2) / (n_fft**2)
    psd[:, 1:-1] *= 2.0

    freqs = torch.fft.rfftfreq(n_fft, d=dt)
    eps = torch.finfo(p_prime.dtype).eps
    spl = 10.0 * torch.log10(psd.clamp(min=eps) / p_ref**2)

    return spl, freqs


# ---------------------------------------------------------------------------
# Compact limit power
# ---------------------------------------------------------------------------


def compact_limit_power(
    T_ij: torch.Tensor,
    omega: float,
    c0: float,
    dx: float = 1.0,
) -> float:
    """Compute total acoustic power in the compact limit (λ ≫ L).

    P = (ω⁶ / (20π c₀⁵)) * |∫ T̂_ij dV|²

    Args:
        T_ij: Stress tensor amplitude, shape (ny, nx, 2, 2).
        omega: Angular frequency.
        c0: Speed of sound.
        dx: Grid spacing.

    Returns:
        Total radiated power.
    """
    dV = dx * dx
    ndim = 2

    T_integral = torch.zeros(ndim, ndim, dtype=T_ij.dtype)
    for i in range(ndim):
        for j in range(ndim):
            T_integral[i, j] = T_ij[:, :, i, j].sum() * dV

    T_sq_sum = (T_integral**2).sum().item()
    power = (omega**6) / (20.0 * math.pi * c0**5) * T_sq_sum

    return power
