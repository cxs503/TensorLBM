"""Lighthill volume-quadrupole acoustics for TensorLBM.

Computes far-field acoustic pressure from the Lighthill stress tensor
T_ij = ρ u_i u_j + δ_ij (p - c₀² ρ)

via the Ffowcs Williams–Hawkings volume integral (porous surface → ∞):

    p'_q(x, t) = ∂²/∂x_i ∂x_j  ∫_V  T_ij(y, τ) / (4π |x-y|)  dV(y)      (3D)

For 2-D problems the free-space Green's function in the frequency domain is
used:

    G(r, ω) = (i/4) H₀⁽¹⁾(k r),   k = ω/c₀

and the quadrupole pressure becomes

    p̂'_q(x, ω) = ∂²/∂x_i ∂x_j  ∫_V  T̂_ij(y, ω) G(|x-y|, ω)  dA(y)

The spatial derivatives of G are computed analytically using Bessel recurrences.

References
----------
Lighthill, M.J. (1952). "On sound generated aerodynamically I. General theory."
    Proc. R. Soc. Lond. A 211, 564–587.
Ffowcs Williams, J.E. & Hawkings, D.L. (1969).
    Phil. Trans. R. Soc. Lond. A 264, 321–342.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import hankel1

__all__ = [
    "LighthillTensor",
    "VolumeQuadrupoleResult",
    "compute_lighthill_tensor",
    "volume_quadrupole_frequency_domain",
    "volume_quadrupole_2d",
    "analytical_gaussian_quadrupole_farfield",
    "compact_quadrupole_power",
    "kirchhoff_surface_2d",
]

_TWO_PI = 2.0 * math.pi
_REF_PRESSURE = 20.0e-6


@dataclass
class LighthillTensor:
    """Lighthill stress tensor field on a regular grid."""
    T: np.ndarray
    dx: float
    dy: Optional[float] = None
    dz: Optional[float] = None
    origin: Tuple[float, ...] = (0.0, 0.0)
    c0: float = 1.0
    rho0: float = 1.0
    dt: float = 1.0

    def __post_init__(self):
        if self.dy is None:
            self.dy = self.dx
        if self.dz is None:
            self.dz = self.dx


@dataclass
class VolumeQuadrupoleResult:
    """Result of volume quadrupole computation."""
    p_prime: np.ndarray
    time: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None
    observers: Optional[np.ndarray] = None
    method: str = "volume_quadrupole"


# ---------------------------------------------------------------------------
# Lighthill tensor from LBM fields
# ---------------------------------------------------------------------------

def compute_lighthill_tensor(
    rho: np.ndarray,
    u: np.ndarray,
    p: Optional[np.ndarray] = None,
    c0: float = 1.0 / math.sqrt(3.0),
) -> np.ndarray:
    """Compute Lighthill stress tensor T_ij = ρ u_i u_j + δ_ij(p - c₀²ρ).

    Args:
        rho: Density field, shape ``(ny, nx)`` for 2-D.
        u: Velocity field, shape ``(d, ny, nx)`` where d=2 or 3.
        p: Optional thermodynamic pressure. If None, uses p = c₀²ρ.
        c0: Speed of sound.

    Returns:
        T_ij, shape ``(ny, nx, d, d)``.
    """
    d = u.shape[0]
    spatial = u.shape[1:]

    T = np.zeros(spatial + (d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            T[..., i, j] = rho * u[i] * u[j]

    if p is not None:
        correction = p - c0**2 * rho
        for i in range(d):
            T[..., i, i] += correction

    return T


# ---------------------------------------------------------------------------
# Vectorized 2-D Green's function second derivatives
# ---------------------------------------------------------------------------

def _green_2d_d2_vectorized(
    obs_pos: np.ndarray,
    src_pos: np.ndarray,
    omega: float,
    c0: float,
) -> np.ndarray:
    """Vectorized ∂²G/∂x_i∂x_j for all (obs, src) pairs.

    Args:
        obs_pos: ``(N_obs, 2)`` observer positions.
        src_pos: ``(N_src, 2)`` source positions.
        omega: Angular frequency.
        c0: Speed of sound.

    Returns:
        d2G: ``(N_obs, N_src, 2, 2)`` second derivatives of Green's function.
    """
    k = omega / c0
    # r_vec[o, s, :] = obs[o] - src[s]
    r_vec = obs_pos[:, None, :] - src_pos[None, :, :]  # (N_obs, N_src, 2)
    r = np.linalg.norm(r_vec, axis=-1)  # (N_obs, N_src)
    r_safe = np.maximum(r, 1e-15)
    kr = k * r_safe

    H0 = hankel1(0, kr)
    H1 = hankel1(1, kr)

    G_prime = -0.25j * k * H1  # (N_obs, N_src)
    G_double_prime = -0.25j * k**2 * (H0 - H1 / kr)  # (N_obs, N_src)

    r_hat = r_vec / r_safe[..., None]  # (N_obs, N_src, 2)

    # Outer product r̂_i r̂_j
    rr = r_hat[..., :, None] * r_hat[..., None, :]  # (N_obs, N_src, 2, 2)
    delta = np.eye(2, dtype=np.complex128)[None, None, :, :]  # (1, 1, 2, 2)

    d2G = (G_double_prime[..., None, None] * rr +
           (G_prime / r_safe)[..., None, None] * (delta - rr))

    return d2G


def _green_2d_d2_point_source(
    obs_pos: np.ndarray,
    src_center: np.ndarray,
    omega: float,
    c0: float,
) -> np.ndarray:
    """∂²G/∂x_i∂x_j for a single source point (fast path).

    Args:
        obs_pos: ``(N_obs, 2)`` observer positions.
        src_center: ``(2,)`` source center.
        omega: Angular frequency.
        c0: Speed of sound.

    Returns:
        d2G: ``(N_obs, 2, 2)`` second derivatives.
    """
    k = omega / c0
    r_vec = obs_pos - src_center[None, :]  # (N_obs, 2)
    r = np.linalg.norm(r_vec, axis=-1)
    r_safe = np.maximum(r, 1e-15)
    kr = k * r_safe

    H0 = hankel1(0, kr)
    H1 = hankel1(1, kr)

    G_prime = -0.25j * k * H1
    G_double_prime = -0.25j * k**2 * (H0 - H1 / kr)

    r_hat = r_vec / r_safe[:, None]
    rr = r_hat[:, :, None] * r_hat[:, None, :]
    delta = np.eye(2, dtype=np.complex128)[None, :, :]

    return (G_double_prime[:, None, None] * rr +
            (G_prime / r_safe)[:, None, None] * (delta - rr))


# ---------------------------------------------------------------------------
# Volume quadrupole integration (frequency domain, vectorized)
# ---------------------------------------------------------------------------

def volume_quadrupole_frequency_domain(
    T_hat: np.ndarray,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    observer_positions: np.ndarray,
    omega: float,
    c0: float,
    ndim: int = 2,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute volume quadrupole pressure in frequency domain (vectorized).

    p̂'_q(x, ω) = Σ_n Σ_{i,j} T̂_ij(y_n, ω) · ∂²G/∂x_i∂x_j (x - y_n) · w_n

    Args:
        T_hat: ``(N_src, d, d)`` Lighthill tensor in frequency domain.
        source_positions: ``(N_src, d)`` source coordinates.
        source_weights: ``(N_src,)`` quadrature weights.
        observer_positions: ``(N_obs, d)`` observer coordinates.
        omega: Angular frequency.
        c0: Speed of sound.
        ndim: 2 or 3.
        batch_size: Number of observers to process at once (memory control).

    Returns:
        Complex pressure ``(N_obs,)``.
    """
    N_obs = observer_positions.shape[0]
    p_hat = np.zeros(N_obs, dtype=np.complex128)

    # Process in batches to control memory
    for start in range(0, N_obs, batch_size):
        end = min(start + batch_size, N_obs)
        obs_batch = observer_positions[start:end]

        if ndim == 2:
            d2G = _green_2d_d2_vectorized(obs_batch, source_positions, omega, c0)
            # d2G: (batch, N_src, 2, 2)
            # T_hat: (N_src, 2, 2)
            # Contract over (i,j) and sum over sources
            contraction = np.einsum('osij,sij->os', d2G, T_hat)  # (batch, N_src)
        else:
            # 3-D: similar but with d=3
            k = omega / c0
            r_vec = obs_batch[:, None, :] - source_positions[None, :, :]
            r = np.linalg.norm(r_vec, axis=-1)
            r_safe = np.maximum(r, 1e-15)

            G = np.exp(1j * k * r_safe) / (4.0 * np.pi * r_safe)
            G_prime = (1j * k - 1.0 / r_safe) * G
            G_double_prime = (-k**2 - 2j * k / r_safe + 2.0 / r_safe**2) * G

            r_hat = r_vec / r_safe[..., None]
            rr = r_hat[..., :, None] * r_hat[..., None, :]
            delta = np.eye(3, dtype=np.complex128)[None, None, :, :]

            d2G = (G_double_prime[..., None, None] * rr +
                   (G_prime / r_safe)[..., None, None] * (delta - rr))

            contraction = np.einsum('osij,sij->os', d2G, T_hat)

        # Sum over sources with weights
        p_hat[start:end] = np.sum(contraction * source_weights[None, :], axis=1)

    return p_hat


def volume_quadrupole_2d(
    T_field: np.ndarray,
    grid_origin: Tuple[float, float] = (0.0, 0.0),
    dx: float = 1.0,
    dy: float = 1.0,
    observer_positions: Optional[np.ndarray] = None,
    observer_radius: float = 50.0,
    n_observers: int = 36,
    omega: Optional[float] = None,
    c0: float = 1.0 / math.sqrt(3.0),
    dt: float = 1.0,
) -> VolumeQuadrupoleResult:
    """Compute 2-D volume quadrupole far-field acoustics.

    Args:
        T_field: ``(T, ny, nx, 2, 2)`` or ``(ny, nx, 2, 2)`` Lighthill tensor.
        grid_origin: Physical coordinates of grid (0,0) corner.
        dx, dy: Grid spacing.
        observer_positions: ``(N, 2)`` observer positions. If None, circular array.
        observer_radius: Radius for default circular observer array.
        n_observers: Number of observers for default circular array.
        omega: Single angular frequency. If None, uses FFT.
        c0: Speed of sound.
        dt: Time step.

    Returns:
        VolumeQuadrupoleResult.
    """
    if T_field.ndim == 4:
        T_field = T_field[np.newaxis, ...]

    T_steps, ny, nx, _, _ = T_field.shape
    dA = dx * dy

    # Source grid
    x_1d = grid_origin[0] + np.arange(nx) * dx
    y_1d = grid_origin[1] + np.arange(ny) * dy
    xx, yy = np.meshgrid(x_1d, y_1d, indexing='xy')
    src_pos = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    src_weights = np.full(nx * ny, dA)

    # Observer positions
    if observer_positions is None:
        theta = np.linspace(0, 2 * np.pi, n_observers, endpoint=False)
        observer_positions = np.stack([
            observer_radius * np.cos(theta),
            observer_radius * np.sin(theta)
        ], axis=-1)

    n_obs = observer_positions.shape[0]

    if omega is not None:
        # Single frequency
        if T_steps > 1:
            freqs = np.fft.rfftfreq(T_steps, d=dt) * 2 * np.pi
            idx = np.argmin(np.abs(freqs - omega))
            T_hat = np.zeros((ny * nx, 2, 2), dtype=np.complex128)
            for i in range(2):
                for j in range(2):
                    T_comp = T_field[:, :, :, i, j].reshape(T_steps, -1)
                    T_hat[:, i, j] = np.fft.rfft(T_comp, axis=0)[idx]
        else:
            T_hat = T_field[0].reshape(-1, 2, 2).astype(np.complex128)

        p_hat = volume_quadrupole_frequency_domain(
            T_hat, src_pos, src_weights, observer_positions, omega, c0, ndim=2
        )

        return VolumeQuadrupoleResult(
            p_prime=p_hat,
            freq=np.array([omega / (2 * np.pi)]),
            observers=observer_positions,
            method="volume_quadrupole_2d_single_freq",
        )
    else:
        # Full FFT
        freqs_rad = np.fft.rfftfreq(T_steps, d=dt) * 2 * np.pi
        n_f = len(freqs_rad)

        T_hat_all = np.zeros((n_f, ny * nx, 2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                T_comp = T_field[:, :, :, i, j].reshape(T_steps, -1)
                T_hat_all[:, :, i, j] = np.fft.rfft(T_comp, axis=0)

        p_hat_all = np.zeros((n_obs, n_f), dtype=np.complex128)
        for fi in range(1, n_f):
            w = freqs_rad[fi]
            if w < 1e-12:
                continue
            p_hat_all[:, fi] = volume_quadrupole_frequency_domain(
                T_hat_all[fi], src_pos, src_weights, observer_positions, w, c0, ndim=2
            )

        p_time = np.real(np.fft.irfft(p_hat_all, n=T_steps, axis=1))
        freqs_hz = np.fft.rfftfreq(T_steps, d=dt)

        return VolumeQuadrupoleResult(
            p_prime=p_time,
            time=np.arange(T_steps) * dt,
            freq=freqs_hz,
            observers=observer_positions,
            method="volume_quadrupole_2d_fft",
        )


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------

def analytical_gaussian_quadrupole_farfield(
    observer_positions: np.ndarray,
    omega: float,
    c0: float,
    amplitude: float = 1.0,
    sigma: float = 2.0,
    source_center: Tuple[float, float] = (0.0, 0.0),
    quadrupole_type: str = "xx-yy",
) -> np.ndarray:
    """Analytical far-field for a Gaussian quadrupole (point-quadrupole limit).

    For r >> σ, the Gaussian source integrates to a point quadrupole:
        Q_ij = A_ij · π σ²
    and the far-field is:
        p̂(x) = Q_ij ∂²G/∂x_i∂x_j (x - x₀)

    Args:
        observer_positions: ``(N, 2)``.
        omega: Angular frequency.
        c0: Speed of sound.
        amplitude: Source amplitude.
        sigma: Gaussian width.
        source_center: Source center.
        quadrupole_type: "xx-yy" for cos(2θ), "xy" for sin(2θ).

    Returns:
        Complex pressure ``(N,)``.
    """
    if quadrupole_type == "xx-yy":
        A_ij = np.array([[amplitude, 0], [0, -amplitude]], dtype=np.complex128)
    elif quadrupole_type == "xy":
        A_ij = np.array([[0, amplitude], [amplitude, 0]], dtype=np.complex128)
    else:
        raise ValueError(f"Unknown quadrupole_type: {quadrupole_type}")

    Q_ij = A_ij * np.pi * sigma**2

    d2G = _green_2d_d2_point_source(observer_positions, np.array(source_center), omega, c0)

    # Contract: p = Q_ij d2G_ij
    p_hat = np.einsum('ij,oij->o', Q_ij, d2G)
    return p_hat


def compact_quadrupole_power(
    Q_ij: np.ndarray,
    omega: float,
    c0: float,
    rho0: float = 1.0,
    ndim: int = 2,
) -> float:
    """Acoustic power of a compact quadrupole.

    2-D: P/L = ρ₀ ω⁵ / (64 π c₀⁴) · Q_ij Q̄_ij  (trace-free)
    3-D: P = ρ₀ ω⁶ / (120 π c₀⁵) · Q_ij Q̄_ij
    """
    trace = np.trace(Q_ij)
    d = Q_ij.shape[0]
    Q_tf = Q_ij - (trace / d) * np.eye(d)
    contraction = np.sum(Q_tf * np.conj(Q_tf)).real

    if ndim == 3:
        return float(rho0 * omega**6 / (120.0 * np.pi * c0**5) * contraction)
    else:
        return float(rho0 * omega**5 / (64.0 * np.pi * c0**4) * contraction)


# ---------------------------------------------------------------------------
# Kirchhoff surface integral (for cross-validation)
# ---------------------------------------------------------------------------

def kirchhoff_surface_2d(
    p_surface: np.ndarray,
    dp_dn_surface: np.ndarray,
    surface_positions: np.ndarray,
    surface_normals: np.ndarray,
    surface_weights: np.ndarray,
    observer_positions: np.ndarray,
    omega: float,
    c0: float,
) -> np.ndarray:
    """2-D frequency-domain Kirchhoff surface integral (vectorized).

    p̂(x) = ∫_S [p̂ ∂G/∂n - G ∂p̂/∂n] dS
    """
    k = omega / c0
    # r_vec[o, s] = obs[o] - src[s]
    r_vec = observer_positions[:, None, :] - surface_positions[None, :, :]
    r = np.linalg.norm(r_vec, axis=-1)
    r_safe = np.maximum(r, 1e-15)
    kr = k * r_safe

    G = 0.25j * hankel1(0, kr)
    G_prime = -0.25j * k * hankel1(1, kr)

    r_hat = r_vec / r_safe[..., None]
    dG_dn = G_prime * np.sum(r_hat * surface_normals[None, :, :], axis=-1)

    integrand = p_surface[None, :] * dG_dn - G * dp_dn_surface[None, :]
    return np.sum(integrand * surface_weights[None, :], axis=1)
