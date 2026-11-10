"""Amplification matrix utilities and spectral radius computation.

Provides common infrastructure for von Neumann stability analysis:
- Spectral radius computation from amplification matrices
- Stability domain scanning over (θ, τ, Ma) parameter space
- Data structures for analysis results

Sign convention:
  growth_rate = ln(spectral_radius)
  - growth_rate < 0: mode decays (stable)
  - growth_rate > 0: mode grows (unstable)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass
class AmplificationResult:
    """Result of von Neumann analysis at a single (θ, τ, Ma) point.

    Attributes:
        theta: wave number angle (direction of wave propagation)
        k_mag: wave number magnitude |k|
        tau: relaxation time
        ma: Mach number (background flow / cs)
        spectral_radius: max |eigenvalue| of amplification matrix
        eigenvalues: all eigenvalues of the amplification matrix
        dominant_eigenvalue: eigenvalue with largest magnitude
        growth_rate: ln(|λ_max|) per time step (negative = decaying/stable)
        phase_speed: phase speed from dominant eigenvalue argument
        is_stable: True if spectral_radius <= 1.0
    """
    theta: float
    k_mag: float
    tau: float
    ma: float
    spectral_radius: float
    eigenvalues: np.ndarray
    dominant_eigenvalue: complex
    growth_rate: float
    phase_speed: float
    is_stable: bool


@dataclass
class StabilityScanResult:
    """Result of scanning stability over a parameter grid.

    Attributes:
        tau_values: array of τ values scanned
        ma_values: array of Ma values scanned
        theta_values: array of θ values scanned
        spectral_radii: 3D array [i_tau, i_ma, i_theta] of spectral radii
        stable_mask: boolean array, True where spectral_radius <= 1
        boundary_points: list of (τ, Ma) points near the stability boundary
    """
    tau_values: np.ndarray
    ma_values: np.ndarray
    theta_values: np.ndarray
    spectral_radii: np.ndarray
    stable_mask: np.ndarray
    boundary_points: list = field(default_factory=list)


class AmplificationMatrixProvider(Protocol):
    """Protocol for classes that compute amplification matrices."""

    @property
    def n_velocities(self) -> int:
        """Number of discrete velocities (e.g. 9 for D2Q9)."""
        ...

    def amplification_matrix(
        self, kx: float, ky: float, tau: float, ma: float, flow_angle: float = 0.0
    ) -> np.ndarray:
        """Compute the amplification matrix G(k, τ, Ma).

        Args:
            kx: x-component of wave vector
            ky: y-component of wave vector
            tau: relaxation time
            ma: Mach number (|u₀|/cs)
            flow_angle: angle of background flow (rad)

        Returns:
            Complex amplification matrix of shape (Q, Q) where Q = n_velocities.
        """
        ...


def compute_spectral_radius(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray, complex]:
    """Compute spectral radius and eigenvalues of amplification matrix.

    Args:
        matrix: complex amplification matrix

    Returns:
        Tuple of (spectral_radius, eigenvalues, dominant_eigenvalue)
    """
    eigenvalues = np.linalg.eigvals(matrix)
    magnitudes = np.abs(eigenvalues)
    dominant_idx = np.argmax(magnitudes)
    spectral_radius = magnitudes[dominant_idx]
    dominant_eigenvalue = eigenvalues[dominant_idx]
    return spectral_radius, eigenvalues, dominant_eigenvalue


def compute_growth_rate(spectral_radius: float) -> float:
    """Compute growth rate from spectral radius.

    Convention: growth_rate = ln(spectral_radius)
    - Negative: mode decays (stable)
    - Positive: mode grows (unstable)

    Args:
        spectral_radius: max |eigenvalue|

    Returns:
        Growth rate = ln(spectral_radius) per time step
    """
    if spectral_radius <= 0:
        return float('-inf')
    return np.log(spectral_radius)


def compute_phase_speed(dominant_eigenvalue: complex, k_mag: float) -> float:
    """Compute phase speed from dominant eigenvalue.

    The eigenvalue λ = |λ| exp(i φ) where φ = -ω (angular frequency).
    Phase speed = ω / |k| = -arg(λ) / |k|

    Args:
        dominant_eigenvalue: eigenvalue with largest magnitude
        k_mag: wave number magnitude

    Returns:
        Phase speed (lattice units per time step per wave number)
    """
    if k_mag <= 0:
        return 0.0
    phase = np.angle(dominant_eigenvalue)
    return -phase / k_mag


def scan_stability_domain(
    provider: AmplificationMatrixProvider,
    tau_values: np.ndarray,
    ma_values: np.ndarray,
    theta_values: np.ndarray | None = None,
    k_max: float = np.pi,
    n_k: int = 32,
    flow_angle: float = 0.0,
) -> StabilityScanResult:
    """Scan stability domain over (τ, Ma) parameter space.

    For each (τ, Ma) point, find the maximum spectral radius over all
    wave numbers and directions.

    Args:
        provider: amplification matrix provider
        tau_values: array of τ values
        ma_values: array of Ma values
        theta_values: array of wave direction angles (default: 8 angles)
        k_max: maximum wave number magnitude (default: π)
        n_k: number of k magnitudes to sample
        flow_angle: angle of background flow (rad)

    Returns:
        StabilityScanResult with spectral radii and stability information
    """
    if theta_values is None:
        theta_values = np.linspace(0, np.pi / 2, 8, endpoint=False)

    k_magnitudes = np.linspace(0.01, k_max, n_k)

    n_tau = len(tau_values)
    n_ma = len(ma_values)

    # spectral_radii[i_tau, i_ma] = max over all (k, theta) of spectral radius
    spectral_radii = np.zeros((n_tau, n_ma))

    for i_tau, tau in enumerate(tau_values):
        for i_ma, ma in enumerate(ma_values):
            max_rho = 0.0
            for theta in theta_values:
                for k_mag in k_magnitudes:
                    kx = k_mag * np.cos(theta)
                    ky = k_mag * np.sin(theta)
                    G = provider.amplification_matrix(kx, ky, tau, ma, flow_angle)
                    rho, _, _ = compute_spectral_radius(G)
                    max_rho = max(max_rho, rho)
            spectral_radii[i_tau, i_ma] = max_rho

    stable_mask = spectral_radii <= 1.0

    # Find boundary points (where stability changes)
    boundary_points = []
    for i_tau in range(n_tau):
        for i_ma in range(n_ma):
            if stable_mask[i_tau, i_ma]:
                # Check if any neighbor is unstable
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i_tau + di, i_ma + dj
                    if 0 <= ni < n_tau and 0 <= nj < n_ma:
                        if not stable_mask[ni, nj]:
                            boundary_points.append(
                                (tau_values[i_tau], ma_values[i_ma])
                            )
                            break

    return StabilityScanResult(
        tau_values=tau_values,
        ma_values=ma_values,
        theta_values=theta_values,
        spectral_radii=spectral_radii,
        stable_mask=stable_mask,
        boundary_points=boundary_points,
    )
