"""D2Q9 BGK amplification matrix for von Neumann stability analysis.

Implements the linearized amplification matrix for the D2Q9 lattice
with BGK (single-relaxation-time) collision operator.

The analysis linearizes around a uniform state (ρ₀, u₀) and computes
the amplification matrix G(k) whose spectral radius determines stability.

Reference: Lallemand & Luo (2000), JCP 159, 1-37.
"""

from __future__ import annotations

import numpy as np
from .amplification import (
    AmplificationResult,
    compute_growth_rate,
    compute_phase_speed,
    compute_spectral_radius,
)

# D2Q9 lattice velocities
_C2 = np.array([
    [0, 0],      # 0: rest
    [1, 0],      # 1: +x
    [0, 1],      # 2: +y
    [-1, 0],     # 3: -x
    [0, -1],     # 4: -y
    [1, 1],      # 5: +x+y
    [-1, 1],     # 6: -x+y
    [-1, -1],    # 7: -x-y
    [1, -1],     # 8: +x-y
], dtype=np.float64)

# D2Q9 weights
_W2 = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)

# Sound speed squared
_CS2 = 1.0 / 3.0


class D2Q9BGKAmplification:
    """D2Q9 BGK amplification matrix for von Neumann analysis.

    Constructs the 9×9 amplification matrix G(k, τ, Ma) for the
    linearized BGK-LBM around a uniform state.

    Args:
        rho0: background density (default 1.0)
    """

    def __init__(self, rho0: float = 1.0):
        self.rho0 = rho0
        self.Q = 9  # number of velocities

    @property
    def n_velocities(self) -> int:
        return self.Q

    def equilibrium_projection(
        self, tau: float, ma: float, flow_angle: float = 0.0
    ) -> np.ndarray:
        """Compute the equilibrium projection matrix E.

        E_ij = ∂f_i^eq / ∂f_j evaluated at the uniform state.

        For D2Q9 BGK with background flow u₀ = Ma * cs * (cos α, sin α):

        E_ij = w_i {1 - (9/2)(c_i·u₀)² + (3/2)|u₀|²
                     + 3(c_i·c_j) + 9(c_i·u₀)(c_i·c_j) - 3(u₀·c_j)}

        Args:
            tau: relaxation time (not used in E, but kept for interface consistency)
            ma: Mach number |u₀|/cs
            flow_angle: angle of background flow (rad)

        Returns:
            9×9 real matrix E
        """
        cs = np.sqrt(_CS2)
        u0x = ma * cs * np.cos(flow_angle)
        u0y = ma * cs * np.sin(flow_angle)
        u0 = np.array([u0x, u0y])
        u0_sq = u0x**2 + u0y**2

        E = np.zeros((self.Q, self.Q), dtype=np.float64)
        for i in range(self.Q):
            ci = _C2[i]
            ci_dot_u0 = ci[0] * u0x + ci[1] * u0y
            for j in range(self.Q):
                cj = _C2[j]
                ci_dot_cj = ci[0] * cj[0] + ci[1] * cj[1]
                u0_dot_cj = u0x * cj[0] + u0y * cj[1]

                E[i, j] = _W2[i] * (
                    1.0 - 4.5 * ci_dot_u0**2 + 1.5 * u0_sq
                    + 3.0 * ci_dot_cj
                    + 9.0 * ci_dot_u0 * ci_dot_cj
                    - 3.0 * u0_dot_cj
                )
        return E

    def amplification_matrix(
        self, kx: float, ky: float, tau: float, ma: float, flow_angle: float = 0.0
    ) -> np.ndarray:
        """Compute the amplification matrix G(k, τ, Ma).

        G_ij(k) = exp(-i k·c_i) [(1-1/τ) δ_ij + (1/τ) E_ij]

        The eigenvalues of G determine stability:
        - |λ| ≤ 1 for all eigenvalues → stable
        - any |λ| > 1 → unstable

        Args:
            kx: x-component of wave vector
            ky: y-component of wave vector
            tau: relaxation time τ
            ma: Mach number |u₀|/cs
            flow_angle: angle of background flow (rad)

        Returns:
            9×9 complex amplification matrix
        """
        E = self.equilibrium_projection(tau, ma, flow_angle)

        G = np.zeros((self.Q, self.Q), dtype=np.complex128)
        for i in range(self.Q):
            ci = _C2[i]
            phase = kx * ci[0] + ky * ci[1]
            streaming = np.exp(-1j * phase)
            for j in range(self.Q):
                collision = (1.0 - 1.0/tau) * (1 if i == j else 0) + (1.0/tau) * E[i, j]
                G[i, j] = streaming * collision

        return G

    def analyze(
        self, kx: float, ky: float, tau: float, ma: float, flow_angle: float = 0.0
    ) -> AmplificationResult:
        """Full von Neumann analysis at a single (k, τ, Ma) point.

        Args:
            kx: x-component of wave vector
            ky: y-component of wave vector
            tau: relaxation time
            ma: Mach number
            flow_angle: angle of background flow (rad)

        Returns:
            AmplificationResult with spectral radius, eigenvalues, etc.
        """
        G = self.amplification_matrix(kx, ky, tau, ma, flow_angle)
        rho, eigenvalues, dominant = compute_spectral_radius(G)
        k_mag = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)

        return AmplificationResult(
            theta=theta,
            k_mag=k_mag,
            tau=tau,
            ma=ma,
            spectral_radius=rho,
            eigenvalues=eigenvalues,
            dominant_eigenvalue=dominant,
            growth_rate=compute_growth_rate(rho),
            phase_speed=compute_phase_speed(dominant, k_mag),
            is_stable=(rho <= 1.0 + 1e-14),
        )

    def spectral_radius_vs_k(
        self,
        tau: float,
        ma: float,
        theta: float = 0.0,
        n_k: int = 100,
        k_max: float = np.pi,
        flow_angle: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute spectral radius as function of wave number magnitude.

        Args:
            tau: relaxation time
            ma: Mach number
            theta: wave propagation direction (rad)
            n_k: number of k points
            k_max: maximum wave number
            flow_angle: background flow angle (rad)

        Returns:
            (k_values, spectral_radii) arrays
        """
        k_values = np.linspace(0.001, k_max, n_k)
        radii = np.zeros(n_k)
        for idx, k in enumerate(k_values):
            kx = k * np.cos(theta)
            ky = k * np.sin(theta)
            G = self.amplification_matrix(kx, ky, tau, ma, flow_angle)
            radii[idx], _, _ = compute_spectral_radius(G)
        return k_values, radii

    def find_most_unstable_mode(
        self,
        tau: float,
        ma: float,
        n_theta: int = 16,
        n_k: int = 50,
        k_max: float = np.pi,
        flow_angle: float = 0.0,
    ) -> AmplificationResult:
        """Find the most unstable (or least stable) mode.

        Scans over wave number magnitudes and directions to find
        the mode with the largest spectral radius.

        Args:
            tau: relaxation time
            ma: Mach number
            n_theta: number of wave directions
            n_k: number of k magnitudes
            k_max: maximum wave number
            flow_angle: background flow angle

        Returns:
            AmplificationResult for the most critical mode
        """
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        k_values = np.linspace(0.001, k_max, n_k)

        best_result = None
        max_rho = 0.0

        for theta in thetas:
            for k in k_values:
                kx = k * np.cos(theta)
                ky = k * np.sin(theta)
                result = self.analyze(kx, ky, tau, ma, flow_angle)
                if result.spectral_radius > max_rho:
                    max_rho = result.spectral_radius
                    best_result = result

        return best_result
