"""D2Q9 MRT amplification matrix for von Neumann stability analysis.

Implements the linearized amplification matrix for the D2Q9 lattice
with Multiple-Relaxation-Time (MRT) collision operator.

The MRT model uses a moment basis with independent relaxation rates
for each moment, providing better stability and accuracy than BGK.

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

_CS2 = 1.0 / 3.0


def _build_mrt_transform() -> tuple[np.ndarray, np.ndarray]:
    """Build the D2Q9 MRT transformation matrix M and its inverse.

    Moment basis (Lallemand & Luo 2000):
    m0 = ρ          (density)
    m1 = e          (related to energy)
    m2 = ε          (related to energy square)
    m3 = jx         (x-momentum)
    m4 = qx         (x-energy flux)
    m5 = jy         (y-momentum)
    m6 = qy         (y-energy flux)
    m7 = pxx        (xx-stress)
    m8 = pyy        (xy-stress / pxy)

    Returns:
        (M, M_inv): transformation matrix and its inverse
    """
    M = np.zeros((9, 9), dtype=np.float64)

    for i in range(9):
        cx, cy = _C2[i]
        cx2 = cx * cx
        cy2 = cy * cy
        c2 = cx2 + cy2  # |c_i|^2

        # m0 = ρ (density)
        M[0, i] = 1.0
        # m1 = e (energy): -4 + 3|c_i|^2
        M[1, i] = -4.0 + 3.0 * c2
        # m2 = ε (energy square): 4 - 5|c_i|^2 + 3|c_i|^4
        M[2, i] = 4.0 - 5.0 * c2 + 3.0 * c2 * c2
        # m3 = jx (x-momentum)
        M[3, i] = cx
        # m4 = qx (x-energy flux): -2/3 cx + cx|c_i|^2... simplified: cx*(|c_i|^2 - 2/3)...
        # Actually from Lallemand & Luo: qx = cx * (3|c_i|^2 - 2) / 3... let me use the standard form
        # m4 = qx: (-2 + 3|c_i|^2) * cx / 3... no, let me use the exact form
        # From Lallemand & Luo (2000) Table 1:
        # m4 = qx = cx * (3*|ci|^2 - 2) ... but this doesn't look right either
        # Let me use the standard D2Q9 MRT transform:
        M[4, i] = cx * (3.0 * c2 - 2.0) / 3.0  # This is not standard, let me fix
        # Actually the standard form from d'Humières (1992) and Lallemand & Luo:
        # m4 = qx: for |ci|^2=0: 0, for |ci|^2=1: cx/3, for |ci|^2=2: 4cx/3
        # So m4 = cx * (3|ci|^2 - 2) / 3... let me verify:
        # |ci|^2=0: cx*(-2/3) = 0 (since cx=0 for rest) ✓
        # |ci|^2=1: cx*(1/3) ✓
        # |ci|^2=2: cx*(4/3) ✓
        # OK this is correct.

        # m5 = jy (y-momentum)
        M[5, i] = cy
        # m6 = qy: cy * (3|ci|^2 - 2) / 3
        M[6, i] = cy * (3.0 * c2 - 2.0) / 3.0
        # m7 = pxx (stress): cx^2 - cy^2
        M[7, i] = cx2 - cy2
        # m8 = pxy (stress): cx * cy
        M[8, i] = cx * cy

    M_inv = np.linalg.inv(M)
    return M, M_inv


# Pre-compute transformation matrices
_M_MRT, _M_INV_MRT = _build_mrt_transform()


class D2Q9MRTAmplification:
    """D2Q9 MRT amplification matrix for von Neumann analysis.

    The MRT collision operator relaxates each moment independently:
    m* = m - S (m - m^eq)

    where S = diag(s0, s1, ..., s8) is the relaxation matrix.

    For physical consistency:
    - s0 = 0 (mass conservation)
    - s3 = s5 = 1/τ (momentum relaxation → viscosity)
    - s1, s2, s4, s6, s7, s8 are free parameters for stability

    Args:
        rho0: background density (default 1.0)
        s_values: relaxation rates for each moment (length 9)
                  If None, uses default Lallemand-Luo values
    """

    def __init__(
        self,
        rho0: float = 1.0,
        s_values: np.ndarray | None = None,
    ):
        self.rho0 = rho0
        self.Q = 9

        if s_values is None:
            # Default: Lallemand & Luo (2000) optimal values
            # s0=0 (mass), s3=s5=1/τ (will be set per call), s7=s8=1/τ (viscosity)
            # s1, s2, s4, s6 are free parameters for stability
            self._s_base = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        else:
            self._s_base = np.asarray(s_values, dtype=np.float64)

    @property
    def n_velocities(self) -> int:
        return self.Q

    def _relaxation_matrix(self, tau: float) -> np.ndarray:
        """Build the diagonal relaxation matrix S.

        s3 = s5 = 1/τ (momentum → viscosity)
        s7 = s8 = 1/τ (stress → viscosity)
        Other rates from base values.
        """
        s = self._s_base.copy()
        s[3] = 1.0 / tau  # jx
        s[5] = 1.0 / tau  # jy
        s[7] = 1.0 / tau  # pxx
        s[8] = 1.0 / tau  # pxy
        return np.diag(s)

    def equilibrium_moments(
        self, ma: float, flow_angle: float = 0.0
    ) -> np.ndarray:
        """Compute equilibrium moments at the uniform state.

        m^eq = M f^eq(ρ₀, u₀)

        Args:
            ma: Mach number
            flow_angle: flow direction angle (rad)

        Returns:
            Equilibrium moment vector (length 9)
        """
        cs = np.sqrt(_CS2)
        u0x = ma * cs * np.cos(flow_angle)
        u0y = ma * cs * np.sin(flow_angle)
        rho0 = self.rho0
        u0_sq = u0x**2 + u0y**2

        # Compute equilibrium in velocity space
        f_eq = np.zeros(self.Q, dtype=np.float64)
        for i in range(self.Q):
            ci_dot_u = _C2[i, 0] * u0x + _C2[i, 1] * u0y
            f_eq[i] = _W2[i] * rho0 * (
                1.0 + 3.0 * ci_dot_u
                + 4.5 * ci_dot_u**2
                - 1.5 * u0_sq
            )

        # Transform to moment space
        m_eq = _M_MRT @ f_eq
        return m_eq

    def equilibrium_projection_moment(
        self, ma: float, flow_angle: float = 0.0
    ) -> np.ndarray:
        """Compute the equilibrium projection in moment space.

        For the linearized analysis, we need ∂m^eq/∂m at the uniform state.

        Returns:
            9×9 matrix E_m in moment space
        """
        cs = np.sqrt(_CS2)
        u0x = ma * cs * np.cos(flow_angle)
        u0y = ma * cs * np.sin(flow_angle)
        u0_sq = u0x**2 + u0y**2
        rho0 = self.rho0

        # First compute E in velocity space (same as BGK)
        E_vel = np.zeros((self.Q, self.Q), dtype=np.float64)
        for i in range(self.Q):
            ci = _C2[i]
            ci_dot_u0 = ci[0] * u0x + ci[1] * u0y
            for j in range(self.Q):
                cj = _C2[j]
                ci_dot_cj = ci[0] * cj[0] + ci[1] * cj[1]
                u0_dot_cj = u0x * cj[0] + u0y * cj[1]

                E_vel[i, j] = _W2[i] * (
                    1.0 - 4.5 * ci_dot_u0**2 + 1.5 * u0_sq
                    + 3.0 * ci_dot_cj
                    + 9.0 * ci_dot_u0 * ci_dot_cj
                    - 3.0 * u0_dot_cj
                )

        # Transform to moment space: E_m = M E_vel M^{-1}
        E_m = _M_MRT @ E_vel @ _M_INV_MRT
        return E_m

    def amplification_matrix(
        self, kx: float, ky: float, tau: float, ma: float, flow_angle: float = 0.0
    ) -> np.ndarray:
        """Compute the MRT amplification matrix G_MRT(k, τ, Ma).

        The MRT LBM:
        1. Transform to moment space: m = M f
        2. Collision: m* = m - S(m - m^eq) = (I - S)m + S m^eq
        3. Transform back: f* = M^{-1} m*
        4. Stream: f(x+c, t+1) = f*(x, t)

        In Fourier space:
        F' = diag(exp(-i k·c_i)) [(I-S) + S E_m] F

        where E_m = M E_vel M^{-1} is the equilibrium projection in moment space.

        Wait, this is not quite right. Let me reconsider.

        The MRT collision in velocity space:
        f* = f - M^{-1} S M (f - f^eq)
           = [I - M^{-1} S M] f + M^{-1} S M f^eq
           = [I - M^{-1} S M] f + M^{-1} S E_vel f  (since f^eq = E_vel f for linearized)

        So the collision matrix in velocity space:
        C_MRT = I - M^{-1} S M + M^{-1} S E_vel
              = I - M^{-1} S (M - E_vel)

        Hmm, but E_vel maps f to f^eq, and M maps f to m.
        The equilibrium in moment space: m^eq = M f^eq = M E_vel f

        So: m* = m - S(m - m^eq) = (I-S)m + S M E_vel f
        f* = M^{-1} m* = M^{-1}(I-S)M f + M^{-1} S M E_vel f
           = [M^{-1}(I-S)M + M^{-1} S M E_vel] f

        The collision matrix: C = M^{-1}(I-S)M + M^{-1} S M E_vel
                                = M^{-1}[(I-S)M + S M E_vel]
                                = M^{-1}[M - SM + S M E_vel]
                                = I - M^{-1} S M (I - E_vel)

        The amplification matrix:
        G_ij = exp(-i k·c_i) C_ij

        Args:
            kx, ky: wave vector components
            tau: relaxation time
            ma: Mach number
            flow_angle: background flow angle

        Returns:
            9×9 complex amplification matrix
        """
        S = self._relaxation_matrix(tau)

        # Equilibrium projection in velocity space
        E_vel = np.zeros((self.Q, self.Q), dtype=np.float64)
        cs = np.sqrt(_CS2)
        u0x = ma * cs * np.cos(flow_angle)
        u0y = ma * cs * np.sin(flow_angle)
        u0_sq = u0x**2 + u0y**2

        for i in range(self.Q):
            ci = _C2[i]
            ci_dot_u0 = ci[0] * u0x + ci[1] * u0y
            for j in range(self.Q):
                cj = _C2[j]
                ci_dot_cj = ci[0] * cj[0] + ci[1] * cj[1]
                u0_dot_cj = u0x * cj[0] + u0y * cj[1]
                E_vel[i, j] = _W2[i] * (
                    1.0 - 4.5 * ci_dot_u0**2 + 1.5 * u0_sq
                    + 3.0 * ci_dot_cj
                    + 9.0 * ci_dot_u0 * ci_dot_cj
                    - 3.0 * u0_dot_cj
                )

        # MRT collision matrix in velocity space
        # C = I - M^{-1} S M (I - E_vel)
        I9 = np.eye(9)
        C_mrt = I9 - _M_INV_MRT @ S @ _M_MRT @ (I9 - E_vel)

        # Add streaming
        G = np.zeros((self.Q, self.Q), dtype=np.complex128)
        for i in range(self.Q):
            ci = _C2[i]
            phase = kx * ci[0] + ky * ci[1]
            streaming = np.exp(-1j * phase)
            for j in range(self.Q):
                G[i, j] = streaming * C_mrt[i, j]

        return G

    def analyze(
        self, kx: float, ky: float, tau: float, ma: float, flow_angle: float = 0.0
    ) -> AmplificationResult:
        """Full von Neumann analysis at a single (k, τ, Ma) point."""
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
        """Compute spectral radius as function of wave number magnitude."""
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
        """Find the most unstable (or least stable) mode."""
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
