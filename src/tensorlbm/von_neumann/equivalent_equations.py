"""Equivalent equations from Chapman-Enskog expansion for D2Q9 LBM.

Derives the macroscopic equations (Navier-Stokes) that the LBM scheme
actually solves, including numerical viscosity corrections.

For D2Q9 BGK:
  ∂ρ/∂t + ∇·(ρu) = 0
  ∂(ρu)/∂t + ∇·(ρuu) = -∇p + ∇·(ν_eff ∇u) + O(Ma²)

where:
  p = ρ cs² (ideal gas EOS)
  ν = (τ - 1/2) cs² = (τ - 1/2) / 3  (kinematic viscosity)
  ν_eff includes numerical corrections at finite wave number

The numerical viscosity depends on wave number θ = k Δx:
  ν_num(θ) = ν [1 + O(θ²)]

This module computes the effective viscosity as a function of wave number
and compares with the theoretical Chapman-Enskog prediction.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EquivalentEquationResult:
    """Result of equivalent equation analysis.

    Attributes:
        tau: relaxation time
        nu_physical: physical kinematic viscosity (τ-1/2)/3
        nu_numerical: effective numerical viscosity at given wave number
        theta: wave number (k Δx)
        dispersion_coeff: coefficient of dispersive term
        dissipation_coeff: coefficient of dissipative term
        relative_error: |ν_num - ν_phys| / ν_phys
    """
    tau: float
    nu_physical: float
    nu_numerical: float
    theta: float
    dispersion_coeff: float
    dissipation_coeff: float
    relative_error: float


class EquivalentEquationsD2Q9:
    """Chapman-Enskog equivalent equations for D2Q9 LBM.

    Computes the effective transport coefficients that the LBM scheme
    actually implements, including finite-k corrections.
    """

    def __init__(self):
        self.cs2 = 1.0 / 3.0

    def physical_viscosity(self, tau: float) -> float:
        """Physical kinematic viscosity from Chapman-Enskog.

        ν = (τ - 1/2) cs² = (τ - 1/2) / 3

        Args:
            tau: relaxation time

        Returns:
            Kinematic viscosity in lattice units
        """
        return (tau - 0.5) * self.cs2

    def numerical_viscosity_from_eigenvalue(
        self, tau: float, k: float, theta_dir: float = 0.0
    ) -> float:
        """Extract numerical viscosity from the amplification eigenvalue.

        For a diffusive mode, the eigenvalue near k=0 behaves as:
        λ ≈ 1 - ν_num k² + O(k⁴) + i (-cs k + O(k³))

        So: -ln|λ| ≈ ν_num k² for small k
        And: ν_num = -ln|λ| / k²

        Args:
            tau: relaxation time
            k: wave number magnitude
            theta_dir: wave propagation direction

        Returns:
            Numerical viscosity estimated from eigenvalue decay
        """
        from .d2q9_bgk import D2Q9BGKAmplification
        from .amplification import compute_spectral_radius

        bgk = D2Q9BGKAmplification()
        kx = k * np.cos(theta_dir)
        ky = k * np.sin(theta_dir)
        G = bgk.amplification_matrix(kx, ky, tau, ma=0.0)
        rho, eigenvalues, dominant = compute_spectral_radius(G)

        if rho <= 0 or rho >= 1:
            return float('inf')

        # The dominant eigenvalue for the diffusive mode
        # -ln|λ| = ν_num k² → ν_num = -ln|λ| / k²
        nu_num = -np.log(np.abs(dominant)) / (k * k)
        return nu_num

    def equivalent_viscosity_curve(
        self,
        tau: float,
        theta_values: np.ndarray | None = None,
        theta_dir: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute effective viscosity as function of wave number.

        Args:
            tau: relaxation time
            theta_values: wave number values (k Δx)
            theta_dir: wave propagation direction

        Returns:
            (theta_values, nu_numerical, nu_physical) arrays
        """
        if theta_values is None:
            theta_values = np.linspace(0.01, np.pi * 0.9, 50)

        nu_phys = self.physical_viscosity(tau)
        nu_num = np.zeros_like(theta_values)

        for i, k in enumerate(theta_values):
            nu_num[i] = self.numerical_viscosity_from_eigenvalue(tau, k, theta_dir)

        return theta_values, nu_num, np.full_like(theta_values, nu_phys)

    def analyze(
        self, tau: float, k: float, theta_dir: float = 0.0
    ) -> EquivalentEquationResult:
        """Full equivalent equation analysis at a single point.

        Args:
            tau: relaxation time
            k: wave number
            theta_dir: wave direction

        Returns:
            EquivalentEquationResult
        """
        nu_phys = self.physical_viscosity(tau)
        nu_num = self.numerical_viscosity_from_eigenvalue(tau, k, theta_dir)

        rel_err = abs(nu_num - nu_phys) / nu_phys if nu_phys > 0 else float('inf')

        return EquivalentEquationResult(
            tau=tau,
            nu_physical=nu_phys,
            nu_numerical=nu_num,
            theta=k,
            dispersion_coeff=0.0,  # TODO: extract from phase speed
            dissipation_coeff=nu_num,
            relative_error=rel_err,
        )

    def viscosity_pollution_ratio(
        self, tau: float, k: float
    ) -> float:
        """Compute numerical/physical viscosity ratio.

        This quantifies how much the numerical viscosity deviates from
        the physical (Chapman-Enskog) viscosity at finite wave number.

        Ratio = ν_num(k) / ν_phys

        For k → 0: ratio → 1
        For large k: ratio can deviate significantly

        Args:
            tau: relaxation time
            k: wave number

        Returns:
            Ratio ν_num / ν_phys
        """
        nu_phys = self.physical_viscosity(tau)
        if nu_phys <= 0:
            return float('inf')
        nu_num = self.numerical_viscosity_from_eigenvalue(tau, k)
        return nu_num / nu_phys

    def find_low_pollution_region(
        self,
        tau_values: np.ndarray,
        k_values: np.ndarray,
        threshold: float = 0.01,
    ) -> list[tuple[float, float]]:
        """Find (τ, k) region where viscosity pollution < threshold.

        Args:
            tau_values: τ values to scan
            k_values: k values to scan
            threshold: maximum allowed ratio deviation (default 1%)

        Returns:
            List of (τ, k) points where |ν_num/ν_phys - 1| < threshold
        """
        good_points = []
        for tau in tau_values:
            for k in k_values:
                ratio = self.viscosity_pollution_ratio(tau, k)
                if abs(ratio - 1.0) < threshold:
                    good_points.append((tau, k))
        return good_points
