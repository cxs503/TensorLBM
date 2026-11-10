"""Collision operators for von Neumann analysis.

Implements BGK, TRT, and MRT collision matrices in the linearized regime
around a uniform base state (ρ₀=1, u₀=0).

The linearized equilibrium projection is:
    E_ij = w_i [1 + (c_i · c_j) / cs2]

This is exact for the acoustic/shear modes at small Mach number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lattice import LatticeBase


def equilibrium_projection(lattice: LatticeBase) -> np.ndarray:
    """Compute the linearized equilibrium projection matrix E.

    E_ij = w_i [1 + (c_i · c_j) / cs2]

    This maps δf → δf^eq in the linearized regime around (ρ₀=1, u₀=0).
    """
    q = lattice.q
    cs2 = lattice.cs2
    c = lattice.c
    w = lattice.w

    # c_i · c_j for all pairs
    cdot = c @ c.T  # (q, q)

    # E_ij = w_i * (1 + c_i·c_j / cs2)
    E = w[:, None] * (1.0 + cdot / cs2)
    return E


@dataclass
class BGKCollision:
    """BGK (single relaxation time) collision operator.

    C_BGK = I - (1/τ)(I - E)

    where τ is the relaxation time and E is the equilibrium projection.
    The kinematic viscosity is ν = cs2 (τ - 0.5) = (τ - 0.5) / 3.
    """

    lattice: LatticeBase
    tau: float

    def __post_init__(self):
        if self.tau <= 0.5:
            raise ValueError(f"τ must be > 0.5 for stability, got τ={self.tau}")

    @property
    def nu(self) -> float:
        """Kinematic viscosity."""
        return self.lattice.cs2 * (self.tau - 0.5)

    def collision_matrix(self) -> np.ndarray:
        """Return the collision matrix C = I - (1/τ)(I - E)."""
        q = self.lattice.q
        E = equilibrium_projection(self.lattice)
        I = np.eye(q, dtype=np.complex128)
        return I - (1.0 / self.tau) * (I - E)


@dataclass
class TRTCollision:
    """Two-relaxation-time (TRT) collision operator.

    The TRT operator relaxates symmetric (even) and antisymmetric (odd)
    parts of f with different rates τ+ and τ-.

    For the 'magic' parameter Λ = τ+ τ- / 4 = 3/16, the TRT scheme
    achieves optimal isotropy and minimal dispersion errors.

    The relation to viscosity is: ν = cs2 (τ+ - 0.5)
    The free parameter is Λ = (τ- - 0.5)(τ+ - 0.5) / 4, or equivalently:
        τ- = 0.5 + 2Λ / (τ+ - 0.5)
    """

    lattice: LatticeBase
    tau_plus: float
    Lambda: Optional[float] = None  # if None, use BGK equivalent (Λ = (τ+-0.5)²/4)

    def __post_init__(self):
        if self.tau_plus <= 0.5:
            raise ValueError(f"τ+ must be > 0.5, got τ+={self.tau_plus}")
        if self.Lambda is None:
            # Default: BGK equivalent
            self.Lambda = (self.tau_plus - 0.5) ** 2 / 4.0

    @property
    def tau_minus(self) -> float:
        """Compute τ- from τ+ and Λ."""
        return 0.5 + 2.0 * self.Lambda / (self.tau_plus - 0.5)

    @property
    def nu(self) -> float:
        """Kinematic viscosity (determined by τ+)."""
        return self.lattice.cs2 * (self.tau_plus - 0.5)

    def collision_matrix(self) -> np.ndarray:
        """Return the TRT collision matrix.

        TRT splits f into even/odd parts:
            f_i^+ = (f_i + f_ī) / 2  (symmetric, contains density)
            f_i^- = (f_i - f_ī) / 2  (antisymmetric, contains velocity)

        The equilibrium splits as:
            f_i^eq,+ = w_i ρ  (even part)
            f_i^eq,- = w_i (c_i·u)/cs²  (odd part)

        The collision is:
            Ω_i = -(1/τ+)(f_i^+ - f_i^eq,+) - (1/τ-)(f_i^- - f_i^-,eq)
        """
        q = self.lattice.q
        opp = self.lattice.opposite
        w = self.lattice.w
        c = self.lattice.c
        cs2 = self.lattice.cs2

        # Build even and odd equilibrium projection matrices
        # E+_ij: maps f_j to f_i^eq,+ = w_i ρ = w_i Σ_k f_k
        # So E+_ij = w_i
        E_plus = np.tile(w[:, None], (1, q))  # (q, q) with E+_ij = w_i

        # E-_:ij: maps f_j to f_i^eq,- = w_i (c_i·u)/cs² = w_i Σ_a c_ia Σ_k c_ka f_k / cs²
        # So E-_:ij = w_i Σ_a c_ia c_ja / cs² = w_i (c_i·c_j) / cs²
        cdot = c @ c.T  # (q, q)
        E_minus = w[:, None] * cdot / cs2  # (q, q)

        # Even/odd projection operators
        # P+_ij = (δ_ij + δ_{i,opp(j)}) / 2
        # P-_:ij = (δ_ij - δ_{i,opp(j)}) / 2
        P_plus = np.zeros((q, q), dtype=np.complex128)
        P_minus = np.zeros((q, q), dtype=np.complex128)

        for i in range(q):
            j = opp[i]
            if i == j:
                # Rest velocity: purely even
                P_plus[i, i] = 1.0
                P_minus[i, i] = 0.0
            else:
                P_plus[i, j] = 0.5
                P_plus[i, i] += 0.5
                P_minus[i, j] = -0.5
                P_minus[i, i] += 0.5

        # TRT collision: Ω = -(1/τ+) P+ (f - E+ f) - (1/τ-) P- (f - E- f)
        # = -(1/τ+) P+ (I - E+) f - (1/τ-) P- (I - E-) f
        # So C = I - (1/τ+) P+ (I - E+) - (1/τ-) P- (I - E-)

        I = np.eye(q, dtype=np.complex128)
        C = I - (1.0 / self.tau_plus) * P_plus @ (I - E_plus) - (1.0 / self.tau_minus) * P_minus @ (I - E_minus)

        return C


@dataclass
class MRTCollision:
    """Multiple-relaxation-time (MRT) collision operator.

    The MRT operator transforms to moment space via a transformation matrix T,
    applies diagonal relaxation, and transforms back:

        C_MRT = I - T^{-1} diag(s_i) T (I - E)

    where s_i are the relaxation rates for each moment, and E is the equilibrium projection.

    For D2Q9, the standard moments are:
        (ρ, e, ε, jx, qjx, jy, qjy, pxx, pxy)

    where:
        ρ = density (conserved)
        e = energy (non-conserved)
        ε = square of energy (non-conserved)
        jx, jy = momentum (conserved)
        qjx, qjy = energy flux (non-conserved)
        pxx, pxy = stress tensor (non-conserved)
    """

    lattice: LatticeBase
    tau_rho: float = 1.0  # density relaxation (conserved, doesn't matter)
    tau_e: float = 1.0  # energy relaxation
    tau_eps: float = 1.0  # energy-square relaxation
    tau_j: float = 1.0  # momentum relaxation (conserved, doesn't matter)
    tau_qj: float = 1.0  # energy flux relaxation
    tau_pxx: float = 1.0  # stress relaxation
    tau_pxy: float = 1.0  # stress relaxation

    def __post_init__(self):
        if self.lattice.name != "D2Q9":
            raise NotImplementedError("MRT currently only implemented for D2Q9")

    @property
    def nu(self) -> float:
        """Kinematic viscosity (determined by stress relaxation)."""
        return self.lattice.cs2 * (self.tau_pxx - 0.5)

    def _transformation_matrix_d2q9(self) -> np.ndarray:
        """D2Q9 moment transformation matrix (Lallemand & Luo, 2000).

        Rows are moment basis vectors, columns are velocity directions.
        """
        # Standard D2Q9 MRT transformation (Lallemand & Luo 2000, Table 1)
        # Order: ρ, e, ε, jx, qjx, jy, qjy, pxx, pxy
        T = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],           # ρ
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],       # e
            [4, -2, -2, -2, -2, 1, 1, 1, 1],        # ε
            [0, 1, 0, -1, 0, 1, -1, -1, 1],         # jx
            [0, -2, 0, 2, 0, 1, -1, -1, 1],         # qjx
            [0, 0, 1, 0, -1, 1, 1, -1, -1],         # jy
            [0, 0, -2, 0, 2, 1, 1, -1, -1],         # qjy
            [0, 1, -1, 1, -1, 0, 0, 0, 0],          # pxx
            [0, 0, 0, 0, 0, 1, -1, 1, -1],          # pxy
        ], dtype=np.float64)
        return T

    def collision_matrix(self) -> np.ndarray:
        """Return the MRT collision matrix C = I - T^{-1} S T (I - E).

        S = diag(s_i) where s_i = 1/τ_i for each moment.
        For conserved moments (ρ, jx, jy), s = 0 (no relaxation).
        E is the equilibrium projection.
        """
        T = self._transformation_matrix_d2q9()
        T_inv = np.linalg.inv(T)
        E = equilibrium_projection(self.lattice)

        # Relaxation rates (s_i = 1/τ_i)
        # For conserved moments (ρ, jx, jy), s = 0 (no relaxation)
        s = np.array([
            0.0,  # ρ (conserved)
            1.0 / self.tau_e,  # e
            1.0 / self.tau_eps,  # ε
            0.0,  # jx (conserved)
            1.0 / self.tau_qj,  # qjx
            0.0,  # jy (conserved)
            1.0 / self.tau_qj,  # qjy (same relaxation as qjx)
            1.0 / self.tau_pxx,  # pxx
            1.0 / self.tau_pxx,  # pxy (same relaxation as pxx for isotropy)
        ], dtype=np.complex128)

        S = np.diag(s)
        I = np.eye(self.lattice.q, dtype=np.complex128)
        return I - T_inv @ S @ T @ (I - E)
