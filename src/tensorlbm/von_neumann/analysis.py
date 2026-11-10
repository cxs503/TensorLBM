"""Von Neumann stability analysis: amplification matrix and eigenvalue computation.

The LBM evolution in Fourier space is:
    f̂(k, t+1) = M(k) · f̂(k, t)

where M(k) = S(k) · C is the amplification matrix,
    S(k) = diag(exp(-i k·c_j)) is the streaming matrix,
    C is the collision matrix.

The eigenvalues λ of M(k) give the dispersion relation:
    ω(k) = i ln(λ) / Δt

where Re(ω) is the oscillation frequency and Im(ω) < 0 is the damping rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .collision import BGKCollision, MRTCollision, TRTCollision
from .lattice import LatticeBase


@dataclass
class AmplificationMatrix:
    """Amplification matrix M(k) for von Neumann analysis.

    M(k) = S(k) · C
    where S(k) = diag(exp(-i k·c_j)) and C is the collision matrix.
    """

    lattice: LatticeBase
    collision: Union[BGKCollision, TRTCollision, MRTCollision]

    def streaming_matrix(self, k: np.ndarray) -> np.ndarray:
        """Compute the streaming matrix S(k) = diag(exp(-i k·c_j)).

        Args:
            k: wavevector of shape (dim,)

        Returns:
            S: diagonal matrix of shape (q, q)
        """
        # k · c_j for each velocity j
        kc = self.lattice.c @ k  # (q,)
        # exp(-i k·c_j)
        phases = np.exp(-1j * kc)
        return np.diag(phases)

    def __call__(self, k: np.ndarray) -> np.ndarray:
        """Compute the amplification matrix M(k) = S(k) · C."""
        S = self.streaming_matrix(k)
        C = self.collision.collision_matrix()
        return S @ C


def von_neumann_eigenvalues(
    lattice: LatticeBase,
    collision: Union[BGKCollision, TRTCollision, MRTCollision],
    k: np.ndarray,
) -> np.ndarray:
    """Compute eigenvalues of the amplification matrix M(k).

    Args:
        lattice: lattice definition
        collision: collision operator
        k: wavevector of shape (dim,)

    Returns:
        eigenvalues: complex array of shape (q,)
    """
    amp = AmplificationMatrix(lattice, collision)
    M = amp(k)
    eigenvalues = np.linalg.eigvals(M)
    return eigenvalues


def eigenvalues_to_omega(eigenvalues: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Convert eigenvalues to angular frequencies ω.

    ω = i ln(λ) / Δt

    Args:
        eigenvalues: complex eigenvalues
        dt: time step (default 1.0 for lattice units)

    Returns:
        omega: complex frequencies where Re(ω) is oscillation and Im(ω) < 0 is damping
    """
    # Principal branch of logarithm
    omega = 1j * np.log(eigenvalues + 0j) / dt
    return omega


def identify_physical_modes(
    eigenvalues: np.ndarray, lattice: LatticeBase, k: np.ndarray = None, threshold: float = 0.5
) -> dict:
    """Identify physical modes (acoustic, shear) from eigenvalue spectrum.

    Physical modes have |λ| ≈ 1, while kinetic modes have |λ| ≪ 1.

    Args:
        eigenvalues: complex eigenvalues
        lattice: lattice definition
        k: wavevector (optional, for better mode identification)
        threshold: threshold for |λ| to be considered physical

    Returns:
        dict with identified modes
    """
    omega = eigenvalues_to_omega(eigenvalues)
    abs_lambda = np.abs(eigenvalues)

    # Physical modes: |λ| > threshold
    physical_mask = abs_lambda > threshold
    physical_idx = np.where(physical_mask)[0]

    if len(physical_idx) == 0:
        return {"acoustic": [], "shear": [], "all_physical": [], "omega": omega, "eigenvalues": eigenvalues}

    physical_omega = omega[physical_idx]

    # Acoustic modes: |Re(ω)| is large relative to k
    # For small k, Re(ω) ≈ cs * k, so we use a relative threshold
    if k is not None:
        k_mag = np.linalg.norm(k)
        # Acoustic if |Re(ω)| > 0.01 * cs * k (i.e., at least 1% of expected acoustic frequency)
        omega_threshold = 0.01 * np.sqrt(lattice.cs2) * k_mag
    else:
        # Fallback: use absolute threshold
        omega_threshold = 1e-5

    acoustic_mask = np.abs(physical_omega.real) > omega_threshold
    shear_mask = ~acoustic_mask

    acoustic_idx = physical_idx[acoustic_mask]
    shear_idx = physical_idx[shear_mask]

    return {
        "acoustic": acoustic_idx,
        "shear": shear_idx,
        "all_physical": physical_idx,
        "omega": omega,
        "eigenvalues": eigenvalues,
    }


def sort_eigenvalues_by_mode(
    eigenvalues: np.ndarray, lattice: LatticeBase, k: np.ndarray = None
) -> dict:
    """Sort eigenvalues into physical modes (acoustic, shear, kinetic).

    For small k, the physical modes are:
    - 2 acoustic modes (propagating): ω ≈ ±cs|k| - i Γ k²
    - (dim) shear modes (diffusive): ω ≈ -i ν k²
    - remaining kinetic modes (highly damped)
    """
    modes = identify_physical_modes(eigenvalues, lattice, k)
    omega = modes["omega"]

    return {
        "omega": omega,
        "eigenvalues": eigenvalues,
        "acoustic_idx": modes["acoustic"],
        "shear_idx": modes["shear"],
        "physical_idx": modes["all_physical"],
    }
