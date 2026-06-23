"""Proper Orthogonal Decomposition (POD) via the method of snapshots.

POD (also known as Principal Component Analysis applied to flow fields)
extracts the dominant coherent structures from an ensemble of simulation
snapshots.  The first few POD modes typically capture most of the turbulent
kinetic energy and reveal the large-scale dynamics of the flow.

This implementation uses the *method of snapshots* (Sirovich, 1987), which
is computationally efficient when the number of snapshots N is much smaller
than the number of spatial grid points M (N ≪ M):

1. Compute the mean field and subtract it from each snapshot.
2. Assemble the (N × N) snapshot correlation matrix  C = X^T X / N
   where X is the (M × N) centred-snapshot matrix.
3. Solve the eigenvalue problem  C v = λ v  (via SVD of X).
4. Reconstruct the spatial modes  Φ_k = X v_k / (√(N λ_k)).

The temporal coefficients (scores) are returned as  a_k(t) = Φ_k^T x(t).

References
----------
Sirovich, L. (1987). "Turbulence and the dynamics of coherent structures."
    *Quarterly of Applied Mathematics* 45(3), 561–590.
Lumley, J.L. (1967). "The structure of inhomogeneous turbulent flows."
    *Atmospheric Turbulence and Radio Wave Propagation*, 166–178.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "PODResult",
    "compute_pod",
    "reconstruct_field",
    "pod_reconstruction_error",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PODResult:
    """Container for POD decomposition results.

    Attributes
    ----------
    modes:
        Spatial POD modes, shape ``(n_modes, *spatial_shape)``.
    singular_values:
        Singular values σ_k (ordered descending).
    energy_fraction:
        Fractional energy content of each mode: σ_k² / Σ σ_i².
    cumulative_energy:
        Cumulative energy fraction (useful for determining how many modes
        are needed to capture a given percentage of the variance).
    temporal_coefficients:
        Projection of each snapshot onto each mode.
        Shape ``(n_snapshots, n_modes)``.
    mean_field:
        Time-mean field subtracted before decomposition.
        Shape ``(*spatial_shape)``.
    n_snapshots:
        Number of snapshots used.
    n_modes:
        Number of modes retained.
    spatial_shape:
        Original spatial shape of each snapshot.
    """

    modes: torch.Tensor
    singular_values: list[float]
    energy_fraction: list[float]
    cumulative_energy: list[float]
    temporal_coefficients: list[list[float]]
    mean_field: list[float]
    n_snapshots: int
    n_modes: int
    spatial_shape: list[int]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_pod(
    snapshots: torch.Tensor | list[torch.Tensor],
    n_modes: int = 10,
    *,
    subtract_mean: bool = True,
    return_coefficients: bool = True,
) -> PODResult:
    """Perform POD on a set of flow-field snapshots.

    Parameters
    ----------
    snapshots:
        Either a stacked tensor of shape ``(N, *spatial_dims)`` or a list of
        N tensors each with the same shape ``(*spatial_dims)``.  The spatial
        field can be 1-D, 2-D, or 3-D.
    n_modes:
        Number of POD modes to retain (capped at min(N, M)).
    subtract_mean:
        Whether to subtract the ensemble mean before decomposition.
    return_coefficients:
        Whether to compute and return temporal coefficients.

    Returns
    -------
    PODResult
    """
    if isinstance(snapshots, list):
        if not snapshots:
            raise ValueError("snapshots list is empty")
        if len(snapshots) < 2:
            raise ValueError("At least 2 snapshots are required for POD")
        data = torch.stack([s.float() for s in snapshots], dim=0)  # (N, ...)
    else:
        data = snapshots.float()

    if data.dim() < 2:
        raise ValueError("snapshots must have at least 2 dimensions (N, M)")

    n_snaps = data.shape[0]
    spatial_shape = list(data.shape[1:])
    m_pts = data[0].numel()

    # Flatten spatial dimensions: (N, M)
    X = data.reshape(n_snaps, m_pts)  # (N, M)

    mean_field: torch.Tensor
    if subtract_mean:
        mean_field = X.mean(dim=0)  # (M,)
        X = X - mean_field.unsqueeze(0)
    else:
        mean_field = torch.zeros(m_pts)

    # SVD of (N, M) matrix: U (N, N), S (K,), Vh (K, M) where K = min(N, M)
    # Use economy SVD (driver='gesvd' or full_matrices=False equivalent)
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        # Fallback: transpose if N > M (cheaper)
        C = X @ X.T / n_snaps  # (N, N) correlation matrix
        eigvals, eigvecs = torch.linalg.eigh(C)
        # eigh returns ascending order; reverse
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)
        S = eigvals.clamp(min=0.0).sqrt() * math.sqrt(n_snaps)
        # Reconstruct Vh from eigenvectors
        Vh_rows: list[torch.Tensor] = []
        for k in range(min(n_snaps, m_pts)):
            if float(S[k]) > 1e-12:
                phi = X.T @ eigvecs[:, k] / (float(S[k]))
                phi = phi / (phi.norm() + 1e-12)
            else:
                phi = torch.zeros(m_pts)
            Vh_rows.append(phi)
        Vh = torch.stack(Vh_rows, dim=0)  # (K, M)
        U = eigvecs  # (N, N)

    k_max = min(n_modes, len(S), n_snaps, m_pts)
    n_modes_actual = k_max

    # Modes: rows of Vh (already unit-norm spatial modes)
    modes_flat = Vh[:k_max]  # (n_modes, M)
    modes = modes_flat.reshape(n_modes_actual, *spatial_shape)

    sigma = S[:k_max]
    total_energy = float((S ** 2).sum()) + 1e-30
    energy_frac = [(float(s ** 2) / total_energy) for s in sigma]
    cumulative = []
    cum = 0.0
    for ef in energy_frac:
        cum += ef
        cumulative.append(cum)

    temporal_coeffs: list[list[float]] = []
    if return_coefficients:
        # a_k(t_i) = X[i] · Φ_k   shape (N, n_modes)
        coeff_mat = X @ modes_flat.T  # (N, n_modes)
        temporal_coeffs = coeff_mat.tolist()

    return PODResult(
        modes=modes,
        singular_values=sigma.tolist(),
        energy_fraction=energy_frac,
        cumulative_energy=cumulative,
        temporal_coefficients=temporal_coeffs,
        mean_field=mean_field.tolist(),
        n_snapshots=n_snaps,
        n_modes=n_modes_actual,
        spatial_shape=spatial_shape,
    )


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def reconstruct_field(
    result: PODResult,
    snapshot_index: int,
    n_modes: int | None = None,
) -> torch.Tensor:
    """Reconstruct a flow field from POD modes.

    Parameters
    ----------
    result:
        A :class:`PODResult` returned by :func:`compute_pod`.
    snapshot_index:
        Index of the snapshot to reconstruct.
    n_modes:
        Number of leading modes to use.  Uses all retained modes by default.

    Returns
    -------
    torch.Tensor
        Reconstructed field with the mean re-added, shape ``(*spatial_shape)``.
    """
    n = n_modes if n_modes is not None else result.n_modes
    n = min(n, result.n_modes)

    modes_flat = result.modes.reshape(result.n_modes, -1)  # (n_modes, M)
    coeffs = torch.tensor(result.temporal_coefficients[snapshot_index])[:n]  # (n,)
    rec = (coeffs.unsqueeze(1) * modes_flat[:n]).sum(dim=0)  # (M,)
    mean = torch.tensor(result.mean_field)
    return (rec + mean).reshape(result.spatial_shape)


def pod_reconstruction_error(
    original: torch.Tensor,
    result: PODResult,
    snapshot_index: int,
    n_modes: int | None = None,
) -> float:
    """Compute the relative L2 reconstruction error for a single snapshot."""
    rec = reconstruct_field(result, snapshot_index, n_modes=n_modes)
    orig_flat = original.float().flatten()
    rec_flat = rec.float().flatten()
    error = float((orig_flat - rec_flat).norm()) / (float(orig_flat.norm()) + 1e-12)
    return error
