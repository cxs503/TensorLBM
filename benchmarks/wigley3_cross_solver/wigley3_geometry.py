"""Wigley III modified parabolic hull geometry (Cb=0.555).

The standard Wigley parabolic form has Cb≈0.444. The modified Wigley III
uses a different vertical distribution to achieve Cb=0.555, which is a
common ITTC benchmark case.

Geometry definition:
    y(x,z) = (B/2) * (1 - ξ²) * [1 - (1-ζ)^α]
    
where:
    ξ = 2x/L ∈ [-1, 1]  (longitudinal, 0 at midship)
    ζ = z/T ∈ [0, 1]     (vertical, 0 at keel, 1 at waterline)
    α ≈ 4.97             (shape exponent for Cb=0.555)

The block coefficient is:
    Cb = (2/3) * α/(α+1) = 0.555
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


# Wigley III principal dimensions
WIGLEY3_L = 3.048  # m, length between perpendiculars
WIGLEY3_B = 0.406  # m, molded beam
WIGLEY3_T = 0.152  # m, molded draft
WIGLEY3_CB = 0.555  # block coefficient
WIGLEY3_ALPHA = 4.97  # vertical shape exponent (derived from Cb)


def wigley3_half_beam(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Compute half-beam for Wigley III hull.
    
    Parameters
    ----------
    xi : array_like
        Normalized longitudinal coordinate ∈ [-1, 1], 0 at midship.
    zeta : array_like
        Normalized vertical coordinate ∈ [0, 1], 0 at keel, 1 at waterline.
    
    Returns
    -------
    y_half : ndarray
        Half-beam normalized to B/2, range [0, 1].
    """
    xi = np.asarray(xi)
    zeta = np.asarray(zeta)
    
    # Clamp to valid range
    xi_c = np.clip(xi, -1.0, 1.0)
    zeta_c = np.clip(zeta, 0.0, 1.0)
    
    # Longitudinal distribution: parabolic
    f_xi = 1.0 - xi_c**2
    
    # Vertical distribution: modified for Cb=0.555
    f_zeta = 1.0 - (1.0 - zeta_c) ** WIGLEY3_ALPHA
    
    # Half-beam (normalized to B/2)
    y_half = f_xi * f_zeta
    
    # Zero outside hull envelope
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    return np.where(in_hull, y_half, 0.0)


def wigley3_hull_mask(
    nx: int, ny: int, nz: int,
    length: float = WIGLEY3_L,
    beam: float = WIGLEY3_B,
    draft: float = WIGLEY3_T,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate 3D boolean mask for Wigley III hull (both sides).
    
    Parameters
    ----------
    nx, ny, nz : int
        Grid resolution in x, y, z directions.
    length, beam, draft : float
        Hull dimensions [m].
    device : str
        Torch device.
    
    Returns
    -------
    mask : torch.Tensor
        Boolean tensor of shape (nx, ny, nz), True inside hull.
        y ranges from -beam/2 to +beam/2 (both sides).
    """
    # Coordinate arrays
    x = torch.linspace(-length / 2, length / 2, nx, device=device)
    y = torch.linspace(-beam / 2, beam / 2, ny, device=device)  # Both sides
    z = torch.linspace(0, draft, nz, device=device)
    
    # Meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    
    # Normalize coordinates
    xi = X / (length / 2)
    zeta = Z / draft
    
    # Half-beam at each (x, z)
    xi_np = xi.cpu().numpy()
    zeta_np = zeta.cpu().numpy()
    y_half_normalized = wigley3_half_beam(xi_np, zeta_np)
    y_half = torch.tensor(y_half_normalized, device=device) * (beam / 2)
    
    # Hull mask: |y| <= half-beam
    mask = torch.abs(Y) <= y_half
    
    return mask


def compute_block_coefficient(
    mask: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    length: float = WIGLEY3_L,
    beam: float = WIGLEY3_B,
    draft: float = WIGLEY3_T,
) -> float:
    """Compute block coefficient from a hull mask.
    
    Parameters
    ----------
    mask : torch.Tensor
        Boolean hull mask (both sides).
    dx, dy, dz : float
        Grid spacing [m].
    length, beam, draft : float
        Hull dimensions [m].
    
    Returns
    -------
    cb : float
        Block coefficient.
    """
    volume = mask.sum().item() * dx * dy * dz
    envelope = length * beam * draft
    return volume / envelope


def compute_wetted_surface(
    mask: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    """Estimate wetted surface area from hull mask.
    
    Uses face-counting method: count exposed fluid-solid faces.
    
    Parameters
    ----------
    mask : torch.Tensor
        Boolean hull mask.
    dx, dy, dz : float
        Grid spacing [m].
    
    Returns
    -------
    wetted_area : float
        Wetted surface area [m²].
    """
    # Pad mask with False (fluid)
    padded = torch.nn.functional.pad(mask.float(), (1, 1, 1, 1, 1, 1), mode="constant", value=0)
    
    # Count faces between solid and fluid
    faces = 0.0
    for dim in range(3):
        # Shift in +direction
        shifted_pos = torch.roll(padded, shifts=1, dims=dim)
        faces += ((mask.float() > 0) & (shifted_pos[1:-1, 1:-1, 1:-1] == 0)).sum().item()
        # Shift in -direction
        shifted_neg = torch.roll(padded, shifts=-1, dims=dim)
        faces += ((mask.float() > 0) & (shifted_neg[1:-1, 1:-1, 1:-1] == 0)).sum().item()
    
    # Face areas
    face_areas = [dy * dz, dx * dz, dx * dy]
    wetted_area = sum(faces * area for faces, area in zip([faces] * 3, face_areas))
    
    # Simplified: assume uniform face area
    avg_face_area = (dy * dz + dx * dz + dx * dy) / 3
    wetted_area = faces * avg_face_area
    
    return wetted_area


if __name__ == "__main__":
    # Quick validation
    print("Wigley III Geometry Validation")
    print("=" * 50)
    print(f"L = {WIGLEY3_L} m")
    print(f"B = {WIGLEY3_B} m")
    print(f"T = {WIGLEY3_T} m")
    print(f"Cb (target) = {WIGLEY3_CB}")
    print(f"α (shape exponent) = {WIGLEY3_ALPHA}")
    
    # Generate mask at moderate resolution
    nx, ny, nz = 64, 32, 16
    mask = wigley3_hull_mask(nx, ny, nz)
    
    dx = WIGLEY3_L / nx
    dy = WIGLEY3_B / ny
    dz = WIGLEY3_T / nz
    
    cb_computed = compute_block_coefficient(mask, dx, dy, dz)
    print(f"\nGrid: {nx}×{ny}×{nz}")
    print(f"Cb (computed) = {cb_computed:.4f}")
    print(f"Cb error = {abs(cb_computed - WIGLEY3_CB) / WIGLEY3_CB * 100:.2f}%")
    
    wetted = compute_wetted_surface(mask, dx, dy, dz)
    print(f"\nWetted surface ≈ {wetted:.4f} m²")
