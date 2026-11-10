"""Mei-Luo-Shyy weighted isotropic gradient operators for D2Q9.

The standard 4-point central difference gradient on a square lattice has
O(h²) anisotropy error that manifests as direction-dependent spurious
currents at curved interfaces.  The Mei-Luo-Shyy (1997) approach uses
weighted sums over extended stencils to achieve higher-order isotropy.

**9-point stencil (D2Q9 velocities):**
    ∂f/∂x ≈ (1/cs²) Σᵢ wᵢ c_ix f(x + cᵢ)

where wᵢ are the D2Q9 weights and cᵢ the lattice velocities.  This is
exact to O(h²) and isotropic to 4th order (the error tensor is proportional
to δ_αβ, not a higher-rank anisotropic tensor).

**13-point stencil (D2Q9 + 4 next-nearest neighbours at |c|=2):**
    ∂f/∂x ≈ (1/cs²) [Σᵢ₌₀⁸ wᵢ c_ix f(x+cᵢ) + w₂ Σⱼ c₂ⱼₓ f(x+c₂ⱼ)]

where c₂ⱼ ∈ {(±2,0), (0,±2)} with weight w₂ chosen so that the second-order
moment Σ wᵢ cᵢα cᵢβ = cs² δ_αβ is preserved.

References
----------
Mei, Luo & Shyy (1997) J. Comput. Phys. 134, 246–253.
Shan & Chen (1993) Phys. Rev. E 47, 1815.
"""

from __future__ import annotations

import torch

# D2Q9 lattice velocities and weights (standard)
_D2Q9_C = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=torch.int64)

_D2Q9_W = torch.tensor([
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=torch.float64)

# Speed of sound squared
_CS2 = 1.0 / 3.0


def gradient_9point_mls(
    field: torch.Tensor,
    periodic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient using 9-point Mei-Luo-Shyy isotropic stencil.

    ∂f/∂x ≈ (1/cs²) Σᵢ wᵢ c_ix f(x + cᵢ)

    This is the standard D2Q9 weighted isotropic gradient.  For a smooth
    field, the truncation error is O(h²) and the error tensor is isotropic
    (proportional to δ_αβ ∇²f), so there is no directional bias.

    Args:
        field: 2D scalar field of shape (ny, nx).
        periodic: If True, use periodic boundary conditions (roll).

    Returns:
        (dfdx, dfdy) each of shape (ny, nx).
    """
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got {field.ndim}D")

    device = field.device
    dtype = field.dtype
    ny, nx = field.shape

    # Optimized vectorized implementation using torch.roll
    # D2Q9 velocities: (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)
    # Weights: 4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36

    # For gradient, we need Σ wᵢ cᵢ f(x+cᵢ)
    # c_x values: 0, 1, 0, -1, 0, 1, -1, -1, 1
    # c_y values: 0, 0, 1, 0, -1, 1, 1, -1, -1

    # Initialize sums
    grad_x = torch.zeros(ny, nx, device=device, dtype=dtype)
    grad_y = torch.zeros(ny, nx, device=device, dtype=dtype)

    # Process each direction
    # i=0: c=(0,0), w=4/9 → no contribution to gradient
    # i=1: c=(1,0), w=1/9
    grad_x += (1.0/9.0) * torch.roll(field, shifts=(0, -1), dims=(0, 1))
    # i=2: c=(0,1), w=1/9
    grad_y += (1.0/9.0) * torch.roll(field, shifts=(-1, 0), dims=(0, 1))
    # i=3: c=(-1,0), w=1/9
    grad_x -= (1.0/9.0) * torch.roll(field, shifts=(0, 1), dims=(0, 1))
    # i=4: c=(0,-1), w=1/9
    grad_y -= (1.0/9.0) * torch.roll(field, shifts=(1, 0), dims=(0, 1))
    # i=5: c=(1,1), w=1/36
    grad_x += (1.0/36.0) * torch.roll(field, shifts=(-1, -1), dims=(0, 1))
    grad_y += (1.0/36.0) * torch.roll(field, shifts=(-1, -1), dims=(0, 1))
    # i=6: c=(-1,1), w=1/36
    grad_x -= (1.0/36.0) * torch.roll(field, shifts=(-1, 1), dims=(0, 1))
    grad_y += (1.0/36.0) * torch.roll(field, shifts=(-1, 1), dims=(0, 1))
    # i=7: c=(-1,-1), w=1/36
    grad_x -= (1.0/36.0) * torch.roll(field, shifts=(1, 1), dims=(0, 1))
    grad_y -= (1.0/36.0) * torch.roll(field, shifts=(1, 1), dims=(0, 1))
    # i=8: c=(1,-1), w=1/36
    grad_x += (1.0/36.0) * torch.roll(field, shifts=(1, -1), dims=(0, 1))
    grad_y -= (1.0/36.0) * torch.roll(field, shifts=(1, -1), dims=(0, 1))

    # Normalize by cs²
    grad_x /= _CS2
    grad_y /= _CS2

    return grad_x, grad_y


def gradient_13point_mls(
    field: torch.Tensor,
    periodic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient using 13-point Mei-Luo-Shyy isotropic stencil.

    Extends the 9-point D2Q9 gradient with 4 next-nearest neighbours at
    |c|=2.  The additional stencil points reduce the O(h⁴) anisotropy error.

    Args:
        field: 2D scalar field of shape (ny, nx).
        periodic: If True, use periodic boundary conditions.

    Returns:
        (dfdx, dfdy) each of shape (ny, nx).
    """
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got {field.ndim}D")

    # Start with 9-point gradient
    grad_x, grad_y = gradient_9point_mls(field, periodic)

    device = field.device
    dtype = field.dtype

    # Add 4 next-nearest neighbours at |c|=2: (2,0), (0,2), (-2,0), (0,-2)
    # Weight: 1/144 each
    w2 = 1.0 / 144.0

    # (2,0): c_x=2, c_y=0
    grad_x += 2.0 * w2 * torch.roll(field, shifts=(0, -2), dims=(0, 1))
    # (0,2): c_x=0, c_y=2
    grad_y += 2.0 * w2 * torch.roll(field, shifts=(-2, 0), dims=(0, 1))
    # (-2,0): c_x=-2, c_y=0
    grad_x -= 2.0 * w2 * torch.roll(field, shifts=(0, 2), dims=(0, 1))
    # (0,-2): c_x=0, c_y=-2
    grad_y -= 2.0 * w2 * torch.roll(field, shifts=(2, 0), dims=(0, 1))

    # Renormalize by cs²_eff = 7/18
    cs2_eff = 7.0 / 18.0
    grad_x *= (_CS2 / cs2_eff)
    grad_y *= (_CS2 / cs2_eff)

    return grad_x, grad_y


def laplacian_9point_isotropic(
    field: torch.Tensor,
    periodic: bool = True,
) -> torch.Tensor:
    """Compute Laplacian using 9-point isotropic stencil.

    ∇²f ≈ (2/cs²) Σᵢ wᵢ [f(x+cᵢ) - f(x)]

    This is the standard D2Q9 weighted Laplacian, isotropic to O(h²).

    Args:
        field: 2D scalar field of shape (ny, nx).
        periodic: If True, use periodic boundary conditions.

    Returns:
        Laplacian of shape (ny, nx).
    """
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got {field.ndim}D")

    device = field.device
    dtype = field.dtype
    ny, nx = field.shape

    # Sum of weighted neighbours
    neighbour_sum = torch.zeros(ny, nx, device=device, dtype=dtype)

    # i=0: c=(0,0), w=4/9
    neighbour_sum += (4.0/9.0) * field
    # i=1: c=(1,0), w=1/9
    neighbour_sum += (1.0/9.0) * torch.roll(field, shifts=(0, -1), dims=(0, 1))
    # i=2: c=(0,1), w=1/9
    neighbour_sum += (1.0/9.0) * torch.roll(field, shifts=(-1, 0), dims=(0, 1))
    # i=3: c=(-1,0), w=1/9
    neighbour_sum += (1.0/9.0) * torch.roll(field, shifts=(0, 1), dims=(0, 1))
    # i=4: c=(0,-1), w=1/9
    neighbour_sum += (1.0/9.0) * torch.roll(field, shifts=(1, 0), dims=(0, 1))
    # i=5: c=(1,1), w=1/36
    neighbour_sum += (1.0/36.0) * torch.roll(field, shifts=(-1, -1), dims=(0, 1))
    # i=6: c=(-1,1), w=1/36
    neighbour_sum += (1.0/36.0) * torch.roll(field, shifts=(-1, 1), dims=(0, 1))
    # i=7: c=(-1,-1), w=1/36
    neighbour_sum += (1.0/36.0) * torch.roll(field, shifts=(1, 1), dims=(0, 1))
    # i=8: c=(1,-1), w=1/36
    neighbour_sum += (1.0/36.0) * torch.roll(field, shifts=(1, -1), dims=(0, 1))

    # Laplacian = (2/cs²) * (neighbour_sum - f * Σwᵢ)
    # Note: Σwᵢ = 1, so this is (2/cs²) * (neighbour_sum - f)
    laplacian = (2.0 / _CS2) * (neighbour_sum - field)

    return laplacian
