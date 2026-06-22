"""Vortex identification criteria for LBM velocity fields.

Implements three standard vortex identification methods used in industrial
CFD post-processing (PowerFlow, XFlow, Fluent, OpenFOAM):

1. **Q-criterion** (Hunt et al. 1988) – vortices are regions where the
   second invariant Q of the velocity gradient tensor is positive (rotation
   dominates strain).

2. **λ2-criterion** (Jeong & Hussain 1995) – vortices are regions where the
   second eigenvalue λ₂ of (S² + Ω²) is negative.

3. **Ω-criterion** (Liu et al. 2016) – normalised vorticity dominance ratio
   Ω = ‖Ω‖² / (‖S‖² + ‖Ω‖² + ε).  Vortex cores: Ω > 0.52.

Both 2-D and 3-D versions are provided for all criteria.

References
----------
Hunt, J. C. R., Wray, A. A., & Moin, P. (1988).
    Eddies, streams, and convergence zones in turbulent flows. *CTR Report*.
Jeong, J., & Hussain, F. (1995).
    On the identification of a vortex. *J. Fluid Mech.* 285, 69–94.
Liu, C., Wang, Y., Yang, Y., & Duan, Z. (2016).
    New omega vortex identification method. *Sci. China Phys.* 59, 684711.
"""
from __future__ import annotations

import torch

__all__ = [
    "q_criterion_2d",
    "lambda2_criterion_2d",
    "omega_criterion_2d",
    "q_criterion_3d",
    "lambda2_criterion_3d",
    "omega_criterion_3d",
    "vortex_fields_2d",
    "vortex_fields_3d",
]


# ---------------------------------------------------------------------------
# 2-D implementations
# ---------------------------------------------------------------------------

def _velocity_gradients_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 2-D velocity gradient components (central differences on unit grid).

    Returns:
        ``(dux_dx, dux_dy, duy_dx, duy_dy)``
    """
    dux_dx = torch.zeros_like(ux)
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    duy_dy = torch.zeros_like(uy)

    dux_dx[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / 2.0
    dux_dy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / 2.0
    duy_dx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / 2.0
    duy_dy[1:-1, :] = (uy[2:, :] - uy[:-2, :]) / 2.0

    return dux_dx, dux_dy, duy_dx, duy_dy


def q_criterion_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Compute the Q-criterion field for a 2-D velocity field.

    Q = 0.5 * (‖Ω‖² − ‖S‖²)

    For 2-D:
        S_xx = ∂ux/∂x,  S_yy = ∂uy/∂y,  S_xy = 0.5(∂ux/∂y + ∂uy/∂x)
        Ω_xy = 0.5(∂uy/∂x − ∂ux/∂y)

    Positive Q indicates vortex cores.

    Args:
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.

    Returns:
        Q-criterion field, shape ``(ny, nx)``.
    """
    dux_dx, dux_dy, duy_dx, duy_dy = _velocity_gradients_2d(ux, uy)

    # Strain-rate tensor
    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)

    # Rotation-rate tensor (anti-symmetric part)
    omega_xy = 0.5 * (duy_dx - dux_dy)

    norm_S2 = s_xx ** 2 + s_yy ** 2 + 2.0 * s_xy ** 2
    norm_O2 = 2.0 * omega_xy ** 2

    return 0.5 * (norm_O2 - norm_S2)


def lambda2_criterion_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Compute the λ₂-criterion for a 2-D velocity field.

    The λ₂ criterion uses the second eigenvalue of (S² + Ω²).  For 2-D,
    the symmetric matrix A = S² + Ω² has a closed-form eigenvalue solution.

    Negative λ₂ indicates vortex cores.

    Args:
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.

    Returns:
        λ₂ field (second/larger eigenvalue), shape ``(ny, nx)``.
    """
    dux_dx, dux_dy, duy_dx, duy_dy = _velocity_gradients_2d(ux, uy)

    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)
    omega_xy = 0.5 * (duy_dx - dux_dy)

    # A = S² + Ω²  (2×2 symmetric)
    # S² = [[s_xx²+s_xy², s_xx*s_xy+s_xy*s_yy], [...], [s_xy²+s_yy²]]
    # Ω² = [[-omega_xy², 0], [0, -omega_xy²]]
    a11 = s_xx * s_xx + s_xy * s_xy - omega_xy * omega_xy
    a12 = s_xx * s_xy + s_xy * s_yy
    a22 = s_xy * s_xy + s_yy * s_yy - omega_xy * omega_xy

    # Eigenvalues of 2×2 symmetric: λ = (tr ± √(tr²-4det))/2
    tr = a11 + a22
    det = a11 * a22 - a12 * a12
    discriminant = torch.clamp(tr * tr - 4.0 * det, min=0.0)
    sqrt_disc = torch.sqrt(discriminant)
    # λ₂ is the larger (less negative) eigenvalue
    lam2 = (tr + sqrt_disc) / 2.0
    return lam2


def omega_criterion_2d(
    ux: torch.Tensor, uy: torch.Tensor, eps_factor: float = 1e-5
) -> torch.Tensor:
    """Compute the Ω-vortex criterion for a 2-D velocity field.

    Ω = ‖Ω‖² / (‖S‖² + ‖Ω‖² + ε)

    where ε is a small stabiliser proportional to the peak of ‖S‖² + ‖Ω‖².
    Values Ω > 0.52 indicate vortex cores.

    Args:
        ux:         x-velocity, shape ``(ny, nx)``.
        uy:         y-velocity, shape ``(ny, nx)``.
        eps_factor: Fraction of max(‖S‖² + ‖Ω‖²) used as ε (default 1e-5).

    Returns:
        Ω-criterion field in [0, 1], shape ``(ny, nx)``.
    """
    dux_dx, dux_dy, duy_dx, duy_dy = _velocity_gradients_2d(ux, uy)

    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)
    omega_xy = 0.5 * (duy_dx - dux_dy)

    norm_S2 = s_xx ** 2 + s_yy ** 2 + 2.0 * s_xy ** 2
    norm_O2 = 2.0 * omega_xy ** 2

    eps = eps_factor * float((norm_S2 + norm_O2).max().item()) + 1e-30
    return norm_O2 / (norm_S2 + norm_O2 + eps)


# ---------------------------------------------------------------------------
# 3-D implementations
# ---------------------------------------------------------------------------

def _velocity_gradients_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return 3-D velocity gradient tensor components (9 entries)."""
    grads: dict[str, torch.Tensor] = {}
    for name, u in [("ux", ux), ("uy", uy), ("uz", uz)]:
        g_dx = torch.zeros_like(u)
        g_dy = torch.zeros_like(u)
        g_dz = torch.zeros_like(u)
        g_dx[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / 2.0
        g_dy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / 2.0
        g_dz[1:-1, :, :] = (u[2:, :, :] - u[:-2, :, :]) / 2.0
        grads[f"d{name}_dx"] = g_dx
        grads[f"d{name}_dy"] = g_dy
        grads[f"d{name}_dz"] = g_dz
    return grads


def q_criterion_3d(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """Q-criterion for a 3-D velocity field.

    Q = 0.5 * (‖Ω‖² − ‖S‖²)

    Args:
        ux, uy, uz: Velocity components, shape ``(nz, ny, nx)``.

    Returns:
        Q-criterion field, shape ``(nz, ny, nx)``.
    """
    g = _velocity_gradients_3d(ux, uy, uz)

    # Strain-rate (symmetric part)
    s_xx = g["dux_dx"]
    s_yy = g["duy_dy"]
    s_zz = g["duz_dz"]
    s_xy = 0.5 * (g["dux_dy"] + g["duy_dx"])
    s_xz = 0.5 * (g["dux_dz"] + g["duz_dx"])
    s_yz = 0.5 * (g["duy_dz"] + g["duz_dy"])

    # Rotation-rate (anti-symmetric part)
    o_xy = 0.5 * (g["duy_dx"] - g["dux_dy"])
    o_xz = 0.5 * (g["duz_dx"] - g["dux_dz"])
    o_yz = 0.5 * (g["duz_dy"] - g["duy_dz"])

    norm_S2 = (s_xx ** 2 + s_yy ** 2 + s_zz ** 2
               + 2.0 * (s_xy ** 2 + s_xz ** 2 + s_yz ** 2))
    norm_O2 = 2.0 * (o_xy ** 2 + o_xz ** 2 + o_yz ** 2)

    return 0.5 * (norm_O2 - norm_S2)


def lambda2_criterion_3d(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """λ₂-criterion for a 3-D velocity field.

    Computes the second eigenvalue of the symmetric tensor A = S² + Ω² at
    each grid point using PyTorch's batched ``torch.linalg.eigvalsh``.

    Args:
        ux, uy, uz: Velocity components, shape ``(nz, ny, nx)``.

    Returns:
        λ₂ field (second eigenvalue), shape ``(nz, ny, nx)``.
    """
    g = _velocity_gradients_3d(ux, uy, uz)
    nz, ny, nx = ux.shape
    device = ux.device

    s_xx = g["dux_dx"].reshape(-1)
    s_yy = g["duy_dy"].reshape(-1)
    s_zz = g["duz_dz"].reshape(-1)
    s_xy = 0.5 * (g["dux_dy"] + g["duy_dx"]).reshape(-1)
    s_xz = 0.5 * (g["dux_dz"] + g["duz_dx"]).reshape(-1)
    s_yz = 0.5 * (g["duy_dz"] + g["duz_dy"]).reshape(-1)

    o_xy = 0.5 * (g["duy_dx"] - g["dux_dy"]).reshape(-1)
    o_xz = 0.5 * (g["duz_dx"] - g["dux_dz"]).reshape(-1)
    o_yz = 0.5 * (g["duz_dy"] - g["duy_dz"]).reshape(-1)

    N = nz * ny * nx
    # Build (N, 3, 3) tensor A = S² + Ω²
    # S = [[s_xx, s_xy, s_xz],[s_xy, s_yy, s_yz],[s_xz, s_yz, s_zz]]
    # Ω = [[0, o_xy, o_xz],[-o_xy, 0, o_yz],[-o_xz, -o_yz, 0]]
    S = torch.stack([
        torch.stack([s_xx, s_xy, s_xz], dim=1),
        torch.stack([s_xy, s_yy, s_yz], dim=1),
        torch.stack([s_xz, s_yz, s_zz], dim=1),
    ], dim=1)  # (N, 3, 3)

    Om = torch.stack([
        torch.stack([torch.zeros(N, device=device), o_xy, o_xz], dim=1),
        torch.stack([-o_xy, torch.zeros(N, device=device), o_yz], dim=1),
        torch.stack([-o_xz, -o_yz, torch.zeros(N, device=device)], dim=1),
    ], dim=1)  # (N, 3, 3)

    A = torch.bmm(S, S) + torch.bmm(Om, Om)  # (N, 3, 3) – symmetric
    # Symmetrize for numerical stability
    A = 0.5 * (A + A.transpose(1, 2))

    eigenvalues = torch.linalg.eigvalsh(A)  # (N, 3) sorted ascending
    lam2 = eigenvalues[:, 1].reshape(nz, ny, nx)  # second (middle) eigenvalue
    return lam2


def omega_criterion_3d(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    eps_factor: float = 1e-5,
) -> torch.Tensor:
    """Ω-vortex criterion for a 3-D velocity field.

    Args:
        ux, uy, uz: Velocity components, shape ``(nz, ny, nx)``.
        eps_factor: Stabiliser fraction.

    Returns:
        Ω field in [0, 1], shape ``(nz, ny, nx)``.
    """
    g = _velocity_gradients_3d(ux, uy, uz)

    s_xx = g["dux_dx"]
    s_yy = g["duy_dy"]
    s_zz = g["duz_dz"]
    s_xy = 0.5 * (g["dux_dy"] + g["duy_dx"])
    s_xz = 0.5 * (g["dux_dz"] + g["duz_dx"])
    s_yz = 0.5 * (g["duy_dz"] + g["duz_dy"])
    o_xy = 0.5 * (g["duy_dx"] - g["dux_dy"])
    o_xz = 0.5 * (g["duz_dx"] - g["dux_dz"])
    o_yz = 0.5 * (g["duz_dy"] - g["duy_dz"])

    norm_S2 = s_xx**2+s_yy**2+s_zz**2+2.0*(s_xy**2+s_xz**2+s_yz**2)
    norm_O2 = 2.0*(o_xy**2+o_xz**2+o_yz**2)

    eps = eps_factor * float((norm_S2+norm_O2).max().item()) + 1e-30
    return norm_O2 / (norm_S2+norm_O2+eps)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def vortex_fields_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, list]:
    """Compute all three 2-D vortex criterion fields.

    Args:
        ux:   x-velocity, shape ``(ny, nx)``.
        uy:   y-velocity, shape ``(ny, nx)``.
        mask: Optional solid mask; zeroes out solid cells.

    Returns:
        Dictionary with keys ``q``, ``lambda2``, ``omega`` as nested float lists.
    """
    q = q_criterion_2d(ux, uy)
    l2 = lambda2_criterion_2d(ux, uy)
    om = omega_criterion_2d(ux, uy)

    if mask is not None:
        fluid = (~mask).float()
        q = q * fluid
        l2 = l2 * fluid
        om = om * fluid

    return {
        "q": q.cpu().tolist(),
        "lambda2": l2.cpu().tolist(),
        "omega": om.cpu().tolist(),
    }


def vortex_fields_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    mask: torch.Tensor | None = None,
    criteria: list[str] | None = None,
) -> dict[str, list]:
    """Compute vortex criterion fields for a 3-D velocity field.

    Args:
        ux, uy, uz: Velocity components, shape ``(nz, ny, nx)``.
        mask:       Optional solid mask.
        criteria:   Subset of ``['q', 'lambda2', 'omega']`` to compute
                    (default: all three).

    Returns:
        Dictionary with requested criterion fields as nested float lists.
    """
    if criteria is None:
        criteria = ["q", "lambda2", "omega"]

    result: dict[str, list] = {}
    fluid = None if mask is None else (~mask).float()

    if "q" in criteria:
        q = q_criterion_3d(ux, uy, uz)
        if fluid is not None:
            q = q * fluid
        result["q"] = q.cpu().tolist()

    if "lambda2" in criteria:
        l2 = lambda2_criterion_3d(ux, uy, uz)
        if fluid is not None:
            l2 = l2 * fluid
        result["lambda2"] = l2.cpu().tolist()

    if "omega" in criteria:
        om = omega_criterion_3d(ux, uy, uz)
        if fluid is not None:
            om = om * fluid
        result["omega"] = om.cpu().tolist()

    return result
