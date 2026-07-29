"""RANS wall function coupling for LBM.

Complementary to wall_model.py: after wall_function_3d computes u_tau
from the log-law, this module sets the RANS (k-ε / k-ω) near-wall
boundary values that OpenFOAM-style wall functions require.

Ref: OpenFOAM nutkWallFunction, epsilonWallFunction, kqRWallFunction.
"""
import torch

# Standard k-ε constants (same as rans_ke.py)
_C_MU = 0.09
_KAPPA = 0.41


def set_rans_wall_bc_k_epsilon(
    k: torch.Tensor,
    epsilon: torch.Tensor,
    u_tau: torch.Tensor,
    near_wall: torch.Tensor,
    nu: float,
    y_val: float = 0.5,
) -> None:
    """Set k and ε boundary values at near-wall cells (in-place).

    OpenFOAM-equivalent wall functions:
      k_wall      = u_tau² / √C_μ
      ε_wall      = u_tau³ / (κ · y)
      ω_wall      = u_tau / (√C_μ · κ · y)   (for k-ω SST)

    Applied ONLY to near-wall fluid cells (first off-wall cell).

    Args:
        k:           TKE field, shape (nz, ny, nx).  Modified in-place.
        epsilon:     Dissipation field.  Modified in-place.
        u_tau:       Friction velocity from wall_model, shape (nz, ny, nx).
        near_wall:   Boolean mask marking the first off-wall fluid cells.
        nu:          Laminar kinematic viscosity.
        y_val:       Wall-normal distance (default 0.5 lu = half cell).
    """
    mask = near_wall & (u_tau > 1e-12)

    # k_wall = u_tau² / √C_μ
    k_wall = (u_tau * u_tau) / (_C_MU ** 0.5)

    # ε_wall = u_tau³ / (κ · y)
    eps_wall = (u_tau * u_tau * u_tau) / (_KAPPA * y_val)

    # Floor: never go below the molecular-diffusion limit
    eps_min = 1e-12
    k_wall = k_wall.clamp(min=1e-8)
    eps_wall = eps_wall.clamp(min=eps_min)

    k[mask] = k_wall[mask]
    epsilon[mask] = eps_wall[mask]


def compute_near_wall_from_solid(solid: torch.Tensor) -> torch.Tensor:
    """Return mask of fluid cells adjacent to solid (same as wall_function_3d).

    Args:
        solid: Boolean solid mask of shape (nz, ny, nx).
    Returns:
        near_wall: Boolean mask of shape (nz, ny, nx).
    """
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    return near
