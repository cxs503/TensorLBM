"""Force-corrected Non-Equilibrium Bounce-Back (NEBB) boundary condition.

When a body force is present in LBM, the standard half-way bounce-back (BB)
boundary condition introduces a velocity slip error at the wall. This is because
the force term contributes to the momentum balance but is not accounted for in
the BB reflection.

The force-corrected NEBB scheme adds a correction term derived from the Guo
forcing model to eliminate this slip error, achieving second-order accuracy
even with body forces.

Theory
------
Standard BB:  f_i(x_w) = f_opp(i)(x_w)  (no force correction)

Force-corrected NEBB:
    f_i(x_w) = f_opp(i)(x_w) + Δf_i^force

where the force correction is:
    Δf_i^force = (1 - 1/(2τ)) · w_i · [3(c_i - u_w) + 9(c_i·u_w)c_i] · F · δt

For a stationary wall (u_w = 0) with force F = (F_x, 0):
    Δf_i^force = (1 - 1/(2τ)) · w_i · [3·c_ix + 9·(c_i·u_w)·c_ix] · F_x

This correction ensures that the wall velocity remains zero even when a body
force drives the flow, eliminating the O(Δx) slip error of standard BB.

References
----------
* Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the
  forcing term in the lattice Boltzmann method." Phys. Rev. E, 65, 046308.
* Chen, H., & Doolen, G. D. (1998). "Lattice Boltzmann method for fluid flows."
  Annu. Rev. Fluid Mech., 30, 329-364.
* haimingz/vivsim: force-corrected-nebb implementation pattern.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# D2Q9 lattice constants (local copy for self-contained module)
CX = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=int)
CY = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=int)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=float)
OPP = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=int)
CS2 = 1.0 / 3.0


def compute_force_correction(
    tau: float,
    force_x: float,
    force_y: float,
    u_wall: np.ndarray,
    v_wall: np.ndarray,
    rho_wall: np.ndarray,
) -> np.ndarray:
    """Compute the force correction term for NEBB.

    Parameters
    ----------
    tau : float
        Relaxation time.
    force_x, force_y : float
        Body force components (per unit mass).
    u_wall, v_wall : np.ndarray, shape (n,)
        Wall velocity components.
    rho_wall : np.ndarray, shape (n,)
        Wall density.

    Returns
    -------
    correction : np.ndarray, shape (9, n)
        Force correction for each direction.
    """
    n = u_wall.size
    correction = np.zeros((9, n), dtype=float)

    # Guo forcing coefficient
    coef = (1.0 - 0.5 / tau)

    for i in range(9):
        ci_x = CX[i]
        ci_y = CY[i]
        ci_dot_u = ci_x * u_wall + ci_y * v_wall

        # Guo source term structure:
        # S_i = w_i * [3(c_i - u) + 9(c_i·u)c_i] · F
        # The correction is (1 - 1/(2τ)) * S_i
        term_x = 3.0 * (ci_x - u_wall) + 9.0 * ci_dot_u * ci_x
        term_y = 3.0 * (ci_y - v_wall) + 9.0 * ci_dot_u * ci_y

        correction[i] = coef * W[i] * (term_x * force_x + term_y * force_y)

    return correction


def apply_force_corrected_nebb(
    f: np.ndarray,
    solid_mask: np.ndarray,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
    u_wall: Optional[np.ndarray] = None,
    v_wall: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply force-corrected NEBB on solid nodes.

    For each fluid node adjacent to a solid node in direction i, the incoming
    population is set by bouncing back the outgoing population from the solid
    node, plus a force correction term.

    Parameters
    ----------
    f : np.ndarray, shape (9, ny, nx)
        Distribution functions.
    solid_mask : np.ndarray, shape (ny, nx), bool
        True for solid nodes.
    tau : float
        Relaxation time.
    force_x, force_y : float
        Body force components.
    u_wall, v_wall : np.ndarray or None
        Wall velocity (default: zero for stationary walls).

    Returns
    -------
    f : np.ndarray, shape (9, ny, nx)
        Modified distribution functions.
    """
    ny, nx = f.shape[1], f.shape[2]
    f_out = f.copy()

    # Default: stationary wall
    if u_wall is None:
        u_wall = np.zeros(nx, dtype=float)
    if v_wall is None:
        v_wall = np.zeros(nx, dtype=float)

    # Compute force correction (only needed if force is non-zero)
    has_force = abs(force_x) > 1e-15 or abs(force_y) > 1e-15
    if has_force:
        # For a horizontal wall, we need wall velocity and density
        # We'll compute these per-wall-node below
        pass

    for i in range(1, 9):
        opp = OPP[i]
        cx, cy = CX[i], CY[i]

        # Find fluid nodes whose upstream neighbor in direction i is solid
        # (pull convention: f_i at (y,x) comes from (y-cy, x-cx))
        src_y = np.arange(ny) - cy
        src_x = np.arange(nx) - cx

        # Handle periodic wrapping in x
        src_x_wrapped = src_x % nx
        src_y_clipped = np.clip(src_y, 0, ny - 1)

        # Check if source is solid
        is_solid_src = solid_mask[np.ix_(src_y_clipped, src_x_wrapped)]

        # Only apply to fluid nodes with solid upstream
        fluid_nodes = ~solid_mask
        apply_mask = is_solid_src & fluid_nodes

        if not apply_mask.any():
            continue

        # Standard bounce-back: f_i = f_opp at the same location
        f_out[i][apply_mask] = f[opp][apply_mask]

        # Force correction
        if has_force:
            # Extract wall velocity and density at affected nodes
            # For a horizontal wall at y=0 or y=ny-1, the wall velocity
            # is prescribed; for curved walls, it varies.
            # Here we use the prescribed wall velocity.
            affected_y, affected_x = np.where(apply_mask)

            # Get wall velocity at affected x positions
            u_w = u_wall[affected_x % len(u_wall)]
            v_w = v_wall[affected_x % len(v_wall)]

            # Estimate wall density from adjacent fluid nodes
            # Use the density at the fluid node itself as approximation
            rho = f.sum(axis=0)
            rho_w = rho[affected_y, affected_x]

            # Compute correction for this direction
            coef = (1.0 - 0.5 / tau)
            ci_dot_u = CX[i] * u_w + CY[i] * v_w
            term_x = 3.0 * (CX[i] - u_w) + 9.0 * ci_dot_u * CX[i]
            term_y = 3.0 * (CY[i] - v_w) + 9.0 * ci_dot_u * CY[i]
            delta_f = coef * W[i] * (term_x * force_x + term_y * force_y)

            f_out[i][apply_mask] += delta_f

    return f_out


def apply_force_corrected_nebb_1d(
    f: np.ndarray,
    wall_y: int,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
    u_wall: float = 0.0,
    v_wall: float = 0.0,
) -> np.ndarray:
    """Apply force-corrected NEBB for a 1D channel (horizontal walls).

    Optimized for the common case of plane Poiseuille flow with walls at
    y=0 and y=ny-1.

    Parameters
    ----------
    f : np.ndarray, shape (9, ny, nx)
        Distribution functions.
    wall_y : int
        Y-index of the wall (0 for bottom, ny-1 for top).
    tau : float
        Relaxation time.
    force_x, force_y : float
        Body force components.
    u_wall, v_wall : float
        Wall velocity (default: 0 for stationary walls).

    Returns
    -------
    f : np.ndarray, shape (9, ny, nx)
        Modified distribution functions.
    """
    ny, nx = f.shape[1], f.shape[2]
    f_out = f.copy()

    # Determine which directions point into the wall (unknown populations)
    if wall_y == 0:
        # Bottom wall: unknowns have c_iy > 0 (directions 3, 5, 6)
        unknown_dirs = [3, 5, 6]
        known_dirs = [0, 1, 2, 4, 7, 8]
    else:
        # Top wall: unknowns have c_iy < 0 (directions 4, 7, 8)
        unknown_dirs = [4, 7, 8]
        known_dirs = [0, 1, 2, 3, 5, 6]

    # Compute wall density from known populations
    rho_known = sum(f[k, wall_y, :] for k in known_dirs)

    # For stationary wall with v_wall = 0:
    # ρ = ρ_known + Σ_unknown f_i
    # From y-momentum: Σ_unknown f_i * c_iy = Σ_known f_i * |c_iy| (for v=0)
    # This gives us ρ

    # Simple approach: use density from adjacent fluid node
    if wall_y == 0:
        adj_y = 1
    else:
        adj_y = wall_y - 1

    rho = f.sum(axis=0)
    rho_wall = rho[adj_y, :].copy()

    # Apply NEBB with force correction
    for i in unknown_dirs:
        opp = OPP[i]
        # Standard NEBB: f_i = f_opp + (f_i^eq - f_opp^eq)
        # This preserves the non-equilibrium part

        # Compute equilibrium at wall
        w = W[i]
        ci_x = CX[i]
        ci_y = CY[i]
        ci_opp_x = CX[opp]
        ci_opp_y = CY[opp]

        u2 = u_wall**2 + v_wall**2
        ci_dot_u = ci_x * u_wall + ci_y * v_wall
        feq_i = w * rho_wall * (1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u**2 - 1.5 * u2)

        ci_opp_dot_u = ci_opp_x * u_wall + ci_opp_y * v_wall
        feq_opp = W[opp] * rho_wall * (1.0 + 3.0 * ci_opp_dot_u + 4.5 * ci_opp_dot_u**2 - 1.5 * u2)

        # NEBB: f_i = f_opp + feq_i - feq_opp
        f_out[i, wall_y, :] = f[opp, wall_y, :] + feq_i - feq_opp

        # Force correction
        if abs(force_x) > 1e-15 or abs(force_y) > 1e-15:
            coef = (1.0 - 0.5 / tau)
            term_x = 3.0 * (ci_x - u_wall) + 9.0 * ci_dot_u * ci_x
            term_y = 3.0 * (ci_y - v_wall) + 9.0 * ci_dot_u * ci_y
            delta_f = coef * w * (term_x * force_x + term_y * force_y)
            f_out[i, wall_y, :] += delta_f

    return f_out


class ForceCorrectedNEBB:
    """Force-corrected NEBB boundary condition handler.

    Parameters
    ----------
    tau : float
        Relaxation time.
    force_x, force_y : float
        Body force components.
    u_wall, v_wall : float or np.ndarray
        Wall velocity (scalar for uniform, array for spatially varying).
    """

    def __init__(
        self,
        tau: float,
        force_x: float = 0.0,
        force_y: float = 0.0,
        u_wall: float = 0.0,
        v_wall: float = 0.0,
    ):
        self.tau = tau
        self.force_x = force_x
        self.force_y = force_y
        self.u_wall = u_wall
        self.v_wall = v_wall

    def apply(self, f: np.ndarray, solid_mask: np.ndarray) -> np.ndarray:
        """Apply force-corrected NEBB.

        Parameters
        ----------
        f : np.ndarray, shape (9, ny, nx)
        solid_mask : np.ndarray, shape (ny, nx), bool

        Returns
        -------
        f : np.ndarray, shape (9, ny, nx)
        """
        return apply_force_corrected_nebb(
            f, solid_mask, self.tau,
            self.force_x, self.force_y,
            self._broadcast_u(), self._broadcast_v(),
        )

    def _broadcast_u(self) -> np.ndarray:
        if isinstance(self.u_wall, np.ndarray):
            return self.u_wall
        return np.array([self.u_wall])

    def _broadcast_v(self) -> np.ndarray:
        if isinstance(self.v_wall, np.ndarray):
            return self.v_wall
        return np.array([self.v_wall])
