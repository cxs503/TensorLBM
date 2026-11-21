"""Suction wall boundary condition for LBM.

Implements a wall with prescribed normal mass flux (suction/blowing)
and tangential velocity. This is the key BC for ASBL (Asymptotic
Suction Boundary Layer) verification.

Physics:
    At the wall (y=0):
        u_tangential = U∞  (free-stream velocity)
        v_normal = -V       (suction velocity, V > 0 means suction into wall)

    The ASBL exact solution:
        u(y) = U∞ (1 - exp(-V y / ν))
        C_f = 2V / U∞  (exact, independent of Re_x)

Implementation:
    Zou-He-style Dirichlet BC for the wall-normal boundary.
    Unknown populations are reconstructed from:
    1. Prescribed velocity (u=U∞, v=-V)
    2. Density extrapolation from interior
    3. Non-equilibrium bounce-back for unknowns
"""

import torch
from .d2q9 import C2Q9, W2Q9, OPPOSITE_2Q9


class SuctionWallBC:
    """Suction wall boundary condition for D2Q9 LBM.

    Parameters
    ----------
    U_inf : float
        Free-stream (tangential) velocity.
    V_suction : float
        Suction velocity (positive = into wall).
    tau : float
        Relaxation time.
    wall_y : int, default 0
        Y-index of the wall row.
    """

    def __init__(self, U_inf: float, V_suction: float, tau: float, wall_y: int = 0):
        self.U_inf = U_inf
        self.V_suction = V_suction
        self.tau = tau
        self.wall_y = wall_y

        # For bottom wall (wall_y=0), unknown populations are those
        # coming from below: directions 3 (north), 6 (NW), 7 (NE)
        # i.e., c_iy > 0
        self._setup_unknown_directions()

    def _setup_unknown_directions(self):
        """Identify unknown populations at the wall."""
        c = C2Q9
        if self.wall_y == 0:
            # Bottom wall: unknowns have c_iy > 0
            self.unknown_dirs = [i for i in range(9) if c[i, 1] > 0]
        else:
            # Top wall: unknowns have c_iy < 0
            self.unknown_dirs = [i for i in range(9) if c[i, 1] < 0]

    def apply(self, f: torch.Tensor) -> torch.Tensor:
        """Apply suction wall BC to distribution functions.

        Uses Zou-He Dirichlet scheme with prescribed velocity.

        Parameters
        ----------
        f : torch.Tensor, shape (9, ny, nx)

        Returns
        -------
        f : torch.Tensor, shape (9, ny, nx), modified in-place at wall
        """
        y = self.wall_y
        nx = f.shape[2]
        device = f.device
        dtype = f.dtype

        # Prescribed velocity at wall
        u_wall = torch.full((nx,), self.U_inf, device=device, dtype=dtype)
        v_wall = torch.full((nx,), -self.V_suction, device=device, dtype=dtype)

        # For bottom wall with Zou-He:
        # Known: f_0, f_1, f_2, f_4, f_8 (c_iy <= 0)
        # Unknown: f_3, f_6, f_7 (c_iy > 0)
        #
        # From rho = Σ f_i and ρv = Σ f_i c_iy:
        #   rho = (f_0 + f_1 + f_2 + f_4 + f_8 + 2*(f_3 + f_6 + f_7)) / (1 - v_wall)
        #   Wait, let me redo this properly.
        #
        # For bottom wall (y=0):
        # Known populations: 0, 1, 2, 4, 8 (c_iy <= 0)
        # Unknown populations: 3, 6, 7 (c_iy > 0)
        #
        # ρ = f_0 + f_1 + f_2 + f_3 + f_4 + f_6 + f_7 + f_8 + f_5
        # Wait, f_5 has c_iy > 0 too. Let me recheck.
        #
        # D2Q9 directions:
        # 0: (0,0), 1: (1,0), 2: (-1,0), 3: (0,1), 4: (0,-1)
        # 5: (1,1), 6: (-1,1), 7: (1,-1), 8: (-1,-1)
        #
        # For bottom wall, unknowns are those with c_iy > 0: 3, 5, 6
        # Known: 0, 1, 2, 4, 7, 8

        if self.wall_y == 0:
            # Unknowns: 3 (0,1), 5 (1,1), 6 (-1,1)
            known = [0, 1, 2, 4, 7, 8]
            unknowns = [3, 5, 6]

            # Sum of known populations
            rho_known = sum(f[k, y, :] for k in known)

            # From mass conservation: ρ = ρ_known + f_3 + f_5 + f_6
            # From momentum y: ρ v = f_3 + f_5 + f_6 - f_4 - f_7 - f_8
            # So: ρ v = (ρ - ρ_known) - (f_4 + f_7 + f_8)
            # => ρ = (ρ_known + f_4 + f_7 + f_8) / (1 - v_wall)
            # Wait: ρ v_wall = f_3 + f_5 + f_6 - f_4 - f_7 - f_8
            # And ρ = ρ_known + f_3 + f_5 + f_6
            # So f_3 + f_5 + f_6 = ρ - ρ_known
            # ρ v_wall = (ρ - ρ_known) - (f_4 + f_7 + f_8)
            # ρ(1 + v_wall) = ρ_known + f_4 + f_7 + f_8  ... wait
            # ρ v_wall = ρ - ρ_known - f_4 - f_7 - f_8
            # ρ - ρ v_wall = ρ_known + f_4 + f_7 + f_8
            # ρ(1 - v_wall) = ρ_known + f_4 + f_7 + f_8
            # ρ = (ρ_known + f_4 + f_7 + f_8) / (1 - v_wall)

            rho_wall = (rho_known + f[4, y, :] + f[7, y, :] + f[8, y, :]) / (1.0 - v_wall)

            # Now we need 3 equations for 3 unknowns (f_3, f_5, f_6).
            # Use: ρ u_wall = f_1 + f_5 + f_7 - f_2 - f_6 - f_8
            # => f_5 - f_6 = ρ u_wall - f_1 - f_7 + f_2 + f_8
            # And non-equilibrium bounce-back: f_5 - f_5^eq = f_7 - f_7^eq
            #   => f_5 = f_7 + f_5^eq - f_7^eq
            # Similarly: f_6 = f_8 + f_6^eq - f_8^eq  ... wait, that's for opposite pairs
            # Actually for Zou-He: f_5 - f_5^eq = -(f_8 - f_8^eq) is wrong.
            # The standard approach: use NEQ bounce-back for the diagonal unknowns
            # f_5^neq = f_7^neq (mirror in y)
            # f_6^neq = f_8^neq

            # Compute equilibrium at wall for known populations
            w = W2Q9.to(device=device, dtype=dtype)
            u2 = u_wall**2 + v_wall**2

            def feq_i(i):
                ci = C2Q9[i].to(device=device, dtype=dtype)
                uc = ci[0] * u_wall + ci[1] * v_wall
                return w[i] * rho_wall * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2)

            # NEQ bounce-back: f_i^neq at unknown = f_opposite^neq at known
            # f_5^neq = f_7^neq (opposite of 5 is 8, but we use the y-mirror)
            # Actually, the standard Zou-He for bottom wall:
            # f_5 = f_7 + (f_5^eq - f_7^eq)  [NEQ bounce-back in x-direction]
            # Wait, let me use the simpler approach:
            # From ρ u_wall: f_5 - f_6 = ρ u_wall - f_1 + f_2 - f_7 + f_8
            # From ρ v_wall: f_3 + f_5 + f_6 = ρ - ρ_known
            # From NEQ BB: f_5 = f_7 + f_5^eq - f_7^eq

            feq5 = feq_i(5)
            feq7 = feq_i(7)
            f[5, y, :] = f[7, y, :] + (feq5 - feq7)

            feq6 = feq_i(6)
            feq8 = feq_i(8)
            f[6, y, :] = f[8, y, :] + (feq6 - feq8)

            # f_3 from mass conservation
            f[3, y, :] = rho_wall - rho_known - f[5, y, :] - f[6, y, :]

        return f


def apply_suction_wall(
    f: torch.Tensor,
    U_inf: float,
    V_suction: float,
    tau: float,
    wall_y: int = 0,
) -> torch.Tensor:
    """Functional interface for suction wall BC.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    U_inf : float
    V_suction : float
    tau : float
    wall_y : int

    Returns
    -------
    f : torch.Tensor, modified at wall
    """
    bc = SuctionWallBC(U_inf, V_suction, tau, wall_y)
    return bc.apply(f)


def apply_inlet_blasius(
    f: torch.Tensor,
    U_inf: float,
    V_suction: float,
    nu: float,
    x_start: float,
    tau: float,
) -> torch.Tensor:
    """Apply Blasius-like inlet profile with suction.

    For ASBL, the inlet should have the Blasius profile (V=0 case)
    or the ASBL exact solution if suction is active everywhere.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    U_inf : float
    V_suction : float
    nu : float, kinematic viscosity
    x_start : float, x-coordinate of inlet
    tau : float

    Returns
    -------
    f : torch.Tensor, modified at inlet (x=0)
    """
    ny, nx = f.shape[1], f.shape[2]
    device = f.device
    dtype = f.dtype

    # y-coordinates (lattice units, wall at y=0)
    y_coords = torch.arange(ny, device=device, dtype=dtype)

    if V_suction > 0:
        # ASBL exact solution
        u_profile = U_inf * (1.0 - torch.exp(-V_suction * y_coords / nu))
    else:
        # Blasius approximation: u/U = erf(0.25 * y * sqrt(U/(nu*x)))
        # For flat plate starting at x_start
        eta = 0.25 * y_coords * torch.sqrt(torch.tensor(U_inf / (nu * max(x_start, 1.0)),
                                                         device=device, dtype=dtype))
        # Approximate Blasius: u/U ≈ tanh(eta) for simplicity
        u_profile = U_inf * torch.tanh(eta)

    v_profile = torch.full((ny,), -V_suction, device=device, dtype=dtype)

    # Set distributions at inlet (x=0) to equilibrium
    x_in = 0
    rho_in = torch.ones(ny, device=device, dtype=dtype)  # ρ=1 for incompressible

    w = W2Q9.to(device=device, dtype=dtype)
    c = C2Q9.to(device=device, dtype=dtype)

    u2 = u_profile**2 + v_profile**2
    for i in range(9):
        uc = c[i, 0] * u_profile + c[i, 1] * v_profile
        f[i, :, x_in] = w[i] * rho_in * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2)

    return f


def apply_outlet_zerogradient(f: torch.Tensor) -> torch.Tensor:
    """Apply zero-gradient (convective) outlet BC at x = nx-1.

    Simple copy from x=nx-2 to x=nx-1.
    """
    f[:, :, -1] = f[:, :, -2]
    return f


def apply_top_free_stream(
    f: torch.Tensor,
    U_inf: float,
    tau: float,
) -> torch.Tensor:
    """Apply free-stream BC at top boundary (y = ny-1).

    Sets velocity to (U_inf, 0) using equilibrium + NEQ extrapolation.
    """
    ny, nx = f.shape[1], f.shape[2]
    device = f.device
    dtype = f.dtype
    y = ny - 1

    # Unknowns at top: directions with c_iy < 0: 4, 7, 8
    u_top = torch.full((nx,), U_inf, device=device, dtype=dtype)
    v_top = torch.zeros(nx, device=device, dtype=dtype)

    known = [0, 1, 2, 3, 5, 6]
    rho_known = sum(f[k, y, :] for k in known)

    # ρ(1 - v_top) = ρ_known + f_7 + f_8 + f_4... wait
    # For top wall, unknowns: 4(0,-1), 7(1,-1), 8(-1,-1)
    # ρ v_top = -(f_4 + f_7 + f_8) + (f_3 + f_5 + f_6)
    # ρ = ρ_known + f_4 + f_7 + f_8
    # ρ v_top = (f_3 + f_5 + f_6) - (f_4 + f_7 + f_8)
    # Since v_top = 0: f_3 + f_5 + f_6 = f_4 + f_7 + f_8
    # ρ = ρ_known + f_4 + f_7 + f_8

    rho_top = rho_known + f[4, y, :] + f[7, y, :] + f[8, y, :]

    w = W2Q9.to(device=device, dtype=dtype)
    u2 = u_top**2 + v_top**2

    def feq_i(i):
        ci = C2Q9[i].to(device=device, dtype=dtype)
        uc = ci[0] * u_top + ci[1] * v_top
        return w[i] * rho_top * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2)

    # NEQ bounce-back
    feq4 = feq_i(4)
    feq3 = feq_i(3)
    f[4, y, :] = f[3, y, :] + (feq4 - feq3)

    feq7 = feq_i(7)
    feq5 = feq_i(5)
    f[7, y, :] = f[5, y, :] + (feq7 - feq5)

    feq8 = feq_i(8)
    feq6 = feq_i(6)
    f[8, y, :] = f[6, y, :] + (feq8 - feq6)

    return f
