"""Suction wall boundary condition for LBM.

Implements a wall with prescribed normal mass flux (suction/blowing)
and tangential velocity. This is the key BC for ASBL (Asymptotic
Suction Boundary Layer) verification.

Physics:
    At the wall (y=0):
        u_tangential = u_wall  (default 0.0 = no-slip; a moving porous wall may
                                pass u_wall > 0)
        v_normal = -V          (suction velocity, V > 0 means suction into wall)

    .. note:: ASBL / suction-boundary-layer verification requires **no-slip**
       (``u_wall = 0``); the exact solution u(y) = U∞ (1 - e^{-Vy/ν}) satisfies
       u(0) = 0 while the free-stream value U∞ is only the y -> ∞ asymptote.
       Prescribing u_wall = U∞ at the wall erases the boundary layer
       (see benchmarks/verified/asbl_suction_bl for the measured evidence).

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


class SuctionWallBCUnverifiedError(NotImplementedError):
    """Raised when the unverified suction wall BC is used without opt-in."""


class SuctionWallBC:
    """Suction wall boundary condition for D2Q9 LBM.

    Parameters
    ----------
    U_inf : float
        Free-stream (tangential) velocity — the y -> infinity asymptote used by
        the inlet / far-field BCs. The wall value is ``u_wall`` (default 0).
    V_suction : float
        Suction velocity (positive = into wall).
    tau : float
        Relaxation time.
    wall_y : int, default 0
        Y-index of the wall row.
    u_wall : float, optional
        Prescribed tangential velocity **at the wall row**. Defaults to
        ``0.0`` (no-slip), which is the physical wall condition for a
        stationary porous wall. Pass ``u_wall=U_inf`` only for a genuinely
        moving (e.g. sliding) porous wall.
    """

    def __init__(
        self,
        U_inf: float,
        V_suction: float,
        tau: float,
        wall_y: int = 0,
        u_wall: float = 0.0,
    ):
        self.U_inf = U_inf
        self.V_suction = V_suction
        self.tau = tau
        self.wall_y = wall_y
        self.u_wall = u_wall

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

    def apply(self, f: torch.Tensor, allow_unverified: bool = False) -> torch.Tensor:
        """Apply suction wall BC to distribution functions.

        Uses Zou-He Dirichlet scheme with prescribed velocity.

        Raises
        ------
        SuctionWallBCUnverifiedError
            Always, unless ``allow_unverified=True``. The scheme is WITHHELD:
            see the module docstring and ``docs/AUDIT_m16154_SUCTION_ASBL.md``.

        Parameters
        ----------
        f : torch.Tensor, shape (9, ny, nx)

        Returns
        -------
        f : torch.Tensor, shape (9, ny, nx), modified in-place at wall
        """
        if not allow_unverified:
            raise SuctionWallBCUnverifiedError(
                "SuctionWallBC is WITHHELD (audit m16154, 2026-11-12): the "
                "repaired Zou-He scheme still fails a steady-Couette sanity "
                "check (interior shear slope 2x analytic, wall-adjacent row "
                "-0.37 U), so it cannot back a validation claim. Use "
                "apply(..., allow_unverified=True) only for diagnostics; see "
                "docs/AUDIT_m16154_SUCTION_ASBL.md."
            )
        y = self.wall_y
        nx = f.shape[2]
        device = f.device
        dtype = f.dtype

        # Prescribed velocity at wall (default: no-slip tangential + suction normal)
        u_wall = torch.full((nx,), self.u_wall, device=device, dtype=dtype)
        v_wall = torch.full((nx,), -self.V_suction, device=device, dtype=dtype)

        # Zou & He (1997) velocity-Dirichlet reconstruction extended to a
        # prescribed wall-normal velocity (suction / blowing).
        #
        # Unknown populations -- bottom wall: 3, 5, 6 (c_iy > 0)
        #                        top wall   : 4, 7, 8 (c_iy < 0)
        # Three conditions close the 3-unknown system:
        #   (i)   mass + wall-normal momentum  -> rho_wall
        #   (ii)  non-equilibrium bounce-back for ONE diagonal pair
        #   (iii) streamwise (x) momentum      -> the second diagonal
        # Using NEQ bounce-back for BOTH diagonals (the pre-2026-11-12 code)
        # silently drops condition (iii): the prescribed tangential wall
        # velocity then has no effect at all (measured: pure-Couette case
        # returned u == 0 exactly).  See tests/test_suction_wall_bc.py.
        w = W2Q9.to(device=device, dtype=dtype)
        u2 = u_wall**2 + v_wall**2

        def feq(i: int, rho_wall: torch.Tensor) -> torch.Tensor:
            ci = C2Q9[i].to(device=device, dtype=dtype)
            uc = ci[0] * u_wall + ci[1] * v_wall
            return w[i] * rho_wall * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2)

        if y == 0:
            known = [0, 1, 2, 4, 7, 8]
            rho_known = sum(f[k, y, :] for k in known)
            # (i) mass + y-momentum:  rho (1 - v_wall) = rho_known + (f4+f7+f8)
            rho_wall = (rho_known + f[4, y, :] + f[7, y, :] + f[8, y, :]) / (
                1.0 - v_wall
            )
            # (ii) NEQ bounce-back across the wall: f6 - feq6 = f8 - feq8
            f[6, y, :] = f[8, y, :] + (feq(6, rho_wall) - feq(8, rho_wall))
            # (iii) rho u = (f1+f5+f7) - (f2+f6+f8)  -> f5
            f[5, y, :] = (
                rho_wall * u_wall - f[1, y, :] - f[7, y, :] + f[2, y, :]
                + f[6, y, :] + f[8, y, :]
            )
            # (i) mass closure -> f3
            f[3, y, :] = rho_wall - rho_known - f[5, y, :] - f[6, y, :]

        elif y == f.shape[1] - 1:
            known = [0, 1, 2, 3, 5, 6]
            rho_known = sum(f[k, y, :] for k in known)
            # (i) mass + y-momentum:  rho (1 + v_wall) = rho_known + (f3+f5+f6)
            rho_wall = (rho_known + f[3, y, :] + f[5, y, :] + f[6, y, :]) / (
                1.0 + v_wall
            )
            # (ii) NEQ bounce-back across the wall: f7 - feq7 = f5 - feq5
            f[7, y, :] = f[5, y, :] + (feq(7, rho_wall) - feq(5, rho_wall))
            # (iii) rho u = (f1+f5+f8) - (f2+f6+f7)  -> f8
            f[8, y, :] = f[7, y, :] + (
                f[1, y, :] + f[5, y, :] - f[2, y, :] - f[6, y, :]
                - rho_wall * u_wall
            )
            # (i) mass closure -> f4
            f[4, y, :] = rho_wall - rho_known - f[7, y, :] - f[8, y, :]

        else:
            raise ValueError(
                f"SuctionWallBC supports only the bottom (wall_y=0) and top "
                f"(wall_y=ny-1={f.shape[1] - 1}) wall rows, got wall_y={y}"
            )

        return f


def apply_suction_wall(
    f: torch.Tensor,
    U_inf: float,
    V_suction: float,
    tau: float,
    wall_y: int = 0,
    u_wall: float = 0.0,
    allow_unverified: bool = False,
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
    bc = SuctionWallBC(U_inf, V_suction, tau, wall_y, u_wall=u_wall)
    return bc.apply(f, allow_unverified=allow_unverified)


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
