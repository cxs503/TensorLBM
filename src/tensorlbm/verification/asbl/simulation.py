"""ASBL simulation driver for TensorLBM -- **WITHDRAWN (audit m16154)**.

Sets up and runs the Asymptotic Suction Boundary Layer simulation.

.. warning:: This driver is **not a verification capability**. Audit m16154
   (2026-11-12) measured it diverging to ``NaN`` from its own defaults
   (nx=ny=48, U=0.1, V=0.002, nu=0.02: ``rho_min`` 1.00 -> 0.25 by step 1000,
   all-NaN by step ~1100) because the underlying
   :class:`~tensorlbm.boundary.suction.SuctionWallBC` was neither no-slip nor
   conservative (see ``docs/AUDIT_m16154_SUCTION_ASBL.md``). The wall BC is now
   gated (WITHHELD); this driver therefore opts in explicitly with
   ``allow_unverified=True`` so that the failure stays reproducible for
   diagnostics instead of being silently forgotten.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from ...boundary.solver2d import (
    equilibrium2d,
    macroscopic2d,
    stream2d,
    collide_bgk2d,
    compute_wall_shear_stress,
)
from ...boundary.suction import (
    SuctionWallBC,
    apply_inlet_blasius,
    apply_outlet_zerogradient,
    apply_top_free_stream,
)
from .analytical import asbl_exact_velocity, asbl_skin_friction


class ASBLSimulation:
    """ASBL simulation driver.

    Domain: 2D flat plate with suction at the bottom wall.
    - Bottom (y=0): suction wall with velocity (U∞, -V)
    - Top (y=ny-1): free-stream (U∞, 0)
    - Left (x=0): inlet with Blasius/ASBL profile
    - Right (x=nx-1): zero-gradient outlet

    Parameters
    ----------
    nx : int, number of grid points in x
    ny : int, number of grid points in y
    U_inf : float, free-stream velocity (lattice units)
    V_suction : float, suction velocity (lattice units)
    nu : float, kinematic viscosity (lattice units)
    max_steps : int, maximum simulation steps
    conv_tol : float, convergence tolerance for steady state
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        U_inf: float,
        V_suction: float,
        nu: float,
        max_steps: int = 50000,
        conv_tol: float = 1e-6,
        u_wall: float = 0.0,
    ):
        self.nx = nx
        self.ny = ny
        self.U_inf = U_inf
        self.V_suction = V_suction
        self.nu = nu
        self.max_steps = max_steps
        self.conv_tol = conv_tol

        # Relaxation time
        self.tau = 0.5 + nu / (1.0 / 3.0)  # τ = 0.5 + ν/cs²

        # Initialize fields
        self.device = torch.device('cpu')
        self.dtype = torch.float64
        self._initialize_fields()

        # Boundary conditions
        # No-slip at the wall is required by the ASBL exact solution
        # u(y) = U_inf (1 - exp(-V y / nu)), whose value at y = 0 is 0.
        # ``u_wall=U_inf`` (the pre-2026-11-12 behaviour) prescribes the free
        # stream *at the wall* and destroys the boundary layer; it is kept
        # reachable only as a diagnostic negative control.
        self.wall_bc = SuctionWallBC(
            U_inf, V_suction, self.tau, wall_y=0, u_wall=u_wall
        )

        # History
        self.history = []

    def _initialize_fields(self):
        """Initialize distribution functions to equilibrium at free-stream."""
        nx, ny = self.nx, self.ny
        device, dtype = self.device, self.dtype

        # Start with uniform free-stream
        rho = torch.ones(ny, nx, device=device, dtype=dtype)
        ux = torch.full((ny, nx), self.U_inf, device=device, dtype=dtype)
        uy = torch.zeros(ny, nx, device=device, dtype=dtype)

        # Set wall region with Blasius-like profile
        y_coords = torch.arange(ny, device=device, dtype=dtype)
        if self.V_suction > 0:
            u_profile = self.U_inf * (1.0 - torch.exp(-self.V_suction * y_coords / self.nu))
        else:
            # Approximate Blasius at mid-domain
            x_mid = self.nx / 2.0
            eta = y_coords * torch.sqrt(torch.tensor(self.U_inf / (self.nu * x_mid),
                                                      device=device, dtype=dtype))
            u_profile = self.U_inf * torch.tanh(0.332 * eta)

        for j in range(ny):
            ux[j, :] = u_profile[j].item()

        self.f = equilibrium2d(rho, ux, uy)

    def run(self, verbose: bool = False) -> Dict:
        """Run the simulation to steady state.

        Returns
        -------
        result : dict with keys:
            'converged': bool
            'steps': int
            'residual': float
            'u_profile': torch.Tensor, shape (ny,) at mid-domain
            'cf_measured': float
            'cf_analytical': float
        """
        prev_rho = None
        converged = False
        residual = float('inf')

        for step in range(self.max_steps):
            # Collision
            self.f = collide_bgk2d(self.f, self.tau)

            # Streaming
            self.f = stream2d(self.f)

            # Boundary conditions
            self.f = self.wall_bc.apply(self.f, allow_unverified=True)
            self.f = apply_top_free_stream(self.f, self.U_inf, self.tau)
            self.f = apply_inlet_blasius(
                self.f, self.U_inf, self.V_suction, self.nu, 0.0, self.tau
            )
            self.f = apply_outlet_zerogradient(self.f)

            # Check convergence every 100 steps
            if step % 100 == 0:
                rho, ux, uy = macroscopic2d(self.f)
                if prev_rho is not None:
                    residual = (rho - prev_rho).abs().max().item()
                    if verbose and step % 1000 == 0:
                        print(f"Step {step}: residual = {residual:.2e}")
                    if residual < self.conv_tol:
                        converged = True
                        if verbose:
                            print(f"Converged at step {step}, residual = {residual:.2e}")
                        break
                prev_rho = rho.clone()

        # Extract results
        rho, ux, uy = macroscopic2d(self.f)

        # Get velocity profile at mid-domain
        x_mid = self.nx // 2
        u_profile = ux[:, x_mid]

        # Measure skin friction
        tau_w = compute_wall_shear_stress(self.f, self.tau, wall_y=0)
        cf_measured = 2.0 * tau_w.mean().item() / (self.U_inf**2)

        # Analytical C_f
        if self.V_suction > 0:
            cf_analytical = asbl_skin_friction(self.U_inf, self.V_suction)
        else:
            # Blasius at mid-domain
            Re_x = self.U_inf * x_mid / self.nu
            cf_analytical = 0.664 / np.sqrt(Re_x)

        return {
            'converged': converged,
            'steps': step + 1,
            'residual': residual,
            'u_profile': u_profile.cpu(),
            'y_coords': torch.arange(self.ny, dtype=self.dtype),
            'cf_measured': cf_measured,
            'cf_analytical': cf_analytical,
            'tau_w_mean': tau_w.mean().item(),
            'f_final': self.f.cpu(),
        }


def run_asbl_case(
    nx: int,
    ny: int,
    U_inf: float,
    V_suction: float,
    nu: float,
    max_steps: int = 50000,
) -> Dict:
    """Convenience function to run a single ASBL case.

    Parameters
    ----------
    nx, ny : int, grid dimensions
    U_inf : float, free-stream velocity
    V_suction : float, suction velocity
    nu : float, kinematic viscosity
    max_steps : int

    Returns
    -------
    result : dict
    """
    sim = ASBLSimulation(nx, ny, U_inf, V_suction, nu, max_steps)
    return sim.run()
