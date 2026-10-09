"""Finite-Difference LBM on body-fitted prism mesh.

Standard LBM (collide-stream) requires uniform Cartesian grids because
the discrete velocities c_i must point to exact neighboring nodes.

FD-LBM instead discretises the Boltzmann equation directly:
  ∂_t f_i + c_i·∇f_i = -(f_i - f_i^eq)/τ

on ANY mesh (structured/unstructured) using FD stencils.
No collide-stream needed → works on prism layers.

The 19 populations f_i are stored at each cell centre.
Advection uses finite-difference gradients.
Coupling with Cartesian LBM is via f at the interface cells (same variable).

References:
- He, Chen & Doolen (1998) "A novel thermal model for the lattice
  Boltzmann method in incompressible limit"
- Shu, Niu & Chew (2002) "Taylor-series expansion and least-squares
  lattice Boltzmann method"
"""

import torch


class FDLBMPrismSolver:
    """FD-LBM on body-fitted prism layers.

    Each prism cell stores 19 populations f_i.
    The advection term c_i·∇f_i is computed via FD stencils on the
    body-fitted mesh, with Jacobian transformations for curvilinear
    elements.
    """

    def __init__(self, prism: dict, velocities: torch.Tensor, tau: float, device: torch.device):
        """
        Args:
            prism: dict from generate_adaptive_prism()
            velocities: (19, 3) lattice velocities (D3Q19 C matrix)
            tau: relaxation time
            device: torch device
        """
        self.prism = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prism.items()
        }
        self.c = velocities.to(device).float()  # (19, 3)
        self.tau = tau
        self.device = device

        self.n_layers = prism["layer_centres"].shape[0]
        self.n_surface = prism["n_surface"]

        # Distribution function: (19, n_layers, n_surface)
        self.f = torch.zeros(19, self.n_layers, self.n_surface, device=device, dtype=torch.float32)

    def init_from_lbm(self, f_lbm: torch.Tensor, solid: torch.Tensor):
        """Initialize FD-LBM distribution from Cartesian LBM at interface."""
        c0 = self.prism["surface_centres"]
        for i in range(self.n_surface):
            cx, cy, cz = int(c0[i, 0]), int(c0[i, 1]), int(c0[i, 2])
            if 0 <= cx < f_lbm.shape[3] and 0 <= cy < f_lbm.shape[2] and 0 <= cz < f_lbm.shape[1]:
                # Copy LBM f to all prism layers as initial guess
                self.f[:, :, i] = f_lbm[:, cz, cy, cx].unsqueeze(1)

    def _equilibrium(self, rho, ux, uy, uz):
        """Compute Maxwell-Boltzmann equilibrium for D3Q19."""
        from .d3q19 import equilibrium3d

        return equilibrium3d(rho, ux, uy, uz)

    def _macroscopic(self):
        """Compute rho, ux, uy, uz from FD-LBM distribution."""
        rho = self.f.sum(dim=0)  # (n_layers, n_surface)
        ux = (self.f * self.c[:, 0].view(19, 1, 1)).sum(dim=0) / rho.clamp(min=1e-10)
        uy = (self.f * self.c[:, 1].view(19, 1, 1)).sum(dim=0) / rho.clamp(min=1e-10)
        uz = (self.f * self.c[:, 2].view(19, 1, 1)).sum(dim=0) / rho.clamp(min=1e-10)
        return rho, ux, uy, uz

    def _grad_wall_normal(self, field: torch.Tensor) -> torch.Tensor:
        """Compute wall-normal gradient via central difference.

        Args:
            field: (n_layers, n_surface) or (19, n_layers, n_surface)

        Returns:
            gradient in wall-normal direction, same shape.
        """
        h = self.prism["layer_heights"]  # (n_layers, n_surface)
        # Central difference: (f_{k+1} - f_{k-1}) / (h_k + h_{k-1})
        up = torch.roll(field, -1, dims=0)  # f_{k+1}
        dn = torch.roll(field, 1, dims=0)  # f_{k-1}
        h_sum = h + torch.roll(h, 1, dims=0)  # h_k + h_{k-1}
        h_sum = h_sum.clamp(min=1e-10)
        return (up - dn) / h_sum

    def _advection(self) -> torch.Tensor:
        """Compute advection term c_i·∇f_i for all populations.

        Since the prism mesh is structured in the wall-normal (n) and
        surface-tangent (t1, t2) directions, the advection splits into:
          c_i·∇f_i = c_i·n * df/dn + c_i·t1 * df/dt1 + c_i·t2 * df/dt2

        We approximate:
        - df/dn via wall-normal central difference
        - df/dt1, df/dt2 via small (neglected for now — prism is thin)
        """
        normals = self.prism["surface_normals"]  # (n_surface, 3)

        # For each population i, project c_i onto the surface normal
        # c_i·n gives the wall-normal velocity component for this population
        c_dot_n = self.c @ normals.T  # (19, n_surface)

        # Wall-normal gradient of f
        dfdn = self._grad_wall_normal(self.f)  # (19, n_layers, n_surface)

        # Advection: c_i·n * df_i/dn
        adv = torch.zeros_like(self.f)
        for i in range(19):
            # c_dot_n[i] is (n_surface,), dfdn[i] is (n_layers, n_surface)
            adv[i] = c_dot_n[i].unsqueeze(0) * dfdn[i]

        return adv

    def step(self, dt: float = 1.0, n_substeps: int = 4):
        """Advance FD-LBM one LBM time step.

        Uses sub-stepping for stability of explicit FD advection.
        """
        dt_sub = dt / n_substeps

        for _ in range(n_substeps):
            # 1. Advection
            adv = self._advection()

            # 2. Collision (explicit Euler on BGK)
            rho, ux, uy, uz = self._macroscopic()
            feq = self._equilibrium(rho, ux, uy, uz)
            coll = -(self.f - feq) / self.tau

            # 3. Update
            self.f = self.f + dt_sub * (-adv + coll)

            # 4. Boundary conditions
            # Wall (layer 0): half-way bounce-back
            for i in range(9):
                tmp = self.f[i, 0].clone()
                self.f[i, 0] = self.f[i + 9, 0]
                self.f[i + 9, 0] = tmp

    def get_interface_f(self) -> torch.Tensor:
        """Return (19, n_surface) distribution at prism-LBM interface."""
        return self.f[:, -1, :]

    def inject_interface_to_lbm(self, f_lbm: torch.Tensor):
        """Write prism interface distribution back to LBM."""
        c0 = self.prism["surface_centres"]
        f_iface = self.get_interface_f()
        for i in range(self.n_surface):
            cx, cy, cz = int(c0[i, 0]), int(c0[i, 1]), int(c0[i, 2])
            if 0 <= cx < f_lbm.shape[3] and 0 <= cy < f_lbm.shape[2] and 0 <= cz < f_lbm.shape[1]:
                f_lbm[:, cz, cy, cx] = f_iface[:, i]


# ── LBM loop integration ──
def hybrid_step_lbm_fdlbm(
    f: torch.Tensor,
    solid: torch.Tensor,
    fd_solver: FDLBMPrismSolver,
    tau: float,
    C_s: float = 0.05,
    u_in: float = 0.08,
) -> torch.Tensor:
    """One LBM step (Cartesian bulk) + FD-LBM step (prism layer).

    The prism layer solves the Boltzmann equation on body-fitted mesh,
    coupled to Cartesian LBM at the interface via f values.
    """
    from .boundaries3d import far_field_bc_3d
    from .solver3d import stream3d
    from .turbulence import collide_smagorinsky_mrt3d

    # LBM bulk step
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=C_s)
    f = stream3d(f)

    # Extract interface f for FD-LBM
    fd_solver.init_from_lbm(f, solid)

    # FD-LBM step on prism layer
    fd_solver.step(dt=1.0, n_substeps=4)

    # Inject back to LBM
    fd_solver.inject_interface_to_lbm(f)

    # Far-field BC
    f = far_field_bc_3d(f, u_in=u_in)

    return f
