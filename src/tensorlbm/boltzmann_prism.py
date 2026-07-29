"""Boltzmann-FDM solver on prism boundary layers.

Same 19-distribution variable f as LBM, but discretized via finite
difference stencils on the body-fitted prism mesh instead of collide-stream.

Interface: f values are exchanged directly — no velocity/temperature coupling needed.

For each population i:
  ∂_t f_i = -c_i·∇f_i - (f_i - f_i^eq)/τ
"""
import torch
import torch.nn.functional as F
import math, time
from dataclasses import dataclass


@dataclass
class BoltzmannPrismConfig:
    n_layers: int = 5
    first_height: float = 0.02
    growth: float = 1.2
    dt_sub: float = 0.1  # sub-step for explicit FDM


class BoltzmannPrismSolver:
    """FDM discrete Boltzmann solver on body-fitted prism mesh.

    Each prism cell stores the full 19-population distribution f,
    same as LBM. The advection term uses FD stencils with the
    wall-normal spacing from the prism layer heights.

    The coupling to LBM is: at each time step, copy f from LBM
    at the outermost prism layer, and copy f from prism back to
    LBM at the wall-adjacent Cartesian cells.
    """

    def __init__(self, solid: torch.Tensor, n_layers: int = 5,
                 nu: float = 2.4e-6, device: str = 'sdaa:0',
                 first_height: float = 0.02, growth: float = 1.2):
        self.device = torch.device(device)
        self.nu = nu
        self.tau = 3.0 * nu + 0.5

        # Build prism mesh
        from .prism_layer import generate_prism_layers
        cpu_solid = solid.cpu() if solid.device != torch.device('cpu') else solid
        prism = generate_prism_layers(cpu_solid, n_layers=n_layers,
                                       first_height=first_height, growth=growth)
        self.n_layers = prism.n_layers
        self.n_surface = prism.n_surface
        self.layer_heights = prism.layer_heights.to(self.device)
        self.surface_centers = prism.surface_centers.to(self.device)
        self.surface_normals = prism.surface_normals.to(self.device)

        # Initialize distribution (19 populations per prism cell)
        self.f_prism = torch.zeros(19, n_layers, prism.n_surface,
                                   device=self.device, dtype=torch.float32)

        # FD stencils for wall-normal direction
        self._h = self.layer_heights

        # Velocity set (D3Q19)
        from .d3q19 import C as _C
        self.c = _C.to(self.device).float()  # (19, 3)

        # Precompute wall-normal stencil weights for each layer
        # Central difference: df/dn ≈ (f_{k+1} - f_{k-1}) / (h_k + h_{k-1})
        # Laplacian: d²f/dn² ≈ (f_{k+1} - 2f_k + f_{k-1}) / h_k²
        h = self._h
        self.grad_w = torch.zeros(n_layers, 3, device=self.device)
        self.lap_w = torch.zeros(n_layers, 3, device=self.device)
        for k in range(n_layers):
            hp = h[k+1] if k+1 < n_layers else h[k]
            hm = h[k-1] if k > 0 else h[0]
            self.grad_w[k, 1] = 1.0 / (hp + hm)   # +1 direction
            self.grad_w[k, 2] = -1.0 / (hp + hm)  # -1 direction
            self.lap_w[k, 1] = 1.0 / (h[k] * h[k])   # +1
            self.lap_w[k, 0] = -2.0 / (h[k] * h[k])  # center
            self.lap_w[k, 2] = 1.0 / (h[k] * h[k])   # -1

        print(f"BoltzmannPrism: {n_layers} layers × {prism.n_surface} surfaces")

    def init_from_lbm(self, f_lbm: torch.Tensor, solid: torch.Tensor):
        """Initialize prism f from LBM near-wall distribution."""
        # For each surface cell, copy LBM distribution at wall-adjacent cell
        sc = self.surface_centers
        for i in range(self.n_surface):
            sx, sy = int(sc[i, 0]), int(sc[i, 1])
            if 0 <= sx < f_lbm.shape[3] and 0 <= sy < f_lbm.shape[2]:
                # Copy LBM f to all prism layers
                for k in range(self.n_layers):
                    self.f_prism[:, k, i] = f_lbm[:, 0, sy, sx].clone()

    def step(self, f_lbm_interface: torch.Tensor, dt: float = 1.0):
        """Advance prism distribution one LBM time step.

        Args:
            f_lbm_interface: (19, n_surface) — LBM f at wall-adjacent cells
            dt: Time step (typically 1.0 for LBM clock).
        """
        n_sub = max(1, int(dt / 0.05))  # ensure CFL
        dt_sub = dt / n_sub

        f = self.f_prism
        c = self.c  # (19, 3)
        h = self._h

        for _ in range(n_sub):
            # ---- Advection (wall-normal) ----
            f_shift_up = torch.roll(f, -1, dims=1)
            f_shift_dn = torch.roll(f, 1, dims=1)

            # Upwind flux: for direction i with c_x > 0, use f[k-1]; c_x < 0, use f[k+1]
            adv = torch.zeros_like(f)
            for i in range(19):
                cx = c[i, 0].item()
                if cx > 0.01:
                    adv[i] = cx * (f[i] - f_shift_dn[i])
                elif cx < -0.01:
                    adv[i] = cx * (f_shift_up[i] - f[i])

            f = f - dt_sub * adv

            # Bounce-back at wall (layer 0)
            for i in range(9):
                tmp = f[i, 0].clone()
                f[i, 0] = f[i + 9, 0]
                f[i + 9, 0] = tmp

            # Dirichlet at interface (layer -1) from LBM
            f[:, -1] = f_lbm_interface

        self.f_prism = f
        return f

    def get_interface_f(self) -> torch.Tensor:
        """Return (19, n_surface) distribution at prism-LBM interface."""
        return self.f_prism[:, -1, :]

    def drag_from_wall_shear(self) -> float:
        """Compute drag from wall shear at layer 0."""
        # τ_w = ν * du/dn|_wall ≈ ν * u_1 / h_0
        # u = c·f, drag = Σ τ_w * direction_x
        return 0.0  # placeholder


# ── Integration with LBM main loop ──
class LBMPrismHybrid:
    """Combined LBM bulk + Boltzmann-Prism boundary layer."""

    def __init__(self, solid: torch.Tensor, n_prism: int = 5,
                 nu: float = 2.4e-6, device: str = 'sdaa:0'):
        self.device = torch.device(device)
        self.nu = nu
        self.prism = BoltzmannPrismSolver(solid, n_prism, nu, device)
        self.solid = solid.to(self.device)

    def init(self) -> torch.Tensor:
        """Initialize LBM distribution."""
        from .d3q19 import equilibrium3d
        nx, ny, nz = self.solid.shape[2], self.solid.shape[1], self.solid.shape[0]
        r0 = torch.ones(nz, ny, nx, device=self.device)
        u0 = torch.full((nz, ny, nx), 0.06, device=self.device)
        u0[self.solid] = 0.0
        f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
        self.prism.init_from_lbm(f, self.solid)
        return f

    def step(self, f: torch.Tensor, tau: float, C_s: float = 0.05) -> torch.Tensor:
        """One LBM step + prism coupling."""
        from .turbulence import collide_smagorinsky_mrt3d
        from .solver3d import stream3d
        from .wall_model import wall_function_3d
        from .boundaries3d import far_field_bc_3d

        # LBM bulk
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=C_s)
        f = stream3d(f)

        # Extract interface f for prism
        sc = self.prism.surface_centers
        f_iface = torch.zeros(19, self.prism.n_surface, device=self.device)
        for i in range(self.prism.n_surface):
            sx, sy = int(sc[i, 0]), int(sc[i, 1])
            if 0 <= sx < f.shape[3] and 0 <= sy < f.shape[2]:
                f_iface[:, i] = f[:, 0, sy, sx].clone()

        # Prism step
        f_iface_new = self.prism.step(f_iface)

        # Inject prism f back to LBM
        for i in range(self.prism.n_surface):
            sx, sy = int(sc[i, 0]), int(sc[i, 1])
            if 0 <= sx < f.shape[3] and 0 <= sy < f.shape[2]:
                f[:, 0, sy, sx] = f_iface_new[:, i]

        # Wall function for remaining near-wall cells (outside prism)
        f, df, dp = wall_function_3d(f, self.solid, self.nu)
        f = far_field_bc_3d(f, u_in=0.06)
        return f
