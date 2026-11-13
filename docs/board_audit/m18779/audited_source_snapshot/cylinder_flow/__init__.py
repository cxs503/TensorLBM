"""Cylinder flow verification for TensorLBM using immersed boundary markers.

Validates the immersed boundary method (IBM) for static obstacles against
the canonical benchmark of flow past a circular cylinder.

Physics
-------
Two-dimensional incompressible flow past a circular cylinder at Reynolds
numbers Re = U*D/nu = 20 (steady) and Re = 100 (periodic vortex shedding).

The cylinder is represented by Lagrangian markers on its surface, and the
no-slip condition is enforced via Peskin's immersed boundary method with
direct forcing.

Reference data
--------------
Re = 20 (steady):
  - Cd = 2.05  (Dennis & Chang 1970, J. Fluid Mech. 42:471)
  - Lr = 0.93  (recirculation length / D)

Re = 40 (steady):
  - Cd = 1.54  (Dennis & Chang 1970)

Re = 100 (periodic vortex shedding):
  - Cd_mean = 1.35, Cl_amp = 0.33  (Liu et al. 1998, J. Comput. Phys. 143:35)
  - St = 0.165

References
----------
- Dennis, S.C.R. & Chang, G.-Z. (1970). "Numerical solutions for steady flow
  past a circular cylinder at Reynolds numbers up to 100." J. Fluid Mech. 42:471-489.
- Liu, C., Zheng, X. & Sung, C.H. (1998). "Preconditioned multigrid methods for
  unsteady incompressible flows." J. Comput. Phys. 143:35-58.
- Peskin, C.S. (2002). "The immersed boundary method." Acta Numerica 11:479-517.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple, Optional, Dict, List

import torch
import numpy as np

from tensorlbm.d2q9 import C, W, OPPOSITE, CS2, equilibrium, macroscopic, stream, collide_bgk
from tensorlbm.ibm import (
    build_stencil_2d,
    spread_force_2d,
    interpolate_velocity_2d,
    direct_forcing_static,
    build_cylinder_markers_2d,
)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

CYLINDER_CD_2D = {
    20: {"Cd": 2.05, "Lr": 0.93, "source": "Dennis & Chang 1970"},
    40: {"Cd": 1.54, "Lr": 2.25, "source": "Dennis & Chang 1970"},
    100: {"Cd_mean": 1.35, "Cl_amp": 0.33, "St": 0.165, "source": "Liu et al. 1998"},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CylinderFlowConfig:
    """Configuration for cylinder flow simulation."""
    nx: int = 320
    ny: int = 160
    cylinder_x: float = 80.0
    cylinder_y: float = 80.0
    cylinder_radius: float = 20.0
    u_inlet: float = 0.1
    re: float = 20.0
    marker_spacing: float = 0.5
    n_iter_ibm: int = 10
    max_steps: int = 10000
    sample_interval: int = 100
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self):
        if self.re <= 0:
            raise ValueError(f"Re must be positive, got {self.re}")
        if self.u_inlet <= 0 or self.u_inlet >= 0.3:
            raise ValueError(f"u_inlet={self.u_inlet} outside stable range")

    @property
    def diameter(self) -> float:
        return 2.0 * self.cylinder_radius

    @property
    def nu(self) -> float:
        return self.u_inlet * self.diameter / self.re

    @property
    def tau(self) -> float:
        return self.nu / CS2 + 0.5


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class CylinderFlowSolver:
    """D2Q9 LBM solver for flow past a cylinder using IBM markers."""

    def __init__(self, config: CylinderFlowConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Generate cylinder markers
        self.marker_x, self.marker_y = build_cylinder_markers_2d(
            config.cylinder_x, config.cylinder_y, config.cylinder_radius,
            marker_spacing=config.marker_spacing, device=self.device
        )
        self.n_markers = self.marker_x.shape[0]

        # Build stencil
        self.weights, self.indices = build_stencil_2d(
            self.marker_x, self.marker_y, self.ny, self.nx
        )

        # Solid mask for visualization
        self._build_solid_mask()
        self.fluid_mask = ~self.solid_mask

        # Initialize at equilibrium with uniform flow
        rho = torch.ones((self.ny, self.nx), dtype=config.dtype, device=self.device)
        ux = torch.full((self.ny, self.nx), config.u_inlet, dtype=config.dtype, device=self.device)
        uy = torch.zeros((self.ny, self.nx), dtype=config.dtype, device=self.device)
        self.f = equilibrium(rho, ux, uy)

        # Target velocity at markers (zero for static cylinder)
        self.target_ux = torch.zeros(self.n_markers, dtype=config.dtype, device=self.device)
        self.target_uy = torch.zeros(self.n_markers, dtype=config.dtype, device=self.device)

        # Cached IBM force (computed during step, reused for force reporting)
        self._cached_force_x = None
        self._cached_force_y = None

        # Force history
        self.force_history: List[Dict] = []

        # Streaming shifts
        self._shifts = []
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            self._shifts.append((-cy, -cx))

    def _build_solid_mask(self):
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, device=self.device, dtype=torch.float32),
            torch.arange(self.nx, device=self.device, dtype=torch.float32),
            indexing='ij',
        )
        dx = xx - self.cfg.cylinder_x
        dy = yy - self.cfg.cylinder_y
        r = torch.sqrt(dx**2 + dy**2)
        self.solid_mask = r <= self.cfg.cylinder_radius

    def _stream(self, f: torch.Tensor) -> torch.Tensor:
        f_out = torch.empty_like(f)
        for i in range(9):
            sy, sx = self._shifts[i]
            if sy == 0 and sx == 0:
                f_out[i] = f[i]
            else:
                f_out[i] = torch.roll(f[i], shifts=(sy, sx), dims=(0, 1))
        return f_out

    def _apply_inlet_bc(self, f: torch.Tensor, u_inlet: float):
        """Zou/He velocity BC at inlet (x=0)."""
        f0 = f[0, :, 0]
        f2 = f[2, :, 0]
        f3 = f[3, :, 0]
        f4 = f[4, :, 0]
        f6 = f[6, :, 0]
        f7 = f[7, :, 0]

        rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_inlet)

        f[1, :, 0] = f[3, :, 0] + (2.0 / 3.0) * rho_in * u_inlet
        f[5, :, 0] = f[7, :, 0] - 0.5 * (f[2, :, 0] - f[4, :, 0]) + (1.0 / 6.0) * rho_in * u_inlet
        f[8, :, 0] = f[6, :, 0] + 0.5 * (f[2, :, 0] - f[4, :, 0]) + (1.0 / 6.0) * rho_in * u_inlet

    def _apply_outlet_bc(self, f: torch.Tensor):
        f[:, :, -1] = f[:, :, -2].clone()

    def _apply_symmetry_bc(self, f: torch.Tensor):
        f[4, 0, :] = f[3, 1, :]
        f[7, 0, :] = f[6, 1, :]
        f[8, 0, :] = f[5, 1, :]
        f[3, -1, :] = f[4, -2, :]
        f[6, -1, :] = f[7, -2, :]
        f[5, -1, :] = f[8, -2, :]

    def _apply_ibm_force(self, f: torch.Tensor) -> torch.Tensor:
        """Apply IBM body force using Guo forcing scheme. Caches force for later use."""
        rho, ux, uy = macroscopic(f)

        u_marker_x, u_marker_y = interpolate_velocity_2d(
            ux, uy, self.weights, self.indices
        )

        force_x, force_y = direct_forcing_static(
            u_marker_x, u_marker_y,
            self.target_ux, self.target_uy,
            rho, self.weights, self.indices,
            self.ny, self.nx,
            n_iter=self.cfg.n_iter_ibm
        )

        # Cache the force
        self._cached_force_x = force_x
        self._cached_force_y = force_y

        # Apply Guo forcing
        cs2 = CS2
        omega = 1.0 / self.cfg.tau

        c = C.to(device=self.device, dtype=f.dtype)
        w = W.to(device=self.device, dtype=f.dtype)

        force_term = w.view(-1, 1, 1) * (1.0 - 0.5 * omega) / cs2 * (
            (c[:, 0].view(-1, 1, 1) - ux.unsqueeze(0)) * force_x.unsqueeze(0) +
            (c[:, 1].view(-1, 1, 1) - uy.unsqueeze(0)) * force_y.unsqueeze(0)
        )

        return f + force_term

    def _compute_forces(self) -> Tuple[float, float]:
        """Return cached drag and lift forces (force on cylinder = -force on fluid)."""
        if self._cached_force_x is None:
            return 0.0, 0.0
        Fx = -self._cached_force_x.sum().item()
        Fy = -self._cached_force_y.sum().item()
        return Fx, Fy

    def _compute_drag_coefficient(self, Fx: float) -> float:
        rho = 1.0
        U = self.cfg.u_inlet
        D = self.cfg.diameter
        return Fx / (0.5 * rho * U**2 * D)

    def _compute_lift_coefficient(self, Fy: float) -> float:
        rho = 1.0
        U = self.cfg.u_inlet
        D = self.cfg.diameter
        return Fy / (0.5 * rho * U**2 * D)

    def step(self) -> Tuple[float, float]:
        """One LBM step."""
        self.f = collide_bgk(self.f, self.cfg.tau)
        self.f = self._stream(self.f)
        self._apply_inlet_bc(self.f, self.cfg.u_inlet)
        self._apply_outlet_bc(self.f)
        self._apply_symmetry_bc(self.f)
        self.f = self._apply_ibm_force(self.f)
        return self._compute_forces()

    def run(self, callback=None) -> List[Dict]:
        for step in range(self.cfg.max_steps):
            Fx, Fy = self.step()
            if step % self.cfg.sample_interval == 0:
                Cd = self._compute_drag_coefficient(Fx)
                Cl = self._compute_lift_coefficient(Fy)
                record = {"step": step, "Fx": Fx, "Fy": Fy, "Cd": Cd, "Cl": Cl}
                self.force_history.append(record)
                if callback:
                    callback(step, record)
        return self.force_history

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        rho, ux, uy = macroscopic(self.f)
        return ux.cpu().numpy(), uy.cpu().numpy()

    def get_converged_drag(self, warmup_frac: float = 0.5) -> float:
        n = len(self.force_history)
        if n == 0:
            return 0.0
        warmup_idx = int(n * warmup_frac)
        cds = [r["Cd"] for r in self.force_history[warmup_idx:]]
        return float(np.mean(cds))


def run_cylinder_flow(config: CylinderFlowConfig) -> CylinderFlowSolver:
    solver = CylinderFlowSolver(config)
    solver.run()
    return solver


def run_re20_validation() -> Dict:
    config = CylinderFlowConfig(
        nx=320, ny=160,
        cylinder_x=80.0, cylinder_y=80.0, cylinder_radius=20.0,
        u_inlet=0.1, re=20.0,
        max_steps=8000, sample_interval=100,
    )
    solver = CylinderFlowSolver(config)

    def callback(step, record):
        if step % 1000 == 0:
            print(f"Step {step}: Cd = {record['Cd']:.4f}")

    solver.run(callback=callback)

    Cd_computed = solver.get_converged_drag()
    Cd_ref = CYLINDER_CD_2D[20]["Cd"]
    rel_error = abs(Cd_computed - Cd_ref) / Cd_ref * 100

    return {
        "config": {"nx": config.nx, "ny": config.ny, "re": config.re,
                   "u_inlet": config.u_inlet, "cylinder_radius": config.cylinder_radius,
                   "n_markers": solver.n_markers},
        "results": {"Cd_computed": Cd_computed, "Cd_reference": Cd_ref,
                    "rel_error_pct": rel_error, "n_steps": config.max_steps,
                    "n_samples": len(solver.force_history)},
        "force_history": solver.force_history,
        "reference": CYLINDER_CD_2D[20],
    }
