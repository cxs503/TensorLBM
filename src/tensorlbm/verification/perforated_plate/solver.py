"""D2Q9 LBM solver for oscillatory flow through a perforated plate.

Geometry:
    2-D channel of length Lx and height Ly.
    Perforated plate at x = Lx/2 with hole array.
    Top/bottom walls: no-slip (bounce-back).
    Inlet/outlet: periodic (or pressure BC).

Driving:
    Sinusoidal body force: F_x(t) = F₀ sin(ω t)

Boundary conditions:
    - Bounce-back (half-way) on top/bottom walls and plate solid cells
    - Periodic in x-direction (or Zou/He pressure BC)

Force measurement:
    - Momentum exchange on plate cells → drag force F_plate(t)
    - Pressure drop Δp(t) from density difference across plate

Output time series:
    - Q(t): volume flow rate (integral of u_x across channel)
    - Δp(t): pressure drop across plate
    - F_plate(t): total force on plate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from tensorlbm.d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
from tensorlbm.boundary.boundaries import bounce_back_cells, compute_obstacle_forces


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerforatedPlateConfig:
    """Configuration for perforated plate oscillatory flow simulation.

    Parameters
    ----------
    nx, ny : int
        Grid resolution (nx in streamwise, ny in wall-normal).
    plate_col : int
        Column index of the perforated plate (default: nx // 2).
    hole_pattern : str
        'single' (one centered hole) or 'array' (multiple holes).
    n_holes : int
        Number of holes in array pattern (default: 1).
    hole_size : int
        Hole height in lattice units (number of fluid cells).
    beta : float
        Open area ratio (override; if set, hole_size is computed from it).
    tau : float
        BGK relaxation time (controls viscosity: ν = (τ - 0.5) / 3).
    force_amplitude : float
        Amplitude of sinusoidal body force F₀.
    omega : float
        Angular frequency of oscillation ω = 2π/T.
    steps : int
        Total number of time steps.
    warmup_cycles : int
        Number of warmup cycles before data collection.
    samples_per_cycle : int
        Number of samples per oscillation cycle for phase averaging.
    n_cycles_measure : int
        Number of cycles to measure after warmup.
    device : str
        Computation device ('cpu' or 'cuda').
    dtype : str
        Floating-point precision ('float32' or 'float64').
    label : str
        Simulation label for output.
    """
    nx: int = 192
    ny: int = 64
    plate_col: int = -1  # auto: nx // 2
    hole_pattern: str = "single"
    n_holes: int = 1
    hole_size: int = 10
    beta: float = -1.0  # auto-compute from hole_size if < 0
    tau: float = 1.0
    force_amplitude: float = 1e-5
    omega: float = 0.01
    steps: int = 10000
    warmup_cycles: int = 3
    samples_per_cycle: int = 32
    n_cycles_measure: int = 5
    device: str = "cpu"
    dtype: str = "float64"
    label: str = "perforated_plate"

    @property
    def plate_col_auto(self) -> int:
        return self.plate_col if self.plate_col > 0 else self.nx // 2

    @property
    def nu(self) -> float:
        """Kinematic viscosity from BGK tau."""
        return (self.tau - 0.5) / 3.0

    @property
    def period(self) -> float:
        """Oscillation period T = 2π/ω."""
        return 2.0 * math.pi / self.omega

    @property
    def period_steps(self) -> int:
        """Number of time steps per oscillation period."""
        return max(1, round(self.period))

    @property
    def total_cycles(self) -> int:
        return self.warmup_cycles + self.n_cycles_measure

    @property
    def u_max_estimate(self) -> float:
        """Estimate of max velocity from force balance (Poiseuille-like)."""
        # For channel flow: u_max ≈ F₀ · H² / (8ν)  (parabolic profile peak)
        h = self.ny - 2  # channel height in lu
        return self.force_amplitude * h * h / (8.0 * self.nu + 1e-15)

    @property
    def kc_estimate(self) -> float:
        """Estimated Keulegan-Carpenter number."""
        d_char = max(self.hole_size, 1)
        return self.u_max_estimate * self.period / d_char

    @property
    def re_estimate(self) -> float:
        """Estimated Reynolds number based on hole size."""
        d_char = max(self.hole_size, 1)
        return self.u_max_estimate * d_char / (self.nu + 1e-15)

    def validate(self) -> None:
        if self.nx < 32 or self.ny < 16:
            raise ValueError("Grid too small")
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5 for stability")
        if self.force_amplitude <= 0:
            raise ValueError("force_amplitude must be positive")
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        pc = self.plate_col_auto
        if pc < 2 or pc >= self.nx - 2:
            raise ValueError("plate_col must be interior")
        if self.hole_size < 1:
            raise ValueError("hole_size must be >= 1")
        if self.hole_pattern not in ("single", "array"):
            raise ValueError("hole_pattern must be 'single' or 'array'")


# ---------------------------------------------------------------------------
# Geometry: perforated plate mask
# ---------------------------------------------------------------------------

def make_perforated_plate_mask(
    nx: int,
    ny: int,
    plate_col: int,
    hole_pattern: str = "single",
    n_holes: int = 1,
    hole_size: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create boolean mask for perforated plate.

    True = solid (bounce-back), False = fluid (hole).

    The plate is 2 cells thick (plate_col-1, plate_col) for numerical stability.
    Holes are centered vertically and evenly spaced for array pattern.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    plate_col : int
        Center column of the plate.
    hole_pattern : str
        'single' or 'array'.
    n_holes : int
        Number of holes (for array pattern).
    hole_size : int
        Height of each hole in lattice units.
    device : torch.device
        Device for the tensor.

    Returns
    -------
    mask : torch.Tensor
        Boolean mask of shape (ny, nx), True = solid.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)

    # Plate is 2 cells thick for half-way bounce-back
    cols = [plate_col - 1, plate_col]
    for col in cols:
        if col < 0 or col >= nx:
            continue
        if hole_pattern == "single":
            # Single centered hole
            hole_center = ny // 2
            hole_half = hole_size // 2
            hole_start = max(1, hole_center - hole_half)
            hole_end = min(ny - 1, hole_center + hole_half)
            mask[:, col] = True
            mask[hole_start:hole_end, col] = False
        elif hole_pattern == "array":
            # Multiple evenly spaced holes
            if n_holes < 1:
                n_holes = 1
            spacing = (ny - 2) // n_holes
            for h in range(n_holes):
                hole_center = 1 + spacing * h + spacing // 2
                hole_half = hole_size // 2
                hole_start = max(1, hole_center - hole_half)
                hole_end = min(ny - 1, hole_center + hole_half)
                mask[hole_start:hole_end, col] = False
            mask[:, col] = True
            # Re-open holes
            for h in range(n_holes):
                hole_center = 1 + spacing * h + spacing // 2
                hole_half = hole_size // 2
                hole_start = max(1, hole_center - hole_half)
                hole_end = min(ny - 1, hole_center + hole_half)
                mask[hole_start:hole_end, col] = False

    return mask


def compute_open_area_ratio(
    mask: torch.Tensor,
    plate_col: int,
    ny: int,
) -> float:
    """Compute open area ratio β from plate mask."""
    col = plate_col
    if col < 0 or col >= mask.shape[1]:
        return 1.0
    total = mask.shape[0] - 2  # exclude walls
    fluid = (~mask[:, col]).sum().item() - 2  # exclude wall rows
    return max(0.0, min(1.0, fluid / max(total, 1)))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PerforatedPlateSolver:
    """D2Q9 LBM solver for oscillatory flow through a perforated plate.

    Uses BGK collision, half-way bounce-back, and momentum-exchange force
    measurement. Driven by sinusoidal body force.
    """

    def __init__(self, cfg: PerforatedPlateConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dt = torch.float32 if cfg.dtype == "float32" else torch.float64

        self.nx = cfg.nx
        self.ny = cfg.ny
        self.plate_col = cfg.plate_col_auto
        self.tau = cfg.tau
        self.nu = cfg.nu

        # Lattice vectors
        self.C = C.to(self.device)
        self.W = W.to(self.device, dtype=self.dt)
        self.OPPOSITE = OPPOSITE.to(self.device)

        # Wall mask (top and bottom rows)
        self.wall_mask = torch.zeros((self.ny, self.nx), dtype=torch.bool, device=self.device)
        self.wall_mask[0, :] = True
        self.wall_mask[-1, :] = True

        # Plate mask
        self.plate_mask = make_perforated_plate_mask(
            self.nx, self.ny, self.plate_col,
            cfg.hole_pattern, cfg.n_holes, cfg.hole_size,
            self.device,
        )
        self.beta = compute_open_area_ratio(self.plate_mask, self.plate_col, self.ny)

        # Combined solid mask (walls + plate)
        self.solid_mask = self.wall_mask | self.plate_mask

        # Initialize populations at equilibrium (rest)
        rho0 = torch.ones((self.ny, self.nx), device=self.device, dtype=self.dt)
        ux0 = torch.zeros((self.ny, self.nx), device=self.device, dtype=self.dt)
        uy0 = torch.zeros((self.ny, self.nx), device=self.device, dtype=self.dt)
        self.f = equilibrium(rho0, ux0, uy0, device=self.device)

        # Time tracking
        self.step_count = 0

        # History
        self.history: dict = {
            "time": [],
            "force_x": [],
            "force_y": [],
            "flow_rate": [],
            "delta_p": [],
            "u_center": [],
        }

    def _apply_body_force(self, t: float) -> None:
        """Apply sinusoidal body force using Guo forcing scheme.

        F_x(t) = F₀ sin(ωt)
        """
        fx = self.cfg.force_amplitude * math.sin(self.cfg.omega * t)
        cx = self.C[:, 0].view(9, 1, 1).to(self.dt)

        # Guo forcing: f_i += w_i * (1 - 1/(2τ)) * (c_i · F)
        factor = 1.0 - 0.5 / self.tau
        force_term = self.W.view(9, 1, 1) * cx * fx * factor
        self.f = self.f + force_term

    def _collide(self) -> None:
        """BGK collision step."""
        rho, ux, uy = macroscopic(self.f, device=self.device)
        feq = equilibrium(rho, ux, uy, device=self.device)
        self.f = self.f - (self.f - feq) / self.tau

    def _stream(self) -> None:
        """Streaming step with periodic boundaries in x."""
        streamed = torch.empty_like(self.f)
        for i in range(9):
            cy = int(self.C[i, 1])
            cx = int(self.C[i, 0])
            if cy == 0 and cx == 0:
                streamed[i] = self.f[i]
            else:
                streamed[i] = torch.roll(self.f[i], shifts=(cy, cx), dims=(-2, -1))
        self.f = streamed

    def _apply_bounce_back(self) -> None:
        """Half-way bounce-back on solid cells."""
        self.f = bounce_back_cells(self.f, self.solid_mask)

    def _measure_flow_rate(self) -> float:
        """Compute volume flow rate Q = ∫ u_x dy at plate location."""
        rho, ux, uy = macroscopic(self.f, device=self.device)
        # Integrate at plate column (use upstream side)
        col = max(0, self.plate_col - 3)
        q = float(ux[:, col].sum().item())
        return q

    def _measure_pressure_drop(self) -> float:
        """Compute pressure drop Δp across the plate.

        Δp = c_s² (ρ_upstream - ρ_downstream)
        c_s² = 1/3 for D2Q9
        """
        rho, ux, uy = macroscopic(self.f, device=self.device)
        col_up = max(0, self.plate_col - 4)
        col_down = min(self.nx - 1, self.plate_col + 4)

        # Average density in a band around the plate
        band = 3
        rho_up = rho[:, col_up - band:col_up + band + 1].mean().item()
        rho_down = rho[:, col_down - band:col_down + band + 1].mean().item()

        cs2 = 1.0 / 3.0
        return cs2 * (rho_up - rho_down)

    def _measure_plate_force(self) -> tuple[float, float]:
        """Momentum-exchange force on the plate.

        Must be called after streaming, before bounce-back.
        """
        # Only plate cells (not walls)
        plate_only = self.plate_mask & (~self.wall_mask)
        if not plate_only.any():
            return 0.0, 0.0

        cx = self.C[:, 0].view(9, 1, 1).to(self.dt)
        cy = self.C[:, 1].view(9, 1, 1).to(self.dt)
        mask_3d = plate_only.unsqueeze(0)
        f_solid = self.f * mask_3d

        fx = 2.0 * (cx * f_solid).sum().item()
        fy = 2.0 * (cy * f_solid).sum().item()
        return fx, fy

    def step(self, t: float) -> dict:
        """One time step: collide → stream → measure force → bounce-back → force.

        Parameters
        ----------
        t : float
            Physical time (for body force evaluation).

        Returns
        -------
        dict with measured quantities.
        """
        # Collide
        self._collide()

        # Stream
        self._stream()

        # Measure force on plate (after streaming, before bounce-back)
        fx, fy = self._measure_plate_force()

        # Apply bounce-back
        self._apply_bounce_back()

        # Apply body force
        self._apply_body_force(t)

        self.step_count += 1

        # Measure flow rate and pressure drop
        q = self._measure_flow_rate()
        dp = self._measure_pressure_drop()

        # Centerline velocity
        rho, ux, uy = macroscopic(self.f, device=self.device)
        u_center = float(ux[self.ny // 2, self.plate_col - 3].item())

        return {
            "force_x": fx,
            "force_y": fy,
            "flow_rate": q,
            "delta_p": dp,
            "u_center": u_center,
        }

    def run(self) -> dict:
        """Run the full simulation.

        Returns
        -------
        dict with time series and metadata.
        """
        cfg = self.cfg
        total_steps = cfg.steps

        # Pre-compute measurement times
        warmup_steps = cfg.warmup_cycles * cfg.period_steps
        measure_steps = cfg.n_cycles_measure * cfg.period_steps

        time_series = {
            "time": [],
            "force_x": [],
            "force_y": [],
            "flow_rate": [],
            "delta_p": [],
            "u_center": [],
        }

        for step in range(1, total_steps + 1):
            t = step  # lattice time = step number (dt=1)
            result = self.step(t)

            # Record during measurement phase
            if step > warmup_steps:
                time_series["time"].append(t)
                time_series["force_x"].append(result["force_x"])
                time_series["force_y"].append(result["force_y"])
                time_series["flow_rate"].append(result["flow_rate"])
                time_series["delta_p"].append(result["delta_p"])
                time_series["u_center"].append(result["u_center"])

        # Convert to numpy arrays
        for key in time_series:
            time_series[key] = np.array(time_series[key], dtype=np.float64)

        return {
            "time_series": time_series,
            "config": cfg,
            "beta": self.beta,
            "plate_col": self.plate_col,
            "nu": self.nu,
            "tau": self.tau,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
        }


def run_perforated_plate(cfg: PerforatedPlateConfig) -> dict:
    """Convenience function to run a perforated plate simulation."""
    solver = PerforatedPlateSolver(cfg)
    result = solver.run()
    result["solver"] = solver
    return result
