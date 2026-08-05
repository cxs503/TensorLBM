"""SUBOFF bare-hull D3Q19+MRT validation runner.

Runs a small-grid SUBOFF bare-hull case with D3Q19 MRT collision,
bounce-back solid boundary, and static wall.  Verifies the wall-function
admission gate in a real config and produces a ``measured_candidate``
evidence artifact with force/Ct time series.

This runner composes existing cold-path admission control
(:mod:`tensorlbm.wall_function_admission`) with existing solver operators
(:mod:`tensorlbm.solver3d`, :mod:`tensorlbm.obstacles`,
:mod:`tensorlbm.boundaries3d`).  It does **not** modify any solver hot path.

The evidence is deliberately a ``measured_candidate`` — the run produces
real force/Ct observations from an actual D3Q19+MRT+bounce-back loop, but
no physical validation, convergence, or steady-state claim is made.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .boundaries3d import (
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import collide_mrt3d, correct_mass3d, stream3d
from .suboff_cad import SuboffHullType, build_suboff_mask
from .suboff_resistance import _voxel_wetted_area
from .wall_function_admission import WallFunctionRunRequest, require_wall_function_run
from .wall_function_contract import WallFunctionCapability, WallFunctionCompatibilityError

__all__ = [
    "SuboffValidationConfig",
    "SuboffValidationEvidence",
    "run_suboff_d3q19_mrt_validation",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuboffValidationConfig:
    """Configuration for the SUBOFF D3Q19+MRT validation run.

    Defaults are chosen for a fast small-grid run (48×24×24, 20 steps)
    that exercises the full admission→run→force/Ct chain.
    """

    nx: int = 48
    ny: int = 24
    nz: int = 24
    n_steps: int = 20
    warmup: int = 5
    u_in: float = 0.06
    re: float = 200.0
    hull_length: float = 24.0
    device: str = "cpu"
    use_wall_function: bool = False
    # The following two fields exist solely to exercise the admission gate's
    # withholding logic in tests.  They do not alter the solver loop.
    lattice: str = "D3Q19"
    free_surface: bool = False

    def __post_init__(self) -> None:
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, nz must be at least 16, 8, 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.u_in <= 0.0 or self.u_in >= 0.15:
            raise ValueError("u_in must be in (0, 0.15)")
        if self.re <= 0.0:
            raise ValueError("re must be > 0")
        if self.hull_length <= 0.0:
            raise ValueError("hull_length must be > 0")

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """MRT relaxation time for shear stress."""
        return 3.0 * self.nu + 0.5


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuboffValidationEvidence:
    """Measured-candidate evidence from a real D3Q19+MRT+bounce-back run.

    Attributes:
        status: Always ``"measured_candidate"`` — real observations, no
            physical validation claim.
        physical_validation: Always ``False``.
        steady_state: Always ``"diagnostic_withheld"`` — no steady-state
            convergence assertion is made.
        admission: Cold-path admission gate record.
        force_time_series: Per-step (fx, fy, fz) in lattice force units.
        ct_time_series: Per-step (ct, ct_fric, ct_pres) coefficients.
        wetted_area: Voxel wetted area in lattice units.
        dynamic_pressure: 0.5 * rho * U^2 * S in lattice units.
        runtime: Solver runtime evidence (steps, finiteness, density range).
        config: Solver configuration snapshot.
    """

    status: str
    physical_validation: bool
    steady_state: str
    admission: dict[str, Any]
    force_time_series: list[dict[str, Any]]
    ct_time_series: list[dict[str, Any]]
    wetted_area: float
    dynamic_pressure: float
    runtime: dict[str, Any]
    config: dict[str, Any]

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence artifact."""
        return {
            "schema": "tensorlbm.suboff-d3q19-mrt-validation/v1",
            "status": self.status,
            "physical_validation": self.physical_validation,
            "steady_state": self.steady_state,
            "admission": self.admission,
            "force_time_series": self.force_time_series,
            "ct_time_series": self.ct_time_series,
            "wetted_area": self.wetted_area,
            "dynamic_pressure": self.dynamic_pressure,
            "runtime": self.runtime,
            "config": self.config,
        }

    def write_artifact(self, path: str | Path) -> None:
        """Write the evidence artifact as a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_artifact(), sort_keys=True, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _admission_record(config: SuboffValidationConfig) -> dict[str, Any]:
    """Execute the cold-path admission gate and return a record.

    When ``use_wall_function=False`` the gate is skipped entirely — the
    run uses bounce-back only, no wall function.

    When ``use_wall_function=True`` the gate is called with the exact
    D3Q19/MRT_SMAGORINSKY/static_voxel_solid/torch tuple.  Unlisted
    combinations (D3Q27, free-surface, AMR) raise
    ``WallFunctionCompatibilityError`` before any solver execution.
    """
    if not config.use_wall_function:
        return {
            "status": "skipped",
            "reason": (
                "use_wall_function=False; bounce-back solid boundary only, "
                "no wall function admitted"
            ),
        }

    record = require_wall_function_run(WallFunctionRunRequest(
        capability=WallFunctionCapability.LOG_LAW_BODY_FORCE,
        lattice=config.lattice,
        physics="single_phase_incompressible",
        collision="MRT_SMAGORINSKY",
        geometry="static_voxel_solid",
        backend="torch",
        free_surface=config.free_surface,
    ))
    return {
        "status": "admitted",
        "capability": WallFunctionCapability.LOG_LAW_BODY_FORCE.value,
        "lattice": config.lattice,
        "physics": "single_phase_incompressible",
        "collision": "MRT_SMAGORINSKY",
        "geometry": "static_voxel_solid",
        "backend": "torch",
        "validation": "IMPLEMENTATION_ONLY",
        "note": record.note,
    }


def _pressure_drag_x(
    f: torch.Tensor,
    solid: torch.Tensor,
) -> float:
    """Pressure drag on the body in the x-direction (lattice units).

    Uses the same sign convention as the reference D3Q19 wall-function
    solver: a fluid cell with a solid neighbour in +x contributes +p
    streamwise drag.
    """
    rho, _, _, _ = macroscopic3d(f)
    p = (rho - 1.0) / 3.0
    fluid = ~solid
    sp = torch.roll(solid, 1, dims=2)   # solid neighbour in -x
    sm = torch.roll(solid, -1, dims=2)  # solid neighbour in +x
    return float(
        (p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    )


def run_suboff_d3q19_mrt_validation(
    config: SuboffValidationConfig | None = None,
) -> SuboffValidationEvidence:
    """Run SUBOFF bare-hull D3Q19+MRT+bounce-back and produce evidence.

    This function:
      1. Executes the cold-path wall-function admission gate.
      2. Builds a SUBOFF bare-hull solid mask on a small grid.
      3. Runs a real D3Q19 MRT collide→stream→force→bounce-back→BC loop.
      4. Records per-step force/Ct time series.
      5. Returns a ``measured_candidate`` evidence artifact.

    The solver hot path is not modified.  Only existing operators are
    composed.
    """
    if config is None:
        config = SuboffValidationConfig()

    # --- 1. Cold-path admission gate (before any solver execution) ---
    admission = _admission_record(config)

    # --- 2. Build geometry ---
    device = torch.device(config.device)
    cx = config.nx * 0.35
    cy = config.ny / 2.0
    cz = config.nz / 2.0
    solid, _stats = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=config.hull_length,
        device=str(device),
    )
    solid = solid.to(device)
    wall_mask = make_channel_wall_mask_3d(
        config.nz, config.ny, config.nx, solid, device=device,
    )

    # Wetted area and dynamic pressure for Ct normalization
    wetted_area = _voxel_wetted_area(solid, 1.0)
    rho_lu = 1.0
    dynamic_pressure = 0.5 * rho_lu * config.u_in ** 2 * wetted_area

    # --- 3. Initialize populations ---
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    initial_mass = float(f.sum().item())

    tau = config.tau

    # --- 4. Solver loop ---
    force_series: list[dict[str, Any]] = []
    ct_series: list[dict[str, Any]] = []
    completed_steps = 0
    finite_population_checks = 0
    finite_density_checks = 0
    all_populations_finite = True
    all_densities_finite = True
    density_min = float("inf")
    density_max = float("-inf")

    for step in range(1, config.n_steps + 1):
        # Collision (MRT)
        f = collide_mrt3d(f, tau=tau)
        # Streaming (pull scheme)
        f = stream3d(f)
        # Force measurement (momentum exchange, before bounce-back)
        fx_t, fy_t, fz_t = compute_obstacle_forces_3d(f, solid)
        fx = float(fx_t.item())
        fy = float(fy_t.item())
        fz = float(fz_t.item())
        # Pressure drag (from density field)
        dp = _pressure_drag_x(f, solid)
        df = fx - dp  # friction drag = total - pressure

        # Bounce-back on solid (static wall)
        f = bounce_back_cells_3d(f, solid)
        # Boundary conditions
        f = zou_he_inlet_velocity_3d(f, config.u_in)
        f = zou_he_outlet_pressure_3d(f)
        f = bounce_back_cells_3d(f, wall_mask)
        # Mass correction every 10 steps
        if step % 10 == 0:
            f = correct_mass3d(f, initial_mass)

        # Record force time series
        force_series.append({
            "step": step,
            "fx": fx,
            "fy": fy,
            "fz": fz,
        })

        # Record Ct time series
        ct = fx / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_fric = df / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_pres = dp / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_series.append({
            "step": step,
            "ct": ct,
            "ct_fric": ct_fric,
            "ct_pres": ct_pres,
        })

        # Runtime finiteness checks
        completed_steps = step
        populations_finite = bool(torch.isfinite(f).all().item())
        finite_population_checks += 1
        all_populations_finite = all_populations_finite and populations_finite

        rho_step, _, _, _ = macroscopic3d(f)
        densities_finite = bool(torch.isfinite(rho_step).all().item())
        finite_density_checks += 1
        all_densities_finite = all_densities_finite and densities_finite
        if densities_finite:
            density_min = min(density_min, float(rho_step.min().item()))
            density_max = max(density_max, float(rho_step.max().item()))

    # --- 5. Build evidence ---
    runtime = {
        "requested_steps": config.n_steps,
        "completed_steps": completed_steps,
        "finite_population_checks": finite_population_checks,
        "finite_density_checks": finite_density_checks,
        "all_populations_finite": all_populations_finite,
        "all_densities_finite": all_densities_finite,
        "density_min": density_min if density_min != float("inf") else 0.0,
        "density_max": density_max if density_max != float("-inf") else 0.0,
        "device": config.device,
        "tau": tau,
        "nu": config.nu,
    }

    config_snapshot = {
        "lattice": "D3Q19",
        "collision": "MRT",
        "boundary": "bounce_back",
        "wall": "static",
        "hull_type": "bare_hull",
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "n_steps": config.n_steps,
        "warmup": config.warmup,
        "u_in": config.u_in,
        "re": config.re,
        "hull_length": config.hull_length,
        "use_wall_function": config.use_wall_function,
    }

    return SuboffValidationEvidence(
        status="measured_candidate",
        physical_validation=False,
        steady_state="diagnostic_withheld",
        admission=admission,
        force_time_series=force_series,
        ct_time_series=ct_series,
        wetted_area=wetted_area,
        dynamic_pressure=dynamic_pressure,
        runtime=runtime,
        config=config_snapshot,
    )
