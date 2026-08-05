"""Uniform-fine reference benchmark for a 2:1 static LBM interface."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .fixed_nested_transfer import restrict_populations_2to1
from .refinement import BoxRegion
from .solver3d import collide_mrt3d, stream3d
from .static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
    convective_refined_tau,
)


@dataclass(frozen=True)
class AMRInterfaceValidationConfig:
    shape_zyx: tuple[int, int, int] = (20, 24, 40)
    box: BoxRegion = BoxRegion(x0=14, x1=28, y0=5, y1=19, z0=4, z1=16)
    tau_coarse: float = 0.56
    velocity_x: float = 0.04
    perturbation: float = 1.0e-3
    pulse_radius: float = 2.5
    steps: int = 24
    device: str = "cpu"
    regularize_prolongation: bool = False
    reflux_correction_stencil: str = "exterior_cells"
    interface_filter_width: int = 0
    interface_filter_strength: float = 0.0

    def validate(self) -> None:
        nz, ny, nx = self.shape_zyx
        if min(self.shape_zyx) < 8:
            raise ValueError("validation domain is too small")
        if not (
            1 < self.box.x0 < self.box.x1 < nx - 2
            and 1 < self.box.y0 < self.box.y1 < ny - 2
            and 1 < self.box.z0 < self.box.z1 < nz - 2
        ):
            raise ValueError("validation box needs a two-cell exterior margin")
        if self.tau_coarse <= 0.5:
            raise ValueError("tau_coarse must exceed 0.5")
        if not 0.0 < self.velocity_x < 0.15:
            raise ValueError("velocity_x must lie in (0,0.15)")
        if not 0.0 < self.perturbation < 0.05 or self.pulse_radius <= 0.0:
            raise ValueError("invalid pulse parameters")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if (self.interface_filter_width == 0) != (
            self.interface_filter_strength == 0.0
        ):
            raise ValueError(
                "interface filter width and strength must both be zero or positive",
            )
        if self.reflux_correction_stencil not in (
            "exterior_cells", "crossing_links",
        ):
            raise ValueError(
                "reflux_correction_stencil must be exterior_cells or crossing_links",
            )


def _initial_coarse(config: AMRInterfaceValidationConfig, device: torch.device) -> torch.Tensor:
    nz, ny, nx = config.shape_zyx
    z, y, x = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32), indexing="ij",
    )
    center_x = config.box.x1 - 1.5
    center_y = 0.5 * (config.box.y0 + config.box.y1 - 1)
    center_z = 0.5 * (config.box.z0 + config.box.z1 - 1)
    radius2 = (
        (x - center_x).square()
        + (y - center_y).square()
        + (z - center_z).square()
    )
    rho = 1.0 + config.perturbation * torch.exp(
        -radius2 / (2.0 * config.pulse_radius**2),
    )
    ux = torch.full_like(rho, config.velocity_x)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, ux, zero, zero, device=device)


def _repeat_2to1(populations: torch.Tensor) -> torch.Tensor:
    return populations.repeat_interleave(2, dim=1).repeat_interleave(
        2, dim=2,
    ).repeat_interleave(2, dim=3)


def _advance_periodic(state: torch.Tensor, tau: float) -> AMRAdvanceResult:
    post = collide_mrt3d(state, tau=tau)
    return AMRAdvanceResult(stream3d(post), post)


def _rms(field: torch.Tensor, mask: torch.Tensor) -> float:
    return float(torch.sqrt(field[mask].square().mean()).item())


def _interface_shell(
    shape: tuple[int, int, int], box: BoxRegion, device: torch.device,
) -> torch.Tensor:
    shell = torch.zeros(shape, dtype=torch.bool, device=device)
    shell[
        box.z0 - 1:box.z1 + 1,
        box.y0 - 1:box.y1 + 1,
        box.x0 - 1:box.x1 + 1,
    ] = True
    shell[
        box.z0 + 1:box.z1 - 1,
        box.y0 + 1:box.y1 - 1,
        box.x0 + 1:box.x1 - 1,
    ] = False
    return shell


def run_amr_interface_validation(
    config: AMRInterfaceValidationConfig,
) -> dict[str, object]:
    """Compare the composite AMR state with coarse and uniform-fine runs."""
    config.validate()
    device = torch.device(config.device)
    coarse_initial = _initial_coarse(config, device)
    coarse_baseline = coarse_initial.clone()
    uniform_fine = _repeat_2to1(coarse_initial)
    solver = StaticBlockAMR3D(
        coarse_initial.clone(),
        StaticBlockAMRConfig(
            config.box,
            tau_coarse=config.tau_coarse,
            regularize_prolongation=config.regularize_prolongation,
            reflux_correction_stencil=config.reflux_correction_stencil,
            interface_filter_width=config.interface_filter_width,
            interface_filter_strength=config.interface_filter_strength,
        ),
    )
    tau_fine = convective_refined_tau(config.tau_coarse)
    maximum_reflux_residual = 0.0
    maximum_limited_directions = 0
    maximum_raw_kinetic_mismatch = 0.0
    maximum_requested_correction = 0.0
    maximum_applied_correction_fraction = 0.0
    for _ in range(config.steps):
        coarse_baseline = _advance_periodic(
            coarse_baseline, config.tau_coarse,
        ).populations
        for _ in range(2):
            uniform_fine = _advance_periodic(
                uniform_fine, tau_fine,
            ).populations

        def advance(
            state: torch.Tensor, tau: float, level: int, substep: int,
        ) -> AMRAdvanceResult:
            del level, substep
            return _advance_periodic(state, tau)

        ledger = solver.step(advance)
        maximum_reflux_residual = max(
            maximum_reflux_residual, float(ledger.residual.abs().max().item()),
        )
        maximum_limited_directions = max(
            maximum_limited_directions, ledger.limited_directions,
        )
        if ledger.raw_kinetic_mismatch is None:
            raise RuntimeError("AMR ledger omitted raw kinetic mismatch")
        maximum_raw_kinetic_mismatch = max(
            maximum_raw_kinetic_mismatch,
            float(ledger.raw_kinetic_mismatch.abs().max().item()),
        )
        maximum_requested_correction = max(
            maximum_requested_correction,
            float(ledger.replacement_mismatch.abs().max().item()),
        )
        maximum_applied_correction_fraction = max(
            maximum_applied_correction_fraction,
            ledger.maximum_applied_correction_fraction,
        )

    reference_coarse = restrict_populations_2to1(uniform_fine)
    rho_reference, ux_reference, _, _ = macroscopic3d(reference_coarse)
    rho_amr, ux_amr, _, _ = macroscopic3d(solver.coarse_f)
    rho_coarse, ux_coarse, _, _ = macroscopic3d(coarse_baseline)
    inside = torch.zeros(config.shape_zyx, dtype=torch.bool, device=device)
    b = config.box
    inside[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = True
    shell = _interface_shell(config.shape_zyx, b, device)
    density_amr_error = rho_amr - rho_reference
    density_coarse_error = rho_coarse - rho_reference
    velocity_amr_error = ux_amr - ux_reference
    velocity_coarse_error = ux_coarse - ux_reference
    initial_mass = float(coarse_initial.sum().item())
    final_mass = float(solver.coarse_f.sum().item())
    finite = bool(torch.isfinite(solver.coarse_f).all())
    metrics = {
        "density_rms_global_amr": _rms(density_amr_error, torch.ones_like(inside)),
        "density_rms_global_coarse": _rms(density_coarse_error, torch.ones_like(inside)),
        "density_rms_refined_amr": _rms(density_amr_error, inside),
        "density_rms_refined_coarse": _rms(density_coarse_error, inside),
        "density_rms_interface_amr": _rms(density_amr_error, shell),
        "density_rms_interface_coarse": _rms(density_coarse_error, shell),
        "velocity_x_rms_refined_amr": _rms(velocity_amr_error, inside),
        "velocity_x_rms_refined_coarse": _rms(velocity_coarse_error, inside),
        "relative_mass_drift": abs(final_mass - initial_mass) / abs(initial_mass),
        "minimum_population": float(solver.coarse_f.min().item()),
        "maximum_reflux_population_residual": maximum_reflux_residual,
        "maximum_raw_kinetic_mismatch": maximum_raw_kinetic_mismatch,
        "maximum_conserved_moment_correction": maximum_requested_correction,
        "maximum_applied_correction_fraction": (
            maximum_applied_correction_fraction
        ),
        "maximum_limited_directions": maximum_limited_directions,
        "finite": finite,
    }
    refined_improves_density = (
        metrics["density_rms_refined_amr"]
        <= metrics["density_rms_refined_coarse"]
    )
    density_error_ratio = (
        metrics["density_rms_refined_amr"]
        / max(metrics["density_rms_refined_coarse"], 1e-30)
    )
    velocity_error_ratio = (
        metrics["velocity_x_rms_refined_amr"]
        / max(metrics["velocity_x_rms_refined_coarse"], 1e-30)
    )
    velocity_not_materially_worse = velocity_error_ratio <= 1.05
    metrics["refined_to_coarse_density_error_ratio"] = density_error_ratio
    metrics["refined_to_coarse_velocity_x_error_ratio"] = velocity_error_ratio
    admitted = (
        finite
        and metrics["minimum_population"] > 0.0
        and metrics["relative_mass_drift"] <= 1e-5
        and metrics["maximum_reflux_population_residual"] <= 1e-6
        and refined_improves_density
        and velocity_not_materially_worse
    )
    return {
        "schema": "tensorlbm-amr-interface-validation-v2",
        "configuration": {
            "shape_zyx": list(config.shape_zyx),
            "box": vars(config.box),
            "tau_coarse": config.tau_coarse,
            "tau_fine": tau_fine,
            "velocity_x": config.velocity_x,
            "perturbation": config.perturbation,
            "pulse_radius": config.pulse_radius,
            "steps": config.steps,
            "device": config.device,
            "regularize_prolongation": config.regularize_prolongation,
            "reflux_correction_stencil": config.reflux_correction_stencil,
            "interface_filter_width": config.interface_filter_width,
            "interface_filter_strength": config.interface_filter_strength,
            "reflux_method": "face_local_conserved_moment_flux",
        },
        "mesh": {
            "allocated_cells": solver.total_allocated_cells,
            "uniform_fine_cells": solver.uniform_fine_equivalent_cells,
            "saving_fraction": solver.cell_saving_fraction,
        },
        "result": metrics,
        "acceptance": {
            "relative_mass_drift_target": 1e-5,
            "maximum_reflux_population_residual": 1e-6,
            "refined_region_improves_density": refined_improves_density,
            "maximum_velocity_error_ratio": 1.05,
            "refined_region_velocity_not_materially_worse": (
                velocity_not_materially_worse
            ),
            "admitted": admitted,
        },
    }


__all__ = [
    "AMRInterfaceValidationConfig",
    "run_amr_interface_validation",
]
