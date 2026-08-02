"""Canonical unconfined cylinder drag using BFL and control-volume force."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .bfl_d3q19 import bouzidi_bounce_back_d3q19, compute_q_cylinder_d3q19
from .boundaries3d import far_field_bc_3d
from .checkpoint_io import atomic_torch_save
from .chunked_collision import (
    NaturalKBCCollisionExecutor,
    collide_in_z_chunks,
)
from .control_volume_force import box_control_volume, observe_control_volume_force
from .cuda_memory_budget import require_cuda_memory_budget
from .cumulant import collide_cumulant_d3q19
from .d3q19 import equilibrium3d, macroscopic3d
from .external_open_boundary import non_equilibrium_far_field_bc_3d
from .force_convergence import assess_force_stationarity
from .solver3d import stream3d
from .sponge_layer import apply_equilibrium_difference_sponge, build_sponge_sigma_3d

CYLINDER_RE100_CD_REFERENCE = 1.33
CYLINDER_RE100_ST_REFERENCE = 0.164


@dataclass(frozen=True)
class CylinderBFLControlVolumeConfig:
    nx: int = 320
    ny: int = 200
    nz: int = 3
    radius: float = 12.0
    center_x_fraction: float = 0.30
    reynolds: float = 100.0
    lattice_speed: float = 0.06
    steps: int = 8000
    warmup_steps: int = 4000
    ramp_steps: int = 500
    sponge_width: int = 24
    sponge_strength: float = 0.2
    sponge_inlet: bool = False
    cv_margin: int = 8
    far_field_mode: str = "non_equilibrium_extrapolation"
    report_interval: int = 1000
    checkpoint_interval: int = 0
    checkpoint_path: str | None = None
    resume: bool = False
    statistics_window_steps: int = 0
    minimum_shedding_cycles: float = 8.0
    collision_model: str = "cumulant_d3q19_cs0"
    collision_chunk_cells: int = 0
    compile_natural_kbc: bool = False
    device: str = "cpu"

    @property
    def nu(self) -> float:
        return self.lattice_speed * 2.0 * self.radius / self.reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def domain_clearance_diameters(self) -> dict[str, float]:
        """Far-field centre distances in cylinder diameters."""
        diameter = 2.0 * self.radius
        cx = self.nx * self.center_x_fraction
        return {
            "upstream_center_distance": cx / diameter,
            "downstream_center_distance": (self.nx - cx) / diameter,
            "lateral_center_distance": (self.ny / 2.0) / diameter,
        }

    @property
    def domain_reference_adequate(self) -> bool:
        clearance = self.domain_clearance_diameters
        return (
            clearance["upstream_center_distance"] >= 5.0
            and clearance["downstream_center_distance"] >= 10.0
            and clearance["lateral_center_distance"] >= 10.0
        )

    def validate(self) -> None:
        if min(self.nx, self.ny) < 16 or self.nz < 1:
            raise ValueError("cylinder domain is too small")
        if self.radius < 3.0 or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("invalid radius or time window")
        if not 0.0 < self.center_x_fraction < 1.0:
            raise ValueError("center_x_fraction must lie in (0,1)")
        cx = self.nx * self.center_x_fraction
        if min(cx, self.nx - cx, self.ny / 2) <= self.radius + self.cv_margin + 2:
            raise ValueError("cylinder/control volume does not fit")
        if self.far_field_mode not in {
            "non_equilibrium_extrapolation", "legacy_hard_equilibrium",
        }:
            raise ValueError("unknown far_field_mode")
        if self.collision_model not in {
            "cumulant_d3q19_cs0", "natural_kbc_d3q19",
        }:
            raise ValueError("unknown collision_model")
        if self.collision_chunk_cells < 0:
            raise ValueError("collision_chunk_cells must be non-negative")
        if self.compile_natural_kbc and self.collision_model != "natural_kbc_d3q19":
            raise ValueError("compiled natural KBC requires natural_kbc_d3q19")
        if self.report_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError("report/checkpoint intervals must be non-negative")
        if not 0 <= self.statistics_window_steps <= self.steps - self.warmup_steps:
            raise ValueError(
                "statistics_window_steps must be zero or fit after warmup",
            )
        if self.minimum_shedding_cycles < 0.0:
            raise ValueError("minimum_shedding_cycles must be non-negative")
        if self.ramp_steps < 0 or self.sponge_width < 0:
            raise ValueError("ramp_steps and sponge_width must be non-negative")
        if not 0.0 <= self.sponge_strength <= 1.0:
            raise ValueError("sponge_strength must lie in [0,1]")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")


def _ramp(step: int, steps: int) -> float:
    if steps <= 0 or step >= steps:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * step / steps))


def estimate_strouhal_from_lift(
    lift_coefficients: list[float],
    *,
    lattice_speed: float,
    diameter: float,
) -> tuple[float, float]:
    """Estimate shedding Strouhal number and observed cycle count by FFT."""
    count = len(lift_coefficients)
    if count < 16 or lattice_speed <= 0.0 or diameter <= 0.0:
        return math.nan, 0.0
    signal = torch.tensor(lift_coefficients, dtype=torch.float64)
    index = torch.arange(count, dtype=torch.float64)
    centered_index = index - index.mean()
    slope = (
        (centered_index * (signal - signal.mean())).sum()
        / centered_index.square().sum().clamp_min(1e-30)
    )
    detrended = signal - signal.mean() - slope * centered_index
    window = torch.hann_window(count, periodic=True, dtype=torch.float64)
    spectrum = torch.fft.rfft(detrended * window).abs().square()
    spectrum[0] = 0.0
    peak = int(spectrum.argmax().item())
    peak_bin = float(peak)
    if 0 < peak < spectrum.numel() - 1:
        left, center, right = (
            float(spectrum[peak - 1]), float(spectrum[peak]),
            float(spectrum[peak + 1]),
        )
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-30:
            peak_bin += 0.5 * (left - right) / denominator
    frequency = peak_bin / count
    strouhal = frequency * diameter / lattice_speed
    return strouhal, frequency * count


def run_cylinder_bfl_control_volume(
    config: CylinderBFLControlVolumeConfig,
) -> dict[str, object]:
    invocation_started = time.perf_counter()
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    shape = (config.nz, config.ny, config.nx)
    estimated_peak_gib = math.prod(shape) * 1000.0 / 2**30
    memory_budget = require_cuda_memory_budget(
        device, estimated_peak_gib=estimated_peak_gib,
        reserve_gib=1.0, label="cylinder benchmark",
    )
    cx, cy = config.nx * config.center_x_fraction, config.ny / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(config.ny, device=device, dtype=torch.float32),
        torch.arange(config.nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cross_section = (xx - cx).square() + (yy - cy).square() <= config.radius**2
    solid = cross_section.unsqueeze(0).expand(shape)
    bfl_mask, bfl_q = compute_q_cylinder_d3q19(
        config.nx, config.ny, config.nz, cx, cy, config.radius, device,
    )
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, config.lattice_speed)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, zero, zero, device=device)
    solid_q = solid.unsqueeze(0).expand_as(f)
    cv = box_control_volume(
        shape,
        x0=int(math.floor(cx - config.radius)) - config.cv_margin,
        x1=int(math.ceil(cx + config.radius)) + config.cv_margin + 1,
        y0=int(math.floor(cy - config.radius)) - config.cv_margin,
        y1=int(math.ceil(cy + config.radius)) + config.cv_margin + 1,
        z0=0, z1=config.nz, periodic_axes=("z",), device=device,
    )
    sponge_faces = ("x+", "y-", "y+")
    if config.sponge_inlet:
        sponge_faces = ("x-",) + sponge_faces
    sigma = build_sponge_sigma_3d(
        shape, width=config.sponge_width,
        max_strength=config.sponge_strength, device=device,
        faces=sponge_faces,
    )
    forces: list[float] = []
    bfl_forces: list[float] = []
    lift_forces: list[float] = []
    start_step = 0
    checkpoint = Path(config.checkpoint_path) if config.checkpoint_path else None
    checkpoint_signature = {
        "schema_version": 4,
        "shape_zyx": list(shape),
        "radius": config.radius,
        "center_x_fraction": config.center_x_fraction,
        "reynolds": config.reynolds,
        "lattice_speed": config.lattice_speed,
        "collision_model": config.collision_model,
        "collision_chunk_cells": config.collision_chunk_cells,
        "compile_natural_kbc": config.compile_natural_kbc,
        "warmup_steps": config.warmup_steps,
        "ramp_steps": config.ramp_steps,
        "sponge_width": config.sponge_width,
        "sponge_strength": config.sponge_strength,
        "sponge_inlet": config.sponge_inlet,
        "cv_margin": config.cv_margin,
        "far_field_mode": config.far_field_mode,
        "periodic_axes": ["z"],
        "link_force_frame": "laboratory_after_wall_activation",
    }
    if config.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        source_configuration = state.get("configuration")
        if isinstance(source_configuration, dict):
            source_configuration = dict(source_configuration)
            source_configuration.setdefault("collision_chunk_cells", 0)
            source_configuration.setdefault("compile_natural_kbc", False)
        if source_configuration != checkpoint_signature:
            raise ValueError("checkpoint configuration does not match cylinder run")
        f = state["populations"].to(device=device)
        start_step = int(state["step"])
        forces = state["drag_force_history"].tolist()
        bfl_forces = state["bfl_drag_history"].tolist()
        lift_forces = state["lift_force_history"].tolist()
        if start_step >= config.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({
            "schema": "tensorlbm-cylinder-checkpoint-v4",
            "configuration": checkpoint_signature,
            "step": step,
            "populations": f.detach().cpu(),
            "drag_force_history": torch.tensor(forces, dtype=torch.float64),
            "bfl_drag_history": torch.tensor(bfl_forces, dtype=torch.float64),
            "lift_force_history": torch.tensor(lift_forces, dtype=torch.float64),
        }, checkpoint)

    def apply_outer(state: torch.Tensor) -> torch.Tensor:
        if config.far_field_mode == "non_equilibrium_extrapolation":
            return non_equilibrium_far_field_bc_3d(
                state, u_in=config.lattice_speed,
                faces=("x-", "x+", "y-", "y+"),
            )
        return far_field_bc_3d(
            state, u_in=config.lattice_speed,
            bc_config={
                "far_field_faces": ["y-", "y+"],
                "periodic_faces": ["z-", "z+"],
            },
        )

    natural_kbc_executor = NaturalKBCCollisionExecutor(
        compile_enabled=config.compile_natural_kbc,
    )
    for step in range(start_step + 1, config.steps + 1):
        old = f
        if config.collision_model == "natural_kbc_d3q19":
            if config.collision_chunk_cells:
                collided = collide_in_z_chunks(
                    f,
                    lambda slab: natural_kbc_executor(slab, config.tau),
                    chunk_cells=config.collision_chunk_cells,
                )
            else:
                collided = natural_kbc_executor(f, config.tau)
        else:
            collided = collide_cumulant_d3q19(f, config.tau, C_s=0.0)
        post = torch.where(solid_q, old, collided)
        f = apply_outer(stream3d(post))
        rho_post, ux_post, uy_post, uz_post = macroscopic3d(post)
        activation = _ramp(step, config.ramp_steps)
        f, bfl_force = bouzidi_bounce_back_d3q19(
            f, post, bfl_mask, bfl_q,
            wall_velocity=(
                (1.0 - activation) * ux_post,
                (1.0 - activation) * uy_post,
                (1.0 - activation) * uz_post,
            ),
            wall_density=rho_post, return_force=True,
        )
        f = apply_equilibrium_difference_sponge(
            f, sigma, velocity_target=(config.lattice_speed, 0.0, 0.0),
        )
        f = apply_outer(f)
        cv_vector = observe_control_volume_force(
            old, f, post, cv, solid=solid, periodic_axes=("z",),
        ).force_on_body
        cv_force = float(cv_vector[0].item())
        if step > config.warmup_steps:
            forces.append(cv_force)
            bfl_forces.append(bfl_force[0])
            lift_forces.append(float(cv_vector[1].item()))
        if not bool(torch.isfinite(f).all()):
            raise FloatingPointError(f"cylinder benchmark diverged at step {step}")
        if config.report_interval and step % config.report_interval == 0:
            recent = forces[-min(len(forces), config.report_interval):]
            recent_cd = (
                sum(recent) / len(recent)
                / (0.5 * config.lattice_speed**2 * (2.0 * config.radius) * config.nz)
                if recent else math.nan
            )
            print(f"cylinder step={step}/{config.steps} recent_Cd={recent_cd:.6f}", flush=True)
        if (
            checkpoint is not None and config.checkpoint_interval
            and step % config.checkpoint_interval == 0
        ):
            save_checkpoint(step)

    if checkpoint is not None:
        save_checkpoint(config.steps)

    statistics_window = config.statistics_window_steps or len(forces)
    selected_forces = forces[-statistics_window:]
    selected_bfl_forces = bfl_forces[-statistics_window:]
    selected_lift_forces = lift_forces[-statistics_window:]
    mean_force = sum(selected_forces) / len(selected_forces)
    mean_bfl = sum(selected_bfl_forces) / len(selected_bfl_forces)
    denominator = (
        0.5 * config.lattice_speed**2 * (2.0 * config.radius) * config.nz
    )
    cd, cd_bfl = mean_force / denominator, mean_bfl / denominator
    cd_history = [force / denominator for force in selected_forces]
    cy_history = [force / denominator for force in selected_lift_forces]
    stationarity = assess_force_stationarity(
        cd_history,
        block_size=max(1, len(cd_history) // 8),
    )
    strouhal, shedding_cycles = estimate_strouhal_from_lift(
        cy_history, lattice_speed=config.lattice_speed,
        diameter=2.0 * config.radius,
    )
    reference_error = abs(cd - CYLINDER_RE100_CD_REFERENCE) / CYLINDER_RE100_CD_REFERENCE * 100.0
    strouhal_error = (
        abs(strouhal - CYLINDER_RE100_ST_REFERENCE)
        / CYLINDER_RE100_ST_REFERENCE * 100.0
        if math.isfinite(strouhal) else math.inf
    )
    observer_difference = abs(cd - cd_bfl) / max(abs(cd), 1e-30) * 100.0
    numerical_quality_admitted = (
        math.isfinite(cd)
        and stationarity.meets(1.0)
        and observer_difference <= 1.0
        and shedding_cycles >= config.minimum_shedding_cycles
    )
    domain_clearance_diameters = config.domain_clearance_diameters
    domain_reference_adequate = config.domain_reference_adequate
    numerical_quality_admitted = (
        numerical_quality_admitted and domain_reference_adequate
    )
    final_rho, final_ux, final_uy, final_uz = macroscopic3d(f)
    final_speed = torch.sqrt(
        final_ux.square() + final_uy.square() + final_uz.square()
    )
    invocation_elapsed_seconds = time.perf_counter() - invocation_started
    steps_advanced = config.steps - start_step
    return {
        "schema": "tensorlbm-cylinder-bfl-control-volume-v4",
        "configuration": checkpoint_signature | {
            "tau": config.tau,
            "steps": config.steps, "warmup_steps": config.warmup_steps,
            "device": config.device,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "report_interval": config.report_interval,
            "checkpoint_interval": config.checkpoint_interval,
            "statistics_window_steps_resolved": statistics_window,
            "statistics_window_steps_requested": (
                config.statistics_window_steps
            ),
            "minimum_shedding_cycles": config.minimum_shedding_cycles,
            "domain_clearance_diameters": domain_clearance_diameters,
        },
        "result": {
            "cd_control_volume": cd, "cd_bfl_link": cd_bfl,
            "observer_difference_pct": observer_difference,
            "cd_reference": CYLINDER_RE100_CD_REFERENCE,
            "reference_error_pct": reference_error,
            "mean_lift_coefficient": sum(cy_history) / len(cy_history),
            "strouhal": strouhal,
            "strouhal_reference": CYLINDER_RE100_ST_REFERENCE,
            "strouhal_reference_error_pct": strouhal_error,
            "shedding_cycles_observed": shedding_cycles,
            "drag_stationarity": stationarity.to_dict(),
            "density_mean": float(final_rho.mean().item()),
            "density_min_max": [
                float(final_rho.min().item()), float(final_rho.max().item()),
            ],
            "relative_mass_drift": float(final_rho.mean().item() - 1.0),
            "maximum_speed": float(final_speed.max().item()),
            "finite": math.isfinite(cd),
            "collision_execution": natural_kbc_executor.diagnostics(),
        },
        "runtime": {
            "invocation_elapsed_seconds": invocation_elapsed_seconds,
            "steps_advanced": steps_advanced,
            "seconds_per_step": invocation_elapsed_seconds / steps_advanced,
        },
        "acceptance": {
            "drag_error_target_pct": 5.0,
            "strouhal_error_target_pct": 5.0,
            "stationarity_target_pct": 1.0,
            "force_observer_target_pct": 1.0,
            "minimum_shedding_cycles": config.minimum_shedding_cycles,
            "drag_target_met": reference_error <= 5.0,
            "strouhal_target_met": strouhal_error <= 5.0,
            "stationarity_target_met": stationarity.meets(1.0),
            "force_observer_target_met": observer_difference <= 1.0,
            "cycle_target_met": (
                shedding_cycles >= config.minimum_shedding_cycles
            ),
            "domain_reference_target_met": domain_reference_adequate,
            "numerical_quality_admitted": numerical_quality_admitted,
            "admitted": (
                reference_error <= 5.0 and strouhal_error <= 5.0
                and numerical_quality_admitted
            ),
        },
        "cuda_memory_preflight": (
            memory_budget.to_dict() if memory_budget is not None else None
        ),
    }


__all__ = [
    "CYLINDER_RE100_CD_REFERENCE",
    "CYLINDER_RE100_ST_REFERENCE",
    "CylinderBFLControlVolumeConfig",
    "estimate_strouhal_from_lift",
    "run_cylinder_bfl_control_volume",
]
