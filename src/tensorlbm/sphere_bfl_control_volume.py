"""Canonical sphere drag with BFL and an independent control-volume force."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .bfl_d3q19 import bouzidi_bounce_back_d3q19
from .boundaries3d import far_field_bc_3d, sphere_mask
from .checkpoint_io import atomic_torch_save
from .chunked_collision import (
    NaturalKBCCollisionExecutor,
    collide_in_z_chunks,
)
from .control_volume_force import box_control_volume, observe_control_volume_force
from .cuda_memory_budget import require_cuda_memory_budget
from .cumulant import collide_cumulant_d3q19
from .d3q19 import equilibrium3d, macroscopic3d
from .drag_pressure import integrate_bfl_projected_pressure
from .external_open_boundary import non_equilibrium_far_field_bc_3d
from .force_convergence import assess_force_stationarity
from .interpolated_bc import compute_q_sphere
from .open_boundary_audit import audit_open_boundary_history
from .solver3d import stream3d
from .sponge_layer import apply_equilibrium_difference_sponge, build_sponge_sigma_3d


def schiller_naumann_cd(reynolds: float) -> float:
    if reynolds <= 0.0:
        raise ValueError("reynolds must be positive")
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687)


@dataclass(frozen=True)
class SphereBFLControlVolumeConfig:
    nx: int = 192
    ny: int = 96
    nz: int = 96
    radius: float = 12.0
    center_x_fraction: float = 0.30
    reynolds: float = 100.0
    lattice_speed: float = 0.06
    steps: int = 5000
    warmup_steps: int = 2500
    ramp_steps: int = 500
    sponge_width: int = 18
    sponge_strength: float = 0.2
    sponge_inlet: bool = False
    cv_margin: int = 8
    far_field_mode: str = "non_equilibrium_extrapolation"
    report_interval: int = 500
    checkpoint_interval: int = 0
    checkpoint_path: str | None = None
    resume: bool = False
    allow_v2_checkpoint: bool = False
    statistics_window_steps: int = 0
    minimum_statistics_convective_times: float = 5.0
    collision_model: str = "cumulant_d3q19_cs0"
    collision_chunk_cells: int = 0
    compile_natural_kbc: bool = False
    projected_pressure_interval: int = 0
    projected_pressure_reconstruction: str = "linear"
    device: str = "cpu"

    @property
    def nu(self) -> float:
        return self.lattice_speed * 2.0 * self.radius / self.reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    def validate(self) -> None:
        if min(self.nx, self.ny, self.nz) < 16:
            raise ValueError("sphere domain is too small")
        if self.radius < 3.0 or self.steps < 1:
            raise ValueError("radius must be >=3 and steps positive")
        if not 0.0 < self.center_x_fraction < 1.0:
            raise ValueError("center_x_fraction must lie in (0,1)")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0,steps)")
        cx = self.nx * self.center_x_fraction
        if min(cx, self.nx - cx, self.ny / 2, self.nz / 2) <= self.radius + self.cv_margin + 2:
            raise ValueError("sphere/control volume does not fit the domain")
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
        if self.projected_pressure_interval < 0:
            raise ValueError("projected_pressure_interval must be non-negative")
        if self.projected_pressure_reconstruction not in {
            "local", "linear", "quadratic",
        }:
            raise ValueError("unknown projected_pressure_reconstruction")
        if not 0 <= self.statistics_window_steps <= self.steps - self.warmup_steps:
            raise ValueError(
                "statistics_window_steps must be zero or fit after warmup",
            )
        if self.minimum_statistics_convective_times < 0.0:
            raise ValueError("minimum_statistics_convective_times must be non-negative")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_sphere_bfl_control_volume(
    config: SphereBFLControlVolumeConfig,
) -> dict[str, object]:
    """Run the canonical benchmark and return a machine-readable result."""
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
        reserve_gib=1.0, label="sphere benchmark",
    )
    cx, cy, cz = (
        config.nx * config.center_x_fraction,
        config.ny / 2.0,
        config.nz / 2.0,
    )
    solid = sphere_mask(
        config.nx, config.ny, config.nz, cx, cy, cz, config.radius,
        device=device,
    )
    bfl_mask, bfl_q = compute_q_sphere(
        config.nx, config.ny, config.nz, cx, cy, cz, config.radius,
        device=device,
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
        z0=int(math.floor(cz - config.radius)) - config.cv_margin,
        z1=int(math.ceil(cz + config.radius)) + config.cv_margin + 1,
        device=device,
    )
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    if config.sponge_inlet:
        sponge_faces = ("x-",) + sponge_faces
    sigma = build_sponge_sigma_3d(
        shape, width=config.sponge_width,
        max_strength=config.sponge_strength, device=device,
        faces=sponge_faces,
    )
    forces: list[float] = []
    bfl_forces: list[float] = []
    projected_pressure_samples: list[dict[str, object]] = []
    open_boundary_history: list[dict[str, object]] = []
    start_step = 0
    checkpoint = Path(config.checkpoint_path) if config.checkpoint_path else None
    checkpoint_signature = {
        "schema_version": 3,
        "bfl_link_fraction_convention": "ray_parameter_q_equals_t_v2",
        "bfl_population_reconstruction": (
            "post_collision_outgoing_and_upstream_v2"
        ),
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
        "statistics_window_steps": config.statistics_window_steps,
    }
    migration_provenance: dict[str, object] | None = None
    if config.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        source_configuration = state.get("configuration")
        if isinstance(source_configuration, dict):
            source_configuration = dict(source_configuration)
            source_configuration.setdefault("collision_chunk_cells", 0)
            source_configuration.setdefault("compile_natural_kbc", False)
        if source_configuration != checkpoint_signature:
            shared_target = {
                key: value for key, value in checkpoint_signature.items()
                if key not in {"schema_version", "statistics_window_steps"}
            }
            v2_compatible = (
                config.allow_v2_checkpoint
                and state.get("schema") == "tensorlbm-sphere-checkpoint-v2"
                and isinstance(source_configuration, dict)
                and source_configuration.get("schema_version") == 2
                and all(
                    source_configuration.get(key) == value
                    for key, value in shared_target.items()
                )
            )
            if not v2_compatible:
                raise ValueError("checkpoint configuration does not match sphere run")
            migration_provenance = {
                "source_schema": state["schema"],
                "source_step": int(state["step"]),
                "source_checkpoint_sha256": _sha256_file(checkpoint),
                "statistics_policy": "retain_full_history_use_explicit_tail_window",
            }
        else:
            migration_provenance = state.get("migration_provenance")
        f = state["populations"].to(device=device)
        start_step = int(state["step"])
        forces = state["drag_force_history"].tolist()
        bfl_forces = state["bfl_drag_history"].tolist()
        open_boundary_history = list(state.get("open_boundary_history", []))
        projected_pressure_samples = list(
            state.get("projected_pressure_samples", []),
        )
        if (
            config.projected_pressure_interval > 0
            and start_step > config.warmup_steps
            and "projected_pressure_samples" not in state
        ):
            raise ValueError(
                "checkpoint lacks requested projected-pressure history",
            )
        if start_step >= config.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({
            "schema": "tensorlbm-sphere-checkpoint-v3",
            "configuration": checkpoint_signature,
            "step": step,
            "populations": f.detach().cpu(),
            "drag_force_history": torch.tensor(forces, dtype=torch.float64),
            "bfl_drag_history": torch.tensor(bfl_forces, dtype=torch.float64),
            "open_boundary_history": open_boundary_history,
            "projected_pressure_samples": projected_pressure_samples,
            "migration_provenance": migration_provenance,
        }, checkpoint)

    def apply_outer(
        state: torch.Tensor,
        *,
        collect_diagnostics: bool,
        stage: str,
        records: list[dict[str, object]],
    ) -> torch.Tensor:
        if config.far_field_mode == "non_equilibrium_extrapolation":
            result = non_equilibrium_far_field_bc_3d(
                state,
                u_in=config.lattice_speed,
                return_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics:
                state, diagnostics = result
                records.append({"stage": stage, **asdict(diagnostics)})
                return state
            return result
        return far_field_bc_3d(state, u_in=config.lattice_speed)

    dynamic_area = 0.5 * config.lattice_speed**2 * math.pi * config.radius**2
    natural_kbc_executor = NaturalKBCCollisionExecutor(
        compile_enabled=config.compile_natural_kbc,
    )
    for step in range(start_step + 1, config.steps + 1):
        collect_boundary_diagnostics = (
            bool(config.report_interval)
            and step % config.report_interval == 0
            and config.far_field_mode == "non_equilibrium_extrapolation"
        )
        step_boundary_records: list[dict[str, object]] = []
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
        f = stream3d(post)
        f = apply_outer(
            f,
            collect_diagnostics=collect_boundary_diagnostics,
            stage="post_stream_pre_sponge",
            records=step_boundary_records,
        )
        rho_post, ux_post, uy_post, uz_post = macroscopic3d(post)
        activation = _ramp(step, config.ramp_steps)
        wall_velocity = (
            (1.0 - activation) * ux_post,
            (1.0 - activation) * uy_post,
            (1.0 - activation) * uz_post,
        )
        f, bfl_force = bouzidi_bounce_back_d3q19(
            f, post, bfl_mask, bfl_q,
            wall_velocity=wall_velocity, wall_density=rho_post,
            return_force=True,
        )
        f = apply_equilibrium_difference_sponge(
            f, sigma, velocity_target=(config.lattice_speed, 0.0, 0.0),
        )
        f = apply_outer(
            f,
            collect_diagnostics=collect_boundary_diagnostics,
            stage="post_sponge",
            records=step_boundary_records,
        )
        if collect_boundary_diagnostics:
            open_boundary_history.append({
                "step": step,
                "stages": step_boundary_records,
                "mass_delta": sum(
                    float(record["mass_delta"])
                    for record in step_boundary_records
                ),
                "momentum_delta": [
                    sum(
                        float(record["momentum_delta"][axis])
                        for record in step_boundary_records
                    )
                    for axis in range(3)
                ],
                "finite": all(
                    bool(record["finite"])
                    for record in step_boundary_records
                ),
            })
        cv_force = float(observe_control_volume_force(
            old, f, post, cv, solid=solid,
        ).force_on_body[0].item())
        if (
            config.projected_pressure_interval > 0
            and step > config.warmup_steps
            and step % config.projected_pressure_interval == 0
        ):
            pressure = (f.sum(dim=0) - 1.0) / 3.0
            projected_force, projected_diagnostics = (
                integrate_bfl_projected_pressure(
                    pressure,
                    bfl_mask,
                    bfl_q,
                    solid=solid,
                    reconstruction=config.projected_pressure_reconstruction,
                )
            )
            projected_pressure_samples.append({
                "step": step,
                "pressure_force_x": projected_force[0],
                "paired_control_volume_force_x": cv_force,
                "diagnostics": asdict(projected_diagnostics),
            })
        if step > config.warmup_steps:
            forces.append(cv_force)
            bfl_forces.append(bfl_force[0])
        if not bool(torch.isfinite(f).all()):
            raise FloatingPointError(f"sphere benchmark diverged at step {step}")
        if config.report_interval and step % config.report_interval == 0:
            recent = forces[-min(len(forces), config.report_interval):]
            recent_cd = sum(recent) / len(recent) / dynamic_area if recent else math.nan
            print(f"sphere step={step}/{config.steps} recent_Cd={recent_cd:.6f}", flush=True)
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
    mean_force = sum(selected_forces) / len(selected_forces)
    mean_bfl_force = sum(selected_bfl_forces) / len(selected_bfl_forces)
    cd = mean_force / dynamic_area
    cd_bfl = mean_bfl_force / dynamic_area
    reference = schiller_naumann_cd(config.reynolds)
    cd_history = [force / dynamic_area for force in selected_forces]
    stationarity = assess_force_stationarity(
        cd_history, block_size=max(1, len(cd_history) // 8),
    )
    observer_difference = abs(cd - cd_bfl) / max(abs(cd), 1e-30) * 100.0
    reference_error = abs(cd - reference) / reference * 100.0
    statistics_convective_times = (
        len(selected_forces) * config.lattice_speed / (2.0 * config.radius)
    )
    duration_acceptable = (
        statistics_convective_times
        >= config.minimum_statistics_convective_times
    )
    numerical_quality_admitted = (
        math.isfinite(cd)
        and stationarity.meets(1.0)
        and observer_difference <= 1.0
        and duration_acceptable
    )
    invocation_elapsed_seconds = time.perf_counter() - invocation_started
    steps_advanced = config.steps - start_step
    open_boundary_audit = audit_open_boundary_history(
        open_boundary_history,
        reference_mass=float(math.prod(shape)),
        reference_momentum=float(math.prod(shape)) * config.lattice_speed,
    )
    statistics_start_step = config.steps - statistics_window + 1
    selected_projected_samples = [
        sample for sample in projected_pressure_samples
        if int(sample["step"]) >= statistics_start_step
    ]
    projected_pressure_observer: dict[str, object] = {
        "scope": "candidate_diagnostic_only_not_an_acceptance_gate",
        "enabled": config.projected_pressure_interval > 0,
        "used_for_acceptance": False,
        "sample_interval_steps": config.projected_pressure_interval,
        "reconstruction": config.projected_pressure_reconstruction,
        "samples": len(selected_projected_samples),
        "mean_pressure_force": None,
        "paired_control_volume_mean_force": None,
        "mean_force_difference_pct": None,
        "minimum_usable_link_fraction": None,
        "maximum_fallback_cells": None,
    }
    if selected_projected_samples:
        projected_mean = sum(
            float(sample["pressure_force_x"])
            for sample in selected_projected_samples
        ) / len(selected_projected_samples)
        projected_paired_cv_mean = sum(
            float(sample["paired_control_volume_force_x"])
            for sample in selected_projected_samples
        ) / len(selected_projected_samples)
        usable_fractions = [
            float(sample["diagnostics"]["usable_links"])
            / max(float(sample["diagnostics"]["requested_links"]), 1.0)
            for sample in selected_projected_samples
        ]
        projected_pressure_observer.update({
            "mean_pressure_force": projected_mean,
            "paired_control_volume_mean_force": projected_paired_cv_mean,
            "mean_force_difference_pct": (
                abs(projected_mean - projected_paired_cv_mean)
                / max(abs(projected_paired_cv_mean), 1.0e-30)
                * 100.0
            ),
            "minimum_usable_link_fraction": min(usable_fractions),
            "maximum_fallback_cells": max(
                int(sample["diagnostics"]["fallback_cells"])
                for sample in selected_projected_samples
            ),
        })
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v3",
        "configuration": checkpoint_signature | {
            "tau": config.tau,
            "steps": config.steps, "warmup_steps": config.warmup_steps,
            "device": config.device,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "report_interval": config.report_interval,
            "checkpoint_interval": config.checkpoint_interval,
            "statistics_window_steps_resolved": statistics_window,
            "projected_pressure_interval": config.projected_pressure_interval,
            "projected_pressure_reconstruction": (
                config.projected_pressure_reconstruction
            ),
            "statistics_convective_times": statistics_convective_times,
            "minimum_statistics_convective_times": (
                config.minimum_statistics_convective_times
            ),
            "migration_provenance": migration_provenance,
        },
        "result": {
            "cd_control_volume": cd,
            "cd_bfl_link": cd_bfl,
            "observer_difference_pct": observer_difference,
            "cd_reference_schiller_naumann": reference,
            "reference_error_pct": reference_error,
            "drag_stationarity": stationarity.to_dict(),
            "finite": math.isfinite(cd),
            "collision_execution": natural_kbc_executor.diagnostics(),
            "open_boundary_population_delta": open_boundary_history,
            "open_boundary_population_delta_audit": (
                open_boundary_audit.to_dict()
            ),
            "projected_bfl_pressure_observer": projected_pressure_observer,
        },
        "runtime": {
            "invocation_elapsed_seconds": invocation_elapsed_seconds,
            "steps_advanced": steps_advanced,
            "seconds_per_step": invocation_elapsed_seconds / steps_advanced,
        },
        "acceptance": {
            "drag_error_target_pct": 5.0,
            "stationarity_target_pct": 1.0,
            "force_observer_target_pct": 1.0,
            "drag_target_met": reference_error <= 5.0,
            "stationarity_target_met": stationarity.meets(1.0),
            "force_observer_target_met": observer_difference <= 1.0,
            "duration_target_met": duration_acceptable,
            "numerical_quality_admitted": numerical_quality_admitted,
            "admitted": (
                reference_error <= 5.0 and numerical_quality_admitted
            ),
        },
        "measured_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else None
        ),
        "cuda_memory_preflight": (
            memory_budget.to_dict() if memory_budget is not None else None
        ),
    }


__all__ = [
    "SphereBFLControlVolumeConfig",
    "run_sphere_bfl_control_volume",
    "schiller_naumann_cd",
]
