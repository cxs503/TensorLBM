"""Canonical sphere drag with BFL and an independent control-volume force."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from .bfl_d3q19 import bouzidi_bounce_back_d3q19
from .boundaries3d import far_field_bc_3d, sphere_mask
from .control_volume_force import box_control_volume, observe_control_volume_force
from .cumulant import collide_cumulant_d3q19
from .d3q19 import equilibrium3d, macroscopic3d
from .external_open_boundary import non_equilibrium_far_field_bc_3d
from .force_convergence import assess_force_stationarity
from .interpolated_bc import compute_q_sphere
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
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0,steps)")
        cx = self.nx * self.center_x_fraction
        if min(cx, self.nx - cx, self.ny / 2, self.nz / 2) <= self.radius + self.cv_margin + 2:
            raise ValueError("sphere/control volume does not fit the domain")
        if self.far_field_mode not in {
            "non_equilibrium_extrapolation", "legacy_hard_equilibrium",
        }:
            raise ValueError("unknown far_field_mode")
        if self.report_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError("report/checkpoint intervals must be non-negative")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")


def _ramp(step: int, steps: int) -> float:
    if steps <= 0 or step >= steps:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * step / steps))


def run_sphere_bfl_control_volume(
    config: SphereBFLControlVolumeConfig,
) -> dict[str, object]:
    """Run the canonical benchmark and return a machine-readable result."""
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    shape = (config.nz, config.ny, config.nx)
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
    start_step = 0
    checkpoint = Path(config.checkpoint_path) if config.checkpoint_path else None
    if config.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        expected = {
            "shape_zyx": list(shape), "radius": config.radius,
            "reynolds": config.reynolds,
            "lattice_speed": config.lattice_speed,
            "sponge_inlet": config.sponge_inlet,
        }
        if state.get("configuration") != expected:
            raise ValueError("checkpoint configuration does not match sphere run")
        f = state["populations"].to(device=device)
        start_step = int(state["step"])
        forces = state["drag_force_history"].tolist()
        bfl_forces = state["bfl_drag_history"].tolist()
        if start_step >= config.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "tensorlbm-sphere-checkpoint-v1",
            "configuration": {
                "shape_zyx": list(shape), "radius": config.radius,
                "reynolds": config.reynolds,
                "lattice_speed": config.lattice_speed,
                "sponge_inlet": config.sponge_inlet,
            },
            "step": step,
            "populations": f.detach().cpu(),
            "drag_force_history": torch.tensor(forces, dtype=torch.float64),
            "bfl_drag_history": torch.tensor(bfl_forces, dtype=torch.float64),
        }, checkpoint)

    def apply_outer(state: torch.Tensor) -> torch.Tensor:
        if config.far_field_mode == "non_equilibrium_extrapolation":
            return non_equilibrium_far_field_bc_3d(
                state, u_in=config.lattice_speed,
            )
        return far_field_bc_3d(state, u_in=config.lattice_speed)

    dynamic_area = 0.5 * config.lattice_speed**2 * math.pi * config.radius**2
    for step in range(start_step + 1, config.steps + 1):
        old = f
        collided = collide_cumulant_d3q19(f, config.tau, C_s=0.0)
        post = torch.where(solid_q, old, collided)
        f = stream3d(post)
        f = apply_outer(f)
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
        f = apply_outer(f)
        cv_force = float(observe_control_volume_force(
            old, f, post, cv, solid=solid,
        ).force_on_body[0].item())
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

    mean_force = sum(forces) / len(forces)
    mean_bfl_force = sum(bfl_forces) / len(bfl_forces)
    cd = mean_force / dynamic_area
    cd_bfl = mean_bfl_force / dynamic_area
    reference = schiller_naumann_cd(config.reynolds)
    cd_history = [force / dynamic_area for force in forces]
    stationarity = assess_force_stationarity(
        cd_history, block_size=max(1, len(cd_history) // 8),
    )
    observer_difference = abs(cd - cd_bfl) / max(abs(cd), 1e-30) * 100.0
    reference_error = abs(cd - reference) / reference * 100.0
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v1",
        "configuration": {
            "shape_zyx": list(shape), "radius": config.radius,
            "reynolds": config.reynolds, "tau": config.tau,
            "steps": config.steps, "warmup_steps": config.warmup_steps,
            "device": config.device,
            "far_field_mode": config.far_field_mode,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "sponge_inlet": config.sponge_inlet,
        },
        "result": {
            "cd_control_volume": cd,
            "cd_bfl_link": cd_bfl,
            "observer_difference_pct": observer_difference,
            "cd_reference_schiller_naumann": reference,
            "reference_error_pct": reference_error,
            "drag_stationarity": stationarity.to_dict(),
            "finite": math.isfinite(cd),
        },
        "acceptance": {
            "drag_error_target_pct": 5.0,
            "stationarity_target_pct": 1.0,
            "force_observer_target_pct": 1.0,
            "drag_target_met": reference_error <= 5.0,
            "stationarity_target_met": stationarity.meets(1.0),
            "force_observer_target_met": observer_difference <= 1.0,
            "admitted": (
                reference_error <= 5.0 and stationarity.meets(1.0)
                and observer_difference <= 1.0
            ),
        },
        "measured_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else None
        ),
    }


__all__ = [
    "SphereBFLControlVolumeConfig",
    "run_sphere_bfl_control_volume",
    "schiller_naumann_cd",
]
