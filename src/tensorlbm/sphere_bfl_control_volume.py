"""Canonical sphere drag with BFL and an independent control-volume force."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .bfl_d3q19 import bouzidi_bounce_back_d3q19
from .boundaries3d import far_field_bc_3d, sphere_mask
from .control_volume_force import box_control_volume, observe_control_volume_force
from .cumulant import collide_cumulant_d3q19
from .d3q19 import equilibrium3d, macroscopic3d
from .external_open_boundary import non_equilibrium_far_field_bc_3d
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
    cv_margin: int = 8
    far_field_mode: str = "non_equilibrium_extrapolation"
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
    sigma = build_sponge_sigma_3d(
        shape, width=config.sponge_width,
        max_strength=config.sponge_strength, device=device,
        faces=("x+", "y-", "y+", "z-", "z+"),
    )
    forces: list[float] = []
    bfl_forces: list[float] = []

    def apply_outer(state: torch.Tensor) -> torch.Tensor:
        if config.far_field_mode == "non_equilibrium_extrapolation":
            return non_equilibrium_far_field_bc_3d(
                state, u_in=config.lattice_speed,
            )
        return far_field_bc_3d(state, u_in=config.lattice_speed)

    for step in range(1, config.steps + 1):
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

    mean_force = sum(forces) / len(forces)
    mean_bfl_force = sum(bfl_forces) / len(bfl_forces)
    dynamic_area = 0.5 * config.lattice_speed**2 * math.pi * config.radius**2
    cd = mean_force / dynamic_area
    cd_bfl = mean_bfl_force / dynamic_area
    reference = schiller_naumann_cd(config.reynolds)
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v1",
        "configuration": {
            "shape_zyx": list(shape), "radius": config.radius,
            "reynolds": config.reynolds, "tau": config.tau,
            "steps": config.steps, "warmup_steps": config.warmup_steps,
            "device": config.device,
            "far_field_mode": config.far_field_mode,
        },
        "result": {
            "cd_control_volume": cd,
            "cd_bfl_link": cd_bfl,
            "observer_difference_pct": abs(cd - cd_bfl) / abs(cd) * 100.0,
            "cd_reference_schiller_naumann": reference,
            "reference_error_pct": abs(cd - reference) / reference * 100.0,
            "finite": math.isfinite(cd),
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
