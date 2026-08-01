"""Finite flat-plate external-flow benchmark for the BFL wall-stress model."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from .control_volume_force import box_control_volume, observe_control_volume_force
from .checkpoint_io import atomic_torch_save
from .cumulant import collide_cumulant_d3q19
from .d3q19 import C, equilibrium3d
from .external_open_boundary import non_equilibrium_far_field_bc_3d
from .force_convergence import assess_force_stationarity
from .population_positivity import limit_nonequilibrium_for_positivity
from .solver3d import stream3d
from .sponge_layer import apply_equilibrium_difference_sponge, build_sponge_sigma_3d
from .wall_model import bfl_wall_function_3d


def ittc_1957_friction_coefficient(reynolds: float) -> float:
    if reynolds <= 100.0:
        raise ValueError("ITTC-1957 correlation requires Reynolds number above 100")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


@dataclass(frozen=True)
class FlatPlateWallModelConfig:
    nx: int = 512
    ny: int = 128
    nz: int = 3
    plate_length: int = 256
    plate_start_fraction: float = 0.20
    reynolds: float = 1.0e6
    resolved_reynolds: float = 1.0e5
    lattice_speed: float = 0.06
    steps: int = 12000
    warmup_steps: int = 6000
    ramp_steps: int = 1000
    sponge_width: int = 24
    sponge_strength: float = 0.2
    cv_margin: int = 6
    wall_law: str = "log"
    smagorinsky_cs: float = 0.05
    positivity_limiter: bool = True
    report_interval: int = 1000
    checkpoint_interval: int = 0
    checkpoint_path: str | None = None
    resume: bool = False
    device: str = "cpu"

    @property
    def wall_nu(self) -> float:
        return self.lattice_speed * self.plate_length / self.reynolds

    @property
    def collision_nu(self) -> float:
        return self.lattice_speed * self.plate_length / self.resolved_reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.collision_nu

    def validate(self) -> None:
        if min(self.nx, self.ny) < 16 or self.nz < 1:
            raise ValueError("flat-plate domain is too small")
        if not 8 <= self.plate_length < self.nx - 8:
            raise ValueError("plate_length does not fit the domain")
        if not 0.0 < self.plate_start_fraction < 1.0:
            raise ValueError("plate_start_fraction must lie in (0,1)")
        x0 = int(self.nx * self.plate_start_fraction)
        if x0 <= self.cv_margin + 1 or x0 + self.plate_length + self.cv_margin >= self.nx - 1:
            raise ValueError("plate/control volume does not fit streamwise domain")
        if self.ny // 2 - self.cv_margin <= 1 or self.ny // 2 + self.cv_margin >= self.ny - 2:
            raise ValueError("plate/control volume does not fit transverse domain")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("invalid averaging window")
        if self.wall_law not in {"log", "reichardt", "musker"}:
            raise ValueError("unsupported wall law")
        if not 0.0 <= self.smagorinsky_cs < 0.5:
            raise ValueError("smagorinsky_cs must lie in [0,0.5)")
        if self.report_interval < 0:
            raise ValueError("report_interval must be non-negative")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")


def _halfway_links(solid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_count = 19
    masks = torch.zeros((q_count, *solid.shape), dtype=torch.bool, device=solid.device)
    q_field = torch.full(masks.shape, 0.5, dtype=torch.float32, device=solid.device)
    for direction in range(1, q_count):
        cx, cy, cz = (int(value) for value in C[direction].tolist())
        neighbour = torch.roll(solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
        masks[direction] = ~solid & neighbour
    return masks, q_field


def _ramp(step: int, steps: int) -> float:
    if steps <= 0 or step >= steps:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * step / steps))


def run_flat_plate_wall_model(config: FlatPlateWallModelConfig) -> dict[str, object]:
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shape = (config.nz, config.ny, config.nx)
    x0 = int(config.nx * config.plate_start_fraction)
    x1 = x0 + config.plate_length
    plate_y = config.ny // 2
    solid = torch.zeros(shape, dtype=torch.bool, device=device)
    solid[:, plate_y, x0:x1] = True
    near = torch.zeros_like(solid)
    near[:, plate_y - 1, x0:x1] = True
    near[:, plate_y + 1, x0:x1] = True
    normal_x = torch.zeros(shape, device=device)
    normal_y = torch.zeros(shape, device=device)
    normal_z = torch.zeros(shape, device=device)
    normal_y[:, plate_y - 1, x0:x1] = -1.0
    normal_y[:, plate_y + 1, x0:x1] = 1.0
    bfl_mask, bfl_q = _halfway_links(solid)
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, config.lattice_speed)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, zero, zero, device=device)
    solid_q = solid.unsqueeze(0).expand_as(f)
    cv = box_control_volume(
        shape, x0=x0 - config.cv_margin, x1=x1 + config.cv_margin,
        y0=plate_y - config.cv_margin, y1=plate_y + config.cv_margin + 1,
        z0=0, z1=config.nz, periodic_axes=("z",), device=device,
    )
    sigma = build_sponge_sigma_3d(
        shape, width=config.sponge_width,
        max_strength=config.sponge_strength, device=device,
        faces=("x+", "y-", "y+"),
    )
    friction_history: list[float] = []
    cv_history: list[float] = []
    bfl_total_history: list[float] = []
    maximum_limited_fraction = 0.0
    start_step = 0
    checkpoint = Path(config.checkpoint_path) if config.checkpoint_path else None
    if config.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        expected = {
            "shape_zyx": list(shape), "plate_length": config.plate_length,
            "reynolds": config.reynolds,
            "resolved_reynolds": config.resolved_reynolds,
            "lattice_speed": config.lattice_speed, "wall_law": config.wall_law,
        }
        if state.get("configuration") != expected:
            raise ValueError("checkpoint configuration does not match flat-plate run")
        f = state["populations"].to(device=device)
        start_step = int(state["step"])
        friction_history = state["friction_history"].tolist()
        cv_history = state["control_volume_history"].tolist()
        bfl_total_history = state["bfl_total_history"].tolist()
        maximum_limited_fraction = float(state["maximum_limited_fraction"])
        if start_step >= config.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({
            "schema": "tensorlbm-flat-plate-checkpoint-v1",
            "configuration": {
                "shape_zyx": list(shape), "plate_length": config.plate_length,
                "reynolds": config.reynolds,
                "resolved_reynolds": config.resolved_reynolds,
                "lattice_speed": config.lattice_speed,
                "wall_law": config.wall_law,
            },
            "step": step,
            "populations": f.detach().cpu(),
            "friction_history": torch.tensor(friction_history, dtype=torch.float64),
            "control_volume_history": torch.tensor(cv_history, dtype=torch.float64),
            "bfl_total_history": torch.tensor(bfl_total_history, dtype=torch.float64),
            "maximum_limited_fraction": maximum_limited_fraction,
        }, checkpoint)

    def outer(state: torch.Tensor) -> torch.Tensor:
        return non_equilibrium_far_field_bc_3d(
            state, u_in=config.lattice_speed,
            faces=("x-", "x+", "y-", "y+"),
        )

    for step in range(start_step + 1, config.steps + 1):
        old = f
        collided = collide_cumulant_d3q19(
            f, config.tau, C_s=config.smagorinsky_cs,
        )
        if config.positivity_limiter:
            collided, diagnostic = limit_nonequilibrium_for_positivity(collided)
            maximum_limited_fraction = max(
                maximum_limited_fraction, diagnostic.limited_fraction,
            )
        post = torch.where(solid_q, old, collided)
        f = outer(stream3d(post))
        f, friction, bfl_force = bfl_wall_function_3d(
            f, post, solid, config.wall_nu, bfl_mask, bfl_q,
            near_mask=near, wall_normals=(normal_x, normal_y, normal_z),
            bfl_wall_mode="wall_model_slip",
            wall_activation=_ramp(step, config.ramp_steps),
            wall_law=config.wall_law,
        )
        if config.positivity_limiter:
            f, diagnostic = limit_nonequilibrium_for_positivity(f)
            maximum_limited_fraction = max(
                maximum_limited_fraction, diagnostic.limited_fraction,
            )
        f = apply_equilibrium_difference_sponge(
            f, sigma, velocity_target=(config.lattice_speed, 0.0, 0.0),
        )
        f = outer(f)
        cv_force = float(observe_control_volume_force(
            old, f, post, cv, solid=solid, periodic_axes=("z",),
        ).force_on_body[0].item())
        if step > config.warmup_steps:
            friction_history.append(friction)
            cv_history.append(cv_force)
            bfl_total_history.append(friction + bfl_force)
        if not bool(torch.isfinite(f).all()):
            raise FloatingPointError(f"flat-plate benchmark diverged at step {step}")
        if config.report_interval and step % config.report_interval == 0:
            recent = friction_history[-min(len(friction_history), config.report_interval):]
            recent_cf = (
                sum(recent) / len(recent)
                / (0.5 * config.lattice_speed**2 * 2.0 * config.plate_length * config.nz)
                if recent else math.nan
            )
            print(
                f"flat_plate step={step}/{config.steps} recent_Cf={recent_cf:.7f} "
                f"max_limited={maximum_limited_fraction:.3e}",
                flush=True,
            )
        if (
            checkpoint is not None and config.checkpoint_interval
            and step % config.checkpoint_interval == 0
        ):
            save_checkpoint(step)

    if checkpoint is not None:
        save_checkpoint(config.steps)

    area = 2.0 * config.plate_length * config.nz
    denominator = 0.5 * config.lattice_speed**2 * area
    cf_history = [force / denominator for force in friction_history]
    cf = sum(cf_history) / len(cf_history)
    cv_mean = sum(cv_history) / len(cv_history)
    bfl_mean = sum(bfl_total_history) / len(bfl_total_history)
    cf_reference = ittc_1957_friction_coefficient(config.reynolds)
    stationarity = assess_force_stationarity(
        cf_history, block_size=max(1, len(cf_history) // 8),
    )
    reference_error = abs(cf - cf_reference) / cf_reference * 100.0
    observer_difference = (
        abs(cv_mean - bfl_mean) / max(abs(cv_mean), 1e-30) * 100.0
    )
    limiter_acceptable = maximum_limited_fraction <= 1e-3
    admitted = (
        reference_error <= 5.0
        and stationarity.meets(1.0)
        and observer_difference <= 1.0
        and limiter_acceptable
    )
    return {
        "schema": "tensorlbm-flat-plate-wall-model-v1",
        "configuration": {
            "shape_zyx": list(shape), "plate_length": config.plate_length,
            "reynolds": config.reynolds,
            "resolved_reynolds": config.resolved_reynolds,
            "wall_nu": config.wall_nu, "tau": config.tau,
            "steps": config.steps, "warmup_steps": config.warmup_steps,
            "wall_law": config.wall_law, "device": config.device,
            "smagorinsky_cs": config.smagorinsky_cs,
            "positivity_limiter": config.positivity_limiter,
            "report_interval": config.report_interval,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
        },
        "result": {
            "friction_coefficient": cf,
            "ittc_1957_reference": cf_reference,
            "reference_error_pct": reference_error,
            "control_volume_total_force": cv_mean,
            "bfl_link_plus_wall_stress_force": bfl_mean,
            "total_force_observer_difference_pct": observer_difference,
            "drag_stationarity": stationarity.to_dict(),
            "maximum_positivity_limited_fraction": maximum_limited_fraction,
            "finite": math.isfinite(cf),
        },
        "acceptance": {
            "friction_error_target_pct": 5.0,
            "stationarity_target_pct": 1.0,
            "force_observer_target_pct": 1.0,
            "maximum_limiter_fraction": 1e-3,
            "friction_target_met": reference_error <= 5.0,
            "stationarity_target_met": stationarity.meets(1.0),
            "force_observer_target_met": observer_difference <= 1.0,
            "limiter_target_met": limiter_acceptable,
            "admitted": admitted,
        },
    }


__all__ = [
    "FlatPlateWallModelConfig",
    "ittc_1957_friction_coefficient",
    "run_flat_plate_wall_model",
]
