"""Three-dimensional periodic wall-resolved channel reference."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .boundaries3d import bounce_back_cells_3d
from .chunked_collision import NaturalKBCCollisionExecutor, collide_in_z_chunks
from .cumulant import collide_cumulant_d3q19
from .d3q19 import C, equilibrium3d, macroscopic3d_low_memory
from .solver3d import stream3d
from .spalding_wall_model import spalding_u_plus_from_y_plus
from .wall_model import guo_body_force_d3q19


@dataclass(frozen=True)
class WallResolvedChannel3DConfig:
    nx: int = 128
    ny: int = 64
    nz: int = 64
    re_tau: float = 180.0
    u_tau: float = 0.003
    steps: int = 50000
    warmup_steps: int = 20000
    sample_interval: int = 10
    report_interval: int = 500
    checkpoint_interval: int = 5000
    collision_model: str = "natural_kbc"
    collision_chunk_cells: int = 262144
    compile_natural_kbc: bool = True
    perturbation_fraction: float = 0.05
    device: str = "cuda"
    output: Path = Path("results/canonical_wall/channel3d.json")
    checkpoint: Path = Path("results/canonical_wall/channel3d.ckpt")
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))

    @property
    def height(self) -> int:
        return self.ny - 2

    @property
    def nu(self) -> float:
        return self.u_tau * (0.5 * self.height) / self.re_tau

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def body_force_acceleration(self) -> float:
        return 2.0 * self.u_tau**2 / self.height

    def validate(self) -> None:
        if min(self.nx, self.nz) < 8 or self.ny < 10:
            raise ValueError("channel dimensions are too small")
        if self.re_tau <= 0.0 or self.u_tau <= 0.0:
            raise ValueError("re_tau and u_tau must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("require 0 <= warmup_steps < steps")
        for name in (
            "sample_interval", "report_interval", "checkpoint_interval",
            "collision_chunk_cells",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.collision_model not in {"natural_kbc", "cumulant"}:
            raise ValueError("collision_model must be natural_kbc or cumulant")
        if self.compile_natural_kbc and self.collision_model != "natural_kbc":
            raise ValueError("compiled collision requires natural_kbc")
        if not 0.0 <= self.perturbation_fraction <= 0.2:
            raise ValueError("perturbation_fraction must lie in [0,0.2]")
        if self.tau <= 0.5:
            raise ValueError("derived relaxation time must exceed 0.5")


def _initial_velocity(config: WallResolvedChannel3DConfig, device: torch.device):
    z = torch.arange(config.nz, device=device, dtype=torch.float32)[:, None, None]
    y = torch.arange(config.ny, device=device, dtype=torch.float32)[None, :, None]
    x = torch.arange(config.nx, device=device, dtype=torch.float32)[None, None, :]
    distance = torch.minimum(y - 0.5, config.height + 0.5 - y).clamp_min(0.0)
    y_plus = distance * config.u_tau / config.nu
    base = spalding_u_plus_from_y_plus(y_plus) * config.u_tau
    phase_x = 2.0 * math.pi * x / config.nx
    phase_y = math.pi * distance / (0.5 * config.height)
    phase_z = 2.0 * math.pi * z / config.nz
    amplitude = config.perturbation_fraction * config.u_tau
    ux = base.expand(config.nz, config.ny, config.nx).clone()
    ux += amplitude * torch.sin(phase_x) * torch.sin(phase_y) * torch.sin(phase_z)
    uy = amplitude * torch.cos(phase_x) * torch.sin(phase_y) * torch.sin(phase_z)
    uz = -amplitude * torch.sin(phase_x) * torch.sin(phase_y) * torch.cos(phase_z)
    solid = torch.zeros(
        (config.nz, config.ny, config.nx), dtype=torch.bool, device=device,
    )
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    ux[solid] = uy[solid] = uz[solid] = 0.0
    return solid, ux, uy, uz


def _fluid_momentum_x(populations: torch.Tensor, fluid: torch.Tensor) -> float:
    direction_x = C.to(device=populations.device, dtype=populations.dtype)[:, 0]
    momentum = (populations * direction_x[:, None, None, None]).sum(dim=0)
    return float(momentum[fluid].sum().item())


def _save_checkpoint(
    config: WallResolvedChannel3DConfig,
    *,
    step: int,
    populations: torch.Tensor,
    profile_sum: torch.Tensor,
    profile_samples: int,
    reports: list[dict[str, float | int | bool]],
    block_start_step: int,
    block_start_momentum_x: float,
) -> None:
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tensorlbm-wall-resolved-channel3d-checkpoint-v1",
            "configuration": {
                **asdict(config),
                "output": str(config.output),
                "checkpoint": str(config.checkpoint),
                "resume": False,
            },
            "step": step,
            "populations": populations.detach().to(device="cpu"),
            "profile_sum": profile_sum.detach().to(device="cpu"),
            "profile_samples": profile_samples,
            "reports": reports,
            "block_start_step": block_start_step,
            "block_start_momentum_x": block_start_momentum_x,
        },
        config.checkpoint,
    )


def run_wall_resolved_channel3d(
    config: WallResolvedChannel3DConfig,
) -> dict[str, object]:
    """Run the channel and return a fail-closed validation record."""
    config.validate()
    device = torch.device(config.device)
    solid, initial_ux, initial_uy, initial_uz = _initial_velocity(config, device)
    fluid = ~solid
    solid_q = solid.unsqueeze(0)
    profile_sum = torch.zeros(config.ny, device=device, dtype=torch.float64)
    profile_samples = 0
    reports: list[dict[str, float | int | bool]] = []
    start_step = 0
    if config.resume and config.checkpoint.exists():
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        if state.get("schema") != "tensorlbm-wall-resolved-channel3d-checkpoint-v1":
            raise ValueError("unsupported channel checkpoint")
        stored = state.get("configuration")
        expected = {
            **asdict(config),
            "output": str(config.output),
            "checkpoint": str(config.checkpoint),
            "resume": False,
        }
        if stored != expected:
            raise ValueError("channel checkpoint configuration mismatch")
        f = state["populations"].to(device=device)
        profile_sum = state["profile_sum"].to(device=device)
        profile_samples = int(state["profile_samples"])
        reports = list(state["reports"])
        start_step = int(state["step"])
        block_start_step = int(state["block_start_step"])
        block_start_momentum = float(state["block_start_momentum_x"])
    else:
        f = equilibrium3d(
            torch.ones_like(initial_ux), initial_ux, initial_uy, initial_uz,
        )
        block_start_step = 0
        block_start_momentum = _fluid_momentum_x(f, fluid)
    initial_mass = float(f[:, fluid].sum().item())
    fluid_cells = int(fluid.sum().item())
    wall_area = 2 * config.nx * config.nz
    collision = NaturalKBCCollisionExecutor(
        compile_enabled=config.compile_natural_kbc,
    )

    for step in range(start_step + 1, config.steps + 1):
        old = f
        if config.collision_model == "natural_kbc":
            collided = collide_in_z_chunks(
                f,
                lambda slab: collision(slab, config.tau),
                chunk_cells=config.collision_chunk_cells,
            )
        else:
            collided = collide_cumulant_d3q19(f, config.tau, C_s=0.0)
        post = torch.where(solid_q, old, collided)
        rho, ux, uy, uz = macroscopic3d_low_memory(post)
        force_x = rho * config.body_force_acceleration * fluid
        post = guo_body_force_d3q19(
            post,
            force_x,
            torch.zeros_like(force_x),
            torch.zeros_like(force_x),
            ux,
            uy,
            uz,
            direction_chunk_size=4,
        )
        streamed = stream3d(post)
        f = bounce_back_cells_3d(streamed, solid, f_pre=post)

        if step > config.warmup_steps and step % config.sample_interval == 0:
            _, sample_ux, _, _ = macroscopic3d_low_memory(f)
            profile_sum += sample_ux.mean(dim=(0, 2)).to(dtype=torch.float64)
            profile_samples += 1
        if step % config.report_interval == 0 or step == config.steps:
            rho_now, ux_now, uy_now, uz_now = macroscopic3d_low_memory(f)
            speed = torch.sqrt(ux_now.square() + uy_now.square() + uz_now.square())
            momentum = _fluid_momentum_x(f, fluid)
            block_steps = step - block_start_step
            body_impulse_per_step = config.body_force_acceleration * fluid_cells
            wall_force_per_step = (
                body_impulse_per_step
                - (momentum - block_start_momentum) / block_steps
            )
            measured_tau_w = wall_force_per_step / wall_area
            measured_u_tau = math.sqrt(max(measured_tau_w, 0.0))
            report = {
                "step": step,
                "finite": bool(torch.isfinite(f).all().item()),
                "minimum_population": float(f.min().item()),
                "maximum_speed": float(speed[fluid].max().item()),
                "mass_drift_fraction": (
                    float(f[:, fluid].sum().item()) - initial_mass
                ) / initial_mass,
                "momentum_x": momentum,
                "measured_wall_shear": measured_tau_w,
                "measured_friction_velocity": measured_u_tau,
                "friction_velocity_error_pct": (
                    (measured_u_tau - config.u_tau) / config.u_tau * 100.0
                ),
                "collision_limited_fraction": 0.0,
            }
            reports.append(report)
            print("channel3d " + json.dumps(report, separators=(",", ":")), flush=True)
            if not report["finite"] or report["minimum_population"] <= 0.0:
                raise FloatingPointError("channel population health failed")
            block_start_step = step
            block_start_momentum = momentum
        if step % config.checkpoint_interval == 0 or step == config.steps:
            _save_checkpoint(
                config,
                step=step,
                populations=f,
                profile_sum=profile_sum,
                profile_samples=profile_samples,
                reports=reports,
                block_start_step=block_start_step,
                block_start_momentum_x=block_start_momentum,
            )

    if not profile_samples:
        raise RuntimeError("channel collected no profile samples")
    mean_profile = profile_sum / profile_samples
    recent = reports[-3:]
    recent_u_tau = [float(item["measured_friction_velocity"]) for item in recent]
    recent_mean = sum(recent_u_tau) / len(recent_u_tau)
    recent_range_fraction = (
        (max(recent_u_tau) - min(recent_u_tau))
        / max(abs(recent_mean), 1.0e-30)
    )
    mean_error_pct = (recent_mean - config.u_tau) / config.u_tau * 100.0
    result: dict[str, object] = {
        "schema": "tensorlbm-wall-resolved-channel3d-result-v1",
        "configuration": {
            **asdict(config),
            "output": str(config.output),
            "checkpoint": str(config.checkpoint),
        },
        "derived": {
            "height": config.height,
            "nu": config.nu,
            "tau": config.tau,
            "body_force_acceleration": config.body_force_acceleration,
            "target_wall_shear": config.u_tau**2,
            "dx_plus": config.re_tau / (0.5 * config.height),
        },
        "statistics": {
            "profile_samples": profile_samples,
            "mean_velocity_profile": mean_profile.tolist(),
            "recent_friction_velocity_mean": recent_mean,
            "recent_friction_velocity_range_fraction": recent_range_fraction,
            "recent_friction_velocity_error_pct": mean_error_pct,
        },
        "reports": reports,
        "collision_execution": collision.diagnostics(),
        "acceptance": {
            "population_health": all(bool(item["finite"]) for item in reports),
            "positive_populations": min(
                float(item["minimum_population"]) for item in reports
            ) > 0.0,
            "friction_velocity_error_below_2pct": abs(mean_error_pct) <= 2.0,
            "recent_range_below_1pct": recent_range_fraction <= 0.01,
        },
    }
    result["physical_validation"] = all(result["acceptance"].values())
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["WallResolvedChannel3DConfig", "run_wall_resolved_channel3d"]
