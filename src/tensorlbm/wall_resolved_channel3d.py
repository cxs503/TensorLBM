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
    perturbation_fraction: float = 1.0
    random_noise_fraction: float = 0.5
    seed: int = 20260802
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
        if not 0.0 <= self.perturbation_fraction <= 2.0:
            raise ValueError("perturbation_fraction must lie in [0,2]")
        if not 0.0 <= self.random_noise_fraction <= 2.0:
            raise ValueError("random_noise_fraction must lie in [0,2]")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
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
    wall_coordinate = (y - 0.5).clamp(0.0, float(config.height))
    phase_y = math.pi * wall_coordinate / config.height
    phase_z = 2.0 * math.pi * z / config.nz
    amplitude = config.perturbation_fraction * config.u_tau
    ux = base.expand(config.nz, config.ny, config.nx).clone()
    # A streamwise-independent streak plus a cross-plane streamfunction.
    # The latter is discretely smooth, vanishes at both walls and is
    # divergence-free in the continuous y-z plane.
    ux += amplitude * torch.sin(phase_y) * torch.cos(phase_z)
    uy = (
        amplitude * torch.sin(phase_x) * torch.sin(phase_y).square()
        * torch.cos(phase_z)
    )
    uz = (
        -amplitude * (config.nz / config.height) * torch.sin(phase_x)
        * torch.sin(2.0 * phase_y) * torch.sin(phase_z)
    )
    if config.random_noise_fraction:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)
        noise = torch.randn(
            (3, config.nz, config.ny, config.nx),
            device=device,
            dtype=ux.dtype,
            generator=generator,
        )
        noise -= noise.mean(dim=(1, 3), keepdim=True)
        taper = torch.sin(phase_y).expand(config.nz, config.ny, config.nx)
        noise_amplitude = amplitude * config.random_noise_fraction
        ux += noise_amplitude * taper * noise[0]
        uy += noise_amplitude * taper * noise[1]
        uz += noise_amplitude * taper * noise[2]
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
    moment_profile_sum: torch.Tensor,
    profile_samples: int,
    statistics_reset_step: int,
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
            "moment_profile_sum": moment_profile_sum.detach().to(device="cpu"),
            "profile_samples": profile_samples,
            "statistics_reset_step": statistics_reset_step,
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
    # Rows: U,V,W, uu_raw,vv_raw,ww_raw,uv_raw; central moments are formed
    # only after the full statistical mean is available.
    moment_profile_sum = torch.zeros(
        (7, config.ny), device=device, dtype=torch.float64,
    )
    profile_samples = 0
    statistics_reset_step = 0
    reports: list[dict[str, float | int | bool]] = []
    start_step = 0
    if config.resume and config.checkpoint.exists():
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        if state.get("schema") != "tensorlbm-wall-resolved-channel3d-checkpoint-v1":
            raise ValueError("unsupported channel checkpoint")
        stored = dict(state.get("configuration", {}))
        stored.setdefault("random_noise_fraction", 0.0)
        stored.setdefault("seed", 20260802)
        expected = {
            **asdict(config),
            "output": str(config.output),
            "checkpoint": str(config.checkpoint),
            "resume": False,
        }
        if stored != expected:
            raise ValueError("channel checkpoint configuration mismatch")
        f = state["populations"].to(device=device)
        if "moment_profile_sum" in state:
            moment_profile_sum = state["moment_profile_sum"].to(device=device)
            profile_samples = int(state["profile_samples"])
            statistics_reset_step = int(state.get("statistics_reset_step", 0))
        else:
            # v1 checkpoints before Reynolds-stress accumulation contain only
            # U sums.  Mixing those with a shorter second-moment window would
            # invent central moments, so all statistics restart explicitly.
            profile_samples = 0
            statistics_reset_step = int(state["step"])
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
            _, sample_ux, sample_uy, sample_uz = macroscopic3d_low_memory(f)
            moment_profile_sum += torch.stack((
                sample_ux.mean(dim=(0, 2)),
                sample_uy.mean(dim=(0, 2)),
                sample_uz.mean(dim=(0, 2)),
                sample_ux.square().mean(dim=(0, 2)),
                sample_uy.square().mean(dim=(0, 2)),
                sample_uz.square().mean(dim=(0, 2)),
                (sample_ux * sample_uy).mean(dim=(0, 2)),
            )).to(dtype=torch.float64)
            profile_samples += 1
        if step % config.report_interval == 0 or step == config.steps:
            rho_now, ux_now, uy_now, uz_now = macroscopic3d_low_memory(f)
            speed = torch.sqrt(ux_now.square() + uy_now.square() + uz_now.square())
            plane_mean_ux = ux_now.mean(dim=(0, 2), keepdim=True)
            plane_mean_uy = uy_now.mean(dim=(0, 2), keepdim=True)
            plane_mean_uz = uz_now.mean(dim=(0, 2), keepdim=True)
            fluctuation_x = ux_now - plane_mean_ux
            fluctuation_y = uy_now - plane_mean_uy
            fluctuation_z = uz_now - plane_mean_uz
            crossflow_rms = torch.sqrt(
                (fluctuation_y[fluid].square() + fluctuation_z[fluid].square()).mean(),
            )
            turbulent_kinetic_energy = 0.5 * (
                fluctuation_x[fluid].square()
                + fluctuation_y[fluid].square()
                + fluctuation_z[fluid].square()
            ).mean()
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
                "crossflow_rms": float(crossflow_rms.item()),
                "crossflow_rms_over_u_tau": float(
                    crossflow_rms.item() / config.u_tau,
                ),
                "turbulent_kinetic_energy": float(
                    turbulent_kinetic_energy.item(),
                ),
                "reynolds_shear_xy": float(
                    (-(fluctuation_x * fluctuation_y)[fluid].mean()).item(),
                ),
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
                moment_profile_sum=moment_profile_sum,
                profile_samples=profile_samples,
                statistics_reset_step=statistics_reset_step,
                reports=reports,
                block_start_step=block_start_step,
                block_start_momentum_x=block_start_momentum,
            )

    if not profile_samples:
        raise RuntimeError("channel collected no profile samples")
    raw_moments = moment_profile_sum / profile_samples
    mean_velocity_profiles = raw_moments[:3]
    reynolds_stress_profiles = torch.stack((
        raw_moments[3] - mean_velocity_profiles[0].square(),
        raw_moments[4] - mean_velocity_profiles[1].square(),
        raw_moments[5] - mean_velocity_profiles[2].square(),
        raw_moments[6] - mean_velocity_profiles[0] * mean_velocity_profiles[1],
    ))
    recent = reports[-3:]
    recent_u_tau = [float(item["measured_friction_velocity"]) for item in recent]
    recent_mean = sum(recent_u_tau) / len(recent_u_tau)
    recent_range_fraction = (
        (max(recent_u_tau) - min(recent_u_tau))
        / max(abs(recent_mean), 1.0e-30)
    )
    mean_error_pct = (recent_mean - config.u_tau) / config.u_tau * 100.0
    recent_crossflow_ratio = sum(
        float(item["crossflow_rms_over_u_tau"]) for item in recent
    ) / len(recent)
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
            "statistics_reset_step": statistics_reset_step,
            "mean_velocity_profile": mean_velocity_profiles[0].tolist(),
            "mean_velocity_profiles_xyz": mean_velocity_profiles.tolist(),
            "reynolds_stress_profiles_uu_vv_ww_uv": (
                reynolds_stress_profiles.tolist()
            ),
            "recent_friction_velocity_mean": recent_mean,
            "recent_friction_velocity_range_fraction": recent_range_fraction,
            "recent_friction_velocity_error_pct": mean_error_pct,
            "recent_crossflow_rms_over_u_tau": recent_crossflow_ratio,
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
            "sustained_three_dimensional_fluctuations": (
                recent_crossflow_ratio >= 0.1
            ),
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
