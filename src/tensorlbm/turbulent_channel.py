"""Body-force-driven 2D turbulent-channel benchmark with Smagorinsky LES.

The flow is periodic in x and bounded by no-slip walls at the bottom and top.
A constant streamwise body force drives the channel and the resulting mean
velocity profile is compared against the viscous sublayer and logarithmic law
of the wall in wall units.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from .boundaries import bounce_back_cells, make_channel_wall_mask
from .config_io import load_config_json, save_config_json
from .d2q9 import C, W, equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .roughness import apply_rough_wall_damping_2d
from .solver import stream
from .turbulence import collide_smagorinsky_bgk
from .turbulence_stats import TurbulenceStatsAccumulator, compute_turbulence_intensity
from .utils import (
    DiagnosticPoint,
    flow_step_image_path,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TurbulentChannelConfig:
    """Configuration for the 2D turbulent-channel LES benchmark."""

    nx: int = 256
    ny: int = 64
    re_tau: float = 100.0
    u_tau: float = 0.005
    smagorinsky_cs: float = 0.1
    n_steps: int = 50000
    averaging_start: int = 20000
    output_interval: int = 5000
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def H(self) -> int:
        """Fluid-channel height between the two bounce-back walls."""
        return self.ny - 2

    @property
    def nu(self) -> float:
        """Kinematic viscosity implied by Re_tau = u_tau (H/2) / nu."""
        return self.u_tau * (self.H / 2.0) / self.re_tau

    @property
    def tau(self) -> float:
        """BGK relaxation time based on molecular viscosity."""
        return 3.0 * self.nu + 0.5

    @property
    def body_force(self) -> float:
        """Streamwise acceleration that drives the channel."""
        return 2.0 * self.u_tau**2 / self.H

    def validate(self) -> None:
        if self.nx < 8:
            msg = "nx must be >= 8"
            raise ValueError(msg)
        if self.ny < 8:
            msg = "ny must be >= 8"
            raise ValueError(msg)
        if self.re_tau <= 0.0:
            msg = "re_tau must be > 0"
            raise ValueError(msg)
        if self.u_tau <= 0.0:
            msg = "u_tau must be > 0"
            raise ValueError(msg)
        if not (0.0 < self.smagorinsky_cs < 0.5):
            msg = "smagorinsky_cs must lie in (0, 0.5)"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"tau={self.tau} must be > 0.5"
            raise ValueError(msg)
        if self.averaging_start >= self.n_steps:
            msg = "averaging_start must be less than n_steps"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"turbulent_channel_nx{self.nx}_ny{self.ny}_retau{self.re_tau:g}"
            f"_utau{self.u_tau:.4f}_steps{self.n_steps}"
        )

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file."""
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TurbulentChannelConfig:
        """Load a :class:`TurbulentChannelConfig` from a JSON file."""
        return load_config_json(cls, path)


def log_law_velocity(y_plus: float, kappa: float = 0.41, B: float = 5.2) -> float:
    """Return the logarithmic-law velocity u-plus."""
    return (1.0 / kappa) * math.log(y_plus) + B


def viscous_sublayer_velocity(y_plus: float) -> float:
    """Return the viscous-sublayer velocity u-plus = y-plus."""
    return y_plus


def _apply_body_force_2d(f: torch.Tensor, a_x: float) -> torch.Tensor:
    """Apply a first-order Guo/Luo streamwise body-force correction."""
    device = f.device
    c = C.to(device).float()
    w = W.to(device).float()
    rho = f.sum(dim=0)
    cx = c[:, 0].view(9, 1, 1)
    w_view = w.view(9, 1, 1)
    f_new = f + w_view * 3.0 * rho.unsqueeze(0) * cx * a_x
    return f_new


def _reference_velocity(y_plus: float) -> float:
    if y_plus < 5.0:
        return viscous_sublayer_velocity(y_plus)
    if y_plus > 11.0:
        return log_law_velocity(y_plus)
    return float("nan")


def _save_snapshot(run_dir: Path, ux_mean: torch.Tensor, step: int) -> None:
    profile = ux_mean.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = ax.imshow(profile, origin="lower", cmap="magma")
    plt.colorbar(image, ax=ax, fraction=0.03, label="Mean u_x")
    ax.set_title(f"Turbulent channel mean streamwise velocity – step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(flow_step_image_path(run_dir, step), dpi=140)
    write_legacy_snapshot_alias(run_dir, step)
    plt.close(fig)


def _summarize_turbulence_stats(
    stats: TurbulenceStatsAccumulator,
    *,
    u_ref: float,
) -> dict[str, object]:
    tu = compute_turbulence_intensity(stats.tke, u_ref=max(u_ref, 1e-12))
    return {
        "n_samples": stats.count,
        "domain_mean_u": float(stats.mean_u.mean().item()),
        "domain_mean_v": float(stats.mean_v.mean().item()),
        "uu_mean": float(stats.uu.mean().item()),
        "vv_mean": float(stats.vv.mean().item()),
        "uv_mean": float(stats.uv.mean().item()),
        "tke_mean": float(stats.tke.mean().item()),
        "tu_percent_mean": float(tu.mean().item()),
        "wall_profile": {
            "mean_u": stats.mean_u.mean(dim=-1).cpu().tolist(),
            "uu": stats.uu.mean(dim=-1).cpu().tolist(),
            "vv": stats.vv.mean(dim=-1).cpu().tolist(),
            "uv": stats.uv.mean(dim=-1).cpu().tolist(),
            "tke": stats.tke.mean(dim=-1).cpu().tolist(),
            "tu_percent": tu.mean(dim=-1).cpu().tolist(),
        },
    }


def run_turbulent_channel(
    config: TurbulentChannelConfig,
    *,
    rough_wall: dict[str, object] | None = None,
    turbulence_statistics: dict[str, object] | None = None,
    diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    """Run the turbulent-channel LES benchmark."""
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "turbulent_channel",
        config.resolved_run_name(),
        config.overwrite,
    )
    obstacle = torch.zeros((config.ny, config.nx), dtype=torch.bool, device=device)
    wall_mask = make_channel_wall_mask(config.ny, config.nx, obstacle, device)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {
            "H": config.H,
            "nu": config.nu,
            "tau": config.tau,
            "body_force": config.body_force,
        },
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
        "engineering_closure": {
            "rough_wall": rough_wall or {"enabled": False},
            "turbulence_statistics": turbulence_statistics or {"enabled": False},
        },
    }

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = config.u_tau * 0.1 * torch.rand((config.ny, config.nx), device=device)
    uy0 = torch.zeros((config.ny, config.nx), device=device)
    ux0[wall_mask] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    ux_sum = torch.zeros((config.ny, config.nx), device=device)
    sample_count = 0
    diagnostics: list[dict[str, object]] = []
    initial_mass = float(rho0.sum().item())
    roughness_damping_history: list[float] = []
    turbulence_acc: TurbulenceStatsAccumulator | None = None
    turbulence_start_step = config.averaging_start
    turbulence_sample_every = 1

    if turbulence_statistics and bool(turbulence_statistics.get("enabled", False)):
        turbulence_acc = TurbulenceStatsAccumulator()
        turbulence_start_step = int(turbulence_statistics.get("start_step", config.averaging_start))
        turbulence_sample_every = int(turbulence_statistics.get("sample_every", 1))

    logger.info(
        "Running turbulent channel device=%s NX=%s NY=%s Re_tau=%.1f u_tau=%.4f",
        device,
        config.nx,
        config.ny,
        config.re_tau,
        config.u_tau,
    )
    logger.info("Run directory: %s", run_dir)

    step_range = range(1, config.n_steps + 1)
    step_iter = (
        _tqdm(step_range, desc="Turbulent channel", unit="step")
        if _TQDM_AVAILABLE
        else step_range
    )
    for step in step_iter:
        f = collide_smagorinsky_bgk(
            f,
            tau=config.tau,
            C_s=config.smagorinsky_cs,
        )
        f = stream(f)
        f = _apply_body_force_2d(f, config.body_force)
        f = bounce_back_cells(f, wall_mask)
        if rough_wall and bool(rough_wall.get("enabled", False)):
            f, mean_damping = apply_rough_wall_damping_2d(
                f,
                config.nu,
                float(rough_wall.get("ks", 0.5)),
                reference_u_tau=(
                    float(rough_wall["reference_u_tau"])
                    if rough_wall.get("reference_u_tau") is not None
                    else None
                ),
                damping_limit=float(rough_wall.get("damping_limit", 0.75)),
            )
            roughness_damping_history.append(mean_damping)

        rho, ux, uy = macroscopic(f)
        ux = ux.masked_fill(wall_mask, 0.0)
        uy = uy.masked_fill(wall_mask, 0.0)
        if step >= config.averaging_start:
            ux_sum += ux
            sample_count += 1
        if (
            turbulence_acc is not None
            and step >= turbulence_start_step
            and (step - turbulence_start_step) % turbulence_sample_every == 0
        ):
            turbulence_acc.update(ux, uy)

        if step % config.output_interval == 0 or step == config.n_steps:
            speed = torch.sqrt(ux * ux + uy * uy)
            point = DiagnosticPoint(
                step=step,
                mass=float(rho.sum().item()),
                mass_drift=float(rho.sum().item()) - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            if roughness_damping_history:
                diagnostics[-1]["roughness_damping_mean"] = roughness_damping_history[-1]
            if turbulence_acc is not None and turbulence_acc.count > 0:
                diagnostics[-1]["tke_mean"] = float(turbulence_acc.tke.mean().item())
            logger.info(
                "step=%6d max|u|=%.6f mass_drift=%+.6f samples=%d",
                step,
                point.max_speed,
                point.mass_drift,
                sample_count,
            )
            if diagnostic_callback is not None:
                diagnostic_callback(diagnostics[-1])

    ux_mean = ux_sum / max(sample_count, 1)
    fluid_rows = torch.arange(1, config.ny - 1, dtype=torch.float32, device=device)
    y_plus = fluid_rows * (config.u_tau / config.nu)
    u_plus = ux_mean[1:-1, :].mean(dim=-1) / config.u_tau

    # Compute log-law RMS error (30 < y+ < 0.8*Re_tau)
    log_region = (y_plus > 30) & (y_plus < 0.8 * config.re_tau)
    if log_region.any():
        y_plus_log = y_plus[log_region]
        u_plus_log = u_plus[log_region]
        u_plus_ref = torch.tensor(
            [_reference_velocity(float(y)) for y in y_plus_log],
            device=device,
        )
        rms_error = float(torch.sqrt(torch.mean((u_plus_log - u_plus_ref) ** 2)).item())
        metadata["log_law_rms_error"] = rms_error
        logger.info("Log-law RMS error (30 < y+ < 0.8*Re_tau): %.4f", rms_error)

    with (run_dir / "velocity_profile.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["y", "y_plus", "u_plus", "u_plus_loglaw"])
        for y, y_p, u_p in zip(
            fluid_rows.tolist(),
            y_plus.tolist(),
            u_plus.tolist(),
            strict=False,
        ):
            writer.writerow([y, y_p, u_p, _reference_velocity(float(y_p))])

    metadata["diagnostics"] = diagnostics
    metadata["averaging_samples"] = sample_count
    if roughness_damping_history:
        metadata["engineering_closure"]["rough_wall_runtime"] = {
            "mean_damping": (
                sum(roughness_damping_history) / len(roughness_damping_history)
            ),
            "last_damping": roughness_damping_history[-1],
        }
    if turbulence_acc is not None and turbulence_acc.count > 0:
        metadata["engineering_closure"]["turbulence_statistics_runtime"] = (
            _summarize_turbulence_stats(
                turbulence_acc,
                u_ref=config.u_tau,
            )
        )
    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_snapshot(run_dir, ux_mean, config.n_steps)
    logger.info("Saved turbulent-channel metadata: %s", meta_path)
    return run_dir


__all__ = [
    "TurbulentChannelConfig",
    "log_law_velocity",
    "run_turbulent_channel",
    "viscous_sublayer_velocity",
]
