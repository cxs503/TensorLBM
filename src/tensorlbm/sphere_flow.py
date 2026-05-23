from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import apply_simple_channel_boundaries_3d, make_channel_wall_mask_3d, sphere_mask
from .d3q19 import equilibrium3d, macroscopic3d
from .logging_config import configure_logging, logger
from .solver3d import collide_bgk3d, stream3d
from .utils import (
    DiagnosticPoint,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SphereFlowConfig:
    nx: int = 120
    ny: int = 60
    nz: int = 60
    u_in: float = 0.06
    re: float = 50.0
    radius: float = 8.0
    n_steps: int = 500
    output_interval: int = 100
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def nu(self) -> float:
        return self.u_in * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            msg = "nx, ny, and nz must be at least 16, 8, and 8"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)
        if self.u_in <= 0.0 or self.re <= 0.0 or self.radius <= 0.0:
            msg = "u_in, re, and radius must be > 0"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/radius"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"nx{self.nx}_ny{self.ny}_nz{self.nz}_re{re_label}"
            f"_uin{self.u_in:.3f}_steps{self.n_steps}"
        )


def _save_flow_snapshot_3d(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
    nz: int,
) -> None:
    """Save speed magnitude on the mid-z slice as a PNG image."""
    mid_z = nz // 2
    speed_np = speed[mid_z].detach().cpu().numpy()
    obs_np = obstacle[mid_z].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = ax.imshow(speed_np, origin="lower", cmap="viridis")
    ax.contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    ax.set_title(f"Velocity magnitude – mid-z slice (step {step})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_sphere_flow(config: SphereFlowConfig) -> Path:
    """Run a 3D D3Q19 channel flow past a sphere and save results."""
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "sphere_flow",
        config.resolved_run_name(),
        config.overwrite,
    )

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    cx = config.nx * 0.25
    cy = config.ny * 0.5
    cz = config.nz * 0.5
    obstacle = sphere_mask(
        config.nx,
        config.ny,
        config.nz,
        cx,
        cy,
        cz,
        config.radius,
        device=device,
    )
    wall_mask = make_channel_wall_mask_3d(
        config.nz,
        config.ny,
        config.nx,
        obstacle,
        device=device,
    )

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full((config.nz, config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
    uz0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
    ux0[obstacle] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    diagnostics: list[dict[str, float | int]] = []

    logger.info(
        "Running D3Q19 sphere flow device=%s NX=%s NY=%s NZ=%s tau=%.4f steps=%s "
        "output_interval=%s",
        device,
        config.nx,
        config.ny,
        config.nz,
        config.tau,
        config.n_steps,
        config.output_interval,
    )
    logger.info("Run directory: %s", run_dir)

    for step in range(1, config.n_steps + 1):
        f = collide_bgk3d(f, tau=config.tau)
        f = stream3d(f)
        f = apply_simple_channel_boundaries_3d(
            f,
            u_in=config.u_in,
            wall_mask=wall_mask,
            obstacle_mask=obstacle,
        )

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            uz = uz.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
            mass = float(rho.sum().item())

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f",
                point.step,
                point.mass,
                point.mass_drift,
                point.mean_rho,
                point.max_speed,
            )
            _save_flow_snapshot_3d(run_dir, step, speed, obstacle, config.nz)

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir
