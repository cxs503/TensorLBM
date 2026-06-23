"""3-D D3Q27 channel flow past a sphere.

Provides an end-to-end simulation runner analogous to
:mod:`tensorlbm.sphere_flow` but using the D3Q27 lattice (27 velocity
directions) instead of D3Q19.  D3Q27 achieves 4th-order isotropy and
can reduce numerical artefacts in flows with strong corner-region gradients.

Key differences from the D3Q19 runner:

* Collision: :func:`~tensorlbm.d3q27.collide_bgk27`
* Streaming: :func:`~tensorlbm.d3q27.stream27`
* Boundaries: :mod:`tensorlbm.boundaries_d3q27` (D3Q27 Zou/He and bounce-back)
* Forces: :func:`~tensorlbm.obstacles.compute_obstacle_forces_27`
* Mass correction: :func:`~tensorlbm.d3q27.correct_mass27`
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import sphere_mask
from .boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    make_channel_wall_mask_27,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .cylinder_flow import _maybe_compile
from .d3q27 import collide_bgk27, equilibrium27, macroscopic27, stream27
from .logging_config import configure_logging, logger
from .obstacles import compute_obstacle_forces_27
from .utils import (
    DiagnosticPoint,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SphereFlowD3Q27Config:
    """Configuration for a 3-D D3Q27 channel flow past a sphere."""

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
    resume_checkpoint: Path | None = None
    use_compile: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))

    @property
    def nu(self) -> float:
        """Kinematic viscosity derived from Re = u_in · 2r / ν."""
        return self.u_in * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time τ = 3ν + 0.5."""
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is invalid."""
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


def _save_flow_snapshot_d3q27(
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


def run_sphere_flow_d3q27(config: SphereFlowD3Q27Config) -> Path:
    """Run a 3-D D3Q27 channel flow past a sphere and save results.

    Args:
        config: Simulation configuration.

    Returns:
        Path to the run output directory.
    """
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "sphere_flow_d3q27",
        config.resolved_run_name(),
        config.overwrite,
    )

    ckpt_str = str(config.resume_checkpoint) if config.resume_checkpoint else None
    metadata: dict[str, object] = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "resume_checkpoint": ckpt_str,
        },
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
    wall_mask = make_channel_wall_mask_27(
        config.nz,
        config.ny,
        config.nx,
        obstacle,
        device=device,
    )

    # Resume from checkpoint or initialise fresh
    start_step = 1
    restart_info: dict[str, object] = {"resumed": False}
    if config.resume_checkpoint is not None:
        f, resume_step, ckpt_meta = load_checkpoint(
            config.resume_checkpoint,
            device=device,
            expected_shape=(27, config.nz, config.ny, config.nx),
            expected_lattice_directions=27,
        )
        if resume_step >= config.n_steps:
            raise ValueError(
                f"resume checkpoint step {resume_step} is not less than n_steps={config.n_steps}"
            )
        f = f.to(device)
        start_step = resume_step + 1
        logger.info("Resumed from checkpoint %s at step %d", config.resume_checkpoint, resume_step)
        restart_info = {
            "resumed": True,
            "source_checkpoint": str(config.resume_checkpoint),
            "source_step": resume_step,
            "checkpoint_format_version": ckpt_meta.get("format_version"),
        }
    else:
        rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
        ux0 = torch.full((config.nz, config.ny, config.nx), config.u_in, device=device)
        uy0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
        uz0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
        ux0[obstacle] = 0.0
        f = equilibrium27(rho0, ux0, uy0, uz0, device=device)

    rho0_mass = torch.ones((config.nz, config.ny, config.nx), device=device)
    initial_mass = float(rho0_mass.sum().item())
    diagnostics: list[dict[str, float | int]] = []

    diameter = 2.0 * config.radius
    ref_area = diameter
    dyn_pressure = 0.5 * config.u_in**2 * ref_area

    # Optionally JIT-compile the hot-path kernels
    _collide = _maybe_compile(collide_bgk27, config.use_compile)
    _stream = _maybe_compile(stream27, config.use_compile)

    logger.info(
        "Running D3Q27 sphere flow device=%s NX=%s NY=%s NZ=%s tau=%.4f steps=%s "
        "output_interval=%s compile=%s",
        device,
        config.nx,
        config.ny,
        config.nz,
        config.tau,
        config.n_steps,
        config.output_interval,
        config.use_compile,
    )
    logger.info("Run directory: %s", run_dir)

    for step in range(start_step, config.n_steps + 1):
        f = _collide(f, tau=config.tau)
        f = _stream(f)
        fx, fy, fz = compute_obstacle_forces_27(f, obstacle)
        f = apply_zou_he_channel_boundaries_27(
            f,
            u_in=config.u_in,
            wall_mask=wall_mask,
            obstacle_mask=obstacle,
        )

        if step % config.output_interval == 0 or step == config.n_steps:
            # Sync only at output intervals (avoids per-step GPU→CPU stall)
            cd = float(fx.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

            rho, ux, uy, uz = macroscopic27(f)
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
            diag_entry: dict[str, float | int] = {**asdict(point), "cd": cd}
            diagnostics.append(diag_entry)
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f Cd=%.4f",
                point.step,
                point.mass,
                point.mass_drift,
                point.mean_rho,
                point.max_speed,
                cd,
            )
            _save_flow_snapshot_d3q27(run_dir, step, speed, obstacle, config.nz)

            # Save checkpoint at every output step
            save_checkpoint(f, step, run_dir)

    metadata["diagnostics"] = diagnostics
    metadata["restart"] = restart_info
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir


__all__ = [
    "SphereFlowD3Q27Config",
    "run_sphere_flow_d3q27",
]
