from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries import apply_simple_channel_boundaries, cylinder_mask, make_channel_wall_mask
from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CylinderFlowConfig:
    nx: int = 320
    ny: int = 100
    u_in: float = 0.08
    re: float = 100.0
    radius: float = 12.0
    n_steps: int = 1200
    output_interval: int = 200
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
        if self.nx < 16 or self.ny < 8:
            msg = "nx and ny must be at least 16 and 8"
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
        return f"nx{self.nx}_ny{self.ny}_re{re_label}_uin{self.u_in:.3f}_steps{self.n_steps}"


@dataclass(frozen=True)
class DiagnosticPoint:
    step: int
    mass: float
    mass_drift: float
    max_speed: float
    mean_rho: float


def compute_vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    dudy = torch.zeros_like(uy)
    dvdx = torch.zeros_like(ux)
    dudy[1:-1, :] = 0.5 * (uy[2:, :] - uy[:-2, :])
    dvdx[:, 1:-1] = 0.5 * (ux[:, 2:] - ux[:, :-2])
    return dvdx - dudy


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA requested but not available"
            raise RuntimeError(msg)
        return torch.device("cuda")
    msg = f"Unsupported device: {device_name}"
    raise ValueError(msg)


def _prepare_run_dir(config: CylinderFlowConfig) -> Path:
    run_dir = config.output_root / "cylinder_flow" / config.resolved_run_name()
    if config.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_flow_snapshot(run_dir: Path, step: int, speed: torch.Tensor, vort: torch.Tensor, obstacle: torch.Tensor) -> None:
    speed_np = speed.detach().cpu().numpy()
    vort_np = vort.detach().cpu().numpy()
    obs_np = obstacle.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(speed_np, origin="lower", cmap="viridis")
    axes[0].contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    axes[0].set_title("Velocity magnitude")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(vort_np, origin="lower", cmap="coolwarm")
    axes[1].contour(obs_np, levels=[0.5], colors="black", linewidths=0.7)
    axes[1].set_title("Vorticity")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_cylinder_flow(config: CylinderFlowConfig) -> Path:
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = _resolve_device(config.device)
    run_dir = _prepare_run_dir(config)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    cx, cy = config.nx * 0.25, config.ny * 0.5
    obstacle = cylinder_mask(config.nx, config.ny, cx, cy, config.radius, device=device)
    wall_mask = make_channel_wall_mask(config.ny, config.nx, obstacle, device=device)

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = torch.full((config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros((config.ny, config.nx), device=device)
    ux0[obstacle] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    initial_mass = float(rho0.sum().item())
    diagnostics: list[dict[str, float | int]] = []

    print(
        "Running D2Q9 cylinder flow "
        f"device={device} NX={config.nx} NY={config.ny} tau={config.tau:.4f} "
        f"steps={config.n_steps} output_interval={config.output_interval}"
    )
    print(f"Run directory: {run_dir}")

    for step in range(1, config.n_steps + 1):
        f = collide_bgk(f, tau=config.tau)
        f = stream(f)
        f = apply_simple_channel_boundaries(f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=obstacle)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            mass = float(rho.sum().item())

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            print(
                f"step={point.step:5d} mass={point.mass:.6f} "
                f"drift={point.mass_drift:+.6f} mean_rho={point.mean_rho:.6f} "
                f"max|u|={point.max_speed:.6f}"
            )

            vort = compute_vorticity(ux, uy)
            _save_flow_snapshot(run_dir, step, speed, vort, obstacle)

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")
    return run_dir
