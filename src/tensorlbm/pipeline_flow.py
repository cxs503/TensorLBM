"""2D near-bed cylinder benchmark for pipeline free-span flow.

The setup models cross-flow past a circular cylinder located close to a flat
wall. The wall proximity is parameterised by the gap ratio e/D. Drag, lift,
and the dominant shedding frequency are reported as simple engineering
diagnostics for comparison with near-bed pipeline correlations.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

from .boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
)
from .config_io import load_config_json, save_config_json
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, stream
from .utils import (
    DiagnosticPoint,
    flow_step_image_path,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PipelineFlowConfig:
    """Configuration for the near-wall pipeline-flow benchmark."""

    nx: int = 400
    ny: int = 160
    diameter: float = 20.0
    gap_ratio: float = 0.5
    u_in: float = 0.05
    re: float = 200.0
    n_steps: int = 30000
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
    def nu(self) -> float:
        """Kinematic viscosity from Re = U D / nu."""
        return self.u_in * self.diameter / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    @property
    def cylinder_center(self) -> tuple[float, float]:
        """Cylinder centre coordinates in lattice cells."""
        cx = float(self.nx // 4)
        cy = 1.0 + (self.gap_ratio + 0.5) * self.diameter
        return cx, cy

    def validate(self) -> None:
        if self.nx < 20:
            msg = "nx must be >= 20"
            raise ValueError(msg)
        if self.ny < 20:
            msg = "ny must be >= 20"
            raise ValueError(msg)
        if self.diameter < 2.0:
            msg = "diameter must be >= 2"
            raise ValueError(msg)
        if not (0.0 <= self.gap_ratio <= 2.0):
            msg = "gap_ratio must be in [0, 2.0]"
            raise ValueError(msg)
        if self.u_in <= 0.0:
            msg = "u_in must be > 0"
            raise ValueError(msg)
        if self.re <= 0.0:
            msg = "re must be > 0"
            raise ValueError(msg)
        if self.tau <= 0.500001:
            msg = f"tau={self.tau} must be sufficiently above 0.5"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)
        cx, cy = self.cylinder_center
        radius = 0.5 * self.diameter
        if cy + radius >= self.ny - 1:
            msg = "Cylinder must fit below the top wall"
            raise ValueError(msg)
        if cx + radius >= self.nx - 5:
            msg = "Cylinder must fit upstream of the outlet"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"pipeline_nx{self.nx}_ny{self.ny}_D{self.diameter:g}"
            f"_eD{self.gap_ratio:.2f}_re{self.re:g}_steps{self.n_steps}"
        )

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file."""
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> PipelineFlowConfig:
        """Load a :class:`PipelineFlowConfig` from a JSON file."""
        return load_config_json(cls, path)


def measure_strouhal(cl_series: list[float], u_in: float, diameter: float) -> float:
    """Estimate the Strouhal number from a lift-coefficient history."""
    if len(cl_series) < 20 or u_in <= 0.0 or diameter <= 0.0:
        return 0.0
    data = np.asarray(cl_series, dtype=float)
    centered = data - data.mean()
    amplitudes = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=1.0)
    if amplitudes.size <= 1:
        return 0.0
    amplitudes[0] = 0.0
    peak_index = int(np.argmax(amplitudes))
    if peak_index <= 0 or float(amplitudes[peak_index]) <= 1e-12:
        return 0.0
    return float(freqs[peak_index]) * diameter / u_in


def make_pipeline_wall_mask(
    ny: int,
    nx: int,
    obstacle: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return top and bottom channel walls excluding the obstacle."""
    return make_channel_wall_mask(ny, nx, obstacle, device)


def _apply_pipeline_outlet(f: torch.Tensor) -> torch.Tensor:
    f_new = f.clone()
    f_new[:, :, -1] = f[:, :, -2]
    return f_new


def _save_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
) -> None:
    speed_np = speed.detach().cpu().numpy()
    obstacle_np = obstacle.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = ax.imshow(speed_np, origin="lower", cmap="viridis")
    ax.contour(obstacle_np, levels=[0.5], colors="white", linewidths=0.8)
    plt.colorbar(image, ax=ax, fraction=0.03, label="|u|")
    ax.set_title(f"Pipeline flow velocity magnitude – step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(flow_step_image_path(run_dir, step), dpi=140)
    write_legacy_snapshot_alias(run_dir, step)
    plt.close(fig)


def run_pipeline_flow(config: PipelineFlowConfig) -> Path:
    """Run the near-wall pipeline-flow benchmark."""
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "pipeline_flow",
        config.resolved_run_name(),
        config.overwrite,
    )
    cx, cy = config.cylinder_center
    radius = 0.5 * config.diameter
    obstacle = cylinder_mask(config.nx, config.ny, cx, cy, radius, device=device)
    wall_mask = make_pipeline_wall_mask(config.ny, config.nx, obstacle, device)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {
            "nu": config.nu,
            "tau": config.tau,
            "cylinder_center": [cx, cy],
        },
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = torch.full((config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros((config.ny, config.nx), device=device)
    ux0[obstacle | wall_mask] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)
    initial_mass = float(rho0.sum().item())

    diagnostics: list[dict[str, object]] = []
    cl_series: list[float] = []
    dyn_pressure = 0.5 * config.u_in**2 * config.diameter

    logger.info(
        "Running pipeline flow device=%s NX=%s NY=%s D=%.2f gap_ratio=%.2f",
        device,
        config.nx,
        config.ny,
        config.diameter,
        config.gap_ratio,
    )
    logger.info("Run directory: %s", run_dir)

    for step in range(1, config.n_steps + 1):
        f = collide_bgk(f, tau=config.tau)
        f = stream(f)
        f = zou_he_inlet_velocity(f, config.u_in)
        f = _apply_pipeline_outlet(f)
        fx, fy = compute_obstacle_forces(f, obstacle)
        f = bounce_back_cells(f, wall_mask)
        f = bounce_back_cells(f, obstacle)

        cd = float(fx) / dyn_pressure if dyn_pressure > 0.0 else 0.0
        cl = float(fy) / dyn_pressure if dyn_pressure > 0.0 else 0.0
        cl_series.append(cl)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle | wall_mask, 0.0)
            uy = uy.masked_fill(obstacle | wall_mask, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            point = DiagnosticPoint(
                step=step,
                mass=float(rho.sum().item()),
                mass_drift=float(rho.sum().item()) - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            entry = {**asdict(point), "cd": cd, "cl": cl}
            diagnostics.append(entry)
            logger.info(
                "step=%6d Cd=%.5f Cl=%.5f max|u|=%.6f drift=%+.6f",
                step,
                cd,
                cl,
                point.max_speed,
                point.mass_drift,
            )
            _save_snapshot(run_dir, step, speed, obstacle)

    strouhal = measure_strouhal(cl_series[len(cl_series) // 2 :], config.u_in, config.diameter)

    with (run_dir / "forces.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cl"])
        for entry in diagnostics:
            writer.writerow([entry["step"], entry["cd"], entry["cl"]])

    (run_dir / "strouhal.json").write_text(
        json.dumps({"strouhal": strouhal}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata["diagnostics"] = diagnostics
    metadata["strouhal"] = strouhal
    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved pipeline metadata: %s", meta_path)
    return run_dir


__all__ = [
    "PipelineFlowConfig",
    "make_pipeline_wall_mask",
    "measure_strouhal",
    "run_pipeline_flow",
]
