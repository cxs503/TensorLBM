"""3-D Wigley ship hull flow simulation for ship and ocean engineering.

This module provides a self-contained, parameterised runner for a Lattice
Boltzmann simulation of viscous flow past a Wigley parabolic ship hull in a
rectangular channel.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .logging_config import configure_logging, logger
from .obstacles import compute_obstacle_forces_3d, compute_obstacle_moments_3d, wigley_hull_mask
from .turbulence import collide_smagorinsky_mrt3d
from .utils import (
    DiagnosticPoint,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)
from .wave_bc import apply_wave_inlet_3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShipHullFlowConfig:
    """Configuration for a 3-D Wigley hull channel-flow simulation."""

    nx: int = 160
    ny: int = 60
    nz: int = 40
    u_in: float = 0.05
    re: float = 200.0
    hull_length: float = 80.0
    hull_beam: float = 8.0
    hull_draft: float = 12.0
    smagorinsky_cs: float = 0.1
    wave_amp: float = 0.0
    wave_period: float = 200.0
    wave_k: float = 0.05
    water_depth: float = 0.0
    n_steps: int = 2000
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
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def froude(self) -> float:
        if self.wave_k <= 0.0:
            return float("inf")
        g_lbm = (1.0 / 3.0) * self.wave_k
        return self.u_in / (g_lbm * self.hull_length) ** 0.5

    def _effective_water_depth(self) -> float:
        return self.water_depth if self.water_depth > 0.0 else float(self.nz)

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, and nz must be at least 16, 8, and 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0:
            raise ValueError("u_in and re must be > 0")
        if self.hull_length <= 0.0 or self.hull_beam <= 0.0 or self.hull_draft <= 0.0:
            raise ValueError("hull_length, hull_beam, and hull_draft must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/hull_length"
            )
        if self.hull_length >= self.nx:
            raise ValueError("hull_length must be less than nx")
        if self.hull_beam >= self.ny:
            raise ValueError("hull_beam must be less than ny")
        if self.hull_draft >= self.nz:
            raise ValueError("hull_draft must be less than nz")

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"nx{self.nx}_ny{self.ny}_nz{self.nz}_re{re_label}"
            f"_uin{self.u_in:.3f}_L{int(self.hull_length)}"
            f"_B{int(self.hull_beam)}_T{int(self.hull_draft)}"
            f"_steps{self.n_steps}"
        )


def _save_ship_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
    nz: int,
    ny: int,
) -> None:
    """Save speed-magnitude slices (mid-z and mid-y) as PNG images."""
    mid_z = nz // 2
    mid_y = ny // 2

    speed_np_z = speed[mid_z].detach().cpu().numpy()
    speed_np_y = speed[:, mid_y, :].detach().cpu().numpy()
    obs_z = obstacle[mid_z].detach().cpu().float().numpy()
    obs_y = obstacle[:, mid_y, :].detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    im0 = axes[0].imshow(speed_np_z, origin="lower", cmap="viridis")
    axes[0].contour(obs_z, levels=[0.5], colors="white", linewidths=0.8)
    axes[0].set_title(f"Speed – mid-z slice (step {step})")
    axes[0].set_xlabel("x (longitudinal)")
    axes[0].set_ylabel("y (transverse)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(speed_np_y, origin="lower", cmap="viridis")
    axes[1].contour(obs_y, levels=[0.5], colors="white", linewidths=0.8)
    axes[1].set_title(f"Speed – mid-y slice (step {step})")
    axes[1].set_xlabel("x (longitudinal)")
    axes[1].set_ylabel("z (vertical)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.savefig(run_dir / f"flow_step_{step:06d}.png", dpi=140)
    plt.close(fig)


def run_ship_hull_flow(config: ShipHullFlowConfig) -> Path:
    """Run a 3-D Wigley hull channel-flow simulation and save results."""
    from .solver3d import stream3d

    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "ship_hull_flow",
        config.resolved_run_name(),
        config.overwrite,
    )

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau, "froude": config.froude},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    cx_hull = config.nx * 0.35
    cy_hull = config.ny * 0.5
    cz_keel = config.nz * 0.1
    effective_depth = config._effective_water_depth()

    obstacle = wigley_hull_mask(
        config.nx,
        config.ny,
        config.nz,
        cx=cx_hull,
        cy=cy_hull,
        cz_keel=cz_keel,
        length=config.hull_length,
        beam=config.hull_beam,
        draft=config.hull_draft,
        device=device,
    )
    wall_mask = make_channel_wall_mask_3d(config.nz, config.ny, config.nx, obstacle, device=device)

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[obstacle] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    ref_area = config.hull_length * config.hull_draft
    dyn_pressure = 0.5 * config.u_in**2 * ref_area

    logger.info(
        "Running D3Q19 Wigley hull flow device=%s NX=%s NY=%s NZ=%s tau=%.4f Re=%.1f "
        "Fr=%.4f hull L=%s B=%s T=%s wave_amp=%s Cs=%s steps=%s output_interval=%s",
        device,
        config.nx,
        config.ny,
        config.nz,
        config.tau,
        config.re,
        config.froude,
        config.hull_length,
        config.hull_beam,
        config.hull_draft,
        config.wave_amp,
        config.smagorinsky_cs,
        config.n_steps,
        config.output_interval,
    )
    logger.info("Run directory: %s", run_dir)

    diagnostics: list[dict[str, object]] = []
    use_waves = config.wave_amp > 0.0

    for step in range(1, config.n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        f = stream3d(f)

        fx, fy, fz = compute_obstacle_forces_3d(f, obstacle)
        mx, my, mz = compute_obstacle_moments_3d(
            f,
            obstacle,
            cx_hull,
            cy_hull,
            cz_keel + config.hull_draft * 0.5,
        )

        if use_waves:
            f = apply_wave_inlet_3d(
                f,
                step=step,
                wall_mask=wall_mask,
                obstacle_mask=obstacle,
                u_mean=config.u_in,
                wave_amp=config.wave_amp,
                wave_period=config.wave_period,
                wave_k=config.wave_k,
                water_depth=effective_depth,
                z_bed=cz_keel,
            )
        else:
            f = apply_zou_he_channel_boundaries_3d(
                f,
                u_in=config.u_in,
                wall_mask=wall_mask,
                obstacle_mask=obstacle,
            )

        cd = float(fx) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cs_coef = float(fy) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cl = float(fz) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

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
            diag_entry: dict[str, object] = {
                **asdict(point),
                "cd": cd,
                "cs": cs_coef,
                "cl": cl,
                "mx": float(mx),
                "my": float(my),
                "mz": float(mz),
            }
            diagnostics.append(diag_entry)
            logger.info(
                "step=%5d mass=%.5f drift=%+.5f max|u|=%.5f Cd=%.4f Cs=%.4f Cl=%.4f My=%.3f",
                point.step,
                point.mass,
                point.mass_drift,
                point.max_speed,
                cd,
                cs_coef,
                cl,
                float(my),
            )
            _save_ship_snapshot(run_dir, step, speed, obstacle, config.nz, config.ny)

    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cs", "cl", "mx", "my", "mz"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cs"], d["cl"], d["mx"], d["my"], d["mz"]])

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir
