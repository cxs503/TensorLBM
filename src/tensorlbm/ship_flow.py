"""3-D Wigley ship hull flow simulation for ship and ocean engineering.

This module provides a self-contained, parameterised runner for a Lattice
Boltzmann simulation of viscous flow past a Wigley parabolic ship hull in a
rectangular channel.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .config_io import load_config_json, save_config_json
from .d3q19 import equilibrium3d, macroscopic3d
from .logging_config import configure_logging, logger
from .obstacles import compute_obstacle_forces_3d, compute_obstacle_moments_3d
from .postprocess import (
    compute_pressure_coefficient,
    compute_q_criterion,
    compute_recirculation_length,
    compute_velocity_magnitude,
    compute_vorticity_3d,
    extract_wake_profile,
)
from .ship_cad import ShipHullType, build_hull_mask, export_hull_stl, generate_hull_previews
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
    """Configuration for a 3-D parametric ship-hull channel-flow simulation."""

    nx: int = 160
    ny: int = 60
    nz: int = 40
    hull_type: str = ShipHullType.WIGLEY.value
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
    export_stl: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())
        object.__setattr__(self, "hull_type", ShipHullType(self.hull_type).value)

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
        return self.u_in / math.sqrt(g_lbm * self.hull_length)

    def _effective_water_depth(self) -> float:
        return self.water_depth if self.water_depth > 0.0 else float(self.nz)

    def validate(self) -> None:
        ShipHullType(self.hull_type)
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
            f"{self.hull_type}_nx{self.nx}_ny{self.ny}_nz{self.nz}_re{re_label}"
            f"_uin{self.u_in:.3f}_L{int(self.hull_length)}"
            f"_B{int(self.hull_beam)}_T{int(self.hull_draft)}"
            f"_steps{self.n_steps}"
        )

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file.

        Args:
            path: Output file path (should end with ``.json``).

        Returns:
            Resolved path to the written file.
        """
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ShipHullFlowConfig:
        """Load a :class:`ShipHullFlowConfig` from a JSON file.

        Args:
            path: Path to a JSON file written by :meth:`save`.

        Returns:
            Reconstructed :class:`ShipHullFlowConfig` instance.
        """
        return load_config_json(cls, path)


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


def _save_cad_artifacts(run_dir: Path, config: ShipHullFlowConfig) -> dict[str, object]:
    """Save CAD artefacts for the selected hull and return CAD summary data."""
    hull_type = ShipHullType(config.hull_type)
    _, hull_stats = build_hull_mask(
        hull_type=hull_type,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=config.nx * 0.35,
        cy=config.ny * 0.5,
        cz_keel=config.nz * 0.1,
        length=config.hull_length,
        beam=config.hull_beam,
        draft=config.hull_draft,
        device=config.device,
    )
    cb_theoretical = float(hull_stats["Cb"])
    cb_numerical = float(hull_stats["Cb_numerical"])
    cb_error_pct = abs(cb_numerical - cb_theoretical) / max(abs(cb_theoretical), 1e-12) * 100.0
    cad_summary: dict[str, object] = {
        **hull_stats,
        "Cb_theoretical": cb_theoretical,
        "Cb_relative_error_pct": cb_error_pct,
    }

    fig = generate_hull_previews(
        hull_type,
        length=config.hull_length,
        beam=config.hull_beam,
        draft=config.hull_draft,
    )
    fig.savefig(run_dir / "cad_preview.png", dpi=140)
    plt.close(fig)

    if config.export_stl:
        export_hull_stl(
            hull_type,
            length=config.hull_length,
            beam=config.hull_beam,
            draft=config.hull_draft,
            output_path=run_dir / "hull.stl",
        )

    (run_dir / "cad_summary.json").write_text(
        f"{json.dumps(cad_summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return cad_summary


def _save_postprocess_summary(
    run_dir: Path,
    config: ShipHullFlowConfig,
    diagnostics: list[dict[str, object]],
    obstacle: torch.Tensor,
    cad_summary: dict[str, object],
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> dict[str, object]:
    """Save ship post-processing artefacts and return the summary payload."""
    trailing_edge = int(round(float(cad_summary["cx"]) + float(cad_summary["length"]) * 0.5))
    wake_offset = int(config.hull_length * 0.125)
    wake_x = min(config.nx - 1, max(trailing_edge + 1, trailing_edge + wake_offset))
    wake_profile = extract_wake_profile(ux, wake_x).detach().cpu()
    with (run_dir / "wake_profile.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["y_index", "ux"])
        for y_idx, value in enumerate(wake_profile.tolist()):
            writer.writerow([y_idx, value])

    recent = diagnostics[-min(5, len(diagnostics)) :]
    cd_vals = [float(d["cd"]) for d in recent]
    cs_vals = [float(d["cs"]) for d in recent]
    cl_vals = [float(d["cl"]) for d in recent]
    drag_scale = max(abs(sum(cd_vals) / len(cd_vals)), 1e-12)

    cp = compute_pressure_coefficient(rho, config.u_in)
    fluid_cp = cp.masked_select(~obstacle)
    omega_x, omega_y, omega_z = compute_vorticity_3d(ux, uy, uz)
    omega_mag = compute_velocity_magnitude(omega_x, omega_y, omega_z)
    q_field = compute_q_criterion(ux, uy, uz)
    fluid_q = q_field.masked_select(~obstacle)

    summary: dict[str, object] = {
        "forces": {
            "samples_used": len(recent),
            "cd_mean": sum(cd_vals) / len(cd_vals),
            "cs_mean": sum(cs_vals) / len(cs_vals),
            "cl_mean": sum(cl_vals) / len(cl_vals),
            "cs_abs_ratio_to_cd": sum(abs(v) for v in cs_vals) / len(cs_vals) / drag_scale,
            "cl_abs_ratio_to_cd": sum(abs(v) for v in cl_vals) / len(cl_vals) / drag_scale,
        },
        "wake": {
            "x_index": wake_x,
            "ux_min": float(wake_profile.min().item()),
            "ux_mean": float(wake_profile.mean().item()),
            "velocity_deficit_max": config.u_in - float(wake_profile.min().item()),
            "recirculation_length_lu": compute_recirculation_length(ux, obstacle),
        },
        "pressure": {
            "cp_min": float(fluid_cp.min().item()),
            "cp_max": float(fluid_cp.max().item()),
        },
        "vorticity": {
            "omega_max": float(omega_mag.max().item()),
            "q_positive_fraction": float((fluid_q > 0.0).float().mean().item()),
        },
        "acceptance": {
            "cb_within_25pct": float(cad_summary["Cb_relative_error_pct"]) < 25.0,
            "drag_positive": abs(sum(cd_vals) / len(cd_vals)) > 0.0,
            "sideforce_small": (sum(abs(v) for v in cs_vals) / len(cs_vals) / drag_scale) < 0.10,
            "lift_small": (sum(abs(v) for v in cl_vals) / len(cl_vals) / drag_scale) < 0.25,
        },
    }
    summary["acceptance"]["workflow_ok"] = all(summary["acceptance"].values())

    (run_dir / "postprocess_summary.json").write_text(
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return summary


def run_ship_hull_flow(config: ShipHullFlowConfig) -> Path:
    """Run a 3-D ship-hull CAD + channel-flow simulation and save results."""
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
    cad_summary = _save_cad_artifacts(run_dir, config)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau, "froude": config.froude},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
        "cad": cad_summary,
    }

    cx_hull = config.nx * 0.35
    cy_hull = config.ny * 0.5
    cz_keel = config.nz * 0.1
    effective_depth = config._effective_water_depth()

    obstacle, _ = build_hull_mask(
        hull_type=config.hull_type,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=cx_hull,
        cy=cy_hull,
        cz_keel=cz_keel,
        length=config.hull_length,
        beam=config.hull_beam,
        draft=config.hull_draft,
        device=str(device),
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
    final_rho: torch.Tensor | None = None
    final_ux: torch.Tensor | None = None
    final_uy: torch.Tensor | None = None
    final_uz: torch.Tensor | None = None

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

        if step % config.output_interval == 0 or step == config.n_steps:
            # Sync only at output intervals (avoids per-step GPU→CPU stall)
            cd = float(fx.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
            cs_coef = float(fy.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
            cl = float(fz.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

            rho, ux, uy, uz = macroscopic3d(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            uz = uz.masked_fill(obstacle, 0.0)
            speed = compute_velocity_magnitude(ux, uy, uz)
            mass = float(rho.sum().item())
            final_rho, final_ux, final_uy, final_uz = rho, ux, uy, uz

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
                "mx": float(mx.item()),
                "my": float(my.item()),
                "mz": float(mz.item()),
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
                float(my.item()),
            )
            _save_ship_snapshot(run_dir, step, speed, obstacle, config.nz, config.ny)

    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cs", "cl", "mx", "my", "mz"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cs"], d["cl"], d["mx"], d["my"], d["mz"]])

    metadata["diagnostics"] = diagnostics
    if final_rho is None or final_ux is None or final_uy is None or final_uz is None:
        raise RuntimeError("Ship run completed without any diagnostic output")
    metadata["postprocess"] = _save_postprocess_summary(
        run_dir,
        config,
        diagnostics,
        obstacle,
        cad_summary,
        final_rho,
        final_ux,
        final_uy,
        final_uz,
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir
