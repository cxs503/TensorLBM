"""3-D Wigley ship hull flow simulation for ship and ocean engineering.

This module provides a self-contained, parameterised runner for a Lattice
Boltzmann simulation of viscous flow past a Wigley parabolic ship hull in a
rectangular channel.

Features
--------
- Wigley-hull body geometry (classic ITTC benchmark)
- D3Q19 MRT + Smagorinsky LES (recommended for high-Re ship flows)
- Full force/moment diagnostics: drag Cd, lateral force Cs, lift Cl,
  and roll Mx, pitch My, yaw Mz coefficients
- Optional Airy regular-wave inlet for ocean wave–body interaction
- Froude number output and per-step force CSV
- Structured run-directory output with metadata snapshot

Usage
-----
.. code-block:: bash

    PYTHONPATH=src python examples/ship_hull_flow.py \\
        --nx 120 --ny 60 --nz 40 --u-in 0.05 --re 200 \\
        --hull-length 60 --hull-beam 6 --hull-draft 10 \\
        --n-steps 1000 --output-interval 100 --overwrite
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
    wigley_hull_mask,
)
from .turbulence import collide_smagorinsky_mrt3d
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device
from .wave_bc import apply_wave_inlet_3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShipHullFlowConfig:
    """Configuration for a 3-D Wigley hull channel-flow simulation.

    Physical parameters
    -------------------
    nx, ny, nz
        Grid dimensions (longitudinal × transverse × vertical).
    u_in
        Inlet x-velocity (lattice units, should be ≤ 0.1).
    re
        Target Reynolds number Re = u_in · L / ν.

    Hull geometry
    -------------
    hull_length
        Wigley hull length *L* in lattice units.
    hull_beam
        Maximum beam *B* in lattice units.
    hull_draft
        Draft *T* in lattice units (keel to waterline).

    Turbulence
    ----------
    smagorinsky_cs
        Smagorinsky constant *C_s* (default 0.1; set to 0 to disable LES).

    Wave inlet (optional)
    ---------------------
    wave_amp
        Horizontal velocity amplitude at the free surface (lattice units).
        Set to 0 to use a steady inlet.
    wave_period
        Wave period in LBM time steps.
    wave_k
        Wave number k = 2π/λ (1/lattice spacing).
    water_depth
        Water depth in lattice units; defaults to nz when 0.

    Output
    ------
    n_steps, output_interval, output_root, run_name, seed, device, overwrite
        Standard runner parameters.
    """

    # Grid
    nx: int = 160
    ny: int = 60
    nz: int = 40
    # Flow
    u_in: float = 0.05
    re: float = 200.0
    # Hull
    hull_length: float = 80.0
    hull_beam: float = 8.0
    hull_draft: float = 12.0
    # Turbulence
    smagorinsky_cs: float = 0.1
    # Wave inlet
    wave_amp: float = 0.0
    wave_period: float = 200.0
    wave_k: float = 0.05
    water_depth: float = 0.0  # 0 → use nz
    # Output
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
        """Kinematic viscosity derived from Re."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    @property
    def froude(self) -> float:
        """Froude number Fr = U / sqrt(g_lbm · L).

        Uses the LBM lattice gravity g_lbm = (cs · wave_k)² / wave_k = cs² · k
        where cs² = 1/3.  When wave_k is zero, Froude number is undefined (inf).
        """
        if self.wave_k <= 0.0:
            return float("inf")
        g_lbm = (1.0 / 3.0) * self.wave_k  # shallow-water dispersion approximation
        return self.u_in / (g_lbm * self.hull_length) ** 0.5

    def _effective_water_depth(self) -> float:
        return self.water_depth if self.water_depth > 0.0 else float(self.nz)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is invalid."""
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

    def save(self, path: "Path | str") -> None:
        """Serialise this config to a JSON file."""
        from .config_io import save_config_json
        save_config_json(self, path)

    @classmethod
    def load(cls, path: "Path | str") -> "ShipHullFlowConfig":
        """Load a :class:`ShipHullFlowConfig` from a JSON file."""
        from .config_io import load_config_json
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


def run_ship_hull_flow(config: ShipHullFlowConfig) -> Path:
    """Run a 3-D Wigley hull channel-flow simulation and save results.

    The simulation loop:

    1. **Collide** – D3Q19 MRT with Smagorinsky LES at rate ``tau``.
    2. **Stream** – periodic gather streaming.
    3. **Force diagnostics** – momentum-exchange forces + moments before
       bounce-back (Ladd 1994 method).
    4. **Boundaries** – wave inlet (or Zou/He steady inlet) + pressure outlet
       + bounce-back on hull and walls.

    Args:
        config: Fully validated :class:`ShipHullFlowConfig` instance.

    Returns:
        Path to the run output directory.
    """
    from .solver3d import stream3d

    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "ship_hull_flow", config.resolved_run_name(), config.overwrite
    )

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {
            "nu": config.nu,
            "tau": config.tau,
            "froude": config.froude,
        },
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    # ------------------------------------------------------------------
    # Geometry: Wigley hull centred at (nx/4, ny/2) with keel at nz*0.2
    # ------------------------------------------------------------------
    cx_hull = config.nx * 0.35
    cy_hull = config.ny * 0.5
    cz_keel = config.nz * 0.1
    effective_depth = config._effective_water_depth()

    obstacle = wigley_hull_mask(
        config.nx, config.ny, config.nz,
        cx=cx_hull, cy=cy_hull, cz_keel=cz_keel,
        length=config.hull_length,
        beam=config.hull_beam,
        draft=config.hull_draft,
        device=device,
    )
    wall_mask = make_channel_wall_mask_3d(
        config.nz, config.ny, config.nx, obstacle, device=device
    )

    # ------------------------------------------------------------------
    # Initial conditions: uniform flow at u_in
    # ------------------------------------------------------------------
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[obstacle] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())

    # Non-dimensional reference quantities
    ref_area = config.hull_length * config.hull_draft  # frontal wetted-area proxy
    dyn_pressure = 0.5 * 1.0 * config.u_in ** 2 * ref_area  # ½ρU²A

    print(
        "Running D3Q19 Wigley hull flow  "
        f"device={device}  NX={config.nx} NY={config.ny} NZ={config.nz}\n"
        f"  tau={config.tau:.4f}  Re={config.re:.1f}  Fr={config.froude:.4f}\n"
        f"  hull L={config.hull_length} B={config.hull_beam} T={config.hull_draft}\n"
        f"  wave_amp={config.wave_amp}  Cs={config.smagorinsky_cs}\n"
        f"  steps={config.n_steps}  output_interval={config.output_interval}"
    )
    print(f"Run directory: {run_dir}")

    diagnostics: list[dict[str, object]] = []
    use_waves = config.wave_amp > 0.0

    for step in range(1, config.n_steps + 1):
        # 1. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)

        # 2. Streaming
        f = stream3d(f)

        # 3. Force/moment diagnostics BEFORE bounce-back
        fx, fy, fz = compute_obstacle_forces_3d(f, obstacle)
        mx, my, mz = compute_obstacle_moments_3d(
            f, obstacle, cx_hull, cy_hull, cz_keel + config.hull_draft * 0.5
        )

        # 4. Boundary conditions
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
                f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=obstacle
            )

        # 5. Diagnostics output
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
            print(
                f"step={point.step:5d}  mass={point.mass:.5f}  "
                f"drift={point.mass_drift:+.5f}  max|u|={point.max_speed:.5f}  "
                f"Cd={cd:.4f}  Cs={cs_coef:.4f}  Cl={cl:.4f}  "
                f"My={float(my):.3f}"
            )
            _save_ship_snapshot(run_dir, step, speed, obstacle, config.nz, config.ny)

    # Save force/moment time-series as CSV
    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cs", "cl", "mx", "my", "mz"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cs"], d["cl"], d["mx"], d["my"], d["mz"]])

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}")
    return run_dir
