"""3-D sphere water-entry simulation.

Simulates a sphere descending into a pool of water at constant speed.
In the reference frame of the sphere the water flows upward (z-direction)
at the entry velocity *v_entry*, enabling a fixed-geometry D3Q19 LBM
simulation with standard bounce-back boundary conditions.

Physical model
--------------
* D3Q19 BGK collision (or BGK + Smagorinsky LES when *smagorinsky_cs* > 0).
* Velocity inlet at the bottom face (z = 0): water flows upward at *v_entry*
  with a smooth linear ramp-up over the first *n_ramp* steps.
* Pressure outlet at the top face (z = nz − 1): prescribes ρ = 1.
* No-slip bounce-back on the four lateral walls (±x, ±y faces).
* No-slip bounce-back on the sphere surface.

Key outputs
-----------
* ``forces.csv``           – per-step drag force *Fz* and drag coefficient
  *Cd* time-series.
* ``flow_step_XXXXXX.png`` – side-view (xz-plane) speed and vertical-velocity
  snapshots.
* ``force_history.png``    – *Fz* and *Cd* vs simulation step.
* ``run_metadata.json``    – configuration, derived parameters, and
  diagnostic summary.

Physical interpretation
-----------------------
In the sphere's reference frame the hydrodynamic drag force in the
z-direction (*Fz*, computed with Ladd's 1994 momentum-exchange method) equals
the resistance the water exerts on the descending sphere.  The drag
coefficient is:

.. math::

    C_d = \\frac{F_z}{\\tfrac{1}{2} \\rho U^2 \\pi r^2}

where *U* is the current entry velocity and *r* is the sphere radius.

References
----------
* Ladd, A. J. C. (1994). J. Fluid Mech. 271, 285–309.
* Schlichting, H., & Gersten, K. (2017). Boundary-Layer Theory (10th ed.).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import (
    apply_water_entry_boundaries_3d,
    make_tank_wall_mask_3d,
    sphere_mask,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import collide_bgk3d, stream3d
from .turbulence import collide_smagorinsky_bgk3d
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SphereWaterEntryConfig:
    """Configuration for a 3-D sphere water-entry LBM simulation.

    The sphere is held stationary in its own reference frame; the water
    flows upward past it at *v_entry*, which is equivalent to the sphere
    descending through still water at that speed.

    Grid and geometry
    -----------------
    nx, ny : int
        Cross-sectional dimensions of the tank (x and y directions).
        Larger values reduce wall-blockage effects.
    nz : int
        Vertical extent of the tank (z direction, flow direction).
    radius : float
        Sphere radius in lattice units.
    sphere_z_frac : float
        Fractional vertical position of the sphere centre (0 = bottom,
        1 = top).  Default 0.5 places the sphere at mid-height.

    Flow
    ----
    v_entry : float
        Entry (impact) velocity in lattice units (should be ≤ 0.1).
    re : float
        Reynolds number Re = v_entry · 2r / ν.
    n_ramp : int
        Number of steps over which *v_entry* is linearly ramped from 0 to
        its target value.  Set to 0 for an impulsive start.

    Turbulence
    ----------
    smagorinsky_cs : float
        Smagorinsky constant *C_s* (default 0; set > 0 to enable LES).

    Output
    ------
    n_steps, output_interval, output_root, run_name, seed, device, overwrite
        Standard runner parameters.
    """

    # Grid
    nx: int = 48
    ny: int = 48
    nz: int = 96
    # Geometry
    radius: float = 6.0
    sphere_z_frac: float = 0.5
    # Flow
    v_entry: float = 0.05
    re: float = 100.0
    n_ramp: int = 50
    # Turbulence
    smagorinsky_cs: float = 0.0
    # Output
    n_steps: int = 1000
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
        """Kinematic viscosity from Re = v_entry · 2r / ν."""
        return self.v_entry * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is invalid."""
        if self.nx < 16 or self.ny < 16 or self.nz < 16:
            raise ValueError("nx, ny, and nz must be at least 16")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.v_entry <= 0.0 or self.re <= 0.0 or self.radius <= 0.0:
            raise ValueError("v_entry, re, and radius must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"Invalid tau={self.tau:.4f}; increase re or reduce v_entry/radius"
            )
        if not (0.0 < self.sphere_z_frac < 1.0):
            raise ValueError("sphere_z_frac must be in (0, 1)")
        if 2.0 * self.radius >= min(self.nx, self.ny):
            raise ValueError("sphere diameter must be less than min(nx, ny)")
        cz = self.nz * self.sphere_z_frac
        if cz - self.radius < 2 or cz + self.radius > self.nz - 3:
            raise ValueError(
                "sphere must fit inside the domain with at least 2 cells "
                "clearance from the z=0 inlet and z=nz-1 outlet"
            )

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"nx{self.nx}_ny{self.ny}_nz{self.nz}_re{re_label}"
            f"_v{self.v_entry:.3f}_r{self.radius:.1f}_steps{self.n_steps}"
        )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _save_water_entry_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    uz: torch.Tensor,
    obstacle: torch.Tensor,
    ny: int,
) -> None:
    """Save speed and vertical-velocity side-view (xz-plane) snapshots."""
    mid_y = ny // 2
    speed_np = speed[:, mid_y, :].detach().cpu().numpy()   # (nz, nx)
    uz_np = uz[:, mid_y, :].detach().cpu().numpy()          # (nz, nx)
    obs_np = obstacle[:, mid_y, :].detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(speed_np, origin="lower", cmap="viridis")
    axes[0].contour(obs_np, levels=[0.5], colors="white", linewidths=0.8)
    axes[0].set_title(f"Speed |u| – xz side view (step {step})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z  (↑ = upward / flow direction)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    vmax = float(abs(uz_np).max()) or 1e-6
    im1 = axes[1].imshow(uz_np, origin="lower", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
    axes[1].contour(obs_np, levels=[0.5], colors="black", linewidths=0.8)
    axes[1].set_title(f"Vertical velocity uz – xz side view (step {step})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z  (↑ = upward / flow direction)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.savefig(run_dir / f"flow_step_{step:06d}.png", dpi=140)
    plt.close(fig)


def _save_force_plot(
    run_dir: Path,
    force_history: list[tuple[int, float, float, float]],
) -> None:
    """Save drag force Fz and drag coefficient Cd vs simulation step."""
    steps = [r[0] for r in force_history]
    fz_vals = [r[2] for r in force_history]
    cd_vals = [r[3] for r in force_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(steps, fz_vals, linewidth=1.2)
    axes[0].set_xlabel("Simulation step")
    axes[0].set_ylabel("Fz (drag force, lattice units)")
    axes[0].set_title("Drag force vs step")
    axes[0].grid(True, alpha=0.3)

    valid = [(s, c) for s, c in zip(steps, cd_vals) if not math.isnan(c)]
    if valid:
        vs, vc = zip(*valid)
        axes[1].plot(vs, vc, linewidth=1.2)
        axes[1].set_xlabel("Simulation step")
        axes[1].set_ylabel("Cd (drag coefficient)")
        axes[1].set_title("Drag coefficient vs step")
        axes[1].grid(True, alpha=0.3)

    fig.savefig(run_dir / "force_history.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_sphere_water_entry(config: SphereWaterEntryConfig) -> Path:
    """Run the sphere water-entry simulation and save all results.

    Simulation loop (per step *t*):

    1. **Collide** – BGK (or BGK + Smagorinsky LES) at relaxation time *tau*.
    2. **Stream**  – periodic gather streaming.
    3. **Force diagnostics** – Ladd momentum-exchange drag before bounce-back.
    4. **Boundaries** – ramped velocity inlet (z=0) + pressure outlet
       (z=nz-1) + lateral wall and sphere bounce-back.

    Args:
        config: Validated :class:`SphereWaterEntryConfig` instance.

    Returns:
        Path to the run output directory.
    """
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "sphere_water_entry",
        config.resolved_run_name(), config.overwrite,
    )

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    # ------------------------------------------------------------------
    # Geometry: sphere at the horizontal centre, sphere_z_frac up from bottom
    # ------------------------------------------------------------------
    cx = config.nx * 0.5
    cy = config.ny * 0.5
    cz = config.nz * config.sphere_z_frac

    obstacle = sphere_mask(
        config.nx, config.ny, config.nz, cx, cy, cz, config.radius, device=device
    )
    wall_mask = make_tank_wall_mask_3d(
        config.nz, config.ny, config.nx, obstacle, device=device
    )

    # ------------------------------------------------------------------
    # Initial conditions: quiescent water at rest
    # ------------------------------------------------------------------
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.zeros_like(rho0)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0[~obstacle].sum().item())

    # Reference area for Cd: sphere cross-sectional area (π r²)
    ref_area = math.pi * config.radius ** 2
    use_smagorinsky = config.smagorinsky_cs > 0.0

    print(
        "Running D3Q19 sphere water-entry simulation\n"
        f"  device={device}  NX={config.nx} NY={config.ny} NZ={config.nz}\n"
        f"  radius={config.radius}  sphere_z_frac={config.sphere_z_frac}"
        f"  cz={cz:.1f}\n"
        f"  v_entry={config.v_entry}  Re={config.re:.1f}  "
        f"tau={config.tau:.4f}  n_ramp={config.n_ramp}\n"
        f"  Cs={config.smagorinsky_cs}  "
        f"steps={config.n_steps}  output_interval={config.output_interval}"
    )
    print(f"Run directory: {run_dir}")

    diagnostics: list[dict[str, object]] = []
    force_history: list[tuple[int, float, float, float]] = []

    for step in range(1, config.n_steps + 1):
        # Linearly ramp entry velocity from 0 to v_entry over n_ramp steps
        v_actual = (
            config.v_entry * min(step / config.n_ramp, 1.0)
            if config.n_ramp > 0
            else config.v_entry
        )

        # 1. Collision
        if use_smagorinsky:
            f = collide_smagorinsky_bgk3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        else:
            f = collide_bgk3d(f, tau=config.tau)

        # 2. Streaming
        f = stream3d(f)

        # 3. Force diagnostics BEFORE bounce-back (Ladd 1994)
        fx, fy, fz = compute_obstacle_forces_3d(f, obstacle)

        # 4. Boundary conditions
        f = apply_water_entry_boundaries_3d(
            f, v_entry=v_actual, wall_mask=wall_mask, obstacle_mask=obstacle
        )

        # Drag coefficient (undefined during the motionless initial state)
        dyn_q = 0.5 * 1.0 * v_actual ** 2 * ref_area
        cd = float(fz) / dyn_q if dyn_q > 1e-14 else float("nan")
        force_history.append((step, v_actual, float(fz), cd))

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            uz = uz.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
            mass = float(rho[~obstacle].sum().item())

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho[~obstacle].mean().item()),
            )
            diag_entry: dict[str, object] = {
                **asdict(point),
                "v_entry_actual": v_actual,
                "fz": float(fz),
                "cd": cd,
            }
            diagnostics.append(diag_entry)
            print(
                f"step={point.step:5d}  mass={point.mass:.5f}  "
                f"drift={point.mass_drift:+.5f}  max|u|={point.max_speed:.5f}  "
                f"v={v_actual:.5f}  Fz={float(fz):.5f}  "
                f"Cd={'nan' if math.isnan(cd) else f'{cd:.4f}'}"
            )
            _save_water_entry_snapshot(run_dir, step, speed, uz, obstacle, config.ny)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "v_entry_actual", "fz", "cd"])
        for row in force_history:
            writer.writerow(row)

    _save_force_plot(run_dir, force_history)

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}")
    return run_dir


__all__ = [
    "SphereWaterEntryConfig",
    "run_sphere_water_entry",
]
