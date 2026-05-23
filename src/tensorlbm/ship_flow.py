from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
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
from .solver3d import collide_bgk3d, stream3d
from .turbulence import collide_smagorinsky_bgk3d
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShipHullFlowConfig:
    """Configuration for a 3D Wigley hull channel-flow simulation.

    All quantities are expressed in lattice units unless noted.

    The ship is centred at (ix_center, iy_center = ny//2) and placed with its
    keel at iz_keel (default nz//4).  The channel walls enclose the domain on
    all four cross-section faces (±y, ±z) and the inlet/outlet are the x=0
    and x=nx−1 planes.

    Attributes:
        nx: Streamwise grid points.
        ny: Transverse (crossflow) grid points.
        nz: Vertical grid points.
        u_in: Inlet x-velocity (lattice units).
        re: Reynolds number  Re = u_in · length_lbm / ν.
        length_lbm: Ship length in lattice units.
        beam_lbm: Maximum beam (breadth) in lattice units.
        draft_lbm: Draft (keel-to-waterline depth) in lattice units.
        C_s: Smagorinsky constant  (0 → pure BGK; typical 0.1–0.18).
        n_steps: Total simulation steps.
        output_interval: Steps between diagnostics and PNG output.
        output_root: Root directory for outputs.
        run_name: Optional identifier for the run directory.
        seed: Random seed for initialisation.
        device: ``"cpu"`` or ``"cuda"``.
        overwrite: Remove existing run directory when ``True``.
    """

    nx: int = 200
    ny: int = 60
    nz: int = 60
    u_in: float = 0.05
    re: float = 500.0
    length_lbm: int = 80
    beam_lbm: int = 12
    draft_lbm: int = 10
    C_s: float = 0.1
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
        return self.u_in * self.length_lbm / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time derived from ν."""
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 32 or self.ny < 16 or self.nz < 16:
            raise ValueError("nx ≥ 32, ny ≥ 16, nz ≥ 16 required")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0:
            raise ValueError("u_in and re must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"tau={self.tau:.4f} ≤ 0.5; increase re or reduce u_in/length_lbm"
            )
        if self.length_lbm >= self.nx - 20:
            raise ValueError("length_lbm too large for grid nx (need nx > length_lbm + 20)")
        if self.beam_lbm >= self.ny - 4:
            raise ValueError("beam_lbm too large for grid ny (need ny > beam_lbm + 4)")
        if self.draft_lbm >= self.nz - 4:
            raise ValueError("draft_lbm too large for grid nz (need nz > draft_lbm + 4)")

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"wigley_nx{self.nx}_ny{self.ny}_nz{self.nz}"
            f"_re{re_label}_uin{self.u_in:.3f}_steps{self.n_steps}"
        )


def _save_ship_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    hull: torch.Tensor,
    nz: int,
    ny: int,
) -> None:
    """Save speed magnitude on the mid-z and mid-y slices as a PNG."""
    mid_z = nz // 2
    mid_y = ny // 2

    speed_np = speed.detach().cpu().numpy()
    hull_mid_z = hull[mid_z].detach().cpu().float().numpy()
    hull_mid_y = hull[:, mid_y, :].detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    im0 = axes[0].imshow(speed_np[mid_z], origin="lower", cmap="viridis")
    axes[0].contour(hull_mid_z, levels=[0.5], colors="white", linewidths=0.8)
    axes[0].set_title(f"Speed – mid-z slice (step {step})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(speed_np[:, mid_y, :], origin="lower", cmap="viridis")
    axes[1].contour(hull_mid_y, levels=[0.5], colors="white", linewidths=0.8)
    axes[1].set_title(f"Speed – mid-y slice (step {step})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.savefig(run_dir / f"flow_step_{step:06d}.png", dpi=120)
    plt.close(fig)


def run_ship_hull_flow(config: ShipHullFlowConfig) -> Path:
    """Run a 3D Wigley hull channel-flow simulation with Smagorinsky LES.

    Pipeline per time step:

    1. Collision – Smagorinsky-BGK (or pure BGK when ``C_s = 0``).
    2. Streaming.
    3. Momentum-exchange force/moment measurement **before** bounce-back.
    4. Boundary conditions – Zou/He inlet + pressure outlet + bounce-back on
       walls and hull.

    Outputs written to ``<output_root>/ship_hull_flow/<run_name>/``:

    * ``run_metadata.json`` – config, derived quantities, diagnostics history.
    * ``forces.csv`` – per-output-step drag, lift and moment time series.
    * ``flow_step_XXXXXX.png`` – speed snapshots on mid-y and mid-z slices.

    Args:
        config: Simulation configuration.

    Returns:
        Path to the output directory.
    """
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "ship_hull_flow", config.resolved_run_name(), config.overwrite
    )

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    # ---- geometry ----
    ix_center = config.nx // 3
    iy_center = config.ny // 2
    iz_keel = config.nz // 4
    iz_waterline = iz_keel + config.draft_lbm

    hull = wigley_hull_mask(
        config.nx, config.ny, config.nz,
        ix_center, iy_center, iz_keel,
        config.length_lbm, config.beam_lbm, config.draft_lbm,
        device=device,
    )
    wall_mask = make_channel_wall_mask_3d(config.nz, config.ny, config.nx, hull, device=device)

    hull_cells = int(hull.sum().item())
    print(f"Wigley hull: {hull_cells} solid cells "
          f"(L={config.length_lbm} B={config.beam_lbm} T={config.draft_lbm})")
    print(f"  ix_center={ix_center}  iy_center={iy_center}  "
          f"iz_keel={iz_keel}  iz_waterline={iz_waterline}")

    # ---- initialisation ----
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[hull] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    diagnostics: list[dict[str, object]] = []

    # Reference quantities for non-dimensional force coefficients
    dyn_pressure = 0.5 * config.u_in**2 * float(config.length_lbm) * float(config.beam_lbm)
    cx_ref = float(ix_center)
    cy_ref = float(iy_center)
    cz_ref = float(iz_waterline)

    print(
        f"Running Wigley hull flow  "
        f"device={device}  NX={config.nx}×NY={config.ny}×NZ={config.nz}  "
        f"Re={config.re}  τ={config.tau:.4f}  C_s={config.C_s}  "
        f"steps={config.n_steps}  output_interval={config.output_interval}"
    )
    print(f"Run directory: {run_dir}")

    for step in range(1, config.n_steps + 1):
        # Collision
        if config.C_s > 0.0:
            f = collide_smagorinsky_bgk3d(f, tau_0=config.tau, C_s=config.C_s)
        else:
            f = collide_bgk3d(f, tau=config.tau)

        # Streaming
        f = stream3d(f)

        # Momentum-exchange forces/moments BEFORE bounce-back
        fx, fy, fz = compute_obstacle_forces_3d(f, hull)
        Mx, My, Mz = compute_obstacle_moments_3d(f, hull, cx_ref, cy_ref, cz_ref)

        # Boundary conditions (Zou/He inlet + pressure outlet + bounce-back)
        f = apply_zou_he_channel_boundaries_3d(
            f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=hull
        )

        # Non-dimensional coefficients
        cd = float(fx) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cl = float(fz) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux = ux.masked_fill(hull, 0.0)
            uy = uy.masked_fill(hull, 0.0)
            uz = uz.masked_fill(hull, 0.0)
            speed = torch.sqrt(ux**2 + uy**2 + uz**2)
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
                "cl": cl,
                "Mx": float(Mx),
                "My": float(My),
                "Mz": float(Mz),
            }
            diagnostics.append(diag_entry)
            print(
                f"step={point.step:5d}  mass={point.mass:.4f}  "
                f"drift={point.mass_drift:+.4f}  max|u|={point.max_speed:.5f}  "
                f"Cd={cd:.4f}  Cl={cl:.4f}  "
                f"Mx={float(Mx):.3f}  My={float(My):.3f}  Mz={float(Mz):.3f}"
            )

            _save_ship_snapshot(run_dir, step, speed, hull, config.nz, config.ny)

    # ---- save forces CSV ----
    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cl", "Mx", "My", "Mz"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cl"], d["Mx"], d["My"], d["Mz"]])

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}")
    return run_dir
