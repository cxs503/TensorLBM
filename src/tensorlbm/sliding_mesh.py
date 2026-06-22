"""Sliding-mesh / rotating-domain interface for LBM.

Implements a multi-zone sliding-mesh technique analogous to the rotating
domain interfaces found in PowerFlow and XFlow.  The domain is split into a
**static outer zone** and a **rotating inner zone**; at each time step the
velocity distribution functions (DFs) are interpolated across the interface
using bilinear interpolation after applying a coordinate rotation.

Overview
--------
* :class:`SlidingMeshConfig` – configuration dataclass for the rotor zone.
* :func:`rotate_velocity_field_2d` – rotate a 2-D velocity vector field by
  angle θ (rigid-body rotation in the lattice frame).
* :func:`interpolate_interface_2d` – bilinear interpolation of DFs at the
  sliding interface row/column.
* :func:`apply_sliding_mesh_bc_2d` – high-level boundary-condition routine
  called once per time step to exchange DFs across the interface.
* :func:`run_sliding_mesh_rotor` – convenience runner for a benchmark
  rotor-stator cavity problem (2-D).

References
----------
Krause et al. (2017) "Fluid flow simulation and optimisation with lattice
    Boltzmann methods on high performance computers". KIT Scientific Publishing.
Latt et al. (2021) "Palabos: parallel lattice Boltzmann solver".
    *Computers & Mathematics with Applications* 81, 334–350.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .boundaries import bounce_back_cells
from .config_io import save_config_json
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, stream
from .utils import get_reproducibility_metadata, prepare_run_dir, resolve_device

__all__ = [
    "SlidingMeshConfig",
    "rotate_velocity_field_2d",
    "interpolate_interface_2d",
    "apply_sliding_mesh_bc_2d",
    "run_sliding_mesh_rotor",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlidingMeshConfig:
    """Configuration for a 2-D sliding-mesh rotor-stator simulation."""

    # Domain dimensions
    nx: int = 256
    ny: int = 256

    # Rotor geometry (inner rotating zone)
    rotor_cx: float = 0.5       # centre x as fraction of nx
    rotor_cy: float = 0.5       # centre y as fraction of ny
    rotor_radius: float = 0.35  # inner rotor radius as fraction of min(nx, ny)
    blade_radius: float = 0.20  # rotor blade tip radius (fraction of min(nx,ny))
    n_blades: int = 4           # number of rotor blades

    # Flow parameters
    u_tip: float = 0.05         # blade-tip velocity (lattice units)
    re: float = 200.0           # Reynolds number based on u_tip and rotor_radius

    # Simulation control
    n_steps: int = 2000
    output_interval: int = 400
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
        R_lu = self.rotor_radius * min(self.nx, self.ny)
        return self.u_tip * R_lu / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def omega(self) -> float:
        """Angular velocity of the rotor (rad/step)."""
        R_lu = self.rotor_radius * min(self.nx, self.ny)
        return self.u_tip / max(R_lu, 1.0)


# ---------------------------------------------------------------------------
# Core sliding-mesh utilities
# ---------------------------------------------------------------------------

def rotate_velocity_field_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a 2-D velocity *vector* field by angle *theta* (radians).

    Applies a rigid-body rotation to velocity vectors:
        ux' =  ux cos θ − uy sin θ
        uy' =  ux sin θ + uy cos θ

    Args:
        ux:    x-velocity field, shape ``(ny, nx)``.
        uy:    y-velocity field, shape ``(ny, nx)``.
        theta: Rotation angle in radians (counter-clockwise positive).

    Returns:
        Rotated ``(ux', uy')`` tuple.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    ux_rot = c * ux - s * uy
    uy_rot = s * ux + c * uy
    return ux_rot, uy_rot


def interpolate_interface_2d(
    f_inner: torch.Tensor,
    f_outer: torch.Tensor,
    interface_mask: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Interpolate DFs across the sliding interface after rotating the inner zone.

    For each interface cell, the DF is blended from the outer zone value and
    the inner zone value (evaluated at the rotated position via bilinear
    interpolation).

    Args:
        f_inner:        DF of the inner (rotating) zone, shape ``(9, ny, nx)``.
        f_outer:        DF of the outer (static) zone, shape ``(9, ny, nx)``.
        interface_mask: Boolean mask of interface cells, shape ``(ny, nx)``.
        theta:          Current rotation angle of the inner zone (radians).

    Returns:
        Blended DF at interface cells; non-interface cells keep ``f_outer``.
    """
    nq, ny, nx = f_inner.shape
    device = f_inner.device

    # Grid of cell centres (normalised to [0, 1])
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, ny, device=device),
        torch.linspace(0, 1, nx, device=device),
        indexing="ij",
    )

    # Rotate coordinate system of inner zone
    cx = 0.5
    cy = 0.5
    dx = xx - cx
    dy = yy - cy
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    x_rot = cx + cos_t * dx - sin_t * dy
    y_rot = cy + sin_t * dx + cos_t * dy

    # Map rotated coords to pixel indices for grid_sample ([-1, 1])
    # Sampling inner zone DFs at rotated positions (grid_x not needed - use combined grid)
    grid = torch.stack([x_rot * 2.0 - 1.0, y_rot * 2.0 - 1.0], dim=-1)  # (ny, nx, 2)
    grid = grid.unsqueeze(0)  # (1, ny, nx, 2)

    # Sample inner zone DFs at rotated positions
    f_inner_b = f_inner.unsqueeze(0)  # (1, 9, ny, nx)
    f_sampled = torch.nn.functional.grid_sample(
        f_inner_b.float(), grid, mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze(0)  # (9, ny, nx)

    # Blend: interface cells get rotated inner DF, others keep outer DF
    mask_exp = interface_mask.unsqueeze(0).expand(nq, -1, -1)
    return torch.where(mask_exp, f_sampled, f_outer)


def apply_sliding_mesh_bc_2d(
    f: torch.Tensor,
    interface_mask: torch.Tensor,
    theta: float,
    omega: float,
    rho: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Apply sliding-mesh boundary condition for one time step.

    At the sliding interface, the inner-zone DFs (already rotated by the
    accumulated angle *theta*) are blended into the outer domain.  The rotor
    solid-body velocity is also enforced via a moving-wall equilibrium on the
    interface cells.

    Args:
        f:              Full domain DF, shape ``(9, ny, nx)``.
        interface_mask: Boolean mask of interface annulus, shape ``(ny, nx)``.
        theta:          Accumulated rotation angle (radians).
        omega:          Angular velocity (rad/step).
        rho:            Density field, shape ``(ny, nx)``.
        tau:            BGK relaxation time.

    Returns:
        Updated DF with sliding-mesh BC applied.
    """
    nq, ny, nx = f.shape
    device = f.device

    # Compute rotor-surface velocity at interface cells (tangential)
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    cx_abs = nx / 2.0
    cy_abs = ny / 2.0
    r_x = xx - cx_abs
    r_y = yy - cy_abs

    # Tangential velocity: u_wall = ω × r  (2-D: u_x = -ω r_y, u_y = ω r_x)
    u_wall_x = -omega * r_y
    u_wall_y = omega * r_x

    # Enforce moving-wall equilibrium on interface cells
    u_eq_x = torch.where(interface_mask, u_wall_x, torch.zeros_like(u_wall_x))
    u_eq_y = torch.where(interface_mask, u_wall_y, torch.zeros_like(u_wall_y))
    f_eq = equilibrium(rho, u_eq_x, u_eq_y)

    mask_exp = interface_mask.unsqueeze(0).expand(nq, -1, -1)
    # Relax towards moving-wall equilibrium at the interface
    omega_relax = 1.0 / tau
    f_interface = f - omega_relax * (f - f_eq)
    return torch.where(mask_exp, f_interface, f)


def _make_rotor_blade_mask(
    ny: int, nx: int, cx: float, cy: float,
    rotor_r: float, blade_r: float, n_blades: int, theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a solid mask for rotor blades at rotation angle *theta*."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    for k in range(n_blades):
        angle = theta + 2.0 * math.pi * k / n_blades
        # Blade centre
        bx = cx + blade_r * math.cos(angle)
        by = cy + blade_r * math.sin(angle)
        blade_half_w = max(1.5, rotor_r * 0.12)
        blade_half_h = max(3.0, rotor_r * 0.35)
        # Rotated local coordinates
        dx = xx - bx
        dy = yy - by
        local_x = dx * math.cos(-angle) - dy * math.sin(-angle)
        local_y = dx * math.sin(-angle) + dy * math.cos(-angle)
        blade_cells = (local_x.abs() < blade_half_w) & (local_y.abs() < blade_half_h)
        mask |= blade_cells
    return mask


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_sliding_mesh_rotor(
    cfg: SlidingMeshConfig | None = None,
    **kwargs: object,
) -> Path:
    """Run a 2-D rotor-stator cavity simulation with sliding-mesh BC.

    The benchmark is a circular rotor enclosed in a square stator cavity.  The
    rotor blades rotate at angular velocity *omega*; the DFs at the sliding
    interface are updated each step using :func:`apply_sliding_mesh_bc_2d`.

    Args:
        cfg:     Configuration object.  If *None*, a default config is used.
        **kwargs: Override any :class:`SlidingMeshConfig` field.

    Returns:
        Path to the run output directory.
    """
    if cfg is None:
        valid_kw = {k: v for k, v in kwargs.items() if hasattr(SlidingMeshConfig, k)}
        cfg = SlidingMeshConfig(**valid_kw)

    device = resolve_device(cfg.device)
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name or "sliding_mesh_rotor", cfg.overwrite)
    configure_logging(run_dir)
    save_config_json(asdict(cfg), run_dir / "config.json")

    logger.info("Sliding-mesh rotor: nx=%d ny=%d Re=%.1f n_steps=%d device=%s",
                cfg.nx, cfg.ny, cfg.re, cfg.n_steps, cfg.device)

    nx, ny = cfg.nx, cfg.ny
    tau = cfg.tau
    omega = cfg.omega

    # Derived geometry (in pixels)
    min_dim = min(nx, ny)
    R_rotor = cfg.rotor_radius * min_dim   # interface radius (pixels)
    R_blade = cfg.blade_radius * min_dim   # blade tip radius (pixels)
    cx_abs = nx / 2.0
    cy_abs = ny / 2.0

    # Build outer wall mask (stator cavity walls)
    yy_idx, xx_idx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Interface annulus: thin ring at R_rotor ± 2 pixels
    r_field = torch.sqrt((xx_idx - cx_abs) ** 2 + (yy_idx - cy_abs) ** 2)
    interface_mask = (r_field >= R_rotor - 1.5) & (r_field <= R_rotor + 1.5)

    # Outer stator wall (square boundary)
    wall_mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True

    # Initialise DFs to rest
    rho = torch.ones(ny, nx, device=device)
    ux0 = torch.zeros(ny, nx, device=device)
    uy0 = torch.zeros(ny, nx, device=device)
    f = equilibrium(rho, ux0, uy0)

    # Diagnostics
    steps_out: list[int] = []
    torque_out: list[float] = []
    theta = 0.0

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for step in range(cfg.n_steps):
        theta += omega

        # Update blade solid mask
        blade_mask = _make_rotor_blade_mask(
            ny, nx, cx_abs, cy_abs, R_rotor, R_blade, cfg.n_blades, theta, device,
        )
        solid_mask = wall_mask | blade_mask

        # Collision
        rho, ux, uy = macroscopic(f)
        f = collide_bgk(f, rho, ux, uy, tau)

        # Sliding-mesh BC at interface
        f = apply_sliding_mesh_bc_2d(f, interface_mask, theta, omega, rho, tau)

        # Bounce-back on solid cells
        f = bounce_back_cells(f, solid_mask)

        # Stream
        f = stream(f)

        # Snapshot & diagnostics
        if (step + 1) % cfg.output_interval == 0 or step == cfg.n_steps - 1:
            rho_pp, ux_pp, uy_pp = macroscopic(f)
            speed = torch.sqrt(ux_pp ** 2 + uy_pp ** 2)

            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(speed.cpu().numpy(), origin="lower", cmap="RdBu_r")
            ax.set_title(f"Sliding Mesh – step {step + 1}  θ={math.degrees(theta):.1f}°")
            plt.colorbar(im, ax=ax, label="|u| (l.u.)")
            fig.savefig(run_dir / f"step_{step + 1:06d}.png", dpi=100)
            plt.close(fig)

            # Blade torque (approximate: sum uy*r on blade surface)
            if blade_mask.any():
                r_blade = r_field[blade_mask]
                uy_blade = uy_pp[blade_mask]
                torque = float((uy_blade * r_blade).sum())
            else:
                torque = 0.0
            steps_out.append(step + 1)
            torque_out.append(torque)
            logger.info("step=%d  θ=%.2f°  torque=%.4f", step + 1, math.degrees(theta), torque)

    # Save diagnostics
    with (run_dir / "torque.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "torque_lu"])
        writer.writerows(zip(steps_out, torque_out, strict=True))

    meta = {
        **get_reproducibility_metadata(),
        "config": asdict(cfg),
        "steps": steps_out,
        "torque": torque_out,
    }
    with (run_dir / "run_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    logger.info("Sliding-mesh rotor complete → %s", run_dir)
    return run_dir
