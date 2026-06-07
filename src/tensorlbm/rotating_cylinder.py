"""Rotating cylinder (Magnus effect) 2D LBM runner.

A circular cylinder spins about its own axis at angular velocity ω while a
uniform free stream u∞ flows past it. The resulting asymmetric pressure
distribution produces a transverse (lift) force — the Magnus effect.

The non-dimensional spin ratio α = ω R / u∞ controls the wake regime:

- α ≲ 0.5  : nearly symmetric Kármán street, small mean Cl.
- 0.5 ≲ α ≲ 1.9 : suppressed vortex shedding, growing lift coefficient.
- α ≳ 2    : steady wake, Cl saturates at large values (Mittal & Kumar 2003).

This module reuses the D2Q9 + BGK + Zou/He channel infrastructure already
implemented in :mod:`tensorlbm.cylinder_flow` and adds a **moving-wall
bounce-back** boundary condition (Ladd 1994) so that the obstacle surface
carries the prescribed tangential velocity ``u_w = ω × r``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from .config_io import load_config_json, save_config_json
from .d2q9 import C, W, equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, collide_mrt, correct_mass, stream
from .utils import (
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)


@dataclass(frozen=True)
class RotatingCylinderConfig:
    """Configuration for the rotating-cylinder (Magnus) simulation."""

    nx: int = 320
    ny: int = 100
    u_in: float = 0.08
    re: float = 100.0
    radius: float = 12.0
    spin_ratio: float = 1.0
    """Non-dimensional spin α = ω R / u∞. Positive ⇒ counter-clockwise spin."""
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

    @property
    def omega(self) -> float:
        """Angular velocity ω = α u∞ / R (lattice units, radians per step)."""
        return self.spin_ratio * self.u_in / self.radius

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
        return (
            f"nx{self.nx}_ny{self.ny}_re{re_label}_alpha{self.spin_ratio:g}"
            f"_steps{self.n_steps}"
        )

    def save(self, path: str | Path) -> Path:
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> RotatingCylinderConfig:
        return load_config_json(cls, path)


def rotating_wall_velocity(
    obstacle_mask: torch.Tensor,
    cx: float,
    cy: float,
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid-body rotation velocity field u_w = ω × r at every cell.

    Returns ``(ux_w, uy_w)`` of shape ``(ny, nx)``. Only the values on
    *obstacle* cells are physically meaningful; off-obstacle entries are
    still computed but should be masked by the caller when applied.

    For a positive *omega* the velocity field is counter-clockwise:
    ``ux_w = -ω (y − cy)``, ``uy_w = +ω (x − cx)``.
    """
    device = obstacle_mask.device
    ny, nx = obstacle_mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux_w = -omega * (yy - cy)
    uy_w = omega * (xx - cx)
    return ux_w, uy_w


def moving_wall_bounce_back(
    f: torch.Tensor,
    mask: torch.Tensor,
    ux_w: torch.Tensor,
    uy_w: torch.Tensor,
) -> torch.Tensor:
    """Ladd (1994) moving-wall bounce-back for D2Q9.

    On every cell flagged by *mask* (the solid surface) the bounce-back rule
    is augmented by a momentum-source term that imposes the prescribed wall
    velocity ``(ux_w, uy_w)``:

    .. math::
        f_{\\bar i}(x) = f_i(x)
        - 2\\,w_i\\,\\rho_w\\,\\frac{c_i\\cdot u_w}{c_s^2}

    Here ``ρ_w`` is taken as the local density (consistent with the LB
    incompressibility hypothesis at low Mach number). The formula reduces to
    the standard bounce-back when ``u_w = 0``.

    Args:
        f:    Distribution tensor of shape ``(9, ny, nx)``.
        mask: Boolean obstacle mask of shape ``(ny, nx)``.
        ux_w: Wall x-velocity field of shape ``(ny, nx)``.
        uy_w: Wall y-velocity field of shape ``(ny, nx)``.

    Returns:
        Updated distribution tensor.
    """
    device = f.device
    c = C.to(device).to(f.dtype)
    w = W.to(device).to(f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w_view = w.view(9, 1, 1)

    rho = f.sum(dim=0)

    # First: standard bounce-back (reverses all directions on solid cells)
    f_bb = bounce_back_cells(f, mask)

    # Then add the moving-wall momentum-source correction.
    # The sign convention: the corrected outgoing population (after reflection
    # in direction ī) is f_i(x) − 2 w_i ρ c_i·u_w / c_s^2. Since bounce_back
    # already wrote f[ī] := f[i], we subtract 2 w_i ρ c_i·u_w / c_s^2 in the
    # *original* direction i, which after the i↔ī swap means subtracting
    # 2 w_ī ρ c_ī·u_w / c_s^2 in direction ī. Using w_i = w_ī and c_ī = −c_i
    # the correction in direction ī is +2 w_i ρ c_i·u_w / c_s^2.
    cu_w = cx * ux_w.unsqueeze(0) + cy * uy_w.unsqueeze(0)
    correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0  # 1/c_s^2 = 3

    return torch.where(mask.unsqueeze(0), f_bb + correction, f_bb)


def run_rotating_cylinder(config: RotatingCylinderConfig) -> Path:
    """Run a 2D rotating-cylinder (Magnus) simulation and write outputs."""
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "rotating_cylinder",
        config.resolved_run_name(),
        config.overwrite,
    )

    cx_obs, cy_obs = config.nx * 0.25, config.ny * 0.5
    obstacle = cylinder_mask(config.nx, config.ny, cx_obs, cy_obs, config.radius, device=device)
    wall_mask = make_channel_wall_mask(config.ny, config.nx, obstacle, device=device)

    ux_w, uy_w = rotating_wall_velocity(obstacle, cx_obs, cy_obs, config.omega)

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = torch.full((config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros((config.ny, config.nx), device=device)
    ux0[obstacle] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    initial_mass = float(rho0.sum().item())
    diameter = 2.0 * config.radius
    dyn_pressure = 0.5 * config.u_in**2 * diameter

    diagnostics: list[dict[str, float | int]] = []
    cd_series: list[float] = []
    cl_series: list[float] = []

    logger.info(
        "Rotating cylinder device=%s NX=%s NY=%s tau=%.4f alpha=%.3f "
        "omega=%.5f steps=%s",
        device,
        config.nx,
        config.ny,
        config.tau,
        config.spin_ratio,
        config.omega,
        config.n_steps,
    )

    _collide_fn = collide_mrt if config.tau < 0.60 else collide_bgk

    for step in range(1, config.n_steps + 1):
        f = _collide_fn(f, tau=config.tau)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, obstacle)
        # Standard channel BC for inlet/outlet/walls (without obstacle), then
        # moving-wall bounce-back on the rotating cylinder surface.
        f = apply_simple_channel_boundaries(
            f,
            u_in=config.u_in,
            wall_mask=wall_mask,
            obstacle_mask=torch.zeros_like(obstacle),  # skip obstacle here
        )
        f = moving_wall_bounce_back(f, obstacle, ux_w, uy_w)

        cd = float(fx.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cl = float(fy.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cd_series.append(cd)
        cl_series.append(cl)

        if step % config.output_interval == 0:
            f = correct_mass(f, initial_mass)
            rho, ux, uy = macroscopic(f)
            mass = float(rho.sum().item())
            diagnostics.append(
                {
                    "step": step,
                    "cd": cd,
                    "cl": cl,
                    "mass": mass,
                    "mass_drift": mass - initial_mass,
                    "mean_rho": float(rho.mean().item()),
                }
            )
            logger.info(
                "step=%5d Cd=%.4f Cl=%.4f mass_drift=%+.3e",
                step,
                cd,
                cl,
                mass - initial_mass,
            )

    # Mean coefficients over the last half of the run (post-transient).
    half = len(cd_series) // 2
    cd_mean = sum(cd_series[half:]) / max(1, len(cd_series) - half)
    cl_mean = sum(cl_series[half:]) / max(1, len(cl_series) - half)
    logger.info("Mean Cd=%.4f, Mean Cl=%.4f (last half)", cd_mean, cl_mean)

    metadata: dict[str, object] = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
        },
        "derived": {
            "nu": config.nu,
            "tau": config.tau,
            "omega": config.omega,
        },
        "diagnostics": diagnostics,
        "cd_mean": cd_mean,
        "cl_mean": cl_mean,
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cl"])
        for i, (cd, cl) in enumerate(zip(cd_series, cl_series, strict=False), start=1):
            writer.writerow([i, cd, cl])

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir


__all__ = [
    "RotatingCylinderConfig",
    "moving_wall_bounce_back",
    "rotating_wall_velocity",
    "run_rotating_cylinder",
]
