"""3D sphere channel-flow simulation using the D3Q27 lattice.

Provides a higher-fidelity alternative to :mod:`tensorlbm.sphere_flow`
(D3Q19) using the 27-direction D3Q27 lattice with optional MRT or
Smagorinsky LES collision operators.

Exported symbols
----------------
- :class:`SphereFlowD3Q27Config` – simulation configuration
- :func:`run_sphere_flow_d3q27`  – runner
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries3d import make_channel_wall_mask_3d, sphere_mask
from .boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    compute_obstacle_forces_27,
    make_channel_wall_mask_27,
)
from .d3q27 import (
    collide_bgk27,
    collide_mrt27,
    collide_smagorinsky_bgk27,
    collide_smagorinsky_mrt27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SphereFlowD3Q27Config:
    """Configuration for a 3D D3Q27 sphere channel-flow simulation.

    Parameters
    ----------
    collision : str
        Collision operator: ``"bgk"``, ``"mrt"``, ``"smagorinsky_bgk"``,
        or ``"smagorinsky_mrt"`` (default).
    """

    nx: int = 120
    ny: int = 60
    nz: int = 60
    u_in: float = 0.06
    re: float = 50.0
    radius: float = 8.0
    collision: str = "bgk"
    smagorinsky_cs: float = 0.1
    n_steps: int = 500
    output_interval: int = 100
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())
        object.__setattr__(self, "collision", self.collision.lower())

    @property
    def nu(self) -> float:
        return self.u_in * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, nz must be at least 16, 8, 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0 or self.radius <= 0.0:
            raise ValueError("u_in, re, and radius must be > 0")
        if self.tau <= 0.5:
            raise ValueError(f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/radius")
        valid = {"bgk", "mrt", "smagorinsky_bgk", "smagorinsky_mrt"}
        if self.collision not in valid:
            raise ValueError(f"collision must be one of {valid}")

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"d3q27_nx{self.nx}_ny{self.ny}_nz{self.nz}_re{re_label}"
            f"_{self.collision}_uin{self.u_in:.3f}_steps{self.n_steps}"
        )

    def save(self, path: "Path | str") -> None:
        from .config_io import save_config_json
        save_config_json(self, path)

    @classmethod
    def load(cls, path: "Path | str") -> "SphereFlowD3Q27Config":
        from .config_io import load_config_json
        return load_config_json(cls, path)


# ---------------------------------------------------------------------------
# Collision selector
# ---------------------------------------------------------------------------

def _collide(f: torch.Tensor, config: SphereFlowD3Q27Config) -> torch.Tensor:
    tau = config.tau
    C_s = config.smagorinsky_cs
    if config.collision == "bgk":
        return collide_bgk27(f, tau)
    if config.collision == "mrt":
        return collide_mrt27(f, tau)
    if config.collision == "smagorinsky_bgk":
        return collide_smagorinsky_bgk27(f, tau, C_s)
    # smagorinsky_mrt
    return collide_smagorinsky_mrt27(f, tau, C_s)


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

def _save_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
    nz: int,
) -> None:
    mid_z = nz // 2
    speed_np = speed[mid_z].detach().cpu().numpy()
    obs_np = obstacle[mid_z].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = ax.imshow(speed_np, origin="lower", cmap="viridis")
    ax.contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    ax.set_title(f"D3Q27 speed – mid-z (step {step})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(run_dir / f"flow_step_{step:06d}.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sphere_flow_d3q27(config: SphereFlowD3Q27Config) -> Path:
    """Run a 3D D3Q27 channel flow past a sphere.

    Args:
        config: Validated :class:`SphereFlowD3Q27Config`.

    Returns:
        Path to the run output directory.
    """
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "sphere_flow_d3q27",
        config.resolved_run_name(), config.overwrite
    )

    cx = config.nx * 0.25
    cy = config.ny * 0.5
    cz = config.nz * 0.5
    obstacle = sphere_mask(config.nx, config.ny, config.nz, cx, cy, cz, config.radius, device=device)
    wall_mask = make_channel_wall_mask_27(
        config.nz, config.ny, config.nx, obstacle, device=device
    )

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[obstacle] = 0.0
    f = equilibrium27(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())

    # Force coefficient reference
    diameter = 2.0 * config.radius
    dyn_pressure = 0.5 * 1.0 * config.u_in ** 2 * diameter ** 2

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }
    diagnostics: list[dict[str, object]] = []

    print(
        f"Running D3Q27 sphere flow  device={device}  "
        f"NX={config.nx} NY={config.ny} NZ={config.nz}  "
        f"tau={config.tau:.4f}  collision={config.collision}  "
        f"steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    for step in range(1, config.n_steps + 1):
        f = _collide(f, config)
        f = stream27(f)
        fx, fy, fz = compute_obstacle_forces_27(f, obstacle)
        f = apply_zou_he_channel_boundaries_27(
            f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=obstacle
        )

        cd = float(fx) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

        if step % config.output_interval == 0 or step == config.n_steps:
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
            diag: dict[str, object] = {**asdict(point), "cd": cd}
            diagnostics.append(diag)
            print(
                f"step={step:5d}  mass={mass:.5f}  drift={mass-initial_mass:+.5f}  "
                f"max|u|={point.max_speed:.5f}  Cd={cd:.4f}"
            )
            _save_snapshot(run_dir, step, speed, obstacle, config.nz)

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}")
    return run_dir


__all__ = ["SphereFlowD3Q27Config", "run_sphere_flow_d3q27"]
