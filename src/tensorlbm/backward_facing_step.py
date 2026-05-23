"""Backward-facing step: 2D CFD benchmark.

A flow enters a channel and expands suddenly at a step.  The benchmark
quantity is the *reattachment length* X_r / h, where h is the step height,
as a function of Reynolds number.  Widely-used reference data:
Armaly et al. (1983), Erturk (2008).

The channel geometry (BFS):

    ┌──────────────────────────────────────────────────┐  ← y = ny-1
    │   inlet          step                            │
    │ ▶▶▶▶▶▶▶▶▶▶▶▶▶▶│                               │
    │                  │  ← expansion (step height h)  │
    └──────────────────┘──────────────────────────────┘  ← y = 0
                       ↑
                    x = nx_in  (step face)

- Inlet channel: y ∈ [h, ny-1], x ∈ [0, nx_in-1]
- Downstream: y ∈ [0, ny-1], x ∈ [nx_in, nx-1]
- Step wall: y ∈ [0, h-1], x ∈ [0, nx_in-1]

Exported symbols
----------------
- :class:`BackwardFacingStepConfig`  – simulation configuration
- :func:`run_backward_facing_step`   – runner
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .boundaries import bounce_back_cells
from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device


@dataclass(frozen=True)
class BackwardFacingStepConfig:
    """Configuration for a 2D backward-facing step simulation.

    Parameters
    ----------
    nx : int
        Total domain length in lattice cells.
    ny : int
        Total channel height in lattice cells.
    step_height : int
        Step height h in lattice cells (≥ 1, < ny//2).
    nx_in : int
        x-position of the step face (cells 0..nx_in-1 form the inlet channel).
    u_in : float
        Mean inlet velocity.
    re : float
        Reynolds number Re = u_in · (2·h) / ν  (hydraulic diameter = 2h).
    """

    nx: int = 200
    ny: int = 40
    step_height: int = 10
    nx_in: int = 30
    u_in: float = 0.05
    re: float = 100.0
    n_steps: int = 8000
    output_interval: int = 1000
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
        """Kinematic viscosity from Re and hydraulic diameter 2h."""
        return self.u_in * 2.0 * self.step_height / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 32 or self.ny < 8:
            raise ValueError("nx >= 32 and ny >= 8 required")
        if self.step_height < 1 or self.step_height >= self.ny // 2:
            raise ValueError("step_height must be >= 1 and < ny//2")
        if self.nx_in < 2 or self.nx_in >= self.nx - 2:
            raise ValueError("nx_in must be in [2, nx-2)")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0:
            raise ValueError("u_in and re must be > 0")
        if self.tau <= 0.5:
            raise ValueError(f"Invalid tau={self.tau:.4f}; decrease re or u_in")

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"nx{self.nx}_ny{self.ny}_h{self.step_height}_re{re_label}"
            f"_steps{self.n_steps}"
        )

    def save(self, path: Path | str) -> None:
        from .config_io import save_config_json
        save_config_json(self, path)

    @classmethod
    def load(cls, path: Path | str) -> BackwardFacingStepConfig:
        from .config_io import load_config_json
        return load_config_json(cls, path)


def _build_bfs_masks(
    nx: int,
    ny: int,
    step_height: int,
    nx_in: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (obstacle_mask, wall_mask) for the backward-facing step.

    The 'obstacle' is the step block: y ∈ [0, h-1], x ∈ [0, nx_in-1].
    The wall includes top/bottom channel boundaries plus the step face.
    """
    obstacle = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    obstacle[:step_height, :nx_in] = True  # step block

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True     # bottom wall
    wall[-1, :] = True    # top wall
    wall[obstacle] = False  # obstacle cells handled separately
    return obstacle, wall


def _parabolic_inlet_velocity(ny: int, step_height: int, u_max: float, device: torch.device) -> torch.Tensor:
    """Parabolic (Poiseuille) inlet profile for the inlet sub-channel.

    The inlet channel spans y ∈ [step_height, ny-1] (height H = ny - h - 1).
    """
    H = ny - step_height - 1  # inner channel height
    if H <= 0:
        return torch.full((ny,), u_max, device=device)
    ux = torch.zeros(ny, device=device)
    for j in range(step_height, ny):
        y_local = float(j - step_height)
        ux[j] = 4.0 * u_max * y_local * (H - y_local) / (H * H)
    return ux


def run_backward_facing_step(config: BackwardFacingStepConfig) -> Path:
    """Run a 2D backward-facing step simulation.

    Args:
        config: Validated :class:`BackwardFacingStepConfig`.

    Returns:
        Path to the run output directory.
    """
    config.validate()
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "backward_facing_step",
        config.resolved_run_name(), config.overwrite
    )

    obstacle, wall = _build_bfs_masks(
        config.nx, config.ny, config.step_height, config.nx_in, device
    )

    # Parabolic inlet profile
    ux_profile = _parabolic_inlet_velocity(
        config.ny, config.step_height, config.u_in, device
    )

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = ux_profile.unsqueeze(1).expand(config.ny, config.nx).clone()
    ux0[obstacle] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)

    initial_mass = float(rho0.sum().item())
    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    diagnostics: list[dict[str, object]] = []
    reattachment_length = 0.0

    print(
        f"Running 2D backward-facing step  NX={config.nx}  NY={config.ny}  "
        f"h={config.step_height}  Re={config.re:.0f}  tau={config.tau:.4f}  "
        f"steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    for step in range(1, config.n_steps + 1):
        f = collide_bgk(f, tau=config.tau)
        f = stream(f)

        # Zou/He inlet on the inlet sub-channel (x=0, y ≥ step_height)
        # Apply as uniform ux at inlet: simple approach
        rho_in, ux_in_cur, uy_in_cur = macroscopic(f)
        ux_in_cur[:, 0] = ux_profile
        uy_in_cur[:, 0] = 0.0
        rho_in[:, 0] = rho_in[:, 1]
        feq_in = equilibrium(
            rho_in[:, 0:1], ux_in_cur[:, 0:1], uy_in_cur[:, 0:1]
        )
        f[:, :, 0] = feq_in[:, :, 0]

        # Zero-gradient outlet
        f[:, :, -1] = f[:, :, -2]

        # Bounce-back on walls and obstacle
        f = bounce_back_cells(f, wall)
        f = bounce_back_cells(f, obstacle)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle, 0.0)
            mass = float(rho.sum().item())

            # Estimate reattachment length: first x where ux at y=step_height+1 > 0
            row = ux[config.step_height + 1, config.nx_in:]
            pos_idx = (row > 0.0).nonzero(as_tuple=False)
            reattachment_length = float(pos_idx[0].item()) if len(pos_idx) > 0 else 0.0

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(torch.sqrt(ux * ux + uy * uy).max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diag: dict[str, object] = {**asdict(point), "reattachment_length": reattachment_length}
            diagnostics.append(diag)
            print(
                f"step={step:5d}  mass={mass:.5f}  drift={mass-initial_mass:+.5f}  "
                f"max|u|={point.max_speed:.5f}  Xr/h={reattachment_length / max(config.step_height, 1):.3f}"
            )

    metadata["diagnostics"] = diagnostics
    if reattachment_length > 0:
        metadata["reattachment_length"] = reattachment_length
        metadata["reattachment_Xr_over_h"] = reattachment_length / config.step_height

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}")
    return run_dir


__all__ = [
    "BackwardFacingStepConfig",
    "run_backward_facing_step",
]
