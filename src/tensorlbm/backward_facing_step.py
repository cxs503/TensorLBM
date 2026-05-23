"""2D Backward-Facing Step (BFS) benchmark using D2Q9 LBM.

The backward-facing step is a canonical flow-separation benchmark.  A
channel flow encounters a sudden expansion at a step, forming a primary
recirculation zone downstream of the step.  The non-dimensional
reattachment length :math:`x_r / h` (step height h) characterises the
flow and depends on the Reynolds number.

Geometry
--------
::

    y=ny-1  __________________________________________
            |                                        |  ← top wall (no-slip)
            |  →  →  →                              |
    y=step_h|_________________________________________|
    y=0                  ← solid step region ↓       |  ← bottom wall
                   (y < step_h, x < x_step)          |
                                                      |
                                             x=nx-1  outlet

* **Inlet** (x = 0, y = step_h … ny-2): uniform x-velocity ``u_in``.
* **Outlet** (x = nx-1): zero-gradient (copy-from-upstream) condition.
* **Top wall** (y = ny-1): no-slip bounce-back for all x.
* **Bottom wall** (y = 0): no-slip bounce-back for x ≥ x_step.
* **Step solid** (x < x_step, y < step_h): no-slip bounce-back; this
  block represents the pre-step channel and the step face.

Reynolds number is defined as Re = u_in · step_h / ν.

Benchmark diagnostic
--------------------
After the simulation the primary reattachment length

.. math::

    x_r^* = (x_r - x_{step}) / h

is measured by finding the first grid column downstream of the step
corner where the streamwise velocity at y = 1 (first fluid row above the
bottom wall) becomes positive.

Reference values for 2:1 expansion (step_h = ny // 2), uniform inlet,
2D laminar flow (approximate):

* Re = 100  →  x_r* ≈ 2.5 – 3.5
* Re = 200  →  x_r* ≈ 5 – 6
* Re = 400  →  x_r* ≈ 8 – 11

Outputs
-------
* ``run_metadata.json``  – config + diagnostics + reattachment length
* ``reattachment.csv``   – step, max|u|, reattachment length per interval
* ``snapshot_XXXXXX.png`` – velocity-magnitude snapshots

References
----------
Armaly, B. F., Durst, F., Pereira, J. C. F., & Schönung, B. (1983).
Experimental and theoretical investigation of backward-facing step flow.
Journal of Fluid Mechanics, **127**, 473-496.

Erturk, E. (2008). Numerical solutions of 2-D steady incompressible flow
over a backward-facing step, Part I: High Reynolds number solutions.
Computers & Fluids, **37**(6), 633-655.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries import bounce_back_cells, zou_he_inlet_velocity
from .config_io import load_config_json, save_config_json
from .cylinder_flow import _maybe_compile
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, stream
from .utils import (
    DiagnosticPoint,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackwardFacingStepConfig:
    """Configuration for the 2D Backward-Facing Step benchmark.

    The step height is ``step_h`` cells; the expansion ratio is
    ``ny / (ny - step_h)`` (exactly 2:1 when ``step_h = ny // 2``).
    Reynolds number Re = u_in · step_h / ν.

    Args:
        nx: Domain width in grid cells.
        ny: Domain height in grid cells.
        step_h: Step height in grid cells (must satisfy 1 ≤ step_h < ny-2).
        x_step: Length of the pre-step solid block in grid cells (≥ 1).
            A small value keeps the upstream channel short; the step face
            is at x = x_step.
        u_in: Inlet x-velocity (uniform profile above the step).
        re: Reynolds number Re = u_in · step_h / ν.
        n_steps: Total number of time steps.
        output_interval: Save a snapshot every this many steps.
        output_root: Root directory for output files.
        run_name: Override the auto-generated run folder name.
        seed: Random seed.
        device: Torch device string.
        overwrite: Remove existing run directory before writing.
        use_compile: Wrap hot-path kernels with ``torch.compile``.
    """

    nx: int = 400
    ny: int = 80
    step_h: int = 40      # step height (ny//2 → 2:1 expansion)
    x_step: int = 80      # pre-step solid length (upstream channel)
    u_in: float = 0.05
    re: float = 100.0
    n_steps: int = 30000
    output_interval: int = 5000
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False
    use_compile: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re = u_in * step_h / nu."""
        return self.u_in * self.step_h / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 20:
            msg = "nx must be >= 20"
            raise ValueError(msg)
        if self.ny < 6:
            msg = "ny must be >= 6"
            raise ValueError(msg)
        if not (1 <= self.step_h <= self.ny - 3):
            msg = f"step_h={self.step_h} must be in [1, ny-3={self.ny - 3}]"
            raise ValueError(msg)
        if self.x_step < 1 or self.x_step >= self.nx - 5:
            msg = f"x_step={self.x_step} must be in [1, nx-6={self.nx - 6}]"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)
        if self.u_in <= 0.0 or self.re <= 0.0:
            msg = "u_in and re must be > 0"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/step_h"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"bfs_nx{self.nx}_ny{self.ny}_sh{self.step_h}"
            f"_re{re_label}_uin{self.u_in:.3f}_steps{self.n_steps}"
        )

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file."""
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BackwardFacingStepConfig:
        """Load a :class:`BackwardFacingStepConfig` from a JSON file."""
        return load_config_json(cls, path)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def make_bfs_solid_mask(
    ny: int,
    nx: int,
    step_h: int,
    x_step: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the Boolean solid mask for the BFS domain.

    Solid cells include the top and bottom walls, the inlet bottom portion
    (step face), and the step solid block.

    Args:
        ny: Domain height.
        nx: Domain width.
        step_h: Step height (number of solid rows at the bottom-left).
        x_step: Horizontal extent of the step solid (columns 0 … x_step-1).
        device: Target device.

    Returns:
        Boolean tensor ``(ny, nx)`` – ``True`` for solid cells.
    """
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[-1, :] = True            # top wall
    solid[0, x_step:] = True       # bottom wall (after step)

    # Step solid block: x < x_step AND y < step_h
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    step_block = (xx < x_step) & (yy < step_h)
    solid |= step_block
    return solid


def _bfs_inlet_rows(ny: int, step_h: int) -> tuple[int, int]:
    """Return the y-slice (y_lo, y_hi) of the inlet above the step.

    The inlet spans ``y = step_h`` to ``y = ny-2`` (inclusive), i.e.,
    above the solid step block and below the top wall.
    """
    return step_h, ny - 1  # exclusive upper bound for slice


# ---------------------------------------------------------------------------
# Inlet BC (uniform velocity above the step)
# ---------------------------------------------------------------------------


def _apply_bfs_inlet(
    f: torch.Tensor,
    u_in: float,
    step_h: int,
) -> torch.Tensor:
    """Apply Zou/He inlet BC at x = 0 for rows above the step.

    Rows y = 0 … step_h-1 at x = 0 belong to the solid step block and
    will be overwritten by the subsequent bounce-back pass; so we
    safely apply zou_he_inlet_velocity to the full left column and let
    the bounce-back step restore the solid cells.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_in: Inlet x-velocity.
        step_h: Step height (rows below step_h at x = 0 are solid).

    Returns:
        Updated distribution tensor.
    """
    return zou_he_inlet_velocity(f, u_in)


def _apply_bfs_outlet(f: torch.Tensor) -> torch.Tensor:
    """Zero-gradient (copy) outlet BC at x = nx-1."""
    f_new = f.clone()
    f_new[:, :, -1] = f[:, :, -2]
    return f_new


# ---------------------------------------------------------------------------
# Reattachment length
# ---------------------------------------------------------------------------


def measure_reattachment_length(
    ux: torch.Tensor,
    x_step: int,
    step_h: int,
) -> float:
    """Measure the primary reattachment length x_r* = (x_r - x_step) / step_h.

    Scans the row y = 1 (first fluid row above the bottom wall) from the
    step corner (x = x_step) rightward and returns the distance to the
    first column where ux > 0, normalised by the step height.

    Returns 0.0 if the recirculation zone has not yet formed or if no
    reattachment is found within the domain.

    Args:
        ux: Streamwise velocity field of shape ``(ny, nx)``.
        x_step: x-index of the step face (start of the open bottom channel).
        step_h: Step height used for non-dimensionalisation.

    Returns:
        Non-dimensional reattachment length :math:`x_r^* = (x_r - x_{step}) / h`.
    """
    centreline = ux[1, x_step:].cpu()  # first row above bottom wall, post-step

    # Scan for the transition from negative to positive ux
    for i, val in enumerate(centreline.tolist()):
        if val > 0.0:
            return float(i) / max(step_h, 1)

    # No reattachment found inside the domain
    return 0.0


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


def _save_bfs_snapshot(
    run_dir: Path,
    step: int,
    ux: torch.Tensor,
    solid: torch.Tensor,
    config: BackwardFacingStepConfig,
) -> None:
    ux_np = ux.detach().cpu().numpy()
    obs_np = solid.detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    im = ax.imshow(ux_np, origin="lower", cmap="RdBu_r",
                   vmin=-config.u_in, vmax=config.u_in * 2)
    ax.contour(obs_np, levels=[0.5], colors="black", linewidths=0.7)
    ax.set_title(f"BFS ux (step {step})")
    plt.colorbar(im, ax=ax, fraction=0.02, label="ux")

    out = run_dir / f"snapshot_{step:06d}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_backward_facing_step(config: BackwardFacingStepConfig) -> Path:
    """Run the backward-facing step benchmark and save results.

    Args:
        config: Simulation configuration.

    Returns:
        Path of the output directory created for this run.
    """
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "backward_facing_step",
        config.resolved_run_name(),
        config.overwrite,
    )

    ny, nx = config.ny, config.nx

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    # Solid mask (walls + step)
    solid = make_bfs_solid_mask(ny, nx, config.step_h, config.x_step, device)

    # Initialise: rest state with small inlet velocity above the step
    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.zeros((ny, nx), device=device)
    ux0[config.step_h:ny - 1, :] = config.u_in  # prescribe velocity above step
    ux0[solid] = 0.0
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)

    _collide = _maybe_compile(collide_bgk, config.use_compile)
    _stream = _maybe_compile(stream, config.use_compile)

    logger.info(
        "Running BFS device=%s NX=%s NY=%s step_h=%s x_step=%s tau=%.4f Re=%.1f steps=%s",
        device, nx, ny, config.step_h, config.x_step,
        config.tau, config.re, config.n_steps,
    )
    logger.info("Run directory: %s", run_dir)

    diagnostics: list[dict[str, object]] = []
    reattach_series: list[tuple[int, float, float]] = []

    for step in range(1, config.n_steps + 1):
        f = _collide(f, tau=config.tau)
        f = _stream(f)
        # Apply BCs
        f = _apply_bfs_inlet(f, config.u_in, config.step_h)
        f = _apply_bfs_outlet(f)
        f = bounce_back_cells(f, solid)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(solid, 0.0)
            uy = uy.masked_fill(solid, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            x_r_star = measure_reattachment_length(ux, config.x_step, config.step_h)

            point = DiagnosticPoint(
                step=step,
                mass=float(rho.sum().item()),
                mass_drift=float(rho.sum().item()) - float(rho0.sum().item()),
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diag_entry: dict[str, object] = {**asdict(point), "reattach_xr_star": x_r_star}
            diagnostics.append(diag_entry)
            reattach_series.append((step, x_r_star, float(speed.max().item())))

            logger.info(
                "step=%5d mass=%.6f drift=%+.6f max|u|=%.6f xr*=%.3f",
                point.step, point.mass, point.mass_drift,
                point.max_speed, x_r_star,
            )
            _save_bfs_snapshot(run_dir, step, ux, solid, config)

    # Final reattachment length
    rho_f, ux_f, uy_f = macroscopic(f)
    ux_f = ux_f.masked_fill(solid, 0.0)
    final_xr = measure_reattachment_length(ux_f, config.x_step, config.step_h)
    logger.info(
        "Final reattachment length xr* = %.3f  (Re=%.1f, step_h=%d)",
        final_xr, config.re, config.step_h,
    )

    # Write reattachment CSV
    csv_path = run_dir / "reattachment.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "xr_star", "max_speed"])
        writer.writerows(reattach_series)

    metadata["diagnostics"] = diagnostics
    metadata["final_reattachment_xr_star"] = final_xr
    metadata["reattachment_note"] = (
        "xr_star = (x_reattach - x_step) / step_h.  "
        "Primary reattachment length, measured from the step corner.  "
        "Expected approx. 3 for Re=100, 5-6 for Re=200 (2:1 expansion, uniform inlet)."
    )

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", meta_path)
    return run_dir


__all__ = [
    "BackwardFacingStepConfig",
    "make_bfs_solid_mask",
    "measure_reattachment_length",
    "run_backward_facing_step",
]
