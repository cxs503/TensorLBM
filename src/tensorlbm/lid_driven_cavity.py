"""2D Lid-Driven Cavity benchmark using D2Q9 LBM.

The lid-driven cavity is the single most commonly used benchmark in
computational fluid dynamics.  A square domain is enclosed on all four
sides by no-slip walls; the top wall (the "lid") slides at constant
velocity *u_lid* in the +x direction, driving a large primary recirculation
vortex and (at higher Re) smaller corner eddies.

Boundary conditions
-------------------
* **Top wall** (y = ny−1): moving lid at u_x = u_lid, u_y = 0,
  implemented with the Zou/He (1997) analytical BC for interior cells.
  Corner cells are treated with simple bounce-back.
* **Bottom, left, right walls**: stationary no-slip bounce-back.

Validation reference
--------------------
Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
incompressible flow using the Navier-Stokes equations and a multigrid
method.  Journal of Computational Physics, **48**(3), 387-411.

Tabulated velocity profiles at Re = 100, 400, 1000 along the vertical
and horizontal centrelines are stored in :data:`GHIA_RE100`,
:data:`GHIA_RE400`, and :data:`GHIA_RE1000`.

Outputs
-------
* ``run_metadata.json``  – config + diagnostics + Ghia comparison error
* ``ghia_comparison.csv`` – centreline profiles and Ghia reference values
* ``flow_step_XXXXXX.png`` – velocity-magnitude snapshots every
  ``output_interval`` steps
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries import bounce_back_cells
from .config_io import load_config_json, save_config_json
from .cylinder_flow import _maybe_compile
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, stream
from .utils import (
    DiagnosticPoint,
    flow_step_image_path,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data
# ---------------------------------------------------------------------------

# u/u_lid along the vertical centreline x = 0.5 (Re = 100, 129×129 grid)
# Keys: "y" – y/H positions, "u" – u/u_lid values
GHIA_RE100: dict[str, list[float]] = {
    "y": [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
        0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
        0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
        0.0547, 0.0000,
    ],
    "u": [
        1.00000,  0.84123,  0.78871,  0.73722,  0.68717,
        0.23151,  0.00332, -0.13641, -0.20581, -0.21090,
       -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
       -0.03717,  0.00000,
    ],
    # v/u_lid along the horizontal centreline y = 0.5
    "x": [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938,
        0.1563, 0.2266, 0.2344, 0.5000, 0.8047,
        0.8594, 0.9063, 0.9453, 0.9531, 0.9609,
        0.9688, 1.0000,
    ],
    "v": [
        0.00000,  0.09233,  0.10091,  0.10890,  0.12317,
        0.16077,  0.17507,  0.17527,  0.05454, -0.24533,
       -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
       -0.05906,  0.00000,
    ],
}

# u/u_lid along vertical centreline (Re = 400)
GHIA_RE400: dict[str, list[float]] = {
    "y": [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
        0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
        0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
        0.0547, 0.0000,
    ],
    "u": [
        1.00000,  0.75837,  0.68439,  0.61756,  0.55892,
        0.29093,  0.16256,  0.02135, -0.11477, -0.17119,
       -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
       -0.08186,  0.00000,
    ],
    "x": [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938,
        0.1563, 0.2266, 0.2344, 0.5000, 0.8047,
        0.8594, 0.9063, 0.9453, 0.9531, 0.9609,
        0.9688, 1.0000,
    ],
    "v": [
        0.00000,  0.18360,  0.19713,  0.20920,  0.22980,
        0.28003,  0.30174,  0.30203,  0.05186, -0.38598,
       -0.44993, -0.38598, -0.22847, -0.19254, -0.15663,
       -0.12146,  0.00000,
    ],
}

# u/u_lid along vertical centreline (Re = 1000)
GHIA_RE1000: dict[str, list[float]] = {
    "y": [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
        0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
        0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
        0.0547, 0.0000,
    ],
    "u": [
        1.00000,  0.65928,  0.57492,  0.51117,  0.46604,
        0.33304,  0.18719,  0.05702, -0.06080, -0.10648,
       -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
       -0.18109,  0.00000,
    ],
    "x": [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938,
        0.1563, 0.2266, 0.2344, 0.5000, 0.8047,
        0.8594, 0.9063, 0.9453, 0.9531, 0.9609,
        0.9688, 1.0000,
    ],
    "v": [
        0.00000,  0.27485,  0.29012,  0.30353,  0.32627,
        0.37095,  0.33075,  0.32235,  0.02526, -0.31966,
       -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
       -0.21388,  0.00000,
    ],
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LidDrivenCavityConfig:
    """Configuration for the 2D Lid-Driven Cavity benchmark.

    The domain is a square of ``nx × nx`` cells.  The top row slides at
    ``u_lid`` in the x-direction; all other walls are no-slip.

    Args:
        nx: Number of grid cells per side (square domain, so ny = nx).
        u_lid: Lid velocity in lattice units.
        re: Reynolds number Re = u_lid * nx / nu.
        n_steps: Total number of time steps.
        output_interval: Save a snapshot every this many steps.
        output_root: Root directory for output files.
        run_name: Override the auto-generated run folder name.
        seed: Random seed for reproducibility.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        overwrite: If True, remove an existing run directory before writing.
        use_compile: If True, wrap hot-path kernels with ``torch.compile``.
    """

    nx: int = 128
    u_lid: float = 0.1
    re: float = 100.0
    n_steps: int = 10000
    output_interval: int = 2000
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
    def ny(self) -> int:
        """Height equals width for the square cavity."""
        return self.nx

    @property
    def nu(self) -> float:
        """Kinematic viscosity derived from Re = u_lid * nx / nu."""
        return self.u_lid * self.nx / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time τ = 3ν + 0.5."""
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 8:
            msg = "nx must be >= 8"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)
        if self.u_lid <= 0.0 or self.re <= 0.0:
            msg = "u_lid and re must be > 0"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"Invalid tau={self.tau:.4f}; increase re or reduce u_lid/nx"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return f"nx{self.nx}_re{re_label}_uin{self.u_lid:.3f}_steps{self.n_steps}"

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file."""
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> LidDrivenCavityConfig:
        """Load a :class:`LidDrivenCavityConfig` from a JSON file."""
        return load_config_json(cls, path)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def make_cavity_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Create a Boolean mask for all four cavity walls (including corners).

    Args:
        ny: Number of rows.
        nx: Number of columns.
        device: Target device.

    Returns:
        Boolean tensor of shape ``(ny, nx)`` with wall cells set to ``True``.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True   # bottom wall
    mask[-1, :] = True  # top wall (lid row)
    mask[:, 0] = True   # left wall
    mask[:, -1] = True  # right wall
    return mask


def zou_he_moving_lid(f: torch.Tensor, u_lid: float) -> torch.Tensor:
    """Zou/He moving-wall BC for the top row (y = ny-1).

    Prescribes ``ux = u_lid``, ``uy = 0`` at the interior top-wall cells
    (x = 1 to nx-2).  Corner cells (x = 0 and x = nx-1) are left unchanged
    (they are handled by the bounce-back pass that precedes this call).

    D2Q9 direction convention (this module uses the same as :mod:`d2q9`):
    ``0=(0,0), 1=(E), 2=(N), 3=(W), 4=(S), 5=(NE), 6=(NW), 7=(SW), 8=(SE)``

    At the top wall the unknown populations heading *into* the domain are
    f4 (S), f7 (SW), f8 (SE).  Using the Zou & He (1997) analytical
    solution with ``uy = 0``:

    .. math::

        \\rho &= f_0 + f_1 + 2f_2 + f_3 + 2f_5 + 2f_6 \\\\
        f_4 &= f_2 \\\\
        f_7 &= f_5 + \\tfrac{1}{2}(f_1 - f_3) - \\tfrac{1}{2}\\rho\\,u_{lid} \\\\
        f_8 &= f_6 - \\tfrac{1}{2}(f_1 - f_3) + \\tfrac{1}{2}\\rho\\,u_{lid}

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_lid: Lid velocity (x-direction).

    Returns:
        Updated distribution tensor (same shape).
    """
    # Interior cells of the top row only (exclude corners)
    f0 = f[0, -1, 1:-1]
    f1 = f[1, -1, 1:-1]
    f2 = f[2, -1, 1:-1]
    f3 = f[3, -1, 1:-1]
    f5 = f[5, -1, 1:-1]
    f6 = f[6, -1, 1:-1]

    rho = f0 + f1 + 2.0 * f2 + f3 + 2.0 * f5 + 2.0 * f6

    f_new = f.clone()
    f_new[4, -1, 1:-1] = f2
    f_new[7, -1, 1:-1] = f5 + 0.5 * (f1 - f3) - 0.5 * rho * u_lid
    f_new[8, -1, 1:-1] = f6 - 0.5 * (f1 - f3) + 0.5 * rho * u_lid
    return f_new


# ---------------------------------------------------------------------------
# Ghia comparison
# ---------------------------------------------------------------------------


def compare_ghia(
    ux: torch.Tensor,
    uy: torch.Tensor,
    u_lid: float,
    reference: dict[str, list[float]],
) -> dict[str, float]:
    """Compare centreline velocity profiles against Ghia tabulated data.

    Interpolates the LBM fields at the Ghia y/x positions and returns the
    root-mean-square error (RMSE) normalised by ``u_lid``.

    Args:
        ux: x-velocity field of shape ``(ny, nx)``.
        uy: y-velocity field of shape ``(ny, nx)``.
        u_lid: Lid velocity used for normalisation.
        reference: One of :data:`GHIA_RE100`, :data:`GHIA_RE400`, or
            :data:`GHIA_RE1000`.

    Returns:
        Dict with keys ``"rmse_u"`` and ``"rmse_v"``.
    """
    import numpy as np

    ny, nx = ux.shape

    # Normalise by u_lid
    ux_np = ux.detach().cpu().numpy() / u_lid
    uy_np = uy.detach().cpu().numpy() / u_lid

    # Vertical centreline: u at x = nx//2
    x_mid = nx // 2
    u_profile = ux_np[:, x_mid]  # shape (ny,)
    y_pos = np.linspace(0.0, 1.0, ny)
    u_ghia_interp = np.interp(reference["y"], y_pos, u_profile)
    rmse_u = float(np.sqrt(np.mean((u_ghia_interp - np.array(reference["u"])) ** 2)))

    # Horizontal centreline: v at y = ny//2
    y_mid = ny // 2
    v_profile = uy_np[y_mid, :]  # shape (nx,)
    x_pos = np.linspace(0.0, 1.0, nx)
    v_ghia_interp = np.interp(reference["x"], x_pos, v_profile)
    rmse_v = float(np.sqrt(np.mean((v_ghia_interp - np.array(reference["v"])) ** 2)))

    return {"rmse_u": rmse_u, "rmse_v": rmse_v}


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


def _save_cavity_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    vort: torch.Tensor,
    wall_mask: torch.Tensor,
) -> None:
    speed_np = speed.detach().cpu().numpy()
    vort_np = vort.detach().cpu().numpy()
    obs_np = wall_mask.detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    im0 = axes[0].imshow(speed_np, origin="lower", cmap="viridis")
    axes[0].contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    axes[0].set_title(f"Velocity magnitude (step {step})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(vort_np, origin="lower", cmap="coolwarm")
    axes[1].contour(obs_np, levels=[0.5], colors="black", linewidths=0.7)
    axes[1].set_title(f"Vorticity (step {step})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    out = flow_step_image_path(run_dir, step)
    fig.savefig(out, dpi=120)
    write_legacy_snapshot_alias(run_dir, step)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_lid_driven_cavity(config: LidDrivenCavityConfig) -> Path:
    """Run the lid-driven cavity benchmark and save results.

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
        "lid_driven_cavity",
        config.resolved_run_name(),
        config.overwrite,
    )

    ny, nx = config.ny, config.nx

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau, "ny": ny},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    # Initialise: quiescent fluid
    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.zeros((ny, nx), device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    # Top row already at rest; lid starts moving at step 1
    f = equilibrium(rho0, ux0, uy0, device=device)

    wall_mask = make_cavity_wall_mask(ny, nx, device)

    _collide = _maybe_compile(collide_bgk, config.use_compile)
    _stream = _maybe_compile(stream, config.use_compile)

    logger.info(
        "Running lid-driven cavity device=%s NX=%s NY=%s tau=%.4f Re=%.1f steps=%s",
        device, nx, ny, config.tau, config.re, config.n_steps,
    )
    logger.info("Run directory: %s", run_dir)

    diagnostics: list[dict[str, object]] = []

    from .cylinder_flow import compute_vorticity

    for step in range(1, config.n_steps + 1):
        f = _collide(f, tau=config.tau)
        f = _stream(f)
        # Bounce-back on all four walls (including lid row, handles corners)
        f = bounce_back_cells(f, wall_mask)
        # Overwrite interior lid cells with Zou/He moving-wall BC
        f = zou_he_moving_lid(f, config.u_lid)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            # Zero velocity inside walls
            ux = ux.masked_fill(wall_mask, 0.0)
            uy = uy.masked_fill(wall_mask, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            vort = compute_vorticity(ux, uy)

            point = DiagnosticPoint(
                step=step,
                mass=float(rho.sum().item()),
                mass_drift=float(rho.sum().item()) - float(rho0.sum().item()),
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diag_entry: dict[str, object] = {**asdict(point)}
            diagnostics.append(diag_entry)
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f",
                point.step, point.mass, point.mass_drift,
                point.mean_rho, point.max_speed,
            )
            _save_cavity_snapshot(run_dir, step, speed, vort, wall_mask)

    # Final state comparison with Ghia reference data
    rho, ux, uy = macroscopic(f)
    ux = ux.masked_fill(wall_mask, 0.0)
    uy = uy.masked_fill(wall_mask, 0.0)

    re_int = int(config.re)
    if re_int == 100:
        reference = GHIA_RE100
    elif re_int == 400:
        reference = GHIA_RE400
    elif re_int == 1000:
        reference = GHIA_RE1000
    else:
        reference = None

    ghia_errors: dict[str, float] | None = None
    if reference is not None:
        ghia_errors = compare_ghia(ux, uy, config.u_lid, reference)
        logger.info(
            "Ghia comparison (Re=%d): RMSE_u=%.5f  RMSE_v=%.5f",
            re_int, ghia_errors["rmse_u"], ghia_errors["rmse_v"],
        )

    # Write Ghia comparison CSV
    import numpy as np

    ny_final, nx_final = ux.shape
    ux_np = ux.detach().cpu().numpy() / config.u_lid
    uy_np = uy.detach().cpu().numpy() / config.u_lid
    x_mid = nx_final // 2
    y_mid = ny_final // 2
    y_pos = np.linspace(0.0, 1.0, ny_final)
    x_pos = np.linspace(0.0, 1.0, nx_final)

    csv_path = run_dir / "ghia_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["y_pos", "u_lbm", "x_pos", "v_lbm"])
        for j in range(ny_final):
            writer.writerow([
                round(float(y_pos[j]), 6),
                round(float(ux_np[j, x_mid]), 6),
                round(float(x_pos[j]), 6),
                round(float(uy_np[y_mid, j]), 6),
            ])

    metadata["diagnostics"] = diagnostics
    if ghia_errors is not None:
        metadata["ghia_errors"] = ghia_errors
        metadata["ghia_note"] = (
            "rmse_u: RMSE of u/u_lid along vertical centreline vs Ghia (1982). "
            "rmse_v: RMSE of v/u_lid along horizontal centreline vs Ghia (1982)."
        )

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", meta_path)
    return run_dir


__all__ = [
    "GHIA_RE100",
    "GHIA_RE400",
    "GHIA_RE1000",
    "LidDrivenCavityConfig",
    "compare_ghia",
    "make_cavity_wall_mask",
    "run_lid_driven_cavity",
    "zou_he_moving_lid",
]
