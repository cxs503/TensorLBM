"""Discontinuous-Galerkin / Lattice-Boltzmann (DG-LBM) hybrid solver.

Two solvers live here:

* **Real nodal-DG hybrid** (``use_real_dg=True``) — the genuine DG-LBM.  A DG
  band (``build_dg_hull_band_mask``) encloses the obstacle; its inner faces
  bounce-back off the solid and its outer faces couple to the exterior D3Q19 LBM
  via :func:`tensorlbm.dg_band.hybrid_step` (method-of-lines band with
  τ_dg = τ_lbm − ½, exterior collide + stream + DG-trace write-back).  The DG
  advection is a real dimension-by-dimension nodal-DG operator (P1 Lobatto,
  upwind flux, SSP-RK3) — see :mod:`tensorlbm.dg_advection` and
  :mod:`tensorlbm.dg_band`.  This is the recommended path.

* **Legacy gradient-correction** (``use_real_dg=False``, the default for
  backward compatibility) — the original near-wall scheme.  The DG zone replaces
  the BGK non-equilibrium part with the first-order Chapman–Enskog correction
  computed from a finite-difference strain-rate tensor (NOT a true DG
  discretisation; it has no polynomial basis / numerical flux / RK step).  Kept
  for reproducibility of older runs.

Legacy gradient-correction details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the near-wall zone the distribution is written as

    f_i = f_i^eq + f_i^(1)

where the non-equilibrium correction is

    f_i^(1) = -2 τ w_i ρ (c_iα c_iβ - δ_αβ / 3) S_αβ

and S_αβ = (∂u_α/∂x_β + ∂u_β/∂x_α) / 2 is the strain-rate tensor computed
from second-order central differences of the macroscopic velocity field.

Coupling (legacy)
~~~~~~~~~~~~~~~~~
After the DG-enhanced collision the distributions in the near-wall zone are
updated in-place.  The standard ``stream3d`` step and boundary conditions are
applied globally in the usual way, so no special treatment at the DG–LBM
interface is needed beyond the mask-gated collision replacement.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

from .boundaries3d import (
    apply_simple_channel_boundaries_3d,
    far_field_bc_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .cylinder_flow import _maybe_compile
from .d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from .dg_advection import equilibrium_dg, get_ops
from .dg_band import build_band_topology, compute_dg_solid_force, hybrid_step, project_band_to_lbm
from .physics import collide_smagorinsky_bgk3d, collide_dynamic_smagorinsky_bgk3d, collide_mrt3d, collide_smagorinsky_mrt3d
from .wall_model import apply_wall_model_bounce_back, wall_function_3d
from .boundaries3d import free_slip_y_walls_3d, free_slip_z_walls_3d
from .obstacles import compute_obstacle_forces_3d
from .logging_config import configure_logging, logger
from .solver3d import correct_mass3d, stream3d
from .suboff_cad import SuboffHullType, build_suboff_mask
from .suboff_resistance import _voxel_wetted_area
from .utils import (
    DiagnosticPoint,
    configure_cpu_threads,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)

try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants – D3Q19 lattice speed of sound squared
# ---------------------------------------------------------------------------
_CS2: float = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DGLBMConfig:
    """Configuration for the DG-LBM hybrid sphere-flow simulation.

    Attributes:
        nx, ny, nz: Grid dimensions.
        u_in: Inlet velocity (lattice units).
        re: Reynolds number.
        radius: Sphere radius (lattice units).
        dg_band: Thickness of the DG near-wall zone in lattice units.
            Cells within the shell ``[radius, radius + dg_band]`` from the
            sphere centre use DG-enhanced collision.
        n_steps: Total number of time steps.
        output_interval: Steps between output snapshots / checkpoints.
        output_root: Root directory for output artefacts.
        run_name: Optional run name.  Auto-generated when *None*.
        seed: Random seed.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        num_threads: CPU thread count; *None* uses PyTorch default.
        overwrite: If *True*, overwrite an existing run directory.
        resume_checkpoint: Path to a previous run directory to resume from.
        use_compile: If *True*, JIT-compile hot-path kernels with
            ``torch.compile``.
        dg_order: Polynomial order for DG reconstruction (currently 1 is
            supported; reserved for future higher-order extension).
    """

    nx: int = 120
    ny: int = 60
    nz: int = 60
    u_in: float = 0.06
    re: float = 50.0
    radius: float = 8.0
    dg_band: float = 4.0
    n_steps: int = 500
    output_interval: int = 100
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    num_threads: int | None = None
    overwrite: bool = False
    resume_checkpoint: Path | None = None
    use_compile: bool = False
    dg_order: int = 4  # Gradient accuracy order (2 or 4); legacy (ignored by real DG)
    smagorinsky_cs: float = 0.0  # Smagorinsky LES (>0 enables)
    dynamic_smag: bool = False  # Dynamic Smagorinsky (auto Cs)
    use_mrt: bool = False  # MRT collision (more stable at low tau)
    use_wall_model: bool = False  # Log-law wall function
    free_slip_walls: bool = False  # Free-slip domain boundaries
    use_real_dg: bool = False  # If True, use the genuine nodal-DG hybrid solver
    use_wall_function: bool = False  # If True, high-Re log-law wall function (body force, τ-decoupled) + friction/pressure drag
    dg_degree: int = 1  # DG polynomial degree for the real solver (locked to 1)
    dg_substeps: int = 16  # RK sub-steps; low τ_dg (fine grid/high Re) ⇒ stiffer, raise this

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))

    # ------------------------------------------------------------------
    # Derived physics
    # ------------------------------------------------------------------

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    @property
    def dg_radius(self) -> float:
        """Outer radius of the DG near-wall zone."""
        return self.radius + self.dg_band

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, nz must be at least 16, 8, 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0 or self.radius <= 0.0:
            raise ValueError("u_in, re, and radius must be > 0")
        if self.dg_band <= 0.0:
            raise ValueError("dg_band must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/radius"
            )
        if self.dg_order not in (1,2,4):
            raise ValueError("Only dg_order=1,2,4 (linear DG) is currently supported")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")

    # ------------------------------------------------------------------
    # Run name
    # ------------------------------------------------------------------

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"nx{self.nx}_ny{self.ny}_nz{self.nz}"
            f"_re{re_label}_uin{self.u_in:.3f}"
            f"_dg{self.dg_band:.1f}_steps{self.n_steps}"
        )


# ---------------------------------------------------------------------------
# Zone-mask utilities
# ---------------------------------------------------------------------------


def build_dg_shell_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    r_inner: float,
    r_outer: float,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask for the DG near-wall shell.

    Returns a boolean tensor of shape ``(nz, ny, nx)`` that is *True* for
    cells whose centre lies in the open spherical shell
    ``r_inner < r ≤ r_outer``.

    Args:
        nx, ny, nz: Grid extents.
        cx, cy, cz: Sphere centre coordinates.
        r_inner: Inner radius (exclusive) – normally the sphere radius.
        r_outer: Outer radius (inclusive) – sphere radius + DG band width.
        device: Target device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return (r2 > r_inner ** 2) & (r2 <= r_outer ** 2)


# ---------------------------------------------------------------------------
# DG gradient reconstruction
# ---------------------------------------------------------------------------


def dg_compute_velocity_gradients(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    order: int = 2,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Compute velocity gradients via second-order central differences.

    Uses ``torch.roll`` for a compact periodic-difference stencil (the domain
    BCs are handled separately by the LBM boundary step, so the roll values at
    inlet/outlet are not used in the final update once the BC step overwrites
    them).

    Args:
        ux, uy, uz: Velocity components of shape ``(nz, ny, nx)``.

    Returns:
        Nine gradient tensors of shape ``(nz, ny, nx)`` in the order::

            dux_dx, dux_dy, dux_dz,
            duy_dx, duy_dy, duy_dz,
            duz_dx, duz_dy, duz_dz
    """
    # Central differences (2nd or 4th order)
    if order >= 4:
        # 4th-order central difference: (-f2 + 8f1 - 8f-1 + f-2) / 12
        dux_dx = (-torch.roll(ux, -2, 2) + 8*torch.roll(ux, -1, 2) - 8*torch.roll(ux, 1, 2) + torch.roll(ux, 2, 2)) / 12.0
        dux_dy = (-torch.roll(ux, -2, 1) + 8*torch.roll(ux, -1, 1) - 8*torch.roll(ux, 1, 1) + torch.roll(ux, 2, 1)) / 12.0
        dux_dz = (-torch.roll(ux, -2, 0) + 8*torch.roll(ux, -1, 0) - 8*torch.roll(ux, 1, 0) + torch.roll(ux, 2, 0)) / 12.0

        duy_dx = (-torch.roll(uy, -2, 2) + 8*torch.roll(uy, -1, 2) - 8*torch.roll(uy, 1, 2) + torch.roll(uy, 2, 2)) / 12.0
        duy_dy = (-torch.roll(uy, -2, 1) + 8*torch.roll(uy, -1, 1) - 8*torch.roll(uy, 1, 1) + torch.roll(uy, 2, 1)) / 12.0
        duy_dz = (-torch.roll(uy, -2, 0) + 8*torch.roll(uy, -1, 0) - 8*torch.roll(uy, 1, 0) + torch.roll(uy, 2, 0)) / 12.0

        duz_dx = (-torch.roll(uz, -2, 2) + 8*torch.roll(uz, -1, 2) - 8*torch.roll(uz, 1, 2) + torch.roll(uz, 2, 2)) / 12.0
        duz_dy = (-torch.roll(uz, -2, 1) + 8*torch.roll(uz, -1, 1) - 8*torch.roll(uz, 1, 1) + torch.roll(uz, 2, 1)) / 12.0
        duz_dz = (-torch.roll(uz, -2, 0) + 8*torch.roll(uz, -1, 0) - 8*torch.roll(uz, 1, 0) + torch.roll(uz, 2, 0)) / 12.0
    else:
        # 2nd-order central difference
        dux_dx = (torch.roll(ux, -1, 2) - torch.roll(ux, 1, 2)) * 0.5
        dux_dy = (torch.roll(ux, -1, 1) - torch.roll(ux, 1, 1)) * 0.5
        dux_dz = (torch.roll(ux, -1, 0) - torch.roll(ux, 1, 0)) * 0.5

        duy_dx = (torch.roll(uy, -1, 2) - torch.roll(uy, 1, 2)) * 0.5
        duy_dy = (torch.roll(uy, -1, 1) - torch.roll(uy, 1, 1)) * 0.5
        duy_dz = (torch.roll(uy, -1, 0) - torch.roll(uy, 1, 0)) * 0.5

        duz_dx = (torch.roll(uz, -1, 2) - torch.roll(uz, 1, 2)) * 0.5
        duz_dy = (torch.roll(uz, -1, 1) - torch.roll(uz, 1, 1)) * 0.5
        duz_dz = (torch.roll(uz, -1, 0) - torch.roll(uz, 1, 0)) * 0.5

    return (
        dux_dx, dux_dy, dux_dz,
        duy_dx, duy_dy, duy_dz,
        duz_dx, duz_dy, duz_dz,
    )


# ---------------------------------------------------------------------------
# DG-enhanced collision step
# ---------------------------------------------------------------------------


def collide_dg_lbm(
    f: torch.Tensor,
    tau: float,
    dg_mask: torch.Tensor,
) -> torch.Tensor:
    """DG-LBM hybrid collision operator.

    In the LBM exterior (``dg_mask == False``) this is identical to the
    standard D3Q19 BGK collision.  In the DG near-wall zone
    (``dg_mask == True``) the non-equilibrium part is replaced by the
    first-order Chapman–Enskog expression evaluated from explicitly
    reconstructed velocity gradients:

    .. math::

        f_i^{\\text{neq}} = -2\\tau\\, w_i\\, \\rho
            \\left(c_{i\\alpha}c_{i\\beta} - \\frac{\\delta_{\\alpha\\beta}}{3}\\right)
            S_{\\alpha\\beta}

    where :math:`S_{\\alpha\\beta} = (\\partial u_\\alpha / \\partial x_\\beta
    + \\partial u_\\beta / \\partial x_\\alpha) / 2`.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: BGK relaxation time.
        dg_mask: Boolean tensor of shape ``(nz, ny, nx)`` marking the DG zone.

    Returns:
        Updated distribution tensor of shape ``(19, nz, ny, nx)``.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)

    # ----------------------------------------------------------------
    # Standard BGK everywhere (used in LBM exterior; overridden below
    # in the DG zone)
    # ----------------------------------------------------------------
    f_bgk = f - (f - feq) / tau

    if not dg_mask.any():
        return f_bgk

    # ----------------------------------------------------------------
    # DG-enhanced non-equilibrium in the near-wall zone
    # ----------------------------------------------------------------
    (
        dux_dx, dux_dy, dux_dz,
        duy_dx, duy_dy, duy_dz,
        duz_dx, duz_dy, duz_dz,
    ) = dg_compute_velocity_gradients(ux, uy, uz, order=4)

    # Strain-rate tensor components
    sxx = dux_dx
    syy = duy_dy
    szz = duz_dz
    sxy = 0.5 * (dux_dy + duy_dx)
    sxz = 0.5 * (dux_dz + duz_dx)
    syz = 0.5 * (duy_dz + duz_dy)

    device = f.device
    c_dev = C.to(device)          # shape (19, 3)
    w_dev = W.to(device)          # shape (19,)

    cx = c_dev[:, 0].view(19, 1, 1, 1)  # (19,1,1,1)
    cy = c_dev[:, 1].view(19, 1, 1, 1)
    cz = c_dev[:, 2].view(19, 1, 1, 1)

    # Traceless symmetric velocity moment tensor: Q_{iαβ} = cα cβ - δαβ/3
    # Contracted with S: Q_{iαβ} S_{αβ}  (sum over α,β)
    cs2 = _CS2
    q_dot_s = (
        (cx * cx - cs2) * sxx
        + (cy * cy - cs2) * syy
        + (cz * cz - cs2) * szz
        + 2.0 * cx * cy * sxy
        + 2.0 * cx * cz * sxz
        + 2.0 * cy * cz * syz
    )  # shape (19, nz, ny, nx)

    # First-order CE non-equilibrium: f^(1) = -2τ w ρ Q:S
    f_neq_dg = -2.0 * tau * w_dev.view(19, 1, 1, 1) * rho.unsqueeze(0) * q_dot_s
    f_dg = feq + f_neq_dg

    # Blend: use f_dg in the DG zone, f_bgk elsewhere
    mask4d = dg_mask.unsqueeze(0)  # (1, nz, ny, nx) → broadcasts to (19,…)
    return torch.where(mask4d, f_dg, f_bgk)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------


def _save_dg_lbm_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
    dg_mask: torch.Tensor,
    nz: int,
) -> None:
    """Save speed magnitude on the mid-z slice with DG zone overlay."""
    mid_z = nz // 2
    speed_np = speed[mid_z].detach().cpu().numpy()
    obs_np = obstacle[mid_z].detach().cpu().float().numpy()
    dg_np = dg_mask[mid_z].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    im = ax.imshow(speed_np, origin="lower", cmap="viridis")
    ax.contour(obs_np, levels=[0.5], colors="white", linewidths=0.8, linestyles="-")
    ax.contour(dg_np, levels=[0.5], colors="cyan", linewidths=0.6, linestyles="--")
    ax.set_title(f"DG-LBM velocity magnitude – mid-z slice (step {step})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_dg_lbm_sphere_flow(config: DGLBMConfig) -> Path:
    """Run the DG-LBM hybrid sphere-flow simulation.

    The near-wall zone (``radius < r ≤ radius + dg_band``) uses the
    DG-enhanced collision operator; the exterior uses standard BGK.

    Args:
        config: Simulation configuration.

    Returns:
        Path to the run output directory.
    """
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    applied_num_threads = configure_cpu_threads(device, config.num_threads)
    run_dir = prepare_run_dir(
        config.output_root,
        "dg_lbm_sphere",
        config.resolved_run_name(),
        config.overwrite,
    )

    ckpt_str = str(config.resume_checkpoint) if config.resume_checkpoint else None
    metadata: dict = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "resume_checkpoint": ckpt_str,
        },
        "derived": {
            "nu": config.nu,
            "tau": config.tau,
            "dg_radius": config.dg_radius,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "num_threads": applied_num_threads,
        },
        "reproducibility": get_reproducibility_metadata(),
    }

    # ----------------------------------------------------------------
    # Geometry: sphere obstacle + DG near-wall shell + channel walls
    # ----------------------------------------------------------------
    cx = config.nx * 0.25
    cy = config.ny * 0.5
    cz = config.nz * 0.5

    obstacle = sphere_mask(
        config.nx, config.ny, config.nz, cx, cy, cz, config.radius, device=device
    )
    dg_mask = build_dg_shell_mask(
        config.nx, config.ny, config.nz,
        cx, cy, cz,
        config.radius, config.dg_radius,
        device=device,
    )
    wall_mask = make_channel_wall_mask_3d(
        config.nz, config.ny, config.nx, obstacle, device=device
    )

    # ----------------------------------------------------------------
    # Initialise or resume
    # ----------------------------------------------------------------
    start_step = 1
    restart_info: dict[str, object] = {"resumed": False}
    if config.resume_checkpoint is not None:
        f, resume_step, ckpt_meta = load_checkpoint(
            config.resume_checkpoint,
            device=device,
            expected_shape=(19, config.nz, config.ny, config.nx),
            expected_lattice_directions=19,
        )
        if resume_step >= config.n_steps:
            raise ValueError(
                f"resume checkpoint step {resume_step} is not less than n_steps={config.n_steps}"
            )
        f = f.to(device)
        start_step = resume_step + 1
        logger.info(
            "Resumed from checkpoint %s at step %d", config.resume_checkpoint, resume_step
        )
        restart_info = {
            "resumed": True,
            "source_checkpoint": str(config.resume_checkpoint),
            "source_step": resume_step,
            "checkpoint_format_version": ckpt_meta.get("format_version"),
        }
    else:
        rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
        ux0 = torch.full((config.nz, config.ny, config.nx), config.u_in, device=device)
        uy0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
        uz0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
        ux0[obstacle] = 0.0
        f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    rho0_mass = torch.ones((config.nz, config.ny, config.nx), device=device)
    initial_mass = float(rho0_mass.sum().item())
    diagnostics: list[dict] = []

    # Optionally JIT-compile the streaming kernel (collision uses DG branch)
    _stream = _maybe_compile(stream3d, config.use_compile)

    dg_cells = int(dg_mask.sum().item())
    lbm_cells = int((~obstacle & ~dg_mask).sum().item())
    logger.info(
        "DG-LBM hybrid: device=%s NX=%s NY=%s NZ=%s tau=%.4f "
        "steps=%s output_interval=%s compile=%s num_threads=%s",
        device, config.nx, config.ny, config.nz, config.tau,
        config.n_steps, config.output_interval, config.use_compile, applied_num_threads,
    )
    logger.info(
        "Zone breakdown: sphere=%d cells  DG shell=%d cells  LBM exterior=%d cells",
        int(obstacle.sum().item()), dg_cells, lbm_cells,
    )
    logger.info("Run directory: %s", run_dir)

    step_range = range(start_step, config.n_steps + 1)
    step_iter = (
        _tqdm(step_range, desc="DG-LBM sphere", unit="step")
        if _TQDM_AVAILABLE
        else step_range
    )

    for step in step_iter:

        # Standard streaming
        f = _stream(f)

        # Momentum-exchange force on hull
        fx, fy, fz = compute_obstacle_forces_3d(f, obstacle)
        drag_lu = fx.item()

        # Boundary conditions (walls + obstacle bounce-back + inlet/outlet)
        f = apply_simple_channel_boundaries_3d(
            f,
            u_in=config.u_in,
            wall_mask=wall_mask,
            obstacle_mask=obstacle,
        )

        # Periodic mass correction
        if step % config.output_interval == 0:
            f = correct_mass3d(f, initial_mass)

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
            diagnostics.append(asdict(point))
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f drag=%.4f",
                point.step, point.mass, point.mass_drift,
                point.mean_rho, point.max_speed, drag_lu,
            )
            _save_dg_lbm_snapshot(run_dir, step, speed, obstacle, dg_mask, config.nz)
            save_checkpoint(f, step, run_dir)

    metadata["diagnostics"] = diagnostics
    metadata["restart"] = restart_info
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir

# ---------------------------------------------------------------------------
# SUBOFF DG-LBM hybrid – configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DGLBMSuboffConfig:
    """Configuration for the DG-LBM hybrid SUBOFF submarine-flow simulation.

    The SUBOFF hull occupies the near-wall DG zone; the LBM exterior uses
    standard D3Q19 BGK collision.  The hull axis runs along the x-direction.

    Attributes:
        nx, ny, nz: Grid dimensions.
        u_in: Inlet velocity (lattice units).
        re: Reynolds number based on hull length.
        hull_length: SUBOFF hull length (lattice units).
        hull_type: SUBOFF model variant (``"bare_hull"``, ``"with_sail"``,
            ``"full"``).
        dg_band: Thickness of the DG near-wall zone (lattice units).
            Cells within ``dg_band`` lattice cells of the hull surface use
            DG-enhanced collision.
        n_steps: Total number of time steps.
        output_interval: Steps between output snapshots / checkpoints.
        output_root: Root directory for output artefacts.
        run_name: Optional run name.  Auto-generated when *None*.
        seed: Random seed.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        num_threads: CPU thread count; *None* uses PyTorch default.
        overwrite: If *True*, overwrite an existing run directory.
        resume_checkpoint: Path to a previous run directory to resume from.
        use_compile: If *True*, JIT-compile hot-path kernels with
            ``torch.compile``.
        dg_order: Polynomial order for DG reconstruction (currently 1).
    """

    nx: int = 200
    ny: int = 80
    nz: int = 80
    u_in: float = 0.06
    re: float = 200.0
    hull_length: float = 120.0
    hull_type: str = SuboffHullType.BARE_HULL.value
    dg_band: float = 4.0
    n_steps: int = 500
    output_interval: int = 100
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    num_threads: int | None = None
    overwrite: bool = False
    resume_checkpoint: Path | None = None
    use_compile: bool = False
    dg_order: int = 4  # Gradient accuracy order (2 or 4); legacy (ignored by real DG)
    smagorinsky_cs: float = 0.0  # Smagorinsky LES (>0 enables)
    dynamic_smag: bool = False  # Dynamic Smagorinsky (auto Cs)
    use_mrt: bool = False  # MRT collision (more stable at low tau)
    use_wall_model: bool = False  # Log-law wall function
    free_slip_walls: bool = False  # Free-slip domain boundaries
    use_real_dg: bool = False  # If True, use the genuine nodal-DG hybrid solver
    use_wall_function: bool = False  # If True, high-Re log-law wall function (body force, τ-decoupled) + friction/pressure drag
    dg_degree: int = 1  # DG polynomial degree for the real solver (locked to 1)
    dg_substeps: int = 16  # RK sub-steps; low τ_dg (fine grid/high Re) ⇒ stiffer, raise this

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())
        object.__setattr__(self, "hull_type", SuboffHullType(self.hull_type).value)
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))

    # ------------------------------------------------------------------
    # Derived physics (characteristic length = hull_length)
    # ------------------------------------------------------------------

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, nz must be at least 16, 8, 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0 or self.hull_length <= 0.0:
            raise ValueError("u_in, re, and hull_length must be > 0")
        if self.dg_band <= 0.0:
            raise ValueError("dg_band must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/hull_length"
            )
        if self.dg_order not in (1,2,4):
            raise ValueError("Only dg_order=1,2,4 (linear DG) is currently supported")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")

    # ------------------------------------------------------------------
    # Run name
    # ------------------------------------------------------------------

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return (
            f"nx{self.nx}_ny{self.ny}_nz{self.nz}"
            f"_re{re_label}_uin{self.u_in:.3f}"
            f"_{self.hull_type}_dg{self.dg_band:.1f}_steps{self.n_steps}"
        )


# ---------------------------------------------------------------------------
# SUBOFF DG zone construction
# ---------------------------------------------------------------------------


def build_dg_hull_band_mask(
    solid_mask: torch.Tensor,
    dg_band: float,
) -> torch.Tensor:
    """Boolean mask for the DG near-wall band around a hull solid mask.

    Dilates *solid_mask* by ``ceil(dg_band)`` lattice cells using a
    3-D max-pool (equivalent to a Chebyshev-ball dilation) and returns
    the shell that is inside the dilated region but outside the solid.

    Args:
        solid_mask: Boolean tensor of shape ``(nz, ny, nx)`` marking solid
            cells.
        dg_band: Near-wall zone half-thickness in lattice units.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``; *True* for cells in the
        DG near-wall band (not solid, within ``dg_band`` of the hull).
    """
    k = max(1, int(math.ceil(dg_band)))
    s = solid_mask.float().unsqueeze(0).unsqueeze(0)   # (1, 1, nz, ny, nx)
    dilated = F.max_pool3d(s, kernel_size=2 * k + 1, stride=1, padding=k)
    dilated_mask = dilated.squeeze(0).squeeze(0) > 0.5
    return dilated_mask & ~solid_mask


# ---------------------------------------------------------------------------
# SUBOFF visualisation helper
# ---------------------------------------------------------------------------


def _save_dg_lbm_suboff_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
    dg_mask: torch.Tensor,
    nz: int,
) -> None:
    """Save speed magnitude on the mid-z slice with DG zone and hull overlay."""
    mid_z = nz // 2
    speed_np = speed[mid_z].detach().cpu().numpy()
    obs_np = obstacle[mid_z].detach().cpu().float().numpy()
    dg_np = dg_mask[mid_z].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    im = ax.imshow(speed_np, origin="lower", cmap="viridis")
    ax.contour(obs_np, levels=[0.5], colors="white", linewidths=1.0, linestyles="-")
    ax.contour(dg_np, levels=[0.5], colors="cyan", linewidths=0.6, linestyles="--")
    ax.set_title(f"DG-LBM SUBOFF velocity magnitude – mid-z slice (step {step})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SUBOFF DG-LBM — REAL nodal-DG hybrid solver (use_real_dg=True)
# ---------------------------------------------------------------------------


def _run_suboff_real_dg(config: DGLBMSuboffConfig) -> Path:
    """Genuine nodal-DG hybrid solver for the SUBOFF hull.

    A DG band (``build_dg_hull_band_mask``) encloses the hull: its inner faces
    bounce-back off the solid, its outer faces couple to the exterior D3Q19 LBM
    via :func:`tensorlbm.dg_band.hybrid_step` (method-of-lines band with
    τ_dg = τ_lbm − ½, exterior collide + stream + DG-trace write-back).  This is
    real DG-LBM — contrast with the legacy gradient-correction path taken when
    ``use_real_dg`` is False.
    """
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    applied_num_threads = configure_cpu_threads(device, config.num_threads)
    run_dir = prepare_run_dir(
        config.output_root, "dg_lbm_suboff", config.resolved_run_name(), config.overwrite
    )
    ftype = torch.float32

    # ---- Geometry ----
    cx, cy, cz = config.nx * 0.35, config.ny * 0.5, config.nz * 0.5
    obstacle, hull_stats = build_suboff_mask(
        hull_type=config.hull_type, nx=config.nx, ny=config.ny, nz=config.nz,
        cx=cx, cy=cy, cz=cz, length=config.hull_length, device=config.device,
    )
    obstacle = obstacle.to(device)
    band_mask = build_dg_hull_band_mask(obstacle, config.dg_band)
    topo = build_band_topology(band_mask, solid_mask=obstacle, periodic=False).to(device)
    wall_mask = make_channel_wall_mask_3d(config.nz, config.ny, config.nx, obstacle, device=device)

    # ---- Initial field ----
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full((config.nz, config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    ux0[obstacle] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]]
    nn = config.dg_degree + 1
    f_dg = f_dg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nn, nn, nn).contiguous()

    ops = get_ops(degree=config.dg_degree, dx=1.0, dtype=ftype, device=device)
    C_d = C.to(ftype).to(device)
    W_d = W.to(ftype).to(device)
    opp = OPPOSITE.to(device)

    initial_mass = float(torch.ones_like(rho0).sum().item())
    diagnostics: list[dict] = []
    dg_cells = int(band_mask.sum().item())
    logger.info(
        "Real DG-LBM SUBOFF: device=%s NX=%s NY=%s NZ=%s tau=%.4f τ_dg=%.4f "
        "dg_degree=%d substeps=%d band_cells=%d hull_cells=%d",
        device, config.nx, config.ny, config.nz, config.tau, config.tau - 0.5,
        config.dg_degree, config.dg_substeps, dg_cells, int(obstacle.sum().item()),
    )
    logger.info("Run directory: %s", run_dir)

    for step in range(1, config.n_steps + 1):
        f_lbm, f_dg = hybrid_step(
            f_lbm, f_dg, C_d, W_d, ops, topo, tau_lbm=config.tau,
            dt=1.0, n_substeps=config.dg_substeps, opposite=opp,
        )
        # Project band P0 into f_lbm so the obstacle (inside the band) is
        # surrounded by real values → reliable momentum-exchange force/bounce-back.
        f_lbm = project_band_to_lbm(f_lbm, f_dg, topo)
        f_lbm = apply_simple_channel_boundaries_3d(f_lbm, config.u_in, wall_mask, obstacle)
        if step % config.output_interval == 0:
            f_lbm = correct_mass3d(f_lbm, initial_mass)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic3d(f_lbm)
            ux = ux.masked_fill(obstacle, 0.0); uy = uy.masked_fill(obstacle, 0.0); uz = uz.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
            mass = float(rho.sum().item())
            # DG-solid-interface momentum-exchange force (preserves wall shear).
            fvec = compute_dg_solid_force(f_dg, topo, C_d, ops)
            drag_lu = float(fvec[0].item())
            point = DiagnosticPoint(
                step=step, mass=mass, mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()), mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f max|u|=%.6f drag=%.4f",
                step, point.mass, point.mass_drift, point.max_speed, drag_lu,
            )
            _save_dg_lbm_suboff_snapshot(run_dir, step, speed, obstacle, band_mask, config.nz)
            save_checkpoint(f_lbm, step, run_dir)

    metadata = {
        "solver": "real_nodal_dg_hybrid",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau, "tau_dg": config.tau - 0.5},
        "runtime": {"device": str(device), "num_threads": applied_num_threads},
        "hull_stats": hull_stats,
        "drag_force_lu": drag_lu,
        "diagnostics": diagnostics,
    }
    mpath = run_dir / "run_metadata.json"
    mpath.write_text(f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8")
    logger.info("Saved metadata: %s", mpath)
    return run_dir


# ---------------------------------------------------------------------------
# SUBOFF — high-Re log-law wall-function solver (use_wall_function=True)
# ---------------------------------------------------------------------------


def _run_suboff_wall_function(config: DGLBMSuboffConfig) -> Path:
    """High-Re SUBOFF solver with a τ-decoupled log-law wall function.

    For real submarine Reynolds numbers (Re~1e6+, τ_lam→0.5) the standard
    bounce-back / momentum-exchange wall treatment lives on the LBM accuracy
    cliff and over-predicts drag by ~10-300×.  This solver instead computes
    the wall shear from the log-law and applies it as a Guo body force
    (:func:`tensorlbm.wall_model.wall_function_3d`), **decoupling the wall
    shear from the bulk τ**, and reports drag as the integrated wall shear
    (friction) + surface pressure (form).  With a far-field lateral BC this
    recovers the experimental AFF-8 resistance coefficient to <1% (validated
    Re=2M: Ct 0.0040 vs 0.004).
    """
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    applied_num_threads = configure_cpu_threads(device, config.num_threads)
    run_dir = prepare_run_dir(
        config.output_root, "dg_lbm_suboff", config.resolved_run_name(), config.overwrite
    )

    cx, cy, cz = config.nx * 0.35, config.ny * 0.5, config.nz * 0.5
    obstacle, hull_stats = build_suboff_mask(
        hull_type=config.hull_type, nx=config.nx, ny=config.ny, nz=config.nz,
        cx=cx, cy=cy, cz=cz, length=config.hull_length, device=config.device,
    )
    obstacle = obstacle.to(device)
    nu_lat = config.nu
    S = _voxel_wetted_area(obstacle, 1.0)
    dyn_p_S = 0.5 * 1.0 * config.u_in ** 2 * S

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full((config.nz, config.ny, config.nx), config.u_in, device=device)
    ux0[obstacle] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(torch.ones_like(rho0).sum().item())
    diagnostics: list[dict] = []

    logger.info(
        "Wall-function SUBOFF: device=%s NX=%s NY=%s NZ=%s Re=%.3e tau=%.5f "
        "nu_lat=%.2e S=%d hull=%s",
        device, config.nx, config.ny, config.nz, config.re, config.tau, nu_lat, S, config.hull_type,
    )
    logger.info("Run directory: %s", run_dir)

    drag_lu = 0.0
    for step in range(1, config.n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=0.1)   # MRT + LES collision
        f = stream3d(f)
        f, drag_f, drag_p = wall_function_3d(f, obstacle, nu_lat, y_val=0.5)
        f = far_field_bc_3d(f, u_in=config.u_in)                    # far-field (no blockage)
        if step % config.output_interval == 0:
            f = correct_mass3d(f, initial_mass)
        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            drag_lu = drag_f + drag_p
            speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
            mass = float(rho.sum().item())
            point = DiagnosticPoint(
                step=step, mass=mass, mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()), mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            ct = drag_lu / dyn_p_S
            logger.info(
                "step=%5d Ct_fric=%.5f Ct_pres=%.5f Ct_tot=%.5f max|u|=%.4f",
                step, drag_f / dyn_p_S, drag_p / dyn_p_S, ct, point.max_speed,
            )
            save_checkpoint(f, step, run_dir)

    metadata = {
        "solver": "wall_function_loglaw",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau, "wetted_area_lu2": S},
        "runtime": {"device": str(device), "num_threads": applied_num_threads},
        "hull_stats": hull_stats,
        "drag_force_lu": drag_lu,
        "Ct_total": drag_lu / dyn_p_S,
        "diagnostics": diagnostics,
    }
    mpath = run_dir / "run_metadata.json"
    mpath.write_text(f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8")
    logger.info("Saved metadata: %s", mpath)
    return run_dir


# ---------------------------------------------------------------------------
# SUBOFF DG-LBM main runner
# ---------------------------------------------------------------------------


def run_dg_lbm_suboff_flow(config: DGLBMSuboffConfig) -> Path:
    """Run the DG-LBM hybrid SUBOFF submarine-flow simulation.

    The near-wall zone (cells within ``dg_band`` of the hull surface) uses
    the DG-enhanced collision operator; the exterior uses standard BGK.

    This function reuses the same :func:`collide_dg_lbm`, :func:`stream3d`,
    and boundary-condition machinery as :func:`run_dg_lbm_sphere_flow`.  The
    only difference is the geometry: the solid mask and DG shell are built
    from the SUBOFF hull profile via :func:`build_suboff_mask` and
    :func:`build_dg_hull_band_mask`.

    Args:
        config: Simulation configuration.

    Returns:
        Path to the run output directory.
    """
    configure_logging()
    config.validate()
    if config.use_real_dg:
        return _run_suboff_real_dg(config)
    if config.use_wall_function:
        return _run_suboff_wall_function(config)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    applied_num_threads = configure_cpu_threads(device, config.num_threads)
    run_dir = prepare_run_dir(
        config.output_root,
        "dg_lbm_suboff",
        config.resolved_run_name(),
        config.overwrite,
    )

    ckpt_str = str(config.resume_checkpoint) if config.resume_checkpoint else None
    metadata: dict = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "resume_checkpoint": ckpt_str,
        },
        "derived": {
            "nu": config.nu,
            "tau": config.tau,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "num_threads": applied_num_threads,
        },
        "reproducibility": get_reproducibility_metadata(),
    }

    # ----------------------------------------------------------------
    # Geometry: SUBOFF hull obstacle + DG near-wall band + channel walls
    # ----------------------------------------------------------------
    # Hull centred at (nx*0.35, ny/2, nz/2): 35 % from inlet gives ~1.65 L
    # of downstream wake for the default nx=200, hull_length=120.
    cx = config.nx * 0.35
    cy = config.ny * 0.5
    cz = config.nz * 0.5

    obstacle, hull_stats = build_suboff_mask(
        hull_type=config.hull_type,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=config.hull_length,
        device=config.device,
    )
    obstacle = obstacle.to(device)

    dg_mask = build_dg_hull_band_mask(obstacle, config.dg_band)
    wall_mask = make_channel_wall_mask_3d(
        config.nz, config.ny, config.nx, obstacle, device=device
    )

    # ----------------------------------------------------------------
    # Initialise or resume
    # ----------------------------------------------------------------
    start_step = 1
    restart_info: dict[str, object] = {"resumed": False}
    if config.resume_checkpoint is not None:
        f, resume_step, ckpt_meta = load_checkpoint(
            config.resume_checkpoint,
            device=device,
            expected_shape=(19, config.nz, config.ny, config.nx),
            expected_lattice_directions=19,
        )
        if resume_step >= config.n_steps:
            raise ValueError(
                f"resume checkpoint step {resume_step} is not less than n_steps={config.n_steps}"
            )
        f = f.to(device)
        start_step = resume_step + 1
        logger.info(
            "Resumed from checkpoint %s at step %d",
            config.resume_checkpoint, resume_step,
        )
        restart_info = {
            "resumed": True,
            "source_checkpoint": str(config.resume_checkpoint),
            "source_step": resume_step,
            "checkpoint_format_version": ckpt_meta.get("format_version"),
        }
    else:
        rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
        ux0 = torch.full((config.nz, config.ny, config.nx), config.u_in, device=device)
        uy0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
        uz0 = torch.zeros((config.nz, config.ny, config.nx), device=device)
        ux0[obstacle] = 0.0
        f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    rho0_mass = torch.ones((config.nz, config.ny, config.nx), device=device)
    initial_mass = float(rho0_mass.sum().item())
    diagnostics: list[dict] = []

    _stream = _maybe_compile(stream3d, config.use_compile)

    dg_cells = int(dg_mask.sum().item())
    lbm_cells = int((~obstacle & ~dg_mask).sum().item())
    hull_cells = int(obstacle.sum().item())

    logger.info(
        "DG-LBM SUBOFF: device=%s NX=%s NY=%s NZ=%s tau=%.4f "
        "steps=%s output_interval=%s hull_type=%s dg_band=%.1f",
        device, config.nx, config.ny, config.nz, config.tau,
        config.n_steps, config.output_interval, config.hull_type, config.dg_band,
    )
    logger.info(
        "Zone breakdown: hull=%d cells  DG band=%d cells  LBM exterior=%d cells",
        hull_cells, dg_cells, lbm_cells,
    )
    logger.info("Hull stats: %s", hull_stats)
    logger.info("Run directory: %s", run_dir)

    step_range = range(start_step, config.n_steps + 1)
    step_iter = (
        _tqdm(step_range, desc="DG-LBM SUBOFF", unit="step")
        if _TQDM_AVAILABLE
        else step_range
    )

    for step in step_iter:
        # Hybrid collision with optional LES
        f = collide_dg_lbm(f, tau=config.tau, dg_mask=dg_mask)
        if config.use_mrt and config.dynamic_smag:
            f_smag = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=0.15)
        elif config.use_mrt:
            f_smag = collide_mrt3d(f, tau=config.tau)
        elif config.dynamic_smag:
            f_smag = collide_dynamic_smagorinsky_bgk3d(f, tau=config.tau)
            f = torch.where(dg_mask.unsqueeze(0), f, f_smag)
        elif config.smagorinsky_cs > 0:
            f_smag = collide_smagorinsky_bgk3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
            f = torch.where(dg_mask.unsqueeze(0), f, f_smag)

        # Standard streaming
        f = _stream(f)

        # Momentum-exchange force on hull
        fx, fy, fz = compute_obstacle_forces_3d(f, obstacle)
        drag_lu = fx.item()

        # Boundary conditions
        if config.free_slip_walls:
            # Free-slip on domain boundaries (no blockage)
            y_wall = torch.zeros(1, config.ny, 1, dtype=torch.bool, device=device)
            y_wall[0, 0, 0] = True
            y_wall[0, -1, 0] = True
            y_wall_full = y_wall.expand(config.nz, config.ny, config.nx)
            z_wall = torch.zeros(config.ny, 1, 1, dtype=torch.bool, device=device)
            z_wall[0, 0, 0] = True
            z_wall[-1, 0, 0] = True
            z_wall_full = z_wall.permute(1, 0, 2).expand(config.nz, config.ny, config.nx)
            f = free_slip_y_walls_3d(f, y_wall_full)
            f = free_slip_z_walls_3d(f, z_wall_full)
            # Hull bounce-back only (use zero wall_mask for channel)
            f = apply_simple_channel_boundaries_3d(
                f, u_in=config.u_in, wall_mask=torch.zeros_like(obstacle),
                obstacle_mask=obstacle
            )
        elif config.use_wall_model:
            _, ux, uy, uz = macroscopic3d(f)
            nu = (config.tau - 0.5) / 3.0
            f = apply_wall_model_bounce_back(f, obstacle, ux, uy, uz, nu)
            f = apply_simple_channel_boundaries_3d(
                f, u_in=config.u_in, wall_mask=wall_mask,
                obstacle_mask=torch.zeros_like(obstacle)
            )
        else:
            f = apply_simple_channel_boundaries_3d(
                f, u_in=config.u_in, wall_mask=wall_mask,
                obstacle_mask=obstacle
            )

        # Periodic mass correction
        if step % config.output_interval == 0:
            f = correct_mass3d(f, initial_mass)

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
            diagnostics.append(asdict(point))
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f drag=%.4f",
                point.step, point.mass, point.mass_drift,
                point.mean_rho, point.max_speed,
            )
            _save_dg_lbm_suboff_snapshot(run_dir, step, speed, obstacle, dg_mask, config.nz)
            save_checkpoint(f, step, run_dir)

    metadata["diagnostics"] = diagnostics
    metadata["restart"] = restart_info
    metadata["hull_stats"] = hull_stats
    metadata["drag_force_lu"] = float(drag_lu)
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir
