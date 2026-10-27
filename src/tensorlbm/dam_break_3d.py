"""3D dam-break benchmark using multiphase LBM (D3Q19).

Simulates a water column collapse with an obstacle, following the classic
Koshizuka & Oka (1996) / Kleefsman et al. (2005) experimental setup.

Supports:
* ``"cg"`` – Color-Gradient (recommended, most stable in 3D)
* ``"fe"`` – Free-Energy binary-fluid

Benchmark diagnostics
----------------------
* Leading edge (x-front) of water tracked over time
* Comparison with Koshizuka & Oka (1996) experimental data
* Pressure probe at obstacle face (P1)

References
----------
Koshizuka & Oka (1996) Nucl. Sci. Eng. 123, 421-434
Kleefsman et al. (2005) J. Comput. Phys. 206, 363-393
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import torch

from .boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from .d3q19 import equilibrium3d
from .free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_fill_rectangular,
    init_flags_from_fill,
    init_mass_from_fill,
)
from .multiphase3d import (
    collide_cg_mrt_3d,
    collide_sc_mrt_3d,
    collide_sc_two_component_3d,
    color_gradient_step_3d,
    free_energy_step_3d,
    init_free_energy_g_3d,
    init_hydrostatic_pressure_3d,
)
from .solver3d import stream3d
from .utils import (
    prepare_run_dir,
    resolve_device,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

Model3D = Literal["sc", "cg", "fe", "fs"]


def _topology_event(
    previous: tuple[int, int],
    current: tuple[int, int],
    conversion: float,
    redistribution: float,
) -> bool:
    """Whether topology work occurred, including count-neutral exchanges."""
    return current != previous or abs(conversion) + abs(redistribution) > 0.0


def _linear_drift_slope(history: list[tuple[int, float]], window: int) -> tuple[float, int]:
    """Terminal/rolling least-squares slope of cumulative drift versus step."""
    points = history[-window:]
    count = len(points)
    if count < 2:
        return 0.0, count
    mean_step = sum(point[0] for point in points) / count
    mean_drift = sum(point[1] for point in points) / count
    denominator = sum((point[0] - mean_step) ** 2 for point in points)
    return (
        sum((point[0] - mean_step) * (point[1] - mean_drift) for point in points) / denominator,
        count,
    )


def _topology_drift_violates(event: bool, normalized_drift: float, tolerance: float) -> bool:
    """Apply the topology gate for both count-changing and neutral work."""
    return event and abs(normalized_drift) > tolerance


@dataclass(frozen=True)
class DamBreak3DConfig:
    """3D dam-break with obstacle (Koshizuka & Oka 1996).

    Geometry (lattice units)
    ------------------------
    Domain: (nx, ny, nz).  Water column fills x ∈ [0, dam_width),
    y ∈ [0, fill_height), full z.

    Obstacle: (obs_x0, obs_y0, 0) to (obs_x1, obs_y1, nz), centered in z.

    Physics
    -------
    rho_heavy / rho_light: density ratio (water/air ≈ 2.0 for CG stability)
    A: surface tension coefficient (CG model)
    gravity: body force in -y direction
    """

    # Domain
    nx: int = 200
    ny: int = 80
    nz: int = 80
    # Water column
    dam_width: int = 80
    fill_height: int = 80  # must equal ny for Koshizuka-style full-height dam
    # Obstacle (0 = no obstacle).  SC model is unstable with obstacles in 3D;
    # the Koshizuka & Oka front-position data is from the no-obstacle case.
    obs_x0: int = 0
    obs_x1: int = 0
    obs_y0: int = 0
    obs_y1: int = 0
    # Physics
    model: Model3D = "cg"
    rho_heavy: float = 1.0
    rho_light: float = 0.1  # density ratio 10:1 — achievable with MRT+SGS+Guo
    # Surface tension: CG perturbation coefficient for cg/sc; Laplace sigma
    # inside the ABB gas density for fs (Thürey 2007 §4.2).
    A: float = 0.005
    G_sc: float = 0.9  # SC coupling (>0 for two-component separation)
    tau: float = 0.8  # τ=0.8 for CG with 10:1; τ=0.6 possible with MRT+SGS
    gravity: float = 8e-5
    # Free-slip on y-walls (waLBerla pattern — reduces spurious currents)
    free_slip_y: bool = True
    # Free-slip on ALL walls (dam-break validation default, waLBerla
    # DamBreakRectangular.prm uses FreeSlip everywhere).  For fs this slips
    # every solid wall including any obstacle; keep False with an obstacle.
    free_slip_all: bool = False
    # Periodic z (no z-walls): the Martin & Moyce 2D-like setup.  Combines
    # with a small nz and free_slip_all=True.
    periodic_z: bool = False
    # Quantitative front-position gate vs the matching experiment (Koshizuka &
    # Oka 1996 by default; Martin & Moyce 1952 when periodic_z).  Mean
    # advance-based relative error; None = record only, never fail.
    front_error_tolerance: float | None = None
    # MRT + Smagorinsky (waLBerla/OpenLB production stack)
    collision: str = "mrt_smag"  # "bgk" | "mrt" | "mrt_smag"
    C_s: float = 0.1  # Smagorinsky constant (0.1 = waLBerla default)
    use_guo: bool = True  # Guo second-order forcing (waLBerla GuoField)
    # Hydrostatic IC (waLBerla initHydrostaticPressure)
    hydrostatic_init: bool = True
    # Time-stepping
    n_steps: int = 5000
    output_interval: int = 250
    # Output
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False
    # Bookkeeping-residual tolerance (RELATIVE to the initial tracked mass):
    # the Körner exchange is attributed in full, so the residual is an
    # arithmetic identity (~1e-7 relative in float32); anything larger flags a
    # bookkeeping bug, not physics.  Not a physical/PV conservation claim.
    free_surface_unexplained_tolerance: float = 1.0e-6
    # Deprecated: the paired bulk debit no longer exists (kept for config
    # compatibility; the gate is retired).
    free_surface_paired_tolerance: float = 1.0e-5
    # The Körner L/I exchange is a one-sided source by construction, so the
    # tracked mass legitimately breathes at the ~1% level (waLBerla-quality
    # runs stay within a few percent).  5% is the fail-closed bound.
    free_surface_relative_cumulative_drift_tolerance: float = 5.0e-2
    # Deprecated advisory diagnostics (recorded, never fail-closed): the slope
    # and topology-normalized gates tripped on physical transients.
    free_surface_relative_drift_slope_tolerance: float = 1.0e-5
    free_surface_topology_normalized_drift_tolerance: float = 5.0e-2
    free_surface_drift_slope_window: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 32 or self.ny < 16 or self.nz < 16:
            raise ValueError("nx/ny/nz too small")
        if self.dam_width <= 0 or self.dam_width >= self.nx:
            raise ValueError("dam_width must be in (0, nx)")
        if self.fill_height <= 0 or self.fill_height > self.ny - 1:
            raise ValueError("fill_height must be in (0, ny-1)")
        if self.rho_heavy <= self.rho_light:
            raise ValueError("rho_heavy > rho_light")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} <= 0.5")
        if (
            self.free_surface_unexplained_tolerance < 0
            or self.free_surface_paired_tolerance < 0
            or self.free_surface_relative_cumulative_drift_tolerance < 0
            or self.free_surface_relative_drift_slope_tolerance < 0
            or self.free_surface_topology_normalized_drift_tolerance < 0
        ):
            raise ValueError("free-surface accounting tolerances must be non-negative")
        if self.free_surface_drift_slope_window < 2:
            raise ValueError("free_surface_drift_slope_window must be at least two")

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"dam3d_{self.model}_nx{self.nx}_dw{self.dam_width}_steps{self.n_steps}"


# ---------------------------------------------------------------------------
# Koshizuka & Oka (1996) reference data — leading edge position
# ---------------------------------------------------------------------------
# Dimensionless time T = t * sqrt(2g / L), position X = x_front / L
# where L = initial column width
_KOSHIZUKA_FRONT: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.5, 1.2),
    (1.0, 2.0),
    (1.5, 2.8),
    (2.0, 3.2),
    (2.5, 3.8),
    (3.0, 4.2),
    (3.5, 4.6),
    (4.0, 5.0),
    (4.5, 5.2),
    (5.0, 5.4),
    (5.5, 5.6),
    (6.0, 5.8),
]

# ---------------------------------------------------------------------------
# Martin & Moyce (1952) reference data — surge front for rectangular column
# ---------------------------------------------------------------------------
# T = t * sqrt(g / a), Z = x_front / a  where a = column half-width
# (waLBerla DamBreakRectangular validation target)
_MARTIN_MOYCE_FRONT: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.5, 1.1),
    (1.0, 1.4),
    (1.5, 1.8),
    (2.0, 2.2),
    (2.5, 2.7),
    (3.0, 3.1),
    (3.5, 3.5),
    (4.0, 3.8),
    (4.5, 4.1),
    (5.0, 4.3),
    (5.5, 4.5),
    (6.0, 4.7),
]

_MARTIN_MOYCE_HEIGHT: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.5, 0.92),
    (1.0, 0.78),
    (1.5, 0.65),
    (2.0, 0.55),
    (2.5, 0.45),
    (3.0, 0.37),
    (3.5, 0.30),
    (4.0, 0.25),
    (4.5, 0.20),
    (5.0, 0.16),
    (5.5, 0.13),
    (6.0, 0.11),
]


def _reference_advance(t_star: float, periodic_z: bool) -> tuple[float, str] | None:
    """Reference front advance (in units of L) at dimensionless time t*.

    Koshizuka & Oka (walled 3D tank): advance/L = X(T) - 1 with
    T = t*sqrt(2g/L).  Martin & Moyce (2D-like, ``periodic_z``): their
    T = t*sqrt(g/a) with a = L/2 coincides with t*, and advance/L =
    (Z(T) - 1)/2.  Returns ``None`` outside the tabulated range.
    """
    table = _MARTIN_MOYCE_FRONT if periodic_z else _KOSHIZUKA_FRONT
    name = "martin_moyce_1952" if periodic_z else "koshizuka_oka_1996"
    if t_star < table[0][0] or t_star > table[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(table[:-1], table[1:]):
        if t0 <= t_star <= t1:
            frac = 0.0 if t1 == t0 else (t_star - t0) / (t1 - t0)
            value = v0 + frac * (v1 - v0)
            advance = (value - 1.0) / 2.0 if periodic_z else (value - 1.0)
            return advance, name
    return None


def front_advance_error(
    front_series: list[tuple[int, float, float, float]], periodic_z: bool
) -> tuple[float, int, str]:
    """Mean advance-based relative front error vs the matching experiment.

    Compares the simulated advance (x_front - L) against the interpolated
    reference advance, normalized by max(|reference|, L/2) so the
    initiation phase (zero advance) cannot blow up.  Returns
    ``(mean_error, n_samples, reference_name)``; ``(nan, 0, name)`` when no
    sample falls in the tabulated range.
    """
    name = "martin_moyce_1952" if periodic_z else "koshizuka_oka_1996"
    errors: list[float] = []
    for _step, t_star, x_star, _mean_rho in front_series:
        hit = _reference_advance(t_star, periodic_z)
        if hit is None:
            continue
        ref_advance, _ = hit
        sim_advance = x_star - 1.0
        errors.append(abs(sim_advance - ref_advance) / max(abs(ref_advance), 0.5))
    if not errors:
        return float("nan"), 0, name
    return sum(errors) / len(errors), len(errors), name


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def _free_slip_all_walls_3d(
    f: torch.Tensor,
    x_wall: torch.Tensor,
    y_wall: torch.Tensor,
    z_wall: torch.Tensor,
) -> torch.Tensor:
    """Specular (free-slip) reflection on every wall axis (D3Q19)."""
    f = free_slip_cells_3d(f, x_wall, axis=0)
    f = free_slip_cells_3d(f, y_wall, axis=1)
    f = free_slip_cells_3d(f, z_wall, axis=2)
    return f


def _free_slip_bottom_3d(f: torch.Tensor, bottom_mask: torch.Tensor) -> torch.Tensor:
    """Specular (free-slip) reflection at y=0 and y=ny-1 for D3Q19.

    Uses the tensorized :func:`~tensorlbm.boundaries3d.free_slip_cells_3d`
    with axis=1 (y-wall mirroring), matching waLBerla's ``FreeSlip`` pattern.
    """
    return free_slip_cells_3d(f, bottom_mask, axis=1)


def _build_solid_mask(
    config: DamBreak3DConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build separate masks for solid cells and per-axis walls.

    Returns:
        ``(solid_mask, y_wall_mask, x_wall_mask, z_wall_mask)`` where:
          - solid_mask: all wall + obstacle cells (bounce-back)
          - *_wall_mask: only the face walls normal to that axis (FreeSlip)
        With ``periodic_z`` no z-walls are built (streaming wraps in z).
    """
    nx, ny, nz = config.nx, config.ny, config.nz
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    y_wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    x_wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    z_wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

    # z-face walls (skipped when periodic_z: streaming wraps instead)
    if not config.periodic_z:
        solid[0, :, :] = True  # z=0
        solid[-1, :, :] = True  # z=nz-1
        z_wall[0, :, :] = True
        z_wall[-1, :, :] = True

    # x-face walls
    solid[:, :, 0] = True  # x=0
    solid[:, :, -1] = True  # x=nx-1
    x_wall[:, :, 0] = True
    x_wall[:, :, -1] = True

    # y-face walls: separate mask for FreeSlip option
    y_wall[:, 0, :] = True  # y=0  (bottom)
    y_wall[:, -1, :] = True  # y=ny-1 (top)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Obstacle (if enabled) — always bounce-back
    if config.obs_x1 > config.obs_x0:
        solid[:, config.obs_y0 : config.obs_y1, config.obs_x0 : config.obs_x1] = True

    return solid, y_wall, x_wall, z_wall


def _init_cg_3d(
    config: DamBreak3DConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Colour-gradient initialisation: sharp water/air split in 3D.

    Water fills the full height (y ∈ [0, fill_height]), matching the 2D
    dam-break convention where the column spans the entire domain height.
    Only the x-direction has a smooth tanh interface at the dam edge.
    """
    nx, ny, nz = config.nx, config.ny, config.nz
    xs = torch.arange(nx, dtype=torch.float32, device=device)

    # Smooth tanh interface at dam edge (x-direction only, like 2D)
    interface_width = 3.0
    prof = 0.5 * (1.0 + torch.tanh((config.dam_width - 1 - xs) / interface_width))
    # Uniform in y and z
    prof = prof.view(1, 1, nx).expand(nz, config.fill_height, nx)

    frac = 0.05
    rho_water_full = config.rho_heavy * prof + config.rho_heavy * frac * (1.0 - prof)
    rho_air_full = config.rho_light * frac * prof + config.rho_light * (1.0 - prof)

    # Pad with air above fill_height
    rho_water = config.rho_heavy * frac * torch.ones((nz, ny, nx), device=device)
    rho_air = config.rho_light * torch.ones((nz, ny, nx), device=device)
    rho_water[:, : config.fill_height, :] = rho_water_full
    rho_air[:, : config.fill_height, :] = rho_air_full

    zero = torch.zeros((nz, ny, nx), device=device)
    return equilibrium3d(rho_water, zero, zero, zero, device=device), equilibrium3d(
        rho_air, zero, zero, zero, device=device
    )


def _init_sc_3d(
    config: DamBreak3DConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SC two-component initialisation: smooth split at dam edge."""
    nx, ny, nz = config.nx, config.ny, config.nz
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    iw = 3.0
    prof = 0.5 * (1.0 + torch.tanh((config.dam_width - 1 - xs) / iw))
    prof = prof.view(1, 1, nx).expand(nz, config.fill_height, nx)

    frac = 0.15
    rho2_full = config.rho_heavy * prof + config.rho_heavy * frac * (1.0 - prof)
    rho1_full = config.rho_light * frac * prof + config.rho_light * (1.0 - prof)

    rho2 = config.rho_heavy * frac * torch.ones((nz, ny, nx), device=device)
    rho1 = config.rho_light * torch.ones((nz, ny, nx), device=device)
    rho2[:, : config.fill_height, :] = rho2_full
    rho1[:, : config.fill_height, :] = rho1_full

    zero = torch.zeros((nz, ny, nx), device=device)
    return equilibrium3d(rho1, zero, zero, zero, device=device), equilibrium3d(
        rho2, zero, zero, zero, device=device
    )


def _init_fe_3d(
    config: DamBreak3DConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-energy initialisation: phi=+1 in water, phi=-1 in air."""
    nx, ny, nz = config.nx, config.ny, config.nz
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    ys = torch.arange(ny, dtype=torch.float32, device=device)

    iw = 3.0
    prof_x = 0.5 * (1.0 + torch.tanh((config.dam_width - 1 - xs) / iw))
    prof_y = 0.5 * (1.0 + torch.tanh((config.fill_height - 1 - ys) / iw))
    prof = prof_x.view(1, 1, nx) * prof_y.view(1, ny, 1)
    prof = prof.expand(nz, ny, nx)

    phi = 2.0 * prof - 1.0
    rho = torch.ones((nz, ny, nx), device=device)
    zero = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho, zero, zero, zero)
    g = init_free_energy_g_3d(phi, zero, zero, zero)
    return f, g


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _find_front_x_3d(
    rho_water: torch.Tensor,
    rho_air: torch.Tensor,
    solid: torch.Tensor,
    threshold: float = 0.45,
    flags: torch.Tensor | None = None,
) -> float:
    """Find the rightmost x-column where water fraction exceeds threshold.

    For free-surface (flags not None): uses LIQUID+INTERFACE flags directly.
    For multiphase: uses phi = rho_water/(rho_water+rho_air).
    """
    if flags is not None:
        # Free-surface: use flag field
        liquid_or_iface = (flags == LIQUID) | (flags == INTERFACE)
        water_cols = liquid_or_iface.any(dim=1).any(dim=0).int()
        front = water_cols.nonzero(as_tuple=True)[0]
        return float(front.max().item()) if front.numel() > 0 else 0.0

    phi = rho_water / (rho_water + rho_air + 1e-12)
    phi_mean = phi[:, 1:-1, 1:-1].mean(dim=(0, 1))
    water_cols = (phi_mean > threshold).nonzero(as_tuple=True)[0]
    return float(water_cols.max().item()) if water_cols.numel() > 0 else 0.0


def _probe_pressure(
    f1: torch.Tensor,
    f2: torch.Tensor,
    obs_x0: int,
    obs_y0: int,
    obs_y1: int,
) -> float:
    """Pressure at obstacle face (P1 probe, center of obstacle face)."""
    # Use CG rho sum as pressure proxy (ideal gas EOS: p = cs² * rho)
    rho = f1.sum(dim=0) + f2.sum(dim=0)
    z_mid = rho.shape[0] // 2
    y_mid = (obs_y0 + obs_y1) // 2
    if 0 <= z_mid < rho.shape[0] and 0 <= y_mid < rho.shape[1] and obs_x0 > 0:
        rho_local = rho[z_mid, y_mid, obs_x0 - 1].item()
        return rho_local / 3.0  # p = cs² * rho
    return float("nan")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_dam_break_3d(config: DamBreak3DConfig) -> Path:
    """Run a 3D dam-break simulation and return the output directory."""
    device = resolve_device(config.device)
    config.validate()

    nx, ny, nz = config.nx, config.ny, config.nz
    run_dir = prepare_run_dir(
        config.output_root,
        "dam_break_3d",
        config.resolved_run_name(),
        config.overwrite,
    )

    # Build geometry
    solid, y_wall, x_wall, z_wall = _build_solid_mask(config, device)

    # For free-surface, use y_wall_mask directly
    if config.model == "fs":
        solid = y_wall.clone()

    # Initialise fluids
    if config.model == "cg":
        f_water, f_air = _init_cg_3d(config, device)
    elif config.model == "sc":
        f_water, f_air = _init_sc_3d(config, device)
    elif config.model == "fe":
        f_fe, g_fe = _init_fe_3d(config, device)
    elif config.model == "fs":
        # Free-surface LBM: single-phase with fill level tracking
        fill, y_wall_mask = init_fill_rectangular(
            nz,
            ny,
            nx,
            column_width=float(config.dam_width),
            column_height=float(config.fill_height),
            device=device,
        )
        if config.periodic_z:
            # Knock out the z-walls built by the initializer (streaming wraps).
            y_wall_mask = y_wall_mask.clone()
            y_wall_mask[0, :, :] = False
            y_wall_mask[-1, :, :] = False
        flags = init_flags_from_fill(fill, y_wall_mask)
        # Independent Körner mass ledger: initialize once and preserve the
        # returned state rather than reconstructing it from fill each step.
        mass = init_mass_from_fill(fill, flags, config.rho_heavy)
        active = (flags == LIQUID) | (flags == INTERFACE)
        zero_f = torch.zeros((nz, ny, nx), device=device)
        f_water = equilibrium3d(
            torch.where(active, torch.ones((nz, ny, nx), device=device), zero_f),
            zero_f,
            zero_f,
            zero_f,
        )
        # For compatibility with the multi-model loop
        f_air = torch.zeros_like(f_water)
        solid = y_wall_mask  # Free-surface uses y_wall_mask as solid
    else:
        raise ValueError(f"Unknown model: {config.model}")

    # Non-dimensionalisation for comparison
    L = float(config.dam_width)  # characteristic length = column width
    g_lu = config.gravity
    t_scale = math.sqrt(2.0 * g_lu / L) if g_lu > 0 else 0.0

    # Hydrostatic initialisation (waLBerla initHydrostaticPressure pattern).
    # The fs model needs it too: without the rho = 1 + 3gh gradient the whole
    # column starts at uniform pressure and the first ~100 steps are an
    # unphysical acoustic transient.
    if config.hydrostatic_init and config.model == "fs":
        f_water = init_hydrostatic_pressure_3d(
            f_water,
            solid,
            gy=-config.gravity,
            water_height=float(config.fill_height),
            rho0=config.rho_heavy,
        )
        # Gas cells carry no populations in the free-surface model.
        f_water = torch.where(
            (flags == GAS).unsqueeze(0), torch.zeros_like(f_water), f_water
        )
    if config.hydrostatic_init and config.model in ("cg", "sc"):
        f_water = init_hydrostatic_pressure_3d(
            f_water,
            solid,
            gy=-config.gravity,
            water_height=float(config.fill_height),
        )
        if config.model == "cg":
            f_air = init_hydrostatic_pressure_3d(
                f_air,
                solid,
                gy=-config.gravity,
                water_height=float(config.fill_height),
            )

    print(
        f"Running dam-break 3D ({config.model.upper()}/{config.collision})  "
        f"device={device}  NX={nx} NY={ny} NZ={nz}  "
        f"L={L:.0f}  g={g_lu:.1e}  tau={config.tau}  "
        f"rho_h={config.rho_heavy} rho_l={config.rho_light}  "
        f"free_slip={config.free_slip_y}"
        + (f"  C_s={config.C_s}" if config.collision == "mrt_smag" else "")
        + ("  guo" if config.use_guo else "")
        + ("  hydro" if config.hydrostatic_init else "")
        + f"  steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    diagnostics: list[dict[str, object]] = []
    front_series: list[tuple[int, float, float]] = []
    fs_handoff: list[dict[str, object]] = []
    fs_runtime: dict[str, object] = {}
    fs_initial_topology: tuple[int, int] | None = None
    fs_initial_mass: float | None = None
    fs_previous_topology: tuple[int, int] | None = None
    fs_topology_events: list[dict[str, object]] = []
    fs_cumulative_drift_history: list[tuple[int, float]] = []
    fs_topology_changed = False
    gy = -config.gravity
    if config.model == "fs":
        fs_initial_topology = (int((flags == LIQUID).sum()), int((flags == INTERFACE).sum()))
        fs_previous_topology = fs_initial_topology
        fs_initial_mass = float(mass.sum().item())

    for step in range(1, config.n_steps + 1):
        # Collision + streaming
        if config.model == "sc":
            if config.collision == "mrt" or config.collision == "mrt_smag":
                C_s_val = config.C_s if config.collision == "mrt_smag" else 0.0
                f_water, f_air = collide_sc_mrt_3d(
                    f_water,
                    f_air,
                    G_12=config.G_sc,
                    tau=config.tau,
                    gy=gy,
                    solid_mask=solid,
                    C_s=C_s_val,
                    use_guo=config.use_guo,
                )
            else:
                f_water, f_air = collide_sc_two_component_3d(
                    f_water,
                    f_air,
                    G_12=config.G_sc,
                    tau1=config.tau,
                    tau2=config.tau,
                    gy=gy,
                    solid_mask=solid,
                    use_guo=config.use_guo,
                )
            f_water, f_air = stream3d(f_water), stream3d(f_air)
            # Bounce-back on x/z walls and obstacle; FreeSlip on y-walls
            f_water = bounce_back_cells_3d(f_water, solid)
            f_air = bounce_back_cells_3d(f_air, solid)
            if config.free_slip_all:
                f_water = _free_slip_all_walls_3d(f_water, x_wall, y_wall, z_wall)
                f_air = _free_slip_all_walls_3d(f_air, x_wall, y_wall, z_wall)
            elif config.free_slip_y:
                f_water = free_slip_cells_3d(f_water, y_wall, axis=1)
                f_air = free_slip_cells_3d(f_air, y_wall, axis=1)
            rho_heavy = f_air.sum(dim=0)  # component 2 = heavy
            rho_light = f_water.sum(dim=0)  # component 1 = light

        elif config.model == "cg":
            if config.collision == "mrt" or config.collision == "mrt_smag":
                C_s_val = config.C_s if config.collision == "mrt_smag" else 0.0
                f_water, f_air = collide_cg_mrt_3d(
                    f_water,
                    f_air,
                    tau=config.tau,
                    A=config.A,
                    gy=gy,
                    solid_mask=solid,
                    C_s=C_s_val,
                    use_guo=config.use_guo,
                )
            else:
                f_water, f_air = color_gradient_step_3d(
                    f_water,
                    f_air,
                    tau=config.tau,
                    A=config.A,
                    gy=gy,
                    solid_mask=solid,
                )
            f_water, f_air = stream3d(f_water), stream3d(f_air)
            f_water = bounce_back_cells_3d(f_water, solid)
            f_air = bounce_back_cells_3d(f_air, solid)
            if config.free_slip_all:
                f_water = _free_slip_all_walls_3d(f_water, x_wall, y_wall, z_wall)
                f_air = _free_slip_all_walls_3d(f_air, x_wall, y_wall, z_wall)
            elif config.free_slip_y:
                f_water = free_slip_cells_3d(f_water, y_wall, axis=1)
                f_air = free_slip_cells_3d(f_air, y_wall, axis=1)
            rho_heavy = f_water.sum(dim=0)
            rho_light = f_air.sum(dim=0)

        elif config.model == "fs":
            # Free-surface LBM step.  All five solver states are passed on.
            # rho_gas = rho_heavy: the gas is a pressure-matched ambient
            # (waLBerla BubbleModelConstantPressure / Pohl 2008), NOT a real
            # light fluid — the ABB reconstruction is consistent only then.
            f_water, fill, flags, mass, df = free_surface_step(
                f_water,
                fill,
                flags,
                solid,
                mass=mass,
                tau=config.tau,
                gy=gy,
                rho_liquid=config.rho_heavy,
                rho_gas=config.rho_heavy,
                surface_tension=config.A if config.A > 0 else 0.0,
                C_s=config.C_s if config.collision == "mrt_smag" else 0.0,
                free_slip_y=config.free_slip_y,
                y_wall_mask=y_wall,
                free_slip_all=config.free_slip_all,
                runtime_ledger=fs_runtime,
            )
            steps = fs_runtime["steps"]
            assert isinstance(steps, list) and steps
            quality = steps[-1]
            assert isinstance(quality, dict)
            finite = bool(
                torch.isfinite(f_water).all()
                and torch.isfinite(fill).all()
                and torch.isfinite(mass).all()
            )
            quality["finite"] = finite
            quality["flags_finite"] = True  # Integral flags have no NaN representation.
            topology = (int((flags == LIQUID).sum()), int((flags == INTERFACE).sum()))
            quality["liquid_cells"], quality["interface_cells"] = topology
            fs_topology_changed = fs_topology_changed or topology != fs_initial_topology
            assert fs_initial_mass is not None and fs_previous_topology is not None
            cumulative_drift = float(quality["mass_end"]) - fs_initial_mass
            conversion = float(quality["conversion"])
            redistribution = float(quality["redistribution"])
            conversion_redistribution_scale = abs(conversion) + abs(redistribution)
            # Attribute drift relative to the local topology work, rather than
            # subtracting physical conversion/redistribution from a mass delta.
            normalized_drift = (
                float(quality["mass_drift"]) / conversion_redistribution_scale
                if conversion_redistribution_scale > 0.0
                else float(quality["mass_drift"])
            )
            # A count-neutral exchange can still convert or redistribute mass.
            # Therefore topology work, not just net flag counts, defines an
            # event and activates its normalized-drift gate.
            topology_count_changed = topology != fs_previous_topology
            topology_event = _topology_event(
                fs_previous_topology, topology, conversion, redistribution
            )
            relative_cumulative_drift = cumulative_drift / fs_initial_mass
            fs_cumulative_drift_history.append((step, cumulative_drift))
            cumulative_drift_slope, regression_count = _linear_drift_slope(
                fs_cumulative_drift_history,
                config.free_surface_drift_slope_window,
            )
            relative_cumulative_drift_slope = cumulative_drift_slope / fs_initial_mass
            quality.update(
                {
                    "time": float(step),
                    "initial_mass": fs_initial_mass,
                    "instantaneous_mass_drift": float(quality["mass_drift"]),
                    "cumulative_drift": cumulative_drift,
                    "relative_cumulative_drift": relative_cumulative_drift,
                    "cumulative_drift_average_rate": cumulative_drift / step,
                    "relative_cumulative_drift_average_rate": relative_cumulative_drift / step,
                    "cumulative_drift_slope": cumulative_drift_slope,
                    "relative_cumulative_drift_slope": relative_cumulative_drift_slope,
                    "drift_slope_window_steps": regression_count,
                    "conversion_redistribution_normalized_drift": normalized_drift,
                    "topology_event": topology_event,
                    "topology_count_changed": topology_count_changed,
                    "liquid_cell_delta": topology[0] - fs_previous_topology[0],
                    "interface_cell_delta": topology[1] - fs_previous_topology[1],
                }
            )
            if topology_event:
                fs_topology_events.append(
                    {
                        "step": step,
                        "time": float(step),
                        "liquid_cells_before": fs_previous_topology[0],
                        "interface_cells_before": fs_previous_topology[1],
                        "liquid_cells_after": topology[0],
                        "interface_cells_after": topology[1],
                        "conversion": float(quality["conversion"]),
                        "redistribution": float(quality["redistribution"]),
                        "topology_count_changed": topology_count_changed,
                        "conversion_redistribution_normalized_drift": normalized_drift,
                    }
                )
            fs_previous_topology = topology
            violations: list[str] = []
            if not finite:
                violations.append("non-finite free-surface state")
            if int(quality["directLG"]) != 0:
                violations.append("direct liquid/gas link")
            # The bookkeeping residual is an arithmetic identity: judge it
            # relative to the initial tracked mass (float32 rounding sits at
            # ~1e-7 relative).
            relative_unexplained = (
                abs(float(quality["unexplained_residual"])) / fs_initial_mass
            )
            quality["relative_unexplained_residual"] = relative_unexplained
            if relative_unexplained > config.free_surface_unexplained_tolerance:
                violations.append("unexplained tracked-mass residual exceeds tolerance")
            if (
                abs(relative_cumulative_drift)
                > config.free_surface_relative_cumulative_drift_tolerance
            ):
                violations.append("relative cumulative tracked-mass drift exceeds tolerance")
            if violations:
                quality["quality_gate"] = "failed: " + "; ".join(violations)
                raise RuntimeError(
                    f"free-surface quality gate fail-closed at step {step}: {'; '.join(violations)}; "
                    f"record={quality}"
                )
            quality["quality_gate"] = "passed"
            fs_handoff.append(
                {
                    "step": step,
                    "f_shape": list(f_water.shape),
                    "fill_shape": list(fill.shape),
                    "flags_shape": list(flags.shape),
                    "mass_shape": list(mass.shape),
                    "df": float(df.item()),
                    "mass_is_independent": True,
                }
            )
            rho_heavy = f_water.sum(dim=0)
            rho_light = torch.zeros_like(rho_heavy)  # gas has no density field

        elif config.model == "fe":
            f_fe, g_fe = free_energy_step_3d(
                f_fe,
                g_fe,
                tau_f=config.tau,
                gy=gy,
                A=0.1,
                B=0.1,
                kappa=0.02,
                Gamma=0.5,
                rho_heavy=config.rho_heavy,
                rho_light=config.rho_light,
            )
            f_fe, g_fe = stream3d(f_fe), stream3d(g_fe)
            f_fe = bounce_back_cells_3d(f_fe, solid)
            g_fe = bounce_back_cells_3d(g_fe, solid)
            if config.free_slip_all:
                f_fe = _free_slip_all_walls_3d(f_fe, x_wall, y_wall, z_wall)
                g_fe = _free_slip_all_walls_3d(g_fe, x_wall, y_wall, z_wall)
            elif config.free_slip_y:
                f_fe = free_slip_cells_3d(f_fe, y_wall, axis=1)
                g_fe = free_slip_cells_3d(g_fe, y_wall, axis=1)
            phi = g_fe.sum(dim=0).clamp(-1.0, 1.0)
            rho_heavy = 0.5 * (phi + 1.0) * config.rho_heavy
            rho_light = 0.5 * (1.0 - phi) * config.rho_light

        # Diagnostics
        if step % config.output_interval == 0 or step == config.n_steps:
            if config.model == "fs":
                # Free-surface: use flags for front detection
                x_front = _find_front_x_3d(rho_heavy, rho_light, solid, flags=flags)
            else:
                x_front = _find_front_x_3d(rho_heavy, rho_light, solid)
            t_star = step * t_scale
            x_star = x_front / L

            # Mass conservation check
            total_mass = float((rho_heavy + rho_light).sum().item())
            mean_rho = float((rho_heavy + rho_light).mean().item())

            front_series.append((step, t_star, x_star, mean_rho))
            print(f"step={step:5d}  t*={t_star:.3f}  X*={x_star:.3f}  mean_ρ={mean_rho:.4f}")

            # Pressure probe at obstacle
            p_p1 = (
                _probe_pressure(f_water, f_air, config.obs_x0, config.obs_y0, config.obs_y1)
                if config.model == "cg"
                else float("nan")
            )

            diagnostics.append(
                {
                    "step": step,
                    "t_star": t_star,
                    "x_star": x_star,
                    "mean_rho": mean_rho,
                    "total_mass": total_mass,
                    "p_p1": p_p1 if math.isfinite(p_p1) else None,
                }
            )

    # Quantitative front validation vs the matching experiment.
    front_error, front_samples, front_reference = front_advance_error(
        front_series, config.periodic_z
    )
    front_gate: str | None = None
    if config.front_error_tolerance is not None and front_samples > 0:
        if not math.isfinite(front_error) or front_error > config.front_error_tolerance:
            front_gate = (
                f"front advance error {front_error:.3f} exceeds tolerance "
                f"{config.front_error_tolerance} vs {front_reference} "
                f"({front_samples} samples)"
            )
            raise RuntimeError(f"dam-break front validation failed: {front_gate}")
        front_gate = "passed"

    # Save metadata
    metadata: dict[str, object] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "diagnostics": diagnostics,
        "front_validation": {
            "reference": front_reference,
            "mean_advance_error": front_error,
            "n_samples": front_samples,
            "tolerance": config.front_error_tolerance,
            "gate": front_gate,
        },
        "front_series": [
            {"step": s, "t_star": ts, "x_star": xs, "mean_rho": mr}
            for s, ts, xs, mr in front_series
        ],
        **({"fs_handoff": fs_handoff} if config.model == "fs" else {}),
        **(
            {
                "free_surface_quality_curve": fs_runtime.get("steps", []),
                "free_surface_quality_gate": {
                    "passed": True,
                    "unexplained_tolerance": config.free_surface_unexplained_tolerance,
                    "paired_tolerance": config.free_surface_paired_tolerance,
                    "relative_cumulative_drift_tolerance": config.free_surface_relative_cumulative_drift_tolerance,
                    "relative_drift_slope_tolerance": config.free_surface_relative_drift_slope_tolerance,
                    "topology_normalized_drift_tolerance": config.free_surface_topology_normalized_drift_tolerance,
                    "drift_slope_window": config.free_surface_drift_slope_window,
                    "topology_changed": fs_topology_changed,
                    "diagnostic": "tracked-mass accounting only; not a physical/PV closure claim",
                },
                "free_surface_topology_events": fs_topology_events,
            }
            if config.model == "fs"
            else {}
        ),
    }

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    # Save front position CSV
    csv_path = run_dir / "front_position.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "t_star", "X_star", "mean_rho"])
        for s, ts, xs, mr in front_series:
            writer.writerow([s, f"{ts:.6f}", f"{xs:.4f}", f"{mr:.6f}"])

    print(f"Saved metadata → {meta_path}")
    if front_series:
        _save_validation_plot(front_series, config, run_dir)
    return run_dir


def _save_validation_plot(front_series, config, run_dir):
    """Save validation plot comparing front position with experimental data."""
    import matplotlib.pyplot as plt

    ts = [fs[1] for fs in front_series]
    xs = [fs[2] for fs in front_series]
    plt.figure(figsize=(8, 5))
    plt.plot(ts, xs, "b.-", label=f"TensorLBM ({config.model.upper()})", linewidth=1.5)
    if config.periodic_z:
        # Martin & Moyce Z(T) -> runner coords X* = (Z + 1) / 2.
        mm_t, mm_z = zip(*_MARTIN_MOYCE_FRONT)
        mm_x = [(z + 1.0) / 2.0 for z in mm_z]
        plt.plot(mm_t, mm_x, "rs--", label="Martin and Moyce (1952)", markersize=4, alpha=0.6)
    else:
        ref_t, ref_x = zip(*_KOSHIZUKA_FRONT)
        plt.plot(ref_t, ref_x, "k^--", label="Koshizuka and Oka (1996)", markersize=4, alpha=0.6)
    plt.xlabel("Dimensionless time  t*")
    plt.ylabel("Dimensionless front position  X*")
    plt.title(f"3D Dam-Break ({config.model.upper()}, tau={config.tau})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = run_dir / "validation_plot.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved validation plot to {plot_path}")
