"""SUBOFF bare-hull 7-collision × 2-lattice × wall_function full-grid runner.

Runs SUBOFF bare hull at Re=2×10⁶ with **all 7 collision families**
(Cumulant, BGK, MRT, TRT, CM/Cascaded, KBC, RLBM) on **both lattices**
(D3Q19, D3Q27) using the solver-agnostic ``wall_function_common`` module
(log-law, Guo body force) and far-field boundary conditions.

The wall function decouples the wall shear stress from the bulk relaxation
time: ``u_tau`` is computed from the log-law at the first off-wall cell,
then applied as a Guo body force on near-wall fluid cells via
:func:`tensorlbm.wall_function_common.wall_function`.

Loop per step::

    collide → stream → wall_function(compute_u_tau → wall_function)
             → far_field_bc → bounce_back → force

This runner does **not** modify any solver hot path.  Only existing
operators are composed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
)
from .boundaries_d3q27 import (
    bounce_back_cells_27,
    far_field_bc_27,
)
from .cascaded_collision import (
    collide_cascaded_d3q19,
    collide_cascaded_d3q27,
)
from .cumulant import (
    collide_cumulant_d3q19,
    collide_cumulant_d3q27,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import (
    collide_bgk27,
    collide_mrt27,
    collide_rlbm27,
    collide_trt27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .entropic_kbc import (
    collide_kbc_d3q19,
    collide_kbc_d3q27,
)
from .solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    collide_rlbm3d,
    collide_trt3d,
    correct_mass3d,
    stream3d,
)
from .suboff_cad import SuboffHullType, build_suboff_mask
from .suboff_resistance import _voxel_wetted_area
from .wall_function_common import (
    _near_wall_mask,
    compute_u_tau,
    compute_y_plus,
    wall_function,
)

__all__ = [
    "SuboffWallFnFullGridConfig",
    "run_suboff_wallfn_fullgrid",
    "write_artifact",
    "COMBINATIONS",
    "COLLISION_FAMILIES",
    "LATTICES",
    "stream27_roll",
]

# --------------------------------------------------------------------------- #
# Collision family and lattice enumerations
# --------------------------------------------------------------------------- #

COLLISION_FAMILIES: tuple[str, ...] = (
    "CUMULANT",
    "BGK",
    "MRT",
    "TRT",
    "CM",
    "KBC",
    "RLBM",
)

LATTICES: tuple[str, ...] = ("D3Q27", "D3Q19")

# The 11 test combinations from the task specification.
# D3Q27: all 7 families (MRT uses reduced grid 320×160×160 for memory).
# D3Q19: 4 families (Cumulant, BGK, MRT, RLBM).
COMBINATIONS: list[tuple[str, str]] = [
    ("D3Q27", "CUMULANT"),
    ("D3Q27", "BGK"),
    ("D3Q27", "MRT"),
    ("D3Q27", "TRT"),
    ("D3Q27", "CM"),
    ("D3Q27", "KBC"),
    ("D3Q27", "RLBM"),
    ("D3Q19", "CUMULANT"),
    ("D3Q19", "BGK"),
    ("D3Q19", "MRT"),
    ("D3Q19", "RLBM"),
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SuboffWallFnFullGridConfig:
    """Configuration for a single SUBOFF wall-function full-grid run.

    Defaults target Re=2×10⁶ on a 480×240×240 grid with 1000 steps
    on ``sdaa:0``.  For fast tests, override ``nx/ny/nz/n_steps/device``.
    """

    re: float = 2_000_000.0
    lattice: str = "D3Q27"
    collision: str = "CUMULANT"
    nx: int = 480
    ny: int = 240
    nz: int = 240
    n_steps: int = 1000
    u_in: float = 0.06
    hull_length: float = 240.0
    device: str = "sdaa:0"
    # Wall-function parameters
    y_val: float = 0.5
    wall_law: str = "log"
    # ITTC reference
    reference_Ct: float = 0.00405
    reference_source: str = "ITTC-1957"

    def __post_init__(self) -> None:
        if self.lattice.upper() not in LATTICES:
            raise ValueError(
                f"lattice must be one of {LATTICES}; got {self.lattice!r}"
            )
        if self.collision.upper() not in COLLISION_FAMILIES:
            raise ValueError(
                f"collision must be one of {COLLISION_FAMILIES}; "
                f"got {self.collision!r}"
            )
        if self.wall_law not in ("log", "reichardt", "gradient", "hybrid"):
            raise ValueError(
                f"wall_law must be 'log', 'reichardt', 'gradient', or 'hybrid'; "
                f"got {self.wall_law!r}"
            )
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, nz must be at least 16, 8, 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.u_in <= 0.0 or self.u_in >= 0.15:
            raise ValueError("u_in must be in (0, 0.15)")
        if self.re <= 0.0:
            raise ValueError("re must be > 0")
        if self.hull_length <= 0.0:
            raise ValueError("hull_length must be > 0")
        if self.y_val <= 0.0:
            raise ValueError("y_val must be > 0")

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """Shear relaxation time."""
        return 3.0 * self.nu + 0.5


# --------------------------------------------------------------------------- #
# Memory-efficient D3Q27 streaming (torch.roll, no index tensors)
# --------------------------------------------------------------------------- #

# D3Q27 velocity vectors matching d3q27.py _C_DATA ordering.
_D3Q27_SHIFTS: list[tuple[int, int, int]] = [
    (0, 0, 0),        #  0: rest
    (1, 0, 0),        #  1: +x
    (-1, 0, 0),       #  2: -x
    (0, 1, 0),        #  3: +y
    (0, -1, 0),       #  4: -y
    (0, 0, 1),        #  5: +z
    (0, 0, -1),       #  6: -z
    (1, 1, 0),        #  7: +x+y
    (-1, 1, 0),       #  8: -x+y
    (1, -1, 0),       #  9: +x-y
    (-1, -1, 0),      # 10: -x-y
    (1, 0, 1),        # 11: +x+z
    (-1, 0, 1),       # 12: -x+z
    (1, 0, -1),       # 13: +x-z
    (-1, 0, -1),      # 14: -x-z
    (0, 1, 1),        # 15: +y+z
    (0, -1, 1),       # 16: -y+z
    (0, 1, -1),       # 17: +y-z
    (0, -1, -1),      # 18: -y-z
    (1, 1, 1),        # 19: +x+y+z
    (-1, 1, 1),       # 20: -x+y+z
    (1, -1, 1),       # 21: +x-y+z
    (-1, -1, 1),      # 22: -x-y+z
    (1, 1, -1),       # 23: +x+y-z
    (-1, 1, -1),      # 24: -x+y-z
    (1, -1, -1),      # 25: +x-y-z
    (-1, -1, -1),     # 26: -x-y-z
]


def stream27_roll(f: torch.Tensor) -> torch.Tensor:
    """Memory-efficient D3Q27 streaming using torch.roll per direction.

    Unlike :func:`tensorlbm.d3q27.stream27` which caches 4×[27,N] int64
    index tensors (~24GB for 480×240×240), this function uses
    ``torch.roll`` per direction — same approach as the D3Q19
    :func:`stream3d`.  Trades a small speed cost for massive memory
    savings (only 1 extra copy of f vs 24GB of index tensors).

    This is a **runner-local** function that does not modify the solver
    hot path.
    """
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


# --------------------------------------------------------------------------- #
# Lattice dispatch helpers
# --------------------------------------------------------------------------- #

def _macroscopic(lattice: str, f: torch.Tensor):
    """Dispatch to the correct macroscopic function."""
    if lattice == "D3Q19":
        return macroscopic3d(f)
    return macroscopic27(f)


def _equilibrium(lattice: str, rho, ux, uy, uz, device=None):
    """Dispatch to the correct equilibrium function."""
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz, device=device)
    return equilibrium27(rho, ux, uy, uz, device=device)


def _stream(lattice: str, f: torch.Tensor) -> torch.Tensor:
    """Dispatch to the correct streaming function.

    For D3Q27, uses the memory-efficient ``stream27_roll`` (torch.roll)
    instead of the index-tensor-based ``stream27`` to avoid OOM on
    large grids (480×240×240).
    """
    if lattice == "D3Q19":
        return stream3d(f)
    return stream27_roll(f)


def _far_field_bc(lattice: str, f, u_in, obstacle_mask=None):
    """Dispatch to the correct far-field BC function."""
    if lattice == "D3Q19":
        return far_field_bc_3d(f, u_in, obstacle_mask=obstacle_mask)
    return far_field_bc_27(f, u_in, obstacle_mask=obstacle_mask)


def _bounce_back(lattice: str, f, mask):
    """Dispatch to the correct bounce-back function."""
    if lattice == "D3Q19":
        return bounce_back_cells_3d(f, mask)
    return bounce_back_cells_27(f, mask)


def _correct_mass(lattice: str, f, target_mass):
    """Dispatch to the correct mass-correction function."""
    if lattice == "D3Q19":
        return correct_mass3d(f, target_mass)
    return correct_mass27(f, target_mass)


# --------------------------------------------------------------------------- #
# Collision dispatch
# --------------------------------------------------------------------------- #

def _collide(
    lattice: str,
    collision: str,
    f: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Dispatch to the correct collision operator.

    All collision operators take ``(f, tau)`` and return the
    post-collision distribution.  Higher-order parameters use defaults.
    """
    lat = lattice.upper()
    col = collision.upper()

    if lat == "D3Q19":
        if col == "CUMULANT":
            return collide_cumulant_d3q19(f, tau)
        if col == "BGK":
            return collide_bgk3d(f, tau)
        if col == "MRT":
            return collide_mrt3d(f, tau)
        if col == "TRT":
            return collide_trt3d(f, tau)
        if col == "CM":
            return collide_cascaded_d3q19(f, tau)
        if col == "KBC":
            return collide_kbc_d3q19(f, tau)
        if col == "RLBM":
            return collide_rlbm3d(f, tau)
    else:  # D3Q27
        if col == "CUMULANT":
            return collide_cumulant_d3q27(f, tau)
        if col == "BGK":
            return collide_bgk27(f, tau)
        if col == "MRT":
            return collide_mrt27(f, tau)
        if col == "TRT":
            return collide_trt27(f, tau)
        if col == "CM":
            return collide_cascaded_d3q27(f, tau)
        if col == "KBC":
            return collide_kbc_d3q27(f, tau)
        if col == "RLBM":
            return collide_rlbm27(f, tau)

    # Unreachable — validated in __post_init__
    raise ValueError(f"Unknown collision {col!r} for lattice {lat!r}")


# --------------------------------------------------------------------------- #
# Drag computation (wall-function based)
# --------------------------------------------------------------------------- #

def _compute_drags(
    f: torch.Tensor,
    solid: torch.Tensor,
    u_tau: torch.Tensor,
    lattice: str,
) -> tuple[float, float]:
    """Compute friction and pressure drag from the wall-function state.

    * **Friction drag** = Σ (τ_w · u_x/|u|) over near-wall fluid cells.
      This is the integrated wall shear stress projected onto the
      streamwise (x) direction.

    * **Pressure drag** = Σ p · n̂_x over hull faces, where
      p = (ρ − 1)/3 is the gauge pressure and n̂_x is the x-component
      of the outward normal from the fluid into the solid.

    Must be called **after streaming, before wall_function** modifies f.
    """
    rho, ux, uy, uz = _macroscopic(lattice, f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

    # Near-wall mask: fluid cells adjacent to solid (6-connected)
    near = _near_wall_mask(solid)

    # Friction drag: Σ τ_w · (u_x / |u|) · near
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    drag_fric = float(
        (tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item()
    )

    # Pressure drag: Σ p · (solid[+x] − solid[−x]) · fluid
    p = (rho - 1.0) / 3.0
    fluid = ~solid
    # solid[+x neighbour] = roll(solid, -1, dims=2)  → sm
    # solid[−x neighbour] = roll(solid, +1, dims=2)  → sp
    sm = torch.roll(solid, -1, dims=2)   # solid at +x of current cell
    sp = torch.roll(solid, 1, dims=2)    # solid at -x of current cell
    drag_pres = float(
        (p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    )

    return drag_fric, drag_pres


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_suboff_wallfn_fullgrid(
    config: SuboffWallFnFullGridConfig | None = None,
) -> dict[str, Any]:
    """Run SUBOFF bare-hull with wall_function and produce artifact.

    Returns a machine-readable dict with the fields specified by the task::

        lattice, collision, wall_function=log-law,
        Ct_fric, Ct_pres, Ct_total, finite, steps_completed,
        grid, hull_length,
        reference_Ct=0.00405, reference_source=ITTC-1957

    plus ``status="diagnostic_only"`` and ``physical_validation=False``.
    """
    if config is None:
        config = SuboffWallFnFullGridConfig()

    lattice = config.lattice.upper()
    collision = config.collision.upper()
    device = torch.device(config.device)

    # --- 1. Build geometry (on CPU to save device memory) ---
    cx = config.nx * 0.35
    cy = config.ny / 2.0
    cz = config.nz / 2.0
    solid, _stats = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=config.hull_length,
        device="cpu",  # Build on CPU, move to device below
    )
    solid = solid.to(device)

    # Wetted area and dynamic pressure for Ct normalization
    wetted_area = _voxel_wetted_area(solid, 1.0)
    rho_lu = 1.0
    dynamic_pressure = 0.5 * rho_lu * config.u_in ** 2 * wetted_area

    # --- 2. Initialize populations (on CPU, move to device) ---
    rho0 = torch.ones((config.nz, config.ny, config.nx))
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[solid.cpu()] = 0.0
    f = _equilibrium(lattice, rho0, ux0, uy0, uz0)
    f = f.to(device)
    initial_mass = float(f.sum().item())

    tau = config.tau
    nu = config.nu

    # --- 3. Solver loop ---
    force_series: list[dict[str, Any]] = []
    ct_series: list[dict[str, Any]] = []
    completed_steps = 0
    all_finite = True

    for step in range(1, config.n_steps + 1):
        # 1. Collision
        f = _collide(lattice, collision, f, tau)

        # 2. Streaming
        f = _stream(lattice, f)

        # 3. Wall function: compute_u_tau → wall_function
        #    Compute macroscopic fields from post-stream state
        rho, ux, uy, uz = _macroscopic(lattice, f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

        # Compute friction velocity from log-law
        u_tau = compute_u_tau(
            u_mag, nu, y_val=config.y_val, wall_law=config.wall_law,
        )
        y_plus = compute_y_plus(u_tau, nu, y_val=config.y_val)

        # Compute drag from wall shear stress (before wall_function modifies f)
        drag_fric, drag_pres = _compute_drags(f, solid, u_tau, lattice)

        # Apply wall function (Guo body force on near-wall cells)
        f = wall_function(
            f, solid, u_tau, y_plus,
            lattice=lattice, nu=nu, y_val=config.y_val,
        )

        # 4. Far-field BC (free-stream on inlet + lateral faces, zero-grad outlet)
        f = _far_field_bc(lattice, f, config.u_in)

        # 5. Bounce-back on solid
        f = _bounce_back(lattice, f, solid)

        # 6. Force (record friction + pressure drag)
        ct_fric = drag_fric / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_pres = drag_pres / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_total = ct_fric + ct_pres

        force_series.append({
            "step": step,
            "drag_fric": drag_fric,
            "drag_pres": drag_pres,
            "drag_total": drag_fric + drag_pres,
        })
        ct_series.append({
            "step": step,
            "ct_fric": ct_fric,
            "ct_pres": ct_pres,
            "ct_total": ct_total,
        })

        # Mass correction every 100 steps
        if step % 100 == 0:
            f = _correct_mass(lattice, f, initial_mass)

        # Finiteness check
        completed_steps = step
        finite = bool(torch.isfinite(f).all().item())
        all_finite = all_finite and finite
        if not finite:
            break

    # --- 4. Build artifact ---
    # Use time-averaged Ct over last 50% of steps (after warmup)
    warmup = max(1, config.n_steps // 2)
    ct_fric_avg = (
        sum(e["ct_fric"] for e in ct_series[warmup:]) /
        max(len(ct_series[warmup:]), 1)
    )
    ct_pres_avg = (
        sum(e["ct_pres"] for e in ct_series[warmup:]) /
        max(len(ct_series[warmup:]), 1)
    )
    ct_total_avg = ct_fric_avg + ct_pres_avg

    artifact: dict[str, Any] = {
        "schema": "tensorlbm.suboff-wallfn-fullgrid/v1",
        "status": "diagnostic_only",
        "physical_validation": False,
        "lattice": lattice,
        "collision": collision,
        "wall_function": f"log-law (κ=0.41, B=5.0, y_val={config.y_val})",
        "Re": config.re,
        "Ct_fric": ct_fric_avg,
        "Ct_pres": ct_pres_avg,
        "Ct_total": ct_total_avg,
        "finite": all_finite,
        "steps_completed": completed_steps,
        "grid": {"nx": config.nx, "ny": config.ny, "nz": config.nz},
        "hull_length": config.hull_length,
        "u_in": config.u_in,
        "tau": tau,
        "nu": nu,
        "wetted_area": wetted_area,
        "dynamic_pressure": dynamic_pressure,
        "reference_Ct": config.reference_Ct,
        "reference_source": config.reference_source,
        "device": "sdaa" if "sdaa" in config.device else config.device,
        "force_time_series": force_series,
        "ct_time_series": ct_series,
    }
    return artifact


def write_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Write the artifact as a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(artifact, sort_keys=True, indent=2),
        encoding="utf-8",
    )
