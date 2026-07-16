"""Sphere cross-validation runner: D3Q19/D3Q27 × 7 collision families × key turbulence models.

This diagnostic runner executes a 3-D sphere flow at Re=100 on a small grid
for every combination of:

* **Lattice**: D3Q19, D3Q27
* **Collision family**: BGK, TRT, RLBM, MRT, CM, CUMULANT, KBC
* **Turbulence model**: none, Smagorinsky, WALE

All collision dispatch goes through :func:`collide_advanced_3d` — the unified
common-contract entry point.  Turbulence models are applied as a per-step
mean effective-tau adjustment computed from the existing LES helpers in
:mod:`tensorlbm.turbulence` (Smagorinsky non-equilibrium stress norm, WALE
eddy viscosity).  This is a **diagnostic** approximation: the spatial
variation of the eddy viscosity is collapsed to a scalar mean tau before
calling ``collide_advanced_3d``.  No physical accuracy is claimed
(``status="diagnostic_only"``, ``physical_validation=False``).

The output is a machine-readable JSON artifact with one entry per
(lattice, collision_family, turbulence_model) combination, recording Cd,
finiteness, steps completed, and the Schiller-Naumann reference Cd.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .advanced_collision_contract import collide_advanced_3d
from .boundaries3d import (
    apply_simple_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from .boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    make_channel_wall_mask_27,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import equilibrium27, macroscopic27, stream27
from .obstacles import compute_obstacle_forces_27, compute_obstacle_forces_3d
from .solver3d import stream3d
from .turbulence import (
    _neq_stress_norm_27,
    _neq_stress_norm_3d,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _wale_nu_t_3d,
)

SCHEMA_VERSION = "tensorlbm.sphere-cross-validation/v1"

LATTICES: tuple[str, ...] = ("D3Q19", "D3Q27")
COLLISION_FAMILIES: tuple[str, ...] = (
    "BGK", "TRT", "RLBM", "MRT", "CM", "CUMULANT", "KBC",
)
TURBULENCE_MODELS: tuple[str, ...] = ("none", "Smagorinsky", "WALE")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SphereCrossValidationConfig:
    """Configuration for the sphere cross-validation runner."""

    re: float = 100.0
    nx: int = 40
    ny: int = 40
    nz: int = 40
    steps: int = 100
    u_in: float = 0.06
    device: str = "cpu"
    smagorinsky_cs: float = 0.1
    wale_cw: float = 0.5


@dataclass(frozen=True)
class SphereCrossValidationResult:
    """One cell of the Cd comparison matrix."""

    lattice: str
    collision_family: str
    turbulence_model: str
    Cd: float | None
    finite: bool
    steps_completed: int
    reference_Cd: float
    status: str
    physical_validation: bool


@dataclass(frozen=True)
class SphereCrossValidationMatrix:
    """Machine-readable Cd comparison matrix artifact."""

    schema_version: str
    runner: str
    torch_version: str
    config: dict[str, Any]
    reference_Cd: float
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schiller_naumann(re: float) -> float:
    """Schiller-Naumann drag correlation for a sphere."""
    if re < 1e-6:
        return 100.0
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def _equilibrium(
    lattice: str,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz, device=device)
    return equilibrium27(rho, ux, uy, uz, device=device)


def _macroscopic(lattice: str, f: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if lattice == "D3Q19":
        return macroscopic3d(f)
    return macroscopic27(f)


def _stream(lattice: str, f: torch.Tensor) -> torch.Tensor:
    if lattice == "D3Q19":
        return stream3d(f)
    return stream27(f)


def _compute_forces(
    lattice: str, f: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if lattice == "D3Q19":
        return compute_obstacle_forces_3d(f, mask)
    return compute_obstacle_forces_27(f, mask)


def _apply_boundaries(
    lattice: str,
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    if lattice == "D3Q19":
        return apply_simple_channel_boundaries_3d(
            f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle_mask
        )
    return apply_zou_he_channel_boundaries_27(
        f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle_mask
    )


def _make_wall_mask(
    lattice: str,
    nz: int,
    ny: int,
    nx: int,
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if lattice == "D3Q19":
        return make_channel_wall_mask_3d(nz, ny, nx, obstacle_mask, device=device)
    return make_channel_wall_mask_27(nz, ny, nx, obstacle_mask, device=device)


def _compute_tau_eff_smagorinsky(
    lattice: str, f: torch.Tensor, tau: float, cs: float
) -> float:
    """Mean effective tau from the Smagorinsky non-equilibrium stress model."""
    rho, ux, uy, uz = _macroscopic(lattice, f)
    feq = _equilibrium(lattice, rho, ux, uy, uz, f.device)
    f_neq = f - feq
    if lattice == "D3Q19":
        pi_norm = _neq_stress_norm_3d(f_neq)
    else:
        pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, cs)
    return float(tau_eff.mean().item())


def _compute_tau_eff_wale(
    lattice: str, f: torch.Tensor, tau: float, cw: float
) -> float:
    """Mean effective tau from the WALE eddy-viscosity model."""
    rho, ux, uy, uz = _macroscopic(lattice, f)
    nu_t = _wale_nu_t_3d(ux, uy, uz, cw)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    return float(tau_eff.mean().item())


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_single_combination(
    config: SphereCrossValidationConfig,
    lattice: str,
    family: str,
    turbulence: str,
) -> SphereCrossValidationResult:
    """Run one (lattice, family, turbulence) combination and return Cd."""
    nx, ny, nz = config.nx, config.ny, config.nz
    radius = max(4.0, nx * 0.08)
    u_in = config.u_in
    re = config.re
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5
    dev = torch.device(config.device)

    mask = sphere_mask(nx, ny, nz, nx * 0.5, ny * 0.5, nz * 0.5, radius, device=dev)
    wall_mask = _make_wall_mask(lattice, nz, ny, nx, mask, dev)

    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), u_in, device=dev)
    uy = torch.zeros(nz, ny, nx, device=dev)
    uz = torch.zeros(nz, ny, nx, device=dev)
    f = _equilibrium(lattice, rho, ux, uy, uz, dev)

    fx_list: list[float] = []
    steps_completed = 0
    finite = True

    for step in range(1, config.steps + 1):
        # Compute effective tau based on turbulence model
        if turbulence == "none":
            tau_eff = tau
        elif turbulence == "Smagorinsky":
            tau_eff = _compute_tau_eff_smagorinsky(lattice, f, tau, config.smagorinsky_cs)
        elif turbulence == "WALE":
            tau_eff = _compute_tau_eff_wale(lattice, f, tau, config.wale_cw)
        else:
            raise ValueError(f"Unknown turbulence model: {turbulence}")

        # Collide via unified dispatch
        f = collide_advanced_3d(lattice, family, f, tau=tau_eff)

        # Stream
        f = _stream(lattice, f)

        # Forces (before bounce-back)
        fx, _, _ = _compute_forces(lattice, f, mask)

        # Boundaries
        f = _apply_boundaries(lattice, f, u_in, wall_mask, mask)

        steps_completed = step

        if not torch.isfinite(f).all().item():
            finite = False
            break

        if step > config.steps // 2:
            fx_list.append(float(fx.item()))

    # Compute Cd
    cd: float | None
    if fx_list and finite:
        fx_mean = sum(fx_list) / len(fx_list)
        area = math.pi * radius ** 2
        cd = fx_mean / (0.5 * u_in ** 2 * area)
        if not math.isfinite(cd):
            cd = None
            finite = False
    else:
        cd = None
        if not fx_list:
            finite = False

    ref_cd = _schiller_naumann(re)

    return SphereCrossValidationResult(
        lattice=lattice,
        collision_family=family,
        turbulence_model=turbulence,
        Cd=cd,
        finite=finite,
        steps_completed=steps_completed,
        reference_Cd=ref_cd,
        status="diagnostic_only",
        physical_validation=False,
    )


def run_sphere_cross_validation(
    config: SphereCrossValidationConfig | None = None,
) -> SphereCrossValidationMatrix:
    """Run the full cross-validation matrix and return the artifact.

    Iterates over all (lattice, collision_family, turbulence_model)
    combinations, runs a short sphere flow for each, and records Cd.
    """
    if config is None:
        config = SphereCrossValidationConfig()

    results: list[SphereCrossValidationResult] = []
    for lattice in LATTICES:
        for family in COLLISION_FAMILIES:
            for turb in TURBULENCE_MODELS:
                result = _run_single_combination(config, lattice, family, turb)
                results.append(result)

    return SphereCrossValidationMatrix(
        schema_version=SCHEMA_VERSION,
        runner="tensorlbm.sphere_cross_validation.run_sphere_cross_validation",
        torch_version=torch.__version__,
        config=asdict(config),
        reference_Cd=_schiller_naumann(config.re),
        results=[asdict(r) for r in results],
    )


def write_sphere_cross_validation_evidence(
    matrix: SphereCrossValidationMatrix,
    path: str | Path,
) -> Path:
    """Write the cross-validation matrix as machine-readable JSON."""
    output = Path(path)
    payload = matrix.to_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "SCHEMA_VERSION",
    "LATTICES",
    "COLLISION_FAMILIES",
    "TURBULENCE_MODELS",
    "SphereCrossValidationConfig",
    "SphereCrossValidationResult",
    "SphereCrossValidationMatrix",
    "_schiller_naumann",
    "run_sphere_cross_validation",
    "write_sphere_cross_validation_evidence",
]
