"""SUBOFF bare-hull CM/CUMULANT/KBC × SGS diagnostic runner.

Runs SUBOFF bare hull at Re=2×10⁶ with three advanced collision families
(CM, CUMULANT, KBC) coupled with three SGS models (Smagorinsky, WALE,
Vreman).  Uses :func:`collide_advanced_3d` for unified collision dispatch
and far-field boundary conditions.

The SGS coupling computes a per-cell effective relaxation time
``tau_eff = tau_0 + 3*nu_t`` from the local eddy viscosity, then passes
it to the collision kernel:

* **CM / CUMULANT** — accept tensor ``tau`` (element-wise ``omega = 1/tau``).
* **KBC** — uses scalar mean ``tau_eff`` because the entropy bisection
  initialiser (``torch.full``) requires a Python scalar.

This runner does **not** modify any solver hot path.  Only existing
operators are composed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .advanced_collision_contract import collide_advanced_3d
from .boundaries3d import far_field_bc_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import correct_mass3d, stream3d
from .suboff_cad import SuboffHullType, build_suboff_mask
from .suboff_resistance import _voxel_wetted_area
from .turbulence import (
    _neq_stress_norm_3d,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
)

__all__ = [
    "SuboffCmkKbcConfig",
    "run_suboff_cmk_kbc",
    "COMBINATIONS",
]

# The 9 test combinations: 3 collision families × 3 SGS models
COMBINATIONS: list[tuple[str, str]] = [
    ("CM", "smagorinsky"),
    ("CM", "wale"),
    ("CM", "vreman"),
    ("CUMULANT", "smagorinsky"),
    ("CUMULANT", "wale"),
    ("CUMULANT", "vreman"),
    ("KBC", "smagorinsky"),
    ("KBC", "wale"),
    ("KBC", "vreman"),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuboffCmkKbcConfig:
    """Configuration for a single SUBOFF CM/CUMULANT/KBC × SGS run.

    Defaults target Re=2×10⁶ on a 320×160×160 D3Q19 grid with 1000 steps
    on ``sdaa:0``.  For fast tests, override ``nx/ny/nz/n_steps/device``.
    """

    re: float = 2_000_000.0
    collision: str = "CM"
    turbulence_model: str = "smagorinsky"
    nx: int = 320
    ny: int = 160
    nz: int = 160
    n_steps: int = 1000
    u_in: float = 0.06
    hull_length: float = 192.0  # 0.6 * 320
    device: str = "sdaa:0"
    lattice: str = "D3Q19"
    boundary_type: str = "farfield"
    # SGS model constants
    C_s: float = 0.1   # Smagorinsky
    C_w: float = 0.5   # WALE
    C_V: float = 0.025  # Vreman
    # ITTC reference
    reference_Ct: float = 0.00405
    reference_source: str = "ITTC-1957"

    def __post_init__(self) -> None:
        if self.collision.upper() not in {"CM", "CUMULANT", "KBC"}:
            raise ValueError(
                f"collision must be CM, CUMULANT, or KBC; got {self.collision!r}"
            )
        if self.turbulence_model.lower() not in {"smagorinsky", "wale", "vreman"}:
            raise ValueError(
                f"turbulence_model must be smagorinsky, wale, or vreman; "
                f"got {self.turbulence_model!r}"
            )
        if self.lattice.upper() != "D3Q19":
            raise ValueError(
                f"lattice must be D3Q19 (far_field_bc_3d is D3Q19-only); "
                f"got {self.lattice!r}"
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

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """Baseline shear relaxation time."""
        return 3.0 * self.nu + 0.5


# ---------------------------------------------------------------------------
# SGS-coupled collision
# ---------------------------------------------------------------------------

def _compute_sgs_tau_eff(
    f: torch.Tensor,
    config: SuboffCmkKbcConfig,
    tau_base: float,
) -> torch.Tensor:
    """Compute per-cell effective tau from the chosen SGS model.

    * **Smagorinsky** — uses the non-equilibrium stress Frobenius norm
      (LBM-specific, no velocity gradients needed).
    * **WALE / Vreman** — compute eddy viscosity from velocity gradients
      via central differences, then convert to tau_eff.
    """
    rho, ux, uy, uz = macroscopic3d(f)

    model = config.turbulence_model.lower()
    if model == "smagorinsky":
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        pi_norm = _neq_stress_norm_3d(f_neq)
        return _smagorinsky_tau(tau_base, pi_norm, rho, config.C_s)
    elif model == "wale":
        nu_t = _wale_nu_t_3d(ux, uy, uz, config.C_w)
        return _nu_t_to_tau_eff(tau_base, nu_t)
    elif model == "vreman":
        nu_t = _vreman_nu_t_3d(ux, uy, uz, config.C_V)
        return _nu_t_to_tau_eff(tau_base, nu_t)
    # Unreachable — validated in __post_init__
    raise ValueError(f"Unknown turbulence model: {config.turbulence_model}")


def _collide_with_sgs(
    f: torch.Tensor,
    config: SuboffCmkKbcConfig,
    tau_base: float,
) -> torch.Tensor:
    """Apply collision via ``collide_advanced_3d`` with SGS-coupled tau_eff.

    For CM and CUMULANT the per-cell tensor ``tau_eff`` is passed directly
    (both kernels use ``omega = 1/tau`` element-wise).

    For KBC the entropy bisection initialiser (``torch.full``) requires a
    Python scalar, so the spatial mean of ``tau_eff`` is used.
    """
    tau_eff = _compute_sgs_tau_eff(f, config, tau_base)

    if config.collision.upper() == "KBC":
        tau_scalar = float(tau_eff.mean().item())
        return collide_advanced_3d(
            config.lattice, config.collision, f, tau=tau_scalar,
        )
    # CM and CUMULANT accept tensor tau
    return collide_advanced_3d(
        config.lattice, config.collision, f, tau=tau_eff,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suboff_cmk_kbc(
    config: SuboffCmkKbcConfig | None = None,
) -> dict[str, Any]:
    """Run SUBOFF bare-hull CM/CUMULANT/KBC × SGS and produce artifact.

    Returns a machine-readable dict with the fields specified by the task:
    ``Re, collision, turbulence_model, Ct, finite, steps_completed,
    boundary_type, device, reference_Ct, reference_source`` plus
    ``status="diagnostic_only"`` and ``physical_validation=False``.
    """
    if config is None:
        config = SuboffCmkKbcConfig()

    device = torch.device(config.device)

    # --- 1. Build geometry ---
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
        device=str(device),
    )
    solid = solid.to(device)

    # Wetted area and dynamic pressure for Ct normalization
    wetted_area = _voxel_wetted_area(solid, 1.0)
    rho_lu = 1.0
    dynamic_pressure = 0.5 * rho_lu * config.u_in ** 2 * wetted_area

    # --- 2. Initialize populations ---
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    initial_mass = float(f.sum().item())

    tau_base = config.tau

    # --- 3. Solver loop ---
    force_series: list[dict[str, Any]] = []
    ct_series: list[dict[str, Any]] = []
    completed_steps = 0
    all_finite = True

    for step in range(1, config.n_steps + 1):
        # Collision with SGS
        f = _collide_with_sgs(f, config, tau_base)
        # Streaming (pull scheme)
        f = stream3d(f)
        # Force measurement (momentum exchange, before bounce-back)
        fx_t, fy_t, fz_t = compute_obstacle_forces_3d(f, solid)
        fx = float(fx_t.item())
        fy = float(fy_t.item())
        fz = float(fz_t.item())
        # Far-field BC (includes bounce-back on solid)
        f = far_field_bc_3d(f, config.u_in, obstacle_mask=solid)
        # Mass correction every 10 steps
        if step % 10 == 0:
            f = correct_mass3d(f, initial_mass)

        # Record force / Ct
        ct = fx / dynamic_pressure if dynamic_pressure > 0 else 0.0
        force_series.append({"step": step, "fx": fx, "fy": fy, "fz": fz})
        ct_series.append({"step": step, "ct": ct})

        completed_steps = step
        finite = bool(torch.isfinite(f).all().item())
        all_finite = all_finite and finite
        if not finite:
            break

    # --- 4. Build artifact ---
    ct_final = ct_series[-1]["ct"] if ct_series else 0.0
    artifact: dict[str, Any] = {
        "schema": "tensorlbm.suboff-cmk-kbc-sgs/v1",
        "status": "diagnostic_only",
        "physical_validation": False,
        "Re": config.re,
        "collision": config.collision.upper(),
        "turbulence_model": config.turbulence_model.lower(),
        "Ct": ct_final,
        "finite": all_finite,
        "steps_completed": completed_steps,
        "boundary_type": config.boundary_type,
        "device": "sdaa",
        "reference_Ct": config.reference_Ct,
        "reference_source": config.reference_source,
        "lattice": config.lattice,
        "grid": {"nx": config.nx, "ny": config.ny, "nz": config.nz},
        "u_in": config.u_in,
        "tau": tau_base,
        "nu": config.nu,
        "wetted_area": wetted_area,
        "dynamic_pressure": dynamic_pressure,
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
