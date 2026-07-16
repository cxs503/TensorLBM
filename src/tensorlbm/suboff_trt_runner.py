"""SUBOFF bare-hull D3Q27 TRT collision × SGS turbulence model runner.

Runs SUBOFF bare-hull cases with D3Q27 TRT (two-relaxation-time) collision
combined with LES sub-grid-scale (SGS) turbulence models on SDAA devices.

Four combinations are supported:
  1. TRT + Smagorinsky
  2. TRT + WALE
  3. TRT + Vreman
  4. TRT + none (no SGS model)

The collision is dispatched through ``collide_advanced_3d`` for the TRT
family.  When an SGS model is active, the per-cell effective relaxation
time is computed from the SGS eddy viscosity and used as the TRT symmetric
relaxation time (τ₊), with the anti-symmetric rate (τ₋) derived from the
magic parameter Λ = 3/16.

This runner composes existing solver operators and does **not** modify any
solver hot path.  The evidence is deliberately ``diagnostic_only`` — real
force/Ct observations from an actual D3Q27+TRT+far-field loop, but no
physical validation claim is made.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from .advanced_collision_contract import collide_advanced_3d
from .d3q27 import (
    OPPOSITE,
    equilibrium27,
    macroscopic27,
    stream27,
    correct_mass27,
)
from .obstacles import compute_obstacle_forces_27
from .suboff_cad import SuboffHullType, build_suboff_mask
from .suboff_resistance import _voxel_wetted_area
from .turbulence import (
    _neq_stress_norm_27,
    _smagorinsky_tau,
    _nu_t_to_tau_eff,
    _wale_nu_t_3d,
    _vreman_nu_t_3d,
)

__all__ = [
    "SuboffTrtConfig",
    "SuboffTrtEvidence",
    "run_suboff_trt_sgs",
    "run_suboff_trt_sgs_campaign",
    "_collide_trt_sgs_27",
    "_far_field_bc_27",
    "_bounce_back_cells_27",
    "ITTC_1957_REFERENCE_CT",
]

# ITTC-1957 friction coefficient at Re = 2×10⁶
# Cf = 0.075 / (log10(Re) - 2)²
_ITTC_RE = 2.0e6
ITTC_1957_REFERENCE_CT = 0.075 / (math.log10(_ITTC_RE) - 2.0) ** 2

TurbulenceModel = Literal["none", "smagorinsky", "wale", "vreman"]
_VALID_MODELS = {"none", "smagorinsky", "wale", "vreman"}


# ---------------------------------------------------------------------------
# D3Q27 bounce-back (uses D3Q27 OPPOSITE, not D3Q19)
# ---------------------------------------------------------------------------

def _bounce_back_cells_27(
    f: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Bounce-back reflection on selected cells for D3Q27.

    The existing ``boundaries3d.bounce_back_cells_3d`` uses the D3Q19
    OPPOSITE map (19 directions).  This function uses the D3Q27 OPPOSITE
    map (27 directions, including corners) so it is correct for D3Q27
    populations.
    """
    opp = OPPOSITE.to(f.device)  # (27,)
    return torch.where(mask.unsqueeze(0), f[opp], f)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuboffTrtConfig:
    """Configuration for SUBOFF D3Q27 TRT×SGS diagnostic run.

    Defaults target Re=2×10⁶ on a 320×160×160 grid with 1000 steps
    on SDAA device 0.
    """

    nx: int = 320
    ny: int = 160
    nz: int = 160
    n_steps: int = 1000
    u_in: float = 0.05
    re: float = 2.0e6
    hull_length: float = 192.0  # 0.6 * 320
    device: str = "sdaa:0"
    turbulence_model: TurbulenceModel = "none"
    lambda_trt: float = 3.0 / 16.0
    smagorinsky_cs: float = 0.1
    wale_cw: float = 0.5
    vreman_cv: float = 0.025
    mass_correction_interval: int = 10

    def __post_init__(self) -> None:
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
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")
        if self.turbulence_model not in _VALID_MODELS:
            raise ValueError(
                f"turbulence_model must be one of {_VALID_MODELS}, "
                f"got '{self.turbulence_model}'"
            )
        if self.mass_correction_interval < 1:
            raise ValueError("mass_correction_interval must be >= 1")

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """TRT symmetric relaxation time τ₊ (sets viscosity)."""
        return 3.0 * self.nu + 0.5

    @property
    def boundary_type(self) -> str:
        """Boundary condition type (always far-field for this runner)."""
        return "farfield"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuboffTrtEvidence:
    """Machine-readable diagnostic evidence from a TRT×SGS run.

    All fields are JSON-serialisable.  The artifact is deliberately
    ``diagnostic_only`` with ``physical_validation=False``.
    """

    status: str
    physical_validation: bool
    Re: float
    collision: str
    turbulence_model: str
    Ct: float
    finite: bool
    steps_completed: int
    boundary_type: str
    device: str
    reference_Ct: float
    reference_source: str
    # Extended diagnostics
    ct_time_series: list[dict[str, Any]]
    wetted_area: float
    dynamic_pressure: float
    tau: float
    nu: float
    lambda_trt: float
    density_min: float
    density_max: float
    config_snapshot: dict[str, Any]

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable machine-readable artifact."""
        return {
            "schema": "tensorlbm.suboff-trt-sgs-diagnostic/v1",
            "status": self.status,
            "physical_validation": self.physical_validation,
            "Re": self.Re,
            "collision": self.collision,
            "turbulence_model": self.turbulence_model,
            "Ct": self.Ct,
            "finite": self.finite,
            "steps_completed": self.steps_completed,
            "boundary_type": self.boundary_type,
            "device": self.device,
            "reference_Ct": self.reference_Ct,
            "reference_source": self.reference_source,
            "ct_time_series": self.ct_time_series,
            "wetted_area": self.wetted_area,
            "dynamic_pressure": self.dynamic_pressure,
            "tau": self.tau,
            "nu": self.nu,
            "lambda_trt": self.lambda_trt,
            "density_min": self.density_min,
            "density_max": self.density_max,
            "config": self.config_snapshot,
        }

    def write_artifact(self, path: str | Path) -> None:
        """Write the evidence artifact as a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_artifact(), sort_keys=True, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Collision: TRT × SGS for D3Q27
# ---------------------------------------------------------------------------

def _collide_trt_sgs_27(
    f: torch.Tensor,
    *,
    tau: float,
    turbulence_model: TurbulenceModel = "none",
    lambda_trt: float = 3.0 / 16.0,
    smagorinsky_cs: float = 0.1,
    wale_cw: float = 0.5,
    vreman_cv: float = 0.025,
) -> torch.Tensor:
    """Combined D3Q27 TRT collision with optional SGS eddy viscosity.

    For ``turbulence_model="none"`` the collision is dispatched through
    ``collide_advanced_3d`` (unified contract dispatch).

    For SGS models (Smagorinsky, WALE, Vreman) the per-cell effective
    relaxation time τ_eff is computed from the SGS eddy viscosity and
    used as the TRT symmetric relaxation time τ₊.  The anti-symmetric
    rate τ₋ is derived per-cell from the magic parameter Λ.

    This composes existing operators and does not modify any solver
    hot path.
    """
    if turbulence_model == "none":
        return collide_advanced_3d(
            "D3Q27", "TRT", f, tau=tau, lambda_trt=lambda_trt,
        )

    # --- Compute macroscopic variables and equilibrium ---
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    # --- Compute per-cell effective tau from SGS model ---
    if turbulence_model == "smagorinsky":
        pi_norm = _neq_stress_norm_27(f_neq)
        tau_eff = _smagorinsky_tau(tau, pi_norm, rho, smagorinsky_cs)
    elif turbulence_model == "wale":
        nu_t = _wale_nu_t_3d(ux, uy, uz, wale_cw)
        tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    elif turbulence_model == "vreman":
        nu_t = _vreman_nu_t_3d(ux, uy, uz, vreman_cv)
        tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    else:
        raise ValueError(f"Unknown turbulence_model: {turbulence_model}")

    # --- TRT collision with per-cell tau_eff ---
    # tau_eff is (nz, ny, nx); need (1, nz, ny, nx) for broadcasting
    tau_plus = tau_eff  # per-cell
    tau_minus = 0.5 + lambda_trt / (tau_eff - 0.5)  # per-cell

    opp = OPPOSITE.to(f.device)
    f_plus = 0.5 * (f + f[opp])
    f_minus = 0.5 * (f - f[opp])
    feq_plus = 0.5 * (feq + feq[opp])
    feq_minus = 0.5 * (feq - feq[opp])

    return (
        f
        - (f_plus - feq_plus) / tau_plus.unsqueeze(0)
        - (f_minus - feq_minus) / tau_minus.unsqueeze(0)
    )


# ---------------------------------------------------------------------------
# D3Q27 far-field boundary condition
# ---------------------------------------------------------------------------

def _far_field_bc_27(
    f: torch.Tensor,
    u_in: float,
    obstacle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Free-stream (Dirichlet) far-field boundary condition for D3Q27.

    Imposes free-stream equilibrium on the inlet and all four lateral
    faces (y±, z±), zero-gradient outlet at x=nx-1.  This is the D3Q27
    analogue of ``boundaries3d.far_field_bc_3d`` (which is D3Q19-only).
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones((nz, ny, nx), dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1,
        torch.full_like(rho1, u_in),
        torch.zeros_like(rho1),
        torch.zeros_like(rho1),
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet (free stream)
    f[:, :, :, -1] = f[:, :, :, -2]        # outlet (zero gradient)
    f[:, 0, :, :] = feq[:, 0, :, :]        # y- lateral
    f[:, -1, :, :] = feq[:, -1, :, :]      # y+ lateral
    f[:, :, 0, :] = feq[:, :, 0, :]        # z- lateral
    f[:, :, -1, :] = feq[:, :, -1, :]      # z+ lateral
    if obstacle_mask is not None:
        f = _bounce_back_cells_27(f, obstacle_mask)
    return f


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suboff_trt_sgs(
    config: SuboffTrtConfig | None = None,
) -> SuboffTrtEvidence:
    """Run SUBOFF bare-hull D3Q27 TRT×SGS and produce diagnostic evidence.

    This function:
      1. Builds a SUBOFF bare-hull solid mask on the configured grid.
      2. Initializes D3Q27 populations at free-stream equilibrium.
      3. Runs a real collide→stream→force→bounce-back→far-field-BC loop.
      4. Records per-step Ct time series.
      5. Returns a ``diagnostic_only`` evidence artifact.

    The solver hot path is not modified.  Only existing operators are
    composed.
    """
    if config is None:
        config = SuboffTrtConfig()

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
    f = equilibrium27(rho0, ux0, uy0, uz0)
    initial_mass = float(f.sum().item())

    tau = config.tau

    # --- 3. Solver loop ---
    ct_series: list[dict[str, Any]] = []
    completed_steps = 0
    all_finite = True
    density_min = float("inf")
    density_max = float("-inf")

    for step in range(1, config.n_steps + 1):
        # Collision (TRT × SGS via unified dispatch)
        f = _collide_trt_sgs_27(
            f,
            tau=tau,
            turbulence_model=config.turbulence_model,
            lambda_trt=config.lambda_trt,
            smagorinsky_cs=config.smagorinsky_cs,
            wale_cw=config.wale_cw,
            vreman_cv=config.vreman_cv,
        )
        # Streaming (D3Q27 pull scheme)
        f = stream27(f)
        # Force measurement (momentum exchange, before bounce-back)
        fx_t, fy_t, fz_t = compute_obstacle_forces_27(f, solid)
        fx = float(fx_t.item())
        fy = float(fy_t.item())
        fz = float(fz_t.item())
        # Bounce-back on solid (static wall, D3Q27)
        f = _bounce_back_cells_27(f, solid)
        # Far-field boundary condition (D3Q27)
        f = _far_field_bc_27(f, config.u_in, obstacle_mask=solid)
        # Mass correction
        if step % config.mass_correction_interval == 0:
            f = correct_mass27(f, initial_mass)

        # Record Ct
        ct = fx / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_series.append({
            "step": step,
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "ct": ct,
        })

        # Runtime finiteness checks
        completed_steps = step
        pop_finite = bool(torch.isfinite(f).all().item())
        all_finite = all_finite and pop_finite

        rho_step, _, _, _ = macroscopic27(f)
        dens_finite = bool(torch.isfinite(rho_step).all().item())
        if dens_finite:
            density_min = min(density_min, float(rho_step.min().item()))
            density_max = max(density_max, float(rho_step.max().item()))

    # Final Ct (last step)
    final_ct = ct_series[-1]["ct"] if ct_series else 0.0

    # --- 4. Build evidence ---
    config_snapshot = {
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "n_steps": config.n_steps,
        "u_in": config.u_in,
        "re": config.re,
        "hull_length": config.hull_length,
        "turbulence_model": config.turbulence_model,
        "lambda_trt": config.lambda_trt,
        "smagorinsky_cs": config.smagorinsky_cs,
        "wale_cw": config.wale_cw,
        "vreman_cv": config.vreman_cv,
        "lattice": "D3Q27",
        "hull_type": "bare_hull",
    }

    return SuboffTrtEvidence(
        status="diagnostic_only",
        physical_validation=False,
        Re=config.re,
        collision="TRT",
        turbulence_model=config.turbulence_model,
        Ct=final_ct,
        finite=all_finite,
        steps_completed=completed_steps,
        boundary_type=config.boundary_type,
        device=config.device,
        reference_Ct=ITTC_1957_REFERENCE_CT,
        reference_source="ITTC-1957",
        ct_time_series=ct_series,
        wetted_area=wetted_area,
        dynamic_pressure=dynamic_pressure,
        tau=tau,
        nu=config.nu,
        lambda_trt=config.lambda_trt,
        density_min=density_min if density_min != float("inf") else 0.0,
        density_max=density_max if density_max != float("-inf") else 0.0,
        config_snapshot=config_snapshot,
    )


# ---------------------------------------------------------------------------
# Campaign: run multiple configs (optionally in parallel)
# ---------------------------------------------------------------------------

def run_suboff_trt_sgs_campaign(
    configs: list[SuboffTrtConfig],
) -> list[SuboffTrtEvidence]:
    """Run a campaign of TRT×SGS configurations sequentially.

    For parallel execution on separate SDAA devices, use
    ``run_suboff_trt_sgs_campaign_parallel`` or launch individual
    ``run_suboff_trt_sgs`` calls as separate background processes.
    """
    return [run_suboff_trt_sgs(cfg) for cfg in configs]
