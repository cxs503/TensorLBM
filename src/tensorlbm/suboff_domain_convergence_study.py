"""SUBOFF bare-hull domain convergence study runner (D3Q19+MRT, diagnostic only).

This runner fixes the hull length, grid resolution (dx = 1 lattice unit), and
step count, then varies the computational domain size across at least three
levels for the SUBOFF bare hull.  Each level runs a real D3Q19+MRT bounce-back
simulation via :func:`tensorlbm.suboff_validation_runner.run_suboff_d3q19_mrt_validation`
and produces a ``measured_candidate`` evidence artifact with force/Ct time
series.  The study collects the mean Ct per level and computes relative-change
indicators, but deliberately withholds any convergence or physical-validation
claim.

The study is a **diagnostic only**: it shows how the measured Ct candidate
varies across domain sizes (i.e., as blockage ratio decreases), but does not
assert that the solution has converged or that the Ct values are physically
validated.

This module composes existing cold-path runners and does **not** modify any
solver hot path.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, pi
from pathlib import Path
from typing import Any

from .suboff_cad import SuboffConfig
from .suboff_validation_runner import (
    SuboffValidationConfig,
    SuboffValidationEvidence,
    run_suboff_d3q19_mrt_validation,
)

__all__ = [
    "DomainLevel",
    "DomainConvergenceStudyConfig",
    "run_suboff_domain_convergence_study",
]

_SCHEMA = "suboff-domain-convergence-study-r1"
_MIN_DOMAIN_LEVELS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DomainLevel:
    """One domain size level for the domain convergence study.

    The hull axis runs along x (axial / flow direction).  ``nx`` is the axial
    domain dimension; ``ny`` and ``nz`` are the transverse domain dimensions.
    All three vary between levels to change the domain-to-body ratio (blockage
    ratio), while the hull length and grid spacing (dx = 1) stay fixed.
    """

    level_id: str
    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        if not isinstance(self.level_id, str) or not self.level_id:
            raise ValueError("level_id must be a non-empty string")
        for name, minimum in (("nx", 16), ("ny", 8), ("nz", 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True)
class DomainConvergenceStudyConfig:
    """Configuration for the SUBOFF domain convergence study.

    All domain levels share the same D3Q19+MRT composition, hull length, step
    count, inlet velocity, and Reynolds number.  Only the domain size
    (nx, ny, nz) varies between levels, changing the blockage ratio.

    Parameters
    ----------
    domain_levels :
        At least three :class:`DomainLevel` instances with strictly increasing
        domain sizes.
    hull_length :
        Hull length in lattice units, fixed across all levels (default 24.0).
    n_steps :
        Number of solver steps per level, fixed across all levels (default 20).
    warmup :
        Number of warmup steps before Ct averaging begins (default 5).
    u_in :
        Inlet velocity in lattice units, fixed across all levels (default 0.06).
    re :
        Reynolds number, fixed across all levels (default 200.0).
    device :
        PyTorch device string (default "cpu").
    lattice :
        Lattice stencil; must be "D3Q19" (default "D3Q19").
    collision :
        Collision operator; must be "MRT" (default "MRT").
    hull_type :
        Hull type; must be "bare_hull" (default "bare_hull").
    """

    domain_levels: tuple[DomainLevel, ...]
    hull_length: float = 24.0
    n_steps: int = 20
    warmup: int = 5
    u_in: float = 0.06
    re: float = 200.0
    device: str = "cpu"
    lattice: str = "D3Q19"
    collision: str = "MRT"
    hull_type: str = "bare_hull"

    def __post_init__(self) -> None:
        if not isinstance(self.domain_levels, tuple) or len(self.domain_levels) < _MIN_DOMAIN_LEVELS:
            raise ValueError(f"domain convergence study requires at least {_MIN_DOMAIN_LEVELS} domain levels")
        if any(not isinstance(level, DomainLevel) for level in self.domain_levels):
            raise TypeError("domain_levels must contain only DomainLevel instances")
        if self.hull_type != "bare_hull":
            raise ValueError("domain convergence study supports only hull_type='bare_hull'")
        if self.lattice != "D3Q19":
            raise ValueError("domain convergence study requires lattice='D3Q19'")
        if self.collision != "MRT":
            raise ValueError("domain convergence study requires collision='MRT'")
        _finite_positive(self.hull_length, "hull_length")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        _finite_positive(self.u_in, "u_in")
        if not 0.0 < self.u_in < 0.15:
            raise ValueError("u_in must be in (0, 0.15)")
        _finite_positive(self.re, "re")
        # Ensure domain sizes are strictly increasing in nx
        nxs = [level.nx for level in self.domain_levels]
        if nxs != sorted(nxs) or len(set(nxs)) != len(nxs):
            raise ValueError("domain levels must have strictly increasing nx values")


# ---------------------------------------------------------------------------
# Per-level execution
# ---------------------------------------------------------------------------

def _run_one_level(
    level: DomainLevel,
    config: DomainConvergenceStudyConfig,
) -> dict[str, Any]:
    """Run one domain level and extract per-level data.

    Creates a :class:`SuboffValidationConfig` with the level's domain size and
    the fixed hull length / step count, runs the validation runner, and
    extracts the mean Ct over post-warmup steps.
    """
    val_config = SuboffValidationConfig(
        nx=level.nx,
        ny=level.ny,
        nz=level.nz,
        n_steps=config.n_steps,
        warmup=config.warmup,
        u_in=config.u_in,
        re=config.re,
        hull_length=config.hull_length,
        device=config.device,
    )
    evidence: SuboffValidationEvidence = run_suboff_d3q19_mrt_validation(val_config)

    # Extract Ct: mean over post-warmup steps
    ct_series = evidence.ct_time_series
    post_warmup_cts = [
        s["ct"] for s in ct_series if s["step"] > config.warmup
    ]
    if post_warmup_cts:
        ct_mean = sum(post_warmup_cts) / len(post_warmup_cts)
    else:
        # Fallback: use last step's Ct if all steps are within warmup
        ct_mean = ct_series[-1]["ct"] if ct_series else 0.0

    # Compute blockage ratio: hull cross-section / domain cross-section
    cad_config = SuboffConfig()
    hull_radius = cad_config.r_over_l * config.hull_length
    hull_cross_section = pi * hull_radius * hull_radius
    domain_cross_section = level.ny * level.nz
    blockage_ratio = hull_cross_section / domain_cross_section if domain_cross_section > 0 else float("inf")

    # Domain length in lattice units (dx = 1)
    domain_length_lu = float(level.nx)

    return {
        "level_id": level.level_id,
        "nx": level.nx,
        "ny": level.ny,
        "nz": level.nz,
        "domain_length_lu": domain_length_lu,
        "hull_length_lu": config.hull_length,
        "hull_radius_lu": hull_radius,
        "blockage_ratio": blockage_ratio,
        "Ct": ct_mean,
        "ct_mean": ct_mean,
        "ct_time_series": evidence.ct_time_series,
        "force_time_series": evidence.force_time_series,
        "wetted_area": evidence.wetted_area,
        "dynamic_pressure": evidence.dynamic_pressure,
        "evidence_status": evidence.status,
        "physical_validation": evidence.physical_validation,
        "steady_state": evidence.steady_state,
        "runtime": evidence.runtime,
        "admission": evidence.admission,
        "config": evidence.config,
    }


# ---------------------------------------------------------------------------
# Convergence indicator
# ---------------------------------------------------------------------------

def _compute_convergence_indicator(ct_values: list[float]) -> dict[str, Any]:
    """Compute relative-change indicators without claiming convergence."""
    relative_changes: list[float] = []
    for i in range(len(ct_values) - 1):
        ct_prev = ct_values[i]
        ct_next = ct_values[i + 1]
        if ct_prev == 0.0:
            relative_changes.append(float("inf"))
        else:
            relative_changes.append(abs(ct_next - ct_prev) / abs(ct_prev))

    # Determine trend direction
    diffs = []
    for i in range(len(ct_values) - 1):
        diffs.append(ct_values[i + 1] - ct_values[i])
    if diffs and all(d < 0 for d in diffs):
        ct_trend = "decreasing"
    elif diffs and all(d > 0 for d in diffs):
        ct_trend = "increasing"
    else:
        ct_trend = "non_monotonic"

    finite_changes = [c for c in relative_changes if isfinite(c)]
    max_relative_change = max(finite_changes) if finite_changes else float("inf")

    return {
        "relative_ct_changes": relative_changes,
        "ct_trend": ct_trend,
        "max_relative_change": max_relative_change,
        "convergence_claim": "withheld",
        "note": (
            "Relative Ct changes across domain sizes are diagnostic indicators "
            "only; they do not constitute a domain convergence claim or physical "
            "validation."
        ),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_suboff_domain_convergence_study(
    config: DomainConvergenceStudyConfig,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the SUBOFF bare-hull domain convergence study.

    Executes the D3Q19+MRT+bounce-back validation runner at each domain level
    with fixed hull length and step count, collects the mean Ct candidate per
    level, and computes relative-change convergence indicators.  The result is
    a machine-readable artifact with ``status='diagnostic_only'`` and
    ``physical_validation=False``.

    Parameters
    ----------
    config :
        Study configuration with at least three domain levels.
    output_path :
        Optional path to write the JSON artifact.  When provided, the artifact
        is written to disk in addition to being returned.

    Returns
    -------
    dict
        Machine-readable convergence artifact.
    """
    if not isinstance(config, DomainConvergenceStudyConfig):
        raise TypeError("config must be a DomainConvergenceStudyConfig")

    per_level_results = [_run_one_level(level, config) for level in config.domain_levels]
    ct_per_level = [result["Ct"] for result in per_level_results]
    convergence_indicator = _compute_convergence_indicator(ct_per_level)

    domain_level_records = [
        {
            "level_id": level.level_id,
            "nx": level.nx,
            "ny": level.ny,
            "nz": level.nz,
            "domain_length_lu": float(level.nx),
            "blockage_ratio": pr["blockage_ratio"],
        }
        for level, pr in zip(config.domain_levels, per_level_results)
    ]

    provenance = {
        "runner_api": "tensorlbm.suboff_validation_runner.run_suboff_d3q19_mrt_validation",
        "model_identity": {
            "lattice": config.lattice,
            "collision": config.collision,
            "hull_type": config.hull_type,
            "boundary": "bounce_back",
            "wall": "static",
            "physics": "single_phase_incompressible",
        },
        "cad_source": "tensorlbm.suboff_cad.build_suboff_mask",
        "force_method": "d3q19_momentum_exchange",
        "ct_extraction": "mean_over_post_warmup_steps",
        "reference_area_mode": "voxel_wetted_area_lattice_units",
        "fixed_parameters": {
            "hull_length_lu": config.hull_length,
            "n_steps": config.n_steps,
            "warmup": config.warmup,
            "u_in": config.u_in,
            "re": config.re,
            "grid_spacing_dx": 1.0,
        },
        "prohibition": "no_convergence_claim_or_physical_validation",
    }

    payload: dict[str, Any] = {
        "artifact_kind": "suboff_domain_convergence_study",
        "schema": _SCHEMA,
        "status": "diagnostic_only",
        "physical_validation": False,
        "domain_levels": domain_level_records,
        "Ct_per_level": ct_per_level,
        "convergence_indicator": convergence_indicator,
        "per_level_results": per_level_results,
        "provenance": provenance,
    }
    payload["provenance_hash"] = _canonical_hash(payload)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    return payload
