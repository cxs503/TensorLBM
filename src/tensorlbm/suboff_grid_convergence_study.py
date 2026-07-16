"""SUBOFF bare-hull grid convergence study runner (D3Q19+MRT, diagnostic only).

This runner executes the production full-wet force window campaign at three or
more systematically refined grid levels for the SUBOFF bare hull.  It collects
the measured Ct candidate per level and computes relative-change indicators,
but deliberately withholds any convergence or physical-validation claim.

The study is a **diagnostic only**: it shows how the measured Ct candidate
varies across grid resolutions, but does not assert that the solution has
converged or that the Ct values are physically validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite, pi, sqrt
from pathlib import Path
from typing import Any

import torch

from .backends.contracts import DeviceSpec
from .full_wet import FullyWettedFlowConfig, VoxelBodyGeometry
from .marine_geometry import GeometryAsset
from .models.contracts import ModelComposition
from .suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask
from .suboff_full_wet_force_window_campaign import run_suboff_full_wet_force_window_campaign
from .suboff_real_state_force import SuboffRealStateForceConfig

_SCHEMA = "suboff-grid-convergence-study-r1"
_MIN_GRID_LEVELS = 3


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class GridLevel:
    """One systematically refined grid level for the convergence study.

    The hull axis runs along x (axial / flow direction).  ``nx`` is the axial
    dimension; ``ny`` and ``nz`` are the transverse dimensions.
    """

    level_id: str
    nx: int
    ny: int
    nz: int
    steps: int
    capture_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.level_id, str) or not self.level_id:
            raise ValueError("level_id must be a non-empty string")
        for name in ("nx", "ny", "nz", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.capture_steps, tuple) or len(self.capture_steps) < 2:
            raise ValueError("capture_steps must be a tuple of at least two step indices")
        if any(
            isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= self.steps
            for s in self.capture_steps
        ):
            raise ValueError("capture_steps must contain step indices in [1, steps]")
        if tuple(sorted(self.capture_steps)) != self.capture_steps or (
            len(set(self.capture_steps)) != len(self.capture_steps)
        ):
            raise ValueError("capture_steps must be unique and ascending")


@dataclass(frozen=True, slots=True)
class GridConvergenceStudyConfig:
    """Configuration for the SUBOFF grid convergence study.

    All grid levels share the same D3Q19+MRT composition, tau, inlet velocity,
    and bare-hull geometry type.  The reference area for Ct normalization is
    the hull cross-sectional area ``pi * R^2`` in lattice units, computed
    per-level from the parametric SUBOFF geometry.
    """

    grid_levels: tuple[GridLevel, ...]
    tau: float = 0.6
    inlet_velocity: float = 0.03
    hull_type: str = "bare_hull"
    lattice: str = "D3Q19"
    collision: str = "MRT"
    cad_config: SuboffConfig = field(default_factory=SuboffConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.grid_levels, tuple) or len(self.grid_levels) < _MIN_GRID_LEVELS:
            raise ValueError(f"grid convergence study requires at least {_MIN_GRID_LEVELS} grid levels")
        if any(not isinstance(level, GridLevel) for level in self.grid_levels):
            raise TypeError("grid_levels must contain only GridLevel instances")
        if self.hull_type != "bare_hull":
            raise ValueError("grid convergence study supports only hull_type='bare_hull'")
        if self.lattice != "D3Q19":
            raise ValueError("grid convergence study requires lattice='D3Q19'")
        if self.collision != "MRT":
            raise ValueError("grid convergence study requires collision='MRT'")
        _finite_positive(self.tau, "tau")
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")
        _finite_positive(self.inlet_velocity, "inlet_velocity")
        if not 0.0 < self.inlet_velocity < 0.15:
            raise ValueError("inlet_velocity must be in (0, 0.15)")
        if not isinstance(self.cad_config, SuboffConfig):
            raise TypeError("cad_config must be a SuboffConfig")


def _build_composition() -> ModelComposition:
    return ModelComposition(
        lattice="D3Q19",
        collision="MRT",
        turbulence=None,
        forcing=(),
        boundaries=("zou_he_channel", "stationary_bounce_back"),
        physics_modules={"single_phase": "incompressible"},
    )


def _run_one_level(
    level: GridLevel,
    config: GridConvergenceStudyConfig,
) -> dict[str, Any]:
    """Build geometry, run the force window campaign, and extract per-level data."""
    mask, stats = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=level.nx,
        ny=level.ny,
        nz=level.nz,
        config=config.cad_config,
    )
    nz, ny, nx = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
    shape: tuple[int, int, int] = (nz, ny, nx)
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
    length = stats["length"]
    radius = stats["radius"]
    solid_cells = stats["solid_cells"]

    asset = GeometryAsset(
        mask,
        f"suboff-bare-hull-{level.level_id}",
        (cx, cy, cz),
        "lattice",
        "suboff-cad-parametric",
    )

    flow_config = FullyWettedFlowConfig(
        geometry=VoxelBodyGeometry(mask, f"suboff-bare-hull-{level.level_id}", origin=(cx, cy, cz)),
        composition=_build_composition(),
        device_spec=DeviceSpec("cpu", "float32"),
        shape=shape,
        tau=config.tau,
        inlet_velocity=config.inlet_velocity,
        steps=level.steps,
        capture_population_steps=level.capture_steps,
    )

    # Cross-sectional reference area in lattice units: pi * R^2.
    reference_area = pi * radius * radius
    force_config = SuboffRealStateForceConfig(
        rho=1.0,
        U=config.inlet_velocity,
        reference_area=reference_area,
        length=length,
        direction=(1.0, 0.0, 0.0),
    )

    campaign = run_suboff_full_wet_force_window_campaign(asset, flow_config, force_config=force_config)

    force_window = campaign["force_window"]
    ct = force_window["contract"]["Ct"]
    window_forces = force_window["window_forces"]
    force_time_series = [
        [float(v) for v in force] for force in window_forces
    ]
    mean_force = [float(v) for v in campaign["sample_windows"]["mean_force"]]
    std_force = [float(v) for v in campaign["sample_windows"]["std_force"]]

    return {
        "level_id": level.level_id,
        "grid_shape_zyx": list(shape),
        "nx": level.nx,
        "ny": level.ny,
        "nz": level.nz,
        "steps": level.steps,
        "capture_steps": list(level.capture_steps),
        "solid_cells": solid_cells,
        "fluid_cells": stats["fluid_cells"],
        "total_cells": stats["total_cells"],
        "hull_length_lu": length,
        "hull_radius_lu": radius,
        "reference_area_lu2": reference_area,
        "link_count": campaign["sample_windows"]["link_count"],
        "campaign_status": campaign["status"],
        "physical_validation": campaign["physical_validation"],
        "Ct": ct,
        "force_time_series": force_time_series,
        "mean_force": mean_force,
        "std_force": std_force,
        "campaign_provenance_hash": campaign["provenance_hash"],
        "geometry_source_hash": asset.source_hash,
    }


def _compute_convergence_indicator(ct_values: list[float | None]) -> dict[str, Any]:
    """Compute relative-change indicators without claiming convergence."""
    relative_changes: list[float] = []
    for i in range(len(ct_values) - 1):
        ct_prev = ct_values[i]
        ct_next = ct_values[i + 1]
        if ct_prev is None or ct_next is None or ct_prev == 0.0:
            relative_changes.append(float("inf"))
        else:
            relative_changes.append(abs(ct_next - ct_prev) / abs(ct_prev))

    # Determine trend direction
    diffs = []
    for i in range(len(ct_values) - 1):
        ct_prev = ct_values[i]
        ct_next = ct_values[i + 1]
        if ct_prev is not None and ct_next is not None:
            diffs.append(ct_next - ct_prev)
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
            "Relative Ct changes are diagnostic indicators only; "
            "they do not constitute a grid convergence claim or physical validation."
        ),
    }


def run_suboff_grid_convergence_study(
    config: GridConvergenceStudyConfig,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the SUBOFF bare-hull grid convergence study.

    Executes the production full-wet force window campaign at each grid level,
    collects the measured Ct candidate per level, and computes relative-change
    convergence indicators.  The result is a machine-readable artifact with
    ``status='diagnostic_only'`` and ``physical_validation=False``.

    Parameters
    ----------
    config :
        Study configuration with at least three grid levels.
    output_path :
        Optional path to write the JSON artifact.  When provided, the artifact
        is written to disk in addition to being returned.

    Returns
    -------
    dict
        Machine-readable convergence artifact.
    """
    if not isinstance(config, GridConvergenceStudyConfig):
        raise TypeError("config must be a GridConvergenceStudyConfig")

    per_level_results = [_run_one_level(level, config) for level in config.grid_levels]
    ct_per_level = [result["Ct"] for result in per_level_results]
    convergence_indicator = _compute_convergence_indicator(ct_per_level)

    grid_level_records = [
        {
            "level_id": level.level_id,
            "nx": level.nx,
            "ny": level.ny,
            "nz": level.nz,
            "steps": level.steps,
            "capture_steps": list(level.capture_steps),
        }
        for level in config.grid_levels
    ]

    provenance = {
        "runner_api": (
            "tensorlbm.suboff_full_wet_force_window_campaign."
            "run_suboff_full_wet_force_window_campaign"
        ),
        "model_identity": {
            "lattice": config.lattice,
            "collision": config.collision,
            "hull_type": config.hull_type,
            "boundary": "zou_he_channel+stationary_bounce_back",
            "physics": "single_phase_incompressible",
        },
        "cad_source": "tensorlbm.suboff_cad.build_suboff_mask",
        "force_method": "d3q19_linkwise_momentum_exchange",
        "sample_phase": "post_stream_pre_bounce_back",
        "reference_area_mode": "cross_section_pi_R_squared_lattice_units",
        "prohibition": "no_convergence_claim_or_physical_validation",
    }

    payload: dict[str, Any] = {
        "artifact_kind": "suboff_grid_convergence_study",
        "schema": _SCHEMA,
        "status": "diagnostic_only",
        "physical_validation": False,
        "grid_levels": grid_level_records,
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


__all__ = [
    "GridConvergenceStudyConfig",
    "GridLevel",
    "run_suboff_grid_convergence_study",
]
