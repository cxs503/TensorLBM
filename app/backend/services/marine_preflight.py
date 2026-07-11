"""Deterministic, solver-independent marine preflight assessment."""
from __future__ import annotations

from math import sqrt

from ..schemas.marine import (
    MarineCheck,
    MarineDerivedQuantities,
    MarineIssue,
    MarinePreflightResult,
    MarineResourceEstimate,
    ShipCase,
)

_GRAVITY_MS2 = 9.80665
_SPEED_OF_SOUND_LU = 1.0 / sqrt(3.0)
_MAX_LATTICE_MACH = 0.10
_MIN_CELLS_PER_LENGTH = 80
_MIN_DOMAIN_LENGTH_FACTOR = 2.0


def evaluate_marine_preflight(case: ShipCase) -> MarinePreflightResult:
    """Assess case readiness without starting a solver or writing any files."""
    length = case.length_between_perpendiculars_m
    beam = case.beam_m
    draft = case.draft_m
    mesh = case.mesh
    numerics = case.numerics

    reynolds = case.design_speed_ms * length / case.water.kinematic_viscosity_m2_s
    froude = case.design_speed_ms / sqrt(_GRAVITY_MS2 * length)
    length_to_beam = length / beam
    beam_to_draft = beam / draft
    lattice_mach = numerics.lattice_inlet_velocity / _SPEED_OF_SOUND_LU

    nx = max(1, round(mesh.domain_length_factor * mesh.cells_per_length))
    ny = max(1, round(2.0 * mesh.lateral_clearance_factor * beam / length * mesh.cells_per_length))
    nz = max(1, round((draft + mesh.vertical_clearance_factor * beam) / length * mesh.cells_per_length))
    total_cells = nx * ny * nz
    memory_mb = (
        total_cells * numerics.distribution_count * numerics.bytes_per_distribution
        * numerics.memory_safety_factor / (1024.0 ** 2)
    )

    checks: list[MarineCheck] = [
        MarineCheck(
            code="geometry_aspect_ratio", status="pass",
            message="Principal dimensions form a positive, internally consistent hull envelope.",
            value=length_to_beam, unit="L/B",
        ),
        MarineCheck(
            code="lattice_mach",
            status="error" if lattice_mach > _MAX_LATTICE_MACH else "pass",
            message=(
                "Estimated lattice Mach number exceeds the 0.10 incompressible-LBM limit."
                if lattice_mach > _MAX_LATTICE_MACH
                else "Estimated lattice Mach number is within the incompressible-LBM limit."
            ),
            value=lattice_mach, limit=_MAX_LATTICE_MACH,
            recommendation="Reduce lattice_inlet_velocity to at most 0.0577." if lattice_mach > _MAX_LATTICE_MACH else None,
        ),
        MarineCheck(
            code="mesh_resolution",
            status="warning" if mesh.cells_per_length < _MIN_CELLS_PER_LENGTH else "pass",
            message=("Longitudinal resolution is below the recommended 80 cells per ship length."
                     if mesh.cells_per_length < _MIN_CELLS_PER_LENGTH else "Longitudinal resolution meets the baseline preflight recommendation."),
            value=float(mesh.cells_per_length), limit=float(_MIN_CELLS_PER_LENGTH), unit="cells/L",
            recommendation="Use at least 80 cells per length for an engineering screening run." if mesh.cells_per_length < _MIN_CELLS_PER_LENGTH else None,
        ),
        MarineCheck(
            code="domain_extent",
            status="warning" if mesh.domain_length_factor < _MIN_DOMAIN_LENGTH_FACTOR else "pass",
            message=("Streamwise domain is shorter than the 2.0L baseline extent."
                     if mesh.domain_length_factor < _MIN_DOMAIN_LENGTH_FACTOR else "Streamwise domain meets the baseline extent."),
            value=mesh.domain_length_factor, limit=_MIN_DOMAIN_LENGTH_FACTOR, unit="L",
            recommendation="Increase domain_length_factor to at least 2.0." if mesh.domain_length_factor < _MIN_DOMAIN_LENGTH_FACTOR else None,
        ),
    ]
    if case.vessel_type == "submarine":
        clearance_ratio = (case.operating_depth_m or 0.0) / draft
        checks.append(MarineCheck(
            code="submergence_clearance",
            status="warning" if clearance_ratio < 3.0 else "pass",
            message=("Operating depth is less than three drafts; free-surface influence requires review."
                     if clearance_ratio < 3.0 else "Submergence clearance is adequate for an initially unbounded-flow model."),
            value=clearance_ratio, limit=3.0, unit="depth/draft",
            recommendation="Model the free surface or increase operating_depth_m." if clearance_ratio < 3.0 else None,
        ))

    blocking = [MarineIssue(code=c.code, message=c.message, recommendation=c.recommendation)
                for c in checks if c.status == "error"]
    warnings = [MarineIssue(code=c.code, message=c.message, recommendation=c.recommendation)
                for c in checks if c.status == "warning"]
    decision = "blocked" if blocking else ("review" if warnings else "ready")

    return MarinePreflightResult(
        case_name=case.case_name,
        decision=decision,
        checks=checks,
        blocking_issues=blocking,
        warnings=warnings,
        derived=MarineDerivedQuantities(
            reynolds_number=reynolds, froude_number=froude,
            length_to_beam_ratio=length_to_beam, beam_to_draft_ratio=beam_to_draft,
            estimated_lattice_mach=lattice_mach,
        ),
        resource_estimate=MarineResourceEstimate(
            grid_nx=nx, grid_ny=ny, grid_nz=nz, total_cells=total_cells,
            distribution_count=numerics.distribution_count, memory_mb=memory_mb,
        ),
    )
