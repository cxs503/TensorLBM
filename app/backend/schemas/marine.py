"""Versioned backend contracts for marine and ship-engineering workflows.

The models in this module are solver-agnostic.  They make a ship case and a
preflight result serialisable, reviewable API artifacts before a solver is
selected or a job is submitted.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


VesselType = Literal["surface_ship", "submarine", "offshore_structure"]
CheckStatus = Literal["pass", "warning", "error"]
PreflightDecision = Literal["ready", "review", "blocked"]


class WaterProperties(BaseModel):
    """Fluid properties in SI units at the intended operating condition."""

    density_kg_m3: float = Field(1025.0, gt=0.0)
    kinematic_viscosity_m2_s: float = Field(1.05e-6, gt=0.0)
    temperature_c: float | None = None


class MarineMeshSettings(BaseModel):
    """Resolution and outer-domain controls independent of a mesh generator."""

    cells_per_length: int = Field(160, ge=20, le=10_000)
    domain_length_factor: float = Field(3.0, ge=1.0, le=20.0)
    lateral_clearance_factor: float = Field(1.0, ge=0.1, le=20.0)
    vertical_clearance_factor: float = Field(1.0, ge=0.1, le=20.0)


class MarineNumericsSettings(BaseModel):
    """Optional lattice controls used by an LBM-oriented readiness check."""

    lattice_inlet_velocity: float = Field(0.05, gt=0.0, lt=1.0)
    distribution_count: int = Field(19, ge=1, le=128)
    bytes_per_distribution: int = Field(4, ge=2, le=16)
    memory_safety_factor: float = Field(2.0, ge=1.0, le=10.0)


class ShipCase(BaseModel):
    """Validated SI-unit ship/marine engineering case submitted to preflight."""

    case_name: str = Field(..., min_length=1, max_length=160)
    vessel_type: VesselType = "surface_ship"
    length_between_perpendiculars_m: float = Field(..., gt=0.0)
    beam_m: float = Field(..., gt=0.0)
    draft_m: float = Field(..., gt=0.0)
    design_speed_ms: float = Field(..., gt=0.0)
    operating_depth_m: float | None = Field(None, ge=0.0)
    water: WaterProperties = Field(default_factory=WaterProperties)
    mesh: MarineMeshSettings = Field(default_factory=MarineMeshSettings)
    numerics: MarineNumericsSettings = Field(default_factory=MarineNumericsSettings)

    @model_validator(mode="after")
    def _validate_vessel_context(self) -> "ShipCase":
        if self.vessel_type == "submarine" and self.operating_depth_m is None:
            raise ValueError("operating_depth_m is required for submarine cases")
        return self


class MarineCheck(BaseModel):
    """One deterministic preflight finding with a stable machine-readable code."""

    code: str = Field(..., min_length=1)
    status: CheckStatus
    message: str
    value: float | None = None
    limit: float | None = None
    unit: str | None = None
    recommendation: str | None = None


class MarineIssue(BaseModel):
    code: str
    message: str
    recommendation: str | None = None


class MarineDerivedQuantities(BaseModel):
    reynolds_number: float = Field(ge=0.0)
    froude_number: float = Field(ge=0.0)
    length_to_beam_ratio: float = Field(1.0, gt=0.0)
    beam_to_draft_ratio: float = Field(1.0, gt=0.0)
    estimated_lattice_mach: float = Field(0.0, ge=0.0)


class MarineResourceEstimate(BaseModel):
    grid_nx: int = Field(1, ge=1)
    grid_ny: int = Field(1, ge=1)
    grid_nz: int = Field(1, ge=1)
    total_cells: int = Field(ge=1)
    distribution_count: int = Field(ge=1)
    memory_mb: float = Field(gt=0.0)


class MarinePreflightResult(BaseModel):
    """Stable ``marine-preflight/v1`` result contract returned by the API."""

    contract_version: Literal["marine-preflight/v1"] = "marine-preflight/v1"
    case_name: str
    decision: PreflightDecision
    checks: list[MarineCheck]
    blocking_issues: list[MarineIssue]
    warnings: list[MarineIssue]
    derived: MarineDerivedQuantities
    resource_estimate: MarineResourceEstimate
