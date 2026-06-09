"""Pydantic request schemas for CAD endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HullPreviewRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length: float = Field(100.0, gt=0, description="Hull length (lattice units)")
    beam: float = Field(16.0, gt=0, description="Hull beam (lattice units)")
    draft: float = Field(8.0, gt=0, description="Hull draft (lattice units)")
    n_stations: int = Field(11, ge=3, le=41, description="Number of body-plan stations")

class HullMaskRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    nx: int = Field(160, ge=20, description="Grid x-size")
    ny: int = Field(60, ge=10, description="Grid y-size")
    nz: int = Field(40, ge=10, description="Grid z-size")
    length: float = Field(80.0, gt=0)
    beam: float = Field(8.0, gt=0)
    draft: float = Field(12.0, gt=0)
    cx: float | None = Field(None, description="Midship x (default: nx/2)")
    cy: float | None = Field(None, description="Centreline y (default: ny/2)")
    cz_keel: float | None = Field(None, description="Keel z (default: nz/4)")
    device: str = "cpu"

class LBMParametersRequest(BaseModel):
    length_m: float = Field(100.0, gt=0, description="Ship length [m]")
    speed_ms: float = Field(5.0, gt=0, description="Ship speed [m/s]")
    nu_m2s: float = Field(1.139e-6, gt=0, description="Kinematic viscosity [m²/s]")
    lbm_length: float = Field(100.0, gt=0, description="LBM hull length (cells)")
    lbm_speed: float = Field(0.05, gt=0, description="LBM inlet velocity (lu/step)")
    froude_target: float | None = Field(
        None, ge=0, description="Target Froude number (overrides speed_ms)"
    )

class ResistanceEstimateRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length_m: float = Field(100.0, gt=0, description="Ship length [m]")
    beam_m: float = Field(16.0, gt=0, description="Ship beam [m]")
    draft_m: float = Field(8.0, gt=0, description="Ship draft [m]")
    speed_ms: float = Field(5.0, gt=0, description="Ship speed [m/s]")
    nu_m2s: float = Field(1.139e-6, gt=0, description="Kinematic viscosity [m²/s]")
    rho_kgm3: float = Field(1025.0, gt=0, description="Fluid density [kg/m³]")
    residual_ratio: float = Field(0.18, ge=0.0, le=1.0, description="Residual/friction ratio")

class HullSolverRequest(BaseModel):
    """Launch a ship-hull LBM solver job from CAD parameters."""

    hull_type: Literal["wigley", "series60", "kcs"] = "wigley"
    nx: int = Field(160, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(40, ge=10)
    hull_length: float = Field(80.0, gt=0)
    hull_beam: float = Field(8.0, gt=0)
    hull_draft: float = Field(12.0, gt=0)
    u_in: float = Field(0.05, gt=0)
    re: float = Field(200.0, gt=0)
    smagorinsky_cs: float = Field(0.1, ge=0)
    wave_amp: float = Field(0.0, ge=0)
    wave_period: float = Field(200.0, gt=0)
    n_steps: int = Field(2000, ge=1)
    output_interval: int = Field(200, ge=1)
    device: str = "cpu"
    seed: int = 0

class HullSTLRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length: float = Field(100.0, gt=0)
    beam: float = Field(16.0, gt=0)
    draft: float = Field(8.0, gt=0)
    n_long: int = Field(60, ge=4, le=200)
    n_vert: int = Field(30, ge=4, le=100)

class CAD3DCreateRequest(BaseModel):
    source_type: Literal["parametric", "stl", "step"] = "parametric"
    units: str = "lu"
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length: float = Field(100.0, gt=0)
    beam: float = Field(16.0, gt=0)
    draft: float = Field(8.0, gt=0)
    n_long: int = Field(80, ge=4, le=400)
    n_vert: int = Field(40, ge=4, le=200)
    file_b64: str | None = None
    filename: str | None = None

class CAD3DUpdateRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length: float = Field(100.0, gt=0)
    beam: float = Field(16.0, gt=0)
    draft: float = Field(8.0, gt=0)
    n_long: int = Field(80, ge=4, le=400)
    n_vert: int = Field(40, ge=4, le=200)

class CAD3DExportRequest(BaseModel):
    fmt: Literal["gltf", "stl", "step"] = "gltf"

class CAD3DMaskBridgeRequest(BaseModel):
    nx: int = Field(160, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(40, ge=10)
    device: str = "cpu"

class SuboffPreviewRequest(BaseModel):
    hull_type: Literal["bare_hull", "with_sail", "full"] = "bare_hull"
    length: float = Field(100.0, gt=0, description="Hull length (lattice units)")
    radius: float = Field(0.0, ge=0, description="Max radius (lu); 0 = auto from L/D≈8.57")
    bow_fraction: float = Field(0.233, gt=0, lt=0.9)
    stern_fraction: float = Field(0.252, gt=0, lt=0.9)
    stern_exponent: float = Field(2.0, gt=0, le=8.0)

class SuboffMaskRequest(BaseModel):
    hull_type: Literal["bare_hull", "with_sail", "full"] = "bare_hull"
    nx: int = Field(200, ge=20)
    ny: int = Field(80, ge=10)
    nz: int = Field(80, ge=10)
    length: float = Field(120.0, gt=0)
    radius: float = Field(0.0, ge=0, description="Max radius; 0 = auto")
    cx: float | None = Field(None, description="Axial midpoint (default: nx/2)")
    cy: float | None = Field(None, description="Lateral axis (default: ny/2)")
    cz: float | None = Field(None, description="Vertical axis (default: nz/2)")
    device: str = "cpu"

class SuboffSTLRequest(BaseModel):
    hull_type: Literal["bare_hull", "with_sail", "full"] = "bare_hull"
    length: float = Field(100.0, gt=0)
    radius: float = Field(0.0, ge=0, description="Max radius; 0 = auto")
    n_axial: int = Field(80, ge=8, le=400)
    n_circ: int = Field(60, ge=8, le=200)

class OffshorePreviewRequest(BaseModel):
    struct_type: Literal["monopile", "jacket", "spar", "semi_sub"] = "monopile"
    nx: int = Field(80, ge=20)
    ny: int = Field(80, ge=20)
    nz: int = Field(80, ge=20)
    diameter: float | None = Field(None, gt=0)
    leg_diameter: float | None = Field(None, gt=0)
    foot_spread: float | None = Field(None, gt=0)
    head_spread: float | None = Field(None, gt=0)
    hull_diameter: float | None = Field(None, gt=0)
    keel_diameter: float | None = Field(None, gt=0)
    column_diameter: float | None = Field(None, gt=0)
    pontoon_length: float | None = Field(None, gt=0)
    pontoon_width: float | None = Field(None, gt=0)
    pontoon_height: float | None = Field(None, gt=0)
    column_height: float | None = Field(None, gt=0)

class OffshoreSTLRequest(BaseModel):
    struct_type: Literal["monopile", "jacket", "spar", "semi_sub"] = "monopile"
    nx: int = Field(40, ge=10)
    ny: int = Field(40, ge=10)
    nz: int = Field(40, ge=10)
    diameter: float | None = None
    leg_diameter: float | None = None
    foot_spread: float | None = None
    head_spread: float | None = None
    hull_diameter: float | None = None
    keel_diameter: float | None = None
    column_diameter: float | None = None
    pontoon_length: float | None = None
    pontoon_width: float | None = None
    pontoon_height: float | None = None
    column_height: float | None = None

class PropellerOpenWaterRequest(BaseModel):
    J: float = Field(0.7, ge=0.0, le=1.5, description="Advance ratio")
    P_D: float = Field(1.0, ge=0.5, le=1.4, description="Pitch ratio P/D")
    Ae_A0: float = Field(0.6, ge=0.3, le=1.05, description="Blade area ratio Ae/A0")
    Z: int = Field(4, ge=2, le=7, description="Number of blades")
    n_rps: float | None = Field(None, gt=0, description="Shaft speed [rev/s]")
    D_m: float | None = Field(None, gt=0, description="Propeller diameter [m]")
    rho: float = Field(1025.0, gt=0, description="Water density [kg/m³]")

class PropellerCurveRequest(BaseModel):
    P_D: float = Field(1.0, ge=0.5, le=1.4)
    Ae_A0: float = Field(0.6, ge=0.3, le=1.05)
    Z: int = Field(4, ge=2, le=7)
    J_min: float = Field(0.01, ge=0.0, le=0.5)
    J_max: float = Field(1.35, ge=0.5, le=1.5)
    n_points: int = Field(60, ge=10, le=200)

class PropellerDesignRequest(BaseModel):
    thrust_n: float = Field(500_000.0, gt=0, description="Required thrust [N]")
    Va_ms: float = Field(6.0, gt=0, description="Advance speed [m/s]")
    P_D: float = Field(1.0, ge=0.5, le=1.4)
    Ae_A0: float = Field(0.6, ge=0.3, le=1.05)
    Z: int = Field(4, ge=2, le=7)
    n_rps: float = Field(2.0, gt=0, description="Shaft speed [rev/s]")
    rho: float = Field(1025.0, gt=0)
