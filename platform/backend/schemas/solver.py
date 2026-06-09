"""Pydantic schema models shared by solver routers."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TurbulenceModel = Literal["none", "smagorinsky_les", "dynamic_smagorinsky_les"]
MultiphaseModel = Literal["none", "sc", "scmp", "cg", "fe"]
FlowType = Literal["single_phase", "multiphase", "free_surface"]
BoundaryCondition = Literal["standard_bounce_back", "zou_he", "periodic"]
NumericalScheme = Literal["bgk", "trt", "mrt"]


class PhysicsSelection(BaseModel):
    flow_type: FlowType = "single_phase"
    turbulence_model: TurbulenceModel = "none"
    turbulence_params: dict[str, float] = Field(default_factory=dict)
    multiphase_model: MultiphaseModel = "none"
    multiphase_params: dict[str, float] = Field(default_factory=dict)
    boundary_condition: BoundaryCondition = "standard_bounce_back"
    numerical_scheme: NumericalScheme = "bgk"
    preset: str | None = None

class CylinderFlowParams(BaseModel):
    nx: int = Field(320, ge=20, description="Grid width")
    ny: int = Field(100, ge=10, description="Grid height")
    u_in: float = Field(0.08, gt=0, description="Inlet velocity (lattice units)")
    re: float = Field(100.0, gt=0, description="Reynolds number")
    radius: float = Field(12.0, gt=0, description="Cylinder radius (cells)")
    n_steps: int = Field(1200, ge=1, description="Total time steps")
    output_interval: int = Field(200, ge=1, description="Output every N steps")
    device: str = Field("cpu", description="Torch device (cpu / cuda:0 …)")
    seed: int = 0
    physics: PhysicsSelection | None = None


class CylinderFlowScanParams(BaseModel):
    nx: int = Field(320, ge=20, description="Grid width")
    ny: int = Field(100, ge=10, description="Grid height")
    u_in: float = Field(0.08, gt=0, description="Inlet velocity (lattice units)")
    re_values: list[float] = Field(
        ..., min_length=2, max_length=20, description="Reynolds-number sweep values"
    )
    radius: float = Field(12.0, gt=0, description="Cylinder radius (cells)")
    n_steps: int = Field(1200, ge=1, description="Total time steps")
    output_interval: int = Field(200, ge=1, description="Output every N steps")
    device: str = Field("cpu", description="Torch device (cpu / cuda:0 …)")
    seed: int = 0
    physics: PhysicsSelection | None = None

class CylinderFlowScanParams(BaseModel):
    nx: int = Field(320, ge=20, description="Grid width")
    ny: int = Field(100, ge=10, description="Grid height")
    u_in: float = Field(0.08, gt=0, description="Inlet velocity (lattice units)")
    re_values: list[float] = Field(
        ..., min_length=2, max_length=20, description="Reynolds-number sweep values"
    )
    radius: float = Field(12.0, gt=0, description="Cylinder radius (cells)")
    n_steps: int = Field(1200, ge=1, description="Total time steps")
    output_interval: int = Field(200, ge=1, description="Output every N steps")
    device: str = Field("cpu", description="Torch device (cpu / cuda:0 …)")
    seed: int = 0
    physics: PhysicsSelection | None = None

class LidDrivenCavityParams(BaseModel):
    nx: int = Field(128, ge=8, description="Grid size (square, ny = nx)")
    u_lid: float = Field(0.1, gt=0, description="Lid velocity")
    re: float = Field(100.0, gt=0, description="Reynolds number")
    n_steps: int = Field(10000, ge=1)
    output_interval: int = Field(2000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None

class BackwardFacingStepParams(BaseModel):
    nx: int = Field(400, ge=20)
    ny: int = Field(80, ge=6)
    step_h: int = Field(40, ge=1, description="Step height (cells)")
    x_step: int = Field(80, ge=1, description="Pre-step solid length (cells)")
    u_in: float = Field(0.05, gt=0)
    re: float = Field(100.0, gt=0)
    n_steps: int = Field(30000, ge=1)
    output_interval: int = Field(5000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None

class TurbulentChannelParams(BaseModel):
    nx: int = Field(256, ge=16)
    ny: int = Field(64, ge=8)
    re_tau: float = Field(100.0, gt=0, description="Friction Reynolds number Re_τ")
    u_tau: float = Field(0.005, gt=0, description="Friction velocity (lattice units)")
    smagorinsky_cs: float = Field(0.1, gt=0, description="Smagorinsky constant C_s")
    n_steps: int = Field(50000, ge=1)
    averaging_start: int = Field(20000, ge=0)
    output_interval: int = Field(5000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None

class PipelineFlowParams(BaseModel):
    nx: int = Field(400, ge=20)
    ny: int = Field(160, ge=10)
    diameter: float = Field(20.0, gt=0, description="Cylinder diameter (cells)")
    gap_ratio: float = Field(0.5, ge=0, description="Gap e/D")
    u_in: float = Field(0.05, gt=0)
    re: float = Field(200.0, gt=0)
    n_steps: int = Field(30000, ge=1)
    output_interval: int = Field(5000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None

class DamBreakParams(BaseModel):
    nx: int = Field(400, ge=20)
    ny: int = Field(200, ge=10)
    dam_width: int = Field(100, ge=1)
    model: Literal["sc", "scmp", "cg", "fe"] = "cg"
    rho_heavy: float = Field(0.8, gt=0)
    rho_light: float = Field(0.4, gt=0)
    G: float = Field(0.9, description="Coupling constant")
    tau: float = Field(1.0, gt=0.5)
    g: float = Field(5e-5, gt=0, description="Gravity (lattice units)")
    n_steps: int = Field(4000, ge=1)
    output_interval: int = Field(400, ge=1)
    device: str = "cpu"
    physics: PhysicsSelection | None = None

class SloshingTankParams(BaseModel):
    nx: int = Field(200, ge=16)
    ny: int = Field(160, ge=16)
    water_level: int = Field(80, ge=1)
    rho_water: float = Field(0.8, gt=0)
    rho_air: float = Field(0.4, gt=0)
    G: float = Field(0.9, description="Color-gradient surface-tension coefficient")
    tau: float = Field(1.0, gt=0.5)
    g: float = Field(2e-5, gt=0)
    forcing_amp: float = Field(3e-5, ge=0)
    forcing_omega: float = Field(0.0, ge=0, description="0 = use natural frequency")
    n_steps: int = Field(6000, ge=1)
    output_interval: int = Field(600, ge=1)
    device: str = "cpu"
    physics: PhysicsSelection | None = None

class SphereFlowParams(BaseModel):
    nx: int = Field(120, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(60, ge=10)
    u_in: float = Field(0.06, gt=0)
    re: float = Field(50.0, gt=0)
    radius: float = Field(8.0, gt=0)
    n_steps: int = Field(500, ge=1)
    output_interval: int = Field(100, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None

class ShipHullFlowParams(BaseModel):
    nx: int = Field(160, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(40, ge=10)
    u_in: float = Field(0.05, gt=0)
    re: float = Field(200.0, gt=0)
    hull_length: float = Field(80.0, gt=0)
    hull_beam: float = Field(8.0, gt=0)
    hull_draft: float = Field(12.0, gt=0)
    smagorinsky_cs: float = Field(0.1, gt=0)
    wave_amp: float = Field(0.0, ge=0)
    wave_period: float = Field(200.0, gt=0)
    n_steps: int = Field(2000, ge=1)
    output_interval: int = Field(200, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None

class PorousDrainageParams(BaseModel):
    nx: int = Field(160, ge=20)
    ny: int = Field(80, ge=10)
    medium: Literal["random_cylinders", "tube_array"] = "random_cylinders"
    model: Literal["sc", "cg"] = "cg"
    porosity: float = Field(0.6, gt=0, lt=1)
    n_steps: int = Field(5000, ge=1)
    output_interval: int = Field(1000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None
