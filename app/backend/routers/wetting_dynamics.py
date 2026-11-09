"""Wetting dynamics API endpoint.

``POST /api/v1/calculations/wetting_dynamics`` runs the wetting-dynamics
analysis (MKT, Cassie-Baxter/Wenzel, capillary rise, droplet spreading,
Cox-Voinov) and returns a machine-readable report with all computed quantities,
evidence grades, and literature references.

This endpoint is synchronous and performs purely analytical calculations
(no LBM simulation).  For LBM-based wetting studies, use the
:mod:`tensorlbm.contact_line.engine` module directly.

References
----------
- de Gennes, P.-G. (1985). Wetting: statics and dynamics. Rev. Mod. Phys. 57, 827.
- Blake, T. D. (1993). Dynamic contact angles and wetting kinetics.
- Cassie, A. B. D. & Baxter, S. (1944). Trans. Faraday Soc. 40, 546.
- Wenzel, R. N. (1936). Ind. Eng. Chem. 28, 988.
- Jurin, J. (1718). Phil. Trans. 30, 739.
- Tanner, L. H. (1979). J. Phys. D 12, 1473.
- Cox, R. G. (1986). J. Fluid Mech. 168, 169.
- Voinov, O. V. (1976). Fluid Dyn. 11, 714.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tensorlbm.contact_line.wetting_dynamics import (
    GAMMA_WATER,
    G_SI,
    MU_WATER,
    RHO_WATER,
    T_ROOM,
    WettingDynamicsInput,
    run_wetting_dynamics,
)

router = APIRouter()


class WettingDynamicsRequest(BaseModel):
    """Request body for the wetting-dynamics analysis."""

    # Surface properties
    gamma: float = Field(
        GAMMA_WATER, gt=0, description="Liquid-vapour surface tension (N/m)."
    )
    theta_e_deg: float = Field(
        60.0, ge=0.0, le=180.0, description="Equilibrium (Young's) contact angle (degrees)."
    )
    mu: float = Field(MU_WATER, gt=0, description="Dynamic viscosity (Pa·s).")
    rho: float = Field(RHO_WATER, gt=0, description="Liquid density (kg/m³).")

    # Roughness / composite surface
    r: float | None = Field(
        None, ge=1.0, description="Roughness factor (Wenzel, r >= 1)."
    )
    f_s: float | None = Field(
        None, gt=0.0, le=1.0, description="Solid-liquid area fraction (Cassie-Baxter, 0 < f_s <= 1)."
    )

    # Capillary rise
    capillary_R: float | None = Field(
        None, gt=0, description="Capillary tube radius or plate separation (m)."
    )
    capillary_geometry: str = Field(
        "cylindrical", description="Capillary geometry: 'cylindrical' or 'planar'."
    )
    g: float = Field(G_SI, ge=0, description="Gravitational acceleration (m/s²).")

    # Droplet spreading
    R_0: float | None = Field(None, gt=0, description="Initial droplet radius (m).")
    theta_0_deg: float | None = Field(
        None, ge=0.0, le=180.0, description="Initial contact angle for spreading (degrees)."
    )
    spreading_time: float | None = Field(None, ge=0, description="Spreading time (s).")

    # MKT parameters
    mkt_kappa: float | None = Field(None, gt=0, description="Molecular jump frequency (Hz).")
    mkt_lambda: float | None = Field(None, gt=0, description="Molecular jump length (m).")
    mkt_n_sites: float | None = Field(None, gt=0, description="Adsorption site density (m⁻²).")
    mkt_T: float = Field(T_ROOM, gt=0, description="Temperature (K).")
    contact_line_velocity: float | None = Field(
        None, description="Contact-line velocity for MKT inversion (m/s)."
    )

    # Cox-Voinov parameters
    cox_voinov_Ca: list[float] | None = Field(
        None, description="Capillary numbers for Cox-Voinov evaluation."
    )
    cox_voinov_L: float | None = Field(
        None, gt=0, description="Macroscopic length scale (m)."
    )
    cox_voinov_lambda: float | None = Field(
        None, gt=0, description="Microscopic cutoff length (m)."
    )


class WettingDynamicsResponse(BaseModel):
    """Response body for the wetting-dynamics analysis."""

    status: str = "success"
    theta_e_deg: float
    theta_CB_deg: float | None
    theta_W_deg: float | None
    wetting_state: str | None
    capillary_h: float | None
    capillary_zero_g: bool
    spreading_alpha: float | None
    spreading_alpha_error_pct: float | None
    spreading_R_final: float | None
    mkt_theta_d_deg: float | None
    mkt_velocity: float | None
    cox_voinov_results: list[dict[str, float]] | None
    evidence_grade: str
    references: list[str]
    validation: dict[str, Any]


@router.post("/wetting_dynamics")
async def run_wetting_dynamics_endpoint(req: WettingDynamicsRequest) -> WettingDynamicsResponse:
    """Run the wetting-dynamics analysis.

    Computes all applicable wetting quantities from the input parameters:
    - Cassie-Baxter apparent contact angle (if f_s provided)
    - Wenzel apparent contact angle (if r provided)
    - Wetting state determination
    - Capillary rise height (Jurin's law, if capillary_R provided)
    - Droplet spreading exponent (if R_0, theta_0, spreading_time provided)
    - MKT dynamic angle (if contact_line_velocity provided)
    - Cox-Voinov dynamic angles (if cox_voinov_Ca provided)
    """
    try:
        inp = WettingDynamicsInput(
            gamma=req.gamma,
            theta_e_deg=req.theta_e_deg,
            mu=req.mu,
            rho=req.rho,
            r=req.r,
            f_s=req.f_s,
            capillary_R=req.capillary_R,
            capillary_geometry=req.capillary_geometry,
            g=req.g,
            R_0=req.R_0,
            theta_0_deg=req.theta_0_deg,
            spreading_time=req.spreading_time,
            mkt_kappa=req.mkt_kappa,
            mkt_lambda=req.mkt_lambda,
            mkt_n_sites=req.mkt_n_sites,
            mkt_T=req.mkt_T,
            contact_line_velocity=req.contact_line_velocity,
            cox_voinov_Ca=req.cox_voinov_Ca,
            cox_voinov_L=req.cox_voinov_L,
            cox_voinov_lambda=req.cox_voinov_lambda,
        )
        result = run_wetting_dynamics(inp)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return WettingDynamicsResponse(
        status="success",
        theta_e_deg=result.theta_e_deg,
        theta_CB_deg=result.theta_CB_deg,
        theta_W_deg=result.theta_W_deg,
        wetting_state=result.wetting_state,
        capillary_h=result.capillary_h,
        capillary_zero_g=result.capillary_zero_g,
        spreading_alpha=result.spreading_alpha,
        spreading_alpha_error_pct=result.spreading_alpha_error_pct,
        spreading_R_final=result.spreading_R_final,
        mkt_theta_d_deg=result.mkt_theta_d_deg,
        mkt_velocity=result.mkt_velocity,
        cox_voinov_results=result.cox_voinov_results,
        evidence_grade=result.evidence_grade,
        references=result.references,
        validation=result.validation,
    )
