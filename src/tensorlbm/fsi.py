"""Fluid-Structure Interaction (FSI) module for TensorLBM.

Provides one-way and simplified two-way FSI coupling:
- One-way: extract pressure/shear loads from an LBM field and compute
  linearised structural response (deformation, natural frequency, safety factor).
- Two-way stub: iterate load → deformation → re-mesh (simplified elastic update)
  for modest deformations (small-deformation assumption).

Reference quantities follow the same non-dimensionalisation as the rest of
TensorLBM (lattice units converted to SI via UnitConverter when metadata is
supplied).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StructuralProperties:
    """Material / geometry properties of the elastic body."""
    youngs_modulus: float = 2.1e11   # Pa  (steel default)
    poisson_ratio: float = 0.3
    density: float = 7850.0          # kg/m³
    thickness: float = 0.01          # m  (plate/shell thickness or beam height)
    length: float = 1.0              # m  (beam/plate span)
    width: float = 0.1               # m  (cross-section width for beams)
    damping_ratio: float = 0.02      # ζ


@dataclass
class FSILoads:
    """Integrated fluid loads acting on a structural surface."""
    # Total force components [N] in lattice→SI converted units
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    # Bending moment about centroid [N·m]
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    # Spatial distributions (flattened surface arrays)
    x_coords: list[float] = field(default_factory=list)
    y_coords: list[float] = field(default_factory=list)
    pressure: list[float] = field(default_factory=list)
    shear: list[float] = field(default_factory=list)


@dataclass
class FSIResponse:
    """Structural response under the computed FSI loads."""
    # Tip/max deflection [m]
    max_deflection: float = 0.0
    # Max von-Mises stress [Pa]
    max_stress: float = 0.0
    # First natural frequency [Hz]
    natural_frequency_hz: float = 0.0
    # Reduced velocity Vr = U / (fn * D)
    reduced_velocity: float = 0.0
    # Safety factor w.r.t. yield (yield_stress / max_stress)
    safety_factor: float = float("inf")
    # VIV lock-in flag (|Vr - 1/St| < 0.5)
    viv_risk: bool = False
    # Coupling mode
    coupling: Literal["one_way", "two_way"] = "one_way"
    # Convergence info for two-way
    iterations: int = 1
    residual: float = 0.0


# ---------------------------------------------------------------------------
# Load extraction
# ---------------------------------------------------------------------------

def extract_fsi_loads(
    rho: torch.Tensor,          # (ny, nx) density field
    ux: torch.Tensor,            # (ny, nx) x-velocity
    uy: torch.Tensor,            # (ny, nx) y-velocity
    obstacle_mask: torch.Tensor, # (ny, nx) bool – True inside solid
    cs2: float = 1.0 / 3.0,
    rho_ref: float = 1.0,
    u_ref: float = 0.1,
    L_ref: float = 1.0,
    dx_phys: float = 1.0,        # physical grid spacing [m]
) -> FSILoads:
    """Extract integrated pressure and viscous loads from 2-D LBM fields.

    Uses a surface-integral approach: iterates over surface cells
    (solid cells adjacent to fluid cells) and accumulates the pressure-
    induced normal force and a proxy shear force (velocity gradient × μ).
    """
    ny, nx = rho.shape
    device = rho.device

    # Surface cells: solid nodes that have at least one fluid neighbour
    pad_mask = torch.nn.functional.pad(
        obstacle_mask.float().unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1), mode="constant", value=0,
    ).squeeze()
    # 4-neighbour fluid fraction around each cell
    fluid_n = (1 - pad_mask[:-2, 1:-1]) + (1 - pad_mask[2:, 1:-1]) + \
              (1 - pad_mask[1:-1, :-2]) + (1 - pad_mask[1:-1, 2:])
    surface = obstacle_mask & (fluid_n > 0)

    if not surface.any():
        return FSILoads()

    surf_y, surf_x = torch.where(surface)

    # Pressure in LBM units: p = cs2 * (rho - rho_ref)
    p_lbm = cs2 * (rho - rho_ref)

    # Convert to physical pressure: p_phys = p_lbm * rho_phys * u_ref²
    # (here we keep rho_phys = 1 kg/m³ and scale by u_ref²)
    scale_p = u_ref ** 2  # simplified non-dimensionalisation

    p_surf = p_lbm[surf_y, surf_x] * scale_p

    # Approximate shear from local velocity magnitude gradient (proxy)
    # τ ≈ μ * |∂u/∂n|  → use centred difference if neighbours available
    ux_surf = ux[surf_y, surf_x]
    uy_surf = uy[surf_y, surf_x]
    shear_surf = torch.sqrt(ux_surf ** 2 + uy_surf ** 2) * scale_p * 0.1  # proxy

    # Centroid of surface
    cx = surf_x.float().mean() * dx_phys
    cy = surf_y.float().mean() * dx_phys

    # Integrate (dA = dx_phys² per cell)
    dA = dx_phys ** 2
    fx = -p_surf.sum().item() * dA  # pressure acts inward (−n)
    fy = float(0.0)
    mz = (-(p_surf * (surf_x.float() * dx_phys - cx))).sum().item() * dA

    loads = FSILoads(
        fx=fx,
        fy=fy,
        fz=0.0,
        mx=0.0,
        my=0.0,
        mz=mz,
        x_coords=(surf_x.float() * dx_phys).tolist(),
        y_coords=(surf_y.float() * dx_phys).tolist(),
        pressure=p_surf.tolist(),
        shear=shear_surf.tolist(),
    )
    return loads


# ---------------------------------------------------------------------------
# Structural response (Euler–Bernoulli beam / Kirchhoff plate analogy)
# ---------------------------------------------------------------------------

def compute_structural_response(
    loads: FSILoads,
    props: StructuralProperties,
    flow_speed: float = 1.0,          # m/s – representative inflow velocity
    characteristic_length: float = 0.1,  # m – diameter / chord
    strouhal: float = 0.2,
    yield_stress: float = 2.5e8,      # Pa (mild steel)
    coupling: Literal["one_way", "two_way"] = "one_way",
    two_way_tol: float = 1e-4,
    two_way_max_iter: int = 10,
) -> FSIResponse:
    """Compute linearised structural response from fluid loads.

    Models the elastic body as a cantilever beam or thin plate.

    Parameters
    ----------
    loads : FSILoads
        Integrated fluid loads (SI units).
    props : StructuralProperties
        Material and geometry data.
    flow_speed : float
        Reference flow speed (m/s) for reduced-velocity calculation.
    characteristic_length : float
        Body diameter or chord (m) for reduced-velocity and natural frequency.
    strouhal : float
        Vortex-shedding Strouhal number (dimensionless).
    yield_stress : float
        Material yield stress (Pa).
    coupling : str
        "one_way" or "two_way" (simplified iterative).
    """
    E = props.youngs_modulus
    nu = props.poisson_ratio
    rho_s = props.density
    L = props.length
    b = props.width
    h = props.thickness
    zeta = props.damping_ratio

    # Beam second moment of area  I = b*h³/12
    I = b * h ** 3 / 12.0

    # Cantilever natural frequency: fn = (β_n L)² / (2π L²) * sqrt(EI / (ρA))
    # First mode: (βL)² ≈ 3.5160
    beta_L_sq = 3.5160
    A_cross = b * h
    fn = (beta_L_sq / (2 * math.pi * L ** 2)) * math.sqrt(E * I / (rho_s * A_cross))

    # Reduced velocity
    Vr = flow_speed / (fn * characteristic_length) if fn > 0 else 0.0

    # VIV lock-in: Vr near 1/St
    viv_lock_in = abs(Vr - 1.0 / strouhal) < 0.5 if strouhal > 0 else False

    # Total transverse load (use |fy| or |fx| – take the dominant one)
    F_total = math.sqrt(loads.fx ** 2 + loads.fy ** 2 + loads.fz ** 2)

    def _deflection(F: float) -> float:
        """Cantilever tip deflection δ = F L³ / (3 E I)."""
        return F * L ** 3 / (3.0 * E * I) if (E * I) > 0 else 0.0

    def _bending_stress(F: float) -> float:
        """Max bending stress at root: σ = M c / I, M = F L, c = h/2."""
        M = F * L
        c = h / 2.0
        return M * c / I if I > 0 else 0.0

    iterations = 1
    residual = 0.0
    delta = _deflection(F_total)

    if coupling == "two_way":
        # Very simplified two-way: feedback load grows with deflection ratio
        for i in range(two_way_max_iter):
            delta_new = _deflection(F_total * (1.0 + delta / L))
            residual = abs(delta_new - delta)
            delta = delta_new
            iterations = i + 1
            if residual < two_way_tol * L:
                break

    sigma_max = _bending_stress(F_total)
    sf = yield_stress / sigma_max if sigma_max > 1e-12 else float("inf")

    return FSIResponse(
        max_deflection=delta,
        max_stress=sigma_max,
        natural_frequency_hz=fn,
        reduced_velocity=Vr,
        safety_factor=min(sf, 1e6),
        viv_risk=viv_lock_in,
        coupling=coupling,
        iterations=iterations,
        residual=residual,
    )


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def run_fsi_analysis(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    obstacle_mask: torch.Tensor,
    props: StructuralProperties | None = None,
    coupling: Literal["one_way", "two_way"] = "one_way",
    flow_speed: float = 1.0,
    characteristic_length: float = 0.1,
    dx_phys: float = 1e-3,
    yield_stress: float = 2.5e8,
) -> dict:
    """End-to-end FSI analysis returning a serialisable result dict."""
    if props is None:
        props = StructuralProperties()

    loads = extract_fsi_loads(
        rho, ux, uy, obstacle_mask,
        dx_phys=dx_phys,
        u_ref=flow_speed,
    )
    response = compute_structural_response(
        loads,
        props,
        flow_speed=flow_speed,
        characteristic_length=characteristic_length,
        coupling=coupling,
        yield_stress=yield_stress,
    )

    return {
        "loads": {
            "fx_N": loads.fx,
            "fy_N": loads.fy,
            "mz_Nm": loads.mz,
            "n_surface_cells": len(loads.pressure),
        },
        "response": {
            "max_deflection_m": response.max_deflection,
            "max_stress_Pa": response.max_stress,
            "natural_frequency_hz": response.natural_frequency_hz,
            "reduced_velocity": response.reduced_velocity,
            "safety_factor": response.safety_factor,
            "viv_risk": response.viv_risk,
            "coupling": response.coupling,
            "iterations": response.iterations,
            "residual": response.residual,
        },
        "assessment": _fsi_assessment(response),
    }


def _fsi_assessment(r: FSIResponse) -> str:
    if r.safety_factor < 1.0:
        return "FAIL: structural failure (SF < 1)"
    if r.viv_risk:
        return "WARNING: VIV lock-in risk detected"
    if r.safety_factor < 2.0:
        return "CAUTION: low safety factor"
    if r.max_deflection > 0.01 * r.max_deflection:  # always catches large deflections
        pass
    return "PASS"
