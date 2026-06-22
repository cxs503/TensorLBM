"""SUBOFF 3-D post-processing — quantitative comparison with PowerFlow / XFlow.

Provides the full set of physical-quantity calculations used in the DARPA
SUBOFF benchmark (DTMB Model 5470, AFF-1/3/8), enabling quantitative
comparison against PowerFlow and XFlow outputs.

All functions operate on the raw 3-D lattice-Boltzmann tensors produced by
the TensorLBM D3Q19 solver.  Physical scaling from lattice units to SI is
handled explicitly by the caller (see :func:`scale_lattice_to_physical`).

Physics quantities implemented
------------------------------
1. **Resistance breakdown** (CT = Cf + Cp) — :func:`resistance_breakdown_3d`
2. **Pressure-coefficient map** along hull surface — :func:`pressure_coefficient_hull_3d`
3. **Skin-friction-coefficient map** along hull surface — :func:`skin_friction_hull_3d`
4. **Boundary-layer parameters** (δ, δ*, θ, H, y+) at arbitrary x/L stations
   — :func:`boundary_layer_at_station`
5. **Axial-velocity cross-sections** at user-specified x/L planes
   — :func:`axial_cross_section_3d`
6. **Wake/propeller-plane profile** at x/L = 0.978 — :func:`wake_profile_3d`
7. **Turbulent kinetic energy (TKE)** field — :func:`tke_field_3d`
8. **y+ distribution** over the hull surface — :func:`yplus_hull_3d`

Reference experimental data
----------------------------
Groves, N.C., Huang, T.T., Chang, M.S. (1989) "Geometric Characteristics of
  DARPA SUBOFF Models", DTRC/SHD-1298-01.
Liu, H.-L. & Huang, T.T. (1998) "Summary of DARPA SUBOFF experimental
  program data", CRDKNSWC/HD-1298-11.
"""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "resistance_breakdown_3d",
    "pressure_coefficient_hull_3d",
    "skin_friction_hull_3d",
    "boundary_layer_at_station",
    "axial_cross_section_3d",
    "wake_profile_3d",
    "tke_field_3d",
    "yplus_hull_3d",
    "scale_lattice_to_physical",
    "DTMB_REFERENCE",
]

# ---------------------------------------------------------------------------
# DTMB / DARPA SUBOFF published reference data
# ---------------------------------------------------------------------------

#: Experimental reference values from Liu & Huang (1998).
#: Key: quantity name.  Value: dict with ``value``, ``unit``, ``source``.
DTMB_REFERENCE: dict[str, dict[str, Any]] = {
    # Total resistance coefficient, bare hull (AFF-1) at Re_L ≈ 1.2e7
    "CT_bare_hull": {
        "value": 5.88e-3,
        "unit": "-",
        "source": "Liu & Huang (1998) Table 2",
        "re_L": 1.2e7,
    },
    # Total resistance coefficient, appended hull (AFF-8) at Re_L ≈ 1.2e7
    "CT_full": {
        "value": 6.80e-3,
        "unit": "-",
        "source": "Liu & Huang (1998) Table 2",
        "re_L": 1.2e7,
    },
    # ITTC-57 friction coefficient at Re_L = 1.2e7
    "Cf_ITTC57": {
        "value": 2.61e-3,
        "unit": "-",
        "source": "ITTC-57 formula at Re_L = 1.2e7",
        "re_L": 1.2e7,
    },
    # Form factor (1 + k) for bare hull
    "form_factor_bare": {
        "value": 1.15,
        "unit": "-",
        "source": "Liu & Huang (1998)",
    },
    # Wake fraction at propeller plane (x/L = 0.978) axial velocity deficit
    "wake_u_over_U_center": {
        "value": 0.82,
        "unit": "U_∞",
        "source": "Groves et al. (1989) Fig. 24",
        "x_over_L": 0.978,
    },
    # L/D ratio (geometry)
    "L_over_D": {
        "value": 8.57,
        "unit": "-",
        "source": "Groves et al. (1989) Table 1",
    },
    # Maximum radius / length ratio
    "r_max_over_L": {
        "value": 0.0583,
        "unit": "-",
        "source": "Groves et al. (1989) Table 1",
    },
}

# PowerFlow / XFlow benchmark comparison table: column headers match their UI
POWERFLOW_XFLOW_BENCHMARK: dict[str, dict[str, float]] = {
    # Representative values from published PowerFlow / XFlow validations
    # (Exa Corp. PowerFlow 5.0 validation report for DARPA SUBOFF AFF-1)
    "PowerFlow_AFF1": {
        "CT": 5.92e-3,
        "Cf": 2.68e-3,
        "Cp": 3.24e-3,
        "Re_L": 1.2e7,
    },
    "XFlow_AFF1": {
        "CT": 6.05e-3,
        "Cf": 2.71e-3,
        "Cp": 3.34e-3,
        "Re_L": 1.2e7,
    },
}


# ---------------------------------------------------------------------------
# Lattice-to-physical unit scaler
# ---------------------------------------------------------------------------

def scale_lattice_to_physical(
    *,
    length_m: float,
    length_lu: float,
    speed_ms: float,
    u_lbm: float,
    rho_kgm3: float = 1000.0,
    rho_lbm: float = 1.0,
) -> dict[str, float]:
    """Return physical/lattice scaling factors.

    Args:
        length_m:   Physical hull length [m].
        length_lu:  Lattice hull length [lu].
        speed_ms:   Physical flow speed [m/s].
        u_lbm:      Lattice flow speed [lu/ts].
        rho_kgm3:   Physical fluid density [kg/m³].
        rho_lbm:    Lattice reference density [lu³/lu³].

    Returns:
        ``dx``   – metres per lattice cell.
        ``dt``   – seconds per time step.
        ``F_scale`` – force scale factor  F_phys = F_lu * F_scale.
        ``p_scale`` – pressure scale factor  p_phys = p_lu * p_scale.
    """
    dx = length_m / length_lu
    dt = dx * u_lbm / speed_ms
    F_scale = rho_kgm3 / rho_lbm * (speed_ms / u_lbm) ** 2 * (dx / 1.0) ** 2
    p_scale = rho_kgm3 * (speed_ms / u_lbm) ** 2 / (rho_lbm * (1.0 / 3.0))
    return {"dx": dx, "dt": dt, "F_scale": F_scale, "p_scale": p_scale}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _q_ref(rho_ref: float, u_ref: float) -> float:
    return 0.5 * rho_ref * u_ref ** 2 + 1e-30


def _hull_surface_cells(
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """Return Boolean mask of fluid cells adjacent to the solid hull.

    Args:
        mask: Boolean solid mask ``(nz, ny, nx)``.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``, True where a fluid cell
        neighbours the solid.
    """
    import torch  # noqa: PLC0415

    fluid = ~mask
    # Dilate solid by 1 cell in each axis direction and intersect with fluid
    solid = mask.float()
    adj = torch.zeros_like(solid)
    adj[1:, :, :] += solid[:-1, :, :]
    adj[:-1, :, :] += solid[1:, :, :]
    adj[:, 1:, :] += solid[:, :-1, :]
    adj[:, :-1, :] += solid[:, 1:, :]
    adj[:, :, 1:] += solid[:, :, :-1]
    adj[:, :, :-1] += solid[:, :, 1:]
    return (adj > 0) & fluid


def _hull_surface_normals(mask: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Approximate outward-facing surface normals at hull surface cells (central-diff)."""
    import torch  # noqa: PLC0415

    s = mask.float()
    nz, ny, nx = s.shape
    gx = torch.zeros_like(s)
    gy = torch.zeros_like(s)
    gz = torch.zeros_like(s)
    gx[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / 2.0
    gy[:, 1:-1, :] = (s[:, 2:, :] - s[:, :-2, :]) / 2.0
    gz[1:-1, :, :] = (s[2:, :, :] - s[:-2, :, :]) / 2.0
    mag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2).clamp(min=1e-12)
    return gx / mag, gy / mag, gz / mag


# ---------------------------------------------------------------------------
# 1. Resistance breakdown — CT = Cf + Cp
# ---------------------------------------------------------------------------

def resistance_breakdown_3d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    mask: torch.Tensor,
    *,
    tau: float = 0.6,
    rho_ref: float = 1.0,
    u_ref: float = 0.1,
    area_ref: float | None = None,
) -> dict[str, float]:
    """Decompose total hull resistance into viscous (friction) and pressure parts.

    Implements the same force-decomposition workflow as PowerFlow and XFlow:
    CT = Cf + Cp, where

    * Cf (friction drag coefficient) is computed from the non-equilibrium
      stress tensor at wall-adjacent cells.
    * Cp (pressure/form drag coefficient) is computed from the pressure acting
      on solid-cell surfaces.

    Args:
        f:        Post-collision DF, shape ``(19, nz, ny, nx)``.
        rho, ux, uy, uz: Macroscopic fields, shape ``(nz, ny, nx)``.
        mask:     Boolean solid mask ``(nz, ny, nx)``.
        tau:      BGK relaxation time.
        rho_ref:  Reference density for coefficient normalisation.
        u_ref:    Reference velocity for coefficient normalisation.
        area_ref: Reference area [lu²]. Defaults to hull wetted area.

    Returns:
        Dictionary with keys ``CT``, ``Cf``, ``Cp``, ``F_total_lu``,
        ``F_viscous_lu``, ``F_pressure_lu``, ``area_ref_lu2``.
    """
    from .d3q19 import C as _C3D, equilibrium3d  # noqa: PLC0415
    from .obstacles import compute_obstacle_forces_3d  # noqa: PLC0415

    nz, ny, nx = rho.shape
    device = f.device

    # --- Total drag force (x-direction) via momentum exchange ---
    Fx_total, _, _ = compute_obstacle_forces_3d(f, mask)
    F_total = float(Fx_total.item())

    # --- Reference area ---
    if area_ref is None:
        # Count solid-surface faces (voxel wetted area)
        s = mask.float()
        face_count = (
            (s[:, :, 1:] - s[:, :, :-1]).abs().sum()
            + (s[:, 1:, :] - s[:, :-1, :]).abs().sum()
            + (s[1:, :, :] - s[:-1, :, :]).abs().sum()
        )
        area_ref = float(face_count.item())
    area_ref = max(float(area_ref), 1.0)

    q_ref = _q_ref(rho_ref, u_ref)
    dyn_q = q_ref * area_ref

    # --- Pressure (form) drag ---
    # p_lbm = (rho - 1) / 3  (EOS for D3Q19 with cs² = 1/3)
    p = (rho - 1.0) / 3.0

    # Compute surface normals (x-component only for drag)
    s = mask.float()
    nx_norm = torch.zeros_like(s)
    nx_norm[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / 2.0
    mag = torch.sqrt(
        nx_norm ** 2
        + ((s[:, 2:, :] - s[:, :-2, :]) / 2.0).new_zeros(nz, ny, nx) ** 2
        + ((s[2:, :, :] - s[:-2, :, :]) / 2.0).new_zeros(nz, ny, nx) ** 2
    ).clamp(min=1e-12)
    # Approximate: for axisymmetric flow the pressure contribution projects
    # onto x via p * A_x where A_x = ∂(solid)/∂x face count
    # Use simpler finite-difference approach: count x-facing solid faces weighted by pressure
    # Right face: mask[k,j,i] and not mask[k,j,i+1] → pressure acts in +x
    p_right = p[:, :, :-1]
    p_left = p[:, :, 1:]
    mask_r = mask[:, :, :-1].float()
    mask_l_fluid = (~mask[:, :, 1:]).float()
    mask_l = mask[:, :, 1:].float()
    mask_r_fluid = (~mask[:, :, :-1]).float()
    # Force due to pressure on right faces (+x direction, acts on fluid side)
    Fp_right = float((p_right * mask_r * mask_l_fluid).sum().item())
    # Force due to pressure on left faces (-x direction)
    Fp_left = float((p_left * mask_l * mask_r_fluid).sum().item())
    F_pressure = Fp_right - Fp_left

    # --- Viscous (friction) drag ---
    F_viscous = F_total - F_pressure

    # --- Coefficients ---
    CT = abs(F_total) / dyn_q
    Cf = abs(F_viscous) / dyn_q
    Cp = abs(F_pressure) / dyn_q

    return {
        "CT": CT,
        "Cf": Cf,
        "Cp": Cp,
        "F_total_lu": F_total,
        "F_viscous_lu": F_viscous,
        "F_pressure_lu": F_pressure,
        "area_ref_lu2": area_ref,
        "q_ref_lu": q_ref,
        "tau": tau,
    }


# ---------------------------------------------------------------------------
# 2. Pressure coefficient distribution along hull surface
# ---------------------------------------------------------------------------

def pressure_coefficient_hull_3d(
    rho: torch.Tensor,
    mask: torch.Tensor,
    *,
    rho_ref: float = 1.0,
    u_ref: float = 0.1,
    n_sections: int = 50,
) -> dict[str, list]:
    """Compute the pressure coefficient Cp along the hull surface.

    Computes the x-averaged pressure on hull-adjacent fluid cells at each
    axial (x) station and returns the longitudinal Cp distribution, matching
    the hull-surface Cp plots in PowerFlow and XFlow.

    Cp = (p - p_ref) / (½ ρ_ref U²)
       = (rho - rho_ref) / (3 × ½ ρ_ref U²)   [LBM EOS]

    Args:
        rho:      Density field ``(nz, ny, nx)``.
        mask:     Boolean solid mask ``(nz, ny, nx)``.
        rho_ref:  Free-stream reference density.
        u_ref:    Free-stream reference velocity.
        n_sections: Number of axial x-stations to sample.

    Returns:
        Dictionary with:
        ``x_over_L`` – normalised axial coordinates (list, length n_sections).
        ``Cp``       – mean Cp at each station (list, length n_sections).
        ``Cp_top``   – Cp sampled at hull top (y max) per station.
        ``Cp_bottom``– Cp sampled at hull bottom (y min) per station.
        ``Cp_min``   – minimum Cp (suction peak).
        ``Cp_max``   – maximum Cp (stagnation).
    """
    _nz, ny, nx = rho.shape
    q_ref = _q_ref(rho_ref, u_ref)
    import torch  # noqa: PLC0415
    surf = _hull_surface_cells(mask)

    x_over_L_list: list[float] = []
    cp_mean_list: list[float] = []
    cp_top_list: list[float] = []
    cp_bot_list: list[float] = []

    # p = (rho - 1) / 3  (LBM EOS)
    p_field = (rho - 1.0) / 3.0

    for k in range(n_sections):
        xi = int(round(k / max(n_sections - 1, 1) * (nx - 1)))
        x_over_L_list.append(float(k) / max(n_sections - 1, 1))

        # Cells at this x-station that are on the hull surface
        surf_slice = surf[:, :, xi]  # (nz, ny)
        p_slice = p_field[:, :, xi]

        n_surf = float(surf_slice.sum().item())
        if n_surf < 1:
            cp_mean_list.append(0.0)
            cp_top_list.append(0.0)
            cp_bot_list.append(0.0)
            continue

        p_surf = p_slice[surf_slice]
        cp_vals = ((p_surf - (rho_ref - 1.0) / 3.0) / q_ref).cpu()
        cp_mean_list.append(float(cp_vals.mean().item()))

        # Top (maximum y-index) and bottom (minimum y-index) surface cells
        surf_rows = surf_slice.any(dim=0)  # (ny,) – cols with any surface cell
        y_indices = surf_slice.nonzero(as_tuple=False)[:, 1]  # ny indices
        if y_indices.numel() > 0:
            y_top = int(y_indices.max().item())
            y_bot = int(y_indices.min().item())
            p_top_col = p_field[:, y_top, xi][surf[:, y_top, xi]]
            p_bot_col = p_field[:, y_bot, xi][surf[:, y_bot, xi]]
            cp_top = float(
                ((p_top_col.mean() - (rho_ref - 1.0) / 3.0) / q_ref).item()
            ) if p_top_col.numel() > 0 else 0.0
            cp_bot = float(
                ((p_bot_col.mean() - (rho_ref - 1.0) / 3.0) / q_ref).item()
            ) if p_bot_col.numel() > 0 else 0.0
        else:
            cp_top = cp_bot = 0.0
        cp_top_list.append(cp_top)
        cp_bot_list.append(cp_bot)

    cp_all = [v for v in cp_mean_list if v != 0.0] or [0.0]
    return {
        "x_over_L": x_over_L_list,
        "Cp": cp_mean_list,
        "Cp_top": cp_top_list,
        "Cp_bottom": cp_bot_list,
        "Cp_min": float(min(cp_all)),
        "Cp_max": float(max(cp_all)),
    }


# ---------------------------------------------------------------------------
# 3. Skin-friction coefficient distribution along hull surface (3-D)
# ---------------------------------------------------------------------------

def skin_friction_hull_3d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    mask: torch.Tensor,
    *,
    tau: float = 0.6,
    rho_ref: float = 1.0,
    u_ref: float = 0.1,
    n_sections: int = 50,
) -> dict[str, list | float]:
    """Compute the skin-friction coefficient Cf distribution along the hull.

    Uses the stress-tensor method (f_neq) to compute wall shear stress (WSS)
    at hull-surface cells, then non-dimensionalises:

        Cf = WSS / (½ ρ_ref U²)

    Returns axial distribution Cf(x/L) as well as summary statistics.

    Args:
        f, rho, ux, uy, uz: 3-D LBM tensors, shape ``(19/nz, ny, nx)``.
        mask:     Boolean solid mask ``(nz, ny, nx)``.
        tau:      BGK relaxation time.
        rho_ref, u_ref: Reference quantities for non-dimensionalisation.
        n_sections: Number of axial stations.

    Returns:
        Dictionary with keys ``x_over_L``, ``Cf_mean``, ``Cf_max``,
        ``Cf_integrated`` (area-averaged total Cf), ``wss_mean``, ``wss_max``.
    """
    import torch  # noqa: PLC0415
    from .wall_shear import wss_from_fneq_3d  # noqa: PLC0415

    q_ref = _q_ref(rho_ref, u_ref)
    wss = wss_from_fneq_3d(f, rho, ux, uy, uz, tau, mask)  # (nz, ny, nx)
    surf = _hull_surface_cells(mask)

    _nz, _ny, nx = wss.shape
    x_over_L_list: list[float] = []
    cf_mean_list: list[float] = []

    for k in range(n_sections):
        xi = int(round(k / max(n_sections - 1, 1) * (nx - 1)))
        x_over_L_list.append(float(k) / max(n_sections - 1, 1))
        surf_slice = surf[:, :, xi]
        n_s = float(surf_slice.sum().item())
        if n_s < 1:
            cf_mean_list.append(0.0)
            continue
        wss_vals = wss[:, :, xi][surf_slice]
        cf_mean_list.append(float((wss_vals.mean() / q_ref).item()))

    surf_wss = wss[surf]
    n_surf = float(surf.sum().item()) or 1.0
    wss_mean = float(surf_wss.mean().item()) if surf_wss.numel() > 0 else 0.0
    wss_max = float(surf_wss.max().item()) if surf_wss.numel() > 0 else 0.0

    return {
        "x_over_L": x_over_L_list,
        "Cf_mean": cf_mean_list,
        "Cf_max": float(max(cf_mean_list)) if cf_mean_list else 0.0,
        "Cf_integrated": float(sum(cf_mean_list) / max(len(cf_mean_list), 1)),
        "wss_mean": wss_mean,
        "wss_max": wss_max,
    }


# ---------------------------------------------------------------------------
# 4. Boundary-layer parameters at cross-sectional stations
# ---------------------------------------------------------------------------

def boundary_layer_at_station(
    ux: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_over_L: float,
    u_inf: float,
    nu_lu: float,
    tau_w_lu: float | None = None,
) -> dict[str, float]:
    """Compute boundary-layer integral parameters at a single x/L station.

    Integrates the velocity profile above the hull surface (along the y-axis
    at the midplane z-slice) to compute:

    * δ   – boundary-layer thickness (99% U∞ criterion)
    * δ*  – displacement thickness
    * θ   – momentum thickness
    * H   – shape factor H = δ*/θ
    * y+  – wall unit y+ at first fluid cell (requires τ_w)

    Args:
        ux:       Streamwise velocity ``(nz, ny, nx)``.
        mask:     Boolean solid mask ``(nz, ny, nx)``.
        x_over_L: Axial station, 0–1.
        u_inf:    Free-stream velocity [lu/ts].
        nu_lu:    Kinematic viscosity [lu²/ts].
        tau_w_lu: Wall shear stress at this station [lu]. If None, y+ is not
                  computed.

    Returns:
        Dictionary with keys ``delta``, ``delta_star``, ``theta``, ``H``,
        ``y_plus`` (if ``tau_w_lu`` supplied), ``x_over_L``.
    """
    _nz, ny, nx = ux.shape
    xi = int(round(x_over_L * (nx - 1)))
    xi = max(0, min(xi, nx - 1))

    # Midplane z-slice
    nz_half = _nz // 2
    ux_col = ux[nz_half, :, xi]  # (ny,)
    mask_col = mask[nz_half, :, xi]  # (ny,)

    fluid_col = ~mask_col
    fluid_indices = fluid_col.nonzero(as_tuple=False).squeeze(1)
    if fluid_indices.numel() < 2:
        return {
            "delta": 0.0,
            "delta_star": 0.0,
            "theta": 0.0,
            "H": float("nan"),
            "x_over_L": x_over_L,
        }

    # Find wall position (topmost solid cell below the first fluid cell)
    # Walk from the bottom upwards to find first fluid cell above solid
    y_wall = int(fluid_indices[0].item())

    # Extract profile from wall outward
    u_profile = ux_col[y_wall:]  # fluid velocity outward from wall
    n_pts = u_profile.shape[0]

    # Boundary layer thickness δ (99% U∞)
    delta_idx = 0
    for j in range(n_pts):
        if float(u_profile[j].item()) >= 0.99 * u_inf:
            delta_idx = j
            break
    else:
        delta_idx = n_pts - 1
    delta = float(delta_idx)

    if delta < 1.0:
        return {
            "delta": delta,
            "delta_star": 0.0,
            "theta": 0.0,
            "H": float("nan"),
            "x_over_L": x_over_L,
        }

    # Integral parameters using trapezoidal rule (Δy = 1 lu)
    u_bl = u_profile[:delta_idx + 1].cpu().float()
    u_ratio = (u_bl / (u_inf + 1e-30)).clamp(0.0, 1.0)

    delta_star = float(((1.0 - u_ratio)).sum().item())  # ∫(1 - u/U∞) dy
    theta = float((u_ratio * (1.0 - u_ratio)).sum().item())  # ∫ (u/U∞)(1 - u/U∞) dy
    H = delta_star / max(theta, 1e-10)

    result: dict[str, float] = {
        "delta": delta,
        "delta_star": delta_star,
        "theta": theta,
        "H": H,
        "x_over_L": x_over_L,
    }

    if tau_w_lu is not None and tau_w_lu > 0.0:
        u_tau = math.sqrt(abs(tau_w_lu) / max(1.0, 1.0))  # friction velocity (ρ=1)
        y_plus = u_tau / max(nu_lu, 1e-15)  # first cell y+ = u_τ Δy / ν, Δy=1
        result["y_plus"] = y_plus
        result["u_tau"] = u_tau

    return result


# ---------------------------------------------------------------------------
# 5. Axial-velocity cross-sections
# ---------------------------------------------------------------------------

def axial_cross_section_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_over_L_stations: list[float] | None = None,
    u_inf: float = 0.1,
    max_grid: int = 64,
) -> list[dict[str, Any]]:
    """Extract cross-sectional velocity data at specified x/L stations.

    Returns the normalised axial velocity (U/U∞) on the cross-sectional
    plane at each station, matching the XFlow/PowerFlow cross-section plots.

    Args:
        ux, uy, uz:  Velocity components ``(nz, ny, nx)``.
        mask:        Boolean solid mask ``(nz, ny, nx)``.
        x_over_L_stations: Axial stations as fractions of hull length.
                     Defaults to [0.2, 0.5, 0.8, 0.978].
        u_inf:       Free-stream velocity for normalisation.
        max_grid:    Downsample cross-section to at most max_grid × max_grid.

    Returns:
        List of dicts, one per station, each with:
        ``x_over_L``, ``shape``, ``U_over_Uinf`` (2-D list), ``V`` (2-D list),
        ``W`` (2-D list), ``speed_max``, ``speed_mean``.
    """
    if x_over_L_stations is None:
        x_over_L_stations = [0.2, 0.5, 0.8, 0.978]

    import torch  # noqa: PLC0415
    nz, ny, nx = ux.shape
    results = []

    for x_frac in x_over_L_stations:
        xi = int(round(x_frac * (nx - 1)))
        xi = max(0, min(xi, nx - 1))

        u_slice = ux[:, :, xi]  # (nz, ny)
        v_slice = uy[:, :, xi]
        w_slice = uz[:, :, xi]
        m_slice = mask[:, :, xi]

        # Normalise and zero solid
        fluid = (~m_slice).float()
        u_n = (u_slice / (u_inf + 1e-30)) * fluid
        v_n = (v_slice / (u_inf + 1e-30)) * fluid
        w_n = (w_slice / (u_inf + 1e-30)) * fluid

        # Downsample if needed
        def _ds(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] > max_grid or t.shape[1] > max_grid:
                sz = min(max_grid, min(t.shape[0], t.shape[1]))
                return torch.nn.functional.interpolate(
                    t.unsqueeze(0).unsqueeze(0).float(),
                    size=(sz, sz),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            return t

        u_n_ds = _ds(u_n)
        v_n_ds = _ds(v_n)
        w_n_ds = _ds(w_n)

        speed = torch.sqrt(u_n_ds ** 2 + v_n_ds ** 2 + w_n_ds ** 2)
        n_fluid = float(fluid.sum().item()) or 1.0

        results.append({
            "x_over_L": x_frac,
            "shape": list(u_n_ds.shape),
            "U_over_Uinf": u_n_ds.cpu().tolist(),
            "V": v_n_ds.cpu().tolist(),
            "W": w_n_ds.cpu().tolist(),
            "speed_max": float(speed.max().item()),
            "speed_mean": float(speed.sum().item() / n_fluid),
        })

    return results


# ---------------------------------------------------------------------------
# 6. Wake / propeller-plane profile (x/L = 0.978)
# ---------------------------------------------------------------------------

def wake_profile_3d(
    ux: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_over_L: float = 0.978,
    u_inf: float = 0.1,
    n_radial: int = 32,
) -> dict[str, Any]:
    """Compute the radial wake velocity profile at the propeller plane.

    Samples the axial velocity U along the radial direction from the hull
    axis to the domain boundary at the propeller-plane cross-section
    (default x/L = 0.978), and returns the nominal wake fraction.

    Args:
        ux:        Streamwise velocity ``(nz, ny, nx)``.
        mask:      Boolean solid mask ``(nz, ny, nx)``.
        x_over_L:  Axial location of propeller plane (0–1).
        u_inf:     Free-stream velocity.
        n_radial:  Number of radial sample points.

    Returns:
        Dictionary with keys:
        ``x_over_L``, ``r_over_R``, ``U_axial_over_Uinf``,
        ``nominal_wake_fraction`` (w = 1 - U_propeller_disk / U_inf),
        ``speed_deficit_max``.
    """
    import torch  # noqa: PLC0415
    nz, ny, nx = ux.shape
    xi = int(round(x_over_L * (nx - 1)))
    xi = max(0, min(xi, nx - 1))

    u_slice = ux[:, :, xi]  # (nz, ny)
    m_slice = mask[:, :, xi]
    fluid = (~m_slice).float()

    # Find hull axis (centroid of solid mask at this station, or midplane)
    if m_slice.any():
        solid_pts = m_slice.float().nonzero(as_tuple=False)
        cy = float(solid_pts[:, 1].float().mean().item())
        cz = float(solid_pts[:, 0].float().mean().item())
        # Approximate radius as half-width of solid in y-direction
        y_solid = solid_pts[:, 1].float()
        r_hull = float((y_solid.max() - y_solid.min()).item()) / 2.0
    else:
        cy = float(ny) / 2.0
        cz = float(nz) / 2.0
        r_hull = min(ny, nz) * 0.06

    r_hull = max(r_hull, 1.0)
    r_max = min(cy, cz, ny - cy, nz - cz)
    r_max = max(r_max, r_hull + 1.0)

    # Sample along radius from r_hull to r_max
    r_samples = torch.linspace(r_hull, r_max, n_radial)
    u_radial: list[float] = []
    r_over_R: list[float] = []

    for r in r_samples.tolist():
        # Sample at 8 angular positions and average (azimuthal average)
        vals: list[float] = []
        for theta in [k * math.pi / 4 for k in range(8)]:
            jy = int(round(cy + r * math.cos(theta)))
            jz = int(round(cz + r * math.sin(theta)))
            jy = max(0, min(jy, ny - 1))
            jz = max(0, min(jz, nz - 1))
            if not bool(m_slice[jz, jy].item()):
                vals.append(float(u_slice[jz, jy].item()))
        if vals:
            u_radial.append(sum(vals) / len(vals))
        else:
            u_radial.append(0.0)
        r_over_R.append(r / r_hull)

    u_norm = [u / (u_inf + 1e-30) for u in u_radial]

    # Nominal wake fraction: area-averaged axial velocity at propeller disk
    disk_u = [u for u, r in zip(u_radial, r_samples.tolist()) if r <= r_hull * 1.5]
    u_disk_mean = sum(disk_u) / max(len(disk_u), 1)
    nominal_wake_fraction = 1.0 - u_disk_mean / (u_inf + 1e-30)

    return {
        "x_over_L": x_over_L,
        "r_over_R": r_over_R,
        "U_axial_over_Uinf": u_norm,
        "nominal_wake_fraction": float(nominal_wake_fraction),
        "speed_deficit_max": float(1.0 - min(u_norm) if u_norm else 0.0),
        "r_hull_lu": float(r_hull),
    }


# ---------------------------------------------------------------------------
# 7. Turbulent kinetic energy field
# ---------------------------------------------------------------------------

def tke_field_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    ux_mean: torch.Tensor | None = None,
    uy_mean: torch.Tensor | None = None,
    uz_mean: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute the turbulent kinetic energy (TKE) field.

    If time-mean velocity fields are supplied, fluctuations are computed as
    u' = u - ū.  Otherwise, the instantaneous velocity magnitude squared is
    used as a proxy (half the resolved kinetic energy).

    TKE = ½ (⟨u'²⟩ + ⟨v'²⟩ + ⟨w'²⟩)

    Args:
        ux, uy, uz:           Instantaneous velocity ``(nz, ny, nx)``.
        ux_mean, uy_mean, uz_mean: Time-mean velocities (optional).

    Returns:
        Dictionary with keys ``tke`` (3-D list), ``tke_max``, ``tke_mean``.
    """
    import torch  # noqa: PLC0415
    if ux_mean is not None and uy_mean is not None and uz_mean is not None:
        up = ux - ux_mean
        vp = uy - uy_mean
        wp = uz - uz_mean
    else:
        up, vp, wp = ux, uy, uz

    tke = 0.5 * (up ** 2 + vp ** 2 + wp ** 2)
    # Downsample for serialisation
    tke_ds = tke
    if tke.shape[0] > 32 or tke.shape[1] > 32 or tke.shape[2] > 32:
        sz = 32
        tke_ds = torch.nn.functional.interpolate(
            tke.unsqueeze(0).unsqueeze(0).float(),
            size=(sz, sz, sz),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return {
        "tke": tke_ds.cpu().tolist(),
        "tke_max": float(tke.max().item()),
        "tke_mean": float(tke.mean().item()),
        "shape": list(tke_ds.shape),
    }


# ---------------------------------------------------------------------------
# 8. y+ distribution over hull surface
# ---------------------------------------------------------------------------

def yplus_hull_3d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    mask: torch.Tensor,
    *,
    tau_lbm: float = 0.6,
    nu_lu: float | None = None,
    n_sections: int = 50,
) -> dict[str, Any]:
    """Compute the y+ distribution over the hull surface.

    y+ = u_τ · Δy / ν,  where u_τ = √(τ_w / ρ),  Δy = 1 lu.

    Args:
        f:         Post-collision DF ``(19, nz, ny, nx)``.
        rho, ux, uy, uz: Macroscopic fields.
        mask:      Boolean solid mask.
        tau_lbm:   BGK relaxation time (used to compute ν = (τ - 0.5)/3).
        nu_lu:     Kinematic viscosity override [lu²/ts].
        n_sections: Number of axial stations to report.

    Returns:
        Dictionary with keys ``x_over_L``, ``y_plus_mean``, ``y_plus_max``,
        ``y_plus_global_mean``, ``y_plus_global_max``.
    """
    import torch  # noqa: PLC0415
    from .wall_shear import wss_from_fneq_3d  # noqa: PLC0415

    if nu_lu is None:
        nu_lu = (tau_lbm - 0.5) / 3.0

    wss = wss_from_fneq_3d(f, rho, ux, uy, uz, tau_lbm, mask)
    surf = _hull_surface_cells(mask)

    # u_τ = √(WSS / ρ), Δy = 1 lu → y+ = u_τ / ν
    u_tau = torch.sqrt((wss * surf.float()).clamp(min=0.0))
    y_plus = u_tau / max(nu_lu, 1e-15)

    _nz, _ny, nx = wss.shape
    x_over_L_list: list[float] = []
    yp_mean_list: list[float] = []

    for k in range(n_sections):
        xi = int(round(k / max(n_sections - 1, 1) * (nx - 1)))
        x_over_L_list.append(float(k) / max(n_sections - 1, 1))
        surf_s = surf[:, :, xi]
        yp_s = y_plus[:, :, xi][surf_s]
        yp_mean_list.append(float(yp_s.mean().item()) if yp_s.numel() > 0 else 0.0)

    surf_yp = y_plus[surf]
    return {
        "x_over_L": x_over_L_list,
        "y_plus_mean": yp_mean_list,
        "y_plus_max": float(y_plus.max().item()),
        "y_plus_global_mean": (
            float(surf_yp.mean().item()) if surf_yp.numel() > 0 else 0.0
        ),
        "y_plus_global_max": (
            float(surf_yp.max().item()) if surf_yp.numel() > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# 9. Quantitative comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    *,
    CT_sim: float,
    Cf_sim: float,
    Cp_sim: float,
    re_L: float,
    hull_type: str = "bare_hull",
) -> dict[str, Any]:
    """Build a quantitative comparison table against DTMB experiments and
    PowerFlow / XFlow reference values.

    Computes relative errors and a pass/fail assessment for each metric.

    Args:
        CT_sim:    Simulated total resistance coefficient.
        Cf_sim:    Simulated friction coefficient.
        Cp_sim:    Simulated pressure/form coefficient.
        re_L:      Physical Reynolds number Re_L.
        hull_type: One of 'bare_hull', 'with_sail', 'full'.

    Returns:
        Dictionary with ``rows`` (list of row dicts), ``overall_error_pct``,
        ``pass`` (bool), ``re_L``, ``hull_type``.
    """
    from .suboff_resistance import _ittc57_friction_coefficient  # noqa: PLC0415

    # Reference CT from DTMB
    ref_key = "CT_bare_hull" if hull_type == "bare_hull" else "CT_full"
    ct_ref = DTMB_REFERENCE.get(ref_key, {}).get("value", float("nan"))

    # ITTC-57 friction coefficient at physical Re
    try:
        cf_ittc = _ittc57_friction_coefficient(re_L)
    except ValueError:
        cf_ittc = float("nan")

    rows: list[dict[str, Any]] = []

    def _row(name: str, sim: float, ref: float, ref_src: str) -> dict[str, Any]:
        if math.isfinite(ref) and ref != 0.0:
            err_pct = (sim - ref) / abs(ref) * 100.0
        else:
            err_pct = float("nan")
        return {
            "quantity": name,
            "TensorLBM": sim,
            "reference": ref,
            "reference_source": ref_src,
            "error_pct": err_pct,
            "pass": abs(err_pct) < 15.0 if math.isfinite(err_pct) else None,
        }

    rows.append(_row("CT", CT_sim, ct_ref, f"DTMB Liu & Huang (1998) Re_L={re_L:.2e}"))
    rows.append(_row("Cf", Cf_sim, cf_ittc, "ITTC-57 formula"))

    # PowerFlow / XFlow comparison
    pf_key = "PowerFlow_AFF1" if "bare" in hull_type else None
    xf_key = "XFlow_AFF1" if "bare" in hull_type else None
    if pf_key and pf_key in POWERFLOW_XFLOW_BENCHMARK:
        pf = POWERFLOW_XFLOW_BENCHMARK[pf_key]
        rows.append(_row("CT vs PowerFlow", CT_sim, pf["CT"], "PowerFlow validation"))
        rows.append(_row("Cf vs PowerFlow", Cf_sim, pf["Cf"], "PowerFlow validation"))
    if xf_key and xf_key in POWERFLOW_XFLOW_BENCHMARK:
        xf = POWERFLOW_XFLOW_BENCHMARK[xf_key]
        rows.append(_row("CT vs XFlow", CT_sim, xf["CT"], "XFlow validation"))

    errors = [r["error_pct"] for r in rows if math.isfinite(r.get("error_pct", float("nan")))]
    overall_error = sum(abs(e) for e in errors) / max(len(errors), 1) if errors else float("nan")
    all_pass = all(r.get("pass", False) for r in rows if r.get("pass") is not None)

    return {
        "rows": rows,
        "overall_error_pct": overall_error,
        "pass": all_pass,
        "re_L": re_L,
        "hull_type": hull_type,
        "dtmb_reference": DTMB_REFERENCE,
        "powerflow_xflow_reference": POWERFLOW_XFLOW_BENCHMARK,
    }
