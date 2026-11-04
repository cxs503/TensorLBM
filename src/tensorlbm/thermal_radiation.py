"""Thermal radiation module for TensorLBM.

Implements grey-body / solar radiation heat transfer as a source term that
couples with the existing thermal LBM (DDF) solver.  Radiation is treated via
the net-radiation method (radiosity) for grey, diffuse enclosures and via a
direct solar flux model for external-aerodynamics / automotive use cases.

Engineering use cases (matches PowerFlow / XFlow radiation)
------------------------------------------------------------
* **Solar load analysis** – impose a directional solar irradiance on exposed
  surfaces; useful for automotive cabin temperature, building façade, and
  photovoltaic thermal management.
* **Electronics / battery cooling** – internal heat generation (volumetric) +
  radiation from hot components to cool walls.
* **Industrial furnace / combustion enclosure** – full grey-body enclosure
  with view-factor radiosity.

Physical model
--------------
The radiation heat flux q_rad on a surface element is computed from the
net-radiation / radiosity method:

    J_i = ε_i σ T_i^4 + (1 − ε_i) G_i          (radiosity)
    G_i = Σ_j F_ij J_j  +  q_solar,i            (irradiation)
    q_rad,i = (J_i − G_i)                        (net flux)

where:
  ε_i   – surface emissivity (0–1)
  σ     – Stefan–Boltzmann constant (5.670 × 10⁻⁸ W m⁻² K⁻⁴)
  F_ij  – view factor from surface i to surface j
  J_i   – radiosity (total emitted + reflected)
  G_i   – irradiation incident on surface i

For external flows the simplified **solar beam model** is also provided:
  q_solar = α_solar × I_solar × cos(θ)

where α_solar is the solar absorptance and θ is the angle between the surface
normal and the solar direction vector.

The radiation source term is then added to the LBM temperature equation as a
Neumann flux condition on the solid–fluid boundary cells.

References
----------
Modest, M. F. (2013). *Radiative Heat Transfer* (3rd ed.). Elsevier.
Incropera, F. P. et al. (2007). *Fundamentals of Heat and Mass Transfer* (6th
ed.). Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = [
    "STEFAN_BOLTZMANN",
    "SurfaceRadiationProps",
    "SolarSettings",
    "RadiationEnclosureConfig",
    "solar_flux_on_surface",
    "compute_net_radiation_flux",
    "radiosity_matrix_solve",
    "validate_view_factors",
    "radiative_energy_residual",
    "apply_radiation_source",
    "run_radiation_step",
    "RadiationResult",
]

# Stefan–Boltzmann constant [W m⁻² K⁻⁴]
STEFAN_BOLTZMANN: float = 5.670374419e-8


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SurfaceRadiationProps:
    """Radiative properties of a bounding surface."""

    emissivity: float = 0.9  # grey-body emissivity (0–1)
    solar_absorptance: float = 0.5  # fraction of solar irradiance absorbed
    temperature: float = 300.0  # surface temperature [K]
    area: float = 1.0  # representative surface area [m²]


@dataclass
class SolarSettings:
    """Direct solar irradiance specification."""

    enabled: bool = False
    irradiance: float = 1000.0  # W/m²  (air-mass-1 ≈ 1353 W/m²)
    # Unit direction vector pointing FROM the sun TOWARD the scene
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)

    def direction_tensor(self) -> torch.Tensor:
        d = torch.tensor(self.direction, dtype=torch.float64)
        norm = d.norm()
        return d / norm if norm > 1e-12 else d


@dataclass
class RadiationEnclosureConfig:
    """Configuration for a radiating enclosure."""

    surfaces: list[SurfaceRadiationProps] = field(default_factory=list)
    solar: SolarSettings = field(default_factory=SolarSettings)
    # View-factor matrix F[i, j]; if None, equal view factors are assumed
    view_factors: list[list[float]] | None = None
    # Physical reference temperature [K] for lattice→SI conversion
    T_ref: float = 300.0
    # Number of fixed-point iterations for the fallback radiosity solve
    # (the primary path is an exact linear solve; see radiosity_matrix_solve)
    max_iter: int = 50
    tol: float = 1e-6
    # Relative tolerance for view-factor closure / reciprocity and for the
    # enclosure radiative energy balance (see validate_view_factors)
    view_factor_tol: float = 1e-6


@dataclass
class RadiationResult:
    """Output of a radiation step."""

    net_flux: list[float]  # W/m² net radiation flux per surface
    radiosity: list[float]  # W/m² total radiosity per surface
    irradiation: list[float]  # W/m² total irradiation per surface
    solar_flux: list[float]  # W/m² solar contribution per surface
    total_emitted_power: float  # W  total power emitted by all surfaces
    total_absorbed_power: float  # W  total power absorbed


# ---------------------------------------------------------------------------
# Solar flux model
# ---------------------------------------------------------------------------


def solar_flux_on_surface(
    surface_normals: torch.Tensor,
    solar: SolarSettings,
    absorptance: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute direct solar flux on each surface element.

    Parameters
    ----------
    surface_normals:
        Tensor of shape (N, 3) with outward unit normals for each surface patch.
    solar:
        Solar settings (irradiance + direction).
    absorptance:
        Scalar or tensor of shape (N,) with solar absorptance per surface.

    Returns
    -------
    Tensor of shape (N,) with absorbed solar flux [W/m²] per surface.
    """
    # Sun direction pointing FROM surface TOWARD sun (negative of direction)
    sun_vec = -solar.direction_tensor().to(surface_normals.dtype)

    # cos(θ) = n̂ · ŝ, clamped to [0, 1] (back-face shadowed)
    cos_theta = torch.clamp((surface_normals @ sun_vec), min=0.0, max=1.0)

    if isinstance(absorptance, torch.Tensor):
        alpha = absorptance.to(surface_normals.dtype)
    else:
        alpha = torch.tensor(absorptance, dtype=surface_normals.dtype)

    return alpha * solar.irradiance * cos_theta


# ---------------------------------------------------------------------------
# View-factor utilities
# ---------------------------------------------------------------------------


def _build_view_factor_matrix(n: int, vf_list: list[list[float]] | None) -> torch.Tensor:
    """Return the (n, n) view-factor matrix F.

    If *vf_list* is None, equal view factors (F_ij = 1/(n-1) for i≠j, 0 for
    i=j) are assumed (diffuse enclosure approximation).
    """
    if vf_list is not None:
        F = torch.tensor(vf_list, dtype=torch.float64)
        if F.shape != (n, n):
            raise ValueError(f"view_factors must be ({n}×{n}), got {F.shape}")
        return F

    # Default: equal view factors for a diffuse enclosure
    F = torch.zeros(n, n, dtype=torch.float64)
    if n > 1:
        off_diag = 1.0 / (n - 1)
        for i in range(n):
            for j in range(n):
                if i != j:
                    F[i, j] = off_diag
    return F


def validate_view_factors(
    view_factors: torch.Tensor,
    areas: Sequence[float],
    tol: float = 1e-6,
) -> dict:
    """Fail-closed validation of a view-factor matrix against enclosure physics.

    Two statements are checked, both necessary for a physically meaningful
    grey-diffuse enclosure:

    * **reciprocity** ``A_i F_ij = A_j F_ji`` (the energy exchanged by a surface
      pair must be the same seen from either side);
    * **closure** ``Σ_j F_ij = 1`` (all radiation leaving ``i`` lands somewhere
      in the enclosure).  A row sum below one is a legitimate *open* enclosure
      (the deficit leaves to an unmodelled environment) and is reported, not
      rejected.

    Non-negativity is required unconditionally.  Violations raise
    ``ValueError`` instead of silently returning a non-conservative flux.

    Returns
    -------
    dict with ``closed`` (bool), ``max_row_sum_dev`` and
    ``max_reciprocity_dev`` (relative deviations).
    """
    F = view_factors
    n = F.shape[0]
    A = torch.tensor(list(areas), dtype=torch.float64)
    if A.shape[0] != n:
        raise ValueError(f"areas must have {n} entries, got {A.shape[0]}")
    if bool((F < 0).any()):
        raise ValueError("view factors must be non-negative")

    row_sums = F.sum(dim=1)
    row_sum_dev = float((row_sums - 1.0).abs().max().item()) if n else 0.0
    closed = bool(n > 0 and row_sum_dev <= tol)

    exchange = A.unsqueeze(1) * F                      # S_ij = A_i F_ij
    scale = float(exchange.abs().max().item()) or 1.0
    reciprocity_dev = float((exchange - exchange.T).abs().max().item()) / scale
    if reciprocity_dev > tol:
        raise ValueError(
            "view factors violate reciprocity A_i*F_ij = A_j*F_ji "
            f"(max relative deviation {reciprocity_dev:.3e} > tol {tol:.1e}); "
            "the equal-view-factor default is only consistent for equal areas, "
            "so pass an explicit reciprocity-consistent view_factors matrix. "
            f"areas={A.tolist()}, F={F.tolist()}"
        )

    return {
        "closed": closed,
        "max_row_sum_dev": row_sum_dev,
        "max_reciprocity_dev": reciprocity_dev,
    }


def radiative_energy_residual(net_flux: Sequence[float], areas: Sequence[float]) -> float:
    """Return ``Σ_i q_i A_i`` [W] for an enclosure.

    This is the net radiative power leaving all surfaces.  For a closed,
    reciprocity-consistent enclosure without solar input it must be zero; with
    solar input it must equal ``-Σ_i A_i q_solar,i`` (the absorbed solar power).
    """
    return float(sum(q * a for q, a in zip(net_flux, areas)))


# ---------------------------------------------------------------------------
# Radiosity matrix solve
# ---------------------------------------------------------------------------


def radiosity_matrix_solve(
    surfaces: Sequence[SurfaceRadiationProps],
    view_factors: list[list[float]] | None = None,
    solar_flux_per_surface: Sequence[float] | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[list[float], list[float], list[float]]:
    """Solve the net-radiation / radiosity system for a grey diffuse enclosure.

    Returns (radiosity J, irradiation G, net_flux q) for each surface.
    Net flux is defined as: q_i = J_i − G_i  [W/m²].
    """
    n = len(surfaces)
    sigma = STEFAN_BOLTZMANN

    T = torch.tensor([s.temperature for s in surfaces], dtype=torch.float64)
    eps = torch.tensor([s.emissivity for s in surfaces], dtype=torch.float64)
    F = _build_view_factor_matrix(n, view_factors)

    # Initial radiosity: blackbody emission
    J = eps * sigma * T**4

    # Solar contribution per surface
    if solar_flux_per_surface is not None:
        q_solar = torch.tensor(solar_flux_per_surface, dtype=torch.float64)
    else:
        q_solar = torch.zeros(n, dtype=torch.float64)

    # Radiosity system (Modest 2013 §5.4 / net-radiation method):
    #   J = ε σ T⁴ + (1 − ε)(F J + q_solar)
    #   (I − diag(1 − ε) F) J = ε σ T⁴ + (1 − ε) q_solar
    # Solved exactly.  The previous undamped fixed-point iteration only
    # converged for spectral radius (1 − ε)·‖F‖ < 1 and, with the old default
    # max_iter=50 / tol=1e-6, under-resolved low-emissivity enclosures by up to
    # +7.3 % (ε = 0.05 two-plate case; see
    # tests/test_thermal_radiation_analytic.py and
    # docs/thermal_radiation_radiosity_verification_r1.md).
    system = torch.eye(n, dtype=torch.float64) - (1.0 - eps).unsqueeze(1) * F
    rhs = eps * sigma * T**4 + (1.0 - eps) * q_solar
    solved = True
    try:
        J = torch.linalg.solve(system, rhs)
    except RuntimeError:  # singular system, e.g. every surface with ε = 0
        solved = False
        J = eps * sigma * T**4

    if not solved:
        # Fallback undamped fixed-point iteration, kept only for the singular
        # case above where the linear system has no unique solution.
        for _ in range(max_iter):
            # Irradiation G_i = Σ_j F_ij J_j + q_solar_i
            G = F @ J + q_solar
            J_new = eps * sigma * T**4 + (1.0 - eps) * G
            if (J_new - J).abs().max().item() < tol:
                J = J_new
                break
            J = J_new

    G = F @ J + q_solar
    q_net = J - G  # positive = net outgoing flux [W/m²]

    return J.tolist(), G.tolist(), q_net.tolist()


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def compute_net_radiation_flux(
    cfg: RadiationEnclosureConfig,
    surface_normals: torch.Tensor | None = None,
) -> RadiationResult:
    """Compute net radiation fluxes for all surfaces in the enclosure.

    Parameters
    ----------
    cfg:
        Enclosure configuration.
    surface_normals:
        Optional (N, 3) tensor of outward normals, used if ``cfg.solar.enabled``
        is True to compute the directional solar contribution per surface.

    Returns
    -------
    RadiationResult
    """
    n = len(cfg.surfaces)
    sigma = STEFAN_BOLTZMANN

    # --- Solar flux per surface ---
    solar_per_surface: list[float] = [0.0] * n
    if cfg.solar.enabled:
        if surface_normals is not None:
            q_sol = solar_flux_on_surface(
                surface_normals,
                cfg.solar,
                absorptance=torch.tensor(
                    [s.solar_absorptance for s in cfg.surfaces],
                    dtype=surface_normals.dtype,
                ),
            )
            solar_per_surface = q_sol.tolist()
        else:
            # Fallback: mean normal facing sun at 45°, all surfaces equal
            cos45 = math.cos(math.radians(45))
            solar_per_surface = [
                s.solar_absorptance * cfg.solar.irradiance * cos45 for s in cfg.surfaces
            ]

    areas = [s.area for s in cfg.surfaces]
    vf_tensor = _build_view_factor_matrix(n, cfg.view_factors)
    vf_diag = validate_view_factors(vf_tensor, areas, tol=cfg.view_factor_tol)

    J, G, q_net = radiosity_matrix_solve(
        cfg.surfaces,
        view_factors=cfg.view_factors,
        solar_flux_per_surface=solar_per_surface,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )

    # Energy balance of a closed enclosure: Σ_i q_i A_i = −Σ_i A_i q_solar,i
    # (net power leaving the surfaces equals the absorbed solar input).
    if vf_diag["closed"]:
        residual = radiative_energy_residual(q_net, areas)
        solar_power = float(sum(a * q for a, q in zip(areas, solar_per_surface)))
        scale = float(sum(abs(q) * a for q, a in zip(q_net, areas))) or 1.0
        tol_power = cfg.view_factor_tol * scale
        if abs(residual + solar_power) > tol_power:
            raise ValueError(
                "radiative energy balance violated: sum(q_i*A_i) = "
                f"{residual:.6e} W vs absorbed solar power {solar_power:.6e} W "
                f"(|imbalance| = {abs(residual + solar_power):.3e} W "
                f"> {tol_power:.3e} W tolerance)"
            )

    T_list = [s.temperature for s in cfg.surfaces]
    A_list = [s.area for s in cfg.surfaces]

    total_emitted = sum(sigma * (T_list[i] ** 4) * A_list[i] for i in range(n))
    total_absorbed = sum(max(0.0, -q_net[i]) * A_list[i] for i in range(n))

    return RadiationResult(
        net_flux=q_net,
        radiosity=J,
        irradiation=G,
        solar_flux=solar_per_surface,
        total_emitted_power=total_emitted,
        total_absorbed_power=total_absorbed,
    )


def apply_radiation_source(
    T: torch.Tensor,
    solid_mask: torch.Tensor,
    net_flux_W_per_m2: float,
    dx_phys: float,
    rho_phys: float,
    cp_phys: float,
    dt_phys: float,
) -> torch.Tensor:
    """Add radiation source term to the temperature field on boundary cells.

    Converts the surface radiation flux [W/m²] to a volumetric source
    [K/step] that can be added directly to the temperature field::

        ΔT = q_rad [W/m²] × dt / (ρ cp dx)

    Parameters
    ----------
    T:
        Temperature field tensor (ny, nx) in physical units [K].
    solid_mask:
        Boolean tensor (ny, nx); True = solid cell.
    net_flux_W_per_m2:
        Net absorbed radiation flux [W/m²] (positive = heating).
    dx_phys, rho_phys, cp_phys, dt_phys:
        Physical length [m], density [kg/m³], heat capacity [J/(kg K)],
        and time step [s].
    """
    # Boundary cells = fluid cells adjacent to at least one solid cell
    padded = solid_mask.float()
    is_boundary = (
        ~solid_mask
        & (
            (torch.roll(padded, 1, 0) > 0)
            | (torch.roll(padded, -1, 0) > 0)
            | (torch.roll(padded, 1, 1) > 0)
            | (torch.roll(padded, -1, 1) > 0)
        ).bool()
    )

    dT = net_flux_W_per_m2 * dt_phys / (rho_phys * cp_phys * dx_phys)
    T_new = T.clone()
    T_new[is_boundary] = T_new[is_boundary] + dT
    return T_new


def run_radiation_step(
    cfg: RadiationEnclosureConfig,
    surface_normals: torch.Tensor | None = None,
) -> dict:
    """Convenience wrapper: compute radiation and return a JSON-serialisable dict."""
    result = compute_net_radiation_flux(cfg, surface_normals=surface_normals)
    return {
        "n_surfaces": len(cfg.surfaces),
        "solar_enabled": cfg.solar.enabled,
        "solar_irradiance_W_m2": cfg.solar.irradiance if cfg.solar.enabled else 0.0,
        "surfaces": [
            {
                "index": i,
                "temperature_K": cfg.surfaces[i].temperature,
                "emissivity": cfg.surfaces[i].emissivity,
                "net_flux_W_m2": result.net_flux[i],
                "radiosity_W_m2": result.radiosity[i],
                "irradiation_W_m2": result.irradiation[i],
                "solar_flux_W_m2": result.solar_flux[i],
            }
            for i in range(len(cfg.surfaces))
        ],
        "total_emitted_power_W": result.total_emitted_power,
        "total_absorbed_power_W": result.total_absorbed_power,
    }
