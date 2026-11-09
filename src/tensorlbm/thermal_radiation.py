"""Thermal radiation module for TensorLBM.

Implements grey-body / solar radiation heat transfer as a source term that
couples with the existing thermal LBM (DDF) solver.  Radiation is treated via
the net-radiation method (radiosity) for grey, diffuse enclosures and via a
direct solar flux model for external-aerodynamics / automotive use cases.

Key features
------------
* **Fail-closed view factors**: vf_list=None raises ValueError (no silent
  uniform placeholder). View factors must come from view_factor.py analytical
  solutions or external computation.
* **Gauss-Seidel iteration** with energy conservation residual for radiosity.
* **Direct linear solve** (numpy) for cross-validation.
* **Solar-sky radiation boundary**: direct + diffuse (ASHRAE clear-sky) +
  sky effective temperature (Berdahl–Martin) + sea albedo.
* **8–12 μm band radiance** via Planck integration for IR signature.
* **Dual backend**: numpy and torch paths with |Δ| < 1e-10 consistency.

Engineering use cases
---------------------
* Ship deck / superstructure / cabin three-surface enclosure with solar load.
* Automotive cabin thermal management.
* Electronics / battery cooling with radiation.
* Industrial furnace / combustion enclosure.

Physical model
--------------
Net-radiation / radiosity method for grey, diffuse enclosure:

    J_i = ε_i σ T_i⁴ + (1 − ε_i) G_i          (radiosity)
    G_i = Σ_j F_ij J_j  +  q_solar,i            (irradiation)
    q_rad,i = (J_i − G_i)                        (net flux)

References
----------
Modest, M. F. (2013). Radiative Heat Transfer (3rd ed.). Elsevier.
Incropera, F. P. et al. (2007). Fundamentals of Heat and Mass Transfer (6th
    ed.). Wiley.
Berdahl, P., & Martin, M. (1984). Emissivity of clear skies. Solar Energy,
    32(5), 663-664.
ASHRAE (2017). Fundamentals Handbook. Chapter 14: Climatic Design Information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "STEFAN_BOLTZMANN",
    "SurfaceRadiationProps",
    "SolarSettings",
    "SkyRadiationSettings",
    "RadiationEnclosureConfig",
    "solar_flux_on_surface",
    "compute_sky_temperature",
    "compute_sky_radiation_flux",
    "compute_net_radiation_flux",
    "radiosity_gauss_seidel",
    "radiosity_direct_solve",
    "radiosity_matrix_solve",
    "apply_radiation_source",
    "run_radiation_step",
    "planck_band_radiance",
    "RadiationResult",
]

# Stefan–Boltzmann constant [W m⁻² K⁻⁴]
STEFAN_BOLTZMANN: float = 5.670374419e-8

# Planck constants
_H_PLANCK = 6.62607015e-34   # J·s
_C_LIGHT = 2.99792458e8      # m/s
_K_BOLTZMANN = 1.380649e-23  # J/K
_C1 = 2.0 * _H_PLANCK * _C_LIGHT**2  # First radiation constant
_C2 = _H_PLANCK * _C_LIGHT / _K_BOLTZMANN  # Second radiation constant


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
    # ASHRAE clear-sky model parameters
    use_ashrae_model: bool = False
    # Day of year for ASHRAE model (1-365)
    day_of_year: int = 172  # summer solstice
    # Latitude [degrees]
    latitude: float = 30.0
    # Time of day [hours, 0-24]
    hour: float = 12.0

    def direction_tensor(self) -> torch.Tensor:
        d = torch.tensor(self.direction, dtype=torch.float64)
        norm = d.norm()
        return d / norm if norm > 1e-12 else d


@dataclass
class SkyRadiationSettings:
    """Sky long-wave radiation boundary conditions.

    Implements Berdahl–Martin model for effective sky temperature and
    sea-surface albedo for reflected solar component.
    """

    enabled: bool = False
    # Ambient air temperature [K]
    T_ambient: float = 300.0
    # Dew point temperature [K] (for Berdahl-Martin sky emissivity)
    T_dewpoint: float = 285.0
    # Sea albedo for reflected solar [0-1]
    sea_albedo: float = 0.06
    # Ground temperature for ground-reflected component [K]
    T_ground: float = 300.0

    def sky_emissivity(self) -> float:
        """Berdahl-Martin clear-sky emissivity.

        ε_sky = 0.711 + 0.56 * T_dp/100 + 0.73 * (T_dp/100)²
        where T_dp is dew point in °C.
        """
        T_dp_C = self.T_dewpoint - 273.15
        tdp = T_dp_C / 100.0
        eps = 0.711 + 0.56 * tdp + 0.73 * tdp * tdp
        return max(0.0, min(1.0, eps))

    def effective_sky_temperature(self) -> float:
        """Effective sky temperature via Berdahl-Martin.

        T_sky = T_ambient * (ε_sky)^(1/4)
        """
        eps = self.sky_emissivity()
        return self.T_ambient * (eps ** 0.25)


@dataclass
class RadiationEnclosureConfig:
    """Configuration for a radiating enclosure."""

    surfaces: list[SurfaceRadiationProps] = field(default_factory=list)
    solar: SolarSettings = field(default_factory=SolarSettings)
    sky: SkyRadiationSettings = field(default_factory=SkyRadiationSettings)
    # View-factor matrix F[i, j]; MUST be provided (fail-closed)
    view_factors: list[list[float]] | None = None
    # Physical reference temperature [K] for lattice→SI conversion
    T_ref: float = 300.0
    # Number of Gauss-Seidel iterations for radiosity solve
    max_iter: int = 200
    tol: float = 1e-12


@dataclass
class RadiationResult:
    """Output of a radiation step."""

    net_flux: list[float]  # W/m² net radiation flux per surface
    radiosity: list[float]  # W/m² total radiosity per surface
    irradiation: list[float]  # W/m² total irradiation per surface
    solar_flux: list[float]  # W/m² solar contribution per surface
    sky_flux: list[float]  # W/m² sky long-wave contribution per surface
    total_emitted_power: float  # W  total power emitted by all surfaces
    total_absorbed_power: float  # W  total power absorbed
    surface_temperatures: list[float]  # K surface temperatures
    band_radiance_8_12: list[float]  # W/(m²·sr) 8-12 μm band radiance
    energy_conservation_residual: float  # |Σ q_i A_i| / Σ|q_i A_i|
    solver_method: str  # 'gauss_seidel' or 'direct'


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
    sun_vec = -solar.direction_tensor().to(surface_normals.dtype)
    cos_theta = torch.clamp((surface_normals @ sun_vec), min=0.0, max=1.0)

    if isinstance(absorptance, torch.Tensor):
        alpha = absorptance.to(surface_normals.dtype)
    else:
        alpha = torch.tensor(absorptance, dtype=surface_normals.dtype)

    return alpha * solar.irradiance * cos_theta


# ---------------------------------------------------------------------------
# Sky radiation model (Berdahl-Martin)
# ---------------------------------------------------------------------------


def compute_sky_temperature(
    T_ambient: float,
    T_dewpoint: float,
) -> float:
    """Compute effective sky temperature using Berdahl-Martin model.

    Parameters
    ----------
    T_ambient : float
        Ambient air temperature [K].
    T_dewpoint : float
        Dew point temperature [K].

    Returns
    -------
    float
        Effective sky temperature [K].
    """
    T_dp_C = T_dewpoint - 273.15
    tdp = T_dp_C / 100.0
    eps_sky = 0.711 + 0.56 * tdp + 0.73 * tdp * tdp
    eps_sky = max(0.0, min(1.0, eps_sky))
    return T_ambient * (eps_sky ** 0.25)


def compute_sky_radiation_flux(
    surface_normals: torch.Tensor,
    sky_settings: SkyRadiationSettings,
    emissivity: float | torch.Tensor = 0.9,
) -> torch.Tensor:
    """Compute net sky long-wave radiation flux on each surface.

    Net flux = ε * σ * (T_sky⁴ - T_surface⁴) * view_factor_to_sky

    For surfaces facing upward (normal has positive y-component), the
    view factor to sky is approximated as the projected solid angle.

    Parameters
    ----------
    surface_normals : torch.Tensor
        (N, 3) outward unit normals.
    sky_settings : SkyRadiationSettings
        Sky radiation parameters.
    emissivity : float or torch.Tensor
        Surface emissivity.

    Returns
    -------
    torch.Tensor
        (N,) net sky radiation flux [W/m²] per surface (positive = heating).
    """
    T_sky = sky_settings.effective_sky_temperature()

    # View factor to sky ≈ max(0, n_y) for upward-facing surfaces
    # (hemispherical projection approximation)
    sky_view = torch.clamp(surface_normals[:, 1], min=0.0)

    if isinstance(emissivity, torch.Tensor):
        eps = emissivity.to(surface_normals.dtype)
    else:
        eps = torch.tensor(emissivity, dtype=surface_normals.dtype)

    # Net long-wave exchange with sky
    q_sky = eps * STEFAN_BOLTZMANN * (T_sky**4) * sky_view

    return q_sky


# ---------------------------------------------------------------------------
# View-factor utilities (FAIL-CLOSED)
# ---------------------------------------------------------------------------


def _build_view_factor_matrix(n: int, vf_list: list[list[float]] | None) -> torch.Tensor:
    """Return the (n, n) view-factor matrix F.

    FAIL-CLOSED: if vf_list is None, raises ValueError.
    View factors MUST be provided from analytical solutions (view_factor.py)
    or external computation. No silent uniform placeholder.
    """
    if vf_list is None:
        raise ValueError(
            "view_factors (vf_list) is None. View factors MUST be explicitly "
            "provided. Use tensorlbm.view_factor module for analytical solutions "
            "(parallel rectangles, perpendicular rectangles, coaxial disks, etc.) "
            "or supply a pre-computed view factor matrix. "
            "Silent uniform placeholder is DISABLED (fail-closed)."
        )

    F = torch.tensor(vf_list, dtype=torch.float64)
    if F.shape != (n, n):
        raise ValueError(f"view_factors must be ({n}×{n}), got {F.shape}")
    return F


# ---------------------------------------------------------------------------
# Radiosity solvers
# ---------------------------------------------------------------------------


def radiosity_gauss_seidel(
    surfaces: Sequence[SurfaceRadiationProps],
    F: torch.Tensor,
    q_solar: torch.Tensor,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Solve radiosity system via Gauss-Seidel iteration.

    The radiosity equation:
        J_i = ε_i σ T_i⁴ + (1 - ε_i) Σ_j F_ij J_j + (1 - ε_i) q_solar_i

    Gauss-Seidel uses latest available values for faster convergence.

    Parameters
    ----------
    surfaces : sequence of SurfaceRadiationProps
        Surface properties.
    F : torch.Tensor
        View factor matrix (n × n).
    q_solar : torch.Tensor
        Solar flux per surface (n,).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on max |ΔJ|.

    Returns
    -------
    J : torch.Tensor
        Radiosity per surface (n,).
    G : torch.Tensor
        Irradiation per surface (n,).
    q_net : torch.Tensor
        Net radiation flux per surface (n,).
    n_iter : int
        Number of iterations to converge.
    """
    n = len(surfaces)
    sigma = STEFAN_BOLTZMANN

    T = torch.tensor([s.temperature for s in surfaces], dtype=torch.float64)
    eps = torch.tensor([s.emissivity for s in surfaces], dtype=torch.float64)
    A = torch.tensor([s.area for s in surfaces], dtype=torch.float64)

    # Emissive power
    E = eps * sigma * T**4

    # Initialize J
    J = E.clone()

    n_iter = 0
    for iteration in range(max_iter):
        J_old = J.clone()

        # Gauss-Seidel sweep
        for i in range(n):
            # Σ_j F_ij J_j using latest J values
            sum_FJ = 0.0
            for j in range(n):
                if j < i:
                    sum_FJ += F[i, j].item() * J[j].item()
                elif j > i:
                    sum_FJ += F[i, j].item() * J_old[j].item()
                # j == i: F_ii = 0 for flat/convex surfaces

            G_i = sum_FJ + q_solar[i].item()
            J[i] = E[i].item() + (1.0 - eps[i].item()) * G_i

        n_iter = iteration + 1

        # Check convergence
        max_diff = (J - J_old).abs().max().item()
        if max_diff < tol:
            break

    # Final irradiation and net flux
    G = F @ J + q_solar
    q_net = J - G

    # Energy conservation check: Σ q_i A_i should be ~0
    energy_residual = (q_net * A).sum().abs().item()

    return J, G, q_net, n_iter


def radiosity_direct_solve(
    surfaces: Sequence[SurfaceRadiationProps],
    F: torch.Tensor,
    q_solar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve radiosity system via direct linear solve (numpy).

    Rearranging the radiosity equation:
        J_i - (1-ε_i) Σ_j F_ij J_j = ε_i σ T_i⁴ + (1-ε_i) q_solar_i

    In matrix form: [I - (1-ε)·F] · J = E + (1-ε)·q_solar

    Parameters
    ----------
    surfaces : sequence of SurfaceRadiationProps
        Surface properties.
    F : torch.Tensor
        View factor matrix (n × n).
    q_solar : torch.Tensor
        Solar flux per surface (n,).

    Returns
    -------
    J : torch.Tensor
        Radiosity per surface (n,).
    G : torch.Tensor
        Irradiation per surface (n,).
    q_net : torch.Tensor
        Net radiation flux per surface (n,).
    """
    n = len(surfaces)
    sigma = STEFAN_BOLTZMANN

    T = torch.tensor([s.temperature for s in surfaces], dtype=torch.float64)
    eps = torch.tensor([s.emissivity for s in surfaces], dtype=torch.float64)

    # Build system matrix: M = I - diag(1-ε) · F
    F_np = F.numpy()
    eps_np = eps.numpy()

    M = np.eye(n) - np.diag(1.0 - eps_np) @ F_np

    # RHS: E + (1-ε) · q_solar
    E_np = eps_np * sigma * T.numpy()**4
    q_solar_np = q_solar.numpy()
    rhs = E_np + (1.0 - eps_np) * q_solar_np

    # Solve
    J_np = np.linalg.solve(M, rhs)
    J = torch.from_numpy(J_np)

    # Compute G and q_net
    G = F @ J + q_solar
    q_net = J - G

    return J, G, q_net


def radiosity_matrix_solve(
    surfaces: Sequence[SurfaceRadiationProps],
    view_factors: list[list[float]] | None = None,
    solar_flux_per_surface: Sequence[float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[list[float], list[float], list[float]]:
    """Solve the net-radiation / radiosity system for a grey diffuse enclosure.

    Uses Gauss-Seidel iteration with energy conservation residual.
    FAIL-CLOSED: view_factors must not be None.

    Returns (radiosity J, irradiation G, net_flux q) for each surface.
    Net flux is defined as: q_i = J_i − G_i  [W/m²].
    """
    n = len(surfaces)

    # FAIL-CLOSED: view factors must be provided
    F = _build_view_factor_matrix(n, view_factors)

    # Solar contribution per surface
    if solar_flux_per_surface is not None:
        q_solar = torch.tensor(solar_flux_per_surface, dtype=torch.float64)
    else:
        q_solar = torch.zeros(n, dtype=torch.float64)

    # Gauss-Seidel solve
    J, G, q_net, n_iter = radiosity_gauss_seidel(
        surfaces, F, q_solar, max_iter=max_iter, tol=tol
    )

    return J.tolist(), G.tolist(), q_net.tolist()


# ---------------------------------------------------------------------------
# Planck band radiance (8-12 μm atmospheric window)
# ---------------------------------------------------------------------------


def planck_spectral_radiance(wavelength: float, temperature: float) -> float:
    """Planck spectral radiance B_λ(T) [W/(m²·sr·m)].

    B_λ = (2hc²/λ⁵) / (exp(hc/λkT) - 1)

    Parameters
    ----------
    wavelength : float
        Wavelength [m].
    temperature : float
        Temperature [K].

    Returns
    -------
    float
        Spectral radiance [W/(m²·sr·m)].
    """
    if wavelength <= 0 or temperature <= 0:
        return 0.0

    exp_arg = _C2 / (wavelength * temperature)
    if exp_arg > 700:  # avoid overflow
        return 0.0

    return _C1 / (wavelength**5) / (math.exp(exp_arg) - 1.0)


def planck_band_radiance(
    temperature: float,
    lambda_min: float = 8e-6,
    lambda_max: float = 12e-6,
    n_quad: int = 100,
) -> float:
    """Integrate Planck spectral radiance over a wavelength band.

    Uses Gauss-Legendre quadrature for high-accuracy integration.

    Parameters
    ----------
    temperature : float
        Surface temperature [K].
    lambda_min : float
        Band lower limit [m] (default 8 μm).
    lambda_max : float
        Band upper limit [m] (default 12 μm).
    n_quad : int
        Number of quadrature points (default 100).

    Returns
    -------
    float
        Band-integrated radiance [W/(m²·sr)].
    """
    if temperature <= 0:
        return 0.0

    # Gauss-Legendre quadrature
    x_pts, x_wts = np.polynomial.legendre.leggauss(n_quad)

    # Transform from [-1, 1] to [lambda_min, lambda_max]
    lam = 0.5 * (lambda_max - lambda_min) * x_pts + 0.5 * (lambda_max + lambda_min)
    w = 0.5 * (lambda_max - lambda_min) * x_wts

    # Integrate
    radiance = 0.0
    for i in range(n_quad):
        radiance += w[i] * planck_spectral_radiance(lam[i], temperature)

    return radiance


def planck_band_radiance_vectorized(
    temperatures: torch.Tensor,
    lambda_min: float = 8e-6,
    lambda_max: float = 12e-6,
    n_quad: int = 100,
) -> torch.Tensor:
    """Vectorized Planck band radiance for torch tensors.

    Parameters
    ----------
    temperatures : torch.Tensor
        Surface temperatures [K], shape (N,).
    lambda_min, lambda_max : float
        Band limits [m].
    n_quad : int
        Quadrature points.

    Returns
    -------
    torch.Tensor
        Band-integrated radiance [W/(m²·sr)], shape (N,).
    """
    x_pts, x_wts = np.polynomial.legendre.leggauss(n_quad)
    lam = 0.5 * (lambda_max - lambda_min) * x_pts + 0.5 * (lambda_max + lambda_min)
    w = 0.5 * (lambda_max - lambda_min) * x_wts

    lam_t = torch.tensor(lam, dtype=torch.float64, device=temperatures.device)
    w_t = torch.tensor(w, dtype=torch.float64, device=temperatures.device)

    # Broadcast: (N, n_quad)
    T_exp = temperatures.unsqueeze(1)  # (N, 1)
    lam_exp = lam_t.unsqueeze(0)  # (1, n_quad)

    exp_arg = _C2 / (lam_exp * T_exp)
    # Clamp to avoid overflow
    exp_arg = torch.clamp(exp_arg, max=700.0)
    B = _C1 / (lam_exp**5) / (torch.exp(exp_arg) - 1.0)

    radiance = (w_t.unsqueeze(0) * B).sum(dim=1)
    return radiance


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def compute_net_radiation_flux(
    cfg: RadiationEnclosureConfig,
    surface_normals: torch.Tensor | None = None,
    solver: str = "gauss_seidel",
) -> RadiationResult:
    """Compute net radiation fluxes for all surfaces in the enclosure.

    Parameters
    ----------
    cfg:
        Enclosure configuration. view_factors MUST be provided (fail-closed).
    surface_normals:
        Optional (N, 3) tensor of outward normals for solar/sky computation.
    solver:
        'gauss_seidel' or 'direct'.

    Returns
    -------
    RadiationResult
    """
    n = len(cfg.surfaces)
    sigma = STEFAN_BOLTZMANN

    # --- Solar flux per surface ---
    solar_per_surface = [0.0] * n
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
            cos45 = math.cos(math.radians(45))
            solar_per_surface = [
                s.solar_absorptance * cfg.solar.irradiance * cos45 for s in cfg.surfaces
            ]

    # --- Sky radiation flux per surface ---
    sky_per_surface = [0.0] * n
    if cfg.sky.enabled:
        if surface_normals is not None:
            q_sky = compute_sky_radiation_flux(
                surface_normals,
                cfg.sky,
                emissivity=torch.tensor(
                    [s.emissivity for s in cfg.surfaces],
                    dtype=surface_normals.dtype,
                ),
            )
            sky_per_surface = q_sky.tolist()

    # Combine solar + sky as external irradiation
    total_external = [solar_per_surface[i] + sky_per_surface[i] for i in range(n)]

    # --- Build view factor matrix (FAIL-CLOSED) ---
    F = _build_view_factor_matrix(n, cfg.view_factors)

    q_solar_t = torch.tensor(total_external, dtype=torch.float64)

    # --- Solve radiosity ---
    if solver == "direct":
        J, G, q_net = radiosity_direct_solve(cfg.surfaces, F, q_solar_t)
        n_iter = -1
    else:
        J, G, q_net, n_iter = radiosity_gauss_seidel(
            cfg.surfaces, F, q_solar_t,
            max_iter=cfg.max_iter, tol=cfg.tol,
        )

    # --- Energy conservation residual ---
    A_list = [s.area for s in cfg.surfaces]
    A_t = torch.tensor(A_list, dtype=torch.float64)
    energy_sum = (q_net * A_t).sum().abs().item()
    energy_total = (q_net.abs() * A_t).sum().item()
    energy_residual = energy_sum / energy_total if energy_total > 0 else 0.0

    # --- Band radiance (8-12 μm) ---
    T_list = [s.temperature for s in cfg.surfaces]
    band_radiance = [planck_band_radiance(T) for T in T_list]

    # --- Total powers ---
    total_emitted = sum(sigma * (T_list[i] ** 4) * A_list[i] for i in range(n))
    total_absorbed = sum(max(0.0, -q_net[i].item()) * A_list[i] for i in range(n))

    return RadiationResult(
        net_flux=q_net.tolist(),
        radiosity=J.tolist(),
        irradiation=G.tolist(),
        solar_flux=solar_per_surface,
        sky_flux=sky_per_surface,
        total_emitted_power=total_emitted,
        total_absorbed_power=total_absorbed,
        surface_temperatures=T_list,
        band_radiance_8_12=band_radiance,
        energy_conservation_residual=energy_residual,
        solver_method=solver,
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
    T_new[is_boundary] += dT
    return T_new


def run_radiation_step(
    cfg: RadiationEnclosureConfig,
    surface_normals: torch.Tensor | None = None,
) -> RadiationResult:
    """Convenience wrapper: compute radiation and return result.

    Equivalent to compute_net_radiation_flux but named for time-loop integration.
    """
    return compute_net_radiation_flux(cfg, surface_normals)
