"""Turbulent inlet boundary-condition profiles for LBM simulations.

Provides physically motivated inlet velocity distributions to replace the
uniform inflow condition.  These profiles are analogous to the inlet-BC
options available in PowerFlow and XFlow and are required for accurate
predictions of turbulent wall-bounded flows, atmospheric boundary layers,
and pipe / channel flows.

Profiles
--------
:func:`log_law_profile`
    Logarithmic law-of-the-wall (von Kármán) profile for fully developed
    turbulent channel / boundary-layer flow.
:func:`power_law_profile`
    1/n power-law profile; commonly used for pipe flow and atmospheric
    boundary layers.
:func:`womersley_profile`
    Oscillatory pressure-driven (pulsatile) Womersley profile for
    cardiovascular / reciprocating-machine applications.
:func:`parabolic_profile`
    Laminar Hagen–Poiseuille parabolic profile for low-Re validation.
:func:`blasius_profile`
    Flat-plate boundary-layer profile from the Blasius similarity solution.
:func:`synthetic_turbulence_2d`
    Superimpose synthetic random-phase turbulent fluctuations on a mean
    profile using the Random Fourier Modes method (Smirnov *et al.* 2001).
:func:`apply_inlet_profile_2d`
    Apply a 1-D velocity profile to the left inlet plane of a 2-D domain.
:func:`apply_inlet_profile_3d`
    Apply a 2-D velocity profile to the left inlet plane of a 3-D domain.

References
----------
Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.
Smirnov, A., Shi, S., & Celik, I. (2001). Random flow generation technique for
    large eddy simulations and particle-dynamics modeling. *J. Fluids Eng.* 123,
    359–371.
Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow
    and viscous drag in arteries when the pressure gradient is known. *J. Physiol.*
    127, 553–563.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# 1-D profiles (return tensors of shape (n,) over y ∈ [0, 1])
# ---------------------------------------------------------------------------

def log_law_profile(
    n: int,
    u_bulk: float,
    re_tau: float,
    nu: float,
    kappa: float = 0.41,
    b: float = 5.0,
    n_iter: int = 20,
) -> torch.Tensor:
    """Log-law velocity profile for fully developed turbulent channel flow.

    Matches the log law ``U+ = (1/κ) ln(y+) + B`` in the log region and a
    linear ``U+ = y+`` in the viscous sub-layer (y+ < 11.6).  The bulk
    velocity is iteratively scaled to match *u_bulk*.

    Args:
        n:        Number of grid cells across the channel (wall to wall).
        u_bulk:   Target bulk (mean) velocity in lattice units.
        re_tau:   Friction Reynolds number Re_τ = u_τ h / ν  (h = half-width).
        nu:       Kinematic viscosity in lattice units.
        kappa:    Von Kármán constant (default 0.41).
        b:        Log-law intercept (default 5.0).
        n_iter:   Iterations to match bulk velocity (default 20).

    Returns:
        Velocity profile tensor of shape ``(n,)`` in lattice units.
    """
    # Build y+ array from wall to centerline (0 → Re_τ)
    # Symmetric channel: half-width h = n/2
    h = n / 2.0
    u_tau_init = re_tau * nu / h

    y_grid = torch.linspace(0.5, n - 0.5, n)          # cell centres
    y_dist = torch.minimum(y_grid, n - y_grid)         # distance from nearest wall
    y_plus = y_dist * u_tau_init / nu

    # Composite profile: linear sub-layer + log outer region
    u_plus = torch.where(
        y_plus < 11.6,
        y_plus,
        (1.0 / kappa) * torch.log(y_plus.clamp(min=1e-8)) + b,
    )

    u_profile = u_plus * u_tau_init

    # Scale to match target bulk velocity
    u_current = float(u_profile.mean())
    if u_current > 1e-12:
        u_profile = u_profile * (u_bulk / u_current)

    return u_profile


def power_law_profile(
    n: int,
    u_centerline: float,
    exponent: float = 7.0,
) -> torch.Tensor:
    """1/n power-law velocity profile (symmetric about channel centre).

    U(y) = U_cl · (1 − |2y/D − 1|)^(1/n)

    where D = channel width (= n cells) and y ∈ [0, D].

    Args:
        n:             Number of grid cells across the channel.
        u_centerline:  Centreline velocity.
        exponent:      Power-law exponent (n=7 for turbulent pipe at moderate Re).

    Returns:
        Velocity profile tensor of shape ``(n,)``.
    """
    y = torch.linspace(0.5, n - 0.5, n)
    eta = 1.0 - (2.0 * y / n - 1.0).abs()   # 0 at walls, 1 at centre
    return u_centerline * eta.pow(1.0 / exponent)


def parabolic_profile(
    n: int,
    u_centerline: float,
) -> torch.Tensor:
    """Laminar Hagen–Poiseuille parabolic profile.

    U(y) = U_cl · (1 − (2y/D − 1)²)

    Args:
        n:             Number of grid cells.
        u_centerline:  Centreline velocity.

    Returns:
        Velocity profile tensor of shape ``(n,)``.
    """
    y = torch.linspace(0.5, n - 0.5, n)
    eta = 2.0 * y / n - 1.0            # −1 to +1
    return u_centerline * (1.0 - eta * eta)


def blasius_profile(
    n: int,
    u_inf: float,
    delta_99: float = 0.3,
) -> torch.Tensor:
    """Blasius flat-plate boundary-layer profile (tabulated similarity solution).

    The profile is defined for y ∈ [0, n] with the boundary layer of
    thickness *delta_99* (in lattice units).  Above the B.L. the velocity
    equals *u_inf*.

    Args:
        n:        Number of grid cells in wall-normal direction.
        u_inf:    Free-stream velocity.
        delta_99: 99% boundary-layer thickness in lattice units.

    Returns:
        Velocity profile tensor of shape ``(n,)``.
    """
    # Blasius f'(η) table (η, f')  – 14-point tabulation
    _eta = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 5.0]
    _fp  = [0.0, 0.066, 0.133, 0.199, 0.265, 0.330, 0.394, 0.516, 0.630, 0.729, 0.812, 0.876, 0.923, 0.992]
    eta_t = torch.tensor(_eta, dtype=torch.float32)
    fp_t  = torch.tensor(_fp,  dtype=torch.float32)

    # η at which f'(η) ≈ 0.99 is approximately 4.91
    eta_99 = 4.91

    y = torch.linspace(0.5, n - 0.5, n)
    eta_grid = y * (eta_99 / delta_99)

    # Interpolate f'(η) at each grid point
    u_profile = torch.zeros(n)
    for i in range(n):
        eta_i = float(eta_grid[i])
        if eta_i >= eta_t[-1]:
            u_profile[i] = u_inf
        else:
            # Linear interpolation in table
            idx = int((eta_t <= eta_i).sum()) - 1
            idx = max(0, min(len(_eta) - 2, idx))
            t = (eta_i - float(eta_t[idx])) / max(float(eta_t[idx + 1] - eta_t[idx]), 1e-8)
            fp_i = float(fp_t[idx]) + t * float(fp_t[idx + 1] - fp_t[idx])
            u_profile[i] = fp_i * u_inf

    return u_profile


def womersley_profile(
    n: int,
    u_mean: float,
    wo: float,
    phase: float = 0.0,
    n_harmonics: int = 4,
) -> torch.Tensor:
    """Womersley oscillatory profile for pulsatile pressure-driven pipe flow.

    The Womersley number Wo = R √(ω/ν) characterises the ratio of oscillatory
    inertia to viscous effects.  For Wo < 2 the profile is nearly parabolic;
    for Wo > 10 it is nearly plug-like with thin Stokes layers at the wall.

    This function returns the **instantaneous** radial velocity profile at a
    given *phase* angle (in radians) as a superposition of *n_harmonics* modes.

    Args:
        n:           Number of grid cells across the diameter.
        u_mean:      Time-averaged mean velocity (Hagen–Poiseuille component).
        wo:          Womersley number.
        phase:       Phase angle ωt in radians.
        n_harmonics: Number of harmonics (1 = fundamental only).

    Returns:
        Instantaneous velocity profile tensor of shape ``(n,)``.
    """
    # Normalised radial coordinate r ∈ [0, 1]  (0 = centre, 1 = wall)
    y = torch.linspace(0.5, n - 0.5, n)
    r = (2.0 * y / n - 1.0).abs()    # 0 at centre, 1 at wall

    # Mean parabolic component
    u = u_mean * (1.0 - r * r) * 2.0   # factor 2 so mean = u_mean

    # Oscillatory component: amplitude decays as 1/k²
    for k in range(1, n_harmonics + 1):
        wo_k = wo * math.sqrt(k)
        amp = u_mean / (k * k)
        # Approximate Womersley function: plug core + Stokes layer (boundary layer ≈ δ_s = sqrt(2/ω))
        # Full solution involves Bessel functions J_0; here we use a boundary-layer approximation
        delta_s = 1.0 / (wo_k + 1e-8)
        stokes = torch.exp(-(1.0 - r) / max(delta_s, 1e-3))
        u_osc = amp * (1.0 - stokes) * math.cos(k * phase)
        u = u + u_osc

    u = u.clamp(min=0.0)
    return u


# ---------------------------------------------------------------------------
# Synthetic turbulence
# ---------------------------------------------------------------------------

def synthetic_turbulence_2d(
    u_mean: torch.Tensor,
    turbulence_intensity: float = 0.05,
    n_modes: int = 64,
    length_scale: float = 5.0,
    seed: int = 42,
) -> torch.Tensor:
    """Superimpose synthetic turbulent fluctuations on a mean profile (2-D inlet).

    Uses the Random Fourier Modes method: fluctuations are generated as a sum
    of Fourier modes with random phases.  The RMS fluctuation matches the
    target turbulence intensity.

    Args:
        u_mean:               Mean velocity profile, shape ``(n,)``.
        turbulence_intensity: u'/U_bulk (fraction; default 5%).
        n_modes:              Number of Fourier modes (default 64).
        length_scale:         Turbulence integral length scale in lattice units.
        seed:                 Random seed for reproducibility.

    Returns:
        Perturbed velocity profile of shape ``(n,)`` with fluctuations.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = u_mean.numel()

    u_bulk = float(u_mean.mean())
    target_rms = turbulence_intensity * abs(u_bulk)

    y = torch.linspace(0.0, float(n - 1), n)
    fluctuation = torch.zeros(n)

    for _ in range(n_modes):
        k = torch.rand(1, generator=rng).item() * (math.pi / length_scale)
        phi = torch.rand(1, generator=rng).item() * 2.0 * math.pi
        amp = torch.randn(1, generator=rng).item()
        fluctuation = fluctuation + amp * torch.cos(k * y + phi)

    # Normalise to target RMS
    rms = float(fluctuation.std())
    if rms > 1e-10:
        fluctuation = fluctuation * (target_rms / rms)

    return u_mean + fluctuation.to(u_mean.device)


# ---------------------------------------------------------------------------
# Apply profile to a domain
# ---------------------------------------------------------------------------

def apply_inlet_profile_2d(
    f: torch.Tensor,
    u_profile: torch.Tensor,
    rho_in: float = 1.0,
    x_inlet: int = 0,
) -> torch.Tensor:
    """Apply a 1-D velocity profile to the inlet plane of a 2-D domain.

    Sets the equilibrium distribution at the inlet column (x = *x_inlet*)
    using the given streamwise velocity profile and a uniform density *rho_in*.

    Args:
        f:          Distribution function, shape ``(9, ny, nx)``.
        u_profile:  Streamwise velocity profile, shape ``(ny,)``.
        rho_in:     Inlet density (default 1.0).
        x_inlet:    x-index of the inlet plane (default 0 = left boundary).

    Returns:
        Updated distribution tensor with inlet populations set.
    """
    from .d2q9 import equilibrium  # noqa: PLC0415

    ny = u_profile.numel()
    device = f.device
    # equilibrium expects 2-D inputs (ny, nx); use shape (1, ny) then squeeze
    rho = torch.full((1, ny), rho_in, dtype=f.dtype, device=device)
    ux  = u_profile.to(device=device, dtype=f.dtype).unsqueeze(0)  # (1, ny)
    uy  = torch.zeros(1, ny, dtype=f.dtype, device=device)
    feq = equilibrium(rho, ux, uy)   # (9, 1, ny)
    f_out = f.clone()
    f_out[:, :, x_inlet] = feq[:, 0, :]  # (9, ny)
    return f_out


def apply_inlet_profile_3d(
    f: torch.Tensor,
    u_profile: torch.Tensor,
    rho_in: float = 1.0,
    x_inlet: int = 0,
) -> torch.Tensor:
    """Apply a 2-D velocity profile to the inlet plane of a 3-D domain.

    Args:
        f:          Distribution function, shape ``(19, nz, ny, nx)`` (D3Q19).
        u_profile:  Streamwise velocity field at inlet, shape ``(nz, ny)``.
        rho_in:     Inlet density (default 1.0).
        x_inlet:    x-index of the inlet plane (default 0).

    Returns:
        Updated distribution tensor with inlet populations set.
    """
    from .d3q19 import equilibrium3d  # noqa: PLC0415

    nz, ny = u_profile.shape
    device = f.device
    # equilibrium3d expects (nz, ny, nx); use shape (nz, ny, 1) then squeeze
    rho = torch.full((nz, ny, 1), rho_in, dtype=f.dtype, device=device)
    ux  = u_profile.to(device=device, dtype=f.dtype).unsqueeze(-1)  # (nz, ny, 1)
    uy  = torch.zeros(nz, ny, 1, dtype=f.dtype, device=device)
    uz  = torch.zeros(nz, ny, 1, dtype=f.dtype, device=device)
    feq = equilibrium3d(rho, ux, uy, uz)   # (19, nz, ny, 1)
    f_out = f.clone()
    f_out[:, :, :, x_inlet] = feq[:, :, :, 0]  # (19, nz, ny)
    return f_out


__all__ = [
    "log_law_profile",
    "power_law_profile",
    "parabolic_profile",
    "blasius_profile",
    "womersley_profile",
    "synthetic_turbulence_2d",
    "apply_inlet_profile_2d",
    "apply_inlet_profile_3d",
]
