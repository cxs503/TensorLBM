"""Passive scalar (concentration) transport for Taylor–Aris dispersion verification.

D2Q5 BGK advection–diffusion solver, unidirectionally coupled to a D2Q9 flow
field.  Provides:

* Independent scalar relaxation time τ_s (diffusivity D = (τ_s − ½)/3)
* Total-mass conservation projection (optional, default off)
* Non-negativity protection (optional, default off)
* Second-moment extractor σ²(t) → D_eff
* Poiseuille body-force driver for channel flow
* Three canonical test cases: pure diffusion, pure convection, pipe dispersion

Physics
-------
The scalar field C obeys  ∂C/∂t + u·∇C = D∇²C.

Taylor–Aris dispersion (long-time limit, Pe = Ua/D):
    D_eff = D + a²U² / (48 D)

where a = half-channel-height, U = mean velocity.

Early-time regime (t ≪ a²/D):
    σ²(t) ∝ t²  (ballistic / shear-dominated)

Late-time regime (t ≫ a²/D):
    σ²(t) = 2 D_eff t  (diffusive)

References
----------
Taylor, G. I. (1953). Dispersion of soluble matter in solvent flowing
    slowly through a tube. *Proc. R. Soc. Lond. A* 219, 186–203.
Aris, R. (1956). On the dispersion of a solute in a fluid flowing
    through a tube. *Proc. R. Soc. Lond. A* 235, 67–77.
Shi, B. & Guo, Z. (2009). LBM for nonlinear convection–diffusion.
    *Phys. Rev. E* 79, 016701.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .d2q9 import OPPOSITE as _OPP9
from .d2q9 import equilibrium as f_equilibrium
from .d2q9 import macroscopic as f_macroscopic
from .solver import collide_bgk, stream

# ── D2Q5 scalar lattice ──────────────────────────────────────────────────────
# Order: 0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1)
_C5 = torch.tensor([0, 1, 0, -1, 0], dtype=torch.int64)
_CY5 = torch.tensor([0, 0, 1, 0, -1], dtype=torch.int64)
_W5_EXACT = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
_OPP5 = torch.tensor([0, 3, 4, 1, 2], dtype=torch.int64)
_CS2 = 1.0 / 3.0  # D2Q5 sound speed squared

# Stream index cache
_stream5_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _w5_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """D2Q5 weights in the requested dtype."""
    return torch.tensor(_W5_EXACT, dtype=dtype, device=device)


# ── Core LBM operators ───────────────────────────────────────────────────────

def scalar_equilibrium(
    c: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """D2Q5 scalar equilibrium: g_i^eq = w_i C (1 + 3 c_i·u).

    Args:
        c:  Concentration field, shape (ny, nx).
        ux: x-velocity, shape (ny, nx).
        uy: y-velocity, shape (ny, nx).

    Returns:
        Equilibrium distributions, shape (5, ny, nx).
    """
    device = c.device
    dtype = c.dtype
    w = _w5_tensor(dtype, device).view(5, 1, 1)
    cx = _C5.to(device).view(5, 1, 1).to(dtype)
    cy = _CY5.to(device).view(5, 1, 1).to(dtype)
    cu = cx * ux + cy * uy
    return w * c.unsqueeze(0) * (1.0 + 3.0 * cu)


def scalar_collide_bgk(
    g: torch.Tensor,
    c: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_s: float,
) -> torch.Tensor:
    """BGK collision for scalar: g* = g − (g − g_eq)/τ_s."""
    g_eq = scalar_equilibrium(c, ux, uy)
    return g - (g - g_eq) / tau_s


def scalar_stream(g: torch.Tensor) -> torch.Tensor:
    """Periodic streaming for D2Q5 scalar distributions."""
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    key = (ny, nx, device.type, getattr(device, "index", None))
    if key not in _stream5_cache:
        c = _C5.to(device)
        cy = _CY5.to(device)
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - cy.unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c.unsqueeze(1)) % nx
        q_idx = torch.arange(5, device=device).view(5, 1, 1).expand(5, ny, nx)
        y_idx = y_src.unsqueeze(2).expand(5, ny, nx)
        x_idx = x_src.unsqueeze(1).expand(5, ny, nx)
        _stream5_cache[key] = (q_idx, y_idx, x_idx)
    q_idx, y_idx, x_idx = _stream5_cache[key]
    return g[q_idx, y_idx, x_idx]


def scalar_macroscopic(g: torch.Tensor) -> torch.Tensor:
    """Recover concentration: C = Σ_i g_i."""
    return g.sum(dim=0)


def scalar_diffusivity(tau_s: float) -> float:
    """Physical diffusivity from relaxation time: D = (τ_s − ½) / 3."""
    return (tau_s - 0.5) * _CS2


def tau_from_diffusivity(D: float) -> float:
    """Relaxation time from target diffusivity."""
    return D / _CS2 + 0.5


# ── Conservation & non-negativity ────────────────────────────────────────────

def project_mass_conservation(g: torch.Tensor, target_mass: float | None = None) -> torch.Tensor:
    """Rescale all populations so that ΣC = target_mass (or initial mass).

    Args:
        g: Scalar distributions, shape (5, ny, nx).
        target_mass: Desired total mass. If None, uses the mass of g[0]
            (assumed to be the initial rest-population mass reference).

    Returns:
        Rescaled distributions.
    """
    current_mass = g.sum().item()
    if target_mass is None or current_mass == 0.0:
        return g
    scale = target_mass / current_mass
    return g * scale


def enforce_nonnegativity(g: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Clip negative concentrations to floor (redistribute mass to rest pop).

    Args:
        g: Scalar distributions, shape (5, ny, nx).
        floor: Minimum allowed value (default 0).

    Returns:
        Non-negative distributions with mass conserved.
    """
    c = scalar_macroscopic(g)
    neg_mask = c < floor
    if not neg_mask.any():
        return g
    # Compute mass deficit from clipping
    c_clipped = torch.clamp(c, min=floor)
    deficit = (c - c_clipped).sum()
    # Redistribute deficit to rest population uniformly
    g_new = g.clone()
    c_new = c_clipped
    # Reconstruct: set all populations proportional to equilibrium at u=0
    # g_i = w_i * C_new + (g_i - w_i * C) to preserve non-equilibrium
    w = _w5_tensor(g.dtype, g.device).view(5, 1, 1)
    g_eq_new = w * c_new.unsqueeze(0)
    g_neq = g - w * c.unsqueeze(0)
    g_new = g_eq_new + g_neq
    # Final mass correction
    g_new = project_mass_conservation(g_new, g.sum().item())
    return g_new


# ── Boundary conditions ──────────────────────────────────────────────────────

def scalar_bounce_back(g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero-flux (bounce-back) for D2Q5 scalar on solid cells."""
    opp = _OPP5.to(g.device)
    return torch.where(mask.unsqueeze(0), g[opp], g)


def momentum_bounce_back(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back for D2Q9 momentum on solid cells."""
    opp = _OPP9.to(f.device)
    return torch.where(mask.unsqueeze(0), f[opp], f)


def scalar_dirichlet(g: torch.Tensor, c_val: float, mask: torch.Tensor,
                     ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Dirichlet BC: set scalar to c_val at masked cells."""
    c_bc = torch.full_like(ux, c_val)
    g_eq = scalar_equilibrium(c_bc, ux, uy)
    m = mask.unsqueeze(0).expand_as(g)
    return torch.where(m, g_eq, g)


# ── Poiseuille body force ────────────────────────────────────────────────────

def poiseuille_velocity_profile(y: torch.Tensor, ny: int, u_max: float) -> torch.Tensor:
    """Parabolic Poiseuille profile: u(y) = 4 u_max (y/H)(1 − y/H).

    Args:
        y: y-coordinate tensor (ny,).
        ny: Channel height in lattice units.
        u_max: Centerline velocity.

    Returns:
        Velocity profile (ny,).
    """
    y_norm = y.float() / (ny - 1)
    return 4.0 * u_max * y_norm * (1.0 - y_norm)


def apply_body_force(f: torch.Tensor, fx: float, fy: float,
                     solid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Apply uniform body force to momentum distribution via velocity shift.

    u' = u + F/ρ, then re-equilibrate the change.
    """
    rho, ux, uy = f_macroscopic(f)
    ux_new = ux + fx / rho
    uy_new = uy + fy / rho
    if solid_mask is not None:
        ux_new = torch.where(solid_mask, torch.zeros_like(ux_new), ux_new)
        uy_new = torch.where(solid_mask, torch.zeros_like(uy_new), uy_new)
    # Add the force contribution to the distribution
    feq_old = f_equilibrium(rho, ux, uy)
    feq_new = f_equilibrium(rho, ux_new, uy_new)
    return f + (feq_new - feq_old)


# ── Second moment analysis ───────────────────────────────────────────────────

def second_moment_x(c: torch.Tensor, dx: float = 1.0) -> dict[str, float]:
    """Compute the x-direction second central moment σ²_x of concentration.

    σ²_x = Σ (x - x̄)² C / Σ C

    Args:
        c: Concentration field, shape (ny, nx).
        dx: Lattice spacing (default 1.0).

    Returns:
        Dict with 'mean', 'variance', 'total_mass'.
    """
    ny, nx = c.shape
    device = c.device
    x = torch.arange(nx, device=device, dtype=c.dtype) * dx
    y = torch.arange(ny, device=device, dtype=c.dtype) * dx

    total = c.sum()
    if total < 1e-15:
        return {"mean": 0.0, "variance": 0.0, "total_mass": 0.0}

    # Mean position (mass-weighted)
    x_mean = (x.unsqueeze(0) * c).sum() / total
    y_mean = (y.unsqueeze(1) * c).sum() / total

    # Variance
    var_x = (((x.unsqueeze(0) - x_mean) ** 2) * c).sum() / total
    var_y = (((y.unsqueeze(1) - y_mean) ** 2) * c).sum() / total

    return {
        "x_mean": float(x_mean.item()),
        "y_mean": float(y_mean.item()),
        "var_x": float(var_x.item()),
        "var_y": float(var_y.item()),
        "total_mass": float(total.item()),
    }


def fit_diffusivity(times: list[float], variances: list[float],
                    t_min: float = 0.0) -> dict[str, float]:
    """Fit D_eff from σ²(t) = σ²₀ + 2 D_eff t (linear regression).

    Args:
        times: Time stamps.
        variances: σ² values.
        t_min: Minimum time to include in fit.

    Returns:
        Dict with 'D_eff', 'r_squared', 'intercept'.
    """
    t_arr = np.array([t for t, v in zip(times, variances) if t >= t_min])
    v_arr = np.array([v for t, v in zip(times, variances) if t >= t_min])

    if len(t_arr) < 2:
        return {"D_eff": 0.0, "r_squared": 0.0, "intercept": 0.0}

    # Linear fit: v = intercept + 2*D_eff * t
    coeffs = np.polyfit(t_arr, v_arr, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    D_eff = slope / 2.0

    # R²
    v_pred = np.polyval(coeffs, t_arr)
    ss_res = np.sum((v_arr - v_pred) ** 2)
    ss_tot = np.sum((v_arr - np.mean(v_arr)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"D_eff": float(D_eff), "r_squared": float(r_squared), "intercept": float(intercept)}


def fit_early_slope(times: list[float], variances: list[float],
                    t_max: float = float("inf")) -> dict[str, float]:
    """Fit log-log slope of σ²(t) in early-time regime.

    Expected: σ² ∝ t² → slope ≈ 2.

    Args:
        times: Time stamps.
        variances: σ² values.
        t_max: Maximum time to include.

    Returns:
        Dict with 'slope', 'r_squared'.
    """
    t_arr = np.array([t for t, v in zip(times, variances) if t > 0 and t <= t_max])
    v_arr = np.array([v for t, v in zip(times, variances) if t > 0 and t <= t_max])

    if len(t_arr) < 3:
        return {"slope": 0.0, "r_squared": 0.0}

    log_t = np.log(t_arr)
    log_v = np.log(v_arr)

    coeffs = np.polyfit(log_t, log_v, 1)
    slope = coeffs[0]

    v_pred = np.polyval(coeffs, log_t)
    ss_res = np.sum((log_v - v_pred) ** 2)
    ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"slope": float(slope), "r_squared": float(r_squared)}


# ── Taylor–Aris analytical solution ──────────────────────────────────────────

def taylor_aris_deff(D: float, U_mean: float, half_height: float) -> float:
    """Taylor–Aris effective diffusivity: D_eff = D + a²U²/(48D).

    Args:
        D: Molecular diffusivity.
        U_mean: Mean (bulk) velocity.
        half_height: Half channel height (a).

    Returns:
        Effective diffusivity.
    """
    if D <= 0:
        return float("inf")
    return D + (half_height ** 2) * (U_mean ** 2) / (48.0 * D)


# ── Simulation runners ───────────────────────────────────────────────────────

@dataclass
class ScalarSimState:
    """Runtime state for scalar simulation."""
    f: torch.Tensor  # momentum distributions (9, ny, nx)
    g: torch.Tensor  # scalar distributions (5, ny, nx)
    solid_mask: torch.Tensor
    tau_f: float
    tau_s: float
    step: int = 0
    initial_mass: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    min_concentration: float = 0.0


def init_channel_flow(
    ny: int, nx: int, u_max: float, tau_f: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize Poiseuille flow in a 2D channel.

    Returns (f, solid_mask) where solid_mask marks top/bottom walls.
    """
    y = torch.arange(ny, device=device, dtype=torch.float32)
    ux = poiseuille_velocity_profile(y, ny, u_max).unsqueeze(1).expand(ny, nx)
    uy = torch.zeros(ny, nx, device=device)
    rho = torch.ones(ny, nx, device=device)

    f = f_equilibrium(rho, ux, uy)

    solid_mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    solid_mask[0, :] = True
    solid_mask[-1, :] = True

    return f, solid_mask


def init_scalar_pulse(
    g: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor,
    x_center: int, y_center: int,
    sigma: float = 3.0,
    total_mass: float = 1.0,
) -> torch.Tensor:
    """Initialize a Gaussian scalar pulse."""
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    dtype = g.dtype

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nx, device=device, dtype=dtype),
        indexing="ij",
    )
    c_init = total_mass / (2 * math.pi * sigma ** 2) * torch.exp(
        -0.5 * ((xx - x_center) ** 2 + (yy - y_center) ** 2) / sigma ** 2
    )

    g = scalar_equilibrium(c_init, ux, uy)
    return g


def init_scalar_line(
    g: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor,
    x_center: int,
    sigma: float = 2.0,
    total_mass: float = 1.0,
) -> torch.Tensor:
    """Initialize a line source (uniform in y, Gaussian in x)."""
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    dtype = g.dtype

    x = torch.arange(nx, device=device, dtype=dtype)
    c_init = total_mass / (math.sqrt(2 * math.pi) * sigma * ny) * torch.exp(
        -0.5 * ((x - x_center) ** 2) / sigma ** 2
    ).unsqueeze(0).expand(ny, nx)

    g = scalar_equilibrium(c_init, ux, uy)
    return g


def run_scalar_step(
    state: ScalarSimState,
    force_x: float = 0.0,
    conserve_mass: bool = False,
    enforce_nonneg: bool = False,
) -> ScalarSimState:
    """One coupled flow+scalar time step.

    Flow: collide → body force → stream → bounce-back
    Scalar: collide → stream → bounce-back
    """
    # --- Flow step ---
    state.f = collide_bgk(state.f, state.tau_f)

    # Body force
    if force_x != 0.0:
        state.f = apply_body_force(state.f, force_x, 0.0, state.solid_mask)

    state.f = stream(state.f)
    state.f = momentum_bounce_back(state.f, state.solid_mask)

    # --- Scalar step ---
    rho_f, ux_f, uy_f = f_macroscopic(state.f)
    # Zero velocity in solid
    ux_f = torch.where(state.solid_mask, torch.zeros_like(ux_f), ux_f)
    uy_f = torch.where(state.solid_mask, torch.zeros_like(uy_f), uy_f)

    c = scalar_macroscopic(state.g)
    state.g = scalar_collide_bgk(state.g, c, ux_f, uy_f, state.tau_s)
    state.g = scalar_stream(state.g)
    state.g = scalar_bounce_back(state.g, state.solid_mask)

    # Optional conservation / non-negativity
    if conserve_mass and state.initial_mass > 0:
        state.g = project_mass_conservation(state.g, state.initial_mass)
    if enforce_nonneg:
        state.g = enforce_nonnegativity(state.g)

    state.step += 1

    # Track minimum concentration
    c_now = scalar_macroscopic(state.g)
    state.min_concentration = min(state.min_concentration, float(c_now.min().item()))

    return state


def run_poiseuille_dispersion(
    ny: int, nx: int,
    u_max: float,
    tau_f: float,
    tau_s: float,
    n_steps: int,
    x_center: int | None = None,
    sigma: float = 3.0,
    total_mass: float = 1.0,
    device: torch.device | str = "cpu",
    conserve_mass: bool = False,
    enforce_nonneg: bool = False,
    record_interval: int = 10,
    force_x: float = 0.0,
) -> ScalarSimState:
    """Run Poiseuille channel dispersion simulation.

    Args:
        ny, nx: Grid dimensions.
        u_max: Centerline velocity (for initial profile).
        tau_f: Flow relaxation time.
        tau_s: Scalar relaxation time.
        n_steps: Number of time steps.
        x_center: Pulse injection x-position (default nx//4).
        sigma: Initial pulse width.
        total_mass: Total scalar mass.
        device: Compute device.
        conserve_mass: Enable mass conservation projection.
        enforce_nonneg: Enable non-negativity protection.
        record_interval: Record history every N steps.
        force_x: Body force to drive flow (if > 0, overrides u_max init).

    Returns:
        ScalarSimState with history.
    """
    device = torch.device(device)
    if x_center is None:
        x_center = nx // 4

    # Initialize flow
    f, solid_mask = init_channel_flow(ny, nx, u_max, tau_f, device)

    # Initialize scalar
    rho, ux, uy = f_macroscopic(f)
    g = torch.zeros(5, ny, nx, device=device, dtype=torch.float32)
    g = init_scalar_line(g, ux, uy, x_center, sigma, total_mass)

    state = ScalarSimState(
        f=f, g=g, solid_mask=solid_mask,
        tau_f=tau_f, tau_s=tau_s,
        initial_mass=total_mass,
    )

    for step in range(n_steps):
        state = run_scalar_step(state, force_x=force_x,
                                conserve_mass=conserve_mass,
                                enforce_nonneg=enforce_nonneg)

        if (step + 1) % record_interval == 0 or step == 0:
            c = scalar_macroscopic(state.g)
            mom = second_moment_x(c)
            state.history.append({
                "step": step + 1,
                "time": float(step + 1),
                **mom,
            })

    return state


def run_pure_diffusion(
    nx: int, ny: int,
    tau_s: float,
    n_steps: int,
    sigma0: float = 5.0,
    x_center: int | None = None,
    y_center: int | None = None,
    total_mass: float = 1.0,
    device: torch.device | str = "cpu",
    record_interval: int = 10,
) -> ScalarSimState:
    """Pure diffusion test: zero velocity, Gaussian blob.

    Expected: σ²(t) = σ²₀ + 2Dt.
    """
    device = torch.device(device)
    if x_center is None:
        x_center = nx // 2
    if y_center is None:
        y_center = ny // 2

    ux = torch.zeros(ny, nx, device=device)
    uy = torch.zeros(ny, nx, device=device)
    g = torch.zeros(5, ny, nx, device=device, dtype=torch.float32)
    g = init_scalar_pulse(g, ux, uy, x_center, y_center, sigma0, total_mass)

    f = f_equilibrium(torch.ones(ny, nx, device=device), ux, uy)
    solid_mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)

    state = ScalarSimState(
        f=f, g=g, solid_mask=solid_mask,
        tau_f=1.0, tau_s=tau_s,
        initial_mass=total_mass,
    )

    for step in range(n_steps):
        # Scalar only (no flow)
        c = scalar_macroscopic(state.g)
        state.g = scalar_collide_bgk(state.g, c, ux, uy, tau_s)
        state.g = scalar_stream(state.g)
        state.step += 1

        if (step + 1) % record_interval == 0 or step == 0:
            c_now = scalar_macroscopic(state.g)
            mom = second_moment_x(c_now)
            state.history.append({
                "step": step + 1,
                "time": float(step + 1),
                **mom,
            })
            state.min_concentration = min(state.min_concentration,
                                          float(c_now.min().item()))

    return state


def run_pure_convection(
    nx: int, ny: int,
    ux_val: float,
    tau_s: float,
    n_steps: int,
    sigma: float = 3.0,
    x_center: int | None = None,
    device: torch.device | str = "cpu",
    record_interval: int = 10,
) -> ScalarSimState:
    """Pure convection test: uniform flow, Gaussian blob.

    Expected: blob translates at speed ux, σ² grows only from D_num.
    """
    device = torch.device(device)
    if x_center is None:
        x_center = nx // 4

    ux = torch.full((ny, nx), ux_val, device=device)
    uy = torch.zeros(ny, nx, device=device)
    g = torch.zeros(5, ny, nx, device=device, dtype=torch.float32)
    g = init_scalar_pulse(g, ux, uy, x_center, ny // 2, sigma, 1.0)

    f = f_equilibrium(torch.ones(ny, nx, device=device), ux, uy)
    solid_mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)

    state = ScalarSimState(
        f=f, g=g, solid_mask=solid_mask,
        tau_f=1.0, tau_s=tau_s,
        initial_mass=1.0,
    )

    for step in range(n_steps):
        c = scalar_macroscopic(state.g)
        state.g = scalar_collide_bgk(state.g, c, ux, uy, tau_s)
        state.g = scalar_stream(state.g)
        state.step += 1

        if (step + 1) % record_interval == 0 or step == 0:
            c_now = scalar_macroscopic(state.g)
            mom = second_moment_x(c_now)
            state.history.append({
                "step": step + 1,
                "time": float(step + 1),
                **mom,
            })
            state.min_concentration = min(state.min_concentration,
                                          float(c_now.min().item()))

    return state


__all__ = [
    "ScalarSimState",
    "apply_body_force",
    "enforce_nonnegativity",
    "fit_diffusivity",
    "fit_early_slope",
    "init_channel_flow",
    "init_scalar_line",
    "init_scalar_pulse",
    "momentum_bounce_back",
    "poiseuille_velocity_profile",
    "project_mass_conservation",
    "run_poiseuille_dispersion",
    "run_pure_convection",
    "run_pure_diffusion",
    "run_scalar_step",
    "scalar_bounce_back",
    "scalar_collide_bgk",
    "scalar_diffusivity",
    "scalar_dirichlet",
    "scalar_equilibrium",
    "scalar_macroscopic",
    "scalar_stream",
    "second_moment_x",
    "taylor_aris_deff",
    "tau_from_diffusivity",
]
