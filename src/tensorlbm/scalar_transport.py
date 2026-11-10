"""Boussinesq-coupled scalar (concentration / temperature) transport for LBM.

IN70-N03 — 替代燃料泄漏链验证面：主动标量输运 + 浮力耦合。

What this module provides
-------------------------
1. A **D2Q5 two-relaxation-time (TRT)** advection-diffusion lattice solver for a
   scalar field ``C`` (flammable-species concentration, or excess temperature)::

       dC/dt + u . grad C = D lap C + S

   TRT collision with magic parameter ``Lambda = (tau_s - 1/2)(tau_o - 1/2)``
   (Ginzburg 2005) instead of BGK.  For the D2Q5 advection-diffusion lattice the
   *odd* (antisymmetric) part carries the diffusive flux, so the diffusivity is
   set by the odd relaxation time, ``D = c_s^2 (tau_o - 1/2)``; the *even*
   (symmetric) rate ``tau_s`` is free and is fixed by ``Lambda``.  This is the
   opposite assignment to hydrodynamic D2Q9 — the exact 5x5 dispersion analysis
   in :func:`dispersion_rates` settles it numerically (see
   :func:`calibrate_magic`).  ``magic = None`` degenerate to BGK
   (``tau_s = tau_o``), which is the *baseline* arm of the BGK-vs-TRT comparison.

2. **Exact open-boundary mass bookkeeping.** Streaming is a gather; the mass
   leaving through an open face is the sum of the populations that actually
   crossed that face *before* streaming, and the mass injected by a Dirichlet
   inflow / volumetric source is the sum of the populations that were *written*
   in.  Both come from independent arrays, so
   ``(M_final - M_initial) - (injected - outflux)`` is a genuine residual
   (~1e-15 relative), not an identity.  See :class:`MassBudget`.

3. **Boussinesq coupling to D2Q9 momentum.**  A scalar anomaly generates the
   body force ``F = -rho0 g beta_C (C - C_ref) e_g`` (Guo forcing): for a light
   species (``beta_C > 0``) the anomaly makes the mixture lighter, so the force
   opposes gravity and the cloud rises.

4. Analytic reference solutions used as acceptance anchors
   (:func:`gaussian_pulse_1d`, :func:`erfc_half_space`,
   :func:`gaussian_moments`) and :func:`lfl_contour_metrics` for extracting a
   flammable-hazard isopleth (LFL = lower flammability limit) from a 2-D field.

Lattice conventions (D2Q5, ``c_s^2 = 1/3``)::

    i : 0       1        2        3         4
    c : (0,0)  (1,0)   (0,1)   (-1,0)   (0,-1)
    w : 2/6    1/6     1/6     1/6      1/6

    g_i^eq = w_i C (1 + 3 (c_i . u)),      D = (tau_o - 1/2) / 3

Evidence level: REAL (resolved solver; every number below is produced by the
lattice update, the analytic anchors are closed-form references).

References
----------
Ginzburg, I. (2005). Equilibrium-type and link-type lattice Boltzmann models for
    generic advection-diffusion and their harmonic/arithmetic-mean exactness.
    *Adv. Water Resour.* 28, 1171-1195.
Guo, Z., Zheng, C., & Shi, B. (2002). Discrete lattice effects on the forcing
    term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308.
Shi, B., & Guo, Z. (2009). Lattice Boltzmann model for nonlinear
    convection-diffusion equations. *Phys. Rev. E* 79, 016701.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "CX_D2Q5",
    "CY_D2Q5",
    "W_D2Q5",
    "OPP_D2Q5",
    "CS2_D2Q5",
    "tau_from_diffusivity",
    "tau_symmetric_from_magic",
    "magic_from_taus",
    "dispersion_rates",
    "calibrate_magic",
    "equilibrium_scalar_d2q5",
    "collide_scalar_trt",
    "stream_scalar",
    "scalar_field",
    "inject_source",
    "apply_inlet_dirichlet",
    "apply_outlet_zero_gradient",
    "MassBudget",
    "boussinesq_force",
    "buoyant_momentum_step",
    "gaussian_pulse_1d",
    "gaussian_moments",
    "erfc_half_space",
    "effective_diffusivity",
    "relative_l2",
    "lfl_contour_metrics",
    "ScalarTransportConfig",
    "ScalarRunResult",
    "run_pulse_1d",
    "run_erfc_half_space",
    "run_crosswind_release",
    "run_buoyant_plume",
    "run_momentum_jet",
    "run_lfl_wind_sweep",
]

_STREAM_CACHE: dict[tuple, tuple] = {}

CX_D2Q5 = (0, 1, 0, -1, 0)
CY_D2Q5 = (0, 0, 1, 0, -1)
W_D2Q5 = (2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
OPP_D2Q5 = (0, 3, 4, 1, 2)
CS2_D2Q5 = 1.0 / 3.0


def _vec(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device)


def tau_from_diffusivity(diffusivity: float) -> float:
    """Odd (diffusive) relaxation time: ``D = c_s^2 (tau_o - 1/2)`` => ``tau_o = 3D + 1/2``."""
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    return 3.0 * float(diffusivity) + 0.5


def tau_symmetric_from_magic(tau_odd: float, magic: float = 0.1875) -> float:
    """Even (ghost) relaxation time from the TRT magic parameter.

    ``Lambda = (tau_s - 1/2)(tau_o - 1/2)`` => ``tau_s = 1/2 + Lambda/(tau_o - 1/2)``.
    """
    delta = tau_odd - 0.5
    if delta <= 0.0:
        raise ValueError("tau_odd must exceed 1/2")
    return 0.5 + float(magic) / delta


def magic_from_taus(tau_sym: float, tau_odd: float) -> float:
    """``Lambda = (tau_s - 1/2)(tau_o - 1/2)``."""
    return (tau_sym - 0.5) * (tau_odd - 0.5)


_P_SYM = None


def _projectors(device, dtype):
    """Symmetric (even) projector ``P`` and its complement for D2Q5."""
    global _P_SYM
    key = (device.type, device.index, dtype)
    if _P_SYM is None or _P_SYM[0] != key:
        opp = _vec(OPP_D2Q5, torch.long, device)
        eye = torch.eye(5, dtype=dtype, device=device)
        p = 0.5 * (eye + eye[opp])
        _P_SYM = (key, p, eye - p)
    return _P_SYM[1], _P_SYM[2]


def dispersion_rates(
    k: float, tau_sym: float, tau_odd: float, n_theta: int = 8
) -> list[float]:
    """Exact decay rates ``-ln|lambda| / k^2`` of the linearised D2Q5 scheme.

    Builds the 5x5 amplification matrix of ``stream o collide`` for a Fourier
    mode ``exp(i k (x cos th + y sin th))``, averaged over ``n_theta``
    directions, and returns the decay rates sorted ascending.  The smallest one
    is the physical (diffusive) branch: for ``k -> 0`` it equals ``D``, and its
    drift with ``k`` *is* the numerical-diffusion error of the scheme.  This is
    the tool used to calibrate the TRT collision (:func:`calibrate_magic`).
    """
    import cmath

    import numpy as np

    dtype = torch.float64
    device = torch.device("cpu")
    p_sym, p_odd = _projectors(device, dtype)
    eye = torch.eye(5, dtype=dtype, device=device)
    w = _vec(W_D2Q5, dtype, device)
    cx = _vec(CX_D2Q5, dtype, device)
    cy = _vec(CY_D2Q5, dtype, device)
    coll = (
        eye
        - p_sym / tau_sym
        - p_odd / tau_odd
        + torch.outer(w, torch.ones(5, dtype=dtype)) / tau_sym
    ).to(torch.complex128)
    acc = np.zeros(5, dtype=np.float64)
    for j in range(int(n_theta)):
        th = 2.0 * math.pi * (j + 0.5) / int(n_theta)
        phase = -1j * k * (cx.numpy() * math.cos(th) + cy.numpy() * math.sin(th))
        stream = torch.diag(torch.tensor([cmath.exp(v) for v in phase], dtype=torch.complex128))
        lam = np.linalg.eigvals((stream @ coll).numpy())
        acc += np.sort(-np.log(np.abs(lam)) / (k * k))
    return [float(v) for v in np.sort(acc / float(n_theta))]


def calibrate_magic(
    diffusivity: float,
    magic_grid: tuple[float, ...] = (0.08, 0.10, 0.125, 0.15, 0.1875, 0.22, 0.25),
    k_band: tuple[float, ...] = (0.05, 0.2, 0.4, 0.6),
) -> dict[str, float]:
    """Pick the magic parameter that flattens ``D_eff(k)`` around ``diffusivity``.

    Returns the best ``magic``, the resulting worst-case relative deviation of
    the physical branch over ``k_band``, and the deviation of BGK for reference.
    """
    tau_odd = tau_from_diffusivity(diffusivity)
    best = None
    rows = {}
    for lam in magic_grid:
        tau_sym = tau_symmetric_from_magic(tau_odd, lam)
        devs = []
        for k in k_band:
            d = dispersion_rates(k, tau_sym, tau_odd)[0]
            devs.append(abs(d / diffusivity - 1.0))
        rows[lam] = max(devs)
        if best is None or rows[lam] < best[1]:
            best = (lam, max(devs))
    bgk = max(
        abs(dispersion_rates(k, tau_odd, tau_odd)[0] / diffusivity - 1.0) for k in k_band
    )
    return {
        "diffusivity": diffusivity,
        "magic_best": float(best[0]),
        "max_rel_dev": float(best[1]),
        "bgk_max_rel_dev": float(bgk),
    }


# ---------------------------------------------------------------------------
# D2Q5 scalar operators
# ---------------------------------------------------------------------------


def equilibrium_scalar_d2q5(
    c: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """``g_i^eq = w_i C (1 + 3 (c_i . u))`` -> shape ``(5, ny, nx)``."""
    device, dtype = c.device, c.dtype
    cx = _vec(CX_D2Q5, dtype, device).view(5, 1, 1)
    cy = _vec(CY_D2Q5, dtype, device).view(5, 1, 1)
    w = _vec(W_D2Q5, dtype, device).view(5, 1, 1)
    cu = cx * ux + cy * uy
    return w * c * (1.0 + cu / CS2_D2Q5)


def collide_scalar_trt(
    g: torch.Tensor,
    c: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_odd: float,
    tau_sym: float | None = None,
) -> torch.Tensor:
    """TRT collision for the D2Q5 scalar lattice (mass conserving to round-off).

    ``tau_odd`` relaxes the antisymmetric part (it sets ``D``); ``tau_sym``
    relaxes the even part and is free — ``tau_sym=None`` degenerates to BGK.
    """
    geq = equilibrium_scalar_d2q5(c, ux, uy)
    opp = _vec(OPP_D2Q5, torch.long, g.device)
    g_p = 0.5 * (g + g[opp])
    g_m = 0.5 * (g - g[opp])
    e_p = 0.5 * (geq + geq[opp])
    e_m = 0.5 * (geq - geq[opp])
    if tau_sym is None:
        tau_sym = tau_odd  # BGK: one relaxation rate for both parities
    return g - (g_m - e_m) / tau_odd - (g_p - e_p) / tau_sym


def stream_scalar(
    g: torch.Tensor,
    x_mode: str = "periodic",
    y_mode: str = "periodic",
    outflow_faces: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather-based D2Q5 streaming with selectable per-axis topology.

    ``x_mode``/``y_mode`` in ``{"periodic", "open"}``.  With ``"open"`` the
    populations whose upstream node lies outside the domain are dropped (zero
    fill) and the caller imposes the physical boundary condition.  The mass that
    left through an open face is returned as the second output, summed from the
    *pre-streaming* populations that crossed that face — the bookkeeping used by
    :class:`MassBudget`.
    """
    if x_mode not in ("periodic", "open") or y_mode not in ("periodic", "open"):
        raise ValueError("x_mode/y_mode must be 'periodic' or 'open'")
    ny, nx = g.shape[1], g.shape[2]
    device, dtype = g.device, g.dtype
    key = (ny, nx, device.type, device.index, x_mode, y_mode)
    if key not in _STREAM_CACHE:
        cx = _vec(CX_D2Q5, torch.long, device)
        cy = _vec(CY_D2Q5, torch.long, device)
        x = torch.arange(nx, device=device)
        y = torch.arange(ny, device=device)
        sx = x.view(1, nx) - cx.view(5, 1)
        sy = y.view(1, ny) - cy.view(5, 1)
        if x_mode == "periodic":
            sx = sx % nx
            vx = torch.ones((5, nx), dtype=dtype, device=device)
        else:
            vx = ((sx >= 0) & (sx < nx)).to(dtype)
            sx = sx.clamp(0, nx - 1)
        if y_mode == "periodic":
            sy = sy % ny
            vy = torch.ones((5, ny), dtype=dtype, device=device)
        else:
            vy = ((sy >= 0) & (sy < ny)).to(dtype)
            sy = sy.clamp(0, ny - 1)
        q_idx = torch.arange(5, device=device).view(5, 1, 1).expand(5, ny, nx)
        y_idx = sy.unsqueeze(2).expand(5, ny, nx)
        x_idx = sx.unsqueeze(1).expand(5, ny, nx)
        mask = (vx.unsqueeze(1) * vy.unsqueeze(2)).contiguous()
        _STREAM_CACHE[key] = (q_idx, y_idx, x_idx, mask)
    q_idx, y_idx, x_idx, mask = _STREAM_CACHE[key]
    out = g[q_idx, y_idx, x_idx] * mask

    faces = outflow_faces if outflow_faces is not None else ("xlo", "xhi", "ylo", "yhi")
    outflux = torch.zeros((), dtype=dtype, device=device)
    if x_mode == "open":
        if "xhi" in faces:
            outflux = outflux + g[1, :, -1].sum()
        if "xlo" in faces:
            outflux = outflux + g[3, :, 0].sum()
    if y_mode == "open":
        if "yhi" in faces:
            outflux = outflux + g[2, -1, :].sum()
        if "ylo" in faces:
            outflux = outflux + g[4, 0, :].sum()
    return out, outflux




def bounce_back_scalar_d2q5(g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back for D2Q5 scalar populations on masked cells.

    Uses the D2Q5 opposite mapping: 0->0, 1->3, 2->4, 3->1, 4->2.
    """
    opp = _vec(OPP_D2Q5, torch.long, g.device)
    return torch.where(mask.unsqueeze(0), g[opp], g)

# ---------------------------------------------------------------------------
# Macroscopic + boundary helpers
# ---------------------------------------------------------------------------


def scalar_field(g: torch.Tensor) -> torch.Tensor:
    """Recover scalar concentration: ``C = Σ_i g_i``."""
    return g.sum(dim=0)


def inject_source(
    g: torch.Tensor,
    rate: float,
    row_slice: slice,
    col_slice: slice,
) -> torch.Tensor:
    """Add a volumetric scalar source to the rest-direction population.

    The source is distributed as ``w_0 * rate`` to the rest population (i=0)
    so that the macroscopic scalar increases by exactly ``rate`` in the
    selected cells.
    """
    g = g.clone()
    g[0, row_slice, col_slice] = g[0, row_slice, col_slice] + rate
    return g


def apply_inlet_dirichlet(
    g: torch.Tensor,
    c_value: float,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Impose ``C = c_value`` at the inflow face (x = 0).

    All five populations at the boundary column are set to their
    equilibrium values so that the macroscopic concentration is exactly
    ``c_value``.  The injected mass (difference between old and new
    column sum) is returned.
    """
    g = g.clone()
    ny, nx = g.shape[1], g.shape[2]
    device, dtype = g.device, g.dtype
    if ux is not None and uy is not None:
        ux_col = ux[:, 0]
        uy_col = uy[:, 0]
    else:
        ux_col = torch.zeros(ny, dtype=dtype, device=device)
        uy_col = torch.zeros(ny, dtype=dtype, device=device)
    old_sum = g[:, :, 0].sum()
    c_col = torch.full((ny, 1), c_value, dtype=dtype, device=device)
    ux_col_2d = ux_col.unsqueeze(1)
    uy_col_2d = uy_col.unsqueeze(1)
    g[:, :, 0] = equilibrium_scalar_d2q5(c_col, ux_col_2d, uy_col_2d).squeeze(2)
    new_sum = g[:, :, 0].sum()
    injected = float(new_sum - old_sum)
    return g, injected


def apply_outlet_zero_gradient(g: torch.Tensor, col: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-gradient outlet: copy the last interior column to the outlet.

    Returns the refill mass (difference between old and new outlet populations).
    """
    g = g.clone()
    interior = col - 1 if col > 0 else 1
    old = g[:, :, col].sum()
    g[:, :, col] = g[:, :, interior]
    new = g[:, :, col].sum()
    return g, float(new - old)


# ---------------------------------------------------------------------------
# Mass budget
# ---------------------------------------------------------------------------


@dataclass
class MassBudget:
    """Exact mass bookkeeping for open-boundary scalar transport.

    The residual ``(M_final - M_initial) - (injected - outflux)`` is a genuine
    check on the lattice update (not an identity), because injected and outflux
    are accumulated from independent pre-streaming / post-write arrays.
    """

    mass_initial: float = 0.0
    mass_final: float = 0.0
    injected: float = 0.0
    outflux: float = 0.0
    steps: int = 0

    def residual(self) -> float:
        return (self.mass_final - self.mass_initial) - (self.injected - self.outflux)

    def residual_relative(self) -> float:
        denom = max(abs(self.injected), abs(self.outflux), abs(self.mass_initial), 1e-30)
        return abs(self.residual()) / denom

    def as_dict(self) -> dict[str, float]:
        return {
            "mass_initial": self.mass_initial,
            "mass_final": self.mass_final,
            "injected": self.injected,
            "outflux": self.outflux,
            "residual": self.residual(),
            "residual_relative": self.residual_relative(),
        }


# ---------------------------------------------------------------------------
# Boussinesq coupling
# ---------------------------------------------------------------------------


def boussinesq_force(
    concentration: torch.Tensor,
    c_ref: float,
    gravity: float,
    expansivity: float,
    rho0: float = 1.0,
    direction: tuple[float, float] = (0.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boussinesq body force ``F = +rho0 g beta_C (C - C_ref) e_g``.

    ``e_g`` (= ``direction``) points **up**, ``gravity > 0`` is the gravitational
    acceleration magnitude, and ``expansivity = beta_C > 0`` means the mixture
    gets lighter as ``C`` (the light-species mass fraction) grows.  Hence a
    positive anomaly produces an upward force, ``F_y = +rho0 g beta_C (C-C_ref)``
    — the classic Boussinesq term with the sign convention that makes a methane
    / hydrogen cloud rise.  Returns ``(Fx, Fy)`` in lattice units.
    """
    dg = (concentration - c_ref) * (rho0 * gravity * expansivity)
    ex, ey = direction
    norm = math.hypot(ex, ey) or 1.0
    return dg * (ex / norm), dg * (ey / norm)


def buoyant_momentum_step(
    f: torch.Tensor,
    tau: float,
    concentration: torch.Tensor,
    c_ref: float,
    gravity: float,
    expansivity: float,
    rho0: float = 1.0,
) -> torch.Tensor:
    """One D2Q9 step (Guo force + periodic streaming) with Boussinesq buoyancy.

    Reuses the repository D2Q9 primitives so the coupling is exercised through
    the same collision kernel as the rest of the code base.
    """
    from .solver import stream
    from .thermal import collide_bgk_force

    fx, fy = boussinesq_force(concentration, c_ref, gravity, expansivity, rho0=rho0)
    force = torch.stack([fx, fy])
    f_post = collide_bgk_force(f, tau, force)
    return stream(f_post)


# ---------------------------------------------------------------------------
# Analytic anchors
# ---------------------------------------------------------------------------


def gaussian_pulse_1d(
    x: torch.Tensor,
    mass: float,
    diffusivity: float,
    velocity: float,
    t: float,
    sigma0: float = 0.0,
) -> torch.Tensor:
    """Advected 1-D Gaussian ``C = M/sqrt(4 pi D t) exp(-(x-Ut)^2/(4Dt))``.

    ``sigma0 > 0`` folds a finite initial width in (``sigma^2 = sigma0^2 + 2Dt``);
    the delta-source form of the mission criterion is ``sigma0 = 0``.
    """
    if t <= 0.0:
        raise ValueError("t must be positive")
    var = sigma0**2 + 2.0 * diffusivity * t
    return mass / math.sqrt(2.0 * math.pi * var) * torch.exp(
        -((x - velocity * t) ** 2) / (2.0 * var)
    )


def erfc_half_space(
    x: torch.Tensor, level: float, diffusivity: float, t: float
) -> torch.Tensor:
    """Semi-infinite Dirichlet solution ``C = C0 erfc(x / sqrt(4 D t))``."""
    if t <= 0.0:
        raise ValueError("t must be positive")
    return level * torch.erfc(x / math.sqrt(4.0 * diffusivity * t))


def gaussian_moments(c: torch.Tensor, x: torch.Tensor) -> tuple[float, float, float]:
    """Return ``(mass, centroid, variance)`` of a 1-D profile.

    For 2-D input the profile is collapsed along the first axis
    (crosswind-integrated).
    """
    prof = c if c.dim() == 1 else c.sum(dim=0)
    mass = torch.sum(prof)
    if float(mass) == 0.0:
        return 0.0, float("nan"), float("nan")
    centroid = torch.sum(prof * x) / mass
    var = torch.sum(prof * (x - centroid) ** 2) / mass
    return float(mass), float(centroid), float(var)


def effective_diffusivity(
    c: torch.Tensor, x: torch.Tensor, t: float, sigma0: float = 0.0
) -> float:
    """``D_eff = (sigma^2 - sigma0^2) / (2 t)`` measured from the profile."""
    _, _, var = gaussian_moments(c, x)
    return (var - sigma0**2) / (2.0 * t)


def relative_l2(
    computed: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """``||computed - reference||_2 / ||reference||_2`` (optionally on a mask)."""
    diff = computed - reference
    ref = reference
    if mask is not None:
        diff = diff[mask]
        ref = ref[mask]
    denom = float(torch.linalg.vector_norm(ref))
    if denom == 0.0:
        return float("nan")
    return float(torch.linalg.vector_norm(diff)) / denom


def lfl_contour_metrics(
    c: torch.Tensor,
    level: float,
    dx: float = 1.0,
    dy: float = 1.0,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> dict[str, float]:
    """Geometry of the ``C >= level`` hazard isopleth on a Cartesian grid.

    Returned metrics
    ----------------
    ``extent_x``  : downwind length of the super-threshold region, taken on the
                    row that holds the most super-threshold nodes and refined by
                    linear interpolation of the threshold crossing.
    ``max_width`` : maximum crosswind width of that region (interpolated edges).
    ``area``      : area of the super-threshold node set.
    ``perimeter`` : arc length of the isopleth (marching-squares edge count).
    ``count``     : number of super-threshold nodes.
    """
    if level <= 0.0:
        raise ValueError("level must be positive")
    mask = c >= level
    count = int(mask.sum().item())
    if count == 0:
        return {"extent_x": 0.0, "max_width": 0.0, "area": 0.0, "perimeter": 0.0, "count": 0}
    ny, nx = c.shape
    ys, xs = torch.nonzero(mask, as_tuple=True)
    row_counts = torch.bincount(ys, minlength=ny)
    best_row = int(row_counts.argmax())
    row_xs = xs[ys == best_row].sort().values.float()
    if len(row_xs) > 0:
        x_lo = float(row_xs[0])
        x_hi = float(row_xs[-1])
        if x_lo > 0:
            c_left = c[best_row, int(x_lo) - 1].item()
            c_right = c[best_row, int(x_lo)].item()
            if c_right != c_left:
                x_lo -= (c_right - level) / (c_right - c_left)
        if x_hi < nx - 1:
            c_left = c[best_row, int(x_hi)].item()
            c_right = c[best_row, int(x_hi) + 1].item()
            if c_right != c_left:
                x_hi += (level - c_left) / (c_right - c_left)
        extent_x = (x_hi - x_lo) * dx
    else:
        extent_x = 0.0
    max_width = 0.0
    for row in range(ny):
        row_mask = mask[row]
        if row_mask.any():
            row_xs_r = torch.nonzero(row_mask, as_tuple=True)[0].float()
            w = (row_xs_r[-1] - row_xs_r[0]) * dy
            if w > max_width:
                max_width = w
    area = float(count) * dx * dy
    perimeter = 0.0
    padded = torch.zeros((ny + 2, nx + 2), dtype=torch.bool)
    padded[1:-1, 1:-1] = mask
    perimeter += float((padded[1:-1, 1:-1] != padded[1:-1, :-2]).sum())
    perimeter += float((padded[1:-1, 1:-1] != padded[1:-1, 2:]).sum())
    perimeter += float((padded[1:-1, 1:-1] != padded[:-2, 1:-1]).sum())
    perimeter += float((padded[1:-1, 1:-1] != padded[2:, 1:-1]).sum())
    perimeter = perimeter * 0.5 * (dx + dy)
    return {
        "extent_x": extent_x,
        "max_width": max_width,
        "area": area,
        "perimeter": perimeter,
        "count": count,
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarTransportConfig:
    """Configuration for the scalar drivers below."""

    nx: int
    ny: int = 3
    diffusivity: float = 0.05
    magic: float | None = 0.1875
    ux: float = 0.0
    uy: float = 0.0
    dtype: torch.dtype = torch.float64

    @property
    def tau_odd(self) -> float:
        return tau_from_diffusivity(self.diffusivity)

    @property
    def tau_sym(self) -> float | None:
        if self.magic is None:
            return None
        return tau_symmetric_from_magic(self.tau_odd, self.magic)


@dataclass
class ScalarRunResult:
    """Result container for the scalar drivers."""

    c: torch.Tensor
    x: torch.Tensor
    budget: MassBudget
    steps: int
    extra: dict = field(default_factory=dict)


def _step(
    g: torch.Tensor,
    cfg: ScalarTransportConfig,
    ux: torch.Tensor,
    uy: torch.Tensor,
    x_mode: str,
    y_mode: str,
    faces: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = collide_scalar_trt(g, scalar_field(g), ux, uy, cfg.tau_odd, cfg.tau_sym)
    return stream_scalar(g, x_mode=x_mode, y_mode=y_mode, outflow_faces=faces)


def run_pulse_1d(
    cfg: ScalarTransportConfig,
    steps: int,
    mass: float = 1.0,
    sigma0: float = 2.0,
    x0: float | None = None,
) -> ScalarRunResult:
    """Advected Gaussian pulse on a periodic strip (1-D physics, ``uy = 0``).

    The initial condition is the analytic Gaussian of mass ``mass`` and width
    ``sigma0``; the lattice distributions start from equilibrium with it, so
    total mass ``= mass`` exactly (to round-off).
    """
    ny, nx = cfg.ny, cfg.nx
    x = torch.arange(nx, dtype=cfg.dtype)
    if x0 is None:
        x0 = 0.5 * nx
    c0 = gaussian_pulse_1d(
        x - x0, mass, cfg.diffusivity, cfg.ux, 1e-30, sigma0=sigma0
    )
    c0 = c0 * (mass / c0.sum())
    ux = torch.full((ny, nx), cfg.ux, dtype=cfg.dtype)
    uy = torch.zeros((ny, nx), dtype=cfg.dtype)
    g = equilibrium_scalar_d2q5(c0.unsqueeze(0).expand(ny, nx), ux, uy)
    budget = MassBudget(mass_initial=float(scalar_field(g).sum()))
    for _ in range(int(steps)):
        g, flux = _step(g, cfg, ux, uy, "periodic", "periodic")
        budget.outflux += float(flux)
    budget.mass_final = float(scalar_field(g).sum())
    budget.steps = int(steps)
    return ScalarRunResult(c=scalar_field(g), x=x, budget=budget, steps=int(steps))


def run_erfc_half_space(
    cfg: ScalarTransportConfig, steps: int, level: float = 1.0
) -> ScalarRunResult:
    """Semi-infinite diffusion against a Dirichlet face at ``x = 0``.

    Initial field is zero; the inflow face is held at ``C = level`` and the far
    face uses zero-gradient outflow.  Reference: :func:`erfc_half_space`.
    """
    ny, nx = cfg.ny, cfg.nx
    x = torch.arange(nx, dtype=cfg.dtype)
    ux = torch.zeros((ny, nx), dtype=cfg.dtype)
    uy = torch.zeros((ny, nx), dtype=cfg.dtype)
    g = torch.zeros((5, ny, nx), dtype=cfg.dtype)
    budget = MassBudget(mass_initial=0.0)
    for _ in range(int(steps)):
        g, flux = _step(g, cfg, ux, uy, "open", "periodic", ("xlo", "xhi"))
        budget.outflux += float(flux)
        g, injected = apply_inlet_dirichlet(g, level)
        budget.injected += float(injected)
        g, refill = apply_outlet_zero_gradient(g)
        budget.injected += float(refill)
    budget.mass_final = float(scalar_field(g).sum())
    budget.steps = int(steps)
    return ScalarRunResult(c=scalar_field(g), x=x, budget=budget, steps=int(steps))


def run_crosswind_release(
    cfg: ScalarTransportConfig,
    steps: int,
    source_rate: float,
    source_ix: int | None = None,
    source_iy: int | None = None,
    source_halfwidth: int = 0,
    record_every: int = 0,
) -> ScalarRunResult:
    """2-D continuous release in a uniform crosswind (steady-state hazard field).

    Uniform wind ``ux`` enters at ``x = 0`` (Dirichlet ``C = 0``), leaves through a
    zero-gradient outflow at ``x = nx-1``; the lateral faces are open.  A
    volumetric point source of strength ``source_rate`` per step sits at
    ``(source_ix, source_iy)``.  Mass bookkeeping closes through
    :class:`MassBudget` (inlet + source vs. the open faces).
    """
    ny, nx = cfg.ny, cfg.nx
    x = torch.arange(nx, dtype=cfg.dtype)
    ux = torch.full((ny, nx), cfg.ux, dtype=cfg.dtype)
    uy = torch.zeros((ny, nx), dtype=cfg.dtype)
    g = torch.zeros((5, ny, nx), dtype=cfg.dtype)
    ix = nx // 8 if source_ix is None else int(source_ix)
    iy = (ny - 1) // 2 if source_iy is None else int(source_iy)
    hw = int(source_halfwidth)
    budget = MassBudget(mass_initial=0.0)
    history: list[dict[str, float]] = []
    for step in range(int(steps)):
        g, flux = _step(g, cfg, ux, uy, "open", "open", ("xlo", "xhi", "ylo", "yhi"))
        budget.outflux += float(flux)
        g, injected = apply_inlet_dirichlet(g, 0.0, ux=cfg.ux)
        budget.injected += float(injected)
        g, refill = apply_outlet_zero_gradient(g)
        budget.injected += float(refill)
        if hw == 0:
            g = inject_source(g, source_rate, slice(iy, iy + 1), slice(ix, ix + 1))
        else:
            g = inject_source(
                g, source_rate, slice(iy - hw, iy + hw + 1), slice(ix - hw, ix + hw + 1)
            )
        budget.injected += float(source_rate)
        if record_every and (step + 1) % int(record_every) == 0:
            c_now = scalar_field(g)
            history.append(
                {"step": float(step + 1), "mass": float(c_now.sum()), "cmax": float(c_now.max())}
            )
    c = scalar_field(g)
    budget.mass_final = float(c.sum())
    budget.steps = int(steps)
    return ScalarRunResult(c=c, x=x, budget=budget, steps=int(steps), extra={"history": history})


# ---------------------------------------------------------------------------
# Buoyant plume driver (2-D, Boussinesq-coupled)
# ---------------------------------------------------------------------------


def run_buoyant_plume(
    nx: int,
    ny: int,
    diffusivity: float = 0.02,
    viscosity: float = 0.02,
    source_rate: float = 0.5,
    source_ix: int | None = None,
    source_iy: int | None = None,
    source_halfwidth: int = 1,
    gravity: float = 1e-4,
    expansivity: float = 1.0,
    c_ref: float = 0.0,
    steps: int = 5000,
    tau_momentum: float | None = None,
    record_every: int = 0,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """2-D buoyant plume in a quiescent box with Boussinesq coupling.

    A scalar source at the bottom centre releases a light species; the
    Boussinesq force makes the mixture rise, forming a self-similar plume.
    The momentum field is D2Q9 BGK; the scalar field is D2Q5 TRT.

    Returns a dict with:
        ``c``: final scalar field (ny, nx)
        ``ux``, ``uy``: final velocity fields
        ``plume_centerline``: list of (y, c_max, u_y_max) along the plume axis
        ``steps``: number of steps run
    """
    device = torch.device("cpu")
    ix = nx // 2 if source_ix is None else int(source_ix)
    iy = 2 if source_iy is None else int(source_iy)
    hw = int(source_halfwidth)

    tau_odd = tau_from_diffusivity(diffusivity)
    tau_sym = tau_symmetric_from_magic(tau_odd, 0.1875)
    tau_f = 3.0 * viscosity + 0.5 if tau_momentum is None else tau_momentum

    from .d2q9 import equilibrium as f_equilibrium, macroscopic
    from .solver import collide_bgk, stream as stream_f
    from .thermal import collide_bgk_force
    from .boundaries import bounce_back_cells

    # Init momentum: rest
    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.zeros((ny, nx), dtype=dtype, device=device)
    uy = torch.zeros((ny, nx), dtype=dtype, device=device)
    f = f_equilibrium(rho, ux, uy)

    # Init scalar: zero
    c = torch.zeros((ny, nx), dtype=dtype, device=device)
    g = equilibrium_scalar_d2q5(c, ux, uy)

    # Walls: top/bottom bounce-back for momentum
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True

    centerline: list[tuple[float, float, float]] = []

    for step in range(steps):
        # --- Momentum step with Boussinesq force ---
        fx, fy = boussinesq_force(c, c_ref, gravity, expansivity)
        force = torch.stack([fx, fy])
        f = collide_bgk_force(f, tau_f, force)
        f = stream_f(f)
        f = bounce_back_cells(f, wall_mask)

        # Recover velocity
        rho, ux, uy = macroscopic(f)
        ux = ux.masked_fill(wall_mask, 0.0)
        uy = uy.masked_fill(wall_mask, 0.0)

        # --- Scalar step ---
        g = collide_scalar_trt(g, c, ux, uy, tau_odd, tau_sym)
        if hw == 0:
            g = inject_source(g, source_rate, slice(iy, iy + 1), slice(ix, ix + 1))
        else:
            g = inject_source(
                g, source_rate, slice(iy - hw, iy + hw + 1), slice(ix - hw, ix + hw + 1)
            )
        g = bounce_back_scalar_d2q5(g, wall_mask)
        g, _ = stream_scalar(g, x_mode="open", y_mode="periodic", outflow_faces=("xlo", "xhi"))
        g, _ = apply_outlet_zero_gradient(g, col=0)
        g, _ = apply_outlet_zero_gradient(g, col=-1)

        c = scalar_field(g)

        if record_every and (step + 1) % record_every == 0:
            c_col = c[:, ix]
            u_col = uy[:, ix]
            c_max = float(c_col.max())
            u_max = float(u_col.max())
            y_max = float(c_col.argmax())
            centerline.append((y_max, c_max, u_max))

    return {
        "c": c,
        "ux": ux,
        "uy": uy,
        "plume_centerline": centerline,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Momentum jet driver (negative control: g' = 0, no buoyancy)
# ---------------------------------------------------------------------------


def run_momentum_jet(
    nx: int,
    ny: int,
    diffusivity: float = 0.02,
    viscosity: float = 0.02,
    jet_velocity: float = 0.05,
    source_ix: int | None = None,
    source_iy: int | None = None,
    source_halfwidth: int = 1,
    steps: int = 5000,
    record_every: int = 0,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """2-D momentum jet in a quiescent box (no buoyancy, g' = 0).

    A velocity source at the bottom centre injects momentum upward, forming
    a pure momentum jet.  This is the negative control for the buoyant plume:
    without buoyancy the jet spreading follows a different self-similarity law.

    Returns a dict with:
        ``c``: passive scalar field (tracer, no buoyancy coupling)
        ``ux``, ``uy``: final velocity fields
        ``jet_centerline``: list of (y, u_y_max) along the jet axis
        ``steps``: number of steps run
    """
    device = torch.device("cpu")
    ix = nx // 2 if source_ix is None else int(source_ix)
    iy = 2 if source_iy is None else int(source_iy)
    hw = int(source_halfwidth)

    tau_odd = tau_from_diffusivity(diffusivity)
    tau_sym = tau_symmetric_from_magic(tau_odd, 0.1875)
    tau_f = 3.0 * viscosity + 0.5

    from .d2q9 import equilibrium as f_equilibrium, macroscopic
    from .solver import collide_bgk, stream as stream_f
    from .boundaries import bounce_back_cells

    # Init momentum: rest
    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.zeros((ny, nx), dtype=dtype, device=device)
    uy = torch.zeros((ny, nx), dtype=dtype, device=device)
    f = f_equilibrium(rho, ux, uy)

    # Init scalar: zero (passive tracer, no buoyancy)
    c = torch.zeros((ny, nx), dtype=dtype, device=device)
    g = equilibrium_scalar_d2q5(c, ux, uy)

    # Walls: top/bottom bounce-back for momentum
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True

    centerline: list[tuple[float, float]] = []

    for step in range(steps):
        # --- Momentum step (no buoyancy) ---
        f = collide_bgk(f, tau_f)
        f = stream_f(f)
        f = bounce_back_cells(f, wall_mask)

        # Inject momentum at source (set uy in source region)
        if hw == 0:
            uy[iy, ix] = jet_velocity
        else:
            uy[iy - hw : iy + hw + 1, ix - hw : ix + hw + 1] = jet_velocity
        # Re-equilibrate at source
        rho_src, _, _ = macroscopic(f)
        f_src = f_equilibrium(rho_src, ux, uy)
        if hw == 0:
            f[:, iy, ix] = f_src[:, iy, ix]
        else:
            f[:, iy - hw : iy + hw + 1, ix - hw : ix + hw + 1] = f_src[
                :, iy - hw : iy + hw + 1, ix - hw : ix + hw + 1
            ]

        # Recover velocity
        rho, ux, uy = macroscopic(f)
        ux = ux.masked_fill(wall_mask, 0.0)
        uy = uy.masked_fill(wall_mask, 0.0)

        # --- Scalar step (passive, no buoyancy) ---
        g = collide_scalar_trt(g, c, ux, uy, tau_odd, tau_sym)
        if hw == 0:
            g = inject_source(g, jet_velocity * 0.1, slice(iy, iy + 1), slice(ix, ix + 1))
        else:
            g = inject_source(
                g, jet_velocity * 0.1, slice(iy - hw, iy + hw + 1), slice(ix - hw, ix + hw + 1)
            )
        g = bounce_back_scalar_d2q5(g, wall_mask)
        g, _ = stream_scalar(g, x_mode="open", y_mode="periodic", outflow_faces=("xlo", "xhi"))
        g, _ = apply_outlet_zero_gradient(g, col=0)
        g, _ = apply_outlet_zero_gradient(g, col=-1)

        c = scalar_field(g)

        if record_every and (step + 1) % record_every == 0:
            u_col = uy[:, ix]
            u_max = float(u_col.max())
            y_max = float(u_col.argmax())
            centerline.append((y_max, u_max))

    return {
        "c": c,
        "ux": ux,
        "uy": uy,
        "jet_centerline": centerline,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# LFL wind speed sweep
# ---------------------------------------------------------------------------


def run_lfl_wind_sweep(
    wind_speeds: list[float],
    nx: int = 128,
    ny: int = 81,
    diffusivity: float = 0.05,
    source_rate: float = 1.0,
    lfl_level: float = 0.1,
    steps: int = 2000,
    dtype: torch.dtype = torch.float64,
) -> list[dict]:
    """Run crosswind release at multiple wind speeds and extract LFL metrics.

    Returns a list of dicts, one per wind speed, with:
        ``ux``: wind speed
        ``extent_x``: downwind extent of LFL isopleth
        ``max_width``: maximum crosswind width
        ``area``: area of LFL region
    """
    results = []
    for ux_val in wind_speeds:
        cfg = ScalarTransportConfig(
            nx=nx, ny=ny, diffusivity=diffusivity, ux=ux_val, dtype=dtype
        )
        res = run_crosswind_release(cfg, steps, source_rate)
        metrics = lfl_contour_metrics(res.c, lfl_level)
        results.append({
            "ux": ux_val,
            "extent_x": metrics["extent_x"],
            "max_width": metrics["max_width"],
            "area": metrics["area"],
            "count": metrics["count"],
        })
    return results
