"""Common wall-function module — solver-agnostic wall correction.

This module extracts the wall-function mechanics from
:mod:`tensorlbm.wall_model` into a **common, solver-agnostic** interface
that can be combined with any collision operator or turbulence model.

The key design decision is that the wall function takes **pre-computed**
``u_tau`` and ``y_plus`` fields as input.  This decouples the wall
correction from how the wall quantities were computed (which could come
from a RANS model, LES closure, or any other turbulence model).  The
caller computes ``u_tau`` and ``y_plus`` using
:func:`compute_u_tau` / :func:`compute_y_plus` (or their own turbulence
model) and passes them to :func:`wall_function`.

Supported lattices: ``D3Q19``, ``D3Q27``.

The module does **not** modify any solver hot path.  It only provides
reusable wall-function mechanics that a solver may call from its own
boundary-condition step.
"""

from __future__ import annotations

import torch

SUPPORTED_LATTICES: tuple[str, ...] = ("D3Q19", "D3Q27")

# von Kármán constant and log-law offset (smooth wall).
_KAPPA = 0.41
_B_LOG = 5.0


def _validate_lattice(lattice: str) -> str:
    """Return *lattice* if supported, else raise ValueError."""
    if lattice not in SUPPORTED_LATTICES:
        raise ValueError(f"Unsupported lattice {lattice!r}; supported: {SUPPORTED_LATTICES}")
    return lattice


def _macroscopic(lattice: str, f: torch.Tensor):
    """Dispatch to the correct macroscopic function for *lattice*."""
    if lattice == "D3Q19":
        from .d3q19 import macroscopic3d

        return macroscopic3d(f)
    elif lattice == "D3Q27":
        from .d3q27 import macroscopic27

        return macroscopic27(f)
    raise ValueError(f"Unsupported lattice: {lattice!r}")


# ---------------------------------------------------------------------------
# Wall-quantity computation helpers (lattice-agnostic)
# ---------------------------------------------------------------------------


def compute_u_tau(
    u_mag: torch.Tensor,
    nu: float,
    y_val: float = 0.5,
    wall_law: str = "log",
) -> torch.Tensor:
    """Compute the friction velocity ``u_tau`` from velocity magnitude.

    This is a **lattice-agnostic** helper: it operates on a scalar velocity
    magnitude field and does not reference any lattice-specific functions.
    The caller is responsible for providing the correct near-wall velocity.

    Args:
        u_mag:    Velocity magnitude field ``(nz, ny, nx)``.
        nu:       Kinematic viscosity (lattice units).
        y_val:    Distance from the near-wall cell centre to the wall.
        wall_law: ``"log"`` (standard log-law, y+>30) or ``"reichardt"``
                  (Reichardt unified law, valid for all y+).

    Returns:
        Friction velocity field, same shape as *u_mag*.
    """
    u_mag = u_mag.clamp(min=1e-12)

    if wall_law == "reichardt":
        # Reichardt unified wall law (1951): valid for all y+.
        ut = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
        for _ in range(12):
            yp = (y_val * ut / nu).clamp(min=1e-6)
            up = (1.0 / _KAPPA) * torch.log1p(_KAPPA * yp) + 7.8 * (
                1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)
            )
            ut = (u_mag / up.clamp(min=1e-6)).clamp(min=1e-12)
        return ut

    if wall_law == "log":
        # Newton iteration for log-law: u = u_tau·(ln(y+)/κ + B)
        u_tau = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
        y_plus = y_val * u_tau / nu
        turb = y_plus > 11.6
        if bool(turb.any()):
            ut = u_tau[turb].clone()
            um = u_mag[turb]
            for _ in range(8):
                lyp = torch.log(y_val * ut / nu)
                fv = ut * (lyp / _KAPPA + _B_LOG) - um
                fp = (lyp / _KAPPA + _B_LOG) + 1.0 / _KAPPA
                ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
            u_tau[turb] = ut
        return u_tau

    if wall_law == "gradient":
        # Direct velocity-gradient method: τ_w = ν·u_mag / y_val
        # Bug 11 fix: correct formula is ν·u/y_val (NOT 2ν·u/y_val).
        # For half-way BB with y_val=0.5: τ_w = ν·u/0.5 = 2ν·u (correct).
        # The old 2ν·u/y_val gave 4ν·u (2× too strong).
        # No log-law assumption; works for all y+ including buffer/viscous sublayer.
        tau_w = nu * u_mag / y_val
        return torch.sqrt(tau_w.clamp(min=1e-30))

    if wall_law == "hybrid":
        # y+ threshold for switching: y+_thresh = 11.6 (viscous-log transition)
        # u_tau_vis = sqrt(nu * u / y)  (laminar shear, u+ = y+ for y+ < 5)
        # u_tau_log = Newton(log-law)    (for y+ >= 11.6)
        yp_thresh = 11.6
        u_tau_vis = torch.sqrt(nu * u_mag / max(y_val, 1e-10))
        y_plus = y_val * u_tau_vis / nu

        u_tau = u_tau_vis.clone()
        turb = (y_plus > yp_thresh) & (u_mag > 1e-10)
        if bool(turb.any()):
            from math import log as _log, exp as _exp

            u_tau_g = u_tau_vis[turb].clone()
            um = u_mag[turb]
            for _ in range(8):
                lyp = torch.log(y_val * u_tau_g / nu)
                fv = u_tau_g * (lyp / _KAPPA + _B_LOG) - um
                fp = (lyp / _KAPPA + _B_LOG) + 1.0 / _KAPPA
                u_tau_g = (u_tau_g - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
            u_tau[turb] = u_tau_g
        return u_tau

    raise ValueError(
        f"Unknown wall_law {wall_law!r}; supported: 'log', 'reichardt', 'gradient', 'hybrid'"
    )


def compute_y_plus(
    u_tau: torch.Tensor,
    nu: float,
    y_val: float = 0.5,
) -> torch.Tensor:
    """Compute the dimensionless wall distance ``y+`` from friction velocity.

    Args:
        u_tau: Friction velocity field ``(nz, ny, nx)``.
        nu:    Kinematic viscosity (lattice units).
        y_val: Distance from the near-wall cell centre to the wall.

    Returns:
        y+ field, same shape as *u_tau*.
    """
    return (y_val * u_tau / nu).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Near-wall mask computation
# ---------------------------------------------------------------------------


def _near_wall_mask(solid: torch.Tensor) -> torch.Tensor:
    """Identify fluid cells adjacent to solid cells (6-connected).

    Bug 9 fix: do NOT use ``torch.roll`` (which wraps periodically in z
    for 2-D extruded simulations).  Instead use interior-only slicing
    with one-sided boundary handling.
    """
    fluid = ~solid
    near = torch.zeros_like(solid)
    nz, ny, nx = solid.shape

    # x-direction (interior only, no periodic wrap)
    near[:, :, 1:-1] |= (solid[:, :, 2:] | solid[:, :, :-2]) & fluid[:, :, 1:-1]
    # y-direction
    near[:, 1:-1, :] |= (solid[:, 2:, :] | solid[:, :-2, :]) & fluid[:, 1:-1, :]
    # z-direction (no periodic wrap for 2-D simulations)
    if nz > 1:
        near[1:-1] |= (solid[2:] | solid[:-2]) & fluid[1:-1]
        near[0] |= solid[1] & fluid[0]
        near[-1] |= solid[-2] & fluid[-1]
    return near


def _compute_wall_normal(
    solid: torch.Tensor,
    near: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute outward wall normal (from solid toward fluid) at near-wall cells.

    Uses the gradient of the solid mask.  Returns zero outside near-wall.
    """
    nz, ny, nx = solid.shape
    sf = solid.to(torch.float32)

    gx = torch.zeros_like(sf)
    gy = torch.zeros_like(sf)
    gz = torch.zeros_like(sf)

    gx[:, :, 1:-1] = (sf[:, :, 2:] - sf[:, :, :-2]) * 0.5
    gx[:, :, 0] = sf[:, :, 1] - sf[:, :, 0]
    gx[:, :, -1] = sf[:, :, -1] - sf[:, :, -2]

    gy[:, 1:-1, :] = (sf[:, 2:, :] - sf[:, :-2, :]) * 0.5
    gy[:, 0, :] = sf[:, 1, :] - sf[:, 0, :]
    gy[:, -1, :] = sf[:, -1, :] - sf[:, -2, :]

    if nz > 1:
        gz[1:-1] = (sf[2:] - sf[:-2]) * 0.5
        gz[0] = sf[1] - sf[0]
        gz[-1] = sf[-1] - sf[-2]

    nx_n = -gx
    ny_n = -gy
    nz_n = -gz

    mag = torch.sqrt(nx_n * nx_n + ny_n * ny_n + nz_n * nz_n)
    has_normal = mag > 1e-10
    inv_mag = torch.where(has_normal, 1.0 / mag, torch.zeros_like(mag))
    near_f = near.to(torch.float32)
    return nx_n * inv_mag * near_f, ny_n * inv_mag * near_f, nz_n * inv_mag * near_f


# --------------------------------------------------------------------------- #
# Public wall_function interface
# --------------------------------------------------------------------------- #
def _apply_body_force(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    lattice: str,
    *,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a Guo body-force correction to a 3-D distribution.

    This is a **lattice-agnostic** helper: it dispatches to the correct
    velocity vectors (``C``, ``W``) for D3Q19 or D3Q27.  The Guo forcing
    term is ``w_i * 3 * (c_i · F)`` added to the distribution.

    If *ux*, *uy*, *uz* are provided, they are used directly instead of
    recomputing macroscopic fields from *f*.
    """
    if lattice == "D3Q19":
        from .d3q19 import C as C_LAT, W as W_LAT

        q = 19
    elif lattice == "D3Q27":
        from .d3q27 import C as C_LAT, W as W_LAT

        q = 27
    else:
        raise ValueError(f"Unsupported lattice: {lattice!r}")

    device = f.device
    c = C_LAT.to(device).float()
    w = W_LAT.to(device).float()
    cx = c[:, 0].view(q, 1, 1, 1)
    cy = c[:, 1].view(q, 1, 1, 1)
    cz = c[:, 2].view(q, 1, 1, 1)
    w_view = w.view(q, 1, 1, 1)

    # Full Guo forcing: w_i * (1 + c_i·u/c_s²) * (c_i·F) / c_s²
    # c_s² = 1/3 for both D3Q19 and D3Q27, so 1/c_s² = 3.
    # The (1 + c·u/cs²) velocity-correction term is essential for
    # correct force application at non-trivial velocities.
    cs2 = 1.0 / 3.0
    cu = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
    # Need velocity field for the correction term; use pre-computed if available.
    if ux is not None and uy is not None and uz is not None:
        _ux, _uy, _uz = ux, uy, uz
    else:
        if lattice == "D3Q19":
            from .d3q19 import macroscopic3d as _macro
        else:
            from .d3q27 import macroscopic27 as _macro
        _rho, _ux, _uy, _uz = _macro(f)
    cu_u = cx * _ux.unsqueeze(0) + cy * _uy.unsqueeze(0) + cz * _uz.unsqueeze(0)
    forcing = w_view * (1.0 + cu_u / cs2) * cu / cs2
    return f + forcing


# ---------------------------------------------------------------------------
# Public wall_function interface
# ---------------------------------------------------------------------------


def wall_function(
    f: torch.Tensor,
    mask: torch.Tensor,
    u_tau: torch.Tensor,
    y_plus: torch.Tensor,
    *,
    lattice: str = "D3Q19",
    nu: float = 0.02,
    y_val: float = 0.5,
    rho: torch.Tensor | None = None,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a wall-function correction to the distribution.

    This is a **solver-agnostic** operation: it takes the distribution *f*,
    a solid *mask*, and **pre-computed** ``u_tau`` and ``y_plus`` fields,
    and returns the corrected distribution ``f_corrected``.

    The correction is applied as a Guo body force on near-wall fluid cells,
    decoupling the wall shear stress from the bulk relaxation time.  The
    body force decelerates the tangential velocity component:

        F = -(τ_w / dy) · û

    where ``τ_w = u_tau²`` and ``û`` is the unit tangential velocity vector.

    Because ``u_tau`` and ``y_plus`` are pre-computed by the caller, this
    function can be combined with any turbulence model (RANS, LES, etc.)
    or collision operator (BGK, MRT, etc.).

    If *rho*, *ux*, *uy*, *uz* are provided, they are used directly instead
    of recomputing macroscopic fields from *f*.  This enables macroscopic
    reuse: the solver computes (rho, ux, uy, uz) once per step and passes
    them to both the collision operator and the wall function.

    Args:
        f:      Distribution tensor ``(Q, nz, ny, nx)``.
        mask:   Boolean solid mask ``(nz, ny, nx)``.  ``True`` = solid.
        u_tau:  Friction velocity field ``(nz, ny, nx)``.
        y_plus: Dimensionless wall distance field ``(nz, ny, nx)``.
        lattice: Lattice name (``"D3Q19"`` or ``"D3Q27"``).
        nu:     Kinematic viscosity (lattice units).
        y_val:  Distance from the near-wall cell centre to the wall.
        rho:    Optional pre-computed density field ``(nz, ny, nx)``.
        ux:     Optional pre-computed x-velocity field ``(nz, ny, nx)``.
        uy:     Optional pre-computed y-velocity field ``(nz, ny, nx)``.
        uz:     Optional pre-computed z-velocity field ``(nz, ny, nx)``.

    Returns:
        Corrected distribution, same shape as *f*.
    """
    _validate_lattice(lattice)

    # If u_tau is zero everywhere, no correction is needed.
    if not u_tau.any():
        return f

    near = _near_wall_mask(mask)
    if rho is not None and ux is not None and uy is not None and uz is not None:
        # Use pre-computed macroscopic fields (no recompute)
        pass
    else:
        rho, ux, uy, uz = _macroscopic(lattice, f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

    # Wall shear stress from pre-computed u_tau
    tau_w = u_tau * u_tau

    # Body force on near-wall cells: F = -(τ_w / dy) · û
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)
    fx = coef * (ux * inv_umag)
    fy = coef * (uy * inv_umag)
    fz = coef * (uz * inv_umag)

    return _apply_body_force(f, fx, fy, fz, lattice, ux=ux, uy=uy, uz=uz)


def log_law(u_tau: torch.Tensor, y_plus: torch.Tensor) -> torch.Tensor:
    """Standard log-law of the wall: ``u+ = (1/κ)·ln(y+) + B``.

    Args:
        u_tau:  Friction velocity field (used only for shape/device inference
                when *y_plus* is scalar; not part of the formula itself).
        y_plus: Dimensionless wall distance field ``(nz, ny, nx)``.

    Returns:
        u+ field, same shape as *y_plus*.  Values are clamped to ``y+``
        (viscous sublayer) for ``y+ < 11.6``.
    """
    yp = y_plus.clamp(min=1e-6)
    up_log = (1.0 / _KAPPA) * torch.log(yp) + _B_LOG
    # In the viscous sublayer (y+ < 11.6) u+ = y+
    up_vis = yp
    return torch.where(yp < 11.6, up_vis, up_log)


def velocity_ramp(
    step: int,
    u_target: float,
    u_start: float = 0.02,
    ramp_steps: int = 1000,
) -> float:
    """Linear velocity ramp for high-Re stability.

    Ramps the inlet velocity from *u_start* to *u_target* over
    *ramp_steps* steps, then holds at *u_target*.  Prevents the initial
    transient from diverging at high Reynolds numbers where the full
    velocity creates an enormous pressure spike at the body surface.

    Args:
        step:       Current step number (1-indexed).
        u_target:   Target free-stream velocity.
        u_start:    Initial velocity (default 0.02).
        ramp_steps: Number of steps to reach *u_target* (default 1000).

    Returns:
        Current inlet velocity.
    """
    if step >= ramp_steps:
        return u_target
    frac = step / ramp_steps
    return u_start + (u_target - u_start) * frac


def apply_wall_function(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor | None,
    nu: float,
    y_val: float = 1.0,
    *,
    lattice: str = "D3Q19",
    y_plus_threshold: float = 11.6,
    rho: torch.Tensor | None = None,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Apply wall function that **REPLACES** bounce-back (not additive).

    This is the key function for the WF (wall-function) mode.  It is
    called **after streaming** (not before, like bounce-back).  The
    wall treatment is selected per-cell based on the local y+ value:

    - ``y+ < y_plus_threshold`` (viscous sublayer): apply bounce-back
      at those near-wall cells.  BB is accurate in the viscous sublayer.
    - ``y+ >= y_plus_threshold`` (log-law region): apply the log-law
      wall function as a Guo body force.  This decouples the wall shear
      from the bulk relaxation time, enabling high-Re simulations.

    The function returns the modified distribution and a diagnostics
    dict with y+ statistics.

    Args:
        f:                 Distribution tensor ``(Q, nz, ny, nx)``.
        solid:             Boolean solid mask ``(nz, ny, nx)``.
        near:              Near-wall mask.  If ``None``, computed internally.
        nu:                Kinematic viscosity (lattice units).
        y_val:             Distance from near-wall cell centre to wall.
        lattice:           ``"D3Q19"`` or ``"D3Q27"``.
        y_plus_threshold:  y+ below which BB is used (default 11.6).
        rho, ux, uy, uz:   Optional pre-computed macroscopic fields.

    Returns:
        ``(f_corrected, diagnostics)`` where diagnostics contains:
        ``y_plus_mean``, ``y_plus_max``, ``n_bb_cells``, ``n_wf_cells``.
    """
    _validate_lattice(lattice)

    if near is None:
        near = _near_wall_mask(solid)

    # Compute macroscopic fields
    if rho is not None and ux is not None and uy is not None and uz is not None:
        pass
    else:
        rho, ux, uy, uz = _macroscopic(lattice, f)

    # Compute wall normal and tangential velocity
    nx_n, ny_n, nz_n = _compute_wall_normal(solid, near)
    u_dot_n = ux * nx_n + uy * ny_n + uz * nz_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_z = uz - u_dot_n * nz_n
    u_tan_mag = torch.sqrt(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z).clamp(min=1e-12)
    has_tan = u_tan_mag > 1e-10
    u_tan_mag = torch.where(
        has_tan, u_tan_mag, torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    )
    ut_x = torch.where(has_tan, ut_x, ux)
    ut_y = torch.where(has_tan, ut_y, uy)
    ut_z = torch.where(has_tan, ut_z, uz)
    inv_utan = 1.0 / u_tan_mag

    # Compute u_tau using log-law Newton iteration
    u_tau = compute_u_tau(u_tan_mag, nu, y_val=y_val, wall_law="log")
    u_tau = torch.where(near, u_tau, torch.zeros_like(u_tau))

    # Compute y+
    y_plus = compute_y_plus(u_tau, nu, y_val=y_val)
    y_plus = torch.where(near, y_plus, torch.zeros_like(y_plus))

    # Classify cells: BB (viscous) vs WF (log-law)
    use_bb = near & (y_plus < y_plus_threshold)
    use_wf = near & (y_plus >= y_plus_threshold)

    n_bb = int(use_bb.sum().item())
    n_wf = int(use_wf.sum().item())

    # --- 1. Bounce-back for viscous sublayer cells ---
    if n_bb > 0:
        if lattice == "D3Q19":
            from .d3q19 import OPPOSITE as OPP

            opp = OPP.to(f.device)
        else:
            from .d3q27 import OPPOSITE as OPP

            opp = OPP.to(f.device)
        bb_mask = use_bb.unsqueeze(0).expand_as(f)
        f = torch.where(bb_mask, f[opp], f)

    # --- 2. Wall function (Guo body force) for log-law cells ---
    if n_wf > 0:
        tau_w = u_tau * u_tau
        # Body force: F = -tau_w * û_tan (decelerate tangential flow)
        coef = -tau_w * use_wf.to(f.dtype)
        fx = coef * (ut_x * inv_utan)
        fy = coef * (ut_y * inv_utan)
        fz = coef * (ut_z * inv_utan)
        f = _apply_body_force(f, fx, fy, fz, lattice, ux=ux, uy=uy, uz=uz)

    # Diagnostics
    y_plus_near = y_plus[near]
    if y_plus_near.numel() > 0:
        yp_mean = float(y_plus_near.mean().item())
        yp_max = float(y_plus_near.max().item())
    else:
        yp_mean = 0.0
        yp_max = 0.0

    diag = {
        "y_plus_mean": yp_mean,
        "y_plus_max": yp_max,
        "n_bb_cells": n_bb,
        "n_wf_cells": n_wf,
    }
    return f, diag


__all__ = [
    "SUPPORTED_LATTICES",
    "compute_u_tau",
    "compute_y_plus",
    "log_law",
    "velocity_ramp",
    "apply_wall_function",
    "wall_function",
]
