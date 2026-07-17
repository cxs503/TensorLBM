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
        raise ValueError(
            f"Unsupported lattice {lattice!r}; supported: {SUPPORTED_LATTICES}"
        )
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

    raise ValueError(f"Unknown wall_law {wall_law!r}; supported: 'log', 'reichardt'")


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
    """Identify fluid cells adjacent to solid cells (6-connected)."""
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    return near


# ---------------------------------------------------------------------------
# Public wall_function interface
# ---------------------------------------------------------------------------

def _apply_body_force(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    lattice: str,
) -> torch.Tensor:
    """Apply a Guo body-force correction to a 3-D distribution.

    This is a **lattice-agnostic** helper: it dispatches to the correct
    velocity vectors (``C``, ``W``) for D3Q19 or D3Q27.  The Guo forcing
    term is ``w_i * 3 * (c_i · F)`` added to the distribution.
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
    # Need velocity field for the correction term; extract from f.
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

    Args:
        f:      Distribution tensor ``(Q, nz, ny, nx)``.
        mask:   Boolean solid mask ``(nz, ny, nx)``.  ``True`` = solid.
        u_tau:  Friction velocity field ``(nz, ny, nx)``.
        y_plus: Dimensionless wall distance field ``(nz, ny, nx)``.
        lattice: Lattice name (``"D3Q19"`` or ``"D3Q27"``).
        nu:     Kinematic viscosity (lattice units).
        y_val:  Distance from the near-wall cell centre to the wall.

    Returns:
        Corrected distribution, same shape as *f*.
    """
    _validate_lattice(lattice)

    # If u_tau is zero everywhere, no correction is needed.
    if not u_tau.any():
        return f

    near = _near_wall_mask(mask)
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

    return _apply_body_force(f, fx, fy, fz, lattice)


__all__ = [
    "SUPPORTED_LATTICES",
    "compute_u_tau",
    "compute_y_plus",
    "wall_function",
]
