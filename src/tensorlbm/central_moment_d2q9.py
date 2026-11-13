"""D2Q9 central-moment collision with partial relaxation (LIFE-compatible).

This module ports the ``CENTRAL_MOMENTS`` collision operator of
`joconnor22/LIFE <https://github.com/joconnor22/LIFE>`_ (``src/Grid.cpp``,
branch ``streamCollide``) into TensorLBM, and generalises the relaxation
rates so the scheme can be ablated against BGK / full-order relaxation.

Upstream scheme (as read from the source, not from the paper)
--------------------------------------------------------------
LIFE keeps the **mass-weighted** (unweighted) central moments of the nine
monomials spanned by D2Q9,

    κ = Σ_v f_v P_m(c_v - u),   P ∈ {1, x, y, x²+y², x²-y², xy, x²y, xy², x²y²},

with the ordering ``κ = (κ0, κx, κy, κtr, κd, κxy, κxxy, κxyy, κxxyy)``.
Its post-collision moment vector is (``src/Grid.cpp``, CENTRAL_MOMENTS
branch, with ``c_s² = 1/3``, ``ω = 1/τ``, ``F`` = body/IBM force):

    κ0*    = ρ                      (mass)
    κx*    = Fx / 2                 (half force step, Guo-style)
    κy*    = Fy / 2
    κtr*   = 2 ρ c_s²               (equilibrium energy)
    κd*    = (1 - ω) κd             ** only the deviatoric stress is relaxed **
    κxy*   = (1 - ω) κxy            ** at ω; this is the "partial" in the name **
    κxxy*  = Fy c_s² / 2            (equilibrium 0 + 3rd-order force term)
    κxyy*  = Fx c_s² / 2
    κxxyy* = ρ c_s⁴                 (equilibrium 4th-order moment)

so orders 0, 1, 2-trace, 3 and 4 are *projected onto their equilibrium (or
force) values* while the shear stress alone is relaxed at ``ω``.  The
equilibrium used upstream is the fourth-order Hermite expansion

    f_eq = ρ w_v / 4 · (9 c_x² u_x² + 6 c_x u_x - 3 u_x² + 2)
                     · (9 c_y² u_y² + 6 c_y u_y - 3 u_y² + 2)

which has exactly the central moments above (verified numerically: κtr =
2ρc_s², κd = κxy = κxxy = κxyy = 0, κxxyy = ρc_s⁴ for arbitrary u).

Generalisation
--------------
The relaxation rates of the non-hydrodynamic moments are exposed as
``s_bulk`` (κtr), ``s3`` (κxxy, κxyy) and ``s4`` (κxxyy).  ``s_* = 1``
reproduces LIFE exactly; ``s_* = ω`` is the naïve "relax every order at the
shear rate" variant used as the ablation arm; ``s_bulk = 1, s3 = s4 = 1`` is
the upstream behaviour.  The deviatoric stress moment always relaxes at ``ω``.

Assumptions / scope
-------------------
* 2-D, D2Q9, single-phase, BGK-type viscosity ``ν = (1/ω - 1/2)/3``.
* The molecule-to-population map is ``f = M₀⁻¹ · shift⁻¹(κ)`` where ``M₀``
  is the unweighted monomial matrix at ``u = 0``; ``M₀⁻¹`` is a module
  constant.  This is algebraically identical to upstream's hand-expanded
  ``fStar[v]`` formulas (verified to 5e-16, see
  ``runs/xval_life_cm_partial/result.json`` → ``operator_equivalence``).
* Force is injected in moment space exactly as upstream does; the Guo
  half-step shows up in :func:`macroscopic_guo` as ``u = (Σ c f + F/2)/ρ``.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from .d2q9 import C, _w_for

__all__ = [
    "M0",
    "M0INV",
    "equilibrium_hermite4",
    "central_moments",
    "moments_to_populations",
    "collide_central_moment_d2q9",
    "macroscopic_guo",
]

_CS2 = 1.0 / 3.0
_CS4 = _CS2 * _CS2


# ---------------------------------------------------------------------------
# Moment transform matrices
# ---------------------------------------------------------------------------
def _basis(cx: float, cy: float) -> tuple[float, ...]:
    """The nine LIFE moment monomials evaluated at ``(cx, cy)``."""
    return (
        1.0,
        cx,
        cy,
        cx * cx + cy * cy,
        cx * cx - cy * cy,
        cx * cy,
        cx * cx * cy,
        cx * cy * cy,
        cx * cx * cy * cy,
    )


def _build_m0() -> np.ndarray:
    """``M0[v, m]`` = monomial ``m`` at lattice velocity ``v`` (u = 0)."""
    rows = []
    for v in range(9):
        cx = float(C[v, 0].item())
        cy = float(C[v, 1].item())
        rows.append(_basis(cx, cy))
    return np.asarray(rows, dtype=np.float64)


M0: np.ndarray = _build_m0()
"""(9, 9) unweighted moment matrix, rows = directions, cols = moments."""

# κ_m = Σ_v M0[v, m] f_v is a *contraction over directions*, so the moment
# vector is κ = M0ᵀ f and the population vector is f = (M0ᵀ)⁻¹ κ.  Storing the
# transpose of the inverse lets ``moments_to_populations`` do one mat-vec per
# moment component: f_v = Σ_m M0INV[v, m] κ_m.
M0INV: np.ndarray = np.linalg.inv(M0).T
"""``(M0ᵀ)⁻¹``: ``f = M0INV @ κ`` at ``u = 0`` (``f_v = Σ_m M0INV[v,m] κ_m``)."""

_M0_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_M0INV_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _m0(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _M0_CACHE:
        _M0_CACHE[key] = torch.as_tensor(M0, dtype=dtype, device=device)
    return _M0_CACHE[key]


def _m0inv(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _M0INV_CACHE:
        _M0INV_CACHE[key] = torch.as_tensor(M0INV, dtype=dtype, device=device)
    return _M0INV_CACHE[key]


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------
def equilibrium_hermite4(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Fourth-order Hermite equilibrium of upstream LIFE (D2Q9).

    ``f_eq = ρ w_v / 4 (9 c_x² u_x² + 6 c_x u_x - 3 u_x² + 2)
    (9 c_y² u_y² + 6 c_y u_y - 3 u_y² + 2)``.

    Unlike :func:`tensorlbm.d2q9.equilibrium` (second-order in ``u``) this
    keeps the O(u²) diagonal terms of both axes and is the equilibrium whose
    central moments are exactly ``(ρ, 0, 0, 2ρc_s², 0, 0, 0, 0, ρc_s⁴)``.
    """
    device, dtype = rho.device, rho.dtype
    cx = C[:, 0].to(device=device, dtype=dtype).view(9, 1, 1)
    cy = C[:, 1].to(device=device, dtype=dtype).view(9, 1, 1)
    w = _w_for(dtype, device).view(9, 1, 1)
    hx = 9.0 * cx * cx * ux * ux + 6.0 * cx * ux - 3.0 * ux * ux + 2.0
    hy = 9.0 * cy * cy * uy * uy + 6.0 * cy * uy - 3.0 * uy * uy + 2.0
    return 0.25 * rho * w * hx * hy


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------
def central_moments(
    f: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Mass-weighted central moments of ``f`` in the LIFE ordering.

    Returns ``(κ0, κx, κy, κtr, κd, κxy, κxxy, κxyy, κxxyy)``, each of shape
    ``(ny, nx)`` -- i.e. ``Σ_v f_v (c_v - u)^a (c_v - u)^b`` with lags summed.
    """
    device, dtype = f.device, f.dtype
    cx = C[:, 0].to(device=device, dtype=dtype).view(9, 1, 1)
    cy = C[:, 1].to(device=device, dtype=dtype).view(9, 1, 1)
    x = cx - ux
    y = cy - uy
    x2 = x * x
    y2 = y * y
    return (
        f.sum(dim=0),
        (f * x).sum(dim=0),
        (f * y).sum(dim=0),
        (f * (x2 + y2)).sum(dim=0),
        (f * (x2 - y2)).sum(dim=0),
        (f * (x * y)).sum(dim=0),
        (f * (x2 * y)).sum(dim=0),
        (f * (x * y2)).sum(dim=0),
        (f * (x2 * y2)).sum(dim=0),
    )


def moments_to_populations(
    kappa: tuple[torch.Tensor, ...], ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """Inverse transform ``κ → f`` (unshift then ``M0⁻¹``).

    ``κ`` follows the LIFE ordering; returns a ``(9, ny, nx)`` tensor.
    """
    k0, kx, ky, ktr, kd, kxy, kxxy, kxyy, kxxyy = kappa
    ux2 = ux * ux
    uy2 = uy * uy
    tr_d = 0.5 * (ktr + kd)
    tr_m = 0.5 * (ktr - kd)

    m00 = k0
    m10 = kx + ux * k0
    m01 = ky + uy * k0
    m20 = tr_d + 2.0 * ux * kx + ux2 * k0
    m02 = tr_m + 2.0 * uy * ky + uy2 * k0
    m11 = kxy + uy * kx + ux * ky + ux * uy * k0
    m21 = (
        kxxy
        + 2.0 * ux * kxy
        + uy * tr_d
        + 2.0 * ux * uy * kx
        + ux2 * ky
        + ux2 * uy * k0
    )
    m12 = (
        kxyy
        + 2.0 * uy * kxy
        + ux * tr_m
        + uy2 * kx
        + 2.0 * ux * uy * ky
        + ux * uy2 * k0
    )
    m22 = (
        kxxyy
        + 2.0 * uy * kxxy
        + 2.0 * ux * kxyy
        + 4.0 * ux * uy * kxy
        + ux2 * tr_m
        + uy2 * tr_d
        + 2.0 * ux2 * uy * ky
        + 2.0 * ux * uy2 * kx
        + ux2 * uy2 * k0
    )
    # ``M0`` columns follow the basis order (1, x, y, x²+y², x²−y², xy, x²y,
    # xy², x²y²), so the trace/deviatoric pair must be re-combined here.
    m = torch.stack(
        [m00, m10, m01, m20 + m02, m20 - m02, m11, m21, m12, m22], dim=0
    )
    return torch.einsum("vm,mij->vij", _m0inv(m00.device, m00.dtype), m)


# ---------------------------------------------------------------------------
# Macroscopic quantities
# ---------------------------------------------------------------------------
def macroscopic_guo(
    f: torch.Tensor,
    force: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ρ, u)`` with the Guo half-force velocity ``u = (Σ c f + F/2)/ρ``.

    Mirrors ``GridClass::macroscopic`` upstream (which adds ``0.5 * force``).
    """
    device, dtype = f.device, f.dtype
    cx = C[:, 0].to(device=device, dtype=dtype).view(9, 1, 1)
    cy = C[:, 1].to(device=device, dtype=dtype).view(9, 1, 1)
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-30)
    mx = (f * cx).sum(dim=0)
    my = (f * cy).sum(dim=0)
    if force is not None:
        mx = mx + 0.5 * force[0]
        my = my + 0.5 * force[1]
    return rho, mx / rho_safe, my / rho_safe


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------
def collide_central_moment_d2q9(
    f: torch.Tensor,
    omega: float,
    *,
    force: tuple[torch.Tensor, torch.Tensor] | None = None,
    s_bulk: float = 1.0,
    s3: float = 1.0,
    s4: float = 1.0,
) -> torch.Tensor:
    """Central-moment D2Q9 collision; ``s3 = s4 = s_bulk = 1`` is LIFE.

    Args:
        f:     pre-collision populations, shape ``(9, ny, nx)``.
        omega: shear relaxation rate ``ω = 1/τ`` applied to ``κd`` and ``κxy``.
        force: optional ``(Fx, Fy)`` body force per cell, lattice units.
        s_bulk: relaxation rate of the energy (trace) mode ``κtr`` towards
            equilibrium.  ``1`` = upstream LIFE (moment replaced by 2ρc_s²).
        s3:    rate for the third-order moments ``κxxy``, ``κxyy``.
        s4:    rate for the fourth-order moment ``κxxyy``.

    Returns:
        Post-collision populations, same shape as ``f``.  Streaming is *not*
        done here.
    """
    if not (0.0 < omega <= 2.0):
        raise ValueError(f"omega must satisfy 0 < omega <= 2, got {omega}")
    for name, s in (("s_bulk", s_bulk), ("s3", s3), ("s4", s4)):
        if not (0.0 <= s <= 2.0):
            raise ValueError(f"{name} must satisfy 0 <= {name} <= 2, got {s}")

    rho, ux, uy = macroscopic_guo(f, force)
    k0, kx, ky, ktr, kd, kxy, kxxy, kxyy, kxxyy = central_moments(f, ux, uy)

    fx = torch.zeros_like(rho) if force is None else force[0]
    fy = torch.zeros_like(rho) if force is None else force[1]

    # Post-collision moments (LIFE: everything but the shear stress is reset).
    k0_star = k0                                # mass
    kx_star = 0.5 * fx                          # half force step
    ky_star = 0.5 * fy
    ktr_star = (1.0 - s_bulk) * ktr + s_bulk * (2.0 * rho * _CS2)
    kd_star = (1.0 - omega) * kd
    kxy_star = (1.0 - omega) * kxy
    kxxy_star = (1.0 - s3) * kxxy + 0.5 * fy * _CS2
    kxyy_star = (1.0 - s3) * kxyy + 0.5 * fx * _CS2
    kxxyy_star = (1.0 - s4) * kxxyy + s4 * (rho * _CS4)

    return moments_to_populations(
        (
            k0_star,
            kx_star,
            ky_star,
            ktr_star,
            kd_star,
            kxy_star,
            kxxy_star,
            kxyy_star,
            kxxyy_star,
        ),
        ux,
        uy,
    )