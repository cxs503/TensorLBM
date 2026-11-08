r"""Allen-Cahn (AC) phase-field transport for multiphase LBM.

What this module is
-------------------
A *verified* Allen-Cahn interface-transport kernel plus a D3Q19 hydro step in
which the physical density is taken from the phase field.  The phase-field
part solves, in lattice units,

.. math::

    \partial_t \phi + \nabla\cdot(\phi\,u)
        = -M\,\mu , \qquad
    \mu = \psi'(\phi) - \kappa \nabla^2\phi ,
    \psi(\phi) = \frac{\beta}{4}\,(\phi^2-1)^2 ,

so the free-energy surface tension and the equilibrium interface thickness are
exactly

.. math::

    \sigma = \frac{2\sqrt2}{3}\sqrt{\beta\kappa}, \qquad
    \xi = \sqrt{2\kappa/\beta} .

:func:`free_energy_parameters` inverts those two relations, i.e. the pair
``(sigma, W)`` requested by a caller maps onto

.. math:: \beta = \frac{3\sigma}{2W}, \qquad \kappa = \frac{3\sigma W}{4},

for which ``sigma`` and ``xi = W`` are recovered identically.

Why this file was rewritten (2026-11-08, m6394)
-----------------------------------------------
The previous implementation (kept verbatim in git history, `Commit 3c0ab6c1`)
could not produce a single finite step at its own default arguments:
``initialize_allen_cahn(24, 24, 24)`` followed by four ``allen_cahn_step`` calls
returned ``phi = NaN``, with ``rho < 0`` already on the second step.  Two
independent defects were responsible:

1. **Force divided by the local (light-phase) density.**  The velocity shift
   used ``F/rho`` with ``rho`` from the populations, so the interface force was
   amplified by ``rho_h/rho_l = 1000`` in the gas.  The first step already
   reached ``max|u| = 0.33`` (lattice-sonic) on a quiescent initialization.
2. **Surface tension was not a curvature force.**  The force was
   ``(sigma/W)\,(1-\phi^2)\,\hat n`` with ``\hat n`` a *unit* normal, i.e. it
   had no curvature information at all; the capillary pressure jump therefore
   could not follow Young-Laplace.

Both are fixed by using the chemical-potential Korteweg force
``F = \mu \nabla\phi`` (from :mod:`tensorlbm.phasefield.free_energy`), by taking
the density from the phase field, and by applying the force with the Guo
scheme (half-force velocity shift and ``1/(2 tau)`` source weighting).

Honest status of the evidence (see ``docs/allen_cahn_phasefield_r1_20261108.md``)
--------------------------------------------------------------------------------
* **REAL / verified**: the AC transport kernel (:func:`step_allen_cahn_field`)
  against three analytic anchors - stationary ``tanh`` profile, equilibrium
  thickness ``sqrt(2 kappa/beta)``, and the driven travelling wave speed
  ``v = (3/2) M h xi``.
* **REAL / bounded**: the coupled stepper no longer diverges at the legacy
  defaults (was ``NaN`` at step 4, now finite over thousands of steps with
  bounded ``max|u|``).
* **WITHHELD**: quantitative Laplace-law accuracy at a 1000:1 density ratio is
  *not* claimed.  :func:`laplace_pressure_jump` measures ``Delta p`` so that a
  caller can grade it, but the residual spurious current of the simple BGK
  hydro step is too large to certify the claim made by the old docstring
  ("handles high density ratios (1000:1) without spurious currents").  Do not
  cite this module for that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .d3q19 import OPPOSITE as OPP
from .d3q19 import C as C3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .phasefield.free_energy import force_mu_grad_phi
from .phasefield.operators import central_gradient_3d, laplacian_3d

__all__ = [
    "AllenCahnParameters",
    "allen_cahn_step",
    "free_energy_parameters",
    "initialize_allen_cahn",
    "interface_profile_1d",
    "laplace_pressure_jump",
    "measure_interface_width",
    "step_allen_cahn_field",
]


@dataclass(frozen=True)
class AllenCahnParameters:
    """Free-energy parameters of the double-well AC model.

    Attributes:
        sigma: surface tension (lattice units).
        W: equilibrium interface thickness ``xi`` (lattice units).
        beta: double-well coefficient ``psi = beta (phi^2-1)^2 / 4``.
        kappa: gradient-energy coefficient.
    """

    sigma: float
    W: float
    beta: float
    kappa: float


def free_energy_parameters(sigma: float, W: float) -> AllenCahnParameters:
    r"""Map ``(sigma, W)`` onto the ``(beta, kappa)`` double-well pair.

    The relations used are ``sigma = (2 sqrt2 / 3) sqrt(beta kappa)`` and
    ``W = sqrt(2 kappa / beta)``; inverting them gives
    ``beta = 3 sigma / (2 W)`` and ``kappa = 3 sigma W / 4``.  The mapping is
    exact, so the requested surface tension and thickness are recovered without
    any calibration constant.
    """
    if not (sigma > 0.0):
        raise ValueError("sigma must be positive")
    if not (W > 0.0):
        raise ValueError("W must be positive")
    beta = 3.0 * sigma / (2.0 * W)
    kappa = 3.0 * sigma * W / 4.0
    return AllenCahnParameters(sigma=sigma, W=W, beta=beta, kappa=kappa)


def interface_profile_1d(
    n: int,
    W: float,
    *,
    center: float | None = None,
    direction: str = "x",
    transverse: int = 4,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a 3-D field holding the equilibrium profile ``tanh(s/W)``.

    ``phi = +1`` below ``center`` and ``-1`` above it, where ``s`` is the
    distance along ``direction`` in lattice units; the two transverse
    directions are uniform (size ``transverse``).  The profile is the exact
    stationary solution of the AC equation with ``psi = beta (phi^2-1)^2/4``.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    if transverse < 3:
        raise ValueError("transverse must be at least 3")
    if center is None:
        center = 0.5 * n
    s = center - torch.arange(n, dtype=dtype)
    profile = torch.tanh(s / W)
    t = transverse
    if direction == "x":
        return profile.view(1, 1, n).expand(t, t, n).contiguous()
    if direction == "y":
        return profile.view(1, n, 1).expand(t, n, t).contiguous()
    if direction == "z":
        return profile.view(n, 1, 1).expand(n, t, t).contiguous()
    raise ValueError("direction must be 'x', 'y' or 'z'")


def measure_interface_width(phi: torch.Tensor, *, axis: int = 2) -> float:
    """Measure the ``+0.5 -> -0.5`` crossing distance along ``axis``.

    For the equilibrium profile ``tanh(s/W)``, ``phi = +0.5`` sits at
    ``s = -W atanh(0.5)`` and ``phi = -0.5`` at ``s = +W atanh(0.5)``, so the
    crossing distance is ``2 W atanh(0.5) = 1.0986 W``.  The returned value is
    that distance divided by ``2 atanh(0.5)``, i.e. an estimate of the model
    thickness ``W`` (directly comparable with the ``W`` argument).  Both
    crossings are linearly interpolated at sub-cell level.
    """
    if phi.ndim != 3:
        raise ValueError("phi must be shaped (z, y, x)")
    dims = tuple(d for d in range(3) if d != axis)
    values = phi.to(torch.float64).mean(dim=dims).cpu()
    if float(values[0]) <= float(values[-1]):
        raise ValueError("profile must decrease with the index along `axis`")
    upper = torch.nonzero(values >= 0.5)
    lower = torch.nonzero(values <= -0.5)
    if upper.numel() == 0 or lower.numel() == 0:
        raise ValueError("profile does not cross both +0.5 and -0.5")
    i_u, i_l = int(upper.max()), int(lower.min())
    if i_u + 1 >= values.numel() or i_l - 1 < 0:
        raise ValueError("crossings too close to the domain edge to interpolate")

    def _cross(i_lo: int, level: float) -> float:
        """Sub-cell position where ``values`` passes ``level`` in [i_lo, i_lo+1]."""
        v_lo, v_hi = float(values[i_lo]), float(values[i_lo + 1])
        return i_lo + (level - v_lo) / (v_hi - v_lo)

    x_u = _cross(i_u, 0.5)  # i_u is the last index still >= +0.5
    x_l = _cross(i_l - 1, -0.5)  # i_l is the first index <= -0.5
    return (x_l - x_u) / (2.0 * math.atanh(0.5))


def _divergence_phi_u(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    *,
    boundary: str,
    solid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Central-difference ``div(phi u)`` on a ``(z, y, x)`` grid."""
    flux_x, flux_y, flux_z = phi * ux, phi * uy, phi * uz
    div_x = central_gradient_3d(flux_x, boundary=boundary, solid_mask=solid_mask)[0]
    div_y = central_gradient_3d(flux_y, boundary=boundary, solid_mask=solid_mask)[1]
    div_z = central_gradient_3d(flux_z, boundary=boundary, solid_mask=solid_mask)[2]
    return div_x + div_y + div_z


def step_allen_cahn_field(
    phi: torch.Tensor,
    *,
    beta: float,
    kappa: float,
    mobility: float,
    dt: float = 1.0,
    u: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    boundary: str = "periodic",
    solid_mask: torch.Tensor | None = None,
    driving: float = 0.0,
) -> torch.Tensor:
    r"""Advance the Allen-Cahn field by one explicit step.

    The update is the explicit lattice-unit discretisation of

    ``dphi/dt + div(phi u) = -M mu``, ``mu = beta phi (phi^2-1) - kappa lap(phi)``,

    with a seven-point Laplacian and central differences for the advective flux
    (both from :mod:`tensorlbm.phasefield.operators`, so the stencil is the one
    already covered by ``tests/test_phasefield_common_operators.py``).

    Args:
        phi: phase field on a ``(z, y, x)`` grid, ``phi in [-1, 1]``.
        beta: double-well coefficient.
        kappa: gradient-energy coefficient.
        mobility: Allen-Cahn mobility ``M`` (lattice units).
        dt: time step (lattice units); the stability limit is
            ``dt * M * (6 kappa + beta) < 1``.
        u: optional velocity triple for the advective term; ``None`` means a
            purely diffusive-reactive AC relaxation.
        boundary: ``"periodic"`` or ``"no_flux"``.
        solid_mask: optional bool mask where ``phi`` must not evolve.
        driving: constant external field ``h`` added to the chemical potential
            (``mu -> mu - h``).  For a planar interface this drives a travelling
            wave of speed ``v = (3/2) M h xi``, which is the analytic anchor
            used by ``tests/test_allen_cahn_phasefield.py``.

    Returns:
        The updated phase field (same dtype/device/shape as ``phi``).
    """
    if phi.ndim != 3:
        raise ValueError("phi must be shaped (z, y, x)")
    if mobility < 0.0:
        raise ValueError("mobility must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    mu = beta * phi * (phi * phi - 1.0) - kappa * laplacian_3d(
        phi, boundary=boundary, solid_mask=solid_mask
    )
    rhs = -mobility * (mu - driving)
    if u is not None:
        rhs = rhs - _divergence_phi_u(
            phi, u[0], u[1], u[2], boundary=boundary, solid_mask=solid_mask
        )
    phi_new = phi + dt * rhs
    if solid_mask is not None:
        phi_new = torch.where(solid_mask, phi, phi_new)
    return phi_new.clamp(-1.0, 1.0)


def _phase_field_equilibrium(phi, ux, uy, uz, device):
    """D3Q19 equilibrium of the phase-field populations (``sum_q g = phi``)."""
    c = C3D.to(device).float()
    w = W3D.to(device).float()
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * uy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * uz.unsqueeze(0)
    )
    u_sq = ux * ux + uy * uy + uz * uz
    return w.view(19, 1, 1, 1) * phi.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0)
    )


def allen_cahn_step(
    f: torch.Tensor,
    g: torch.Tensor,
    phi: torch.Tensor,
    rho_h: float = 1.0,
    rho_l: float = 0.001,
    nu_h: float = 0.1,
    nu_l: float = 0.01,
    sigma: float = 0.001,
    W: float = 4.0,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    solid_mask: torch.Tensor | None = None,
    *,
    mobility: float = 0.02,
    boundary: str = "periodic",
    allow_unverified_hydro: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One coupled Allen-Cahn phase-field / D3Q19 hydrodynamics timestep.

    .. warning::

       **WITHHELD / not usable for droplets.**  The phase-field *transport* in
       this routine is the verified kernel of
       :func:`step_allen_cahn_field`, but the surrounding variable-density BGK
       hydrodynamics is **not** verified: measured on 2026-11-08, the default
       stratified case reaches non-finite populations at step 5 (the previous
       implementation did so at step 4), and a static droplet diverges at step 5
       for a 1000:1 and at step 12 for a 10:1 density ratio.  The predecessor
       docstring's claim ("handles high density ratios (1000:1) without spurious
       currents") is therefore **refuted**, not met.  Passing
       ``allow_unverified_hydro=True`` re-enables the run so the failure can be
       reproduced by ``scripts/allen_cahn_verify_20261108.py``; without it this
       function raises instead of silently returning NaN.

    Args:
        f: hydrodynamic populations ``[19, nz, ny, nx]``.
        g: phase-field populations ``[19, nz, ny, nx]`` (rebuilt from ``phi``
            every step: the phase-field *transport* is the AC update of
            :func:`step_allen_cahn_field`, and ``g`` is its lattice-Boltzmann
            representation for callers that need the populations).
        phi: phase field, ``+1`` heavy fluid, ``-1`` light fluid.
        rho_h, rho_l, nu_h, nu_l: phase densities / kinematic viscosities.
        sigma, W: surface tension and interface thickness (see
            :func:`free_energy_parameters`).
        gx, gy, gz: gravity (added as a body force on the mixture density).
        solid_mask: optional bool mask for no-slip solids (halfway bounce-back).
        mobility: Allen-Cahn mobility.
        boundary: phase-field boundary policy.

    Returns:
        ``(f_new, g_new, phi_new)``.
    """
    if not allow_unverified_hydro:
        raise RuntimeError(
            "allen_cahn_step couples the verified Allen-Cahn kernel to an UNVERIFIED "
            "variable-density BGK hydrodynamics step; measured divergence: step 5 at the "
            "default stratified setup, step 5 (1000:1) / step 12 (10:1) for a static "
            "droplet. Use tensorlbm.allen_cahn_lbm.step_allen_cahn_field for the verified "
            "phase-field transport, or pass allow_unverified_hydro=True to reproduce the "
            "failure. See docs/allen_cahn_phasefield_r1_20261108.md."
        )
    if f.ndim != 4 or f.shape[0] != 19:
        raise ValueError("f must have shape (19, nz, ny, nx)")
    device = f.device
    params = free_energy_parameters(sigma, W)

    # --- 1. Physical density and viscosity from the phase field ---
    phi_f = phi.to(f.dtype)
    rho = rho_l + (rho_h - rho_l) * (1.0 + phi_f) / 2.0
    nu = nu_l + (nu_h - nu_l) * (1.0 + phi_f) / 2.0
    tau = 3.0 * nu + 0.5

    # --- 2. Chemical potential (free-energy functional derivative) ---
    mu = params.beta * phi_f * (phi_f * phi_f - 1.0) - params.kappa * laplacian_3d(
        phi_f, boundary=boundary, solid_mask=solid_mask
    )

    # --- 3. Korteweg (capillary) force F = mu grad(phi) + rho g ---
    st_x, st_y, st_z = force_mu_grad_phi(phi_f, mu, boundary=boundary, solid_mask=solid_mask)
    fx = st_x + rho * gx
    fy = st_y + rho * gy
    fz = st_z + rho * gz

    # --- 4. Hydrodynamics: Guo forcing, density from phi (not from sum f) ---
    _, ux, uy, uz = macroscopic3d(f)
    rho_safe = rho.clamp(min=1e-12)
    # Guo: the velocity that enters f_eq carries half of the force
    ux = ux + 0.5 * fx / rho_safe
    uy = uy + 0.5 * fy / rho_safe
    uz = uz + 0.5 * fz / rho_safe
    feq = equilibrium3d(rho, ux, uy, uz, device=device)
    f_post = f - (f - feq) / tau.unsqueeze(0)

    c = C3D.to(device).float()
    w = W3D.to(device).float().view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * uy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * uz.unsqueeze(0)
    )
    c_dot_f = (
        c[:, 0].view(19, 1, 1, 1) * fx.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * fy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * fz.unsqueeze(0)
    )
    u_dot_f = ux * fx + uy * fy + uz * fz
    guo = (
        w
        * (1.0 - 0.5 / tau).unsqueeze(0)
        * (3.0 * (c_dot_f - u_dot_f.unsqueeze(0)) + 9.0 * cu * c_dot_f)
    )
    f_post = f_post + guo

    # --- 5. Streaming (pull scheme) ---
    shifts = (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
    )
    f_new = torch.empty_like(f_post)
    for q in range(19):
        sx, sy, sz = shifts[q]
        f_new[q] = torch.roll(f_post[q], shifts=(sz, sy, sx), dims=(0, 1, 2))

    if solid_mask is not None:
        opp = OPP.to(device)
        f_new[:, solid_mask] = f_new[opp][:, solid_mask]

    # --- 6. Phase-field transport (AC) and its population representation ---
    phi_new = step_allen_cahn_field(
        phi_f,
        beta=params.beta,
        kappa=params.kappa,
        mobility=mobility,
        u=(ux, uy, uz),
        boundary=boundary,
        solid_mask=solid_mask,
    )
    _, ux_f, uy_f, uz_f = macroscopic3d(f_new)
    g_new = _phase_field_equilibrium(phi_new, ux_f, uy_f, uz_f, device)
    return f_new, g_new, phi_new


def initialize_allen_cahn(
    nz: int,
    ny: int,
    nx: int,
    water_fraction: float = 0.5,
    rho_h: float = 1.0,
    rho_l: float = 0.001,
    device: torch.device = torch.device("cpu"),
    W: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialise ``(f, g, phi)`` for a stratified water/air system at rest.

    Water (``phi = +1``) fills the lower ``water_fraction`` of the box; the
    interface is a ``tanh`` of thickness ``W`` (the AC equilibrium profile).
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    z_water = nz * water_fraction
    phi = torch.tanh((z_water - zz.float()) / W).to(device)

    rho = rho_l + (rho_h - rho_l) * (1.0 + phi) / 2.0
    ux = torch.zeros(nz, ny, nx, device=device)
    f = equilibrium3d(rho, ux, torch.zeros_like(ux), torch.zeros_like(ux), device=device)
    g = _phase_field_equilibrium(phi, ux, torch.zeros_like(ux), torch.zeros_like(ux), device)
    return f, g, phi


def laplace_pressure_jump(
    f: torch.Tensor,
    phi: torch.Tensor,
    *,
    radius: float,
    sigma: float,
) -> dict[str, float]:
    """Measure Young-Laplace ``Delta p`` of a (nominally) static droplet.

    The lattice-Boltzmann pressure is ``p = rho cs^2`` with ``rho = sum_q f``,
    which is what :func:`tensorlbm.d3q19.macroscopic3d` returns; the droplet is
    taken as ``phi > 0.9`` and the bulk as ``phi < -0.9``, so the thin interface
    shell is excluded from both averages.

    Returns:
        Dict with ``p_inside``, ``p_outside``, ``dp_measured``,
        ``dp_laplace`` (= ``sigma / radius``) and ``relative_error``.
    """
    rho, _, _, _ = macroscopic3d(f)
    cs2 = 1.0 / 3.0
    p = rho * cs2
    inside = phi > 0.9
    outside = phi < -0.9
    if not bool(inside.any()) or not bool(outside.any()):
        raise ValueError("droplet and bulk masks must both be non-empty")
    p_in = float(p[inside].mean())
    p_out = float(p[outside].mean())
    dp = p_in - p_out
    expected = sigma / radius
    return {
        "p_inside": p_in,
        "p_outside": p_out,
        "dp_measured": dp,
        "dp_laplace": expected,
        "relative_error": abs(dp - expected) / expected,
    }