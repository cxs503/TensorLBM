"""Herschel-Bulkley yield-stress rheology: analytic anchors (kinematic units).

Constitutive law
----------------
A Herschel-Bulkley (H-B) material carries a *yield stress* ``tau_y``: it does
not deform while the local shear stress stays below the threshold, and once
yielded it follows a power law,

    tau = tau_y + K * gamma_dot**n      (tau > tau_y)
    gamma_dot = 0                        (tau <= tau_y)

with ``K`` the consistency index and ``n`` the flow index.  Special cases:

* ``tau_y = 0, n = 1`` -> Newtonian (``nu = K``),
* ``tau_y = 0``        -> power law (see :mod:`tensorlbm.powerlaw`),
* ``n = 1``            -> Bingham plastic.

The unyielded material moves as a rigid body; in a body-force / pressure-driven
duct that rigid body is the **plug core**, and its boundary is the **yield
surface**.

Two facts organise this module
------------------------------
**(1) Yield surface from momentum balance (rheology independent).**
For a pipe of radius ``R`` driven by a body-force acceleration ``a`` the total
shear stress at radius ``r`` is fixed by the axial momentum balance,
``sigma(r) = rho*a*r/2``, hence ``sigma_w = rho*a*R/2``.  The plug boundary is
where ``sigma = tau_y``:

    r_c = R * tau_y / sigma_w.

For the plane (slit) channel of gap ``H`` the same balance gives
``sigma(y) = rho*a*y`` (``y`` from the mid-plane), so the plug half-thickness is
``y_c = (H/2) * tau_y / sigma_w`` -- the *same* ratio to the half-width as in the
pipe.  Note the campaign brief quotes ``r_c = R(1 - tau_y/tau_w)``: the correct
plug relation is ``r_c = R * tau_y/tau_w``; the quantity ``R(1 - tau_y/tau_w)``
is the **thickness of the yielded annulus**.  This module exposes both
(:func:`hb_pipe_yield_radius` returns ``r_c`` and
``yielded_annulus_thickness``) so the benchmark can gate on either.

**(2) Couette with relative plate motion.**
In a *pure* plane Couette flow the shear stress is uniform across the gap, so
yield is global: either the whole gap shears (``sigma_w > tau_y``, linear
profile) or the whole gap is a rigid plug translating rigidly with the moving
plate (``sigma_w <= tau_y``).  There is no interior yield surface -- that
degeneracy is a negative control (:func:`hb_couette_regime`).  An interior yield
surface at exactly ``y_yield = h*tau_y/tau_w`` exists as soon as the stress
varies linearly across the gap, i.e. for Couette-Poiseuille with an *adverse*
gradient chosen so that the stress vanishes at the stationary wall
(:func:`hb_couette_adverse_profile`): then ``sigma(y) = beta*y``,
``tau_w = beta*h``, and the plug is the static layer ``0 <= y <= y_yield``.

Regularisation used by the lattice-Boltzmann solver
---------------------------------------------------
The H-B viscosity ``nu = tau_y/gamma_dot + K*gamma_dot**(n-1)`` is singular as
``gamma_dot -> 0``, which is exactly where the plug lives.  Following
Papanastasiou (1987) the yield term is smoothed,

    nu(g) = tau_y*(1 - exp(-m*g))/g + K*g**(n-1),

so the plug is a very viscous but finite fluid, ``nu_plug -> tau_y*m``.
``nu_max`` additionally clamps the plug viscosity (standard practice): the
clamped plug creeps at ``gamma_dot ~ tau_y/nu_max``, which must stay small
compared with the wall shear rate -- the benchmark reports that ratio.  The
momentum balance above is *unaffected* by the regularisation, and the
regularised constitutive stress is monotonic in ``g``, so the numerical yield
surface is defined unambiguously as the locus where the regularised stress
equals ``tau_y`` (:func:`yield_surface_gamma_dot`).

Units
-----
Everything is *kinematic* (per unit density): stresses are ``tau/rho``, ``K`` is
a kinematic consistency index, and the resulting viscosity is the lattice
viscosity ``nu = (tau_relax - 1/2)/3``.  With the lattice density ``rho ~ 1``
this is identical to dividing the dynamic quantities by the constant density,
which is exact in the incompressible limit used here.

The lattice-Boltzmann side (and the plug/yield-surface measurement) lives in
:mod:`tensorlbm.herschel_bulkley_lbm`.

References
----------
* Herschel, W.H. & Bulkley, R. (1926) *Kolloid Z.* 39, 291.
* Bingham, E.C. (1922) *Fluidity and Plasticity*, McGraw-Hill.
* Papanastasiou, T.C. (1987) *J. Rheol.* 31, 385 (regularisation).
* Chhabra, R.P. & Richardson, J.F. (2008) *Non-Newtonian Flow and Applied
  Rheology*, 2nd ed., Butterworth-Heinemann (pipe/slit H-B solutions).
* Bird, R.B., Armstrong, R.C. & Hassager, O. (1987) *Dynamics of Polymeric
  Liquids*, Vol. 1, Wiley (Couette-Poiseuille solutions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "HBRheology",
    "validate_rheology",
    "hb_regularized_viscosity",
    "hb_constitutive_stress",
    "yield_surface_gamma_dot",
    "hb_pipe_yield_radius",
    "hb_pipe_velocity_profile",
    "hb_pipe_flow_rate",
    "hb_pipe_wall_stress_from_flow_rate",
    "hb_pipe_pressure_gradient",
    "hb_plane_yield_half_height",
    "hb_plane_velocity_profile",
    "hb_plane_flow_rate",
    "hb_couette_regime",
    "hb_pipe_design_case",
    "hb_couette_adverse_wall_velocity",
    "hb_couette_adverse_profile",
    "hb_regularized_reference_profile",
    "slush_rheology",
]


# ---------------------------------------------------------------------------
# Rheology container + validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HBRheology:
    """Kinematic Herschel-Bulkley parameters (per unit density).

    Attributes:
        tau_y: Yield stress (kinematic). ``0`` -> no yield stress (power law).
        K:     Consistency index (kinematic), ``K > 0``.
        n:     Flow index, ``n > 0``. ``n < 1`` shear thinning, ``n = 1``
               Bingham, ``n > 1`` shear thickening.
    """

    tau_y: float
    K: float
    n: float

    def as_dict(self) -> dict[str, float]:
        return {"tau_y": float(self.tau_y), "K": float(self.K), "n": float(self.n)}

    def apparent_viscosity(self, gamma_dot: float) -> float:
        """Exact (unregularised) apparent viscosity ``tau_y/g + K g**(n-1)``."""
        g = float(gamma_dot)
        if g <= 0.0:
            return math.inf
        return self.tau_y / g + self.K * g ** (self.n - 1.0)


def validate_rheology(tau_y: float, K: float, n: float) -> HBRheology:
    """Validate H-B parameters and return the immutable container.

    ``tau_y < 0`` is rejected: a negative offset is not a yield stress and has
    no physical meaning (the material would "flow" under zero stress).
    """
    if not math.isfinite(tau_y) or tau_y < 0.0:
        msg = f"tau_y must be finite and >= 0, got {tau_y}"
        raise ValueError(msg)
    if not math.isfinite(K) or K <= 0.0:
        msg = f"K must be finite and > 0, got {K}"
        raise ValueError(msg)
    if not math.isfinite(n) or n <= 0.0:
        msg = f"n must be finite and > 0, got {n}"
        raise ValueError(msg)
    return HBRheology(float(tau_y), float(K), float(n))


# ---------------------------------------------------------------------------
# Regularised viscosity / constitutive stress / yield-surface locus
# ---------------------------------------------------------------------------


def hb_regularized_viscosity(
    gamma_dot: torch.Tensor | float,
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_min: float = 1.0e-8,
    nu_max: float | None = None,
) -> torch.Tensor:
    """Papanastasiou-regularised H-B kinematic viscosity.

        nu(g) = tau_y * (1 - exp(-m g)) / g + K * g**(n-1)

    The removable ``0/0`` singularity at ``g = 0`` is handled analytically
    (the bracket tends to ``tau_y*m``), so no epsilon division is needed.

    Args:
        gamma_dot: Shear rate (non-negative scalar or tensor).
        rheo: H-B parameters.
        m: Regularisation rate (units of 1/shear rate). ``m -> inf`` recovers
            the exact H-B law; large ``m`` stiffens the plug but also raises
            ``nu_plug ~ tau_y*m``.
        nu_min: Lower clamp (guards shear-thickening tail).
        nu_max: Optional upper clamp (``None`` -> unclamped).

    Returns:
        Viscosity tensor with the same shape as ``gamma_dot``.
    """
    if m <= 0.0:
        msg = f"m must be > 0, got {m}"
        raise ValueError(msg)
    g = gamma_dot if isinstance(gamma_dot, torch.Tensor) else torch.as_tensor(gamma_dot, dtype=torch.float64)
    g = torch.clamp(g, min=0.0)
    tiny = torch.tensor(1.0e-12, dtype=g.dtype, device=g.device)
    g_safe = torch.clamp(g, min=tiny)
    # (1 - exp(-m g)) / g  with the g -> 0 limit (m) used below the switch-over
    with torch.no_grad():
        ratio = (1.0 - torch.exp(-m * g_safe)) / g_safe
    ratio = torch.where(g > 1.0e-9, ratio, torch.full_like(g, m))
    nu = rheo.tau_y * ratio + rheo.K * torch.pow(g_safe, rheo.n - 1.0)
    nu = torch.clamp(nu, min=nu_min)
    if nu_max is not None:
        nu = torch.clamp(nu, max=nu_max)
    return nu


def hb_constitutive_stress(
    gamma_dot: torch.Tensor | float,
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_min: float = 1.0e-8,
    nu_max: float | None = None,
) -> torch.Tensor:
    """Regularised shear stress ``tau = nu_reg(g) * g`` (kinematic).

    ``nu_max`` is *not* re-applied here by default because the clamp is a
    numerical device applied when building the relaxation time; pass the same
    ``nu_max`` if the stress must be consistent with the collision operator.
    """
    g = gamma_dot if isinstance(gamma_dot, torch.Tensor) else torch.as_tensor(gamma_dot, dtype=torch.float64)
    nu = hb_regularized_viscosity(g, rheo, m=m, nu_min=nu_min, nu_max=nu_max)
    return nu * torch.clamp(g, min=0.0)


def yield_surface_gamma_dot(
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_max: float | None = None,
    nu_min: float = 1.0e-8,
) -> float:
    """Shear rate at which the regularised constitutive stress equals ``tau_y``.

    Solves ``nu_reg(g) * g = tau_y`` for the *same* regularised law (including
    the ``nu_max`` clamp) that the collision operator uses, so the numerically
    detected yield surface is consistent with the discretisation.  The function
    ``nu_reg(g)*g`` is monotone increasing in ``g`` (clamped viscosity times
    ``g`` near zero, ``tau_y + K g^n`` further out), so the root is unique.

    Without the clamp and for ``m -> inf`` this reduces to the exact H-B yield
    shear rate ``(tau_y/K)**(1/n)``... which for Bingham (``n = 1``) is *not*
    the physical yield shear rate (a Bingham fluid yields at ``g -> 0``); the
    regularised value ``g ~ tau_y/nu_plug`` is the correct lattice analogue and
    is what the plug-creep estimate uses.

    Returns:
        The yield-locus shear rate (``0.0`` when ``tau_y == 0``).
    """
    if rheo.tau_y <= 0.0:
        return 0.0
    hi = 10.0 * (rheo.tau_y / rheo.K) ** (1.0 / rheo.n)
    lo = 0.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        stress = float(
            hb_constitutive_stress(
                torch.tensor(mid, dtype=torch.float64), rheo, m=m,
                nu_min=nu_min, nu_max=nu_max,
            )
        )
        if stress < rheo.tau_y:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1.0e-15 * max(hi, 1.0e-300):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 1. Pipe (capillary) Poiseuille: plug core radius + flow-rate law
# ---------------------------------------------------------------------------


def hb_pipe_yield_radius(R: float, tau_w: float, tau_y: float) -> dict[str, float]:
    """Plug-core radius from the momentum balance.

    ``sigma(r) = tau_w * r/R`` for a fully developed pipe flow, so the yield
    surface sits at ``r_c = R * tau_y / tau_w`` (valid when ``tau_w > tau_y``).

    Returns a dict with:

    * ``r_c``: plug radius,
    * ``yielded_annulus_thickness``: ``R - r_c = R*(1 - tau_y/tau_w)``,
    * ``plug_fraction``: ``r_c/R = tau_y/tau_w``,
    * ``yielded_fraction``: ``1 - plug_fraction``,
    * ``plugged``: ``True`` when ``tau_w > tau_y`` (otherwise the whole
      cross-section is unyielded and ``r_c = R``).
    """
    if R <= 0.0:
        msg = f"R must be > 0, got {R}"
        raise ValueError(msg)
    if tau_w <= 0.0:
        msg = f"tau_w must be > 0, got {tau_w}"
        raise ValueError(msg)
    if tau_y < 0.0:
        msg = f"tau_y must be >= 0, got {tau_y}"
        raise ValueError(msg)
    plugged = tau_w > tau_y
    frac = min(1.0, tau_y / tau_w)
    r_c = R * frac
    return {
        "r_c": r_c,
        "plug_fraction": frac,
        "yielded_fraction": 1.0 - frac,
        "yielded_annulus_thickness": R * (1.0 - frac),
        "tau_w": tau_w,
        "plugged": plugged,
    }


def hb_pipe_velocity_profile(
    r: torch.Tensor,
    R: float,
    tau_w: float,
    rheo: HBRheology,
) -> torch.Tensor:
    """Exact H-B pipe velocity profile (no-slip at ``R``), body-force driven.

        u(r) = (n/(n+1)) K^(-1/n) (R/tau_w) *
               [ (tau_w - tau_y)^((n+1)/n) - (tau_w r/R - tau_y)^((n+1)/n) ]

    for ``r >= r_c``, and ``u = u_c = u(r_c)`` inside the plug.  Reduces to
    ``tau_w/(4K) R (1 - (r/R)^2)`` for Newtonian and to the power-law profile
    for ``tau_y = 0``.
    """
    p = (rheo.n + 1.0) / rheo.n
    pref = (rheo.n / (rheo.n + 1.0)) * rheo.K ** (-1.0 / rheo.n) * (R / tau_w)
    sig = tau_w * torch.clamp(r / R, min=0.0)
    inner = torch.clamp(sig - rheo.tau_y, min=0.0)
    u = pref * ((tau_w - rheo.tau_y) ** p - torch.pow(inner, p))
    # inside the plug (sig < tau_y) the bracket is constant -> u = u_c automatically
    return u


def hb_pipe_flow_rate(tau_w: float, R: float, rheo: HBRheology) -> float:
    """Exact H-B pipe flow rate ``Q(tau_w)`` (closed form, no quadrature).

    Derived by integrating the analytic profile, ``Q = 2*pi*int_0^R u r dr``:

        Q = 2 pi R^3 K^(-1/n) (n/(n+1)) tau_w^(1/n) (1-phi)^((n+1)/n) *
            [ 1/2 - n(1-phi)^2/(3n+1) - n phi (1-phi)/(2n+1) ],  phi = tau_y/tau_w.

    Limits (verified in the test suite): ``n=1`` -> Bing ham
    ``pi R^3 tau_w/(4K) (1 - 4phi/3 + phi^4/3)``; ``tau_y=0`` -> power law
    ``pi n R^3/(3n+1) (tau_w/K)^(1/n)``.
    """
    if tau_w <= 0.0:
        msg = f"tau_w must be > 0, got {tau_w}"
        raise ValueError(msg)
    if R <= 0.0:
        msg = f"R must be > 0, got {R}"
        raise ValueError(msg)
    phi = min(1.0, rheo.tau_y / tau_w)
    if phi >= 1.0:
        return 0.0
    n = rheo.n
    p = (n + 1.0) / n
    bracket = 0.5 - n * (1.0 - phi) ** 2 / (3.0 * n + 1.0) - n * phi * (1.0 - phi) / (2.0 * n + 1.0)
    return (
        2.0
        * math.pi
        * R**3
        * rheo.K ** (-1.0 / n)
        * (n / (n + 1.0))
        * tau_w ** (1.0 / n)
        * (1.0 - phi) ** p
        * bracket
    )


def hb_pipe_wall_stress_from_flow_rate(
    Q: float,
    R: float,
    rheo: HBRheology,
    tol: float = 1.0e-12,
    max_iter: int = 400,
) -> float:
    """Invert ``Q(tau_w)`` -> ``tau_w`` by bisection (``Q`` monotone in ``tau_w``).

    Raises:
        ValueError: when ``Q <= 0`` (no flow without stress) or when the
            bracketing fails (unreachable for physical parameters).
    """
    if Q <= 0.0:
        raise ValueError(f"Q must be > 0, got {Q}")
    lo = max(rheo.tau_y, 1.0e-30) * (1.0 + 1.0e-12)
    hi = max(lo * 10.0, 1.0e-12)
    while hb_pipe_flow_rate(hi, R, rheo) < Q:
        hi *= 2.0
        if hi > 1.0e30:  # pragma: no cover - unreachable for physical inputs
            msg = "failed to bracket tau_w for the requested Q"
            raise ValueError(msg)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hb_pipe_flow_rate(mid, R, rheo) < Q:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break
    return 0.5 * (lo + hi)


def hb_pipe_pressure_gradient(Q: float, R: float, rheo: HBRheology) -> float:
    """Pressure gradient ``dP/dx`` (per unit density) needed to pass ``Q``.

    Fully developed pipe balance: ``dP/dx = 2 tau_w / R``.
    """
    tau_w = hb_pipe_wall_stress_from_flow_rate(Q, R, rheo)
    return 2.0 * tau_w / R


# ---------------------------------------------------------------------------
# 2. Plane (slit) Poiseuille: plug half-height + flow-rate law
# ---------------------------------------------------------------------------


def hb_plane_yield_half_height(H: float, tau_w: float, tau_y: float) -> dict[str, float]:
    """Plug half-height of a plane channel of gap ``H`` (walls at ``+-H/2``).

    ``sigma(y) = tau_w * 2|y|/H`` -> ``y_c = (H/2) tau_y/tau_w``; the plug
    thickness is ``H tau_y/tau_w``.
    """
    if tau_w <= 0.0:
        raise ValueError(f"tau_w must be > 0, got {tau_w}")
    if tau_y < 0.0:
        raise ValueError(f"tau_y must be >= 0, got {tau_y}")
    frac = min(1.0, tau_y / tau_w)
    return {
        "y_c": 0.5 * H * frac,
        "plug_fraction": frac,
        "plug_thickness": H * frac,
        "plugged": tau_w > tau_y,
        "tau_w": tau_w,
    }


def hb_plane_velocity_profile(
    y: torch.Tensor,
    H: float,
    tau_w: float,
    rheo: HBRheology,
) -> torch.Tensor:
    """Exact H-B plane-Poiseuille velocity profile, walls at ``+-H/2``.

    ``u(y) = (n/(n+1)) K^(-1/n) (H/(2 tau_w)) [ (tau_w - tau_y)^((n+1)/n)
    - (2 tau_w |y|/H - tau_y)^((n+1)/n) ]`` with ``u = u_plug`` for
    ``|y| <= y_c``.  Newtonian limit: ``u = tau_w H/(4K) (1 - (2y/H)^2)``.
    """
    p = (rheo.n + 1.0) / rheo.n
    pref = (rheo.n / (rheo.n + 1.0)) * rheo.K ** (-1.0 / rheo.n) * (H / (2.0 * tau_w))
    sig = tau_w * 2.0 * torch.clamp(torch.abs(y) / H, min=0.0)
    inner = torch.clamp(sig - rheo.tau_y, min=0.0)
    return pref * ((tau_w - rheo.tau_y) ** p - torch.pow(inner, p))


def hb_plane_flow_rate(tau_w: float, H: float, rheo: HBRheology) -> float:
    """Exact H-B plane-channel flow rate per unit width.

        q = H^2/2 K^(-1/n) tau_w^(1/n) (n/(n+1)) (1-phi)^((n+1)/n) *
            [ 1 - n(1-phi)/(2n+1) ],   phi = tau_y/tau_w.

    Newtonian limit: ``q = H^2 tau_w/(6K)`` (= ``a H^3/(12 nu)``).
    """
    if tau_w <= 0.0:
        raise ValueError(f"tau_w must be > 0, got {tau_w}")
    phi = min(1.0, rheo.tau_y / tau_w)
    if phi >= 1.0:
        return 0.0
    n = rheo.n
    p = (n + 1.0) / n
    return (
        0.5
        * H**2
        * rheo.K ** (-1.0 / n)
        * tau_w ** (1.0 / n)
        * (n / (n + 1.0))
        * (1.0 - phi) ** p
        * (1.0 - n * (1.0 - phi) / (2.0 * n + 1.0))
    )


# ---------------------------------------------------------------------------
# 3. Plane Couette: global yield (no interior surface) and the
#    adverse-gradient Couette-Poiseuille case with y_yield = h tau_y/tau_w
# ---------------------------------------------------------------------------


def hb_couette_regime(h: float, tau_w: float, tau_y: float) -> dict[str, float | str | bool]:
    """Yield state of a *pure* plane Couette gap of height ``h``.

    Pure Couette has a uniform stress ``tau = tau_w`` across the gap, so the
    unyielded set is not an interior surface:

    * ``tau_w > tau_y``  -> ``regime = "sheared"``: the whole gap yields, the
      profile is linear, ``gamma_dot = ((tau_w - tau_y)/K)**(1/n)``, and the
      fraction of the gap that is plug is **0**.
    * ``tau_w <= tau_y`` -> ``regime = "rigid_plug"``: nothing yields, the
      material translates rigidly with the moving plate (``gamma_dot = 0``),
      and the plug fraction is **1**.

    ``yield_surface_y`` is therefore ``None`` -- documented degeneracy, not a
    missing feature.
    """
    if tau_w > tau_y:
        return {
            "regime": "sheared",
            "plug_fraction": 0.0,
            "yield_surface_y": None,
            "gamma_dot": None,  # filled by hb_couette_profile (needs K, n)
            "tau_w": tau_w,
        }
    return {
        "regime": "rigid_plug",
        "plug_fraction": 1.0,
        "yield_surface_y": None,
        "gamma_dot": 0.0,
        "tau_w": tau_w,
    }


def hb_pipe_design_case(
    n: float,
    phi: float,
    u_target: float,
    nu_ref: float,
    R: float,
) -> dict[str, float | HBRheology]:
    """Pick H-B parameters that put a pipe simulation in a well-conditioned regime.

    Inverting the pipe solution for a *chosen* plug fraction ``phi = tau_y/tau_w``,
    wall shear rate ``g_w`` and wall apparent viscosity ``nu_ref`` gives a very
    compact set (see the module docstring for the momentum balance):

        g_w = u_target (n+1) / (n R (1-phi))       [from u_c = (n/(n+1)) R g_w (1-phi)]
        K   = nu_ref (1-phi) g_w**(1-n)
        tau_y = phi nu_ref g_w
        a   = 2 nu_ref g_w / R                     [body force]

    so ``nu(g_w) = tau_y/g_w + K g_w**(n-1) = nu_ref/(1-phi)``... note the
    apparent viscosity at the wall is ``nu_ref/(1-phi)`` by construction, i.e.
    ``nu_ref`` is the *power-law part* ``K g_w**(n-1)``.  This helper exists to
    document the parameter choice of the benchmark in one place instead of
    scattering magic numbers.

    Args:
        n: Flow index.
        phi: Target plug fraction ``tau_y/tau_w`` in ``(0, 1)``.
        u_target: Target centreline (plug) velocity in lattice units.
        nu_ref: Target power-law viscosity at the wall ``K g_w**(n-1)``.
        R: Pipe radius in lattice units.

    Returns:
        dict with ``g_w``, ``a``, ``tau_w``, ``u_c``, ``nu_wall`` and the
        ``rheo`` (:class:`HBRheology`).
    """
    if not 0.0 < phi < 1.0:
        raise ValueError(f"phi must be in (0, 1), got {phi}")
    if n <= 0.0 or R <= 0.0 or u_target <= 0.0 or nu_ref <= 0.0:
        raise ValueError("n, R, u_target and nu_ref must all be > 0")
    g_w = u_target * (n + 1.0) / (n * R * (1.0 - phi))
    K = nu_ref * (1.0 - phi) * g_w ** (1.0 - n)
    tau_y = phi * nu_ref * g_w
    tau_w = nu_ref * g_w
    rheo = validate_rheology(tau_y, K, n)
    return {
        "g_w": g_w,
        "a": 2.0 * tau_w / R,
        "tau_w": tau_w,
        "u_c": u_target,
        "nu_wall": rheo.tau_y / g_w + K * g_w ** (n - 1.0),
        "rheo": rheo,
    }


def hb_couette_adverse_wall_velocity(
    beta: float,
    h: float,
    rheo: HBRheology,
) -> dict[str, float]:
    """Plate speed that makes a Couette-Poiseuille gap have ``sigma(0) = 0``.

    Configuration: lower plate at rest (``y = 0``), upper plate at speed ``U``
    (``y = h``), uniform decelerating body force ``-beta`` (equivalently an
    adverse pressure gradient ``dP/dx = beta``).  The stress is then
    ``sigma(y) = beta*y`` (vanishing at the lower plate), the plug is the static
    layer ``0 <= y <= y_yield`` with

        y_yield = tau_y / beta = h * tau_y / tau_w,   tau_w = beta*h,

    and the sheared layer ``y_yield <= y <= h`` follows the H-B law.  This dst
    dict returns the required plate speed

        U = (n/(n+1)) K^(-1/n) beta^(-1) (tau_w - tau_y)^((n+1)/n)

    together with ``y_yield``, ``tau_w`` and the plug fraction ``y_yield/h``.
    Requires ``tau_w > tau_y`` (otherwise the whole gap is a static plug).
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    tau_w = beta * h
    if tau_w <= rheo.tau_y:
        raise ValueError(
            f"tau_w = beta*h = {tau_w} must exceed tau_y = {rheo.tau_y} for a yielded layer"
        )
    p = (rheo.n + 1.0) / rheo.n
    U = (
        (rheo.n / (rheo.n + 1.0))
        * rheo.K ** (-1.0 / rheo.n)
        / beta
        * (tau_w - rheo.tau_y) ** p
    )
    return {
        "U": U,
        "tau_w": tau_w,
        "y_yield": rheo.tau_y / beta,
        "plug_fraction": rheo.tau_y / tau_w,
        "gamma_dot_wall": ((tau_w - rheo.tau_y) / rheo.K) ** (1.0 / rheo.n),
    }


def hb_couette_adverse_profile(
    y: torch.Tensor,
    beta: float,
    h: float,
    rheo: HBRheology,
) -> torch.Tensor:
    """Exact velocity profile of the adverse-gradient Couette-Poiseuille case.

        u(y) = 0                                                    y <= y_yield
        u(y) = (n/(n+1)) K^(-1/n) beta^(-1) (beta y - tau_y)^((n+1)/n) otherwise

    i.e. a static plug of thickness ``y_yield = tau_y/beta = h tau_y/tau_w``
    glued to the stationary plate, and a sheared H-B layer above it reaching
    ``U = hb_couette_adverse_wall_velocity(...)["U"]`` at ``y = h``.
    """
    yy = torch.as_tensor(y, dtype=torch.float64)
    p = (rheo.n + 1.0) / rheo.n
    pref = (rheo.n / (rheo.n + 1.0)) * rheo.K ** (-1.0 / rheo.n) / beta
    inner = torch.clamp(beta * yy - rheo.tau_y, min=0.0)
    return pref * torch.pow(inner, p)


def hb_regularized_reference_profile(
    coord: torch.Tensor,
    stress: torch.Tensor,
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_min: float = 1.0e-8,
    nu_max: float | None = None,
    wall_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    """1-D reference solution of the *regularised* H-B model.

    A regularised yield-stress model does not have a truly rigid plug: where the
    momentum balance gives ``sigma < tau_y`` the material creeps at
    ``gamma_dot = sigma/nu_plug``.  The plug extent that a lattice simulation
    can reproduce is therefore the plug of the *regularised* model, not of the
    ideal H-B law.  This routine solves that model exactly in one dimension:

    1. per radius/cross-stream station, solve ``nu_reg(g) g = sigma`` for ``g``
       (monotone in ``g`` -> bisection, unique root);
    2. integrate ``u`` inwards from the wall, where ``u = wall_value``.

    Comparing a lattice plug radius against this profile isolates *discretisation*
    error; comparing it against the ideal H-B profile
    (:func:`hb_pipe_velocity_profile`) additionally contains the *regularisation*
    bias.  Both numbers are reported by the benchmark.

    Args:
        coord: Ascending coordinates measured from the unyielded side (axis or
            mid-plane); the wall is the **last** entry.
        stress: Shear-stress magnitude at each coordinate (monotone increasing
            towards the wall), e.g. ``a*r/2`` for a pipe.
        rheo: H-B parameters.
        m, nu_min, nu_max: Regularisation / clamping used by the solver.
        wall_value: Velocity at the wall (no-slip -> 0).

    Returns:
        dict with ``gamma`` (shear rate), ``u`` (velocity, ``u[-1] =
        wall_value``) and ``coord``.
    """
    if coord.shape != stress.shape:
        msg = f"coord {tuple(coord.shape)} and stress {tuple(stress.shape)} must match"
        raise ValueError(msg)
    # vectorised bisection on nu_reg(g) g = sigma with a common bracket derived
    # from the largest stress (the constitutive stress is monotone in g)
    sig_max = float(stress.max())
    hi = torch.full_like(stress, 1.0)
    for _ in range(200):
        if float(hb_constitutive_stress(hi, rheo, m=m, nu_min=nu_min, nu_max=nu_max).min()) >= sig_max:
            break
        hi = hi * 2.0
    lo = torch.zeros_like(stress)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        below = hb_constitutive_stress(mid, rheo, m=m, nu_min=nu_min, nu_max=nu_max) < stress
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
        if float((hi - lo).max()) <= 1.0e-16 * max(float(hi.max()), 1.0e-300):
            break
    gamma = 0.5 * (lo + hi)
    # trapezoidal integration inwards from the wall (last entry)
    increments = 0.5 * (gamma[1:] + gamma[:-1]) * (coord[1:] - coord[:-1])
    rev = torch.flip(increments, [0]).cumsum(0)
    u = torch.zeros_like(stress)
    u[:-1] = wall_value + torch.flip(rev, [0])
    u[-1] = wall_value
    return {"gamma": gamma, "u": u, "coord": coord}


# ---------------------------------------------------------------------------
# 4. Slush (ice-slurry) closure: concentration -> H-B parameters
# ---------------------------------------------------------------------------


def slush_rheology(
    phi: float,
    *,
    phi_max: float = 0.60,
    intrinsic_viscosity: float = 2.5,
    n_water: float = 1.0,
    n_packed: float = 0.45,
    phi_yield: float = 0.10,
    tau_y_ref: float = 2.0,
    yield_exponent: float = 2.5,
) -> HBRheology:
    """Ice-slurry (slush) concentration -> H-B parameters.

    Closures (all *representative*, not fitted to any particular slurry):

    * **Consistency index.** Krieger-Dougherty effective-medium viscosity of the
      carrier liquid, ``mu_r(phi) = (1 - phi/phi_max)^(-[eta] phi_max)`` with
      ``[eta] = 2.5``, ``phi_max = 0.60`` (same closure as
      :mod:`tensorlbm.frazil_rheology`), used as ``K = mu_r(phi)`` in lattice
      units where the carrier viscosity is 1.
    * **Flow index.** Linear blend from clear water (``n = 1``) to the packed
      floc limit ``n_packed = 0.45``: ``n(phi) = 1 - (1 - n_packed) phi/phi_max``
      (flocs break down under shear -> shear thinning grows with ice fraction;
      Daly 1984, Barnes 1997).
    * **Yield stress.** Slush develops a yield stress only above a percolation
      threshold ``phi_yield ~ 0.10``; beyond it the network strength grows as a
      power law ``tau_y(phi) = tau_y_ref ((phi - phi_yield)/(phi_max - phi_yield))
      ** yield_exponent`` (exponent 2.5 in the range quoted for ice slurries /
      frazil floes: ~2-3).

    Args:
        phi: Ice volume fraction, ``0 <= phi < phi_max``.

    Returns:
        :class:`HBRheology` in kinematic lattice units (carrier viscosity 1).

    Raises:
        ValueError: for ``phi`` outside ``[0, phi_max)`` -- extrapolating past
            the packing limit is meaningless (the material is no longer
            flowing) and is refused rather than silently returning a number.
    """
    if not math.isfinite(phi) or phi < 0.0 or phi >= phi_max:
        msg = f"phi must satisfy 0 <= phi < phi_max={phi_max}, got {phi}"
        raise ValueError(msg)
    mu_r = (1.0 - phi / phi_max) ** (-intrinsic_viscosity * phi_max)
    n = n_water - (n_water - n_packed) * (phi / phi_max)
    if phi <= phi_yield:
        tau_y = 0.0
    else:
        s = (phi - phi_yield) / (phi_max - phi_yield)
        tau_y = tau_y_ref * s**yield_exponent
    return validate_rheology(tau_y, mu_r, n)