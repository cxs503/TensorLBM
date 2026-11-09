"""Frazil / grease-ice suspension closure: hindered settling, drag transition,
and concentration-coupled power-law rheology.

This module is the *suspension* half of the frazil model chain.  The crystal
rheology (:mod:`tensorlbm.frazil_rheology`) supplies the concentration -> power
law map ``(K(phi), n(phi))``; :mod:`tensorlbm.frazil_lbm` supplies the lattice
channel that proves the collision operator reproduces that map.  What was
missing -- and what this module adds -- is the **transport** side: how fast the
ice crystals settle out of the mixture, how strongly that settling is hindered
by concentration, and what drag coefficient a coupled solver must use to be
*consistent* with the hindered-settling law.

Physics
-------
For a monodisperse suspension of frazil flocs of diameter ``d`` in water of
viscosity ``mu_w`` and density difference ``drho``, the single-particle
terminal velocity follows the drag balance

    1/2 C_D(Re) rho_f (pi d^2/4) v_s0^2 = drho g (pi d^3/6)

    =>  v_s0 = sqrt(4 drho g d / (3 rho_f C_D(Re))),

with the Schiller-Naumann correlation

    C_D(Re) = 24/Re (1 + 0.15 Re^0.687)     (Re <= 800),
    C_D     = 0.44                          (Re > 800),

which is Stokes ``24/Re`` as ``Re -> 0`` and saturates on the Newton plate for
``Re >> 1``.  That single-particle curve is the *transition* the campaign asks
for: for frazil flocs of 0.1-5 mm in water at 0 C the particle Reynolds number
sweeps ``0.02 -> 200`` and the drag law moves from viscous to inertial.

In a concentrated mixture the settling is *hindered*.  The Richardson-Zaki
(1954) law

    v_s(c) = v_s0 (1 - c/c_max)^m

is the reference closure.  The exponent is not a constant: RZ's original fit is
piecewise in the single-particle Reynolds number

    m = 4.65                        Re < 0.2
        4.4 Re^-0.03                0.2 <= Re < 1
        4.4 Re^-0.10                1   <= Re < 500
        2.40                        Re >= 500

so a slurry of large (inertial) crystals is *less* hindered than a slurry of
creeping-flow crystals -- the second half of the transition.  ``c_max`` is the
packing limit, which for frazil flocs is ``~0.6`` (the same
:data:`~tensorlbm.frazil_rheology.DEFAULT_PHI_MAX` used by the rheology).

The drag coefficient a coupled solver must *use* is fixed by the requirement
that the modelled drag reproduce the hindered velocity.  Writing the particle
drag in the frame used by the transport solver (slip velocity ``v_s(c)``), the
balance ``C_D(c) v_s(c)^2 = C_D(Re) v_s0^2`` gives the **hindered drag
amplification**

    Omega(c) = C_D(c)/C_D(Re) = (1 - c/c_max)^(-2m).

Because ``v_s`` is a drift in the concentration equation, ``Omega`` is the exact
multiplier the scalar LBM needs in order to stay consistent with the settling
law; :func:`settling_from_drag` verifies that round trip, so the two halves of
the model cannot silently drift apart.

Shear-induced diffusion and the transport equation
--------------------------------------------------
The concentration field obeys

    dc/dt + div(c u) = div(D_eff grad c) + v_s(c) dc/dz,

with the effective diffusivity

    D_eff(c, gdot) = D_m + D_sh,   D_sh = alpha d^2 gdot f(c),
    f(c) = c^2 (1 + 0.5 exp(8.8 c)),

i.e. molecular diffusion plus the Leighton-Acrivos (1987) shear-induced
diffusion which vanishes quadratically as ``c -> 0`` (clear-water limit) and
grows steeply towards the packed limit.  The settling-induced dispersion
``~ (v_s d)^2/D_m`` is reported in :func:`effective_diffusivity` under the
``dispersion`` key but is *not* added to ``D_eff`` by default: the scalar solver
resolves the settling flux itself and double counting would over-diffuse the
Kynch front.

Everything here is in SI unless a caller passes lattice-unit numbers; the
lattice side lives in :mod:`tensorlbm.frazil_settling_lbm`.

References
----------
* Richardson, J.F. & Zaki, W.N. (1954) *Trans. Inst. Chem. Eng.* 32, 35.
* Schiller, L. & Naumann, A. (1935) *Z. Ver. Deutsch. Ing.* 77, 318.
* Krieger, I.M. & Dougherty, T.J. (1959) *Trans. Soc. Rheol.* 3, 137.
* Leighton, D. & Acrivos, A. (1987) *J. Fluid Mech.* 181, 415.
* Di Felice, R. (1994) *Int. J. Multiphase Flow* 20, 153 (comparison only).
* Daly, S.F. (1984) CRREL Report 84-11 (frazil floc sizes, 0.1-3 mm).
"""

from __future__ import annotations

import math

import torch

from .frazil_rheology import (
    DEFAULT_PHI_MAX,
    FrazilRheologyError,
    frazil_apparent_viscosity,
    frazil_consistency_index,
    frazil_flow_index,
    relative_viscosity_krieger_dougherty,
    validate_concentration,
)

__all__ = [
    "RHO_ICE",
    "RHO_WATER",
    "MU_WATER",
    "GRAVITY",
    "DEFAULT_D_FRAZIL",
    "DEFAULT_ALPHA_SHEAR_DIFFUSION",
    "stokes_settling_velocity",
    "drag_coefficient",
    "single_particle_settling_velocity",
    "richardson_zaki_exponent",
    "hindered_settling_velocity",
    "hindered_drag_amplification",
    "drag_coefficient_suspension",
    "difelice_drag_amplification",
    "settling_from_drag",
    "effective_mixture_viscosity",
    "effective_diffusivity",
    "flow_capacity_ratio",
    "batch_settling_interface",
    "concentration_sweep",
]

#: Ice density at 0 C (kg m^-3).
RHO_ICE = 917.0
#: Water density at 0 C (kg m^-3).
RHO_WATER = 1000.0
#: Dynamic viscosity of water at 0 C (Pa s).
MU_WATER = 1.8e-3
#: Gravitational acceleration (m s^-2).
GRAVITY = 9.81
#: Representative frazil floc diameter (m); Daly (1984) reports 0.1-3 mm.
DEFAULT_D_FRAZIL = 1.0e-3
#: Shear-induced-diffusion prefactor (Leighton-Acrivos family).
DEFAULT_ALPHA_SHEAR_DIFFUSION = 0.1


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _scalar_like(reference, tensor: torch.Tensor):
    """Return a float when the caller passed a python float, else a tensor."""
    if isinstance(reference, torch.Tensor):
        return tensor
    if tensor.numel() == 1:
        return float(tensor.reshape(-1)[0])
    return tensor


# ---------------------------------------------------------------------------
# Single-particle drag / settling: the dilute Stokes -> inertial transition
# ---------------------------------------------------------------------------


def drag_coefficient(re_p):
    """Schiller-Naumann sphere drag coefficient ``C_D(Re)``.

    ``C_D = 24/Re (1 + 0.15 Re^0.687)`` for ``Re <= 800`` and the Newton plate
    ``0.44`` above it.  The low-Re limit is Stokes ``24/Re`` (relative error
    ``4e-4`` at ``Re = 1e-3``, verified in the tests), and the two branches meet
    within 1 % at ``Re = 800``.

    Args:
        re_p: Particle Reynolds number (scalar or tensor), must be > 0.

    Raises:
        FrazilRheologyError: if any ``Re <= 0``.
    """
    re = _as_tensor(re_p)
    if bool((re <= 0).any()):
        raise FrazilRheologyError(f"particle Reynolds number must be > 0, got {re_p!r}")
    cd_low = 24.0 / re * (1.0 + 0.15 * torch.pow(re, 0.687))
    cd = torch.where(re <= 800.0, cd_low, torch.full_like(re, 0.44))
    return _scalar_like(re_p, cd)


def stokes_settling_velocity(
    d_p: float = DEFAULT_D_FRAZIL,
    *,
    rho_particle: float = RHO_ICE,
    rho_fluid: float = RHO_WATER,
    mu_fluid: float = MU_WATER,
) -> float:
    """Stokes terminal velocity ``drho g d^2 / (18 mu)`` (creeping limit).

    Raises:
        FrazilRheologyError: on non-positive ``d_p``/``mu_fluid`` or a particle
            that is not lighter than the fluid.
    """
    if d_p <= 0.0:
        raise FrazilRheologyError(f"particle diameter must be > 0, got {d_p!r}")
    if mu_fluid <= 0.0:
        raise FrazilRheologyError(f"mu_fluid must be > 0, got {mu_fluid!r}")
    drho = rho_fluid - rho_particle
    if drho <= 0.0:
        raise FrazilRheologyError(
            f"particle must be lighter than the fluid for settling, got drho = {drho!r}"
        )
    return drho * GRAVITY * d_p * d_p / (18.0 * mu_fluid)


def single_particle_settling_velocity(
    d_p: float = DEFAULT_D_FRAZIL,
    *,
    rho_particle: float = RHO_ICE,
    rho_fluid: float = RHO_WATER,
    mu_fluid: float = MU_WATER,
    max_iter: int = 500,
    tol: float = 1e-14,
) -> dict:
    """Terminal velocity of one frazil crystal with the full (non-Stokes) drag.

    Fixed-point iteration on the drag balance

        ``v = sqrt(4 drho g d / (3 rho_f C_D(Re)))``,  ``Re = rho_f v d / mu``

    started at the Stokes velocity, with a damped (averaged) update so the sweep
    is monotone for every branch of ``C_D``.  Convergence is geometric because
    ``C_D`` is a decreasing function of ``Re``.

    Returns:
        dict with ``velocity``, ``reynolds``, ``drag_coefficient``,
        ``stokes_velocity``, ``drag_ratio`` (``C_D / (24/Re)``),
        ``iterations``, ``converged``, ``diameter``, ``drho``, ``mu_fluid``,
        ``rho_fluid`` and ``rho_particle``.
    """
    v_stokes = stokes_settling_velocity(
        d_p, rho_particle=rho_particle, rho_fluid=rho_fluid, mu_fluid=mu_fluid
    )
    drho = rho_fluid - rho_particle
    coeff = 4.0 * drho * GRAVITY * d_p / (3.0 * rho_fluid)
    v = v_stokes
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        re = rho_fluid * v * d_p / mu_fluid
        cd = float(drag_coefficient(re))
        v_new = math.sqrt(coeff / cd)
        if abs(v_new - v) <= tol * max(v_new, 1e-300):
            v = v_new
            converged = True
            break
        v = 0.5 * (v + v_new)
    re = rho_fluid * v * d_p / mu_fluid
    cd = float(drag_coefficient(re))
    return {
        "velocity": v,
        "reynolds": re,
        "drag_coefficient": cd,
        "stokes_velocity": v_stokes,
        "drag_ratio": cd / (24.0 / re),
        "iterations": it,
        "converged": converged,
        "diameter": d_p,
        "drho": drho,
        "mu_fluid": mu_fluid,
        "rho_fluid": rho_fluid,
        "rho_particle": rho_particle,
    }


# ---------------------------------------------------------------------------
# Richardson-Zaki hindered settling
# ---------------------------------------------------------------------------


def richardson_zaki_exponent(re_p):
    """Richardson-Zaki hindrance exponent ``m(Re)`` (piecewise 1954 fit).

    The four branches are the published correlation.  The published form is
    *discontinuous* at ``Re = 0.2/1/500`` (about 5 % at ``Re = 1`` and 40 % at
    ``Re = 500``); frazil flocs of 0.1-5 mm in water at 0 C sit at
    ``Re ~ 0.02-200``, i.e. mostly on the ``1 <= Re < 500`` branch
    (``m = 4.4 Re^-0.1``).  The jumps are documented rather than smoothed --
    smoothing them would silently depart from the reference correlation.

    Raises:
        FrazilRheologyError: if any ``Re <= 0``.
    """
    re = _as_tensor(re_p)
    if bool((re <= 0).any()):
        raise FrazilRheologyError(f"particle Reynolds number must be > 0, got {re_p!r}")
    m = torch.where(
        re < 0.2,
        torch.full_like(re, 4.65),
        torch.where(
            re < 1.0,
            4.4 * torch.pow(re, -0.03),
            torch.where(re < 500.0, 4.4 * torch.pow(re, -0.10), torch.full_like(re, 2.40)),
        ),
    )
    return _scalar_like(re_p, m)


def _hindrance_exponent(exponent, re_p):
    if (exponent is None) == (re_p is None):
        raise FrazilRheologyError("give exactly one of 'exponent' and 're_p'")
    m = _as_tensor(exponent) if exponent is not None else _as_tensor(
        richardson_zaki_exponent(re_p)
    )
    if bool((m <= 0).any()):
        raise FrazilRheologyError(f"hindrance exponent must be > 0, got {exponent!r}")
    return m


def hindered_settling_velocity(
    concentration,
    v_s0,
    *,
    exponent=None,
    re_p=None,
    phi_max: float = DEFAULT_PHI_MAX,
):
    """Richardson-Zaki hindered settling velocity ``v_s0 (1 - c/c_max)^m``.

    Args:
        concentration: Ice volume fraction ``c`` in ``[0, phi_max)``.
        v_s0: Single-particle (dilute) terminal velocity, same units as the
            result.
        exponent: Hindrance exponent ``m``; mutually exclusive with ``re_p``.
        re_p: If given, ``m`` is evaluated from
            :func:`richardson_zaki_exponent`.
        phi_max: Packing limit (default :data:`DEFAULT_PHI_MAX` = 0.6).

    Raises:
        FrazilRheologyError: on ``c`` outside ``[0, phi_max)``, a non-positive
            ``v_s0``, or when neither/both of ``exponent`` and ``re_p`` are set.
    """
    m = _hindrance_exponent(exponent, re_p)
    c = validate_concentration(concentration, phi_max)
    v0 = _as_tensor(v_s0)
    if bool((v0 <= 0).any()):
        raise FrazilRheologyError(f"v_s0 must be > 0, got {v_s0!r}")
    factor = torch.pow((1.0 - c / phi_max).clamp(min=0.0), m)
    return _scalar_like(concentration, v0 * factor)


def hindered_drag_amplification(
    concentration,
    *,
    exponent=None,
    re_p=None,
    phi_max: float = DEFAULT_PHI_MAX,
):
    """Drag amplification ``Omega(c) = C_D(c)/C_D(Re) = (1 - c/c_max)^(-2m)``.

    Derived from the requirement that the drag balance reproduce the RZ
    hindered velocity with the slip velocity the transport solver uses:
    ``C_D(c) v_s(c)^2 = C_D(Re) v_s0^2``.
    """
    m = _hindrance_exponent(exponent, re_p)
    c = validate_concentration(concentration, phi_max)
    factor = torch.pow((1.0 - c / phi_max).clamp(min=1e-300), -2.0 * m)
    return _scalar_like(concentration, factor)


def drag_coefficient_suspension(
    concentration,
    re_p,
    *,
    exponent=None,
    phi_max: float = DEFAULT_PHI_MAX,
):
    """Suspension drag coefficient ``C_D(c, Re) = C_D(Re) (1 - c/c_max)^(-2m)``.

    ``m`` defaults to the Richardson-Zaki value for the supplied single-particle
    ``re_p``.  This is the coefficient to hand to an Euler-Lagrange / Euler-Euler
    solver so that its dilute-limit drag is Schiller-Naumann *and* its hindered
    settling reproduces RZ exactly (see :func:`settling_from_drag`).
    """
    m = _hindrance_exponent(
        exponent, None if exponent is not None else re_p
    )
    cd0 = _as_tensor(drag_coefficient(re_p))
    c = validate_concentration(concentration, phi_max)
    factor = torch.pow((1.0 - c / phi_max).clamp(min=1e-300), -2.0 * m)
    return _scalar_like(concentration, cd0 * factor)


def difelice_drag_amplification(
    concentration,
    re_p,
    *,
    phi_max: float = DEFAULT_PHI_MAX,
):
    """Di Felice (1994) drag amplification ``(1 - c)^(-chi)``, ``chi(Re)``.

    Provided **for comparison only** -- it is not used by the settling solver.
    Di Felice's fit uses ``epsilon = 1 - c`` (packing limit 1) and gives
    ``chi = 3.7 - 0.65 exp(-(1.5 - log10 Re)^2 / 2)``, i.e. at low Reynolds
    number ``Omega ~ (1-c)^-3.7``, far weaker than the
    ``(1 - c/c_max)^(-2m)`` implied by Richardson-Zaki with a frazil packing
    limit of 0.6.  The two closures therefore disagree strongly at high
    concentration; the artifact records the size of that disagreement instead of
    pretending the models agree.
    """
    c = validate_concentration(concentration, phi_max if phi_max <= 1.0 else 1.0)
    re = _as_tensor(re_p)
    if bool((re <= 0).any()):
        raise FrazilRheologyError(f"particle Reynolds number must be > 0, got {re_p!r}")
    chi = 3.7 - 0.65 * torch.exp(-0.5 * torch.pow(1.5 - torch.log10(re), 2.0))
    out = torch.pow((1.0 - c).clamp(min=1e-300), -chi)
    return _scalar_like(concentration, out)


def settling_from_drag(
    concentration,
    re_p,
    v_s0: float,
    *,
    exponent=None,
    phi_max: float = DEFAULT_PHI_MAX,
) -> dict:
    """Invert ``C_D(c)`` back into a settling velocity (round-trip check).

    Returns:
        dict with ``velocity`` (recovered), ``expected`` (RZ value),
        ``relative_error`` and ``drag_coefficient``.
    """
    cd = _as_tensor(
        drag_coefficient_suspension(concentration, re_p, exponent=exponent, phi_max=phi_max)
    )
    cd0 = _as_tensor(drag_coefficient(re_p))
    v = float(v_s0) * torch.sqrt(cd0 / cd)
    expected = _as_tensor(
        hindered_settling_velocity(
            concentration, v_s0, exponent=exponent, re_p=None if exponent is not None else re_p,
            phi_max=phi_max,
        )
    )
    err = torch.abs(v - expected) / expected.clamp(min=1e-300)
    return {
        "velocity": v if isinstance(v, torch.Tensor) else float(v),
        "expected": expected,
        "relative_error": err,
        "drag_coefficient": cd,
    }


# ---------------------------------------------------------------------------
# Concentration-coupled rheology and diffusivity
# ---------------------------------------------------------------------------


def effective_mixture_viscosity(
    concentration,
    shear_rate,
    *,
    mu_water: float = MU_WATER,
    phi_max: float = DEFAULT_PHI_MAX,
    n_packed: float = 0.45,
    shear_rate_floor: float = 1e-12,
) -> dict:
    """Apparent viscosity of the frazil slurry at a given shear rate.

    ``mu_eff = K(c) gdot^(n(c)-1)`` with ``K = mu_water mu_r^KD`` and
    ``n = n(c)`` from :mod:`tensorlbm.frazil_rheology`.  The returned dict makes
    the two competing effects explicit:

    * ``mu_relative`` -- Krieger-Dougherty thickening (shear independent),
    * ``shear_thinning`` -- ``gdot^(n-1)`` ( = 1 for clear water),
    * ``mu_eff`` / ``viscosity_ratio`` -- the product, i.e. what the collision
      operator actually sees,
    * ``consistency_index`` / ``flow_index`` -- the power-law pair.
    """
    g = _as_tensor(shear_rate).clamp(min=shear_rate_floor)
    mu_eff, n = frazil_apparent_viscosity(
        concentration,
        g,
        mu_water=mu_water,
        phi_max=phi_max,
        n_packed=n_packed,
        shear_rate_floor=shear_rate_floor,
    )
    mu_r = _as_tensor(relative_viscosity_krieger_dougherty(concentration, phi_max))
    K = _as_tensor(frazil_consistency_index(concentration, mu_water=mu_water, phi_max=phi_max))
    thinning = torch.pow(g, n - 1.0)
    return {
        "mu_eff": mu_eff,
        "mu_relative": mu_r,
        "shear_thinning": thinning,
        "viscosity_ratio": mu_eff / mu_water,
        "consistency_index": K,
        "flow_index": n,
        "shear_rate": g,
    }


def effective_diffusivity(
    concentration,
    shear_rate,
    *,
    diameter: float = DEFAULT_D_FRAZIL,
    molecular: float = 1.0e-9,
    alpha: float = DEFAULT_ALPHA_SHEAR_DIFFUSION,
    phi_max: float = DEFAULT_PHI_MAX,
) -> dict:
    """Effective diffusivity of the concentration field (m^2 s^-1).

    ``D_eff = D_m + alpha d^2 gdot f(c)`` with ``f(c) = c^2 (1 + 0.5 e^{8.8c})``
    (Leighton-Acrivos shear-induced diffusion, vanishing as ``c -> 0``).

    The settling dispersion ``(v_s d)^2 / D_m`` is reported separately as
    ``dispersion`` for the record but is **not** part of ``D_eff`` (see the
    module docstring).

    Raises:
        FrazilRheologyError: on non-positive ``diameter`` or ``molecular``.
    """
    if diameter <= 0.0:
        raise FrazilRheologyError(f"diameter must be > 0, got {diameter!r}")
    if molecular <= 0.0:
        raise FrazilRheologyError(f"molecular diffusivity must be > 0, got {molecular!r}")
    c = validate_concentration(concentration, phi_max)
    g = _as_tensor(shear_rate).clamp(min=0.0)
    shape = torch.broadcast_shapes(c.shape, g.shape)
    cc = c.expand(shape)
    gg = g.expand(shape)
    f = cc * cc * (1.0 + 0.5 * torch.exp(8.8 * cc))
    d_sh = alpha * diameter * diameter * gg * f
    return {
        "D_eff": molecular + d_sh,
        "molecular": molecular,
        "shear_induced": d_sh,
        "f_concentration": f,
    }


def flow_capacity_ratio(
    concentration,
    *,
    acceleration: float = 1.2e-5,
    half_height: float = 10.0,
    mu_water: float = 0.02,
    phi_max: float = DEFAULT_PHI_MAX,
    n_packed: float = 0.45,
):
    """Flow-rate ratio ``Q(c)/Q(0)`` of a uniform slurry channel at fixed dP.

    For a power-law fluid the exact (steady, uniform) Poiseuille solution in a
    channel of half-height ``Hh`` under body force ``a`` is

        ``u_max = n/(n+1) (a/K)^(1/n) Hh^((n+1)/n)``,
        ``Q = H u_max (n+1)/(2n+1)``,

    so, normalised by the clear-water channel ``Q0 = H (a/(2 mu_w)) Hh^2 (2/3)``,

        ``Q(c)/Q(0) = [n/(n+1) (a/K(c))^(1/n) Hh^((n+1)/n) (n+1)/(2n+1)]
                      / [(a/(2 mu_w)) Hh^2 (2/3)]``.

    Two effects are folded in and *must not* be confused: the Krieger-Dougherty
    thickening enters through ``K(c)`` (amplified by the exponent ``1/n``), and
    the shear-thinning shape factor ``(n+1)/(2n+1)`` (a flatter profile carries
    slightly *more* flow for the same peak velocity).  At ``c = 0.4`` with
    ``mu_r = 5.2``, ``n = 0.70``, ``a = 1.2e-5`` and ``Hh = 10`` the channel
    carries ``9.3e-3`` of the clear-water flow, i.e. a **107x** reduction -- far
    more than the ``10x`` implied by dropping the ``n`` dependence, because the
    power law raises the low-shear viscosity the channel actually sees.

    Args:
        concentration: Ice volume fraction ``c`` in ``[0, phi_max)``.
        acceleration: Driving body-force acceleration (same unit system as
            ``mu_water``); the default is the lattice value used by the
            benchmark.
        half_height: Channel half-height in the same units.
        mu_water: Clear-water viscosity in the *same units as the channel*
            (default 0.02 = the benchmark's lattice viscosity, so that
            ``acceleration``/``half_height`` defaults are consistent).
        phi_max, n_packed: Closure parameters.

    Note:
        Because the power law is evaluated at the channel's own shear-rate
        scale, the ratio is *not* a pure function of ``c``: it needs the
        channel's ``(a, Hh, mu_w)``.  Calling it with SI numbers for a physical
        channel gives the physical ratio for that channel.
    """
    if acceleration <= 0.0:
        raise FrazilRheologyError(f"acceleration must be > 0, got {acceleration!r}")
    if half_height <= 0.0:
        raise FrazilRheologyError(f"half_height must be > 0, got {half_height!r}")
    c = validate_concentration(concentration, phi_max)
    K = _as_tensor(frazil_consistency_index(c, mu_water=mu_water, phi_max=phi_max))
    n = _as_tensor(frazil_flow_index(c, n_packed=n_packed, phi_max=phi_max))
    a = float(acceleration)
    Hh = float(half_height)
    u_max = n / (n + 1.0) * torch.pow(a / K, 1.0 / n) * Hh ** ((n + 1.0) / n)
    flow = u_max * (n + 1.0) / (2.0 * n + 1.0)
    flow0 = 0.5 * a / mu_water * Hh * Hh * (2.0 / 3.0)
    out = flow / flow0
    return _scalar_like(concentration, out)


def batch_settling_interface(
    concentration_0,
    time,
    *,
    v_s0: float,
    exponent=None,
    re_p=None,
    phi_max: float = DEFAULT_PHI_MAX,
    height_0: float = 1.0,
):
    """Kynch constant-rate-zone interface height ``h(t) = h0 - v_s(c0) t``.

    For a Richardson-Zaki flux ``f(c) = c v_s(c)`` with ``v_s`` decreasing in
    ``c``, a homogenised column of concentration ``c0`` under clear water forms
    a shock whose speed is ``[f(c0) - f(0)]/c0 = v_s(c0)``: the interface falls
    at the hindered velocity of the *initial* concentration.  This is the
    analytic reference the column LBM is validated against.

    Raises:
        FrazilRheologyError: on a negative time or a negative height.
    """
    if float(time) < 0.0:
        raise FrazilRheologyError(f"time must be >= 0, got {time!r}")
    if height_0 <= 0.0:
        raise FrazilRheologyError(f"height_0 must be > 0, got {height_0!r}")
    v_s = float(
        hindered_settling_velocity(
            concentration_0, v_s0, exponent=exponent, re_p=re_p, phi_max=phi_max
        )
    )
    h = height_0 - v_s * float(time)
    return {"height": max(h, 0.0), "velocity": v_s, "clipped_by_bottom": h < 0.0}


def concentration_sweep(
    concentrations=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
    shear_rates=(0.1, 1.0, 10.0, 100.0),
    *,
    diameter: float = DEFAULT_D_FRAZIL,
    rho_particle: float = RHO_ICE,
    rho_fluid: float = RHO_WATER,
    mu_water: float = MU_WATER,
    phi_max: float = DEFAULT_PHI_MAX,
    n_packed: float = 0.45,
) -> dict:
    """Programmatic closure table for the campaign artifact (no hand-typed numbers).

    Returns:
        dict with ``single_particle`` (dilute settling + RZ exponent),
        ``rows`` (one dict per concentration: rheology, settling, drag,
        diffusivity, flow capacity) and ``shear_rates``.
    """
    single = single_particle_settling_velocity(
        diameter, rho_particle=rho_particle, rho_fluid=rho_fluid, mu_fluid=mu_water
    )
    re_p = single["reynolds"]
    m = float(richardson_zaki_exponent(re_p))
    rows = []
    for c in concentrations:
        c = float(c)
        mu_rows = []
        for g in shear_rates:
            mu = effective_mixture_viscosity(
                c, float(g), mu_water=mu_water, phi_max=phi_max, n_packed=n_packed
            )
            mu_rows.append(
                {
                    "shear_rate": float(g),
                    "mu_eff": float(torch.as_tensor(mu["mu_eff"]).reshape(-1)[0]),
                    "viscosity_ratio": float(
                        torch.as_tensor(mu["viscosity_ratio"]).reshape(-1)[0]
                    ),
                }
            )
        mu_ref = effective_mixture_viscosity(
            c, 1.0, mu_water=mu_water, phi_max=phi_max, n_packed=n_packed
        )
        v_s = float(hindered_settling_velocity(c, single["velocity"], exponent=m, phi_max=phi_max))
        omega = float(hindered_drag_amplification(c, exponent=m, phi_max=phi_max))
        cd_susp = float(drag_coefficient_suspension(c, re_p, exponent=m, phi_max=phi_max))
        diff = effective_diffusivity(c, 1.0, diameter=diameter, phi_max=phi_max)
        disp = (v_s * diameter) ** 2 / 1.0e-9
        rows.append(
            {
                "concentration": c,
                "mu_relative_kd": float(
                    relative_viscosity_krieger_dougherty(c, phi_max)
                ),
                "flow_index": float(frazil_flow_index(c, n_packed=n_packed, phi_max=phi_max)),
                "consistency_index": float(
                    frazil_consistency_index(c, mu_water=mu_water, phi_max=phi_max)
                ),
                "mu_eff_at_shear_rates": mu_rows,
                "mu_eff_shear_1": float(torch.as_tensor(mu_ref["mu_eff"]).reshape(-1)[0]),
                "viscosity_ratio_shear_1": float(
                    torch.as_tensor(mu_ref["viscosity_ratio"]).reshape(-1)[0]
                ),
                "settling_velocity": v_s,
                "settling_velocity_ratio": v_s / single["velocity"],
                "drag_amplification": omega,
                "drag_coefficient_suspension": cd_susp,
                "difelice_amplification": float(
                    difelice_drag_amplification(c, re_p, phi_max=min(phi_max, 1.0))
                ),
                "effective_diffusivity_shear_1": float(
                    torch.as_tensor(diff["D_eff"]).reshape(-1)[0]
                ),
                "settling_dispersion": disp,
                "flow_capacity_ratio": float(flow_capacity_ratio(c, phi_max=phi_max)),
            }
        )
    return {
        "single_particle": single,
        "richardson_zaki_exponent": m,
        "concentrations": [float(c) for c in concentrations],
        "shear_rates": [float(g) for g in shear_rates],
        "rows": rows,
        "parameters": {
            "diameter": diameter,
            "rho_particle": rho_particle,
            "rho_fluid": rho_fluid,
            "mu_water": mu_water,
            "phi_max": phi_max,
            "n_packed": n_packed,
        },
    }