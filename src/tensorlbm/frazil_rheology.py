"""Frazil-ice slurry rheology surface on top of the D2Q9 power-law LBM.

Physical background
-------------------
Frazil ice forms when turbulent, supercooled water (bulk temperature a few
hundredths of a kelvin below the freezing point) nucleates small disc-shaped
crystals.  The resulting *slurry* is a suspension of ice particles in water
whose rheology is the missing link between the thermodynamic state
(supercooling / ice volume fraction) and the hydrodynamic response used by
resistance and thrust-loss models:

* **Concentration → viscosity (effective-medium).**  Einstein's dilute limit
  :math:`\\mu_r = 1 + [\\eta]\\phi` is only valid for :math:`\\phi \\lesssim 0.05`.
  For frazil, whose flocs are non-spherical and interact strongly, the
  Krieger–Dougherty (1959) hard-sphere-suspension result

      .. math::  \\mu_r(\\phi) = \\left(1 - \\phi/\\phi_{max}\\right)^{-[\\eta]\\phi_{max}}

  is the standard closure in the sea-ice literature (Osterkamp & Gosink 1983;
  Daly 1984), with :math:`[\\eta] = 2.5` and a packed/floc limit
  :math:`\\phi_{max} \\approx 0.6`.  As :math:`\\phi \\to \\phi_{max}` the
  viscosity diverges — the **viscosity jump window**: over the last
  :math:`\\sim 0.05` of volume fraction the relative viscosity climbs from
  :math:`O(10)` to :math:`O(10^2\\!-\\!10^3)`.  Extrapolating *past*
  :math:`\\phi_{max}` is meaningless (the suspension is no longer flowing) and
  this module refuses it instead of returning a number
  (:class:`FrazilRheologyError`).

* **Shear thinning.**  Frazil slurries are strongly shear-thinning: the flocs
  break down under shear and the apparent viscosity follows an Ostwald–de
  Waele / power-law law :math:`\\mu_{app} = K(\\phi)\\,\\dot\\gamma^{\\,n(\\phi)-1}`
  with :math:`n` decreasing from 1 (clear water) towards
  :math:`n_{packed} \\approx 0.45` as the ice fraction grows
  (Daly 1984; Barnes 1997 for the review of "shear-thinning at high
  concentration").  Only the shear-thinning branch :math:`0 < n \\le 1` is
  physically admissible for frazil; other indices are rejected
  (:func:`validate_flow_index`).

* **Supercooling → crystallization growth-rate window.**  Crystallization is
  dormant below a nucleation threshold :math:`\\Delta T_{nuc}` (≈0.02 K for
  natural waters), then proceeds through two resistances in series: the
  interfacial (kinetic) attachment law :math:`v_{kin} = A\\,\\Delta T^{b}`
  with :math:`b \\approx 1.5` (Michel 1978; Daly 1984) and the diffusive limit
  :math:`v_{diff} = k_w \\Delta T / (\\rho_i L \\delta)`.  The harmonic mean
  gives the familiar growth-rate *window*: nothing below
  :math:`\\Delta T_{nuc}`, a steep rise just above it and a linear,
  diffusion-limited tail.  The model is only valid inside the frazil regime
  :math:`\\Delta T \\lesssim 1` K; beyond it columnar/dendritic ice forms and
  the closure is flagged out of range.

Everything in this module is *pure rheology* — no lattice units are implied.
The lattice-Boltzmann side (channel validation, viscosity-jump survival)
lives in :mod:`tensorlbm.frazil_lbm`.

References (self-built anchors — see ``benchmarks/frazil_rheology_bench.py``)
----------------------------------------------------------------------------
* Krieger, I.M. & Dougherty, T.J. (1959) *Trans. Soc. Rheol.* 3, 137.
* Osterkamp, T.E. & Gosink, J.P. (1983) *J. Glaciol.* 29, 267.
* Daly, S.F. (1984) CRREL Report 84-11.
* Michel, B. (1978) *Ice Mechanics*, Laval Univ. Press.
* Barnes, H.A. (1997) *J. Non-Newtonian Fluid Mech.* 70, 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "DEFAULT_PHI_MAX",
    "DEFAULT_INTRINSIC_VISCOSITY",
    "DEFAULT_N_PACKED",
    "FrazilRheologyError",
    "validate_concentration",
    "validate_flow_index",
    "relative_viscosity_krieger_dougherty",
    "frazil_flow_index",
    "frazil_consistency_index",
    "frazil_apparent_viscosity",
    "viscosity_jump_window",
    "growth_rate_window",
    "concentration_viscosity_curve",
    "RheologyPoint",
]

# ---------------------------------------------------------------------------
# Physical constants / closure parameters (all documented, none fitted here)
# ---------------------------------------------------------------------------

DEFAULT_PHI_MAX = 0.60  # packed / floc limit of a frazil suspension
DEFAULT_INTRINSIC_VISCOSITY = 2.5  # Einstein coefficient, rigid spheres
DEFAULT_N_PACKED = 0.45  # power-law index at the packed limit (frazil flocs)
DEFAULT_INDEX_EXPONENT = 1.5  # how fast n(phi) collapses with concentration

#: Water / ice properties used by the growth-rate window (SI).
K_WATER = 0.561  # W m^-1 K^-1, thermal conductivity of water at 0 C
RHO_ICE = 917.0  # kg m^-3
L_FUSION = 3.3355e5  # J kg^-1
DELTA_THERMAL = 1.0e-4  # m, crystal thermal boundary-layer thickness (upper bound)
A_KINETIC = 2.5e-4  # m s^-1 K^-1.5, interfacial attachment coefficient
B_KINETIC = 1.5  # exponent of the kinetic growth law
DT_NUCLEATION = 0.02  # K, nucleation threshold of natural water (0.01-0.05 K)
DT_REGIME_MAX = 1.0  # K, upper edge of the frazil (equiaxed) regime


class FrazilRheologyError(ValueError):
    """Raised when a request leaves the validated envelope of the closure.

    Used both for *unphysical inputs* (negative concentration, a power-law
    index outside :math:`0 < n \\le 1`) and for *honest truncation* requests:
    extrapolating a Krieger–Dougherty fit through and past the packed limit is
    rejected rather than answered with a fictitious viscosity.
    """


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64)


def validate_concentration(phi, phi_max: float = DEFAULT_PHI_MAX, *, allow_zero: bool = True):
    """Reject concentrations outside the physical envelope ``[0, phi_max)``.

    Args:
        phi: Concentration (scalar or tensor).
        phi_max: Packed limit; the closure is not defined at or above it.
        allow_zero: When ``False`` a strictly positive concentration is required
            (used by the apparent-viscosity helpers that divide by ``phi``).

    Raises:
        FrazilRheologyError: on ``phi < 0`` or ``phi >= phi_max`` — i.e. the
            extrapolation past the packed limit is refused, not silently
            clipped, unless the caller asks for clipping explicitly via
            :func:`concentration_viscosity_curve`.
    """
    t = _as_tensor(phi)
    lo = 0.0 if allow_zero else 0.0
    if t.numel() and float(t.min()) < lo:
        raise FrazilRheologyError(f"concentration must be >= 0, got min={float(t.min()):.6g}")
    if not allow_zero and t.numel() and float(t.min()) == 0.0:
        raise FrazilRheologyError("concentration must be > 0 for this call")
    if t.numel() and float(t.max()) >= phi_max:
        raise FrazilRheologyError(
            f"concentration {float(t.max()):.6g} >= packed limit {phi_max:.6g}: "
            "Krieger-Dougherty extrapolation is refused (suspension is not flowing "
            "above the packed limit); truncate the request below phi_max instead"
        )
    return t


def validate_flow_index(n, *, upper: float = 1.0) -> float:
    """Accept only physically admissible power-law indices ``0 < n <= 1``.

    The frazil surface is a *shear-thinning* surface: ``n <= 0`` is
    mathematically singular (ν ∝ γ̇^(n-1) diverges), and ``n > 1``
    (shear-thickening) is not observed for ice slurries and would silently
    turn the resistance model in the wrong direction, so both are rejected.

    Raises:
        FrazilRheologyError: when ``n <= 0`` or ``n > upper``.
    """
    n = float(n)
    if not math.isfinite(n) or n <= 0.0:
        raise FrazilRheologyError(
            f"power-law index must be > 0 (shear-thinning frazil), got n={n!r}"
        )
    if n > upper + 1e-12:
        raise FrazilRheologyError(
            f"power-law index must be <= {upper:g} for a shear-thinning frazil "
            f"slurry, got n={n!r} (shear-thickening is rejected)"
        )
    return n


# ---------------------------------------------------------------------------
# Effective-medium viscosity: Krieger-Dougherty + jump window
# ---------------------------------------------------------------------------


def relative_viscosity_krieger_dougherty(
    phi,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
):
    """Relative viscosity ``mu_r = (1 - phi/phi_max)^(-[eta]*phi_max)``.

    Args:
        phi: Ice volume fraction, ``0 <= phi < phi_max``.
        phi_max: Packed (floc) limit, default 0.60.
        intrinsic: Intrinsic viscosity ``[eta]``, default 2.5 (rigid spheres).

    Returns:
        Relative viscosity (unitless) with the same container type as ``phi``
        (float in → float out, tensor in → tensor out).

    Raises:
        FrazilRheologyError: for ``phi < 0`` or ``phi >= phi_max``.
    """
    if phi_max <= 0.0:
        raise FrazilRheologyError(f"phi_max must be > 0, got {phi_max!r}")
    if intrinsic <= 0.0:
        raise FrazilRheologyError(f"intrinsic viscosity must be > 0, got {intrinsic!r}")
    t = validate_concentration(phi, phi_max)
    mu_r = (1.0 - t / phi_max) ** (-intrinsic * phi_max)
    if isinstance(phi, torch.Tensor):
        return mu_r
    return float(mu_r)


def frazil_flow_index(
    phi,
    *,
    n_water: float = 1.0,
    n_packed: float = DEFAULT_N_PACKED,
    phi_max: float = DEFAULT_PHI_MAX,
    exponent: float = DEFAULT_INDEX_EXPONENT,
):
    """Concentration-dependent power-law index ``n(phi)`` of a frazil slurry.

    The collapse law is

        .. math:: n(\\phi) = n_w - (n_w - n_{packed})\\,(\\phi/\\phi_{max})^{p}

    with :math:`n_w = 1` (clear water is Newtonian), :math:`n_{packed} = 0.45`
    at the packed limit and :math:`p = 1.5` controlling how early the thinning
    sets in.  ``n`` is clipped to ``[n_packed, 1]`` so that the return value is
    always a valid shear-thinning index.

    Raises:
        FrazilRheologyError: on ``phi`` outside ``[0, phi_max)`` or on
            non-physical closure parameters.
    """
    validate_flow_index(n_water, upper=1.0 + 1e-12)
    validate_flow_index(n_packed)
    if n_packed > n_water:
        raise FrazilRheologyError("n_packed must not exceed n_water")
    if exponent <= 0.0:
        raise FrazilRheologyError(f"exponent must be > 0, got {exponent!r}")
    t = validate_concentration(phi, phi_max)
    frac = (t / phi_max).clamp(0.0, 1.0) ** exponent
    n = n_water - (n_water - n_packed) * frac
    n = n.clamp(min=n_packed, max=n_water)
    if isinstance(phi, torch.Tensor):
        return n
    return float(n)


def frazil_consistency_index(
    phi,
    *,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
):
    """Consistency index ``K(phi) = mu_water * mu_r(phi)`` (units of viscosity).

    ``K`` is the apparent viscosity at the reference shear rate
    ``gamma_dot = 1`` (in whatever unit system the caller works in), so the
    power-law pair ``(K, n)`` reduces to ``mu_water`` at ``phi = 0`` and is
    consistent with the Krieger–Dougherty curve at ``gamma_dot = 1``.
    """
    if mu_water <= 0.0:
        raise FrazilRheologyError(f"mu_water must be > 0, got {mu_water!r}")
    mu_r = relative_viscosity_krieger_dougherty(phi, phi_max, intrinsic)
    if isinstance(mu_r, torch.Tensor):
        return mu_r * mu_water
    return mu_r * mu_water


def viscosity_jump_window(
    *,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    width: float = 0.05,
) -> dict:
    """Characterise the *viscosity jump window* just below the packed limit.

    The Krieger–Dougherty curve is asymptotically steep: inside the last
    ``width`` of volume fraction the relative viscosity rises by orders of
    magnitude.  The lattice solver must survive that jump, so the window is
    reported explicitly (bounds and the viscosity ratio across it) and used by
    :mod:`tensorlbm.frazil_lbm` to bound the relaxation time.

    Returns:
        dict with ``phi_lower``, ``phi_upper`` (= ``phi_max``), ``mu_ratio``,
        ``mu_ratio_evaluation_phi`` and the derivative ``d ln(mu_r)/d phi`` at
        the window centre.  The ratio is evaluated over
        ``[phi_lower, phi_max - 0.005]``: the last half percent of volume
        fraction is where the suspension stops flowing, so quoting a ratio
        against an epsilon inside the divergence would be meaningless.
    """
    if width <= 0.0 or width >= phi_max:
        raise FrazilRheologyError(f"window width must be in (0, {phi_max}), got {width!r}")
    phi_lo = phi_max - width
    phi_hi = phi_max - 0.005
    mu_lo = float(relative_viscosity_krieger_dougherty(phi_lo, phi_max, intrinsic))
    mu_hi = float(relative_viscosity_krieger_dougherty(phi_hi, phi_max, intrinsic))
    centre = 0.5 * (phi_lo + phi_max)
    slope = intrinsic * phi_max / (phi_max - centre)  # d ln(mu_r)/d phi
    return {
        "phi_lower": phi_lo,
        "phi_upper": phi_max,
        "width": width,
        "mu_relative_at_lower": mu_lo,
        "mu_relative_at_upper": mu_hi,
        "mu_ratio": mu_hi / mu_lo,
        "mu_ratio_evaluation_phi": phi_hi,
        "dln_mu_dphi_at_centre": slope,
        "intrinsic_viscosity": intrinsic,
    }


def frazil_apparent_viscosity(
    phi,
    shear_rate,
    *,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
    index_exponent: float = DEFAULT_INDEX_EXPONENT,
    shear_rate_floor: float = 1e-12,
):
    """Power-law apparent viscosity of a frazil slurry ``K(phi)*gdot^(n(phi)-1)``.

    Args:
        phi: Ice volume fraction(s), ``0 <= phi < phi_max``.
        shear_rate: Shear rate magnitude, same units as the ``1``-reference of
            :func:`frazil_consistency_index`.
        mu_water: Water viscosity in the working unit system.
        shear_rate_floor: Positive floor that keeps the ``gdot^(n-1)`` singular
            limit (``gdot -> 0``, ``n < 1``) finite.

    Returns:
        ``(mu_app, n_phi)`` — apparent viscosity and the local power-law index,
        broadcast to the shape of ``phi``/``shear_rate``.
    """
    if shear_rate_floor <= 0.0:
        raise FrazilRheologyError("shear_rate_floor must be > 0")
    K = frazil_consistency_index(phi, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    n = frazil_flow_index(phi, n_packed=n_packed, phi_max=phi_max, exponent=index_exponent)
    g = _as_tensor(shear_rate).clamp(min=shear_rate_floor)
    if isinstance(K, float):
        K = torch.as_tensor(K, dtype=torch.float64)
    mu = K * torch.pow(g, n - 1.0)
    return mu, n


# ---------------------------------------------------------------------------
# Supercooling -> growth-rate window
# ---------------------------------------------------------------------------


def growth_rate_window(
    supercooling,
    *,
    dt_nucleation: float = DT_NUCLEATION,
    a_kinetic: float = A_KINETIC,
    b_kinetic: float = B_KINETIC,
    k_water: float = K_WATER,
    rho_ice: float = RHO_ICE,
    latent_heat: float = L_FUSION,
    delta_thermal: float = DELTA_THERMAL,
    dt_regime_max: float = DT_REGIME_MAX,
) -> dict:
    """Frazil crystallization growth rate versus supercooling (a *window*).

    Two resistances in series (interfacial attachment and heat diffusion):

        .. math::
            v_{kin} = A (\\Delta T - \\Delta T_{nuc})^b, \\quad
            v_{diff} = \\frac{k_w \\Delta T}{\\rho_i L \\delta}, \\quad
            v = \\left(v_{kin}^{-1} + v_{diff}^{-1}\\right)^{-1}

    for ``Delta T > dt_nucleation`` and ``v = 0`` otherwise, so the growth rate
    is dormant below nucleation, rises steeply just above it and becomes
    diffusion-limited (linear in supercooling) further out.  0.02 K is the
    nucleation threshold of natural water and 1 K the upper edge of the
    equiaxed (frazil) regime.

    Args:
        supercooling: Supercooling ``Delta T`` in kelvin (scalar or array-like).

    Returns:
        dict with ``delta_T``, ``v`` (m/s), ``v_kinetic``, ``v_diffusive``,
        ``active`` (bool array: inside the nucleation window) and
        ``in_regime`` (bool array: ``Delta T <= dt_regime_max``) plus the
        scalar limiting values used.
    """
    if dt_nucleation < 0.0:
        raise FrazilRheologyError("dt_nucleation must be >= 0")
    if a_kinetic <= 0.0 or b_kinetic <= 0.0:
        raise FrazilRheologyError("kinetic coefficients must be > 0")
    if delta_thermal <= 0.0:
        raise FrazilRheologyError("delta_thermal must be > 0")
    dT = _as_tensor(supercooling).clamp(min=0.0)
    drive = (dT - dt_nucleation).clamp(min=0.0)
    v_kin = a_kinetic * torch.pow(drive, b_kinetic)
    beta = k_water / (rho_ice * latent_heat * delta_thermal)
    v_diff = beta * dT

    # Crossover of the two resistances: A (dT - dT_nuc)^b = beta dT, solved by
    # bisection on [dT_nuc, dT_regime_max] (the kinetic branch is rate limiting
    # below it, the diffusive branch above it -- this is the growth *window*).
    def _gap(dt: float) -> float:
        return a_kinetic * max(dt - dt_nucleation, 0.0) ** b_kinetic - beta * dt

    crossover = float("nan")
    lo, hi = dt_nucleation, max(dt_regime_max, dt_nucleation + 1e-6)
    if _gap(lo) < 0.0 <= _gap(hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _gap(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        crossover = 0.5 * (lo + hi)
    tiny = torch.finfo(torch.float64).tiny
    v = 1.0 / (1.0 / v_kin.clamp(min=tiny) + 1.0 / v_diff.clamp(min=tiny))
    active = dT > dt_nucleation
    v = torch.where(active, v, torch.zeros_like(v))
    out = {
        "delta_T": dT,
        "v": v,
        "v_kinetic": torch.where(active, v_kin, torch.zeros_like(v_kin)),
        "v_diffusive": torch.where(active, v_diff, torch.zeros_like(v_diff)),
        "active": active,
        "in_regime": dT <= dt_regime_max,
        "dt_nucleation": dt_nucleation,
        "dt_regime_max": dt_regime_max,
        "a_kinetic": a_kinetic,
        "b_kinetic": b_kinetic,
        "diffusive_coefficient": beta,
        "series_crossover_delta_T": crossover,
    }
    if not isinstance(supercooling, torch.Tensor):
        out["delta_T"] = out["delta_T"].tolist()
        out["v"] = out["v"].tolist()
        out["v_kinetic"] = out["v_kinetic"].tolist()
        out["v_diffusive"] = out["v_diffusive"].tolist()
        out["active"] = out["active"].tolist()
        out["in_regime"] = out["in_regime"].tolist()
    return out


# ---------------------------------------------------------------------------
# Concentration-viscosity curve (the artifact payload)
# ---------------------------------------------------------------------------


@dataclass
class RheologyPoint:
    """One point of the concentration-viscosity curve."""

    phi: float
    relative_viscosity: float
    flow_index: float
    consistency_index: float
    apparent_viscosity_at_shear: dict = field(default_factory=dict)


def concentration_viscosity_curve(
    phi_values,
    *,
    shear_rates=(0.1, 1.0, 10.0),
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
    index_exponent: float = DEFAULT_INDEX_EXPONENT,
    truncate: bool = True,
) -> dict:
    """Evaluate the concentration-viscosity curve with honest truncation.

    Requests at or above the packed limit are never extrapolated.  With
    ``truncate=True`` (default) the out-of-range concentrations are *dropped*
    from the returned curve and counted in ``n_truncated`` /
    ``truncated_above``; with ``truncate=False`` the call raises
    :class:`FrazilRheologyError` instead.  Either way no fictitious viscosity
    above ``phi_max`` is produced.

    Returns:
        dict with ``phi``, ``relative_viscosity``, ``flow_index``,
        ``consistency_index``, ``apparent_viscosity`` (per requested shear
        rate) and the truncation bookkeeping.
    """
    phis = [float(p) for p in phi_values]
    if not phis:
        raise FrazilRheologyError("empty concentration grid")
    kept, dropped = [], []
    for p in phis:
        if p < 0.0:
            raise FrazilRheologyError(f"concentration must be >= 0, got {p!r}")
        if p >= phi_max:
            dropped.append(p)
        else:
            kept.append(p)
    if dropped and not truncate:
        raise FrazilRheologyError(
            f"{len(dropped)} concentration(s) >= packed limit {phi_max}: "
            "extrapolation refused (pass truncate=True to drop them)"
        )
    if not kept:
        raise FrazilRheologyError("all requested concentrations are above the packed limit")
    phi_t = torch.tensor(kept, dtype=torch.float64)
    mu_r = relative_viscosity_krieger_dougherty(phi_t, phi_max, intrinsic)
    n = frazil_flow_index(phi_t, n_packed=n_packed, phi_max=phi_max, exponent=index_exponent)
    K = frazil_consistency_index(phi_t, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    apparent = {}
    for g in shear_rates:
        mu, _ = frazil_apparent_viscosity(
            phi_t,
            g,
            mu_water=mu_water,
            phi_max=phi_max,
            intrinsic=intrinsic,
            n_packed=n_packed,
            index_exponent=index_exponent,
        )
        apparent[f"{float(g):g}"] = mu.tolist()
    return {
        "phi": kept,
        "relative_viscosity": mu_r.tolist(),
        "flow_index": n.tolist(),
        "consistency_index": K.tolist(),
        "apparent_viscosity": apparent,
        "shear_rates": [float(g) for g in shear_rates],
        "n_truncated": len(dropped),
        "truncated_above": dropped,
        "phi_max": phi_max,
        "intrinsic_viscosity": intrinsic,
        "n_packed": n_packed,
        "index_exponent": index_exponent,
        "mu_water": mu_water,
    }
