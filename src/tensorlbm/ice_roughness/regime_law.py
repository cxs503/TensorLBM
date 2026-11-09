"""Three-zone roughness closure for the ice/water drag coefficient ``C_w``.

Physics
-------
Below a drifting ice sheet the turbulent boundary layer obeys the rough-wall
log law

    U^+(z) = (1/kappa) ln(z u_tau / nu) + B - dU^+(k_s^+, lambda/k_s)        (1)

with ``B = 5.2`` the smooth-wall intercept (kappa = 0.41).  Everything the
roughness does to the drag is contained in the **roughness function**
``dU^+ >= 0`` -- the downward shift of the velocity profile.  The mission
supplies ``dU^+`` from wall-resolved LBM channel simulations and asks for a
three-zone parameterisation in ``k_s^+ = k_s u_tau / nu`` at fixed spacing
ratio ``lambda/k_s``.

The form adopted here is the one-parameter family

    dU^+(k_s^+) = (1/kappa) ln(1 + k_s^+ / k0),        k0 = k0(lambda/k_s)    (2)

which has the three regimes of Nikuradse built in analytically:

* **hydraulically smooth**, ``k_s^+ << k0``: ``dU^+ -> k_s^+/ (kappa k0) -> 0``
  (the roughness is buried in the viscous sublayer, no drag penalty);
* **transitional**, ``k_s^+ ~ k0``: smooth monotone interpolation with a
  continuous first derivative (``d dU^+/d ln k_s^+ = k_s^+/(kappa(k_s^+ + k0))``,
  strictly between 0 and 1/kappa);
* **fully rough**, ``k_s^+ >> k0``: ``dU^+ = (1/kappa) ln(k_s^+/k0)``, i.e. the
  Nikuradse sand-grain line with slope ``1/kappa`` and an *equivalent sand
  grain height* ``k_s,eff / k_s = k0 / k0_Nik``, where

      k0_Nik = exp(kappa (8.5 - B)) = 3.87                                (3)

  is fixed by requiring Nikuradse's fully rough law
  ``U^+ = (1/kappa) ln(z/k_s) + 8.5`` to coincide with (1) at ``dU^+`` from
  (2).  A morphology that reproduces the sand-grain drag therefore returns
  ``k0 = 3.87``; ``k0 > k0_Nik`` means *less* drag than sand at the same
  geometric height (skimming / d-type flow), ``k0 < k0_Nik`` means more.

Only ``k0`` is fitted; the slope ``1/kappa`` and the smooth intercept ``B`` are
imposed.  That is deliberate: one free parameter per morphology is what a
15-case campaign can determine honestly, and it makes C2 (fully-rough
agreement with Nikuradse) a genuine prediction of the *intercept* rather than
a tautology.

``C_w``
------
In the drift momentum balance the water drag per unit ice area is
``tau_w = rho_w C_w |U_rel| U_rel`` and the boundary layer is thin compared to
the ocean depth, so the appropriate reference velocity is the ice-water
relative velocity at the ice underside.  Eliminating ``u_tau`` from (1) at a
reference height ``z_ref`` (the OBL thickness over which the drift shear is
communicated, default 10 m below the ice) gives

    C_w = [ kappa / ( ln(z_ref u_tau/nu) + kappa (B - dU^+) ) ]^2          (4)

which is implicit in ``u_tau`` through both ``ln z^+`` and ``dU^+(k_s^+)``;
:func:`solve_friction_velocity` iterates it to convergence (fixed point on
``u_tau``, contraction factor ``~ kappa/ln^2 << 1``).  In the smooth limit
(``dU^+ = 0``) (4) is the classical smooth-plate drag law; in the fully rough
limit it reduces to ``C_w = [kappa/ln(z_ref/z0)]^2`` with
``z0 = (nu/u_tau) exp(kappa(dU^+ - B)) -> k_s exp(-kappa(B-8.5))/... `` the
Nikuradse equivalent sand roughness height over 30.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "KAPPA",
    "B_SMOOTH",
    "K0_NIKURADSE",
    "B_FULLY_ROUGH",
    "ThreeZoneLaw",
    "delta_u_plus",
    "nikuradse_delta_u_plus",
    "equivalent_sand_grain_ratio",
    "k0_from_delta_u_plus",
    "fit_k0",
    "cf_channel_smooth_reference",
    "cf_from_bulk_plus",
    "solve_friction_velocity",
    "cw_ice_water",
    "regime_of",
    "ks_plus_boundaries",
    "DEFAULT_REFERENCE_HEIGHT_M",
    "NU_WATER",
    "RHO_WATER",
]

KAPPA: float = 0.41
#: Smooth-wall log-law intercept, U+ = (1/kappa) ln y+ + B.
B_SMOOTH: float = 5.2
#: Nikuradse fully-rough additive constant, U+ = (1/kappa) ln(y/ks) + 8.5.
B_FULLY_ROUGH: float = 8.5
#: Onset parameter that makes (2) coincide with Nikuradse's sand-grain law.
K0_NIKURADSE: float = math.exp(KAPPA * (B_FULLY_ROUGH - B_SMOOTH))  # 3.87

#: Default ocean-side reference height for C_w: the OBL thickness over which
#: the ice drift shear is transmitted (10 m, cf. McPhee 2008 "ice-ocean
#: boundary layer" thickness).
DEFAULT_REFERENCE_HEIGHT_M: float = 10.0
NU_WATER: float = 1.83e-6      # m^2/s, sea water at 0 C, S = 34 psu
RHO_WATER: float = 1025.0      # kg/m^3

#: Nikuradse's zone boundaries, rescaled by the onset parameter of the
#: morphology: k_s^+ < 5 (k0/k0_Nik) is smooth, > 70 (k0/k0_Nik) is fully rough.
_KS_PLUS_SMOOTH_NIK: float = 5.0
_KS_PLUS_FULLY_ROUGH_NIK: float = 70.0


def delta_u_plus(ks_plus: float, k0: float) -> float:
    """Roughness function (2): ``(1/kappa) ln(1 + ks+/k0)``."""
    ks = np.asarray(ks_plus, dtype=float)
    if k0 <= 0.0:
        raise ValueError("k0 must be positive")
    return np.log1p(np.maximum(ks, 0.0) / float(k0)) / KAPPA


def nikuradse_delta_u_plus(ks_plus: float) -> float:
    """Nikuradse sand-grain roughness function, slope ``1/kappa``, ``k0=3.87``."""
    return delta_u_plus(ks_plus, K0_NIKURADSE)


def equivalent_sand_grain_ratio(k0: float) -> float:
    """``k_s,eff / k_s = k0 / k0_Nik`` (fully-rough intercept ratio)."""
    if k0 <= 0.0:
        raise ValueError("k0 must be positive")
    return float(k0) / K0_NIKURADSE


def k0_from_delta_u_plus(ks_plus: float, d_u_plus: float) -> float:
    """Invert (2) for a single measurement: ``k0 = ks+ / (exp(kappa dU+) - 1)``."""
    ks = float(ks_plus)
    d = float(d_u_plus)
    if not (ks > 0.0) or not math.isfinite(d) or d <= 0.0:
        raise ValueError("k0 inversion needs ks+ > 0 and a finite positive dU+")
    return ks / math.expm1(KAPPA * d)


def fit_k0(
    ks_plus: Sequence[float],
    d_u_plus: Sequence[float],
    *,
    min_delta_u_plus: float = 0.35,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Least-squares fit of ``k0`` over the informative (transitional+rough) points.

    Points with ``dU^+ < min_delta_u_plus`` carry almost no information about
    ``k0`` (they sit in the hydraulically smooth plateau where (2) gives
    ``dU^+ ~ ks+``), so they are excluded; the number of points used is
    reported.  A robust geometric-median estimate and the LSQ value are both
    returned so that a morphology whose ``dU^+`` is not perfectly described by
    (2) cannot hide behind a single number.
    """
    ks = np.asarray(ks_plus, dtype=float)
    du = np.asarray(d_u_plus, dtype=float)
    ok = np.isfinite(ks) & np.isfinite(du) & (ks > 0.0) & (du >= min_delta_u_plus)
    n_used = int(ok.sum())
    if n_used < 1:
        return {"k0": float("nan"), "k0_median": float("nan"), "k0_low": float("nan"),
                "k0_high": float("nan"), "n_points": n_used, "n_available": int(ks.size)}
    ks_u, du_u = ks[ok], du[ok]
    w = np.ones_like(ks_u) if weights is None else np.asarray(weights, dtype=float)[ok]
    # geometric least squares: minimise sum w [ln k0_pred - ln k0]^2
    ln_k0_pt = np.log(ks_u / np.expm1(KAPPA * du_u))
    ln_k0_fit = float(np.sum(w * ln_k0_pt) / np.sum(w))
    resid = ln_k0_pt - ln_k0_fit
    return {
        "k0": float(math.exp(ln_k0_fit)),
        "k0_median": float(math.exp(float(np.median(ln_k0_pt)))),
        "k0_low": float(math.exp(float(np.percentile(ln_k0_pt, 16)))),
        "k0_high": float(math.exp(float(np.percentile(ln_k0_pt, 84)))),
        "rms_log_residual": float(np.sqrt(np.mean(resid**2))),
        "n_points": n_used,
        "n_available": int(ks.size),
        "min_delta_u_plus": float(min_delta_u_plus),
    }


def ks_plus_boundaries(k0: float) -> dict[str, float]:
    """Zone boundaries of (2), rescaled from Nikuradse's 5 / 70 by ``k0/k0_Nik``."""
    scale = float(k0) / K0_NIKURADSE
    return {"ks_plus_smooth_upper": _KS_PLUS_SMOOTH_NIK * scale,
            "ks_plus_fully_rough_lower": _KS_PLUS_FULLY_ROUGH_NIK * scale}


def regime_of(ks_plus: float, k0: float) -> str:
    """``"hydraulically_smooth"`` / ``"transitional"`` / ``"fully_rough"``."""
    b = ks_plus_boundaries(k0)
    if ks_plus <= b["ks_plus_smooth_upper"]:
        return "hydraulically_smooth"
    if ks_plus >= b["ks_plus_fully_rough_lower"]:
        return "fully_rough"
    return "transitional"


def cf_channel_smooth_reference(re_tau: float) -> float:
    """DNS-backed smooth-channel skin friction, ``Cf = 2 tau_w/(rho U_b^2)``.

    ``U_b^+ = 2.439 ln(Re_tau) + 5.65`` reproduces the Moser-Rodi-Kim (1998)
    and Lee-Moser (2015) channel DNS to <1% over ``Re_tau = 180 ... 5200``
    (Re_tau=180: U_b+ = 18.3, Cf = 0.00597; Re_tau=590: U_b+ = 21.2,
    Cf = 0.00445).  It is the reference against which the smooth LBM baseline
    is adjudicated (criterion C1); flat-plate laws (Prandtl-Schlichting,
    ITTC-57) are *not* interchangeable with a channel Cf and are only quoted
    for context.
    """
    re_tau = float(re_tau)
    if re_tau <= 1.0:
        raise ValueError("Re_tau must exceed 1")
    u_b_plus = 2.439 * math.log(re_tau) + 5.65
    return 2.0 / u_b_plus**2


def cf_from_bulk_plus(u_plus_bulk: float) -> float:
    """Channel skin friction from the bulk velocity in wall units."""
    return 2.0 / float(u_plus_bulk) ** 2


@dataclass(frozen=True)
class ThreeZoneLaw:
    """Closed three-zone ``dU^+`` / ``C_w`` law for one spacing ratio.

    ``k0`` is the only fitted constant (see the module docstring); ``k0_source``
    records how it was obtained (``"lbm-fit"``, ``"nikuradse"``, ``"prior"``).
    """

    lambda_over_ks: float
    k0: float
    k0_source: str = "lbm-fit"
    k0_band: tuple[float, float] | None = None
    n_points: int = 0
    fit_residual: float = float("nan")

    def __post_init__(self) -> None:
        if self.k0 <= 0.0:
            raise ValueError("k0 must be positive")

    # -- roughness function ------------------------------------------------
    def delta_u_plus(self, ks_plus: float) -> float:
        return float(delta_u_plus(ks_plus, self.k0))

    def cf_ratio(self, ks_plus: float, cf_smooth: float) -> float:
        """``Cf_rough / Cf_smooth`` at matched friction velocity (Clauser)."""
        u_b_plus_smooth = math.sqrt(2.0 / float(cf_smooth))
        u_b_plus_rough = u_b_plus_smooth - self.delta_u_plus(ks_plus)
        if u_b_plus_rough <= 0.0:
            raise ValueError("roughness function exceeds the smooth bulk velocity")
        return (u_b_plus_smooth / u_b_plus_rough) ** 2

    def cw_amplification(self, ks_plus: float, z_ref_plus: float) -> float:
        """``C_w/C_w,smooth`` from the log-law ratio at the same ``z_ref^+``.

        ``A = [ (ln z+ + kappa B) / (ln z+ + kappa (B - dU+)) ]^2``.  This is
        the amplification the drift model needs: it is referenced to the *same*
        ocean-boundary-layer height for the smooth and the rough wall, so no
        arbitrary reference-height convention enters the ratio.
        """
        if z_ref_plus <= 0.0:
            raise ValueError("z_ref_plus must be positive")
        d_u = self.delta_u_plus(ks_plus)
        d_smooth = math.log(float(z_ref_plus)) + KAPPA * B_SMOOTH
        d_rough = d_smooth - KAPPA * d_u
        if d_rough <= 0.0:
            raise ValueError("roughness function exceeds the log-law denominator")
        return (d_smooth / d_rough) ** 2

    def regimes(self) -> dict[str, float]:
        return ks_plus_boundaries(self.k0)

    def regime_of(self, ks_plus: float) -> str:
        return regime_of(ks_plus, self.k0)

    def equivalent_sand_grain_ratio(self) -> float:
        return equivalent_sand_grain_ratio(self.k0)

    def curve(self, ks_plus: Iterable[float]) -> list[dict[str, float]]:
        out = []
        for ks in ks_plus:
            out.append({
                "ks_plus": float(ks),
                "dU_plus": self.delta_u_plus(ks),
                "dU_plus_nikuradse": float(nikuradse_delta_u_plus(ks)),
                "regime": self.regime_of(ks),
            })
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "lambda_over_ks": float(self.lambda_over_ks),
            "k0": float(self.k0),
            "k0_source": self.k0_source,
            "k0_band": list(self.k0_band) if self.k0_band else None,
            "n_points": int(self.n_points),
            "fit_residual": float(self.fit_residual),
            "k0_nikuradse": float(K0_NIKURADSE),
            "equivalent_sand_grain_ratio": self.equivalent_sand_grain_ratio(),
            "regimes": self.regimes(),
        }


def solve_friction_velocity(
    u_ref: float,
    k_s_m: float,
    law: ThreeZoneLaw,
    *,
    z_ref_m: float = DEFAULT_REFERENCE_HEIGHT_M,
    nu: float = NU_WATER,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict[str, float]:
    """Solve (1) for ``u_tau`` at reference velocity ``u_ref`` (fixed point).

    Returns ``u_tau``, ``k_s^+``, ``dU^+``, ``C_w`` and the regime, all mutually
    consistent: ``u_tau`` fixes ``k_s^+``, which fixes ``dU^+``, which fixes
    the log-law slope, which fixes ``u_tau``.
    """
    u_ref = float(u_ref)
    if not (u_ref > 0.0):
        raise ValueError("u_ref must be positive")
    if not (k_s_m > 0.0):
        raise ValueError("k_s must be positive")
    # start from the fully rough estimate with z0 = k_s/30
    u_tau = KAPPA * u_ref / math.log(max(z_ref_m / (float(k_s_m) / 30.0), 1.0 + 1e-12))
    for _ in range(max_iter):
        ks_plus = float(k_s_m) * u_tau / nu
        d_u = law.delta_u_plus(ks_plus)
        denom = math.log(z_ref_m * u_tau / nu) + KAPPA * (B_SMOOTH - d_u)
        if denom <= 0.0:
            raise ValueError("log-law denominator non-positive: reference height too small")
        new = KAPPA * u_ref / denom
        if abs(new - u_tau) <= tol * max(abs(new), 1e-30):
            u_tau = new
            break
        u_tau = 0.5 * (u_tau + new)
    ks_plus = float(k_s_m) * u_tau / nu
    d_u = law.delta_u_plus(ks_plus)
    cw = (u_tau / u_ref) ** 2
    return {
        "u_tau": float(u_tau),
        "ks_plus": float(ks_plus),
        "dU_plus": float(d_u),
        "C_w": float(cw),
        "regime": law.regime_of(ks_plus),
        "z_ref_m": float(z_ref_m),
        "nu": float(nu),
    }


def cw_ice_water(
    u_rel_m_s: float,
    k_s_m: float,
    lambda_over_ks: float,
    *,
    law: ThreeZoneLaw | None = None,
    laws: Mapping[float, ThreeZoneLaw] | None = None,
    z_ref_m: float = DEFAULT_REFERENCE_HEIGHT_M,
    nu: float = NU_WATER,
) -> dict[str, Any]:
    """Roughness-dependent ice/water drag coefficient -- the back-fill entry.

    ``C_w`` such that ``tau_w = rho_w C_w |u_rel| u_rel``.  The law for the
    requested ``lambda/k_s`` is taken from ``laws`` (log-linear interpolation
    of ``ln k0`` between the fitted spacing ratios, clamped outside the fitted
    range and flagged ``extrapolated``).
    """
    from_laws = laws is not None
    if law is None and from_laws:
        law = interpolate_law(float(lambda_over_ks), laws)  # type: ignore[arg-type]
    if law is None:
        raise ValueError("provide law= or laws=")
    sol = solve_friction_velocity(u_rel_m_s, k_s_m, law, z_ref_m=z_ref_m, nu=nu)
    sol.update({
        "C_w": float(sol["C_w"]),
        "lambda_over_ks": float(lambda_over_ks),
        "k_s_m": float(k_s_m),
        "u_rel_m_s": float(u_rel_m_s),
        "k0": float(law.k0),
        "k0_source": law.k0_source,
        "law_lambda_over_ks": float(law.lambda_over_ks),
        "extrapolated": bool(from_laws and abs(float(law.lambda_over_ks) - float(lambda_over_ks)) > 1e-12),
    })
    return sol


def interpolate_law(lambda_over_ks: float, laws: Mapping[float, ThreeZoneLaw]) -> ThreeZoneLaw:
    """Interpolate ``ln k0`` linearly in ``ln(lambda/k_s)`` between fitted laws."""
    ratios = sorted(float(k) for k in laws)
    if not ratios:
        raise ValueError("empty law table")
    target = float(lambda_over_ks)
    if target <= ratios[0]:
        base = laws[ratios[0]]
        return ThreeZoneLaw(base.lambda_over_ks, base.k0, k0_source=base.k0_source + "+clamped-low",
                            k0_band=base.k0_band, n_points=base.n_points, fit_residual=base.fit_residual)
    if target >= ratios[-1]:
        base = laws[ratios[-1]]
        return ThreeZoneLaw(base.lambda_over_ks, base.k0, k0_source=base.k0_source + "+clamped-high",
                            k0_band=base.k0_band, n_points=base.n_points, fit_residual=base.fit_residual)
    for lo, hi in zip(ratios[:-1], ratios[1:]):
        if lo <= target <= hi:
            a, b = laws[lo], laws[hi]
            t = (math.log(target) - math.log(lo)) / (math.log(hi) - math.log(lo))
            k0 = math.exp((1.0 - t) * math.log(a.k0) + t * math.log(b.k0))
            return ThreeZoneLaw(target, k0, k0_source="interpolated",
                                k0_band=(min(a.k0, b.k0), max(a.k0, b.k0)),
                                n_points=a.n_points + b.n_points)
    raise AssertionError("unreachable")  # pragma: no cover


def laws_from_campaign(summary: Mapping[str, Any]) -> dict[float, ThreeZoneLaw]:
    """Build the law table consumed by :func:`cw_ice_water` from a campaign result."""
    laws: dict[float, ThreeZoneLaw] = {}
    for group in summary.get("groups", []):
        if group.get("lambda_over_ks") is None:
            # the Nikuradse sand control group has no pitch; it is a check
            # quantity (criterion C2), not a C_w table entry.
            continue
        law = ThreeZoneLaw(
            lambda_over_ks=float(group["lambda_over_ks"]),
            k0=float(group["fit"]["k0"]),
            k0_source="lbm-fit",
            k0_band=(float(group["fit"]["k0_low"]), float(group["fit"]["k0_high"])),
            n_points=int(group["fit"]["n_points"]),
            fit_residual=float(group["fit"].get("rms_log_residual", float("nan"))),
        )
        laws[law.lambda_over_ks] = law
    if not laws:
        raise ValueError("campaign summary contains no fitted groups")
    return laws