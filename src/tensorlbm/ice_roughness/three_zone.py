"""Three-lens cross-adjudication of the ice-underside drag closure.

Mission P103W2C-N01 asks the question that P103W2B-N02 left open: is the
``C_w(k_s^+, lambda/k_s)`` law extracted from the wall-resolved LBM channel a
*closure* or just a fit?  A one-parameter fit (``k0``) can always be drawn
through five points of one morphology, so the honest test is a **three-lens
cross-adjudication**: three friction relations that were derived from
independent physics (or independent experiments) must each reproduce the LBM
data *in the zone where that relation is supposed to hold*, and must disagree
outside it.

The three lenses
----------------
**Lens S -- hydraulically smooth power law.**  With the roughness buried in the
viscous sublayer the wall shear is set by viscosity alone, and the friction
coefficient obeys a power law in the bulk Reynolds number,

    C_f = a Re_b^n ,      n ~ -0.2 ... -0.25                               (S)

whose exponent for the DNS-backed smooth-channel correlation
``U_b^+ = 2.439 ln Re_tau + 5.65`` is :func:`smooth_power_law_exponent`.  Lens S
predicts *no* roughness effect at all: ``C_f,S = C_f,smooth(Re_tau)``.  The
measured exponent over the smooth baseline series is criterion **C3** (the
mission's ``n = -0.22 +- 0.03``).

**Lens C -- Colebrook-White implicit relation.**  Colebrook (1939) interpolates
between the smooth and fully rough asymptotes with an *implicit* equation for
the Darcy friction factor ``lambda = 4 C_f``,

    1/sqrt(lambda) = -2 log10( k_s/(3.7 D_h) + 2.51/(Re_D sqrt(lambda)) )    (C)

with ``D_h`` the hydraulic diameter and ``Re_D = U_b D_h/nu``.  It was fitted to
Moody's experimental database, i.e. it is *not* a restatement of the log law,
and it is the classical transitional-zone relation: its roughness enters
through the relative sand-grain height ``k_s/D_h``.  For the plane channel the
hydraulic diameter is ``D_h = 4h`` (``h`` = channel half-height), the standard
duct convention; that convention is itself an approximation, which is exactly
why the cross-adjudication is informative.  Lens C predicts the transitional
and fully rough *sand* limits without any free parameter once ``k_s`` is given.

**Lens R -- Nikuradse / von Karman fully-rough log law.**  When the roughness
elements pierce the buffer layer the velocity deficit becomes a pure log shift,

    dU^+ = (1/kappa) ln(k_s^+ / k0_Nik),   U_b^+ = U_b,smooth^+ - dU^+       (R)

with ``k0_Nik = exp(kappa (8.5 - B)) = 3.87`` fixed by Nikuradse's additive
constant.  The **slope** of (R) is ``1/kappa``; fitting it to the LBM data is
criterion **C2** (``kappa = 0.41 +- 0.02``).  ``kappa`` therefore enters as a
*check quantity*, never as an imposed input: nothing in (R) is fitted to the
LBM records except the vertical position of each morphology (its ``k0``).

Adjudication
------------
For every measured case the module computes all three predictions and the
relative error of each, then groups the cases into the Nikuradse zones by the
*measured* ``k_s^+``.  The verdict table answers: in the smooth zone does lens
S win, in the transitional zone does lens C win, in the fully rough zone does
lens R win?  A lens is credited only if its median absolute error in its own
zone is below :data:`LENS_AGREEMENT_TOLERANCE` (15 %, the platform's accuracy
bar) *and* it is not beaten by more than a factor of two by another lens in
that zone.

The Colebrook inversion is also used as a **measurement**: given the measured
``C_f`` and ``Re_D``, (C) is solved for the equivalent sand-grain height.  The
ratio ``k_s,eff/k_s`` obtained this way is a morphology attribute comparable
across the ``lambda/k_s`` groups and against Nikuradse's sand (``= 1`` by
construction).

Everything here is pure post-processing of the case records produced by
:mod:`tensorlbm.ice_roughness.sweep`; no number is entered by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import regime_law as rl

__all__ = [
    "LENS_AGREEMENT_TOLERANCE",
    "SMOOTH_KS_PLUS_UPPER",
    "FULLY_ROUGH_KS_PLUS_LOWER",
    "CaseRow",
    "rows_from_records",
    "cf_from_u_plus",
    "re_bulk_from_record",
    "smooth_power_law_exponent",
    "fit_power_law",
    "colebrook_darcy",
    "colebrook_cf_channel",
    "ks_over_d_from_cf_colebrook",
    "fully_rough_cf",
    "cf_smooth_lens",
    "lens_predictions",
    "zone_of_ks_plus",
    "fit_kappa_group_effects",
    "gci_three_grid",
    "resolution_audit",
    "geometric_cap",
    "wall_resolved_cost",
    "adjudicate",
]

#: A lens is credited in its own zone only if its median absolute error there
#: is below this (platform accuracy bar: 15 %).
LENS_AGREEMENT_TOLERANCE: float = 0.15
#: Nikuradse zone boundaries in ``k_s^+`` (sand).  The morphology-specific
#: boundaries of the fitted law are reported alongside.
SMOOTH_KS_PLUS_UPPER: float = 5.0
FULLY_ROUGH_KS_PLUS_LOWER: float = 70.0


# ---------------------------------------------------------------------------
# Case records -> flat rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseRow:
    """One measured channel case, flattened for the adjudication."""

    case_id: str
    kind: str                  # "smooth" | "rough"
    group: str                 # "2" | "5" | "10" | "sand" | "smooth"
    ks_plus: float
    re_tau: float
    re_b: float                # bulk Reynolds number on 2h
    cf: float
    u_plus_bulk: float
    dy_plus: float
    ks_nodes: float
    b_fit: float               # log-law intercept of the measured profile
    d_u_plus: float            # bulk (Hama/Clauser) roughness function
    kappa_self: float          # self-consistent kappa of the measured profile
    log_window_points: int
    h_eff: float = float("nan")  # channel half-height in lattice units
    max_mach: float = float("nan")   # peak bulk Mach number of the run
    tau_relax: float = float("nan")  # LBM relaxation time (margin to 0.5)

    @property
    def log_window_valid(self) -> bool:
        return self.log_window_points >= 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "kind": self.kind, "group": self.group,
            "ks_plus": self.ks_plus, "re_tau": self.re_tau, "re_b": self.re_b,
            "cf": self.cf, "u_plus_bulk": self.u_plus_bulk, "dy_plus": self.dy_plus,
            "ks_nodes": self.ks_nodes, "b_fit": self.b_fit,
            "dU_plus": self.d_u_plus, "kappa_self": self.kappa_self,
            "log_window_points": self.log_window_points,
            "log_window_valid": self.log_window_valid, "h_eff": self.h_eff,
            "max_mach": self.max_mach, "tau_relax": self.tau_relax,
        }


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def re_bulk_from_record(record: Mapping[str, Any]) -> float:
    """Bulk Reynolds number ``U_b * 2h / nu`` of one channel record."""
    u_bulk = _f(record.get("u_bulk"))
    h_eff = _f(record.get("h_eff"))
    nu = _f(record.get("nu"))
    if not (u_bulk > 0.0 and h_eff > 0.0 and nu > 0.0):
        # fall back on the definition through u_tau: Re_b = U_b^+ * 2 Re_tau
        u_plus = _f(record.get("u_plus_bulk"))
        re_tau = _f(record.get("re_tau"))
        return u_plus * 2.0 * re_tau
    return u_bulk * 2.0 * h_eff / nu


def cf_from_u_plus(u_plus_bulk: float) -> float:
    """``C_f = 2 u_tau^2/U_b^2 = 2/(U_b^+)^2`` for a constant-stress channel."""
    return 2.0 / float(u_plus_bulk) ** 2


def rows_from_records(records: Mapping[str, Mapping[str, Any]]) -> list[CaseRow]:
    """Flatten case records into :class:`CaseRow`s (smooth baseline first)."""
    # bulk roughness function of every rough case needs a smooth reference at
    # the *same* Re_tau; the campaign supplies exact matches.
    smooth_by_retau: dict[float, float] = {}
    for rec in records.values():
        if rec.get("kind") == "smooth":
            try:
                smooth_by_retau[round(float(rec["re_tau"]), 6)] = float(rec["u_plus_bulk"])
            except (KeyError, TypeError, ValueError):
                continue

    rows: list[CaseRow] = []
    for cid, rec in sorted(records.items()):
        case = rec.get("case") or {}
        # the engine labels the geometry: "smooth" | "ribs" | "sand"; the
        # adjudication only distinguishes "smooth" from "rough".
        raw_kind = str(rec.get("kind") or case.get("kind") or "rough")
        kind = "smooth" if raw_kind == "smooth" else "rough"
        lam = case.get("lambda_over_ks")
        if kind == "smooth":
            group = "smooth"
        elif case.get("morphology") == "sand" or raw_kind == "sand":
            group = "sand"
        else:
            try:
                group = f"{float(lam):g}"
            except (TypeError, ValueError):
                group = "unknown"
        u_plus = _f(rec.get("u_plus_bulk"))
        cf = _f(rec.get("cf"))
        if not math.isfinite(cf) and u_plus > 0:
            cf = cf_from_u_plus(u_plus)
        ref = smooth_by_retau.get(round(_f(rec.get("re_tau")), 6))
        d_u = (ref - u_plus) if (ref is not None and kind != "smooth") else 0.0
        fit = rec.get("log_fit") or {}
        free_kappa = _f((rec.get("log_fit_free_kappa") or {}).get("kappa")
                        if isinstance(rec.get("log_fit_free_kappa"), Mapping) else
                        rec.get("log_fit_free_kappa"))
        rows.append(CaseRow(
            case_id=cid, kind=kind, group=group,
            ks_plus=_f(rec.get("ks_plus")), re_tau=_f(rec.get("re_tau")),
            re_b=re_bulk_from_record(rec), cf=cf, u_plus_bulk=u_plus,
            dy_plus=_f(rec.get("dy_plus")), ks_nodes=_f(rec.get("ks_nodes")),
            b_fit=_f(fit.get("b")), d_u_plus=d_u, kappa_self=free_kappa,
            log_window_points=int((rec.get("log_window") or {}).get("n_points", 0) or 0),
            h_eff=_f(rec.get("h_eff")), max_mach=_f(rec.get("max_mach")),
            tau_relax=_f(rec.get("tau")),
        ))
    return rows


# ---------------------------------------------------------------------------
# Lens S: hydraulically smooth power law
# ---------------------------------------------------------------------------

def fit_power_law(re_b: Sequence[float], cf: Sequence[float]) -> dict[str, Any]:
    """Least-squares ``ln C_f = ln a + n ln Re_b``.

    Returns the exponent ``n``, its standard error, the prefactor ``a`` and the
    RMS residual in ``ln C_f``.  The error is the honest statistical one from
    the fit; with five baselines it is a *measurement* of the smooth-zone
    exponent, not a copy of a textbook number.
    """
    x = np.log(np.asarray(re_b, dtype=float))
    y = np.log(np.asarray(cf, dtype=float))
    if x.size < 2:
        return {"n": float("nan"), "n_stderr": float("nan"), "a": float("nan"),
                "rms_ln_residual": float("nan"), "n_points": int(x.size)}
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(x.size - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return {
        "n": float(coef[1]),
        "n_stderr": float(math.sqrt(max(cov[1, 1], 0.0))),
        "a": float(math.exp(coef[0])),
        "rms_ln_residual": float(math.sqrt(float(resid @ resid) / max(x.size, 1))),
        "n_points": int(x.size),
    }


def smooth_power_law_exponent(re_tau: float, *, dln: float = 1e-3) -> float:
    """Local exponent ``d ln C_f,smooth/d ln Re_b`` of the smooth channel law.

    The smooth series is *not* exactly a power law, so the local exponent drifts
    with Reynolds number; this function returns the value of (S) at a given
    ``Re_tau`` and is what the measured C3 slope should be compared against.
    """
    def cf(rt: float) -> float:
        return rl.cf_channel_smooth_reference(rt)

    def re_b(rt: float) -> float:
        u_plus = math.sqrt(2.0 / cf(rt))
        return u_plus * 2.0 * rt

    a, b = math.log(re_tau), math.log(re_tau) + dln
    return (math.log(cf(math.exp(b))) - math.log(cf(math.exp(a)))) / (
        math.log(re_b(math.exp(b))) - math.log(re_b(math.exp(a))))


# ---------------------------------------------------------------------------
# Lens C: Colebrook-White implicit relation
# ---------------------------------------------------------------------------

def colebrook_darcy(re_d: float, rel_rough: float = 0.0,
                    *, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Darcy friction factor from the Colebrook-White equation (C).

    Solved for ``sqrt(lambda)`` by fixed-point iteration on
    ``1/sqrt(lambda) = -2 log10(rel + 2.51/(Re sqrt(lambda)))``, which is a
    contraction (``|d/dx| < 0.3``) and therefore needs no bracketing.  Seed is
    the Swamee-Jain explicit approximation, which is within 1 % everywhere.
    """
    re_d = float(re_d)
    rel = max(float(rel_rough), 0.0)
    if re_d <= 0.0:
        raise ValueError("Re must be positive")
    # Swamee-Jain explicit seed (1976)
    if rel > 0.0:
        seed = 0.25 / (math.log10(rel / 3.7 + 5.74 / re_d ** 0.9)) ** 2
    else:
        seed = 0.3164 / re_d ** 0.25 if re_d < 1e5 else 0.25 / (
            math.log10(5.74 / re_d ** 0.9)) ** 2
    lam = seed
    for _ in range(max_iter):
        x = 1.0 / math.sqrt(lam)
        rhs = -2.0 * math.log10(rel / 3.7 + 2.51 * x / re_d)
        lam_new = 1.0 / rhs ** 2
        if abs(lam_new - lam) <= tol * max(lam_new, 1e-30):
            return float(lam_new)
        lam = 0.5 * (lam + lam_new)
    return float(lam)


def colebrook_cf_channel(re_b: float, ks_over_h: float, *,
                         hydraulic_diameter_factor: float = 4.0) -> dict[str, float]:
    """Colebrook prediction of the channel ``C_f`` (lens C).

    ``D_h = hydraulic_diameter_factor * h`` (default ``4h``, the standard duct
    convention for a plane channel), ``Re_D = U_b D_h/nu = Re_b * factor/2``.
    """
    d_h_factor = float(hydraulic_diameter_factor)
    re_d = float(re_b) * d_h_factor / 2.0
    rel = float(ks_over_h) / d_h_factor
    lam = colebrook_darcy(re_d, rel)
    return {
        "darcy_lambda": float(lam),
        "C_f": float(lam / 4.0),
        "Re_D": float(re_d),
        "ks_over_D_h": float(rel),
        "D_h_over_h": d_h_factor,
    }


def ks_over_d_from_cf_colebrook(cf: float, re_d: float) -> float:
    """Invert (C): the relative sand roughness that reproduces a measured ``C_f``.

    Used as a *measurement*: it yields the equivalent sand-grain height of the
    simulated morphology, ``k_s,eff/k_s``.  Returns ``nan`` when the measured
    ``C_f`` lies above what the Colebrook relation can produce at that
    Reynolds number (i.e. the geometry is "rougher than sand" in the Colebrook
    sense), which is reported rather than hidden.
    """
    cf = float(cf)
    re_d = float(re_d)
    if not (cf > 0.0 and re_d > 0.0):
        return float("nan")
    lam = 4.0 * cf
    x = 1.0 / math.sqrt(lam)
    # from 1/sqrt(lam) = -2 log10(rel/3.7 + 2.51 x/Re): solve for rel
    inner = 10.0 ** (-x / 2.0)
    rel = 3.7 * (inner - 2.51 * x / re_d)
    return float(rel) if rel > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# Lens R: Nikuradse / von Karman fully-rough log law
# ---------------------------------------------------------------------------

def cf_smooth_lens(re_tau: float) -> float:
    """Lens S prediction: the smooth channel ``C_f`` at this ``Re_tau``."""
    return rl.cf_channel_smooth_reference(float(re_tau))


def fully_rough_cf(re_tau: float, ks_plus: float, *,
                   kappa: float = rl.KAPPA,
                   k0: float = rl.K0_NIKURADSE) -> float:
    """Lens R prediction of ``C_f`` from the Nikuradse log-shift (R).

    ``U_b^+ = U_b,smooth^+(Re_tau) - (1/kappa) ln(k_s^+/k0)``; ``k0`` defaults
    to the Nikuradse sand value :data:`tensorlbm.ice_roughness.regime_law.K0_NIKURADSE`
    so that sand is reproduced exactly, and can be set to a fitted morphological
    value (``k_s,eff/k_s = k0/k0_Nik``).
    """
    u_smooth = math.sqrt(2.0 / cf_smooth_lens(re_tau))
    d_u = (1.0 / float(kappa)) * math.log(max(float(ks_plus), 1e-12) / float(k0))
    u_rough = u_smooth - d_u
    if u_rough <= 0.0:
        return float("nan")
    return 2.0 / u_rough ** 2


def zone_of_ks_plus(ks_plus: float, *,
                    smooth_upper: float = SMOOTH_KS_PLUS_UPPER,
                    fully_rough_lower: float = FULLY_ROUGH_KS_PLUS_LOWER) -> str:
    """Nikuradse zone label of a *measured* ``k_s^+``."""
    if not math.isfinite(float(ks_plus)):
        return "unknown"
    if ks_plus <= smooth_upper:
        return "smooth"
    if ks_plus >= fully_rough_lower:
        return "fully_rough"
    return "transitional"


def lens_predictions(row: CaseRow, *,
                     kappa: float = rl.KAPPA,
                     k0_morphology: float | None = None,
                     hydraulic_diameter_factor: float = 4.0) -> dict[str, Any]:
    """All three lens predictions for one measured case, with their errors."""
    cf_s = cf_smooth_lens(row.re_tau)
    h_eff = _h_eff_of(row)
    cf_c = colebrook_cf_channel(
        row.re_b, row.ks_nodes / h_eff if h_eff > 0 else float("nan"),
        hydraulic_diameter_factor=hydraulic_diameter_factor)["C_f"]
    cf_r = fully_rough_cf(row.re_tau, row.ks_plus, kappa=kappa,
                          k0=k0_morphology if k0_morphology else rl.K0_NIKURADSE)
    cf_nik = fully_rough_cf(row.re_tau, row.ks_plus)   # sand reference

    def err(pred: float) -> float:
        if not (math.isfinite(pred) and pred > 0.0 and row.cf > 0.0):
            return float("nan")
        return (row.cf - pred) / pred

    return {
        "case_id": row.case_id, "group": row.group, "k_s_plus": row.ks_plus,
        "zone": zone_of_ks_plus(row.ks_plus),
        "cf_measured": row.cf, "cf_lens_smooth": cf_s, "cf_lens_colebrook": cf_c,
        "cf_lens_fully_rough": cf_r, "cf_lens_nikuradse_sand": cf_nik,
        "rel_err_lens_smooth": err(cf_s), "rel_err_lens_colebrook": err(cf_c),
        "rel_err_lens_fully_rough": err(cf_r),
        "rel_err_vs_nikuradse_sand": err(cf_nik),
    }


def _h_eff_of(row: CaseRow) -> float:
    """Channel half-height in lattice units of the record (measured if present).

    ``k_s^+ = Re_tau * k_s / h_eff`` is the campaign's design identity, so
    ``h_eff = Re_tau * k_s / k_s^+`` is the fallback when the record does not
    carry the measured value (the identity is exact to the ``k_s`` discretisation
    used by the geometry builder).
    """
    if math.isfinite(float(row.h_eff)) and row.h_eff > 0.0:
        return float(row.h_eff)
    ks_plus = float(row.ks_plus)
    if not (ks_plus > 0.0 and row.re_tau > 0.0 and row.ks_nodes > 0.0):
        return float("nan")
    return row.re_tau * row.ks_nodes / ks_plus


# ---------------------------------------------------------------------------
# kappa from the fully rough data (criterion C2)
# ---------------------------------------------------------------------------

def fit_kappa_group_effects(rows: Iterable[CaseRow], *,
                            ks_plus_min: float = FULLY_ROUGH_KS_PLUS_LOWER,
                            require_log_window: bool = True) -> dict[str, Any]:
    """Fit ``1/kappa`` in ``dU^+ = c_group + (1/kappa) ln k_s^+`` over all groups.

    A common slope with per-morphology intercepts: the slope is the log-law
    constant *shared by every roughness geometry*, the intercepts absorb
    ``k0``.  This is the cleanest estimator of ``kappa`` the campaign supports,
    and it is exactly the criterion C2 question ("is the fully-rough slope
    1/0.41?").  ``kappa`` is never an input here.
    """
    use = [r for r in rows
           if r.kind == "rough" and math.isfinite(r.ks_plus) and r.ks_plus >= ks_plus_min
           and math.isfinite(r.d_u_plus)
           and (r.log_window_valid or not require_log_window)]
    groups = sorted({r.group for r in use})
    if len(use) < 3 or len(groups) < 1:
        return {"kappa": float("nan"), "slope": float("nan"),
                "slope_stderr": float("nan"), "n_points": len(use),
                "groups": groups, "passed": False,
                "reason": "fewer than 3 fully-rough points with a valid log window"}
    # design matrix: one intercept per group + common slope on ln k_s^+
    cols = [np.array([1.0 if r.group == g else 0.0 for r in use]) for g in groups]
    cols.append(np.array([math.log(r.ks_plus) for r in use]))
    A = np.vstack(cols).T
    y = np.array([r.d_u_plus for r in use])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[-1])
    resid = y - A @ coef
    dof = max(use.__len__() - A.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr = float(math.sqrt(max(cov[-1, -1], 0.0)))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
        stderr = float("nan")
    kappa = 1.0 / slope if slope > 0 else float("nan")
    kappa_err = stderr / slope ** 2 if slope > 0 and math.isfinite(stderr) else float("nan")
    return {
        "kappa": float(kappa),
        "kappa_stderr": float(kappa_err),
        "slope": float(slope),
        "slope_stderr": float(stderr),
        "intercepts": {g: float(coef[i]) for i, g in enumerate(groups)},
        "n_points": len(use),
        "groups": groups,
        "rms_dU_plus_residual": float(math.sqrt(float(resid @ resid) / max(len(use), 1))),
        "ks_plus_min": float(ks_plus_min),
        "passed_band": bool(math.isfinite(kappa) and abs(kappa - rl.KAPPA) <= 0.02),
    }


# ---------------------------------------------------------------------------
# Grid convergence index (criterion C4)
# ---------------------------------------------------------------------------

def gci_three_grid(fine: float, medium: float, coarse: float,
                   *, ratio: float = 2.0, safety: float = 1.25) -> dict[str, Any]:
    """Roache GCI for a uniform refinement ratio (three grids).

    Standard three-grid procedure: the observed order ``p`` solves
    ``(f_coarse - f_medium)/(f_medium - fine) = r^p`` (iteratively, with the
    sign-safe Newton iteration of Roache 1994); then
    ``GCI = Fs |(f_medium-fine)/fine| / (r^p - 1)``.  Oscillatory (non
    monotone) data are reported with ``p = nan`` and the bracket
    ``|eps|/|r-1|`` so that a failed convergence cannot masquerade as a pass.
    """
    f1, f2, f3 = float(fine), float(medium), float(coarse)
    if not all(math.isfinite(v) and v != 0.0 for v in (f1, f2, f3)):
        return {"p": float("nan"), "gci_fine": float("nan"), "passed": False,
                "reason": "non-finite values"}
    eps21 = (f2 - f1) / f1
    eps32 = (f3 - f2) / f2
    r = float(ratio)
    if eps21 == 0.0 or eps32 / eps21 <= 0.0:
        return {"p": float("nan"), "gci_fine": float("nan"),
                "eps_medium_fine": float(eps21), "eps_coarse_medium": float(eps32),
                "passed": False, "reason": "non-monotone or identical values"}
    q = eps32 / eps21
    p = math.log(q) / math.log(r)
    gci = safety * abs(eps21) / (r ** p - 1.0) if abs(r ** p - 1.0) > 1e-12 else float("inf")
    return {
        "p": float(p), "gci_fine": float(gci),
        "eps_medium_fine": float(eps21), "eps_coarse_medium": float(eps32),
        "ratio": r, "safety_factor": float(safety),
        "passed": bool(math.isfinite(gci) and abs(gci) <= 0.05),
    }


# ---------------------------------------------------------------------------
# Resolution audit -- can this campaign support a roughness function at all?
# ---------------------------------------------------------------------------

def resolution_audit(rows: Sequence[CaseRow], *,
                     dy_plus_wall_resolved: float = 2.0,
                     min_log_window_points: int = 3) -> dict[str, Any]:
    """Quantify whether the channel runs resolve the wall layer they are read with.

    A wall-resolved smooth-channel measurement needs ``dy^+ = u_tau/nu`` (the
    first-node wall distance) of order one; the roughness function read from a
    log-layer fit needs enough nodes between ``3 k_s^+`` and ``0.35 Re_tau``.
    This function reports, from the records alone:

    * ``dy^+`` per case and its growth with ``Re_tau`` (fixed by the design,
      ``dy^+ = Re_tau/h_eff``), so the wall-resolved fraction is a number;
    * the smooth-baseline error as a function of ``dy^+`` and its fitted
      exponent -- the error is a *discretisation* effect, not a turbulence
      model difference, if the exponent is positive and monotone;
    * how many rough cases actually have a usable log window;
    * the compressibility (``max_mach``) and LBM stability (``tau_relax``)
      margins.

    The verdict string says in words whether the campaign can carry the
    closure; it is *not* an opinion, each clause quotes the number that
    triggered it.
    """
    smooth = [r for r in rows if r.kind == "smooth"]
    rough = [r for r in rows if r.kind == "rough"]

    err_rows = []
    for r in smooth:
        ref = cf_smooth_lens(r.re_tau)
        if not (math.isfinite(ref) and ref > 0.0 and r.cf > 0.0):
            continue
        err_rows.append({"case_id": r.case_id, "re_tau": r.re_tau, "dy_plus": r.dy_plus,
                         "cf": r.cf, "cf_reference": float(ref),
                         "rel_error": float((r.cf - ref) / ref)})
    exponent = float("nan")
    if len(err_rows) >= 2:
        x = [math.log(max(e["dy_plus"], 1e-12)) for e in err_rows]
        y = [math.log(abs(e["rel_error"])) for e in err_rows if abs(e["rel_error"]) > 0]
        if len(y) == len(x):
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            den = sum((xi - xm) ** 2 for xi in x)
            if den > 0:
                exponent = float(sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / den)

    dy = [r.dy_plus for r in rows if math.isfinite(r.dy_plus)]
    nwin = [r.log_window_points for r in rough]
    usable = [r for r in rough if r.log_window_valid and r.log_window_points >= min_log_window_points]
    mach = [r.max_mach for r in rows if math.isfinite(r.max_mach)]
    tau = [r.tau_relax for r in rows if math.isfinite(r.tau_relax)]

    wall_resolved = bool(dy) and max(dy) <= dy_plus_wall_resolved
    clauses: list[str] = []
    if not wall_resolved:
        clauses.append(f"dy^+ spans {min(dy):.1f}-{max(dy):.1f} (wall-resolved needs <= "
                       f"{dy_plus_wall_resolved:g})")
    if usable == []:
        clauses.append(f"0 of {len(rough)} rough cases have a log window of >= "
                       f"{min_log_window_points} nodes (max {max(nwin) if nwin else 0})")
    if err_rows and max(abs(e["rel_error"]) for e in err_rows) > 0.03:
        clauses.append(f"smooth baseline error reaches "
                       f"{100 * max(abs(e['rel_error']) for e in err_rows):.0f} % (threshold 3 %)")
    if mach and max(mach) > 0.2:
        clauses.append(f"peak bulk Mach {max(mach):.2f} (> 0.2: compressibility error is no longer negligible)")
    if tau and min(tau) < 0.505:
        clauses.append(f"minimum relaxation time {min(tau):.4f} (margin to the 0.5 stability edge "
                       f"is {min(tau) - 0.5:.4f})")

    return {
        "dy_plus_range": [float(min(dy)), float(max(dy))] if dy else [],
        "dy_plus_per_case": {r.case_id: r.dy_plus for r in rows if math.isfinite(r.dy_plus)},
        "wall_resolved": wall_resolved,
        "smooth_error_vs_dy_plus": err_rows,
        "smooth_error_dy_plus_exponent": exponent,
        "rough_log_window_points": {
            "max": int(max(nwin)) if nwin else 0,
            "median": float(np.median(nwin)) if nwin else float("nan"),
            "n_usable": len(usable),
            "n_rough": len(rough),
        },
        "max_mach": float(max(mach)) if mach else float("nan"),
        "min_tau_relax": float(min(tau)) if tau else float("nan"),
        "verdict": ("campaign is adequate" if not clauses else
                    "campaign cannot carry the closure: " + "; ".join(clauses)),
        "passed": not clauses,
    }


def wall_resolved_cost(*, ks_plus_max: float, dy_plus_target: float = 1.0,
                       ks_over_h: float = 0.1, turnovers: float = 4.0,
                       u_tau: float = 0.05, cells_per_update: float = 1.0,
                       throughput_mlups: float = 100.0) -> dict[str, Any]:
    """Lattice cost of a *wall-resolved* campaign that would reach ``k_s^+``.

    The design identities of the campaign fix everything once the target
    ``k_s^+`` and the wall resolution ``dy^+ = u_tau/nu`` (wall units per lattice
    node) are chosen::

        k_s^+ = k_s_nodes * dy^+          -> k_s_nodes = k_s^+ / dy^+
        k_s/h <= ks_over_h                -> h = k_s_nodes / ks_over_h
        Re_tau = dy^+ h = k_s^+/ks_over_h (independent of dy^+)

    so a *tighter* wall resolution does **not** raise the Reynolds-number
    requirement -- it shrinks the elements and grows the grid.  The function
    returns the grid, the step count for ``turnovers`` eddy turnovers
    (``2h/u_tau`` lattice steps each) and the wall time on the given throughput
    -- the number that decides whether a criterion is affordable.
    """
    ks_nodes = float(ks_plus_max) / float(dy_plus_target)
    h = ks_nodes / float(ks_over_h)
    re_tau = float(dy_plus_target) * h
    cells = h * h * (2.0 * h + 2.0)
    steps = float(turnovers) * 2.0 * h / float(u_tau)
    updates = cells * steps * float(cells_per_update)
    seconds = updates / (float(throughput_mlups) * 1.0e6)
    return {
        "ks_plus_max": float(ks_plus_max),
        "dy_plus_target": float(dy_plus_target),
        "ks_over_h": float(ks_over_h),
        "ks_nodes": float(ks_nodes),
        "h_nodes": float(h),
        "re_tau": float(re_tau),
        "cells": float(cells),
        "steps": float(steps),
        "cell_updates": float(updates),
        "turns": float(updates / 1.0e12),
        "wall_hours_at_100_MLUPS": float(seconds / 3600.0),
        "memory_GB_double_19": float(cells * 19 * 2 * 8 / 1e9),
        "feasible": bool(seconds <= 24 * 3600 and cells * 19 * 2 * 8 / 1e9 <= 64.0),
    }


def geometric_cap(rows: Sequence[CaseRow], *,
                  kappa: float = rl.KAPPA, b_fully_rough: float = rl.B_FULLY_ROUGH) -> dict[str, Any]:
    """Largest roughness function a channel of relative roughness ``k_s/h`` can show.

    In the fully rough limit the *slowest possible* bulk flow is the log law
    evaluated at the channel centreline,

        U_b^+ >= (1/kappa) ln(h_eff/k_s) + 8.5                             (cap)

    (nothing can be slower than "fully rough everywhere"), so the roughness
    function measured against the smooth baseline is bounded by

        dU^+ <= U_b,smooth^+(Re_tau) - [(1/kappa) ln(h_eff/k_s) + 8.5].

    With ``k_s/h = 0.106`` (this campaign's geometry) that ceiling is only
    ~5 at ``Re_tau = 850``, while Nikuradse's sand-grain ``dU^+`` is already
    9.3 at ``k_s^+ = 93``: **the geometry cannot represent sand-grain roughness
    above k_s^+ ~ 33**, no matter how well it is resolved.  This is a *design*
    limit, independent of ``dy^+``, and it is why the fitted ``k0`` values of
    the dense-pitch groups come out an order of magnitude too large.
    """
    per_case = []
    capped = 0
    for r in rows:
        if r.kind != "rough" or not math.isfinite(r.ks_plus):
            continue
        h_eff = _h_eff_of(r)
        ks = r.ks_nodes
        if not (h_eff > 0 and ks > 0):
            continue
        u_smooth = math.sqrt(2.0 / cf_smooth_lens(r.re_tau))
        ceiling = u_smooth - ((1.0 / kappa) * math.log(h_eff / ks) + b_fully_rough)
        fraction = (r.d_u_plus / ceiling) if ceiling > 0 else float("nan")
        # the k_s^+ above which Nikuradse's law would exceed the ceiling
        ks_plus_cap = rl.K0_NIKURADSE * math.exp(kappa * ceiling) if ceiling > 0 else float("nan")
        over = bool(math.isfinite(fraction) and fraction > 0.8)
        capped += int(over)
        per_case.append({
            "case_id": r.case_id, "ks_plus": r.ks_plus, "re_tau": r.re_tau,
            "ks_over_h": float(ks / h_eff), "dU_plus": r.d_u_plus,
            "dU_plus_ceiling": float(ceiling), "ceiling_fraction": float(fraction),
            "nikuradse_dU_plus": float(rl.nikuradse_delta_u_plus(r.ks_plus)),
            "ks_plus_of_the_ceiling": float(ks_plus_cap),
            "at_or_above_ceiling": over,
        })
    fractions = [c["ceiling_fraction"] for c in per_case if math.isfinite(c["ceiling_fraction"])]
    return {
        "relation": "dU^+ <= U_b,smooth^+ - [(1/kappa) ln(h_eff/k_s) + 8.5]",
        "n_cases": len(per_case),
        "n_at_or_above_ceiling": capped,
        "median_ceiling_fraction": float(np.median(fractions)) if fractions else float("nan"),
        "ks_plus_where_ceiling_binds": float(np.median(
            [c["ks_plus_of_the_ceiling"] for c in per_case])) if per_case else float("nan"),
        "per_case": per_case,
        "verdict": ("geometry can carry the roughness function" if capped == 0 else
                    f"{capped} of {len(per_case)} cases sit at or above the geometric ceiling "
                    f"(k_s/h ~ {np.median([c['ks_over_h'] for c in per_case]):.3f}); the "
                    "sand-grain law is out of reach there"),
        "passed": bool(capped == 0),
    }


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def adjudicate(rows: Sequence[CaseRow], *,
               tolerance: float = LENS_AGREEMENT_TOLERANCE,
               c3_target: float = -0.22,
               c3_band: float = 0.03) -> dict[str, Any]:
    """Run the three-lens cross-adjudication and the C1-C3 verdicts.

    Returns a JSON-ready dict; every number is computed from ``rows``.
    """
    rough = [r for r in rows if r.kind == "rough"]
    smooth = [r for r in rows if r.kind == "smooth"]

    # ---- C1: smooth baselines against the DNS-backed channel law ----------
    c1_rows = []
    for r in smooth:
        ref = cf_smooth_lens(r.re_tau)
        c1_rows.append({
            "case_id": r.case_id, "re_tau": r.re_tau, "re_b": r.re_b,
            "cf": r.cf, "cf_reference": float(ref),
            "rel_error": float((r.cf - ref) / ref), "dy_plus": r.dy_plus,
        })
    max_c1 = max((abs(x["rel_error"]) for x in c1_rows), default=float("nan"))
    c1 = {
        "reference": "U_b^+ = 2.439 ln(Re_tau) + 5.65 (Moser 1999 / Lee-Moser 2015)",
        "threshold_rel_error": 0.03, "max_abs_rel_error": float(max_c1),
        "n_cases": len(c1_rows),
        "passed": bool(c1_rows) and max_c1 <= 0.03, "rows": c1_rows,
    }

    # ---- C3: smooth-zone exponent of the power law ------------------------
    power = fit_power_law([r.re_b for r in smooth], [r.cf for r in smooth])
    exponents_local = [smooth_power_law_exponent(r.re_tau) for r in smooth]
    c3 = {
        "target": float(c3_target), "band": float(c3_band),
        "fit": power,
        "local_exponent_reference": [float(e) for e in exponents_local],
        "local_exponent_mean": _median(exponents_local),
        "passed": bool(math.isfinite(power["n"])
                       and abs(power["n"] - c3_target) <= c3_band),
        "note": "measured slope of ln Cf vs ln Re_b over the smooth baseline series",
    }

    # ---- lens comparison --------------------------------------------------
    per_group = {g: r for g, r in ((g, _group_k0(rough, g)) for g in sorted({r.group for r in rough}))}
    lens_rows: list[dict[str, Any]] = []
    for r in rough:
        if not math.isfinite(r.ks_plus):
            continue
        k0_g = per_group.get(r.group)
        lens_rows.append(lens_predictions(r, k0_morphology=k0_g))

    zone_table: dict[str, dict[str, Any]] = {}
    for zone in ("smooth", "transitional", "fully_rough"):
        sel = [x for x in lens_rows if x["zone"] == zone]
        if not sel:
            zone_table[zone] = {"n_cases": 0, "passed": False,
                                "reason": "no measured case in this zone"}
            continue
        med = {
            "lens_smooth": _median([x["rel_err_lens_smooth"] for x in sel]),
            "lens_colebrook": _median([x["rel_err_lens_colebrook"] for x in sel]),
            "lens_fully_rough": _median([x["rel_err_lens_fully_rough"] for x in sel]),
        }
        expected = {"smooth": "lens_smooth", "transitional": "lens_colebrook",
                    "fully_rough": "lens_fully_rough"}[zone]
        best = min(med, key=lambda k: abs(med[k]))
        zone_table[zone] = {
            "n_cases": len(sel),
            "median_rel_error": med,
            "median_abs_error": {k: abs(v) for k, v in med.items()},
            "expected_lens": expected,
            "best_lens": best,
            "expected_lens_abs_error": abs(med[expected]),
            "cases": sel,
            "passed": bool(abs(med[expected]) <= tolerance and best == expected),
        }

    # ---- C2: kappa from the fully rough slope ----------------------------
    # Two scopes: the profile (log-intercept) measurement, which needs a real
    # log window, and the bulk (Hama/Clauser) measurement, which is available
    # for every case but inherits the smooth-baseline error.  The criterion is
    # judged on the stricter scope when it has points; if the campaign has no
    # usable log window the fallback is used *and reported as such*.
    kappa_strict = fit_kappa_group_effects(rough, require_log_window=True)
    kappa_bulk = fit_kappa_group_effects(rough, require_log_window=False)
    kappa_fit = kappa_strict if kappa_strict.get("n_points", 0) >= 3 else kappa_bulk
    scope = "log_intercept" if kappa_fit is kappa_strict else "bulk_hama_clauser"
    kappa_self = _median([r.kappa_self for r in rough if r.log_window_valid])
    c2 = {
        "target_kappa": float(rl.KAPPA), "band": 0.02,
        "scope": scope,
        "fit": kappa_fit,
        "fit_log_intercept_scope": kappa_strict,
        "fit_bulk_scope": kappa_bulk,
        "kappa_from_self_consistent_log_fit_median": float(kappa_self),
        "passed": bool(kappa_fit.get("passed_band")),
    }

    audit = resolution_audit(rows)
    audit["geometric_cap"] = geometric_cap(rows)
    scenarios = [
        ("k_s_plus_500_dy_plus_1", dict(ks_plus_max=500.0, dy_plus_target=1.0)),
        ("k_s_plus_500_dy_plus_10", dict(ks_plus_max=500.0, dy_plus_target=10.0)),
        ("k_s_plus_100_dy_plus_10", dict(ks_plus_max=100.0, dy_plus_target=10.0)),
        ("k_s_plus_100_dy_plus_5", dict(ks_plus_max=100.0, dy_plus_target=5.0)),
        ("k_s_plus_50_dy_plus_5", dict(ks_plus_max=50.0, dy_plus_target=5.0)),
    ]
    cost = wall_resolved_cost(**scenarios[0][1])
    cost_scoped = {name: wall_resolved_cost(**kw) for name, kw in scenarios}

    return {
        "generated_by": "tensorlbm.ice_roughness.three_zone.adjudicate",
        "resolution_audit": audit,
        "wall_resolved_cost_k_s_plus_500": cost,
        "wall_resolved_cost_scenarios": cost_scoped,
        "n_rough": len(rough), "n_smooth": len(smooth),
        "C1_smooth_lens": c1,
        "C2_kappa_fully_rough": c2,
        "C3_smooth_zone_exponent": c3,
        "lenses": {
            "definitions": {
                "lens_smooth": "Cf = Cf_smooth(Re_tau)  [no roughness]",
                "lens_colebrook": "Colebrook-White (1939) implicit, D_h = 4h, sand height k_s",
                "lens_fully_rough": "Nikuradse/von Karman log shift, k0 = k0_Nik (sand) or fitted k0",
            },
            "zone_adjudication": zone_table,
            "tolerance": float(tolerance),
        },
        "all_passed": bool(c1["passed"] and c2["passed"] and c3["passed"]
                           and all(z.get("passed") for z in zone_table.values())),
    }


def _group_k0(rows: Sequence[CaseRow], group: str) -> float:
    """Fitted morphology ``k0`` of a ``lambda/k_s`` group from its own points.

    Uses the bulk roughness function and the same estimator as the campaign
    (:func:`tensorlbm.ice_roughness.regime_law.k0_from_delta_u_plus`), averaged
    in ``ln`` over the fully-rough points of the group.
    """
    vals = []
    for r in rows:
        if r.group != group or not math.isfinite(r.d_u_plus) or r.d_u_plus <= 0.0:
            continue
        if r.ks_plus < FULLY_ROUGH_KS_PLUS_LOWER:
            continue
        k0 = rl.k0_from_delta_u_plus(r.ks_plus, r.d_u_plus)
        if math.isfinite(k0) and k0 > 0.0:
            vals.append(math.log(k0))
    return float(math.exp(float(np.mean(vals)))) if vals else rl.K0_NIKURADSE