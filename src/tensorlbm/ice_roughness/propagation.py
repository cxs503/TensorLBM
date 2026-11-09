"""Propagation of the three-zone roughness closure into the sea-ice drift model.

Mission P103W2C-N01 criterion **C5** is a *propagation* criterion: a closure
that is numerically perfect but changes nothing in the drift model would not
have earned its place.  The constant entry point of the drift model (the
``C_w = 5.5e-3`` of ``m5682``) is therefore replaced by the measured closure and
the answer is read off the free-drift balance of
:mod:`tensorlbm.ice_roughness.freedrift`:

    drift speed   U  ->  U * sqrt(C_w,smooth/C_w)          (leading order)
    drift angle   alpha: tan(alpha) = rho_i h_i f/(rho_w C_w U)

so a *larger* ``C_w`` (rougher ice) makes the ice drift **faster** and
**straighter**.  The mission quotes the expected size of that effect
(10-40 % in speed, 5-15 deg in Nansen angle) and demands an explicit answer to
two questions:

1. does the propagated change actually fall inside those bands (i.e. is the
   closure strong enough to matter and weak enough to stay credible)?
2. does the resulting drift state itself lie inside the observational band
   (Nansen angle 20-40 deg, canonical 28 deg; observed drift/wind ratio
   1.5-4 %, Nansen's 2 % rule) -- and if it does *not*, that must be reported
   as an honest FAIL, not softened.

Two back-fill forms are distinguished, because they are *not* equivalent:

**absolute** (:func:`tensorlbm.ice_roughness.freedrift.free_drift_solution`)
    the log law with the measured roughness function, ``C_w`` obtained from the
    velocity at ``z_ref``.  No constant enters, so nothing is counted twice --
    this is the form a drift model should consume, and it is the default here.

**amplification**
    (:func:`tensorlbm.ice_roughness.freedrift.drift_with_roughness_amplification`)
    ``C_w = C_w,const * Cf_rough/Cf_smooth``.  It is only legitimate if the
    constant it multiplies is a *smooth-ice* value; the ``m5682`` entry
    (``5.5e-3``) already contains the roughness of real ice, so this form
    double-counts and is reported as a sensitivity bound, not as the answer.

Both questions are answered per condition in :func:`propagation_verdict`, with
the band membership spelled out as booleans so a downstream reader can neither
over- nor under-read it.  Nothing here is tuned: the ice-keel geometry comes
from :mod:`tensorlbm.ice_roughness.geometry`, the closure from the LBM fit, and
the environment defaults from Lepparanta (2011).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import freedrift as fd
from . import geometry as ice_geom
from . import regime_law as rl

__all__ = [
    "M5682_CONSTANT_CW",
    "SPEED_CHANGE_BAND_PCT",
    "ANGLE_CHANGE_BAND_DEG",
    "NANSEN_ANGLE_BAND_DEG",
    "WIND_FACTOR_BAND",
    "DEFAULT_KS_M",
    "DEFAULT_LAMBDA_OVER_KS",
    "cw_table",
    "propagation_verdict",
    "nansen_bands",
]

#: The constant ice/water drag coefficient replaced by this mission (``m5682``).
M5682_CONSTANT_CW: float = 5.5e-3
#: Required *change* of the propagated drift speed (mission, percent).
SPEED_CHANGE_BAND_PCT: tuple[float, float] = (10.0, 40.0)
#: Required *change* of the propagated Nansen angle (mission, degrees).
ANGLE_CHANGE_BAND_DEG: tuple[float, float] = (5.0, 15.0)
#: Observed Nansen deflection band (Lepparanta 2011 ch. 4; Nansen 1902).
NANSEN_ANGLE_BAND_DEG: tuple[float, float] = (20.0, 40.0)
#: Observed free-drift / wind speed ratio band (Nansen's 2 % rule sits in it).
WIND_FACTOR_BAND: tuple[float, float] = (0.015, 0.040)
#: Ice-keel RMS height levels used for the C_w table (m).  Pressure-ridge
#: keels under level ice are 0.2 m (thin first-year) to 1.0 m (ridged) --
#: cf. :func:`tensorlbm.ice_roughness.geometry.ice_keel_roughness`.
DEFAULT_KS_M: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 1.00)
DEFAULT_LAMBDA_OVER_KS: tuple[float, ...] = (2.0, 5.0, 10.0)


def _band(value: float, band: Sequence[float]) -> bool:
    lo, hi = float(band[0]), float(band[1])
    return bool(math.isfinite(value) and lo <= value <= hi)


def cw_table(
    laws: Mapping[float, rl.ThreeZoneLaw],
    *,
    ks_m: Iterable[float] = DEFAULT_KS_M,
    lambda_over_ks: Iterable[float] = DEFAULT_LAMBDA_OVER_KS,
    u_rel: float = 0.15,
    z_ref_m: float = rl.DEFAULT_REFERENCE_HEIGHT_M,
    nu: float = rl.NU_WATER,
) -> list[dict[str, Any]]:
    """``C_w`` table over (ice-keel roughness, spacing ratio) -- the back-fill.

    ``u_rel = 0.15 m/s`` is the reference ice/water relative velocity; ``C_w``
    is only weakly velocity dependent (it enters through ``ln z^+``), so the
    table is usable as a look-up.  Every row carries ``k_s^+``, the zone and
    the amplification over the smooth case, so the consumer can see whether it
    is inside the fitted range or in an extrapolated corner.
    """
    rows: list[dict[str, Any]] = []
    for lam in lambda_over_ks:
        law = rl.interpolate_law(float(lam), laws)
        extrapolated = abs(float(law.lambda_over_ks) - float(lam)) > 1e-12
        for ks in ks_m:
            try:
                sol = rl.cw_ice_water(u_rel, float(ks), float(lam), law=law,
                                      z_ref_m=z_ref_m, nu=nu)
            except ValueError as exc:  # denormalized point (huge k_s, tiny u_rel)
                rows.append({"lambda_over_ks": float(lam), "k_s_m": float(ks),
                             "error": str(exc)})
                continue
            rows.append({
                "lambda_over_ks": float(lam),
                "k_s_m": float(ks),
                "u_rel_m_s": float(u_rel),
                "k_s_plus": float(sol["ks_plus"]),
                "dU_plus": float(sol["dU_plus"]),
                "C_w": float(sol["C_w"]),
                "amplification_over_smooth": float(sol["C_w"] / _cw_smooth(u_rel, z_ref_m, nu)),
                "regime": sol["regime"],
                "k0": float(law.k0),
                "k_s_eff_over_k_s": float(law.equivalent_sand_grain_ratio()),
                "extrapolated_in_lambda": bool(extrapolated),
            })
    return rows


def _cw_smooth(u_rel: float, z_ref_m: float, nu: float) -> float:
    """Smooth-limit ``C_w`` of the same log-law closure (``k_s -> 0``)."""
    law = rl.ThreeZoneLaw(lambda_over_ks=float("nan"), k0=rl.K0_NIKURADSE,
                          k0_source="smooth-limit")
    return float(rl.solve_friction_velocity(u_rel, 1e-9, law,
                                            z_ref_m=z_ref_m, nu=nu)["C_w"])


def nansen_bands(
    laws: Mapping[float, rl.ThreeZoneLaw],
    *,
    keel: ice_geom.IceKeelRoughness | None = None,
    env: fd.DriftEnvironment | None = None,
    wind_speeds: Iterable[float] = (3.0, 5.0, 7.0, 10.0),
    constant_cw: float = M5682_CONSTANT_CW,
) -> list[dict[str, Any]]:
    """Nansen angle and wind factor as functions of ice-keel roughness.

    For every ``lambda/k_s`` of the fit and every ``k_s`` level this returns the
    drift state under the closure and under the constant entry, so the figure
    "drift angle / wind factor vs ice-underside roughness, with the observed
    band shaded" can be drawn straight from the numbers.
    """
    keel = keel or ice_geom.default_ice_keel()
    env0 = env or fd.DriftEnvironment()
    rows: list[dict[str, Any]] = []
    for lam in sorted(float(k) for k in laws):
        law = laws[lam]
        for ks in DEFAULT_KS_M:
            for ua in wind_speeds:
                e = fd.DriftEnvironment(**{**env0.__dict__, "wind_speed": float(ua)})
                cl = fd.drift_with_roughness_amplification(e, float(ks), law)
                ab = fd.free_drift_solution(e, float(ks), lam, law=law)
                cst = fd.drift_angle(e, float(constant_cw))
                rows.append({
                    "lambda_over_ks": float(lam), "k_s_m": float(ks),
                    "wind_speed_m_s": float(ua),
                    "closure_C_w": cl["C_w"], "closure_ks_plus": cl["ks_plus"],
                    "closure_regime": cl["regime"],
                    "closure_drift_m_s": cl["drift_speed_m_s"],
                    "closure_angle_deg": cl["drift_angle_deg"],
                    "closure_wind_factor": cl["drift_speed_m_s"] / float(ua),
                    "constant_C_w": float(constant_cw),
                    "constant_drift_m_s": cst["drift_speed_m_s"],
                    "constant_angle_deg": cst["drift_angle_deg"],
                    "constant_wind_factor": cst["drift_speed_m_s"] / float(ua),
                    "speed_change_pct": 100.0 * (cl["drift_speed_m_s"] / cst["drift_speed_m_s"] - 1.0),
                    "angle_change_deg": cl["drift_angle_deg"] - cst["drift_angle_deg"],
                    "absolute_C_w": ab["C_w"],
                    "absolute_ks_plus": ab["ks_plus"],
                    "absolute_regime": ab["regime"],
                    "absolute_drift_m_s": ab["drift_speed_m_s"],
                    "absolute_angle_deg": ab["drift_angle_deg"],
                    "absolute_wind_factor": ab["drift_speed_m_s"] / float(ua),
                    "absolute_speed_change_pct": 100.0 * (ab["drift_speed_m_s"]
                                                          / cst["drift_speed_m_s"] - 1.0),
                    "absolute_angle_change_deg": ab["drift_angle_deg"] - cst["drift_angle_deg"],
                    "absolute_in_nansen_band": _band(ab["drift_angle_deg"], NANSEN_ANGLE_BAND_DEG),
                    "absolute_in_wind_factor_band": _band(ab["drift_speed_m_s"] / float(ua),
                                                          WIND_FACTOR_BAND),
                    "in_nansen_band": _band(cl["drift_angle_deg"], NANSEN_ANGLE_BAND_DEG),
                    "in_wind_factor_band": _band(cl["drift_speed_m_s"] / float(ua), WIND_FACTOR_BAND),
                    "keel_reference_k_s_m": keel.ks_m,
                })
    return rows


def _variant_row(
    env: fd.DriftEnvironment,
    keel: ice_geom.IceKeelRoughness,
    law: rl.ThreeZoneLaw,
    constant_cw: float,
    form: str,
) -> dict[str, Any]:
    """One wind speed, one back-fill form, with the band bookkeeping."""
    if form == "absolute":
        cl = fd.free_drift_solution(env, keel.ks_m, float(law.lambda_over_ks), law=law)
    elif form == "amplification":
        cl = fd.drift_with_roughness_amplification(env, keel.ks_m, law)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown back-fill form: {form}")
    cst = fd.drift_angle(env, float(constant_cw))
    speed_change = 100.0 * (cl["drift_speed_m_s"] / cst["drift_speed_m_s"] - 1.0)
    angle_change = cl["drift_angle_deg"] - cst["drift_angle_deg"]
    wf = cl["drift_speed_m_s"] / float(env.wind_speed)
    return {
        "wind_speed_m_s": float(env.wind_speed),
        "form": form,
        "closure": {k: cl[k] for k in ("C_w", "ks_plus", "regime", "drift_speed_m_s",
                                       "drift_angle_deg")},
        "closure_wind_factor": float(wf),
        "constant": {"C_w": float(constant_cw),
                     "drift_speed_m_s": cst["drift_speed_m_s"],
                     "drift_angle_deg": cst["drift_angle_deg"]},
        "speed_change_pct": float(speed_change),
        "angle_change_deg": float(angle_change),
        "speed_change_in_band": _band(abs(speed_change), SPEED_CHANGE_BAND_PCT),
        "angle_change_in_band": _band(abs(angle_change), ANGLE_CHANGE_BAND_DEG),
        "in_observed_angle_band": _band(cl["drift_angle_deg"], NANSEN_ANGLE_BAND_DEG),
        "in_observed_wind_factor_band": _band(wf, WIND_FACTOR_BAND),
    }


def propagation_verdict(
    laws: Mapping[float, rl.ThreeZoneLaw],
    *,
    keel: ice_geom.IceKeelRoughness | None = None,
    env: fd.DriftEnvironment | None = None,
    reference_lambda_over_ks: float = 5.0,
    constant_cw: float = M5682_CONSTANT_CW,
    wind_speeds: Iterable[float] = (3.0, 5.0, 7.0, 10.0),
    forms: Sequence[str] = ("absolute", "amplification"),
) -> dict[str, Any]:
    """Criterion C5: propagate the closure and adjudicate it *explicitly*.

    Returns, per back-fill form and per wind speed, the closure and constant
    drift states, the propagated changes, and four booleans:

    ``speed_change_in_band``        ``|dU/U|`` in 10-40 %
    ``angle_change_in_band``        ``|dalpha|`` in 5-15 deg
    ``in_observed_angle_band``      closure angle in 20-40 deg
    ``in_observed_wind_factor_band`` closure ``U/U_a`` in 1.5-4 %

    The headline ``passed`` belongs to the **absolute** form (the recommended
    back-fill); ``honest_fail`` is set with a plain-language reason whenever
    anything misses, and the numbers that missed stay in the payload.  The
    other forms are reported in ``variants`` for sensitivity.
    """
    keel = keel or ice_geom.default_ice_keel(lambda_over_ks=reference_lambda_over_ks)
    env0 = env or fd.DriftEnvironment()
    law = rl.interpolate_law(float(reference_lambda_over_ks), laws)

    variants: dict[str, Any] = {}
    for form in forms:
        rows = [_variant_row(fd.DriftEnvironment(**{**env0.__dict__, "wind_speed": float(ua)}),
                             keel, law, constant_cw, form)
                for ua in wind_speeds]
        speed_ok = [r["speed_change_in_band"] for r in rows]
        angle_ok = [r["angle_change_in_band"] for r in rows]
        angle_band_ok = [r["in_observed_angle_band"] for r in rows]
        wf_band_ok = [r["in_observed_wind_factor_band"] for r in rows]

        reasons: list[str] = []
        if not all(speed_ok):
            reasons.append("drift-speed change outside 10-40 % at "
                           f"{[r['wind_speed_m_s'] for r in rows if not r['speed_change_in_band']]} m/s")
        if not all(angle_ok):
            reasons.append("Nansen-angle change outside 5-15 deg at "
                           f"{[r['wind_speed_m_s'] for r in rows if not r['angle_change_in_band']]} m/s")
        if not all(angle_band_ok):
            reasons.append("drift angle outside the observed 20-40 deg band at "
                           f"{[r['wind_speed_m_s'] for r in rows if not r['in_observed_angle_band']]} m/s")
        if not all(wf_band_ok):
            reasons.append("wind factor outside the observed 1.5-4 % band at "
                           f"{[r['wind_speed_m_s'] for r in rows if not r['in_observed_wind_factor_band']]} m/s")
        passed = bool(all(speed_ok) and all(angle_ok) and all(angle_band_ok) and all(wf_band_ok))
        variants[form] = {
            "rows": rows,
            "speed_change_pct_range": [float(min(r["speed_change_pct"] for r in rows)),
                                       float(max(r["speed_change_pct"] for r in rows))],
            "angle_change_deg_range": [float(min(r["angle_change_deg"] for r in rows)),
                                       float(max(r["angle_change_deg"] for r in rows))],
            "closure_C_w_range": [float(min(r["closure"]["C_w"] for r in rows)),
                                  float(max(r["closure"]["C_w"] for r in rows))],
            "closure_angle_deg_range": [float(min(r["closure"]["drift_angle_deg"] for r in rows)),
                                        float(max(r["closure"]["drift_angle_deg"] for r in rows))],
            "closure_wind_factor_range": [float(min(r["closure_wind_factor"] for r in rows)),
                                          float(max(r["closure_wind_factor"] for r in rows))],
            "passed": passed,
            "honest_fail": not passed,
            "fail_reasons": reasons,
        }

    head = variants.get("absolute") or next(iter(variants.values()))
    return {
        "criterion": "C5 propagation of the closure into the free-drift model",
        "recommended_form": "absolute",
        "form_note": ("'absolute' = log law + measured roughness function, no constant enters; "
                      "'amplification' = C_w,const * Cf_rough/Cf_smooth, which double-counts "
                      "roughness because the m5682 constant already contains it"),
        "ice_keel": keel.as_dict(),
        "environment": env0.as_dict(),
        "law": law.as_dict(),
        "lambda_over_ks": float(reference_lambda_over_ks),
        "constant_C_w": float(constant_cw),
        "bands": {
            "speed_change_pct": list(SPEED_CHANGE_BAND_PCT),
            "angle_change_deg": list(ANGLE_CHANGE_BAND_DEG),
            "nansen_angle_deg": list(NANSEN_ANGLE_BAND_DEG),
            "wind_factor": list(WIND_FACTOR_BAND),
            "band_interpretation": "|change| is compared with the band (sign reported separately)",
        },
        "variants": variants,
        "rows": head["rows"],
        "speed_change_pct_range": head["speed_change_pct_range"],
        "angle_change_deg_range": head["angle_change_deg_range"],
        "closure_angle_deg_range": head["closure_angle_deg_range"],
        "closure_wind_factor_range": head["closure_wind_factor_range"],
        "passed": head["passed"],
        "honest_fail": head["honest_fail"],
        "fail_reasons": head["fail_reasons"],
    }
