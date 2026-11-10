"""DNS-grade verification face for the plane turbulent channel at Re_tau = 180/395/590.

What this module adds on top of :mod:`tensorlbm.turbulence.les_channel`
(wall-unit reduction, MKM comparison, LES gate set ``C1..C6``)
------------------------------------------------------------------
The ``C``-gate set certifies an *LES* campaign.  A turbulence-model validation
face needs the stricter question: **is the run itself DNS-grade, and does it
reproduce the DNS statistics?**  This module answers that with an explicit,
fail-closed gate set:

``D0`` resolution attestation        ``dy+ <= 1.5, dx+ <= 10, dz+ <= 5``
                                     (Moin & Mahesh 1998; Kim et al. 1987 used
                                     ``dy+ < 1``, ``dx+ ~ 6``, ``dz+ ~ 3``)
``D1`` mean profile vs MKM-1999      buffer ``5 < y+ < 30`` rms ``<= 5 %``,
                                     log ``30 < y+ <= 0.3 Re_tau`` rms ``<= 3 %``
``D2`` Reynolds stresses             ``u_rms`` and ``-<u'v'>`` peaks within
                                     ``10 %`` of MKM **and** the anisotropy
                                     ordering ``u > w > v`` on ``>= 90 %`` rows
``D3`` wall law                      fitted ``kappa in [0.36, 0.45]``,
                                     ``B in [4.2, 6.2]``, rms residual ``<= 0.5``
``D4`` TKE spectrum                  ``-5/3`` slope within ``+-0.20`` with
                                     ``r2 >= 0.85`` on a log-layer plane, using
                                     the a-priori band of
                                     :func:`tensorlbm.turbulence.channel_spectra.fit_loglog_slope`
``D5`` friction coefficient          ``C_f = 2/U_b+^2`` vs MKM ``<= 5 %`` and vs
                                     Dean (1978) ``0.073 Re_b^-1/4`` ``<= 10 %``
``D6`` statistical convergence       batch-mean bulk velocity spread ``<= 2 %``,
                                     three-route ``u_tau`` spread ``<= 3 %``

Missing inputs give ``WITHHELD`` (never ``pass``), and a *study* is only
reported as ``DNS_VERIFIED`` when ``D0`` and ``D1..D6`` all pass.  A case that
is physically under-resolved is therefore reported as ``UNDER_RESOLVED`` with
the measured ``dy+``, not quietly graded as a DNS success -- this is the honest
FAIL path the node requires.

Empirical reference: MOSER, KIM & MANSOUR (1999) DNS tables vendored in
``tensorlbm/data/mkm1999`` and SHA-256 verified by
:mod:`tensorlbm.validation.mkm1999`; Dean's correlation from
:func:`tensorlbm.validation.wall_law.dean_channel_cf`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..validation import mkm1999
from ..validation.wall_law import dean_channel_cf
from .channel_spectra import (
    load_plane_dump,
    plane_spectra_legs,
    read_plane_meta,
)
from .les_channel import compare_to_mkm, wall_units

__all__ = [
    "DNS_TARGETS",
    "MINUS_FIVE_THIRDS",
    "cf_targets",
    "dns_cells_required",
    "evaluate_dns_gates",
    "load_dns_case",
    "momentum_balance",
    "turbulence_level",
    "resolution_metrics",
    "spectrum_slope_gate",
    "study_status",
]

#: Resolution thresholds that separate a DNS from a wall-resolved LES.
#: ``dy_plus`` follows Kim et al. (1987) / Moin & Mahesh (1998): the first
#: off-wall node must sit inside the viscous sublayer (``y+ < 2``), so a uniform
#: grid with ``dy+ <= 1.5`` is the operational target.
DNS_TARGETS = {
    "dy_plus": 1.5,
    "dx_plus": 10.0,
    "dz_plus": 5.0,
}

#: Kolmogorov inertial-range slope used as the reference for ``D4``.
MINUS_FIVE_THIRDS = -5.0 / 3.0


# ---------------------------------------------------------------------------
# resolution attestation
# ---------------------------------------------------------------------------
def resolution_metrics(
    *, re_tau: float, u_tau: float, ny: int, nx: int, nz: int, nu: float | None = None
) -> dict:
    """Wall-unit grid spacing, domain extent and resolution grade of a case.

    Lattice spacing is 1 in every direction, so ``dy+ = dx+ = dz+`` *in lattice
    units* unless the caller rescales; the three numbers are reported separately
    because they are compared against different thresholds.
    """
    if ny < 4:
        raise ValueError("ny must be >= 4")
    h = 0.5 * (ny - 2)
    if nu is None:
        if u_tau <= 0.0 or re_tau <= 0.0 or h <= 0.0:
            raise ValueError("u_tau, re_tau and h must be positive")
        nu = u_tau * h / re_tau
    spacing_plus = u_tau / nu
    metrics = {
        "re_tau": float(re_tau),
        "u_tau": float(u_tau),
        "nu": float(nu),
        "half_height": float(h),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "cells": int(nx * ny * nz),
        "dy_plus": spacing_plus,
        "dx_plus": spacing_plus,
        "dz_plus": spacing_plus,
        "lx_over_h": nx / h,
        "lz_over_h": nz / h,
        "first_node_y_plus": 0.5 * spacing_plus,
        "targets": dict(DNS_TARGETS),
    }
    ok = (
        metrics["dy_plus"] <= DNS_TARGETS["dy_plus"]
        and metrics["dx_plus"] <= DNS_TARGETS["dx_plus"]
        and metrics["dz_plus"] <= DNS_TARGETS["dz_plus"]
    )
    metrics["dns_resolved"] = bool(ok)
    metrics["grade"] = (
        "dns"
        if ok
        else "quasi_dns"
        if metrics["dy_plus"] <= 2.5 and metrics["dx_plus"] <= 15.0
        else "under_resolved"
    )
    return metrics


def dns_cells_required(
    re_tau: float,
    *,
    dy_plus: float = 1.0,
    lx_over_h: float = 2.0 * np.pi,
    lz_over_h: float = np.pi,
) -> dict:
    """Grid a textbook DNS box would need at this ``Re_tau`` (cost projection).

    ``h = Re_tau / dy_plus`` sets the wall-normal count; the box is
    ``Lx = lx_over_h * h``, ``Lz = lz_over_h * h`` with unit lattice spacing.
    Reported so the gap between what the solver *can* do inside a job budget and
    what DNS asks for is a number, not a claim.
    """
    if re_tau <= 0.0 or dy_plus <= 0.0:
        raise ValueError("re_tau and dy_plus must be positive")
    h = re_tau / dy_plus
    nx = int(round(lx_over_h * h))
    ny = int(round(2.0 * h)) + 2
    nz = int(round(lz_over_h * h))
    return {
        "dy_plus": float(dy_plus),
        "half_height": float(h),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "cells": int(nx * ny * nz),
        "lx_over_h": float(lx_over_h),
        "lz_over_h": float(lz_over_h),
    }


# ---------------------------------------------------------------------------
# friction coefficient routes
# ---------------------------------------------------------------------------
def cf_targets(profile: Mapping[str, Any], mkm: mkm1999.MKMProfile | None) -> dict:
    """``C_f`` from the simulated bulk velocity vs MKM and Dean (1978).

    ``C_f = 2 tau_w / (rho U_b^2) = 2 / U_b+^2``; ``Re_b = U_b * 2h / nu`` uses
    the hydraulic diameter of the plane channel (``2h``), which is the
    definition Dean's correlation is quoted with in the channel literature.
    """
    u_bulk_plus = float(profile["u_bulk_plus"])
    cf = float(profile["cf"])
    re_tau_effective = float(profile["re_tau_effective"])
    re_bulk = 2.0 * re_tau_effective * u_bulk_plus
    cf_dean = dean_channel_cf(re_bulk)
    out = {
        "cf": cf,
        "bulk_velocity_plus": u_bulk_plus,
        "bulk_velocity": float(profile["u_bulk"]),
        "re_bulk": re_bulk,
        "cf_dean": cf_dean,
        "dean_relative_pct": 100.0 * (cf - cf_dean) / cf_dean,
        "dean_correlation_range": [1.0e4, 1.0e5],
        "dean_in_validity_range": bool(1.0e4 <= re_bulk <= 1.0e5),
        "reference": "Dean (1978) C_f = 0.073 Re_b^-0.25",
    }
    if mkm is not None:
        cf_mkm = mkm.friction_coefficient()
        out["cf_mkm"] = cf_mkm
        out["mkm_relative_pct"] = 100.0 * (cf - cf_mkm) / cf_mkm
        out["re_bulk_mkm"] = 2.0 * float(mkm.re_tau) * mkm.bulk_velocity_plus()
    return out


# ---------------------------------------------------------------------------
# Reynolds-stress peaks (u_rms peak is in compare_to_mkm; v/w/ordering here)
# ---------------------------------------------------------------------------
def stress_peaks(profile: Mapping[str, Any], mkm: mkm1999.MKMProfile | None) -> dict:
    """Peak magnitudes of the three normal stresses and the shear stress."""
    out: dict[str, Any] = {
        "computed": {
            "u_rms": profile["peaks"]["u_rms"],
            "v_rms": profile["peaks"]["v_rms"],
            "w_rms": profile["peaks"]["w_rms"],
            "minus_uv": profile["peaks"]["minus_uv"],
        },
        "anisotropy": dict(profile.get("anisotropy", {})),
    }
    if mkm is None:
        out["available"] = False
        return out
    out["available"] = True
    out["reference"] = "MKM-1999"
    for key in ("u_rms", "v_rms", "w_rms", "minus_uv"):
        ref = mkm.peak(key)
        got = profile["peaks"][key]
        out[key] = {
            "reference_y_plus": ref["y_plus"],
            "reference_value": ref["value"],
            "computed_y_plus": got["y_plus"],
            "computed_value": got["value"],
            "value_relative_pct": 100.0
            * (got["value"] - ref["value"])
            / ref["value"],
            "y_plus_error": got["y_plus"] - ref["y_plus"],
        }
    return out


# ---------------------------------------------------------------------------
# TKE spectrum gate
# ---------------------------------------------------------------------------
def spectrum_slope_gate(
    spectra: Mapping[str, Any] | None,
    *,
    re_tau: float,
    slope_tol: float = 0.20,
    r2_min: float = 0.85,
    y_plus_band: tuple[float, float] = (20.0, 0.5),
) -> dict:
    """Best log-layer streamwise spectrum leg vs the ``-5/3`` law.

    Only legs inside the log layer (``y+ >= 20`` and ``y+ <= 0.5 Re_tau``) are
    eligible -- the inertial range is a property of that region -- and among
    them the leg with slope closest to ``-5/3`` is reported, together with all
    eligible legs.  The fit band itself is fixed a priori by
    :func:`channel_spectra.fit_loglog_slope`, so nothing here can tune the slope.
    """
    if not spectra:
        return {"valid": False, "reason": "no spectrum block in the artefact"}
    lo, frac = y_plus_band
    hi = frac * re_tau
    candidates = []
    for leg in spectra.get("streamwise", []):
        y_plus = float(leg.get("y_plus", float("nan")))
        fit = leg.get("fit", {})
        if not fit.get("valid") or not (lo <= y_plus <= hi):
            continue
        candidates.append(
            {
                "y_plus": y_plus,
                "slope": float(fit["slope"]),
                "r2": float(fit["r2"]),
                "k_min": float(fit.get("k_min", float("nan"))),
                "k_max": float(fit.get("k_max", float("nan"))),
                "n_points": int(fit.get("n_points", 0)),
            }
        )
    if not candidates:
        return {
            "valid": False,
            "reason": "no eligible fit (need a dumped plane inside the log layer)",
            "samples": spectra.get("samples"),
        }
    best = min(candidates, key=lambda c: abs(c["slope"] - MINUS_FIVE_THIRDS))
    best = dict(best)
    best.update(
        {
            "valid": True,
            "target_slope": MINUS_FIVE_THIRDS,
            "slope_error": best["slope"] - MINUS_FIVE_THIRDS,
            "slope_tol": slope_tol,
            "r2_min": r2_min,
            "passed": bool(
                abs(best["slope"] - MINUS_FIVE_THIRDS) <= slope_tol
                and best["r2"] >= r2_min
            ),
            "samples": spectra.get("samples"),
            "legs": candidates,
        }
    )
    return best


# ---------------------------------------------------------------------------
# exact momentum-balance identity of the fully developed channel
# ---------------------------------------------------------------------------
def momentum_balance(profile: Mapping[str, Any]) -> dict:
    """Residual of ``nu dU/dy - <u'v'> = u_tau^2 (1 - y/h)`` in wall units.

    The identity is exact for a statistically stationary plane channel driven by
    a constant body force, so it is a *physics* check that does not depend on the
    MKM tables: it verifies that the resolved shear plus the viscous term carry
    the imposed wall stress all the way across the channel.  Reported as the
    wall-unit residual ``T+ - (1 - y+/Re_tau)`` over the inner 80 % of the half
    channel (the outer 20 % is where the finite-difference derivative of a coarse
    profile is least accurate).
    """
    y = np.asarray(profile["y"], dtype=float)
    u_plus = np.asarray(profile["u_plus"], dtype=float)
    uv_plus = np.asarray(profile["uv_plus"], dtype=float)
    y_plus = np.asarray(profile["y_plus"], dtype=float)
    u_tau = float(profile["u_tau"]["momentum_balance"])
    nu = float(profile["nu"])
    re_tau = float(profile["re_tau_effective"])
    if y.size < 4:
        return {"valid": False, "reason": "profile too short"}
    du_dy = np.gradient(u_plus * u_tau, y)
    viscous_plus = nu * du_dy / u_tau**2
    total_plus = viscous_plus - uv_plus
    target = 1.0 - y_plus / re_tau
    mask = (y_plus >= 2.0) & (y_plus <= 0.8 * re_tau)
    if int(mask.sum()) < 3:
        return {"valid": False, "reason": "fewer than 3 rows inside the inner layer"}
    residual = total_plus[mask] - target[mask]
    return {
        "valid": True,
        "y_plus": y_plus[mask].tolist(),
        "total_stress_plus": total_plus[mask].tolist(),
        "target_plus": target[mask].tolist(),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "relative_rms_pct": float(
            100.0 * np.sqrt(np.mean(residual**2)) / max(float(np.mean(target[mask])), 1e-12)
        ),
        "n_points": int(mask.sum()),
        "identity": "nu dU/dy - <u'v'> = u_tau^2 (1 - y/h)",
    }


def turbulence_level(profile: Mapping[str, Any], mkm: mkm1999.MKMProfile | None) -> dict:
    """Is the sampled state turbulent at all?  (fraction of the DNS fluctuation)

    A channel run can be *numerically fine and physically useless*: the initial
    perturbation may decay and the flow relaminarises before the averaging window
    opens.  Comparing such a profile against DNS then measures nothing but the
    initial condition, so the ratio of the simulated ``u_rms`` peak to the DNS
    peak is recorded explicitly and a state below half the DNS level is flagged
    as ``laminarised``.  This is a diagnostic, not a pass/fail gate: the DNS
    comparison gates already fail, but the flag says *why*.
    """
    peaks = profile.get("peaks", {})
    u_rms = float(peaks.get("u_rms", {}).get("value", float("nan")))
    uv = float(peaks.get("minus_uv", {}).get("value", float("nan")))
    if mkm is None:
        return {"available": False, "u_rms_peak_plus": u_rms, "minus_uv_peak_plus": uv}
    ref_u = float(mkm.peak("u_rms")["value"])
    ref_uv = float(mkm.peak("minus_uv")["value"])
    frac = u_rms / ref_u if ref_u > 0 else float("nan")
    frac_uv = uv / ref_uv if ref_uv > 0 else float("nan")
    return {
        "available": True,
        "u_rms_peak_plus": u_rms,
        "u_rms_peak_reference": ref_u,
        "u_rms_fraction_of_dns": frac,
        "minus_uv_peak_plus": uv,
        "minus_uv_peak_reference": ref_uv,
        "minus_uv_fraction_of_dns": frac_uv,
        "laminarised": bool(frac < 0.5 or frac_uv < 0.5),
        "criterion": "flagged when either fluctuation peak is below 50% of MKM-1999",
    }


# ---------------------------------------------------------------------------
# statistical convergence from the kernel's 4 batch means
# ---------------------------------------------------------------------------
def batch_convergence(result: Mapping[str, Any]) -> dict:
    """Block-mean (4 sub-window) spread of the bulk velocity and ``u_tau``.

    The kernel stores four consecutive batch means; their spread is a direct
    (conservative) estimate of the statistical error of the sampling window, so
    a case whose mean profile has not settled is flagged instead of being
    compared against DNS.
    """
    batches = result.get("batch_means") or []
    rows = [
        b for b in batches if b.get("samples") and isinstance(b.get("mean_ux"), list)
    ]
    if len(rows) < 2:
        return {"valid": False, "reason": "fewer than two non-empty batch means"}
    bulks = []
    for b in rows:
        prof = np.asarray(b["mean_ux"], dtype=float)
        if prof.size < 2 or not np.all(np.isfinite(prof)):
            return {"valid": False, "reason": "non-finite batch mean profile"}
        bulks.append(float(np.mean(prof)))
    bulks_arr = np.asarray(bulks)
    mean = float(bulks_arr.mean())
    spread = float((bulks_arr.max() - bulks_arr.min()) / mean * 100.0) if mean else float("nan")
    return {
        "valid": True,
        "bulk_velocity_batches": bulks_arr.tolist(),
        "bulk_velocity_mean": mean,
        "bulk_velocity_spread_pct": spread,
        "n_batches": len(bulks),
    }


# ---------------------------------------------------------------------------
# artefact loading
# ---------------------------------------------------------------------------
def _load_json_lenient(path: Path) -> tuple[dict, bool]:
    """Load a kernel artefact, tolerating lowercase ``nan``/``inf`` from C.

    A diverged run must remain *readable* so it can be reported as an honest
    FAIL; the C writer emits ``nan`` which is not valid JSON, so those tokens are
    replaced by ``null`` and the substitution is recorded.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        import re

        cleaned = re.sub(r"(?<![\w.])-?(nan|inf)(?![\w.])", "null", text)
        return json.loads(cleaned), True


def load_dns_case(
    run_dir: str | Path,
    name: str,
    *,
    re_tau_anchor: int,
    k_min: float = 4.0,
) -> dict | None:
    """Reduce one native run (+ optional spectrum dump) into a case record.

    Returns ``None`` when the artefact is missing or the run diverged (a
    ``None`` mean profile), so the gate layer records ``withheld`` rather than
    inventing numbers.
    """
    run_dir = Path(run_dir)
    path = run_dir / f"{name}.json"
    if not path.exists():
        return None
    result, sanitized = _load_json_lenient(path)
    probe = result.get("profiles", {}).get("mean_ux", [])
    if not probe or any(v is None or v != v for v in probe[:4]):
        config = result.get("configuration", {})
        derived = result.get("derived", {})
        resolution = None
        try:
            resolution = resolution_metrics(
                re_tau=float(config.get("re_tau", 0.0)),
                u_tau=float(config.get("u_tau", 0.0)),
                ny=int(config.get("ny", 0)),
                nx=int(config.get("nx", 0)),
                nz=int(config.get("nz", 0)),
                nu=float(derived["nu"]) if derived.get("nu") else None,
            )
        except (TypeError, ValueError, KeyError):
            resolution = None
        return {
            "name": name,
            "diverged": True,
            "path": str(path),
            "sanitized": sanitized,
            "anchor": int(re_tau_anchor),
            "resolution": resolution,
            "cs": config.get("cs"),
            "regularized": config.get("regularized"),
            "u_tau": config.get("u_tau"),
            "steps": config.get("steps"),
            "max_speed": result.get("diagnostics", {}).get("max_speed"),
            "samples": int(derived.get("samples", 0) or 0),
        }
    profile = wall_units(result)
    mkm = mkm1999.load_profile(re_tau_anchor)
    comparison = compare_to_mkm(profile, mkm)
    derived = result.get("derived", {})
    config = result.get("configuration", {})
    resolution = resolution_metrics(
        re_tau=float(config.get("re_tau", mkm.re_tau)),
        u_tau=float(config.get("u_tau", profile["u_tau"]["momentum_balance"])),
        ny=int(config.get("ny", 0)),
        nx=int(config.get("nx", 0)),
        nz=int(config.get("nz", 0)),
        nu=float(derived.get("nu", float("nan"))),
    )
    spectra = None
    spec_bin = run_dir / f"{name}_spec.bin"
    spec_meta = run_dir / f"{name}_spec.json"
    if spec_bin.exists() and spec_meta.exists():
        meta = read_plane_meta(spec_meta)
        dump = load_plane_dump(spec_bin, meta)
        spectra = plane_spectra_legs(
            dump,
            u_tau=float(profile["u_tau"]["momentum_balance"]),
            nu=float(derived["nu"]),
            k_min=k_min,
        )
    return {
        "name": name,
        "diverged": False,
        "path": str(path),
        "sanitized": sanitized,
        "anchor": int(re_tau_anchor),
        "result": result,
        "profile": profile,
        "comparison": comparison,
        "resolution": resolution,
        "spectra": spectra,
        "cf": cf_targets(profile, mkm),
        "stress": stress_peaks(profile, mkm),
        "convergence": batch_convergence(result),
        "momentum_balance": momentum_balance(profile),
        "turbulence": turbulence_level(profile, mkm),
        "eddy_turnovers": float(derived.get("statistics_eddy_turnovers", float("nan"))),
        "cs": config.get("cs"),
        "regularized": config.get("regularized"),
        "samples": int(derived.get("samples", 0)),
    }


# ---------------------------------------------------------------------------
# gate assembly
# ---------------------------------------------------------------------------
def _gate(name: str, criterion: str, passed: bool | None, evidence: dict, problems=None) -> dict:
    return {
        "gate": name,
        "criterion": criterion,
        "passed": passed,
        "status": "pass" if passed is True else ("fail" if passed is False else "withheld"),
        "evidence": evidence,
        "problems": list(problems or []),
    }


def _gates_for_case(case: dict, *, kappa_band=(0.36, 0.45), b_band=(4.2, 6.2)) -> list[dict]:
    name = case["name"]
    if case.get("diverged"):
        return [
            _gate(f"{name}.{g}", "not evaluable", None, {"diverged": True},
                  ["run diverged (non-finite mean profile)"])
            for g in ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7")
        ]
    res = case["resolution"]
    gates: list[dict] = []

    # D0 resolution attestation
    d0_problems = []
    for key, target in (
        ("dy_plus", DNS_TARGETS["dy_plus"]),
        ("dx_plus", DNS_TARGETS["dx_plus"]),
        ("dz_plus", DNS_TARGETS["dz_plus"]),
    ):
        if res[key] > target:
            d0_problems.append(f"{key}={res[key]:.2f} > {target}")
    gates.append(
        _gate(
            f"{name}.D0_resolution",
            "dy+ <= 1.5, dx+ <= 10, dz+ <= 5 (DNS-grade wall-normal spacing)",
            res["dns_resolved"],
            res,
            d0_problems,
        )
    )

    comp = case["comparison"]
    buf = comp.get("buffer", {})
    log = comp.get("log", {})
    d1_ok = (
        bool(buf.get("ok"))
        and bool(log.get("ok"))
        and buf["rms_relative_pct"] <= 5.0
        and log["rms_relative_pct"] <= 3.0
    )
    d1_problems = []
    if buf.get("ok") and buf["rms_relative_pct"] > 5.0:
        d1_problems.append(f"buffer rms {buf['rms_relative_pct']:.2f}% > 5%")
    if log.get("ok") and log["rms_relative_pct"] > 3.0:
        d1_problems.append(f"log-layer rms {log['rms_relative_pct']:.2f}% > 3%")
    gates.append(
        _gate(
            f"{name}.D1_u_plus_vs_mkm",
            "buffer(5<y+<30) rms <= 5%, log(30<y+<0.3Re) rms <= 3%",
            d1_ok if (buf.get("ok") and log.get("ok")) else None,
            {"buffer": buf, "log": log},
            d1_problems,
        )
    )

    stress = case["stress"]
    if stress.get("available"):
        uu = stress["u_rms"]
        uv = stress["minus_uv"]
        ani = stress.get("anisotropy", {})
        stations = int(ani.get("stations", 0))
        ordering = int(ani.get("ordering_u_w_v", 0))
        frac = ordering / stations if stations else 0.0
        d2_ok = (
            abs(uu["value_relative_pct"]) <= 10.0
            and abs(uv["value_relative_pct"]) <= 10.0
            and frac >= 0.90
        )
        d2_problems = []
        if abs(uu["value_relative_pct"]) > 10.0:
            d2_problems.append(f"u_rms peak error {uu['value_relative_pct']:.1f}% > 10%")
        if abs(uv["value_relative_pct"]) > 10.0:
            d2_problems.append(f"-<u'v'> peak error {uv['value_relative_pct']:.1f}% > 10%")
        if frac < 0.90:
            d2_problems.append(f"u>w>v ordering on only {frac * 100:.0f}% of rows")
        gates.append(
            _gate(
                f"{name}.D2_reynolds_stress",
                "u_rms and -<u'v'> peaks within 10% of MKM; u>w>v on >= 90% rows",
                d2_ok,
                {"u_rms": uu, "minus_uv": uv, "ordering_fraction": frac,
                 "v_rms": stress["v_rms"], "w_rms": stress["w_rms"]},
                d2_problems,
            )
        )
    else:
        gates.append(_gate(f"{name}.D2_reynolds_stress", "vs MKM-1999", None, {},
                           ["no MKM reference for this Re_tau"]))

    law = case["profile"].get("wall_law", {})
    if law.get("ok"):
        kappa = float(law["kappa"])
        b = float(law["B"])
        rms = float(law["rms_residual"])
        d3_ok = (
            kappa_band[0] <= kappa <= kappa_band[1]
            and b_band[0] <= b <= b_band[1]
            and rms <= 0.5
        )
        d3_problems = []
        if not kappa_band[0] <= kappa <= kappa_band[1]:
            d3_problems.append(f"kappa={kappa:.3f} outside {kappa_band}")
        if not b_band[0] <= b <= b_band[1]:
            d3_problems.append(f"B={b:.3f} outside {b_band}")
        if rms > 0.5:
            d3_problems.append(f"log-law rms residual {rms:.3f} > 0.5")
        gates.append(
            _gate(
                f"{name}.D3_wall_law",
                "kappa in [0.36,0.45], B in [4.2,6.2], rms residual <= 0.5",
                d3_ok,
                {"kappa": kappa, "B": b, "rms_residual": rms,
                 "window": law.get("window"), "n_points": law.get("n_points")},
                d3_problems,
            )
        )
    else:
        gates.append(_gate(f"{name}.D3_wall_law", "log-law fit", None,
                           {"wall_law": law}, ["no log-layer nodes to fit"]))

    spec = spectrum_slope_gate(
        case.get("spectra"), re_tau=float(case["profile"]["re_tau_effective"])
    )
    gates.append(
        _gate(
            f"{name}.D4_tke_spectrum",
            "streamwise E(k) slope = -5/3 +-0.20 with r2 >= 0.85 on a log-layer plane",
            spec.get("passed") if spec.get("valid") else None,
            spec,
            [] if spec.get("passed") else [spec.get("reason", "slope outside band")],
        )
    )

    cf = case["cf"]
    d5_ok = abs(cf["mkm_relative_pct"]) <= 5.0 and abs(cf["dean_relative_pct"]) <= 10.0
    d5_problems = []
    if abs(cf["mkm_relative_pct"]) > 5.0:
        d5_problems.append(f"C_f vs MKM {cf['mkm_relative_pct']:.2f}% > 5%")
    if abs(cf["dean_relative_pct"]) > 10.0:
        d5_problems.append(f"C_f vs Dean {cf['dean_relative_pct']:.2f}% > 10%")
    gates.append(
        _gate(
            f"{name}.D5_friction_coefficient",
            "C_f = 2/U_b+^2 vs MKM <= 5%, vs Dean(1978) <= 10%",
            d5_ok,
            cf,
            d5_problems,
        )
    )

    conv = case["convergence"]
    spread_ut = case["profile"].get("max_pairwise_u_tau_deviation_pct")
    if conv.get("valid") and spread_ut is not None:
        d6_ok = conv["bulk_velocity_spread_pct"] <= 2.0 and spread_ut <= 3.0
        d6_problems = []
        if conv["bulk_velocity_spread_pct"] > 2.0:
            d6_problems.append(
                f"batch bulk spread {conv['bulk_velocity_spread_pct']:.2f}% > 2%"
            )
        if spread_ut > 3.0:
            d6_problems.append(f"u_tau route spread {spread_ut:.2f}% > 3%")
        gates.append(
            _gate(
                f"{name}.D6_statistical_convergence",
                "batch-mean bulk spread <= 2%, u_tau route spread <= 3%",
                d6_ok,
                {"batch": conv, "u_tau_route_spread_pct": spread_ut,
                 "eddy_turnovers": case["eddy_turnovers"]},
                d6_problems,
            )
        )
    else:
        gates.append(
            _gate(f"{name}.D6_statistical_convergence", "batch means", None,
                  {"batch": conv, "u_tau_route_spread_pct": spread_ut},
                  ["batch means unavailable"])
        )
    mb = case.get("momentum_balance") or {}
    if mb.get("valid"):
        ok = mb["relative_rms_pct"] <= 3.0
        gates.append(
            _gate(
                f"{name}.D7_momentum_balance",
                "rms of (nu dU/dy - <u'v'> + y/h - 1) <= 3% of u_tau^2 over y+ in [2, 0.8Re]",
                ok,
                mb,
                [] if ok else [f"momentum-balance rms {mb['relative_rms_pct']:.2f}% > 3%"],
            )
        )
    else:
        gates.append(
            _gate(f"{name}.D7_momentum_balance", "channel stress identity", None, mb,
                  [mb.get("reason", "not computable")])
        )
    return gates


def evaluate_dns_gates(cases: Mapping[str, dict | None], *, require: tuple[str, ...] = ()) -> dict:
    """Evaluate ``D0..D6`` for every case and roll them up per Reynolds number.

    ``cases`` maps a case name to the record from :func:`load_dns_case` (or
    ``None`` when the artefact is missing -> all gates ``withheld``).
    ``require`` lists case names that must be present for the study to be
    graded; a missing required case makes the study ``INCOMPLETE``.
    """
    gates: list[dict] = []
    per_case: dict[str, dict] = {}
    for name, case in cases.items():
        if case is None:
            name_gates = [
                _gate(f"{name}.D0_resolution", "missing artefact", None, {},
                      ["run artefact not found"])
            ]
            gates.extend(name_gates)
            per_case[name] = {"status": "WITHHELD", "gates": name_gates}
            continue
        name_gates = _gates_for_case(case)
        gates.extend(name_gates)
        statuses = [g["status"] for g in name_gates]
        if any(s == "fail" for s in statuses):
            status = "FAIL"
        elif all(s == "pass" for s in statuses):
            status = "PASS"
        else:
            status = "WITHHELD"
        per_case[name] = {
            "status": status,
            "anchor": case.get("anchor"),
            "dns_resolved": None
            if case.get("diverged")
            else case["resolution"]["dns_resolved"],
            "grade": None if case.get("diverged") else case["resolution"]["grade"],
            "failed_gates": [g["gate"] for g in name_gates if g["status"] == "fail"],
            "withheld_gates": [g["gate"] for g in name_gates if g["status"] == "withheld"],
            "gates": name_gates,
        }
    missing = [n for n in require if cases.get(n) is None]
    passed = [n for n, rec in per_case.items() if rec["status"] == "PASS"]
    failed = [n for n, rec in per_case.items() if rec["status"] == "FAIL"]
    withheld = [n for n, rec in per_case.items() if rec["status"] == "WITHHELD"]
    if missing:
        study = "INCOMPLETE"
    elif failed:
        study = "FAIL"
    elif withheld:
        study = "WITHHELD"
    else:
        study = "PASS"
    return {
        "study_status": study,
        "cases": per_case,
        "missing_required": missing,
        "passed_cases": passed,
        "failed_cases": failed,
        "withheld_cases": withheld,
        "gates": gates,
        "criteria": {
            "D0": "dy+ <= 1.5, dx+ <= 10, dz+ <= 5",
            "D1": "U+ vs MKM: buffer rms <= 5%, log rms <= 3%",
            "D2": "u_rms / -<u'v'> peaks within 10% of MKM; u>w>v on >= 90% rows",
            "D3": "log law: kappa in [0.36,0.45], B in [4.2,6.2], rms residual <= 0.5",
            "D4": "E(k) slope = -5/3 +-0.20, r2 >= 0.85 on a log-layer plane",
            "D5": "C_f vs MKM <= 5%, vs Dean(1978) <= 10%",
            "D6": "batch bulk spread <= 2%, u_tau route spread <= 3%",
            "D7": "exact channel momentum balance residual <= 3% of u_tau^2",
        },
        "reference_data": {
            "mean_and_stress": "Moser, Kim & Mansour (1999) DNS, SHA-256 verified "
            "in tensorlbm/data/mkm1999",
            "friction": "Dean (1978) C_f = 0.073 Re_b^-0.25",
        },
    }


def study_status(gates: Mapping[str, Any], *, case: str) -> dict:
    """One-line verdict for a single Reynolds-number leg of the study."""
    rec = gates["cases"].get(case)
    if rec is None:
        return {"case": case, "status": "WITHHELD", "reason": "case not in gate set"}
    return {
        "case": case,
        "status": rec["status"],
        "dns_resolved": rec["dns_resolved"],
        "grade": rec["grade"],
        "failed_gates": rec["failed_gates"],
        "withheld_gates": rec["withheld_gates"],
    }