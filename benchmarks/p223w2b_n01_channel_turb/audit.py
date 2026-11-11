"""Gate-feasibility audit for the p223w2b-n01 turbulent-channel validation surface.

The node asks for five machine criteria ((a) log-law band < 5 %, (b) Reynolds
stress peak < 10 %, (c) inertial-range slope vs -5/3 < 10 %, (d) friction
coefficient vs the Dean correlation < 5 %, (e) grid convergence of ``u_tau``
< 3 %).  Before any of them is applied to a simulation, each is applied to the
MKM-1999 DNS reference itself: a criterion that the *reference* cannot satisfy
is a mis-specified criterion, not a solver failure.

This module is pure reduction code -- it touches no solver and needs no compute.
It is exercised by ``tests/test_p223w2b_n01_channel_turb.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorlbm.validation.mkm1999 import (
    MKMProfile,
    available_cases,
    load_profile,
    load_provenance,
)
from tensorlbm.validation.wall_law import dean_channel_cf

__all__ = [
    "KAPPA",
    "B_LOG",
    "log_law_error",
    "viscous_sublayer_error",
    "cf_audit",
    "stress_peaks",
    "spectral_slope",
    "run_audit",
    "REFERENCE_JSON",
]

KAPPA = 0.41
B_LOG = 5.2

#: y+ window of the log layer used by criterion (a)
LOG_BAND = (30.0, 100.0)


def log_law_error(
    y_plus: np.ndarray,
    u_plus: np.ndarray,
    kappa: float = KAPPA,
    b: float = B_LOG,
    band: tuple[float, float] = LOG_BAND,
) -> dict:
    """Relative error of ``U+`` against ``(1/kappa) ln(y+) + b`` in a y+ band.

    Two numbers are returned because they answer different questions:
    ``max_rel`` is the worst station (what criterion (a) literally asks for) and
    ``rms_rel`` is the band-averaged model-form misfit (what a profile-level
    comparison can actually resolve).
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    lo, hi = band
    mask = (y_plus >= lo) & (y_plus <= hi)
    if not np.any(mask):
        return {"band": [lo, hi], "points": 0, "max_rel": None, "rms_rel": None}
    law = np.log(y_plus[mask]) / kappa + b
    rel = (u_plus[mask] - law) / law
    return {
        "band": [lo, hi],
        "points": int(mask.sum()),
        "max_rel": float(np.max(np.abs(rel))),
        "signed_mean_rel": float(np.mean(rel)),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "y_plus_max_rel": float(y_plus[mask][int(np.argmax(np.abs(rel)))]),
    }


def viscous_sublayer_error(
    y_plus: np.ndarray, u_plus: np.ndarray, limit: float = 5.0
) -> dict:
    """``U+ = y+`` in the viscous sublayer: criterion (a)'s second half."""
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus > 0.0) & (y_plus <= limit)
    if not np.any(mask):
        return {"limit": limit, "points": 0, "max_abs": None}
    diff = u_plus[mask] - y_plus[mask]
    return {
        "limit": limit,
        "points": int(mask.sum()),
        "max_abs": float(np.max(np.abs(diff))),
        "rms_abs": float(np.sqrt(np.mean(diff**2))),
    }


def cf_audit(profile: MKMProfile) -> dict:
    """Friction coefficient: reference value, the Dean correlation, and the fit.

    ``Re_b = U_b 2h / nu = Re_tau * U_b+`` and ``C_f = 2 tau_w / (rho U_b^2)
    = 2 / U_b+^2``; both conventions are stated explicitly because the Dean
    correlation is extremely sensitive to which one is used.
    """
    ub_plus = profile.bulk_velocity_plus()
    cf_dns = profile.friction_coefficient()
    re_b = profile.re_tau * ub_plus
    cf_dean = float(dean_channel_cf(re_b))
    return {
        "re_tau": profile.re_tau,
        "bulk_plus": ub_plus,
        "re_bulk": re_b,
        "cf_dns": cf_dns,
        "cf_dean": cf_dean,
        "rel_dev_dean": (cf_dean - cf_dns) / cf_dns,
    }


def stress_peaks(profile: MKMProfile) -> dict:
    """Reference Reynolds-stress peaks (targets for criterion (b))."""
    out = {}
    for name in ("u_rms", "v_rms", "w_rms", "minus_uv"):
        out[name] = profile.peak(name)
    out["uu_peak"] = {
        "quantity": "uu",
        "y_plus": out["u_rms"]["y_plus"],
        "value": out["u_rms"]["value"] ** 2,
    }
    return out


def spectral_slope(
    k: np.ndarray,
    e: np.ndarray,
    k_lo: float = 0.0,
    k_hi: float = math.inf,
    n_points: int = 8,
) -> dict:
    """Least-squares slope of ``ln E`` vs ``ln k`` (criterion (c) reduction)."""
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)
    mask = (k > k_lo) & (k <= k_hi) & (e > 0.0)
    if mask.sum() < 3:
        return {"points": int(mask.sum()), "slope": None, "rel_err_vs_minus_5_3": None}
    kk, ee = k[mask], e[mask]
    if kk.size > n_points:                       # keep the fit band honest
        idx = np.linspace(0, kk.size - 1, n_points).astype(int)
        kk, ee = kk[idx], ee[idx]
    slope = float(np.polyfit(np.log(kk), np.log(ee), 1)[0])
    return {
        "points": int(kk.size),
        "k_range": [float(kk.min()), float(kk.max())],
        "slope": slope,
        "rel_err_vs_minus_5_3": (slope + 5.0 / 3.0) / (5.0 / 3.0),
    }


@dataclass
class AuditCase:
    nominal_re_tau: int
    re_tau: float
    source_files: tuple[str, str]
    ub_plus: float
    cf_dns: float
    cf_dean: float
    cf_dean_rel_dev: float
    log_law: dict
    log_law_wide: dict
    viscous_sublayer: dict
    stress_peaks: dict


def audit_case(nominal: int) -> AuditCase:
    p = load_profile(nominal)
    cf = cf_audit(p)
    return AuditCase(
        nominal_re_tau=nominal,
        re_tau=p.re_tau,
        source_files=p.source_files,
        ub_plus=cf["bulk_plus"],
        cf_dns=cf["cf_dns"],
        cf_dean=cf["cf_dean"],
        cf_dean_rel_dev=cf["rel_dev_dean"],
        log_law=log_law_error(p.y_plus, p.u_plus, band=(30.0, 100.0)),
        log_law_wide=log_law_error(
            p.y_plus, p.u_plus, band=(30.0, min(0.15 * p.re_tau, 100.0))
        ),
        viscous_sublayer=viscous_sublayer_error(p.y_plus, p.u_plus),
        stress_peaks=stress_peaks(p),
    )


def _power_law_fit(cases: list[AuditCase]) -> dict:
    """``C_f = A Re_b^{-n}`` through the DNS points (the honest low-Re reference)."""
    re_b = np.array([c.re_tau * c.ub_plus for c in cases])
    cf = np.array([c.cf_dns for c in cases])
    n, log_a = np.polyfit(np.log(re_b), np.log(cf), 1)
    pred = np.exp(log_a) * re_b ** n
    return {
        "form": "C_f = A Re_b^{-n}",
        "A": float(np.exp(log_a)),
        "n": float(-n),
        "re_b": re_b.tolist(),
        "cf_dns": cf.tolist(),
        "cf_fit": pred.tolist(),
        "max_rel_dev": float(np.max(np.abs(pred - cf) / cf)),
        "note": (
            "fit through the three MKM DNS points only; it is NOT an independent "
            "correlation and is reported to show that a power law does describe "
            "this Re range, whereas the Dean correlation does not"
        ),
    }


def run_audit() -> dict:
    cases = [audit_case(n) for n in available_cases()]
    prov = load_provenance()
    payload: dict[str, Any] = {
        "schema": "tensorlbm-turbulent-channel-gate-audit-v1",
        "reference": {
            "dataset": prov.get("dataset"),
            "citation": prov.get("citation"),
            "upstream": prov.get("upstream"),
            "upstream_sha256": prov.get("upstream_sha256"),
            "files": {k: v.get("sha256") for k, v in (prov.get("files") or {}).items()},
            "re_tau": {c.nominal_re_tau: c.re_tau for c in cases},
        },
        "constants": {"kappa": KAPPA, "B": B_LOG, "dean": "C_f = 0.073 Re_b^-0.25"},
        "cases": [asdict(c) for c in cases],
        "power_law": _power_law_fit(cases),
        "verdicts": {},
    }
    # ---- per-criterion feasibility, decided on the reference data ---------
    worst_loglaw = max(c.log_law["max_rel"] for c in cases)
    worst_dean = max(abs(c.cf_dean_rel_dev) for c in cases)
    payload["verdicts"]["a_log_law_5pct"] = {
        "criterion": "max |U+ - ((1/0.41) ln y+ + 5.2)| / U+ < 5% in y+ in [30,100]",
        "reference_attains": worst_loglaw < 0.05,
        "worst_reference_max_rel": worst_loglaw,
        "verdict": "FEASIBLE" if worst_loglaw < 0.05 else "MIS-SPECIFIED",
        "comment": (
            "the DNS reference itself is the yardstick: if its own log-law residual "
            "exceeds the tolerance there is no tolerance left for a solver"
        ),
    }
    payload["verdicts"]["d_dean_5pct"] = {
        "criterion": "C_f vs Dean 0.073 Re_b^-0.25 within 5%",
        "reference_attains": worst_dean < 0.05,
        "worst_reference_rel_dev": worst_dean,
        "verdict": "MIS-SPECIFIED" if worst_dean >= 0.05 else "FEASIBLE",
        "comment": (
            "Dean's correlation is a high-Re pipe/duct fit; at the Re_b of a "
            "channel DNS it is a systematic extrapolation, so the criterion can be "
            "failed by exact DNS data. Replacement: the MKM C_f itself, or the "
            "power-law fit reported under 'power_law'."
        ),
    }
    payload["verdicts"]["b_stress_10pct"] = {
        "criterion": "u'u'+ peak within 10% of DNS",
        "reference_targets": {c.nominal_re_tau: c.stress_peaks["uu_peak"] for c in cases},
        "verdict": "FEASIBLE",
        "comment": "peak values are well defined in the reference (see reference_targets)",
    }
    payload["verdicts"]["c_spectrum_slope_10pct"] = {
        "criterion": "inertial-range slope within 10% of -5/3",
        "verdict": "UNVERIFIABLE_FROM_REFERENCE",
        "comment": (
            "the MKM distribution ships profiles only (no spectra), and at "
            "Re_tau <= 590 the -5/3 band is short; the criterion has to be "
            "evaluated against a fitted band declared in advance, not against a "
            "reference spectrum"
        ),
    }
    payload["verdicts"]["e_grid_convergence_3pct"] = {
        "criterion": "u_tau changes < 3% between the two finest grids",
        "verdict": "FEASIBLE",
        "comment": "requires two completed simulations; see the run ledger",
    }
    return payload


REFERENCE_JSON = "artifacts/p223w2b_n01_channel_turb/gate_audit.json"


def write_audit(root: str | Path = ".") -> Path:
    payload = run_audit()
    path = Path(root) / REFERENCE_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False)
    path.write_text(text + "\n")
    payload["artifact_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return path


if __name__ == "__main__":  # pragma: no cover
    print(write_audit())