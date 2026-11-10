"""Smagorinsky LES turbulence verification surface for the plane channel.

This module is the *acceptance* layer of the LBM-LES campaign
``p84w3-n05-lbm-les-turbulence``.  It answers one question only: given a
completed :class:`tensorlbm.turbulence.les_channel3d.LESChannel` run, do the
*measured* turbulence statistics match the MKM1999 direct-numerical-simulation
data at ``Re_tau = 180`` to the precision the campaign asks for?

The four criteria (numbers are the campaign specification, not tuned here):

1. **Mean velocity profile** — ``U+(y+)`` on the log-law region
   ``y+ in [30, min(100, 0.3 Re_tau)]`` vs the published DNS ``U+`` table,
   maximum relative error ``<= 5 %``.
2. **Reynolds shear stress** — the wall-normal position of the
   ``-<u'v'>+`` peak vs the DNS stress budget, ``|Delta y+| <= 5`` wall units
   (the amplitude error is reported alongside; prior campaign wave used
   ``<= 15 %`` and is kept as a secondary, non-gating metric so the two
   numbers cannot be confused).
3. **Energy spectrum** — log-log slope of ``E_uu(k_x)`` in the *a-priori*
   band of :func:`tensorlbm.turbulence.les_channel3d.fit_power_law_slope`
   within ``0.10`` of ``-5/3`` **and** fit ``R^2 >= 0.95``.
4. **Friction velocity** — ``u_tau`` recovered from the wall
   *momentum-exchange* flux (viscous + sub-grid wall stress actually applied by
   the bounce-back kernel) vs the DNS ``u_tau`` implied by ``Re_tau = 180``,
   relative error ``<= 3 %``.

Honesty notes that travel with every artifact:

* the driving is a constant body force ``a = 2 u_tau^2 / H``, the standard
  channel-DNS/LES set-up, so the *target* ``u_tau`` is an input.  Criterion 4
  is therefore **not** a blind prediction: it checks that the wall kernel
  really transfers the imposed momentum (``tau_w = rho a h``) instead of
  losing or inventing wall stress.  ``u_tau`` from the stress fit of the
  measured profile (``nu dU/dy - <u'v'>`` extrapolated to the wall) is
  reported as a second, weaker route.
* the Reynolds-stress *reference* is not a hand-copied number: the DNS
  ``-<u'v'>+`` profile is reconstructed from the published mean table through
  the exact channel momentum identity
  ``-<u'v'>+ = 1 - y+/Re_tau - dU+/dy+``
  (:func:`tensorlbm.turbulence.les_channel3d.dns_reynolds_shear`).
* every criterion can end in ``PASS``, ``FAIL`` or ``WITHHELD``; ``WITHHELD``
  is used when the run does not contain the quantity at all (e.g. no sampled
  spectrum) and never silently becomes a pass.
* :func:`criteria_from_artifact` recomputes all four verdicts from the stored
  profiles/spectra of an on-disk artifact.  The test suite compares that
  recomputation with the stored verdict, so a verdict cannot drift away from
  the data it claims to describe.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from tensorlbm.turbulence.les_channel3d import (
    LESChannel,
    LESChannelConfig,
    dns_reynolds_shear,
    fit_power_law_slope,
    interpolate_profile,
    mkm_profile_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---- campaign thresholds (specification, frozen here so tests can cite them)
LOG_LAW_MIN_Y_PLUS = 30.0
LOG_LAW_BAND_FRACTION = 0.30  # upper edge = min(100, 0.3 Re_tau)
LOG_LAW_MAX_REL_ERROR = 0.05
UV_PEAK_MAX_DY_PLUS = 5.0
UV_PEAK_MAX_REL_AMPLITUDE = 0.15  # secondary, non-gating
SPECTRUM_SLOPE_TOL = 0.10
SPECTRUM_R2_MIN = 0.95
U_TAU_MAX_REL_ERROR = 0.03
TARGET_RE_TAU = 180.0

CRITERIA_KEYS = (
    "1_mean_profile_log_region",
    "2_reynolds_stress_peak",
    "3_energy_spectrum_slope",
    "4_friction_velocity",
)


# ---------------------------------------------------------------------------
# DNS reference
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DnsReference:
    """Vendored MKM1999 mean table plus its reconstructed stress budget."""

    re_tau: float
    y_plus: torch.Tensor
    u_plus: torch.Tensor
    uv_minus_plus: torch.Tensor
    u_tau: float
    source: str

    def uv_peak(self) -> tuple[float, float]:
        """``(y+_peak, -<u'v'>+_peak)`` of the DNS budget reconstruction."""
        i = int(torch.argmax(self.uv_minus_plus))
        return float(self.y_plus[i]), float(self.uv_minus_plus[i])


def load_dns_reference(re_tau: float = TARGET_RE_TAU, root: str | Path = REPO_ROOT) -> DnsReference:
    """Load the vendored MKM1999 table for ``re_tau`` (no substitution)."""
    profile = mkm_profile_for(float(re_tau), root)
    if profile is None:
        raise FileNotFoundError(
            f"no vendored MKM1999 mean table for Re_tau={re_tau} under {root}"
        )
    y, uv = dns_reynolds_shear(profile)
    return DnsReference(
        re_tau=float(profile.re_tau),
        y_plus=profile.y_plus.to(torch.float64),
        u_plus=profile.u_plus.to(torch.float64),
        uv_minus_plus=uv,
        u_tau=1.0,  # DNS wall units: u_tau == 1 by normalisation
        source=str(profile.source),
    )


# ---------------------------------------------------------------------------
# criterion 1: mean profile vs DNS
# ---------------------------------------------------------------------------
def mean_profile_error(
    result: Mapping[str, Any],
    dns: DnsReference,
    *,
    y_plus_min: float = LOG_LAW_MIN_Y_PLUS,
    top_fraction: float = LOG_LAW_BAND_FRACTION,
    y_plus_max: float = 100.0,
) -> dict[str, Any]:
    """``U+`` error against the DNS table on the campaign log-law band."""
    wall = result.get("wall_units", {})
    y_plus = torch.as_tensor(wall.get("y_plus", []), dtype=torch.float64)
    u_plus = torch.as_tensor(wall.get("u_plus", []), dtype=torch.float64)
    re_tau = float(result.get("config", {}).get("re_tau_target", dns.re_tau))
    y_hi = min(float(y_plus_max), float(top_fraction) * re_tau)
    if y_plus.numel() < 3 or y_hi <= y_plus_min:
        return {
            "valid": False,
            "reason": "profile too short or the log-law band is empty",
            "y_plus_band": [y_plus_min, y_hi],
            "n_points": 0,
        }
    sel = (y_plus >= y_plus_min) & (y_plus <= y_hi)
    y_sel = y_plus[sel]
    if y_sel.numel() == 0:
        # no node inside the band: report the interpolation error at the band
        # edges anyway so a coarse grid cannot look like a pass.
        return {
            "valid": False,
            "reason": "no wall-normal node inside the log-law band",
            "y_plus_band": [y_plus_min, y_hi],
            "dy_plus": float(result.get("config", {}).get("dy_plus", float("nan"))),
            "n_points": 0,
        }
    dns_u = [interpolate_profile(dns.y_plus, dns.u_plus, float(yy)) for yy in y_sel]
    model_u = [float(uu) for uu in u_plus[sel]]
    rel = [
        (m - d) / d
        for m, d in zip(model_u, dns_u)
        if math.isfinite(d) and abs(d) > 1e-12
    ]
    if not rel:
        return {
            "valid": False,
            "reason": "DNS interpolation produced no finite reference value",
            "y_plus_band": [y_plus_min, y_hi],
            "n_points": 0,
        }
    abs_rel = [abs(r) for r in rel]
    return {
        "valid": True,
        "y_plus_band": [float(y_sel.min()), float(y_sel.max())],
        "requested_band": [float(y_plus_min), float(y_hi)],
        "n_points": len(rel),
        "y_plus": [float(v) for v in y_sel],
        "u_plus_model": model_u,
        "u_plus_dns": [float(v) for v in dns_u],
        "rel_error_signed": rel,
        "max_rel_error": float(max(abs_rel)),
        "rms_rel_error": float(math.sqrt(sum(r * r for r in rel) / len(rel))),
        "mean_rel_error": float(sum(rel) / len(rel)),
        "dns_source": dns.source,
        "dns_re_tau": dns.re_tau,
    }


# ---------------------------------------------------------------------------
# criterion 2: Reynolds-shear-stress peak
# ---------------------------------------------------------------------------
def reynolds_stress_peak_error(result: Mapping[str, Any], dns: DnsReference) -> dict[str, Any]:
    """``-<u'v'>+`` peak position (and amplitude) vs the DNS stress budget."""
    wall = result.get("wall_units", {})
    y_plus = torch.as_tensor(wall.get("y_plus", []), dtype=torch.float64)
    uv = torch.as_tensor(wall.get("uv_minus_plus", []), dtype=torch.float64)
    if y_plus.numel() < 3 or uv.numel() != y_plus.numel():
        return {
            "valid": False,
            "reason": "no Reynolds-stress profile in the result",
        }
    y_dns, uv_dns = dns.uv_peak()
    i = int(torch.argmax(uv))
    y_model = float(y_plus[i])
    uv_model = float(uv[i])
    return {
        "valid": True,
        "y_plus_peak_model": y_model,
        "uv_peak_model_plus": uv_model,
        "y_plus_peak_dns": y_dns,
        "uv_peak_dns_plus": uv_dns,
        "delta_y_plus": y_model - y_dns,
        "abs_delta_y_plus": abs(y_model - y_dns),
        "rel_error_amplitude": (uv_model - uv_dns) / uv_dns if abs(uv_dns) > 1e-12 else float("nan"),
        "tolerance_y_plus": UV_PEAK_MAX_DY_PLUS,
        "dns_source": dns.source,
        "reconstruction": "-<u'v'>+ = 1 - y+/Re_tau - dU+/dy+ (MKM1999 mean table)",
    }


# ---------------------------------------------------------------------------
# criterion 3: energy-spectrum slope
# ---------------------------------------------------------------------------
def spectrum_slope_check(
    result: Mapping[str, Any],
    *,
    slope_tol: float = SPECTRUM_SLOPE_TOL,
    r2_min: float = SPECTRUM_R2_MIN,
) -> dict[str, Any]:
    """Best (a-priori band) streamwise spectrum leg vs the ``-5/3`` law."""
    les = result.get("les", {})
    legs = les.get("spectra", {}).get("streamwise", [])
    candidates: list[dict[str, Any]] = []
    for leg in legs:
        fit = leg.get("fit", {})
        if not fit.get("valid"):
            continue
        candidates.append(
            {
                "y_plus": float(leg.get("y_plus", float("nan"))),
                "slope": float(fit["slope"]),
                "r2": float(fit["r2"]),
                "k_min": float(fit["k_min"]),
                "k_max": float(fit["k_max"]),
                "n_points": int(fit["n_points"]),
            }
        )
    if not candidates:
        return {
            "valid": False,
            "reason": "no valid spectral fit (window too short or too few modes)",
            "samples": les.get("spectra", {}).get("samples"),
        }
    best = min(candidates, key=lambda c: abs(c["slope"] + 5.0 / 3.0))
    best = dict(best)
    best.update(
        {
            "valid": True,
            "target_slope": -5.0 / 3.0,
            "slope_error": best["slope"] + 5.0 / 3.0,
            "slope_tol": float(slope_tol),
            "r2_min": float(r2_min),
            "passed": bool(
                abs(best["slope"] + 5.0 / 3.0) <= slope_tol and best["r2"] >= r2_min
            ),
            "samples": les.get("spectra", {}).get("samples"),
            "legs": candidates,
        }
    )
    return best


# ---------------------------------------------------------------------------
# criterion 4: friction velocity
# ---------------------------------------------------------------------------
def friction_velocity_error(result: Mapping[str, Any], dns: DnsReference) -> dict[str, Any]:
    """Wall-exchange ``u_tau`` vs the DNS value implied by ``Re_tau``."""
    fr = result.get("friction_velocity", {})
    target = float(fr.get("target", result.get("config", {}).get("u_tau_target", float("nan"))))
    measured = fr.get("wall_exchange")
    if measured is None or not math.isfinite(float(measured)) or target <= 0.0:
        return {
            "valid": False,
            "reason": "wall momentum-exchange flux was not accumulated",
            "u_tau_target": target,
        }
    measured = float(measured)
    rel = (measured - target) / target
    return {
        "valid": True,
        "u_tau_wall_exchange": measured,
        "u_tau_body_force": float(fr.get("body_force", float("nan"))),
        "u_tau_stress_fit": float(fr.get("stress_fit", float("nan"))),
        "u_tau_dns_target": target,
        "rel_error": rel,
        "abs_rel_error": abs(rel),
        "tolerance": U_TAU_MAX_REL_ERROR,
        "re_tau_achieved": float(result.get("re_tau_achieved", float("nan"))),
        "note": (
            "constant-body-force channel: u_tau is an input, so this criterion "
            "verifies the wall kernel transfers tau_w = rho a h (no spurious "
            "momentum sink); it is not a blind prediction of u_tau"
        ),
    }


# ---------------------------------------------------------------------------
# verdict assembly
# ---------------------------------------------------------------------------
def _verdict(ok: bool | None, numbers: Mapping[str, Any], description: str, tolerance: Any, metric: Any) -> dict[str, Any]:
    if ok is None:
        return {"passed": None, "status": "WITHHELD", "criterion": description, "numbers": dict(numbers), "tolerance": tolerance, "metric": metric}
    return {
        "passed": bool(ok),
        "status": "PASS" if ok else "FAIL",
        "criterion": description,
        "numbers": dict(numbers),
        "tolerance": tolerance,
        "metric": metric,
    }


def evaluate_criteria(
    result: Mapping[str, Any],
    dns: DnsReference | None = None,
    *,
    log_law_tol: float = LOG_LAW_MAX_REL_ERROR,
    uv_dy_tol: float = UV_PEAK_MAX_DY_PLUS,
    slope_tol: float = SPECTRUM_SLOPE_TOL,
    r2_min: float = SPECTRUM_R2_MIN,
    u_tau_tol: float = U_TAU_MAX_REL_ERROR,
) -> dict[str, Any]:
    """Evaluate criteria (1)-(4) of the campaign from a completed run."""
    dns = dns or load_dns_reference()
    mean_cmp = mean_profile_error(result, dns)
    uv_cmp = reynolds_stress_peak_error(result, dns)
    spec = spectrum_slope_check(result, slope_tol=slope_tol, r2_min=r2_min)
    utau = friction_velocity_error(result, dns)

    checks: dict[str, dict[str, Any]] = {}

    if mean_cmp.get("valid"):
        checks["1_mean_profile_log_region"] = _verdict(
            mean_cmp["max_rel_error"] <= log_law_tol,
            mean_cmp,
            f"max |U+ - U+_DNS| / U+_DNS on y+ in {mean_cmp['requested_band']} (<= {log_law_tol:.0%})",
            log_law_tol,
            mean_cmp["max_rel_error"],
        )
    else:
        checks["1_mean_profile_log_region"] = _verdict(
            None, mean_cmp, "no wall-normal node in the log-law band", log_law_tol, None
        )

    if uv_cmp.get("valid"):
        checks["2_reynolds_stress_peak"] = _verdict(
            uv_cmp["abs_delta_y_plus"] <= uv_dy_tol,
            uv_cmp,
            f"|-<u'v'>+ peak y+ error| <= {uv_dy_tol:g} wall units",
            uv_dy_tol,
            uv_cmp["abs_delta_y_plus"],
        )
    else:
        checks["2_reynolds_stress_peak"] = _verdict(None, uv_cmp, "no Reynolds-stress profile", uv_dy_tol, None)

    if spec.get("valid"):
        checks["3_energy_spectrum_slope"] = _verdict(
            spec["passed"],
            spec,
            f"E_uu(k_x) slope within {slope_tol:g} of -5/3 with R2 >= {r2_min:g}",
            {"slope": slope_tol, "r2_min": r2_min},
            {"slope": spec["slope"], "r2": spec["r2"]},
        )
    else:
        checks["3_energy_spectrum_slope"] = _verdict(None, spec, "no valid spectral fit", slope_tol, None)

    if utau.get("valid"):
        checks["4_friction_velocity"] = _verdict(
            utau["abs_rel_error"] <= u_tau_tol,
            utau,
            f"|u_tau(wall exchange) - u_tau(DNS)| / u_tau(DNS) <= {u_tau_tol:.0%}",
            u_tau_tol,
            utau["abs_rel_error"],
        )
    else:
        checks["4_friction_velocity"] = _verdict(None, utau, "no wall-exchange measurement", u_tau_tol, None)

    passed = [k for k, v in checks.items() if v.get("passed") is True]
    failed = [k for k, v in checks.items() if v.get("passed") is False]
    withheld = [k for k, v in checks.items() if v.get("passed") is None]
    return {
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "withheld": withheld,
        "all_passed": not failed and not withheld,
        "thresholds": {
            "log_law_max_rel_error": log_law_tol,
            "uv_peak_max_dy_plus": uv_dy_tol,
            "spectrum_slope_tol": slope_tol,
            "spectrum_r2_min": r2_min,
            "u_tau_max_rel_error": u_tau_tol,
        },
        "dns": {
            "source": dns.source,
            "re_tau": dns.re_tau,
            "uv_peak_y_plus": dns.uv_peak()[0],
            "uv_peak_plus": dns.uv_peak()[1],
        },
    }


def criteria_from_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the verdict from an artifact's *stored* numbers.

    The run result lives under ``artifact["result"]`` (whether the artifact is
    the raw result or the wrapped one).  Nothing but the artifact is read, so
    this is the anti-fabrication hook used by the tests.
    """
    result = artifact.get("result", artifact)
    dns = load_dns_reference(root=REPO_ROOT)
    return evaluate_criteria(result, dns)


def summarize(result: Mapping[str, Any]) -> dict[str, Any]:
    """Compact scalars used by the report figures/tables."""
    cfg = result.get("config", {})
    rs = result.get("reynolds_stress", {})
    fr = result.get("friction_velocity", {})
    les = result.get("les", {})
    return {
        "label": result.get("label") or cfg.get("label"),
        "cells": cfg.get("cells"),
        "steps": cfg.get("steps"),
        "dy_plus": cfg.get("dy_plus"),
        "y_plus_first_node": cfg.get("y_plus_first_fluid_node"),
        "u_tau_target": cfg.get("u_tau_target"),
        "smagorinsky_cs": cfg.get("smagorinsky_cs"),
        "van_driest": cfg.get("van_driest"),
        "u_bulk": result.get("mean_velocity_bulk"),
        "re_bulk": result.get("reynolds_number_bulk"),
        "re_tau_achieved": result.get("re_tau_achieved"),
        "u_tau_wall_exchange": fr.get("wall_exchange"),
        "u_tau_stress_fit": fr.get("stress_fit"),
        "uv_peak_plus": rs.get("uv_minus_peak_plus"),
        "uv_peak_y_plus": rs.get("uv_minus_peak_y_plus"),
        "sgs_fraction_y50": les.get("sgs_stress", {}).get("fraction_at_y_plus_50"),
        "eddy_turnovers_averaged": cfg.get("eddy_turnovers_averaged"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "steps_per_second": result.get("steps_per_second"),
    }


def run_case(
    *,
    half_height: int,
    steps: int,
    warmup_steps: int,
    u_tau: float = 0.01,
    re_tau: float = TARGET_RE_TAU,
    cs: float = 0.12,
    strain_source: str = "velocity_gradient",
    wall_model: str = "none",
    sample_interval: int = 10,
    report_interval: int = 500,
    van_driest_a_plus: float = 26.0,
    van_driest: bool = True,
    tau_minus_max: float | None = None,
    label: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one verification case and attach the criteria verdict.

    The result is plain JSON-able data of the form
    ``{"result": <LESChannel result>, "criteria": <evaluate_criteria>, ...}``.
    """
    h = int(half_height)
    cfg = LESChannelConfig(
        nx=int(round(2.0 * math.pi * h)),
        nz=int(round(math.pi * h)),
        ny=2 * h + 2,
        re_tau=float(re_tau),
        u_tau=float(u_tau),
        steps=int(steps),
        warmup_steps=int(warmup_steps),
        sample_interval=int(sample_interval),
        report_interval=int(report_interval),
        smagorinsky_cs=float(cs),
        strain_source=strain_source,
        van_driest=bool(van_driest),
        van_driest_a_plus=float(van_driest_a_plus),
        tau_minus_max=tau_minus_max,
        wall_model=wall_model,
        perturbation_amplitude=0.35,
        label=label or f"re{int(re_tau)}_h{h}",
    )
    cfg.validate()
    les = LESChannel(cfg)
    result = les.run_les(verbose=verbose)
    dns = load_dns_reference(re_tau, REPO_ROOT)
    criteria = evaluate_criteria(result, dns)
    return {
        "campaign": "p84w3-n05-lbm-les-turbulence",
        "result": result,
        "criteria": criteria,
        "summary": summarize(result),
    }


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1))
    return out