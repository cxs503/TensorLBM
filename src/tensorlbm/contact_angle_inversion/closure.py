"""Closure ("round-trip") auditing for contact-angle inversion.

Why this module exists
----------------------
A calibration-based inversion answers the question *"which wall strength
``f_w`` should I impose to obtain ``theta_target``?"*.  Verifying such an
inversion by evaluating the **same fitted curve** forward is a tautology: for
any target inside the calibration range the round trip

    ``theta_target -> (inverse of the fit) -> f_w -> (forward fit) -> theta``

returns ``theta_target`` to machine precision, no matter how wrong the
underlying physics is.  Numbers produced that way look like a validation but
carry no information (:func:`self_interpolation_error` measures exactly this
quantity, so the tautology can be *shown* rather than argued).

A real closure test must break the loop somewhere:

* impose the inverted ``f_w`` in an actual simulation,
* measure ``theta`` on the resulting density field with an estimator that has
  no knowledge of the calibration, and
* (to test transferability) do it on a grid that was **not** the calibration
  grid.

This module holds only the pure post-processing pieces of that workflow --
interpolation, error bookkeeping and the verdict rules -- so that the LBM
driver (:mod:`benchmarks.xval_huang2007_contact_angle_inversion.run_closure`)
can be re-run on any hardware while the acceptance arithmetic stays testable
without a solver.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

__all__ = [
    "piecewise_linear_invert",
    "piecewise_linear_forward",
    "self_interpolation_error",
    "closure_metrics",
    "estimator_disagreement",
    "stage_verdict",
    "CRITERION_PCT",
]

#: Owner精度标准: 对标误差 >15% 不可接受.
CRITERION_PCT = 15.0


def _sorted_pairs(points: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """``[(theta_measured, f_w_or_x), ...]`` sorted by the first entry."""
    pairs = [(float(a), float(b)) for a, b in points]
    return sorted(pairs, key=lambda p: p[0])


def piecewise_linear_invert(
    points: Iterable[Sequence[float]], x: float
) -> float:
    """Invert a measured curve ``(theta_measured, f_w)`` at ``x = theta_target``.

    Out-of-range targets are **clamped** to the end knots (the historical
    behaviour of the round-1 benchmark); a clamped answer is not an inversion
    and callers should treat a clamped target as *not measurable* instead of
    silently accepting it.  ``piecewise_linear_invert`` therefore reports the
    clamp through :func:`is_clamped` semantics documented below: the returned
    value equals the end knot's ``f_w`` whenever ``x`` lies outside the
    measured range.
    """
    pairs = _sorted_pairs(points)
    if len(pairs) < 2:
        raise ValueError("need at least two calibration points")
    if x <= pairs[0][0]:
        return pairs[0][1]
    if x >= pairs[-1][0]:
        return pairs[-1][1]
    for (t0, f0), (t1, f1) in zip(pairs, pairs[1:]):
        if t0 <= x <= t1:
            if t1 == t0:
                return f0
            w = (x - t0) / (t1 - t0)
            return f0 + w * (f1 - f0)
    return pairs[-1][1]


def piecewise_linear_forward(
    points: Iterable[Sequence[float]], x: float
) -> float:
    """Evaluate a curve given as ``(f_w, theta_measured)`` at ``x = f_w``."""
    pairs = _sorted_pairs(points)
    if len(pairs) < 2:
        raise ValueError("need at least two calibration points")
    if x <= pairs[0][0]:
        return pairs[0][1]
    if x >= pairs[-1][0]:
        return pairs[-1][1]
    for (t0, f0), (t1, f1) in zip(pairs, pairs[1:]):
        if t0 <= x <= t1:
            if t1 == t0:
                return f0
            w = (x - t0) / (t1 - t0)
            return f0 + w * (f1 - f0)
    return pairs[-1][1]


def self_interpolation_error(
    calibration_points: Iterable[Sequence[float]],
    theta_targets: Iterable[float],
) -> list[dict]:
    """Error of "invert then forward on the same curve" (the tautology test).

    ``calibration_points`` are ``(f_w, theta_measured)`` pairs.  For each target
    the routine inverts the measured ``theta`` axis to a wall fraction and then
    evaluates the forward fit at that wall fraction.  The returned ``err_pct``
    is what a self-consistent benchmark would report as its "accuracy": it is
    ~0 by construction inside the measured range and only grows where the
    target was clamped, which is precisely why such numbers cannot validate a
    contact-angle inversion.

    Returns one dict per target with ``theta_target``, ``fw``, ``theta_roundtrip``,
    ``err_pct`` and ``clamped``.
    """
    inv_pts = [(float(th), float(fw)) for fw, th in calibration_points]  # (theta, fw)
    fwd_pts = [(float(fw), float(th)) for fw, th in calibration_points]  # (fw, theta)
    thetas = [th for th, _ in _sorted_pairs(inv_pts)]
    out: list[dict] = []
    for tgt in theta_targets:
        tgt = float(tgt)
        fw = piecewise_linear_invert(inv_pts, tgt)
        back = piecewise_linear_forward(fwd_pts, fw)
        clamped = tgt < thetas[0] or tgt > thetas[-1]
        err = abs(back - tgt) / tgt * 100.0 if tgt > 0 else float("nan")
        out.append(
            {
                "theta_target": tgt,
                "fw": fw,
                "theta_roundtrip": back,
                "err_pct": err,
                "clamped": clamped,
            }
        )
    return out


def closure_metrics(rows: Iterable[Mapping[str, object]], criterion_pct: float = CRITERION_PCT) -> dict:
    """Aggregate per-run closure rows into pass/fail counts.

    Each row must carry ``theta_target`` and at least one measured angle under
    ``theta_measured`` (or ``theta_measured_alt`` for a second estimator).  The
    error is recomputed here from the raw numbers so a report can never carry a
    hand-written percentage: ``err_pct = |theta_measured - theta_target| /
    theta_target * 100``.
    """
    per_row: list[dict] = []
    for row in rows:
        tgt = float(row["theta_target"])  # type: ignore[arg-type]
        meas = float(row["theta_measured"])  # type: ignore[arg-type]
        finite = math.isfinite(meas) and math.isfinite(tgt) and tgt > 0
        err = abs(meas - tgt) / tgt * 100.0 if finite else float("nan")
        per_row.append(
            {
                "label": row.get("label", ""),
                "theta_target": tgt,
                "theta_measured": meas,
                "err_pct": err,
                "passed": bool(finite and err <= criterion_pct),
                "status": row.get("status", "ok" if finite else "no_measurement"),
            }
        )
    measured = [r for r in per_row if r["status"] == "ok"]
    passed = [r for r in measured if r["passed"]]
    worst = max((r["err_pct"] for r in measured), default=float("nan"))
    return {
        "criterion_pct": criterion_pct,
        "n_rows": len(per_row),
        "n_measured": len(measured),
        "n_passed": len(passed),
        "max_err_pct": worst,
        "rows": per_row,
        "all_passed": bool(measured) and len(passed) == len(measured),
    }


def estimator_disagreement(
    theta_primary: float, theta_alt: float, tol_deg: float = 5.0
) -> dict:
    """Compare two independent contact-angle estimators on one field.

    The primary estimator (:func:`tensorlbm.sc_surface_tension.contact_angle_from_mask`,
    a linear least-squares circle fit on the cap half-widths) and the alternate
    one (:func:`tensorlbm.contact_line.theta_fit.angle_from_density`, a contour
    point cloud + nonlinear circle fit) share no code.  A closure claim should
    not depend on which of the two is used: a large disagreement flags a
    measurement problem rather than a physics problem.
    """
    if not (math.isfinite(theta_primary) and math.isfinite(theta_alt)):
        return {"delta_deg": float("nan"), "agree": False, "status": "no_measurement"}
    delta = abs(theta_primary - theta_alt)
    return {
        "delta_deg": delta,
        "agree": bool(delta <= tol_deg),
        "tol_deg": tol_deg,
        "status": "ok" if delta <= tol_deg else "estimators_disagree",
    }


def stage_verdict(stages: Mapping[str, Mapping[str, object]], criterion_pct: float = CRITERION_PCT) -> dict:
    """Combine stage metrics into one verdict.

    Rules (no averaging, no cherry-picking):

    * a stage passes only if *every* measured row is within ``criterion_pct``;
    * a stage with zero measured rows is ``inconclusive`` -- never a pass;
    * the overall verdict is ``PASS`` only if the *in-grid* closure stage and
      every transferability stage that was run pass; ``FAIL`` if any run stage
      fails; ``INCONCLUSIVE`` otherwise.
    """
    summary: dict[str, str] = {}
    for name, metrics in stages.items():
        n_meas = int(metrics.get("n_measured", 0) or 0)
        if n_meas == 0:
            summary[name] = "inconclusive"
        elif metrics.get("all_passed"):
            summary[name] = "pass"
        else:
            summary[name] = "fail"

    required = {k: v for k, v in summary.items() if k.startswith("in_grid") or k.startswith("transfer")}
    if not required:
        required = summary
    if any(v == "fail" for v in summary.values()):
        verdict = "FAIL"
    elif required and all(v == "pass" for v in required.values()):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    return {"criterion_pct": criterion_pct, "stages": summary, "verdict": verdict}