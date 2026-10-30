"""Acceptance gate for the verified B6 benchmark (SUBOFF bare hull, Re=1000).

Why this module exists (shift m336 diagnosis, 2026-10-31)
--------------------------------------------------------
The m319 shift reported ``Cd_tot = 0.061495`` (+46.43 % vs Blasius
``1.328/sqrt(1000) = 0.041995``) and attributed the failure to
``compile_status = eager_fallback``.  Both halves of that report are wrong:

* ``eager_fallback`` is **route-neutral by construction** (see
  ``benchmarks/compile_route.py``): the first compiled attempt dies during
  lowering, before producing an output, and every step then runs the *same*
  eager closure — bit-for-bit the ``--compile-mode eager`` trajectory.  The
  in-registry PASS value 0.041513 was itself produced in
  ``eager_fallback`` mode.
* A +46 % drag coefficient is what this case returns from a **short /
  coarse** run: the wake is still developing, so the sampled window mean
  sits far above the converged value.  Quoting such a number as the
  benchmark result is the actual defect.

So the fix is procedural, not numerical: make the runner itself refuse to
present a non-acceptance-eligible number as the benchmark result.  This
module is the pure decision function (no torch / no I/O) so it can be unit
tested; ``run.py`` writes its output as the ``acceptance`` block of
``result.json`` and prints a loud banner when the run is not eligible.

Acceptance definition (unchanged, from README / result.json)
-----------------------------------------------------------
End-to-end ``Cd_tot`` within +/-3 % of the Blasius reference **and** a
time-converged window mean, on a grid / step budget that can represent the
verified configuration (L_cells >= 48, n_steps >= 4000: the registered
coarse control L=48/20000 and the primary L=80/12000 both qualify, a
transient probe like L=32/1200 does not).
"""

from __future__ import annotations

import math

#: Blasius laminar flat-plate reference for the repo's Re=1000 family,
#: normalised on the wetted area S = pi*D*L (see README "参考口径").
CF_BLASIUS_RE1000: float = 1.328 / math.sqrt(1000.0)

#: Grid adequacy: the hull radius must be resolved by several cells.  At
#: L=48 the radius is 2.8 cells (registered coarse control, -3.17 %), at
#: L=80 it is 4.7 cells (primary, -1.15 %).
MIN_RESOLUTION_L: int = 48

#: Step adequacy: the registered runs are L=80/12000 and L=48/20000.
MIN_STEPS: int = 4000

#: Window-drift tolerance used to declare the window mean converged, in
#: percent of Cd_tot, comparing the primary window (last 1000 samples) with
#: the recent window (last 500 samples).
MAX_WINDOW_DRIFT_PCT: float = 1.0

#: Accuracy threshold for acceptance (README: |error| <= 3 %).
MAX_ABS_ERR_PCT: float = 3.0

VERDICTS = (
    "PASS",  # eligible config, converged, |err| <= 3 %
    "FAIL_ACCURACY",  # eligible + converged, but out of tolerance
    "NOT_CONVERGED",  # eligible config, window mean still drifting
    "COARSE_GRID",  # L_cells < MIN_RESOLUTION_L -> not the verified config
    "SHORT_RUN",  # n_steps < MIN_STEPS -> transient, not a steady result
)


def window_drift_pct(cd_total: float, cd_total_recent: float) -> float:
    """Relative difference between the primary and the recent window mean.

    ``cd_total_recent`` is the shorter tail window (last 500 samples in
    ``run.py``), so a converged run has both windows on the same plateau and
    returns a small drift.
    """
    if cd_total == 0.0:
        raise ValueError("cd_total must be non-zero")
    return abs(cd_total - cd_total_recent) / abs(cd_total) * 100.0


def assess(
    *,
    L_cells: int,
    n_steps: int,
    cd_total: float,
    cd_total_recent: float | None = None,
    cf_ref: float = CF_BLASIUS_RE1000,
    max_abs_err_pct: float = MAX_ABS_ERR_PCT,
    max_window_drift_pct: float = MAX_WINDOW_DRIFT_PCT,
    min_resolution_l: int = MIN_RESOLUTION_L,
    min_steps: int = MIN_STEPS,
) -> dict:
    """Decide whether a run may be quoted as the verified benchmark result.

    Parameters mirror ``result.json``: ``cd_total`` is the primary (last
    1000 samples) window mean on the wetted-area normalisation,
    ``cd_total_recent`` the last-500 window mean (``Cd_total_last5000`` in
    ``result.json``).

    Precedence: an ineligible configuration (coarse grid / short run) is
    reported as such *before* accuracy, because a transient number must
    never be presented as an accuracy failure (that is exactly the m319
    misreport); then convergence; then accuracy.
    """
    err_pct = (cd_total - cf_ref) / cf_ref * 100.0
    drift = (
        window_drift_pct(cd_total, cd_total_recent)
        if cd_total_recent is not None
        else float("nan")
    )
    within_tol = abs(err_pct) <= max_abs_err_pct
    converged = (drift <= max_window_drift_pct) if cd_total_recent is not None else True
    grid_ok = L_cells >= min_resolution_l
    steps_ok = n_steps >= min_steps

    if not grid_ok:
        verdict = "COARSE_GRID"
    elif not steps_ok:
        verdict = "SHORT_RUN"
    elif not converged:
        verdict = "NOT_CONVERGED"
    elif within_tol:
        verdict = "PASS"
    else:
        verdict = "FAIL_ACCURACY"

    #: True only when the number equals the registered benchmark statement.
    quotable = verdict == "PASS"
    reasons: list[str] = []
    if not grid_ok:
        reasons.append(
            f"L_cells={L_cells} < {min_resolution_l}: hull radius resolved by too few "
            "cells (discretisation error dominates; registered coarse control is L=48)"
        )
    if not steps_ok:
        reasons.append(f"n_steps={n_steps} < {min_steps}: transient window mean, not steady state")
    if not converged:
        reasons.append(
            f"window drift {drift:.2f}% > {max_window_drift_pct}%: last-1000 and "
            "last-500 window means disagree, flow not time-converged"
        )
    if not within_tol:
        reasons.append(f"|error| {abs(err_pct):.2f}% > {max_abs_err_pct}% vs Blasius {cf_ref:.6f}")

    return {
        "verdict": verdict,
        "quotable_as_benchmark_result": quotable,
        "cf_ref": cf_ref,
        "cd_total": cd_total,
        "error_pct_vs_Blasius": err_pct,
        "within_3pct": within_tol,
        "window_drift_pct": drift,
        "converged": converged,
        "grid_adequate": grid_ok,
        "steps_adequate": steps_ok,
        "criteria": {
            "L_cells_min": min_resolution_l,
            "n_steps_min": min_steps,
            "max_abs_err_pct": max_abs_err_pct,
            "max_window_drift_pct": max_window_drift_pct,
        },
        "reasons": reasons,
    }


def banner(assessment: dict, *, case: str = "suboff_re1000") -> str:
    """Loud, greppable one-block summary for run logs."""
    bar = "=" * 78
    head = (
        f"[acceptance] {case}: {assessment['verdict']} "
        f"(Cd_tot={assessment['cd_total']:.6f}, "
        f"err={assessment['error_pct_vs_Blasius']:+.2f}% vs Blasius "
        f"{assessment['cf_ref']:.6f})"
    )
    tail = "" if assessment["quotable_as_benchmark_result"] else (
        "[acceptance] DO NOT quote this number as the benchmark result: "
        + " | ".join(assessment["reasons"])
    )
    return f"\n{bar}\n{head}\n{tail}\n{bar}\n"