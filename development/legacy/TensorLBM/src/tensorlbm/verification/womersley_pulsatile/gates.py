"""Acceptance gates for the pulsatile (Womersley) verification case.

Predeclared acceptance (case pool ``CASE-LBM-WOMERSLEY-PULSATILE``):

blocking criteria
  * centreline velocity **amplitude ratio** vs the analytic Womersley solution:
    ``| |U_num(0)|/|U_ana(0)| - 1 | <= 5 %``
  * centreline **phase lag** vs analytic, relative (the lag is close to 90 deg
    at alpha=5, so the tolerance is stated relative to it):
    ``|dphi| / |phi_ana| <= 5 %``

informative (reported, not blocking)
  * whole-period phase-averaged **profile L2 error** normalised by the analytic
    peak speed, over the full gap and over the inner core (|y'| or r/R <= 0.8).
    The full-profile number is dominated by the near-wall Stokes layer, which is
    only ~2-4 lattice cells thick at alpha=5; it is therefore reported for
    diagnosis and required only not to grow under refinement.

convergence
  * error must not increase from the coarse to the fine grid (H -> 2H, R -> 2R).

``TOL_PCT`` is fixed a priori and must not be tuned after inspecting results.
"""

from __future__ import annotations

TOL_PCT = 5.0

CRITERIA = [
    {
        "key": "amplitude_rel_pct",
        "name": "centreline amplitude ratio vs Womersley analytic",
        "threshold_pct": TOL_PCT,
        "blocking": True,
    },
    {
        "key": "phase_rel_pct",
        "name": "centreline phase lag vs Womersley analytic (relative)",
        "threshold_pct": TOL_PCT,
        "blocking": True,
    },
    {
        "key": "profile_l2_pct",
        "name": "phase-averaged full-profile L2 / analytic peak speed (informative)",
        "threshold_pct": TOL_PCT,
        "blocking": False,
    },
    {
        "key": "profile_l2_inner_pct",
        "name": "phase-averaged inner-core L2 (|y'| or r/R <= 0.8, informative)",
        "threshold_pct": TOL_PCT,
        "blocking": False,
    },
]


def evaluate_case(metrics: dict, tol_pct: float = TOL_PCT) -> dict:
    """Machine-checkable verdict for one grid / parameter point."""
    checks = {}
    for crit in CRITERIA:
        key = crit["key"]
        value = metrics.get(key)
        checks[key] = {
            "value_pct": value,
            "threshold_pct": tol_pct,
            "blocking": crit["blocking"],
            "passed": bool(value is not None and value <= tol_pct),
        }
    blocking_ok = all(
        c["passed"] for c in checks.values() if c["blocking"]
    )
    informative_ok = all(
        c["passed"] for c in checks.values() if not c["blocking"]
    )
    return {
        "checks": checks,
        "passed": blocking_ok,
        "informative_ok": informative_ok,
    }


def convergence_verdict(case_rows: list[dict]) -> dict:
    """Error must not grow when the resolution is refined (sorted coarse->fine)."""
    rows = sorted(case_rows, key=lambda r: r["resolution"])
    if len(rows) < 2:
        return {"applicable": False, "passed": None}
    amp = [r["metrics"]["amplitude_rel_pct"] for r in rows]
    phase = [r["metrics"]["phase_rel_pct"] for r in rows]
    l2 = [r["metrics"]["profile_l2_pct"] for r in rows]
    l2_inner = [r["metrics"]["profile_l2_inner_pct"] for r in rows]
    checks = {
        "amplitude_not_worse": bool(amp[-1] <= amp[0]),
        "phase_not_worse": bool(phase[-1] <= phase[0]),
        "profile_l2_not_worse": bool(l2[-1] <= l2[0]),
        "profile_l2_inner_not_worse": bool(l2_inner[-1] <= l2_inner[0]),
    }
    return {
        "applicable": True,
        "resolutions": [r["resolution"] for r in rows],
        "amplitude_rel_pct": amp,
        "phase_rel_pct": phase,
        "profile_l2_pct": l2,
        "profile_l2_inner_pct": l2_inner,
        "checks": checks,
        "passed": all(checks.values()),
    }