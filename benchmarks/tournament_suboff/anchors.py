"""Physical numerical anchors for the SUBOFF resistance tournament.

Two anchors, both frozen before any seat ran (prereg):

* **Blasius laminar flat plate** — ``Cf = 1.328/sqrt(Re)``, used for the
  Re=1000 laminar seats.  Value at Re=1000: 0.041995047327036086
  (identical to the frozen reference in
  ``benchmarks/verified/suboff_sail_resistance/prereg.md``).
* **ITTC-1957 5.5-power series** — ``Cf = 0.075/(log10(Re)-2)^2``, used for
  the Re>=1e5 turbulent seats (same convention as the repo's
  ``artifacts/suboff_best_rans_summary.json`` bench2).

Convention (inherited from the verified benchmark): the bare SUBOFF hull is
slender (L/D = 8.57) and friction-dominated, so the *total* drag coefficient
normalised by the full wetted area is compared directly against the
flat-plate friction anchor.  The form-factor offset (pressure drag) is a
known systematic bias at high Re and is disclosed, not absorbed.

Red-team self-tests (T5 discipline, known-solution cases):
* anchor(Re=1000) == 1.328/sqrt(1000) exactly (machine epsilon);
* ITTC-1957(Re=1e5) == 0.075/9 exactly;
* anchors are monotonically decreasing in Re (physical sanity);
* anchor is finite and positive for Re in [100, 1e8];
* the Blasius anchor at Re=1000 equals the frozen verified-benchmark value
  to 1e-15 (cross-repo consistency).
"""

from __future__ import annotations

import math

import numpy as np

# Frozen cross-check constant (verified/suboff_sail_resistance prereg, 2026-10-08)
BLASIUS_REF_RE1000_FROZEN = 0.041995047327036084

BLASIUS_COEF = 1.328
ITTC_COEF = 0.075


def blasius_cf(re: float) -> float:
    """Blasius laminar flat-plate friction coefficient."""
    if re <= 0:
        raise ValueError(f"Re must be positive, got {re}")
    return BLASIUS_COEF / math.sqrt(re)


def ittc1957_cf(re: float) -> float:
    """ITTC-1957 5.5-power turbulent friction coefficient."""
    if re <= 1e4:
        raise ValueError(f"ITTC-1957 requires Re > 1e4, got {re}")
    return ITTC_COEF / (math.log10(re) - 2.0) ** 2


def anchor_for(re: float) -> tuple[str, float]:
    """Return (anchor_name, Cf_ref) for a given Reynolds number.

    Re < 5e4  -> Blasius (laminar regime of the tournament)
    Re >= 5e4 -> ITTC-1957 (turbulent regime of the tournament)
    """
    if re < 5.0e4:
        return ("blasius", blasius_cf(re))
    return ("ittc1957", ittc1957_cf(re))


def red_team_self_test() -> dict:
    """Known-solution red-team checks.  Returns a dict of pass/fail per check.

    Any failure means the anchor layer is broken and NO seat may be scored.
    """
    checks: dict[str, bool] = {}
    checks["blasius_re1000_exact"] = abs(
        blasius_cf(1000.0) - BLASIUS_REF_RE1000_FROZEN
    ) < 1e-15
    checks["blasius_frozen_crosscheck"] = abs(
        blasius_cf(1000.0) - BLASIUS_REF_RE1000_FROZEN
    ) < 1e-14
    checks["ittc_re1e5_exact"] = abs(ittc1957_cf(1.0e5) - 0.075 / 9.0) < 1e-15
    re_grid = np.logspace(2, 8, 40)
    b = np.array([blasius_cf(r) for r in re_grid])
    checks["blasius_monotone_decreasing"] = bool(np.all(np.diff(b) < 0))
    t = np.array([ittc1957_cf(r) for r in re_grid[re_grid >= 1e4]])
    checks["ittc_monotone_decreasing"] = bool(np.all(np.diff(t) < 0))
    checks["positive_finite_range"] = bool(
        np.all(np.isfinite(b)) and np.all(b > 0) and np.all(np.isfinite(t)) and np.all(t > 0)
    )
    checks["regime_switch_consistent"] = (
        anchor_for(1.0e4)[0] == "blasius" and anchor_for(1.0e5)[0] == "ittc1957"
    )
    return checks


def all_red_team_pass() -> bool:
    return all(red_team_self_test().values())


if __name__ == "__main__":
    import json

    print(json.dumps(red_team_self_test(), indent=1))
